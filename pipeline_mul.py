# python pipeline_mul.py

"""
Rule 프리필터 + 모델 추론 통합 파이프라인
지원 모델: esresnext / yamnet / rf
"""

import functools
print = functools.partial(print, flush=True)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import os
import glob
import unicodedata
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from prefilter import rule_prefilter
from common_raw_audio import load_wav_fixed

# ============================================================
#  설정
# ============================================================
AUDIO_ROOT = os.environ.get("AUDIO_ROOT", "./data/validation")  # 음원파일위치
OUT_CSV_PATH = "./result/pipe/rule_model.csv"   # 결과 CSV

RULE_MIN_SCORE = 0.06                                    # rule 프리필터 기준 (현장 로그 14k건 분석 후 0.07→0.06: 화재 catch율 92.6→96.3%)
POSITIVE_PREFIX = {"S1", "S2", "S3", "S8", "S10"}       # 양성 클래스 prefix (GT 기준)

# 3-class 공통 매핑: 0=other, 1=emergency, 2=fire_alarm
CLASS_NAMES = ['other', 'emergency', 'fire_alarm']
POSITIVE_INDICES = {1, 2}  # emergency, fire_alarm → 화재 양성

# ============================================================
#  ★ 활성 모델 선택 ★
#  "esresnext" | "yamnet" | "rf" 중 하나만 지정
# ============================================================
ACTIVE_MODEL = "esresnext"

# 모델별 경로
ESRESNEXT_MODEL_PATH = "./best_3class_esresnext_tuned.pt"
YAMNET_CLASSIFIER_PATH = "./yamnet_transfer_classifier.keras"
YAMNET_HUB_PATH = "./yamnet_local"  # TF Hub YAMNet (임베딩 추출용)
RF_MODEL_PATH = "./best_3class_rf.pkl"

# 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def norm(s: str, form: str = "NFD") -> str:
    return unicodedata.normalize(form, s)


# ============================================================
#  모델 로더들
# ============================================================

def load_model(model_type=None):
    """활성 모델을 로드한다.
    Returns: (model_bundle, model_type, target_sr, out_len)
      - model_bundle: 모델별로 필요한 객체 묶음 (dict)
      - target_sr: 해당 모델의 샘플레이트
      - out_len: 해당 모델의 입력 길이 (samples)
    """
    if model_type is None:
        model_type = ACTIVE_MODEL

    if model_type == "esresnext":
        return _load_esresnext(), model_type, 22050, 88200
    elif model_type == "yamnet":
        return _load_yamnet(), model_type, 16000, 64000
    elif model_type == "rf":
        return _load_rf(), model_type, 22050, 88200
    else:
        raise ValueError(f"지원하지 않는 모델: {model_type}")


# ── ESResNeXtFBSP ──
def _load_esresnext():
    from esresnext import ESResNeXtFBSP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESResNeXtFBSP(
        n_fft=2048, hop_length=561, win_length=1654,
        window='blackmanharris', normalized=True, onesided=True,
        spec_height=-1, spec_width=-1,
        num_classes=3, apply_attention=True, pretrained=False,
    )
    state_dict = torch.load(ESRESNEXT_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"[ESResNeXtFBSP] 로드 완료: {ESRESNEXT_MODEL_PATH}")
    return {"model": model, "device": device}


# ── YAMNet Transfer ──
def _load_yamnet():
    import tensorflow as tf
    import tensorflow_hub as hub

    yamnet = hub.load(YAMNET_HUB_PATH)
    classifier = tf.keras.models.load_model(YAMNET_CLASSIFIER_PATH)
    print(f"[YAMNet Transfer] 로드 완료: {YAMNET_CLASSIFIER_PATH}")
    return {"yamnet": yamnet, "classifier": classifier}


# ── Random Forest ──
def _load_rf():
    import joblib
    import librosa  # noqa: F401 – RF feature 추출에 필요

    pipeline = joblib.load(RF_MODEL_PATH)
    print(f"[Random Forest] 로드 완료: {RF_MODEL_PATH}")
    return {"pipeline": pipeline}


# ============================================================
#  모델별 추론 함수
# ============================================================

def _infer_esresnext(wav, bundle):
    """ESResNeXtFBSP: 오디오 → (pred_idx, pred_prob, all_probs)"""
    model = bundle["model"]
    device = bundle["device"]

    x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(prob.argmax())
    return pred_idx, float(prob[pred_idx]), prob


def _infer_yamnet(wav, bundle):
    """YAMNet Transfer: 오디오 → YAMNet 임베딩 → Keras 분류기 → (pred_idx, pred_prob, all_probs)"""
    import tensorflow as tf
    import numpy as np

    yamnet = bundle["yamnet"]
    classifier = bundle["classifier"]

    # YAMNet 임베딩 추출 (mean + std pooling → 2048-dim)
    waveform = tf.convert_to_tensor(wav, dtype=tf.float32)
    scores, embeddings, _ = yamnet(waveform)
    embeddings = embeddings.numpy()  # [num_frames, 1024]
    feat = np.concatenate([
        embeddings.mean(axis=0),
        embeddings.std(axis=0),
    ]).astype(np.float32)  # [2048]

    # Keras 분류기로 예측
    prob = classifier.predict(feat[np.newaxis, :], verbose=0)[0]

    pred_idx = int(prob.argmax())
    return pred_idx, float(prob[pred_idx]), prob


def _infer_rf(wav, bundle):
    """Random Forest: 오디오 → librosa 피처 → RF → (pred_idx, pred_prob, all_probs)"""
    import librosa

    pipeline = bundle["pipeline"]
    sr = 22050

    # 학습 코드와 동일한 피처 추출
    def summarize(mat):
        return np.concatenate([np.mean(mat, axis=1), np.std(mat, axis=1)])

    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=64), ref=np.max,
    )
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(wav)
    rms = librosa.feature.rms(y=wav)
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=wav, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=wav)

    feat = np.concatenate([
        summarize(mfcc), summarize(delta), summarize(delta2),
        summarize(mel), summarize(chroma),
        summarize(zcr), summarize(rms), summarize(centroid),
        summarize(bandwidth), summarize(rolloff), summarize(flatness),
    ]).astype(np.float32)

    prob = pipeline.predict_proba(feat[np.newaxis, :])[0]
    pred_idx = int(prob.argmax())
    return pred_idx, float(prob[pred_idx]), prob


# 모델별 추론 함수 매핑
_INFER_FN = {
    "esresnext": _infer_esresnext,
    "yamnet": _infer_yamnet,
    "rf": _infer_rf,
}


# ============================================================
#  유틸
# ============================================================

def get_tail_digits(name: str) -> int:
    m = re.search(r"(\d{1,2})$", name)
    return int(m.group(1)) if m else 0


# ============================================================
#  단일 파일 추론 (Rule → 모델)
# ============================================================

def infer_one_file(wav_path, target_sr, out_len, model_bundle, model_type):
    t0 = time.perf_counter()

    rel_path = os.path.relpath(wav_path, AUDIO_ROOT)
    true_label = rel_path.split(os.sep)[0]
    true_label = unicodedata.normalize("NFC", true_label)

    wav = load_wav_fixed(wav_path, sr=target_sr, target_len=out_len)

    # ── Stage 0: rule prefilter ──
    rule_score = rule_prefilter(wav, sr=target_sr, min_db=-35.0)
    if rule_score < RULE_MIN_SCORE:
        elapsed = time.perf_counter() - t0
        return {
            "path": wav_path,
            "stage": "rule_filtered",
            "true_label": true_label,
            "rule_score": rule_score,
            "pred_label": "RULE_FILTERED",
            "pred_prefix": None,
            "pred_prob": None,
            "reason": f"rule_filtered (rule_score={rule_score:.3f} < {RULE_MIN_SCORE})",
            "elapsed": elapsed,
        }

    # ── Stage 1: 모델 추론 ──
    try:
        infer_fn = _INFER_FN[model_type]
        pred_idx, pred_prob, all_probs = infer_fn(wav, model_bundle)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "path": wav_path,
            "stage": "model_failed",
            "true_label": true_label,
            "rule_score": rule_score,
            "pred_label": "MODEL_FAILED",
            "pred_prefix": None,
            "pred_prob": None,
            "reason": f"{model_type}_failed ({e})",
            "elapsed": elapsed,
        }

    pred_label = CLASS_NAMES[pred_idx]
    is_positive = pred_idx in POSITIVE_INDICES
    pred_prefix = "FIRE" if is_positive else "OTHER"

    elapsed = time.perf_counter() - t0
    return {
        "path": wav_path,
        "stage": "passed",
        "true_label": true_label,
        "rule_score": rule_score,
        "pred_label": pred_label,
        "pred_prefix": pred_prefix,
        "pred_prob": pred_prob,
        "reason": "",
        "elapsed": elapsed,
    }


# ============================================================
#  Confusion matrix / 평가
# ============================================================

def plot_confusion_matrix(cm, labels, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45)
    plt.yticks(tick, labels)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_binary_metrics(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    print(f"[{name}] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")


def apply_postprocessor(df_valid, window_size=3, min_positive=2):
    positive_prefix = {"FIRE"}

    if "group" not in df_valid.columns or "seq_idx" not in df_valid.columns:
        raise ValueError("df_valid에 group / seq_idx 없음")

    parts = []
    for gid, sub in df_valid.groupby("group"):
        sub = sub.sort_values("seq_idx").reset_index(drop=True)
        true_bin = sub["true_label"].apply(
            lambda t: 1 if t.split("-")[0] in POSITIVE_PREFIX else 0
        ).to_numpy()
        pred_bin = sub["pred_prefix"].apply(
            lambda p: 1 if p in positive_prefix else 0
        ).to_numpy()
        post = np.zeros_like(pred_bin)

        n = len(pred_bin)
        if n >= window_size:
            for start in range(0, n - window_size + 1):
                end = start + window_size
                if pred_bin[start:end].sum() >= min_positive:
                    post[start:end] = 1

        sub["true_bin"] = true_bin
        sub["pred_bin"] = pred_bin
        sub["post_pred_bin"] = post
        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


# ============================================================
#  메인
# ============================================================

def main():
    model_bundle, model_type, target_sr, out_len = load_model()
    print(f"\n활성 모델: {model_type} | SR={target_sr} | 입력길이={out_len}")

    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)

    wav_paths = sorted(glob.glob(os.path.join(AUDIO_ROOT, "**", "*.wav"), recursive=True))
    print("총 wav:", len(wav_paths))

    results = []
    group_state = {}

    for path in wav_paths:
        rel = os.path.relpath(path, AUDIO_ROOT)
        group = os.path.dirname(rel)
        base = os.path.basename(path)
        seq_idx = get_tail_digits(os.path.splitext(base)[0])

        r = infer_one_file(
            wav_path=path,
            target_sr=target_sr,
            out_len=out_len,
            model_bundle=model_bundle,
            model_type=model_type,
        )

        r["group"] = group
        r["seq_idx"] = seq_idx

        # ── 스트리밍 post-processor (3중2) ──
        st = group_state.setdefault(group, {"seq": [], "pred": [], "post": []})

        pred_bin_cur = 1 if (r["stage"] == "passed" and r["pred_prefix"] == "FIRE") else 0
        st["seq"].append(seq_idx)
        st["pred"].append(pred_bin_cur)

        if len(st["pred"]) >= 3:
            window = st["pred"][-3:]
            post_cur = 1 if sum(window) >= 2 else 0
            print(f"[POST] group={group} seq={seq_idx:02d}  window={window} "
                  f"sum={sum(window)} -> post={post_cur}")
        else:
            post_cur = 0

        st["post"].append(post_cur)
        r["post_pred_bin_stream"] = int(post_cur)
        results.append(r)

        df_tmp = pd.DataFrame(results)
        df_tmp.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")

        post_flag = f" | post={post_cur}"

        if r["stage"] == "passed":
            flag = "ALARM" if r["pred_prefix"] == "FIRE" else "NON"
            print(
                f"[OK] {base}  rule={r['rule_score']:.3f}  "
                f"pred={r['pred_label']} (p={r['pred_prob']:.3f}) "
                f"true={r['true_label']} {flag}{post_flag}"
            )
        elif r["stage"] == "rule_filtered":
            print(f"[RULE] {base}  rule={r['rule_score']:.3f} true={r['true_label']}{post_flag}")
        else:
            print(f"[{r['stage'].upper()}] {base}  reason={r.get('reason','')}{post_flag}")

    # ── CSV 저장 ──
    df = pd.DataFrame(results)

    def compute_stage_final(row):
        if row["stage"] == "rule_filtered":
            return "rule_filtered"
        if row["stage"] == "model_failed":
            return "model_failed"
        true_is_pos = row["true_label"].split("-")[0] in POSITIVE_PREFIX
        pred_is_pos = (row["pred_prefix"] == "FIRE")
        if true_is_pos and pred_is_pos:
            return "tp"
        if true_is_pos and not pred_is_pos:
            return "fn"
        if not true_is_pos and pred_is_pos:
            return "fp"
        return "tn"

    df["stage_final"] = df.apply(compute_stage_final, axis=1)
    df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print("CSV 저장:", OUT_CSV_PATH)

    # ── FN 분석 ──
    def is_pos(label):
        return label.split("-")[0] in POSITIVE_PREFIX

    df["true_bin_all"] = df["true_label"].apply(lambda t: 1 if is_pos(t) else 0)
    df["pred_bin_all"] = df.apply(
        lambda row: 1 if (row["stage"] == "passed" and row["pred_prefix"] == "FIRE") else 0,
        axis=1,
    )

    fn = df[(df["true_bin_all"] == 1) & (df["pred_bin_all"] == 0)]
    print(f"\n========== 파이프라인 FN 분석 ({model_type}) ==========")
    print(fn.groupby("stage")["true_label"].value_counts())
    print("=" * 50)

    fn_model = df[
        (df["true_bin_all"] == 1) & (df["stage"] == "passed") & (df["pred_bin_all"] == 0)
    ]
    print(f"모델 단계 FN 개수: {len(fn_model)}")
    if not fn_model.empty:
        print(fn_model["pred_label"].value_counts())

    # ── 처리 시간 ──
    if "elapsed" in df.columns:
        print(f"\n[TIME] 전체 평균: {df['elapsed'].mean():.3f} s (N={len(df)})")
        for stage_name in ["rule_filtered", "passed", "model_failed"]:
            mask = df["stage"] == stage_name
            if mask.any():
                print(f"[TIME] {stage_name} (N={mask.sum()}): {df.loc[mask, 'elapsed'].mean():.3f} s")

    # ── 성능 계산 ──
    df_valid = df[df["stage"] == "passed"]
    if df_valid.empty:
        print("성능 계산 불가 (passed 없음)")
        return

    true_mc = df_valid["true_label"].tolist()
    pred_mc = df_valid["pred_label"].tolist()
    labels = sorted(set(true_mc) | set(pred_mc))

    cm_mc = confusion_matrix(true_mc, pred_mc, labels=labels)
    plot_confusion_matrix(
        cm_mc, labels,
        "./result/pipe/cm_multiclass.png",
        title=f"Multiclass CM ({model_type})"
    )
    print("멀티클래스 CM 저장 완료")

    true_bin = [1 if t.split("-")[0] in POSITIVE_PREFIX else 0 for t in true_mc]
    pred_bin = [1 if p == "FIRE" else 0 for p in df_valid["pred_prefix"].tolist()]

    cm_bin = confusion_matrix(true_bin, pred_bin, labels=[0, 1])
    plot_confusion_matrix(
        cm_bin, ["Negative", "Positive"],
        "./result/pipe/emb_cm_binary.png",
        title=f"Binary CM ({model_type} stage only)"
    )
    print_binary_metrics(true_bin, pred_bin, f"{model_type}-only (segment)")

    cm_bin_all = confusion_matrix(df["true_bin_all"], df["pred_bin_all"], labels=[0, 1])
    plot_confusion_matrix(
        cm_bin_all, ["Negative", "Positive"],
        "./result/pipe/emb_cm_binary_pipeline.png",
        title=f"Binary CM (Full pipeline - {model_type})"
    )
    print_binary_metrics(df["true_bin_all"], df["pred_bin_all"], f"Full pipeline ({model_type})")

    if "post_pred_bin_stream" in df.columns:
        cm_post_all = confusion_matrix(
            df["true_bin_all"], df["post_pred_bin_stream"], labels=[0, 1],
        )
        plot_confusion_matrix(
            cm_post_all, ["Negative", "Positive"],
            "./result/pipe/emb_cm_binary_pipeline_post.png",
            title=f"Binary CM (Pipeline + streaming post - {model_type})"
        )
        print_binary_metrics(
            df["true_bin_all"], df["post_pred_bin_stream"],
            f"Pipeline + streaming post ({model_type})"
        )

    df_post = apply_postprocessor(df_valid)
    cm_post = confusion_matrix(df_post["true_bin"], df_post["post_pred_bin"], labels=[0, 1])
    plot_confusion_matrix(
        cm_post, ["Negative", "Positive"],
        "./result/pipe/emb_cm_binary_post.png",
        title=f"Binary CM (Offline postprocessor - {model_type})"
    )


if __name__ == "__main__":
    main()
