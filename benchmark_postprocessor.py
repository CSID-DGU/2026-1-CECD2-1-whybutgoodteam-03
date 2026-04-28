#!/usr/bin/env python3
"""
포스트프로세서(3-of-2 voting) ON vs OFF 비교 벤치마크
- 확률/정확도 비교
- CPU 점유율 비교
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import time
import glob
import json
import re
import threading
import collections
import numpy as np
import psutil
import torch

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

from prefilter import rule_prefilter
from common_raw_audio import load_wav_fixed

# ============================================================
#  설정
# ============================================================
AIHUB_DIR = "./aihub_test"
TARGET_SR = 22050
OUT_LEN = 88200  # 4초 @ 22050Hz
RULE_MIN_SCORE = 0.06

CLASS_NAMES = ["other", "emergency", "fire_alarm"]
# GT 매핑: 디렉토리명 -> 클래스 인덱스
GT_MAP = {"other": 0, "emergency": 1, "fire_alarm": 2}
POSITIVE_INDICES = {1, 2}


def load_esresnext():
    from esresnext import ESResNeXtFBSP
    device = torch.device("cpu")
    model = ESResNeXtFBSP(
        n_fft=2048, hop_length=561, win_length=1654,
        window="blackmanharris", normalized=True, onesided=True,
        spec_height=-1, spec_width=-1,
        num_classes=3, apply_attention=True, pretrained=False,
    )
    state_dict = torch.load("./best_3class_esresnext_tuned.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device


def infer_model(model, device, wav):
    x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(prob.argmax())
    return pred_idx, float(prob[pred_idx]), prob


def measure_cpu_during(func, *args, **kwargs):
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=False)

    cpu_proc = []
    cpu_sys = []
    mem = []
    stop = threading.Event()
    result = [None]

    def sampler():
        while not stop.is_set():
            cpu_proc.append(proc.cpu_percent(interval=None))
            cpu_sys.append(psutil.cpu_percent(interval=None, percpu=False))
            mem.append(proc.memory_info().rss / (1024 * 1024))
            stop.wait(0.3)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    t0 = time.perf_counter()
    result[0] = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    stop.set()
    t.join(timeout=2)

    return {
        "result": result[0],
        "elapsed": elapsed,
        "cpu_proc_avg": float(np.mean(cpu_proc)) if cpu_proc else 0,
        "cpu_proc_max": float(np.max(cpu_proc)) if cpu_proc else 0,
        "cpu_sys_avg": float(np.mean(cpu_sys)) if cpu_sys else 0,
        "cpu_sys_max": float(np.max(cpu_sys)) if cpu_sys else 0,
        "mem_avg_mb": float(np.mean(mem)) if mem else 0,
        "mem_max_mb": float(np.max(mem)) if mem else 0,
    }


def parse_group_seq(filename):
    """fire_alarm_g0001_s03.wav -> (group='fire_alarm_g0001', seq=3)"""
    base = os.path.splitext(filename)[0]
    m = re.match(r"(.+_g\d+)_s(\d+)", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, 0


def collect_data():
    """클래스별 그룹별 정렬된 파일 목록"""
    data = {}  # {class_name: {group_name: [(seq, filepath), ...]}}
    for cls in ["fire_alarm", "emergency", "other"]:
        cls_dir = os.path.join(AIHUB_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        data[cls] = {}
        for f in sorted(os.listdir(cls_dir)):
            if not f.endswith(".wav"):
                continue
            group, seq = parse_group_seq(f)
            if group not in data[cls]:
                data[cls][group] = []
            data[cls][group].append((seq, os.path.join(cls_dir, f)))
        # seq 순 정렬
        for g in data[cls]:
            data[cls][g].sort(key=lambda x: x[0])
    return data


def run_pipeline(model, device, data, use_prefilter=True):
    """전체 파이프라인 실행, 그룹 단위 결과 반환"""
    all_results = []

    for cls, groups in data.items():
        gt_idx = GT_MAP[cls]
        gt_is_positive = gt_idx in POSITIVE_INDICES

        for group_name, files in groups.items():
            group_results = []
            for seq, fpath in files:
                t0 = time.perf_counter()
                wav = load_wav_fixed(fpath, sr=TARGET_SR, target_len=OUT_LEN)

                # Stage 0: prefilter
                if use_prefilter:
                    rule_score = rule_prefilter(wav, sr=TARGET_SR, min_db=-35.0)
                    if rule_score < RULE_MIN_SCORE:
                        elapsed = time.perf_counter() - t0
                        group_results.append({
                            "file": os.path.basename(fpath),
                            "group": group_name,
                            "seq": seq,
                            "gt_class": cls,
                            "gt_idx": gt_idx,
                            "gt_is_positive": gt_is_positive,
                            "stage": "rule_filtered",
                            "pred_idx": 0,  # other
                            "pred_prob": 0.0,
                            "pred_is_positive": False,
                            "all_probs": [0, 0, 0],
                            "elapsed": elapsed,
                        })
                        continue

                # Stage 1: model
                pred_idx, pred_prob, all_probs = infer_model(model, device, wav)
                elapsed = time.perf_counter() - t0
                group_results.append({
                    "file": os.path.basename(fpath),
                    "group": group_name,
                    "seq": seq,
                    "gt_class": cls,
                    "gt_idx": gt_idx,
                    "gt_is_positive": gt_is_positive,
                    "stage": "passed",
                    "pred_idx": pred_idx,
                    "pred_prob": pred_prob,
                    "pred_is_positive": pred_idx in POSITIVE_INDICES,
                    "all_probs": all_probs.tolist(),
                    "elapsed": elapsed,
                })

            all_results.extend(group_results)

    return all_results


def apply_postprocessor(results, window=3, min_positive=2):
    """3-of-2 슬라이딩 윈도우 포스트프로세서"""
    # 그룹별로 처리
    groups = {}
    for r in results:
        g = r["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(r)

    post_results = []
    for group_name, items in groups.items():
        items.sort(key=lambda x: x["seq"])
        preds = [1 if r["pred_is_positive"] else 0 for r in items]

        for i, r in enumerate(items):
            r_copy = dict(r)
            if i >= window - 1:
                window_preds = preds[i - window + 1:i + 1]
                r_copy["post_positive"] = sum(window_preds) >= min_positive
            else:
                r_copy["post_positive"] = False
            post_results.append(r_copy)

    return post_results


def compute_metrics(results, use_post=False):
    """정확도, 정밀도, 재현율, F1 계산"""
    tp = fp = tn = fn = 0
    probs_positive_correct = []
    probs_positive_wrong = []

    for r in results:
        gt_pos = r["gt_is_positive"]
        if use_post:
            pred_pos = r.get("post_positive", r["pred_is_positive"])
        else:
            pred_pos = r["pred_is_positive"]

        if gt_pos and pred_pos:
            tp += 1
            probs_positive_correct.append(r["pred_prob"])
        elif gt_pos and not pred_pos:
            fn += 1
            probs_positive_wrong.append(r["pred_prob"])
        elif not gt_pos and pred_pos:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "avg_prob_correct": float(np.mean(probs_positive_correct)) if probs_positive_correct else 0,
        "avg_prob_wrong": float(np.mean(probs_positive_wrong)) if probs_positive_wrong else 0,
    }


def compute_class_metrics(results, use_post=False):
    """클래스별 정확도"""
    class_correct = {c: 0 for c in CLASS_NAMES}
    class_total = {c: 0 for c in CLASS_NAMES}

    for r in results:
        gt = r["gt_class"]
        if gt not in class_total:
            continue
        class_total[gt] += 1

        if use_post:
            pred_pos = r.get("post_positive", r["pred_is_positive"])
        else:
            pred_pos = r["pred_is_positive"]

        gt_pos = r["gt_is_positive"]
        if gt_pos == pred_pos:
            class_correct[gt] += 1

    return {c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0
            for c in CLASS_NAMES}


def main():
    print("=" * 60)
    print("  포스트프로세서(3-of-2) ON vs OFF 비교 벤치마크")
    print("=" * 60)

    # 데이터 수집
    print("\n[1/4] 데이터 수집...")
    data = collect_data()
    for cls, groups in data.items():
        total_files = sum(len(files) for files in groups.values())
        print(f"  {cls}: {len(groups)}그룹, {total_files}청크")

    # 모델 로드
    print("\n[2/4] 모델 로딩...")
    model, device = load_esresnext()

    # 워밍업
    first_cls = list(data.keys())[0]
    first_group = list(data[first_cls].keys())[0]
    first_file = data[first_cls][first_group][0][1]
    wav = load_wav_fixed(first_file, sr=TARGET_SR, target_len=OUT_LEN)
    _ = infer_model(model, device, wav)
    print("  워밍업 완료")

    # ── A) 포스트프로세서 OFF (세그먼트 단위 판정) ──
    print("\n[3/4] 포스트프로세서 OFF 벤치마크...")
    off_metrics = measure_cpu_during(run_pipeline, model, device, data, True)
    off_results = off_metrics["result"]

    # ── B) 포스트프로세서 ON (3-of-2 voting) ──
    print("[4/4] 포스트프로세서 ON 벤치마크...")
    # 동일 추론 결과 사용 (포스트프로세서는 추론 후 적용)
    on_results = apply_postprocessor(off_results, window=3, min_positive=2)

    # 포스트프로세서 자체 CPU 측정 (추론 없이 투표만)
    def run_post_only():
        return apply_postprocessor(off_results, window=3, min_positive=2)

    post_cpu = measure_cpu_during(run_post_only)

    # ============================================================
    #  결과 분석
    # ============================================================
    print("\n" + "=" * 60)
    print("  결과 분석")
    print("=" * 60)

    # 기본 통계
    total = len(off_results)
    filtered = sum(1 for r in off_results if r["stage"] == "rule_filtered")
    passed = total - filtered

    print(f"\n총 청크: {total}")
    print(f"프리필터 통과: {passed} ({passed/total*100:.1f}%)")
    print(f"프리필터 필터링: {filtered} ({filtered/total*100:.1f}%)")

    # ── 1. 성능 비교 ──
    print("\n" + "-" * 60)
    print("  1. 성능(정확도) 비교")
    print("-" * 60)

    m_off = compute_metrics(off_results, use_post=False)
    m_on = compute_metrics(on_results, use_post=True)

    print(f"\n{'지표':<20} {'포스트프로세서 OFF':>20} {'포스트프로세서 ON':>20} {'변화':>15}")
    print("-" * 75)
    for key, label in [("accuracy", "정확도(Accuracy)"),
                       ("precision", "정밀도(Precision)"),
                       ("recall", "재현율(Recall)"),
                       ("f1", "F1 Score")]:
        diff = m_on[key] - m_off[key]
        print(f"{label:<20} {m_off[key]:>20.4f} {m_on[key]:>20.4f} {diff:>+15.4f}")

    print(f"\n{'Confusion Matrix':<20} {'OFF':>20} {'ON':>20}")
    print("-" * 55)
    for key in ["tp", "fp", "tn", "fn"]:
        print(f"  {key.upper():<18} {m_off[key]:>20} {m_on[key]:>20}")

    # 클래스별
    c_off = compute_class_metrics(off_results, use_post=False)
    c_on = compute_class_metrics(on_results, use_post=True)

    print(f"\n{'클래스별 정확도':<20} {'OFF':>20} {'ON':>20} {'변화':>15}")
    print("-" * 75)
    for cls in CLASS_NAMES:
        diff = c_on[cls] - c_off[cls]
        print(f"  {cls:<18} {c_off[cls]:>20.4f} {c_on[cls]:>20.4f} {diff:>+15.4f}")

    # ── 2. 확률 분석 ──
    print("\n" + "-" * 60)
    print("  2. 예측 확률 분석")
    print("-" * 60)

    # 양성 클래스(fire/emergency)의 확률 분포
    pos_results = [r for r in off_results if r["gt_is_positive"] and r["stage"] == "passed"]
    neg_results = [r for r in off_results if not r["gt_is_positive"] and r["stage"] == "passed"]

    if pos_results:
        pos_probs = [r["pred_prob"] for r in pos_results if r["pred_is_positive"]]
        pos_wrong = [r["pred_prob"] for r in pos_results if not r["pred_is_positive"]]
        print(f"\n양성(fire/emergency) 샘플 중 모델 통과: {len(pos_results)}개")
        if pos_probs:
            print(f"  정탐(TP) 평균 확률: {np.mean(pos_probs):.4f} (N={len(pos_probs)})")
            print(f"  정탐(TP) 최소 확률: {np.min(pos_probs):.4f}")
        if pos_wrong:
            print(f"  오탐(FN) 평균 확률: {np.mean(pos_wrong):.4f} (N={len(pos_wrong)})")

    if neg_results:
        neg_fp = [r["pred_prob"] for r in neg_results if r["pred_is_positive"]]
        neg_correct = [r["pred_prob"] for r in neg_results if not r["pred_is_positive"]]
        print(f"\n음성(other) 샘플 중 모델 통과: {len(neg_results)}개")
        if neg_fp:
            print(f"  오경보(FP) 평균 확률: {np.mean(neg_fp):.4f} (N={len(neg_fp)})")
        if neg_correct:
            print(f"  정기각(TN) 평균 확률: {np.mean(neg_correct):.4f} (N={len(neg_correct)})")

    # 포스트프로세서로 수정된 케이스 분석
    off_preds = {r["file"]: r["pred_is_positive"] for r in off_results}
    on_preds = {r["file"]: r.get("post_positive", r["pred_is_positive"]) for r in on_results}

    corrected_fp = 0  # FP -> TN
    corrected_tp = 0  # TP -> FN (부정적)
    added_fp = 0      # TN -> FP
    added_tp = 0      # FN -> TP

    for r in on_results:
        f = r["file"]
        seg_pred = off_preds.get(f, False)
        post_pred = r.get("post_positive", seg_pred)
        gt = r["gt_is_positive"]

        if seg_pred != post_pred:
            if seg_pred and not post_pred:
                if not gt:
                    corrected_fp += 1
                else:
                    corrected_tp += 1
            elif not seg_pred and post_pred:
                if gt:
                    added_tp += 1
                else:
                    added_fp += 1

    print(f"\n포스트프로세서 수정 분석:")
    print(f"  FP -> TN (오경보 제거): {corrected_fp}건")
    print(f"  FN -> TP (미탐 복구):   {added_tp}건")
    print(f"  TP -> FN (정탐 손실):   {corrected_tp}건")
    print(f"  TN -> FP (오경보 추가): {added_fp}건")

    # ── 3. CPU/연산량 비교 ──
    print("\n" + "-" * 60)
    print("  3. 연산량(CPU) 비교")
    print("-" * 60)

    print(f"\n{'항목':<35} {'값':>20}")
    print("-" * 55)
    print(f"{'전체 추론 시간 (프리필터+모델)':<35} {off_metrics['elapsed']:>20.2f}s")
    print(f"{'포스트프로세서 추가 시간':<35} {post_cpu['elapsed']:>20.6f}s")
    print(f"{'포스트프로세서 오버헤드 비율':<35} {post_cpu['elapsed']/off_metrics['elapsed']*100:>19.4f}%")
    print(f"{'추론 CPU 평균 (프로세스)':<35} {off_metrics['cpu_proc_avg']:>20.1f}%")
    print(f"{'추론 CPU 최대 (프로세스)':<35} {off_metrics['cpu_proc_max']:>20.1f}%")
    print(f"{'추론 시스템 CPU 평균':<35} {off_metrics['cpu_sys_avg']:>20.1f}%")
    print(f"{'추론 시스템 CPU 최대':<35} {off_metrics['cpu_sys_max']:>20.1f}%")
    print(f"{'메모리 평균':<35} {off_metrics['mem_avg_mb']:>20.1f}MB")
    print(f"{'메모리 최대':<35} {off_metrics['mem_max_mb']:>20.1f}MB")
    print(f"{'총 청크 수':<35} {total:>20}")
    print(f"{'청크당 평균 추론시간':<35} {off_metrics['elapsed']/total*1000:>19.1f}ms")

    # ── 요약 ──
    print("\n" + "=" * 60)
    print("  요약")
    print("=" * 60)
    print(f"  포스트프로세서 OFF: Acc={m_off['accuracy']:.4f}, Prec={m_off['precision']:.4f}, "
          f"Rec={m_off['recall']:.4f}, F1={m_off['f1']:.4f}")
    print(f"  포스트프로세서 ON:  Acc={m_on['accuracy']:.4f}, Prec={m_on['precision']:.4f}, "
          f"Rec={m_on['recall']:.4f}, F1={m_on['f1']:.4f}")
    print(f"  포스트프로세서 연산 비용: {post_cpu['elapsed']*1000:.3f}ms (전체의 {post_cpu['elapsed']/off_metrics['elapsed']*100:.4f}%)")
    print(f"  오경보 제거: {corrected_fp}건, 정탐 손실: {corrected_tp}건")
    print("=" * 60)

    # JSON 저장
    summary = {
        "total_chunks": total,
        "filtered": filtered,
        "passed": passed,
        "postprocessor_off": m_off,
        "postprocessor_on": m_on,
        "class_metrics_off": c_off,
        "class_metrics_on": c_on,
        "corrections": {
            "fp_removed": corrected_fp,
            "fn_recovered": added_tp,
            "tp_lost": corrected_tp,
            "fp_added": added_fp,
        },
        "cpu": {
            "inference_total_sec": off_metrics["elapsed"],
            "postprocessor_sec": post_cpu["elapsed"],
            "cpu_proc_avg": off_metrics["cpu_proc_avg"],
            "cpu_proc_max": off_metrics["cpu_proc_max"],
            "cpu_sys_avg": off_metrics["cpu_sys_avg"],
            "mem_avg_mb": off_metrics["mem_avg_mb"],
        },
    }
    with open("./benchmark_postprocessor_result.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n결과 저장: benchmark_postprocessor_result.json")


if __name__ == "__main__":
    main()
