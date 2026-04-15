# common_raw_audio.py
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

SR = 16000
N_MELS = 64
HOP_LENGTH = 512
DURATION = 4.0            # 4초 짜리라고 했으니까
TARGET_LEN = int(DURATION * SR)


def list_wav_files(root_dir):
    """
    root_dir/label/..../*.wav 형태로 모두 수집
    group은 label/그 바로 위 폴더 로 할 거라서
    얌넷 임베딩 만들 때랑 최대한 비슷하게 감.
    """
    wav_paths = []
    labels = []
    groups = []

    for label_name in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        for r, _, files in os.walk(label_dir):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                full = os.path.join(r, f)

                rel_to_label = os.path.relpath(full, label_dir)
                parts = rel_to_label.split(os.sep)
                if len(parts) >= 2:
                    parent = parts[-2]
                else:
                    parent = "(root)"

                wav_paths.append(full)
                labels.append(label_name)
                groups.append(f"{label_name}/{parent}")

    return wav_paths, labels, groups


def load_wav_fixed(path, sr=SR, target_len=TARGET_LEN):
    import soundfile as sf
    try:
        import soxr
        data, orig_sr = sf.read(path, dtype='float32', always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if orig_sr != sr:
            data = soxr.resample(data, orig_sr, sr).astype(np.float32)
    except ImportError:
        data, _ = librosa.load(path, sr=sr, mono=True)
    y = data
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def wav_to_mel(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # (n_mels, time)


def extract_simple_features(mel_db):
    """
    RandomForest용: mel의 평균/표준편차만 뽑아서 1D 벡터로.
    """
    mean = mel_db.mean(axis=1)        # (n_mels,)
    std = mel_db.std(axis=1)          # (n_mels,)
    feat = np.concatenate([mean, std], axis=0)
    return feat  # (2 * n_mels,)


def split_by_group(groups, test_size=0.3, val_size=0.5, seed1=42, seed2=43):
    groups = np.array(groups)
    idx = np.arange(len(groups))

    gss1 = GroupShuffleSplit(test_size=test_size, random_state=seed1)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))

    gss2 = GroupShuffleSplit(test_size=val_size, random_state=seed2)
    val_sub, test_sub = next(gss2.split(temp_idx, groups=groups[temp_idx]))

    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]
    return train_idx, val_idx, test_idx


def plot_confusion_matrix(cm, idx_to_label, title, fname):
    num_classes = len(idx_to_label)
    df_cm = pd.DataFrame(
        cm,
        index=[idx_to_label[i] for i in range(num_classes)],
        columns=[idx_to_label[i] for i in range(num_classes)],
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(df_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [idx_to_label[i] for i in range(num_classes)],
               rotation=45, ha="right")
    plt.yticks(tick_marks, [idx_to_label[i] for i in range(num_classes)])

    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs("./result", exist_ok=True)
    out_path = f"./result/{fname}"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 {title} 저장: {out_path}")
