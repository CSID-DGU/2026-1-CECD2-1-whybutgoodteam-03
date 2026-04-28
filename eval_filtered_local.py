"""eval_filtered.py 로컬 변종 — wav 디렉토리는 /tmp/rf_wavs/, 모델은 로컬 .pt 사용."""
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from pipeline_mul import load_model, _INFER_FN, CLASS_NAMES, POSITIVE_INDICES
from common_raw_audio import load_wav_fixed

LOG = '/tmp/det_log.csv'  # Pi에서 받은 최신 detection_log
if not os.path.exists(LOG):
    LOG = os.path.join(ROOT, 'logs', 'detection_log.csv')

WAV_DIR = '/tmp/rf_wavs'
OUTPUT = '/tmp/eval_filtered_result.json'


def main():
    with open(LOG, 'rb') as f:
        text = f.read().replace(b'\x00', b'').decode('utf-8-sig', errors='replace')

    items = []
    for r in csv.DictReader(text.splitlines()):
        if r.get('stage') != 'rule_filtered':
            continue
        fname = (r.get('filename') or '').strip()
        if not fname:
            continue
        full = os.path.join(WAV_DIR, fname)
        if not os.path.isfile(full):
            continue
        try:
            rs = float(r['rule_score'])
        except (ValueError, TypeError):
            continue
        items.append((full, rs))

    print(f'rule_filtered rows with existing wav locally: {len(items)}', flush=True)

    print('Loading model...', flush=True)
    bundle, mtype, target_sr, out_len = load_model()
    infer_fn = _INFER_FN[mtype]
    print(f'Model={mtype} target_sr={target_sr} out_len={out_len}', flush=True)

    counts = {'fire': 0, 'fire_alarm': 0, 'emergency': 0, 'other': 0, 'failed': 0}
    fire_examples = []

    t0 = time.time()
    total = len(items)
    for i, (path, rs) in enumerate(items):
        if i and i % 100 == 0:
            el = time.time() - t0
            eta = (total - i) * el / i
            print(f'[{i}/{total}] el={el:.0f}s eta={eta:.0f}s fire={counts["fire"]}', flush=True)
        try:
            wav = load_wav_fixed(path, sr=target_sr, target_len=out_len)
            pred_idx, pred_prob, _ = infer_fn(wav, bundle)
            label = CLASS_NAMES[pred_idx]
            counts[label] = counts.get(label, 0) + 1
            if pred_idx in POSITIVE_INDICES:
                counts['fire'] += 1
                fire_examples.append({
                    'filename': os.path.basename(path),
                    'rule_score': rs,
                    'pred_label': label,
                    'pred_prob': float(pred_prob),
                })
        except Exception as e:
            counts['failed'] += 1
            print(f'  FAIL {path}: {e}', flush=True)

    fire_rate = counts['fire'] / total if total else 0

    result = {
        'total_evaluated': total,
        'counts': counts,
        'fire_rate': fire_rate,
        'fire_examples_top30_by_prob': sorted(fire_examples, key=lambda x: -x['pred_prob'])[:30],
        'elapsed_sec': time.time() - t0,
    }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n=== RESULT ===', flush=True)
    print(f'Evaluated: {total} rule_filtered files', flush=True)
    print(f'Predicted FIRE: {counts["fire"]}/{total} ({100*fire_rate:.2f}%)', flush=True)
    print(f'  fire_alarm={counts.get("fire_alarm",0)} emergency={counts.get("emergency",0)}', flush=True)
    print(f'Predicted other: {counts.get("other",0)}', flush=True)
    print(f'Failed: {counts["failed"]}', flush=True)
    print(f'Elapsed: {time.time() - t0:.0f}s', flush=True)
    print(f'Result saved to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
