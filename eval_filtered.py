"""
이전에 rule-filtered로 컷된 wav들을 모델에 직접 통과시켜
"실제로 모델이 FIRE라고 분류했을 비율"을 추정한다.

- 입력: logs/detection_log.csv 에서 stage='rule_filtered' 인 행
- 표본: 환경변수 N_SAMPLE (기본 500)
- 출력: stdout + /tmp/eval_filtered_result.json
"""
import csv
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from pipeline_mul import load_model, _INFER_FN, CLASS_NAMES, POSITIVE_INDICES
from common_raw_audio import load_wav_fixed

LOG = os.path.join(ROOT, 'logs', 'detection_log.csv')
RECORDS = os.path.join(ROOT, 'records')
N_SAMPLE = int(os.environ.get('N_SAMPLE', '500'))
SEED = int(os.environ.get('SEED', '42'))
OUTPUT = '/tmp/eval_filtered_result.json'


def main():
    # 1) rule_filtered 행 + 실제 wav 존재하는 것만
    with open(LOG, 'rb') as f:
        text = f.read().replace(b'\x00', b'').decode('utf-8-sig', errors='replace')

    items = []
    for r in csv.DictReader(text.splitlines()):
        if r.get('stage') != 'rule_filtered':
            continue
        fname = (r.get('filename') or '').strip()
        if not fname:
            continue
        full = os.path.join(RECORDS, fname)
        if not os.path.isfile(full):
            continue
        try:
            rs = float(r['rule_score'])
        except (ValueError, TypeError):
            continue
        items.append((full, rs))

    print(f'rule_filtered rows with existing wav: {len(items)}', flush=True)

    random.seed(SEED)
    sample = items if len(items) <= N_SAMPLE else random.sample(items, N_SAMPLE)
    print(f'Sampling {len(sample)} (seed={SEED})...', flush=True)

    print('Loading model...', flush=True)
    bundle, mtype, target_sr, out_len = load_model()
    infer_fn = _INFER_FN[mtype]
    print(f'Model={mtype} target_sr={target_sr} out_len={out_len}', flush=True)

    counts = {'fire': 0, 'fire_alarm': 0, 'emergency': 0, 'other': 0, 'failed': 0}
    fire_examples = []

    t0 = time.time()
    for i, (path, rs) in enumerate(sample):
        if i and i % 25 == 0:
            el = time.time() - t0
            eta = (len(sample) - i) * el / i
            print(f'[{i}/{len(sample)}] el={el:.0f}s eta={eta:.0f}s fire={counts["fire"]}', flush=True)
        try:
            wav = load_wav_fixed(path, sr=target_sr, target_len=out_len)
            pred_idx, pred_prob, _ = infer_fn(wav, bundle)
            label = CLASS_NAMES[pred_idx]
            if pred_idx in POSITIVE_INDICES:
                counts['fire'] += 1
                counts[label] = counts.get(label, 0) + 1
                fire_examples.append({
                    'filename': os.path.basename(path),
                    'rule_score': rs,
                    'pred_label': label,
                    'pred_prob': float(pred_prob),
                })
            else:
                counts[label] = counts.get(label, 0) + 1
        except Exception as e:
            counts['failed'] += 1
            print(f'  FAIL {path}: {e}', flush=True)

    total = len(sample)
    fire_rate = counts['fire'] / total if total else 0
    est_total_fire = round(fire_rate * len(items))

    result = {
        'total_filtered_rows_with_wav': len(items),
        'sample_size': total,
        'seed': SEED,
        'counts': counts,
        'fire_rate': fire_rate,
        'estimated_fires_in_full_filtered_set': est_total_fire,
        'fire_examples_top20_by_prob': sorted(fire_examples, key=lambda x: -x['pred_prob'])[:20],
        'elapsed_sec': time.time() - t0,
    }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n=== RESULT ===', flush=True)
    print(f'Sample: {total} of {len(items)} filtered files', flush=True)
    print(f'Predicted FIRE: {counts["fire"]} ({100*fire_rate:.2f}%)', flush=True)
    print(f'  fire_alarm={counts.get("fire_alarm",0)} emergency={counts.get("emergency",0)}', flush=True)
    print(f'Predicted other: {counts.get("other",0)}', flush=True)
    print(f'Failed: {counts["failed"]}', flush=True)
    print(f'\nExtrapolated to all {len(items)} filtered files: ~{est_total_fire} fires', flush=True)
    print(f'\nResult saved to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
