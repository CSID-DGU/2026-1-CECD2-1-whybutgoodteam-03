#!/usr/bin/env python3
"""
벤치마크: 룰베이스 프리필터 ON vs OFF 비교
- CPU 점유율
- 추론 시간
- 메모리 사용량
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import time
import glob
import json
import threading
import numpy as np
import psutil

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

import torch
from prefilter import rule_prefilter
from common_raw_audio import load_wav_fixed

NUM_SAMPLES = 50
RECORDS_DIR = "./records"
TARGET_SR = 22050
OUT_LEN = 88200
RULE_MIN_SCORE = 0.06
CLASS_NAMES = ["other", "emergency", "fire_alarm"]


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
    return pred_idx, float(prob[pred_idx])


def measure_cpu_during(func, *args, **kwargs):
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=False)

    cpu_samples_proc = []
    cpu_samples_sys = []
    mem_samples = []
    stop_event = threading.Event()
    result_container = [None]

    def sampler():
        while not stop_event.is_set():
            cpu_samples_proc.append(proc.cpu_percent(interval=None))
            cpu_samples_sys.append(psutil.cpu_percent(interval=None, percpu=False))
            mem_samples.append(proc.memory_info().rss / (1024 * 1024))
            stop_event.wait(0.2)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    t0 = time.perf_counter()
    result_container[0] = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    stop_event.set()
    t.join(timeout=2)

    return {
        "result": result_container[0],
        "elapsed": elapsed,
        "cpu_proc_avg": float(np.mean(cpu_samples_proc)) if cpu_samples_proc else 0,
        "cpu_proc_max": float(np.max(cpu_samples_proc)) if cpu_samples_proc else 0,
        "cpu_sys_avg": float(np.mean(cpu_samples_sys)) if cpu_samples_sys else 0,
        "cpu_sys_max": float(np.max(cpu_samples_sys)) if cpu_samples_sys else 0,
        "mem_avg_mb": float(np.mean(mem_samples)) if mem_samples else 0,
        "mem_max_mb": float(np.max(mem_samples)) if mem_samples else 0,
        "n_samples": len(cpu_samples_proc),
    }


def run_benchmark():
    print("=" * 60)
    print("  벤치마크: 룰베이스 프리필터 ON vs OFF")
    print("=" * 60)

    wav_files = sorted(glob.glob(os.path.join(RECORDS_DIR, "*.wav")))
    if len(wav_files) < NUM_SAMPLES:
        wav_subset = wav_files
    else:
        indices = np.linspace(0, len(wav_files) - 1, NUM_SAMPLES, dtype=int)
        wav_subset = [wav_files[i] for i in indices]

    print(f"총 WAV: {len(wav_files)}, 벤치마크 샘플: {len(wav_subset)}")

    print("\n[1/4] 모델 로딩...")
    model, device = load_esresnext()
    print(f"  디바이스: {device}")

    print("\n[2/4] 오디오 프리로드...")
    wavs = []
    for f in wav_subset:
        try:
            wav = load_wav_fixed(f, sr=TARGET_SR, target_len=OUT_LEN)
            wavs.append((f, wav))
        except Exception as e:
            print(f"  스킵: {os.path.basename(f)} ({e})")
    print(f"  로드 완료: {len(wavs)}개")

    print("\n[  ] 워밍업 추론...")
    _ = infer_model(model, device, wavs[0][1])
    _ = rule_prefilter(wavs[0][1], sr=TARGET_SR)
    print("  워밍업 완료")

    # ── A) 프리필터 ON ──
    print("\n[3/4] 프리필터 ON 벤치마크...")

    def run_with_prefilter():
        results = []
        for fname, wav in wavs:
            t0 = time.perf_counter()
            score = rule_prefilter(wav, sr=TARGET_SR, min_db=-35.0)
            prefilter_time = time.perf_counter() - t0

            if score < RULE_MIN_SCORE:
                results.append({
                    "file": os.path.basename(fname),
                    "stage": "rule_filtered",
                    "rule_score": score,
                    "prefilter_time": prefilter_time,
                    "model_time": 0.0,
                    "total_time": prefilter_time,
                    "pred_idx": -1,
                    "pred_prob": 0.0,
                })
            else:
                t1 = time.perf_counter()
                pred_idx, pred_prob = infer_model(model, device, wav)
                model_time = time.perf_counter() - t1
                results.append({
                    "file": os.path.basename(fname),
                    "stage": "passed",
                    "rule_score": score,
                    "prefilter_time": prefilter_time,
                    "model_time": model_time,
                    "total_time": prefilter_time + model_time,
                    "pred_idx": pred_idx,
                    "pred_prob": pred_prob,
                })
        return results

    on_metrics = measure_cpu_during(run_with_prefilter)
    on_results = on_metrics["result"]

    # ── B) 프리필터 OFF ──
    print("[4/4] 프리필터 OFF 벤치마크...")

    def run_without_prefilter():
        results = []
        for fname, wav in wavs:
            t0 = time.perf_counter()
            pred_idx, pred_prob = infer_model(model, device, wav)
            model_time = time.perf_counter() - t0
            results.append({
                "file": os.path.basename(fname),
                "stage": "model_only",
                "rule_score": -1,
                "prefilter_time": 0.0,
                "model_time": model_time,
                "total_time": model_time,
                "pred_idx": pred_idx,
                "pred_prob": pred_prob,
            })
        return results

    off_metrics = measure_cpu_during(run_without_prefilter)
    off_results = off_metrics["result"]

    # ── 결과 분석 ──
    print("\n" + "=" * 60)
    print("  결과 비교")
    print("=" * 60)

    on_filtered = [r for r in on_results if r["stage"] == "rule_filtered"]
    on_passed = [r for r in on_results if r["stage"] == "passed"]
    on_total_times = [r["total_time"] for r in on_results]
    on_prefilter_times = [r["prefilter_time"] for r in on_results]
    on_model_times = [r["model_time"] for r in on_passed]

    off_total_times = [r["total_time"] for r in off_results]

    on_avg = np.mean(on_total_times)
    off_avg = np.mean(off_total_times)
    diff_avg = off_avg - on_avg
    filt_pct = len(on_filtered) / len(on_results) * 100 if on_results else 0

    print(f"\n{'항목':<35} {'프리필터 ON':>15} {'프리필터 OFF':>15} {'차이':>15}")
    print("-" * 80)
    print(f"{'총 샘플 수':<35} {len(on_results):>15} {len(off_results):>15}")
    print(f"{'룰 필터링된 수':<35} {len(on_filtered):>15} {'N/A':>15}")
    print(f"{'모델 추론된 수':<35} {len(on_passed):>15} {len(off_results):>15}")
    print(f"{'필터링 비율':<35} {filt_pct:>14.1f}% {'0.0%':>15}")
    print()
    print(f"{'평균 처리시간/샘플 (s)':<35} {on_avg:>15.4f} {off_avg:>15.4f} {diff_avg:>+15.4f}")
    print(f"{'총 처리시간 (s)':<35} {sum(on_total_times):>15.3f} {sum(off_total_times):>15.3f} {sum(off_total_times)-sum(on_total_times):>+15.3f}")

    if on_prefilter_times:
        print(f"{'프리필터 평균시간 (s)':<35} {np.mean(on_prefilter_times):>15.4f} {'N/A':>15}")
    if on_model_times:
        print(f"{'모델 추론 평균시간 (s)':<35} {np.mean(on_model_times):>15.4f} {np.mean(off_total_times):>15.4f}")

    if on_avg > 0 and off_avg > 0:
        if on_avg < off_avg:
            print(f"{'속도 향상':<35} {off_avg/on_avg:>14.2f}x {'빠름':>15}")
        else:
            print(f"{'속도 차이':<35} {on_avg/off_avg:>14.2f}x {'느림':>15}")

    print()
    print(f"{'프로세스 CPU 평균 (%)':<35} {on_metrics['cpu_proc_avg']:>15.1f} {off_metrics['cpu_proc_avg']:>15.1f} {off_metrics['cpu_proc_avg']-on_metrics['cpu_proc_avg']:>+15.1f}")
    print(f"{'프로세스 CPU 최대 (%)':<35} {on_metrics['cpu_proc_max']:>15.1f} {off_metrics['cpu_proc_max']:>15.1f} {off_metrics['cpu_proc_max']-on_metrics['cpu_proc_max']:>+15.1f}")
    print(f"{'시스템 CPU 평균 (%)':<35} {on_metrics['cpu_sys_avg']:>15.1f} {off_metrics['cpu_sys_avg']:>15.1f} {off_metrics['cpu_sys_avg']-on_metrics['cpu_sys_avg']:>+15.1f}")
    print(f"{'시스템 CPU 최대 (%)':<35} {on_metrics['cpu_sys_max']:>15.1f} {off_metrics['cpu_sys_max']:>15.1f} {off_metrics['cpu_sys_max']-on_metrics['cpu_sys_max']:>+15.1f}")
    print()
    print(f"{'메모리 평균 (MB)':<35} {on_metrics['mem_avg_mb']:>15.1f} {off_metrics['mem_avg_mb']:>15.1f} {off_metrics['mem_avg_mb']-on_metrics['mem_avg_mb']:>+15.1f}")
    print(f"{'메모리 최대 (MB)':<35} {on_metrics['mem_max_mb']:>15.1f} {off_metrics['mem_max_mb']:>15.1f} {off_metrics['mem_max_mb']-on_metrics['mem_max_mb']:>+15.1f}")
    print()

    # 예측 일관성
    on_passed_dict = {r["file"]: r for r in on_results if r["stage"] == "passed"}
    off_dict = {r["file"]: r for r in off_results}

    match_count = 0
    mismatch_count = 0
    for fname, on_r in on_passed_dict.items():
        off_r = off_dict.get(fname)
        if off_r:
            if on_r["pred_idx"] == off_r["pred_idx"]:
                match_count += 1
            else:
                mismatch_count += 1
                print(f"  불일치: {fname} ON={on_r['pred_idx']}({on_r['pred_prob']:.3f}) vs OFF={off_r['pred_idx']}({off_r['pred_prob']:.3f})")

    print(f"{'예측 일치 (passed만)':<35} {match_count:>15} / {match_count + mismatch_count}")

    # 필터링된 샘플의 OFF 결과
    on_filtered_files = {r["file"] for r in on_results if r["stage"] == "rule_filtered"}
    filtered_off_preds = []
    for fname in on_filtered_files:
        off_r = off_dict.get(fname)
        if off_r:
            filtered_off_preds.append(off_r["pred_idx"])

    if filtered_off_preds:
        from collections import Counter
        cnt = Counter(filtered_off_preds)
        print(f"\n필터링된 샘플의 OFF 예측 분포:")
        for idx, count in sorted(cnt.items()):
            label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            pct = count / len(filtered_off_preds) * 100
            print(f"  {label}: {count}건 ({pct:.1f}%)")

        fire_in_filtered = sum(1 for p in filtered_off_preds if p in {1, 2})
        if fire_in_filtered > 0:
            print(f"  >> 프리필터가 화재로 예측될 수 있는 {fire_in_filtered}건을 걸러냄 (잠재적 FN)")
        else:
            print(f"  >> 프리필터가 걸러낸 샘플 중 화재 예측 없음 (안전)")

    # 요약
    print("\n" + "=" * 60)
    print("  요약")
    print("=" * 60)
    print(f"  프리필터 ON:  평균 {on_avg*1000:.1f}ms/샘플, CPU {on_metrics['cpu_proc_avg']:.1f}%")
    print(f"  프리필터 OFF: 평균 {off_avg*1000:.1f}ms/샘플, CPU {off_metrics['cpu_proc_avg']:.1f}%")
    if on_avg > 0 and off_avg > 0:
        saved_pct = (1 - on_avg / off_avg) * 100
        print(f"  프리필터 효과: 처리시간 {saved_pct:.1f}% 절감, 필터링률 {filt_pct:.1f}%")
    print("=" * 60)

    # JSON 저장
    summary = {
        "num_samples": len(wavs),
        "prefilter_on": {
            "total_time": sum(on_total_times),
            "avg_time_per_sample": on_avg,
            "filtered_count": len(on_filtered),
            "passed_count": len(on_passed),
            "filter_rate_pct": filt_pct,
            "cpu_proc_avg": on_metrics["cpu_proc_avg"],
            "cpu_proc_max": on_metrics["cpu_proc_max"],
            "cpu_sys_avg": on_metrics["cpu_sys_avg"],
            "cpu_sys_max": on_metrics["cpu_sys_max"],
            "mem_avg_mb": on_metrics["mem_avg_mb"],
            "mem_max_mb": on_metrics["mem_max_mb"],
        },
        "prefilter_off": {
            "total_time": sum(off_total_times),
            "avg_time_per_sample": off_avg,
            "cpu_proc_avg": off_metrics["cpu_proc_avg"],
            "cpu_proc_max": off_metrics["cpu_proc_max"],
            "cpu_sys_avg": off_metrics["cpu_sys_avg"],
            "cpu_sys_max": off_metrics["cpu_sys_max"],
            "mem_avg_mb": off_metrics["mem_avg_mb"],
            "mem_max_mb": off_metrics["mem_max_mb"],
        },
    }

    with open("./benchmark_result.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n결과 저장: benchmark_result.json")


if __name__ == "__main__":
    run_benchmark()
