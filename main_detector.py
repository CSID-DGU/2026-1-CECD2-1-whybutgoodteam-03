import os
# 1. 라이브러리 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('Agg')

import time
import multiprocessing
import threading
import pyaudio
import wave
import requests
import collections
import numpy as np
import csv
from datetime import datetime

# --- 설정 ---
SERVER_URL = "http://127.0.0.1:5000/api/events"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, "records")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "detection_log.csv")

MIC_RATE = 48000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 4.0
SLIDE_SECONDS = 2.0  # 슬라이딩 간격


def init_system():
    if not os.path.exists(RECORD_DIR): os.makedirs(RECORD_DIR)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            header = ["timestamp", "filename", "stage", "rule_score", "pred_label", "pred_prob", "is_fire", "reason", "elapsed"]
            writer.writerow(header)


def save_log_to_csv(result_dict, is_fire):
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            row = [
                result_dict.get("timestamp", ""),
                os.path.basename(result_dict.get("path", "")),
                result_dict.get("stage", ""),
                f"{result_dict.get('rule_score', 0):.4f}",
                result_dict.get("pred_label", ""),
                f"{result_dict.get('pred_prob', 0):.4f}" if result_dict.get('pred_prob') else "",
                is_fire,
                result_dict.get("reason", ""),
                f"{result_dict.get('elapsed', 0):.4f}"
            ]
            writer.writerow(row)
    except Exception:
        pass


def send_alert_to_server():
    try:
        requests.post(SERVER_URL, json={"event_type": "fire_alarm_detected"}, timeout=2)
        print("🚨 [서버 전송 완료]", flush=True)
    except:
        print("❌ [서버 전송 실패]", flush=True)


# ============================================================
#  추론 프로세스
# ============================================================
def inference_process(infer_q, result_q):
    import pipeline_mul
    from pipeline_mul import load_model, infer_one_file

    print("[추론 프로세스] 모델 로딩 중...", flush=True)
    model_bundle, model_type, target_sr, out_len = load_model()
    print(f"[추론 프로세스] 모델 준비 완료! [{model_type}]", flush=True)

    while True:
        try:
            item = infer_q.get()
            if item is None:
                break

            wav_filename, timestamp_str = item

            result = infer_one_file(
                wav_path=wav_filename,
                target_sr=target_sr,
                out_len=out_len,
                model_bundle=model_bundle,
                model_type=model_type,
            )

            result["timestamp"] = timestamp_str
            result_q.put(result)

        except Exception as e:
            print(f"[추론 프로세스] 에러: {e}", flush=True)


# ============================================================
#  연속 녹음 스레드 (링버퍼에 계속 쌓기)
# ============================================================
def continuous_recording_thread(stream, ring_buffer, ring_lock, stop_event):
    """마이크에서 연속으로 읽어서 링버퍼에 쌓는다."""
    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            with ring_lock:
                ring_buffer.append(data)
        except IOError:
            pass
        except Exception as e:
            print(f"❌ [녹음 스레드 에러] {e}", flush=True)
            break


# ============================================================
#  메인 프로세스
# ============================================================
def main():
    print("\n=== 🔥 가드이어 감지기 (2초 슬라이딩 윈도우) 시작 ===", flush=True)
    init_system()

    # 추론 프로세스
    infer_q = multiprocessing.Queue(maxsize=4)
    result_q = multiprocessing.Queue()

    proc = multiprocessing.Process(
        target=inference_process,
        args=(infer_q, result_q),
        daemon=True,
    )
    proc.start()

    prediction_queue = collections.deque(maxlen=3)

    p = pyaudio.PyAudio()
    stream = None

    # 링버퍼: 4초 분량의 chunk 개수
    chunks_for_4sec = int(MIC_RATE / CHUNK * RECORD_SECONDS)
    # 2초마다 꺼낼 chunk 개수
    chunks_for_slide = int(MIC_RATE / CHUNK * SLIDE_SECONDS)

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=MIC_RATE,
                        input=True, frames_per_buffer=CHUNK, start=True)
        print(f"🎤 마이크 설정 완료: {MIC_RATE}Hz", flush=True)
        print(f"🎧 2초 슬라이딩 윈도우 모드 (녹음={RECORD_SECONDS}s, 간격={SLIDE_SECONDS}s)\n", flush=True)

        # 연속 녹음용 링버퍼
        ring_buffer = collections.deque(maxlen=chunks_for_4sec)
        ring_lock = threading.Lock()
        stop_event = threading.Event()

        rec_thread = threading.Thread(
            target=continuous_recording_thread,
            args=(stream, ring_buffer, ring_lock, stop_event),
            daemon=True,
        )
        rec_thread.start()

        # 처음 4초는 버퍼 채우기
        time.sleep(RECORD_SECONDS)

        while True:
            # ── 1. 링버퍼에서 4초 분량 스냅샷 ──
            with ring_lock:
                if len(ring_buffer) < chunks_for_4sec:
                    time.sleep(0.5)
                    continue
                frames = list(ring_buffer)

            # ── 2. 파일 저장 ──
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            filename_str = now.strftime("%Y%m%d_%H%M%S%f")[:19]  # 밀리초까지
            wav_filename = os.path.join(RECORD_DIR, f"{filename_str}.wav")

            wf = wave.open(wav_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(MIC_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # ── 3. 추론 큐에 넣기 ──
            if infer_q.full():
                try:
                    dropped = infer_q.get_nowait()
                    print(f"⏭️  [추론 밀림] {os.path.basename(dropped[0])} 건너뜀", flush=True)
                except:
                    pass
            infer_q.put((wav_filename, timestamp_str))

            # ── 4. 완료된 추론 결과 처리 ──
            while not result_q.empty():
                try:
                    result = result_q.get_nowait()
                except:
                    break

                is_fire = 0
                if result["stage"] == "passed" and result["pred_prefix"] == "FIRE":
                    is_fire = 1
                    print(f"⚠️  [화재 감지!] {result['pred_label']} ({result['pred_prob']:.2f}) | {result['elapsed']:.1f}s", flush=True)
                elif result["stage"] == "rule_filtered":
                    print(f"💤  [조용함] rule={result['rule_score']:.3f} | {result['elapsed']:.2f}s", flush=True)
                elif result["stage"] == "passed":
                    print(f"ℹ️  [일반 소음] {result['pred_label']} ({result['pred_prob']:.2f}) | {result['elapsed']:.1f}s", flush=True)
                else:
                    print(f"⚠️  [{result['stage']}] {result.get('reason','')}", flush=True)

                save_log_to_csv(result, is_fire)

                prediction_queue.append(is_fire)
                if len(prediction_queue) == 3 and sum(prediction_queue) >= 2:
                    print("\n🔥🔥🔥 [확정] 화재 경보 발송!!! 🔥🔥🔥", flush=True)
                    send_alert_to_server()
                    prediction_queue.clear()
                    time.sleep(3)

            # ── 5. 2초 대기 (슬라이딩 간격) ──
            time.sleep(SLIDE_SECONDS)

    except KeyboardInterrupt:
        print("\n👋 종료합니다.", flush=True)
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}", flush=True)
    finally:
        stop_event.set()
        infer_q.put(None)
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        proc.join(timeout=5)


if __name__ == "__main__":
    main()
