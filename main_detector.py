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
import socket
import numpy as np
import csv
from datetime import datetime

# --- 설정 ---
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000/api/events")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, "records")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "detection_log.csv")

MIC_RATE = 48000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 4.0
SLIDE_SECONDS = 4.0  # 슬라이딩 간격


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


def send_alert_to_server(pred_label="fire_alarm"):
    try:
        requests.post(
            SERVER_URL,
            json={"event_type": "fire_alarm_detected", "pred_label": pred_label},
            timeout=2,
        )
        print(f"🚨 [서버 전송 완료] label={pred_label}", flush=True)
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


class MicDisconnected(Exception):
    pass


# 녹음 스레드 heartbeat 미갱신이 N초 넘으면 끊김으로 간주 (USB 분리 시 stream.read가 영원히 블록되는 케이스 대응)
MIC_HEARTBEAT_TIMEOUT_SEC = 5.0


# ============================================================
#  실시간 라이브 오디오 (Unix socket broadcast)
# ============================================================
LIVE_SOCKET_PATH = "/tmp/gard_audio.sock"
LIVE_BUFFER_CHUNKS = 80  # ~1.7s @ 48kHz, 1024-sample chunks

_live_clients = []  # list of (conn, deque)
_live_cv = threading.Condition()


def _serve_live_client(conn, q):
    try:
        while True:
            with _live_cv:
                while not q:
                    _live_cv.wait(timeout=2.0)
                data = q.popleft()
            try:
                conn.sendall(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
    finally:
        try:
            conn.close()
        except Exception:
            pass
        with _live_cv:
            for i, (c, _) in enumerate(_live_clients):
                if c is conn:
                    _live_clients.pop(i)
                    break


def _live_audio_server():
    """Unix socket 서버: 새 청크가 들어오면 모든 구독자에게 broadcast."""
    try:
        os.unlink(LIVE_SOCKET_PATH)
    except FileNotFoundError:
        pass
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(LIVE_SOCKET_PATH)
    except OSError as e:
        print(f"[live audio] bind 실패: {e}", flush=True)
        return
    try:
        os.chmod(LIVE_SOCKET_PATH, 0o666)
    except OSError:
        pass
    srv.listen(8)
    print(f"[live audio] socket 대기: {LIVE_SOCKET_PATH}", flush=True)
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            time.sleep(0.5)
            continue
        q = collections.deque(maxlen=LIVE_BUFFER_CHUNKS)
        with _live_cv:
            _live_clients.append((conn, q))
        threading.Thread(target=_serve_live_client,
                         args=(conn, q), daemon=True).start()


def broadcast_live_chunk(data):
    with _live_cv:
        if not _live_clients:
            return
        for _, q in _live_clients:
            if len(q) >= q.maxlen:
                q.popleft()
            q.append(data)
        _live_cv.notify_all()


# ============================================================
#  연속 녹음 스레드 (링버퍼에 계속 쌓기)
# ============================================================
def continuous_recording_thread(stream, ring_buffer, ring_lock, stop_event, mic_error_event, heartbeat):
    """마이크에서 연속으로 읽어서 링버퍼에 쌓는다.
    - 매 chunk 성공 시 heartbeat[0] = time.time() 갱신 (메인 루프 watchdog용)
    - 명시적 read 실패 시 mic_error_event 세팅 후 종료
    """
    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            with ring_lock:
                ring_buffer.append(data)
            broadcast_live_chunk(data)
            heartbeat[0] = time.time()
        except (IOError, OSError) as e:
            print(f"❌ [녹음 스레드] read 실패: {e}", flush=True)
            mic_error_event.set()
            return
        except Exception as e:
            print(f"❌ [녹음 스레드 에러] {e}", flush=True)
            mic_error_event.set()
            return


# ============================================================
#  메인 프로세스
# ============================================================
def run_audio_session(infer_q, result_q, prediction_queue):
    """한 번의 마이크 세션. 끊김 감지 시 MicDisconnected 발생."""
    p = pyaudio.PyAudio()
    stream = None
    rec_thread = None
    stop_event = threading.Event()
    mic_error_event = threading.Event()
    heartbeat = [time.time()]  # 녹음 스레드가 chunk 받을 때마다 갱신

    chunks_for_4sec = int(MIC_RATE / CHUNK * RECORD_SECONDS)

    def check_mic_health():
        if mic_error_event.is_set():
            raise MicDisconnected("녹음 스레드 read 실패")
        if not rec_thread.is_alive():
            raise MicDisconnected("녹음 스레드 종료")
        gap = time.time() - heartbeat[0]
        if gap > MIC_HEARTBEAT_TIMEOUT_SEC:
            raise MicDisconnected(f"녹음 스레드 무응답 ({gap:.1f}s) — read가 블록됨")

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=MIC_RATE,
                        input=True, frames_per_buffer=CHUNK, start=True)
        print(f"🎤 마이크 연결 완료: {MIC_RATE}Hz", flush=True)
        print(f"🎧 4초 슬라이딩 윈도우 모드 (녹음={RECORD_SECONDS}s, 간격={SLIDE_SECONDS}s)\n", flush=True)

        ring_buffer = collections.deque(maxlen=chunks_for_4sec)
        ring_lock = threading.Lock()

        heartbeat[0] = time.time()
        rec_thread = threading.Thread(
            target=continuous_recording_thread,
            args=(stream, ring_buffer, ring_lock, stop_event, mic_error_event, heartbeat),
            daemon=True,
        )
        rec_thread.start()

        # 처음 4초는 버퍼 채우기 (끊김 빨리 감지하려고 짧게 폴링)
        for _ in range(int(RECORD_SECONDS * 10)):
            check_mic_health()
            time.sleep(0.1)

        while True:
            check_mic_health()

            # ── 1. 링버퍼에서 4초 분량 스냅샷 ──
            with ring_lock:
                if len(ring_buffer) < chunks_for_4sec:
                    time.sleep(0.5)
                    continue
                frames = list(ring_buffer)

            # ── 2. 파일 저장 ──
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            filename_str = now.strftime("%Y%m%d_%H%M%S%f")[:19]
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
                fire_label = None
                if result["stage"] == "passed" and result["pred_prefix"] == "FIRE":
                    is_fire = 1
                    fire_label = result["pred_label"]  # 'emergency' 또는 'fire_alarm'
                    print(f"⚠️  [화재 감지!] {result['pred_label']} ({result['pred_prob']:.2f}) | {result['elapsed']:.1f}s", flush=True)
                elif result["stage"] == "rule_filtered":
                    print(f"💤  [조용함] rule={result['rule_score']:.3f} | {result['elapsed']:.2f}s", flush=True)
                elif result["stage"] == "passed":
                    print(f"ℹ️  [일반 소음] {result['pred_label']} ({result['pred_prob']:.2f}) | {result['elapsed']:.1f}s", flush=True)
                else:
                    print(f"⚠️  [{result['stage']}] {result.get('reason','')}", flush=True)

                save_log_to_csv(result, is_fire)

                prediction_queue.append((is_fire, fire_label))
                fire_count = sum(1 for f, _ in prediction_queue if f)
                if len(prediction_queue) == 3 and fire_count >= 2:
                    fire_labels = [lbl for f, lbl in prediction_queue if f]
                    # fire_alarm이 한 번이라도 있으면 fire_alarm, 모두 emergency면 emergency
                    confirmed_label = "fire_alarm" if "fire_alarm" in fire_labels else "emergency"
                    print(f"\n🔥🔥🔥 [확정] {confirmed_label} 경보 발송!!! 🔥🔥🔥", flush=True)
                    send_alert_to_server(confirmed_label)
                    prediction_queue.clear()
                    time.sleep(3)

            # ── 5. 4초 대기 (슬라이딩 간격) ──
            time.sleep(SLIDE_SECONDS)
    finally:
        stop_event.set()
        # 스트림을 먼저 닫아서 stuck 상태인 stream.read()를 강제 해제 (USB 분리 시 필수)
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if rec_thread is not None:
            rec_thread.join(timeout=2)
            if rec_thread.is_alive():
                print("⚠️  녹음 스레드 join 타임아웃 (daemon이라 프로세스 종료 시 정리됨)", flush=True)
        try:
            p.terminate()
        except Exception:
            pass


def main():
    print("\n=== 🔥 가드이어 감지기 (4초 슬라이딩 윈도우) 시작 ===", flush=True)
    init_system()

    # 라이브 오디오 broadcast 서버 (Unix socket)
    threading.Thread(target=_live_audio_server, daemon=True).start()

    # 추론 프로세스 (한 번만 시작 — 마이크 재연결 시에도 유지)
    infer_q = multiprocessing.Queue(maxsize=4)
    result_q = multiprocessing.Queue()

    proc = multiprocessing.Process(
        target=inference_process,
        args=(infer_q, result_q),
        daemon=True,
    )
    proc.start()

    prediction_queue = collections.deque(maxlen=3)

    backoff = 1
    try:
        while True:
            try:
                run_audio_session(infer_q, result_q, prediction_queue)
                break  # 정상 종료 (현재 코드 흐름상 도달 안 함)
            except MicDisconnected as e:
                print(f"⚠️  [마이크 끊김] {e} → {backoff}초 후 재연결 시도...", flush=True)
                # 다음 세션이 깨끗하게 시작하도록 확정 큐 초기화
                prediction_queue.clear()
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                print(f"⚠️  [세션 에러] {e} → {backoff}초 후 재시작 시도...", flush=True)
                prediction_queue.clear()
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            else:
                backoff = 1  # 성공 시 리셋
    except KeyboardInterrupt:
        print("\n👋 종료합니다.", flush=True)
    finally:
        infer_q.put(None)
        proc.join(timeout=5)


if __name__ == "__main__":
    main()
