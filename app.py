import sqlite3
import os
import atexit
import socket
import subprocess
import threading
import time
import csv
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, g, send_from_directory, abort, Response
from flask_cors import CORS
from notifications import send_notification_task, send_manual_sms_task
from health import get_health
import telegram_notify

STATUS_REPORT_INTERVAL_SEC = 30 * 60  # 30분

DETECTION_LOG = 'logs/detection_log.csv'
RECORDINGS_DIR = os.path.abspath('records')
RECORDING_WINDOW_SEC = 10  # 이벤트 타임스탬프 ±N초의 녹음본을 매칭

# 보관 정책: 최근 RETENTION_HOURS 시간 wav + 화재 분류 wav 만 남기고 정리.
RETENTION_HOURS = 2
CLEANUP_INTERVAL_SEC = 5 * 60
# 업로드 마커가 없어도 이 시간이 지나면 강제 삭제 (네트워크 장기 다운 백스톱)
UPLOAD_BACKSTOP_HOURS = 6
UPLOADED_DIR = os.path.join(RECORDINGS_DIR, ".uploaded")

# 실시간 라이브 오디오 (detector가 publish하는 Unix socket)
LIVE_AUDIO_SOCKET = "/tmp/gard_audio.sock"
LIVE_SAMPLE_RATE = 48000
LIVE_CHANNELS = 1
LIVE_BITS = 16

DATABASE = 'gard-ear.db'
DEVICE_ID = 'rasp_pi_main'

# 화재 알림 쿨다운: 직전 알림 후 N초 이내면 이벤트는 기록하되 알림(SMS/이메일/텔레그램)은 스킵.
FIRE_ALERT_COOLDOWN_SEC = 5 * 60
_last_fire_alert_ts = 0.0
_fire_alert_lock = threading.Lock()

app = Flask(__name__, template_folder='.')
CORS(app)
app.config['DATABASE'] = DATABASE

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None: db.close()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = [dict(row) for row in cur.fetchall()]
    cur.close()
    return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    try:
        db = get_db()
        db.execute(query, args)
        db.commit()
        return True
    except sqlite3.Error as e:
        print(f"DB Error: {e}")
        return False

import struct as _struct


def _make_streaming_wav_header(sample_rate=LIVE_SAMPLE_RATE,
                                channels=LIVE_CHANNELS,
                                bits=LIVE_BITS):
    """무한 스트리밍용 WAV 헤더. ChunkSize / Subchunk2Size 를 최대값으로 채워
    브라우저가 끝없이 재생하도록 한다."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = 0xFFFFFFFF - 36
    riff_size = 0xFFFFFFFF
    return (
        b'RIFF' + _struct.pack('<I', riff_size) + b'WAVE'
        + b'fmt ' + _struct.pack('<I', 16)
        + _struct.pack('<HHIIHH', 1, channels, sample_rate,
                        byte_rate, block_align, bits)
        + b'data' + _struct.pack('<I', data_size)
    )


@app.route('/api/live_audio')
def live_audio_stream():
    """detector Unix socket → 무한 WAV chunked HTTP 스트림."""
    def generate():
        yield _make_streaming_wav_header()
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(LIVE_AUDIO_SOCKET)
        except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
            print(f'[live_audio] detector socket 연결 실패: {e}', flush=True)
            return
        try:
            sock.settimeout(30)
            while True:
                data = sock.recv(8192)
                if not data:
                    break
                yield data
        except (OSError, socket.timeout):
            pass
        finally:
            try:
                sock.close()
            except Exception:
                pass

    response = Response(generate(), mimetype='audio/wav')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/')
def index():
    return render_template('index.html')

# --- [NEW] 관리자 관련 API ---

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """관리자 비밀번호 확인"""
    data = request.json
    password = data.get('password')
    
    # DB에 저장된 비밀번호와 비교
    settings = query_db("SELECT admin_password FROM SystemSettings WHERE id = 1", one=True)
    if settings and settings['admin_password'] == password:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': '비밀번호가 일치하지 않습니다.'}), 401

@app.route('/api/admin/settings', methods=['GET', 'POST'])
def admin_settings():
    """발신자 정보 및 관리자 비밀번호 수정"""
    # (실제 서비스라면 여기서 세션/토큰 검증을 해야 하지만, 프로토타입이므로 생략)
    
    if request.method == 'POST':
        d = request.json
        # 비밀번호 변경 요청이 있으면 변경, 없으면 기존 유지
        if d.get('new_password'):
            execute_db("UPDATE SystemSettings SET admin_password = ? WHERE id = 1", [d['new_password']])

        # 발신자 정보 업데이트: 빈 문자열이면 기존 값 유지, 값이 있으면 덮어쓰기
        for field in ('gmail_user', 'gmail_password',
                      'solapi_api_key', 'solapi_api_secret', 'solapi_sender_number',
                      'telegram_bot_token', 'telegram_chat_id'):
            value = d.get(field)
            if value:
                execute_db(f"UPDATE SystemSettings SET {field} = ? WHERE id = 1", [value])

        return jsonify({'success': True})

    # GET 요청: 민감 정보는 빈 값으로 마스킹, *_set 플래그로 저장 여부만 노출
    row = query_db("SELECT * FROM SystemSettings WHERE id = 1", one=True)
    if row:
        masked = {
            'location': row.get('location', ''),
            'gmail_user': '',
            'gmail_password': '',
            'solapi_api_key': '',
            'solapi_api_secret': '',
            'solapi_sender_number': '',
            'telegram_bot_token': '',
            'telegram_chat_id': '',
            'solapi_sender_number_preview': row.get('solapi_sender_number') or '',
            'gmail_user_set': bool(row.get('gmail_user')),
            'gmail_password_set': bool(row.get('gmail_password')),
            'solapi_api_key_set': bool(row.get('solapi_api_key')),
            'solapi_api_secret_set': bool(row.get('solapi_api_secret')),
            'solapi_sender_number_set': bool(row.get('solapi_sender_number')),
            'telegram_bot_token_set': bool(row.get('telegram_bot_token')),
            'telegram_chat_id_set': bool(row.get('telegram_chat_id')),
        }
        return jsonify(masked)
    return jsonify({})

@app.route('/api/admin/send-sms', methods=['POST'])
def admin_send_sms():
    """관리자 수동 문자 발송. 본문 + 수신자 ID 리스트(없으면 활성 연락처 전원)."""
    d = request.json or {}
    message = (d.get('message') or '').strip()
    raw_ids = d.get('recipient_ids')

    if not message:
        return jsonify({'success': False, 'error': '본문이 비어 있습니다.'}), 400

    recipient_ids = []
    if isinstance(raw_ids, list):
        for x in raw_ids:
            try:
                recipient_ids.append(int(x))
            except (TypeError, ValueError):
                continue

    if recipient_ids:
        placeholders = ','.join('?' * len(recipient_ids))
        rows = query_db(
            f"SELECT id FROM NotificationContacts "
            f"WHERE id IN ({placeholders}) AND phone IS NOT NULL AND phone != ''",
            recipient_ids,
        )
    else:
        rows = query_db(
            "SELECT id FROM NotificationContacts "
            "WHERE is_active = 1 AND phone IS NOT NULL AND phone != ''"
        )

    queued = len(rows or [])
    if queued == 0:
        return jsonify({'success': False, 'error': '발송 대상 수신자가 없습니다.'}), 400

    threading.Thread(
        target=send_manual_sms_task,
        args=(message, recipient_ids if recipient_ids else None),
    ).start()
    return jsonify({'success': True, 'queued': queued})

# --- 기존 API ---
@app.route('/api/status', methods=['GET'])
def get_status():
    status_data = query_db("SELECT status FROM DeviceStatus WHERE device_id = ?", [DEVICE_ID], one=True)
    settings_data = query_db("SELECT location FROM SystemSettings WHERE id = 1", one=True)
    
    if not status_data:
        execute_db("INSERT INTO DeviceStatus (device_id, status) VALUES (?, ?)", [DEVICE_ID, 'normal'])
        status_data = {'status': 'normal'}

    return jsonify({
        'status': status_data.get('status', 'normal'),
        'location': settings_data.get('location', '미설정')
    })

@app.route('/api/heartbeat', methods=['GET'])
def heartbeat():
    """감지기의 마지막 추론 한 줄 + 최근 사이클 시간 정보"""
    try:
        with open(DETECTION_LOG, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            f.seek(max(0, size - block))
            tail = f.read().decode('utf-8', errors='replace').splitlines()
        if not tail:
            return jsonify({'alive': False})
        last = tail[-1]
        # CSV 파싱
        reader = csv.reader([last])
        row = next(reader)
        if len(row) < 9:
            return jsonify({'alive': False})
        return jsonify({
            'alive': True,
            'timestamp': row[0],
            'stage': row[2],
            'rule_score': float(row[3]) if row[3] else None,
            'pred_label': row[4],
            'pred_prob': float(row[5]) if row[5] else None,
            'is_fire': row[6] == '1',
            'elapsed': float(row[8]) if row[8] else None,
        })
    except Exception as e:
        return jsonify({'alive': False, 'error': str(e)})

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify(get_health())


@app.route('/api/events', methods=['POST'])
def create_event():
    payload = request.get_json(silent=True) or {}
    pred_label = payload.get('pred_label', 'fire_alarm')

    execute_db("UPDATE DeviceStatus SET status = 'alert' WHERE device_id = ?", [DEVICE_ID])
    settings = query_db("SELECT location FROM SystemSettings WHERE id = 1", one=True)
    location = settings.get('location', '미설정') if settings else '미설정'
    execute_db("INSERT INTO Events (device_id, event_type, location) VALUES (?, 'fire_alarm_detected', ?)", [DEVICE_ID, location])

    global _last_fire_alert_ts
    now_ts = time.time()
    with _fire_alert_lock:
        elapsed = now_ts - _last_fire_alert_ts
        if elapsed < FIRE_ALERT_COOLDOWN_SEC:
            remaining = int(FIRE_ALERT_COOLDOWN_SEC - elapsed)
            print(f"[알림 쿨다운] {remaining}초 남음 — 이벤트 기록만 하고 알림 스킵", flush=True)
            return jsonify({'message': 'Alert recorded (notification suppressed by cooldown).',
                            'cooldown_remaining_sec': remaining}), 201
        _last_fire_alert_ts = now_ts

    threading.Thread(target=send_notification_task, args=(location, pred_label)).start()
    return jsonify({'message': 'Alert triggered.'}), 201

@app.route('/api/acknowledge', methods=['POST'])
def acknowledge_event():
    execute_db("UPDATE DeviceStatus SET status = 'normal' WHERE device_id = ?", [DEVICE_ID])
    return jsonify({'message': 'Alert acknowledged.'}), 200

@app.route('/api/events', methods=['GET'])
def get_events():
    return jsonify(query_db("SELECT * FROM Events ORDER BY timestamp DESC"))


def _existing_recordings():
    try:
        return set(os.listdir(RECORDINGS_DIR))
    except OSError:
        return set()


def _read_log_tail_lines(max_bytes):
    """detection_log.csv 끝에서 max_bytes 만큼만 읽어 라인 리스트 반환.
    파일이 더 작으면 전체. 첫 부분 잘린 라인은 버림.
    """
    try:
        size = os.path.getsize(DETECTION_LOG)
    except OSError:
        return []
    try:
        with open(DETECTION_LOG, 'rb') as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # 잘린 라인 버림
                data = f.read()
            else:
                data = f.read()
    except OSError:
        return []
    text = data.decode('utf-8', errors='replace').lstrip('﻿')
    return text.splitlines()


def _parse_detection_row(row_csv):
    """CSV row → dict. 형식 불일치 시 None."""
    if len(row_csv) < 7:
        return None
    ts_str = row_csv[0].strip()
    fname = row_csv[1].strip()
    if not ts_str or not fname:
        return None
    rule_score = None
    if len(row_csv) > 3 and row_csv[3].strip():
        try: rule_score = float(row_csv[3])
        except ValueError: pass
    pred_prob = None
    if len(row_csv) > 5 and row_csv[5].strip():
        try: pred_prob = float(row_csv[5])
        except ValueError: pass
    return {
        'filename': fname,
        'timestamp': ts_str,
        'stage': row_csv[2] if len(row_csv) > 2 else None,
        'rule_score': rule_score,
        'pred_label': (row_csv[4] or None) if len(row_csv) > 4 else None,
        'pred_prob': pred_prob,
        'is_fire': row_csv[6].strip() == '1',
    }


@app.route('/api/events/<int:event_id>/recordings', methods=['GET'])
def get_event_recordings(event_id):
    """이벤트 시각 ±N초 사이의 detection_log 행 + wav 존재 여부를 반환."""
    row = query_db("SELECT timestamp FROM Events WHERE id = ?", [event_id], one=True)
    if not row:
        return jsonify({'error': 'event not found'}), 404
    try:
        ev_ts = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return jsonify({'error': 'invalid event timestamp'}), 500

    lo_str = (ev_ts - timedelta(seconds=RECORDING_WINDOW_SEC)).strftime('%Y-%m-%d %H:%M:%S')
    hi_str = (ev_ts + timedelta(seconds=RECORDING_WINDOW_SEC)).strftime('%Y-%m-%d %H:%M:%S')

    existing = _existing_recordings()
    items = []
    # 보수적 5MB (~50k 행) — 최근 이벤트 대부분 커버. 더 옛날 이벤트면 빈 결과.
    lines = _read_log_tail_lines(5_000_000)
    for row_csv in csv.reader(lines):
        if len(row_csv) < 7:
            continue
        ts_str = row_csv[0].strip()
        if not (lo_str <= ts_str <= hi_str):
            continue
        item = _parse_detection_row(row_csv)
        if not item:
            continue
        item['exists'] = item['filename'] in existing
        items.append(item)

    return jsonify({'event_timestamp': row['timestamp'], 'recordings': items})


_RECENT_CACHE = {}  # seconds -> {'mtime':..., 'ts':..., 'data':...}
_RECENT_CACHE_TTL = 1.5
_RECENT_CACHE_LOCK = threading.Lock()


@app.route('/api/detections/recent', methods=['GET'])
def get_recent_detections():
    """최근 N초 내 detection_log 행 + wav 존재 여부."""
    try:
        seconds = int(request.args.get('seconds', 60))
    except (TypeError, ValueError):
        seconds = 60
    seconds = max(1, min(seconds, 600))

    try:
        log_mtime = os.path.getmtime(DETECTION_LOG)
    except OSError:
        log_mtime = 0
    now = time.time()
    with _RECENT_CACHE_LOCK:
        c = _RECENT_CACHE.get(seconds)
        if c and c['mtime'] == log_mtime and (now - c['ts']) < _RECENT_CACHE_TTL:
            return jsonify({'seconds': seconds, 'recordings': c['data']})

    cutoff_str = (datetime.now() - timedelta(seconds=seconds)).strftime('%Y-%m-%d %H:%M:%S')
    existing = _existing_recordings()
    items = []
    # 600초 max → 2MB(약 20k행)면 여유.
    lines = _read_log_tail_lines(2_000_000)
    for row_csv in csv.reader(lines):
        if len(row_csv) < 7:
            continue
        ts_str = row_csv[0].strip()
        if ts_str < cutoff_str:
            continue
        item = _parse_detection_row(row_csv)
        if not item:
            continue
        item['exists'] = item['filename'] in existing
        items.append(item)

    items.sort(key=lambda x: x['timestamp'], reverse=True)
    with _RECENT_CACHE_LOCK:
        _RECENT_CACHE[seconds] = {'mtime': log_mtime, 'ts': now, 'data': items}
    return jsonify({'seconds': seconds, 'recordings': items})


@app.route('/api/recordings/<path:filename>', methods=['GET'])
def get_recording(filename):
    # 경로 탈출 방지: 파일명에 슬래시/.. 차단, .wav만 허용
    if '/' in filename or '\\' in filename or '..' in filename or not filename.lower().endswith('.wav'):
        abort(400)
    return send_from_directory(RECORDINGS_DIR, filename, mimetype='audio/wav', conditional=True)


def _load_fire_filenames():
    """detection_log.csv에서 is_fire == 1 인 행의 filename 집합."""
    fire = set()
    try:
        with open(DETECTION_LOG, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if (r.get('is_fire') or '').strip() == '1':
                    fname = (r.get('filename') or '').strip()
                    if fname:
                        fire.add(fname)
    except FileNotFoundError:
        pass
    return fire


def cleanup_recordings_once():
    """RETENTION_HOURS 초과 wav 중 화재 분류 아닌 것 삭제. (deleted, kept) 반환.
    업로드 마커가 없으면 UPLOAD_BACKSTOP_HOURS 까지 보관 (gdrive 업로드 race 보호)."""
    if not os.path.isdir(RECORDINGS_DIR):
        return 0, 0
    try:
        os.makedirs(UPLOADED_DIR, exist_ok=True)
    except OSError:
        pass
    fire_set = _load_fire_filenames()
    now = time.time()
    cutoff = now - RETENTION_HOURS * 3600
    backstop_cutoff = now - UPLOAD_BACKSTOP_HOURS * 3600
    deleted = 0
    kept = 0
    for fname in os.listdir(RECORDINGS_DIR):
        if not fname.lower().endswith('.wav'):
            continue
        fpath = os.path.join(RECORDINGS_DIR, fname)
        try:
            mtime = os.path.getmtime(fpath)
        except OSError:
            continue
        if mtime >= cutoff or fname in fire_set:
            kept += 1
            continue
        marker = os.path.join(UPLOADED_DIR, fname)
        if not os.path.exists(marker) and mtime >= backstop_cutoff:
            kept += 1
            continue
        try:
            os.remove(fpath)
            deleted += 1
            try:
                os.remove(marker)
            except OSError:
                pass
        except OSError:
            pass
    return deleted, kept


def recordings_cleanup_loop():
    while True:
        try:
            deleted, kept = cleanup_recordings_once()
            if deleted or kept:
                print(f"[녹음 정리] 삭제 {deleted}개 / 보관 {kept}개 "
                      f"(최근 {RETENTION_HOURS}h + 화재)", flush=True)
        except Exception as e:
            print(f"[녹음 정리] 예외: {e}", flush=True)
        time.sleep(CLEANUP_INTERVAL_SEC)


_RETAINED_CACHE = {'key': None, 'ts': 0.0, 'data': None}
_RETAINED_CACHE_TTL = 3.0
_RETAINED_CACHE_LOCK = threading.Lock()


def _build_retained_index():
    """전체 retained 항목을 한 번만 만들어 반환. (only_fire 필터는 호출측에서)"""
    cutoff = datetime.now() - timedelta(hours=RETENTION_HOURS)
    cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')

    try:
        existing = set(os.listdir(RECORDINGS_DIR))
    except OSError:
        existing = set()

    items_by_fname = {}
    try:
        with open(DETECTION_LOG, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # skip header
            except StopIteration:
                return []
            for row in reader:
                if len(row) < 7:
                    continue
                ts_str = row[0].strip()
                fname = row[1].strip()
                if not fname or fname not in existing:
                    continue
                is_fire = row[6].strip() == '1'
                in_window = ts_str >= cutoff_str  # ISO-like format → 문자열 비교 OK
                if not (is_fire or in_window):
                    continue
                pred_prob = None
                if len(row) > 5 and row[5].strip():
                    try:
                        pred_prob = float(row[5])
                    except ValueError:
                        pred_prob = None
                items_by_fname[fname] = {
                    'filename': fname,
                    'timestamp': ts_str,
                    'stage': row[2] if len(row) > 2 else None,
                    'pred_label': (row[4] or None) if len(row) > 4 else None,
                    'pred_prob': pred_prob,
                    'is_fire': is_fire,
                    'in_window': in_window,
                }
    except FileNotFoundError:
        return []

    items = sorted(items_by_fname.values(), key=lambda x: x['timestamp'], reverse=True)
    return items


@app.route('/api/recordings/retained', methods=['GET'])
def get_retained_recordings():
    """보관 중인 wav 목록 (최근 RETENTION_HOURS 시간 + 화재 분류).
    detection_log.csv 와 실제 파일 존재를 매칭해서 반환.
    """
    try:
        limit = int(request.args.get('limit', 200))
    except (TypeError, ValueError):
        limit = 200
    limit = max(1, min(limit, 10000))
    only_fire = (request.args.get('fire_only') or '').lower() in ('1', 'true', 'yes')

    try:
        log_mtime = os.path.getmtime(DETECTION_LOG)
    except OSError:
        log_mtime = 0
    cache_key = log_mtime

    now = time.time()
    with _RETAINED_CACHE_LOCK:
        cached = _RETAINED_CACHE['data']
        if (cached is not None
                and _RETAINED_CACHE['key'] == cache_key
                and (now - _RETAINED_CACHE['ts']) < _RETAINED_CACHE_TTL):
            items = cached
        else:
            items = _build_retained_index()
            _RETAINED_CACHE['data'] = items
            _RETAINED_CACHE['key'] = cache_key
            _RETAINED_CACHE['ts'] = now

    if only_fire:
        filtered = [x for x in items if x['is_fire']]
    else:
        filtered = items
    fire_count = sum(1 for x in items if x['is_fire'])
    return jsonify({
        'retention_hours': RETENTION_HOURS,
        'total': len(filtered),
        'fire_count': fire_count,
        'limit': limit,
        'recordings': filtered[:limit],
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify(query_db("SELECT * FROM NotificationLogs ORDER BY timestamp DESC"))

@app.route('/api/system-settings', methods=['GET', 'POST'])
def system_settings():
    if request.method == 'POST':
        loc = request.json.get('location')
        execute_db("UPDATE SystemSettings SET location = ? WHERE id = 1", [loc])
        return jsonify({'success': True})
    
    row = query_db("SELECT location FROM SystemSettings WHERE id = 1", one=True)
    return jsonify(row)

@app.route('/api/notification-contacts', methods=['GET', 'POST'])
def manage_contacts():
    if request.method == 'POST':
        d = request.json
        execute_db("INSERT INTO NotificationContacts (name, phone, email) VALUES (?, ?, ?)", [d['name'], d['phone'], d['email']])
        return jsonify({'success': True}), 201
    return jsonify(query_db("SELECT * FROM NotificationContacts ORDER BY name"))

@app.route('/api/notification-contacts/<int:id>', methods=['PATCH', 'DELETE'])
def manage_contact_detail(id):
    if request.method == 'DELETE':
        execute_db("DELETE FROM NotificationContacts WHERE id = ?", [id])
    elif request.method == 'PATCH':
        execute_db("UPDATE NotificationContacts SET is_active = ? WHERE id = ?", [request.json.get('is_active'), id])
    return jsonify({'success': True})

def _get_local_ip():
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.connect(('8.8.8.8', 80))
        ip = sk.getsockname()[0]
        sk.close()
        return ip
    except Exception:
        return None


def _get_tailscale_ip():
    try:
        out = subprocess.check_output(['tailscale', 'ip', '-4'], timeout=2).decode().strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def _read_last_detection():
    try:
        with open(DETECTION_LOG, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode('utf-8', errors='replace').splitlines()
        if not tail:
            return None
        last = next(csv.reader([tail[-1]]))
        return {'timestamp': last[0], 'stage': last[2]} if len(last) >= 9 else None
    except Exception:
        return None


def _count_today_events(db_path):
    try:
        conn = sqlite3.connect(db_path)
        today = datetime.now().strftime('%Y-%m-%d')
        cur = conn.execute("SELECT COUNT(*) FROM Events WHERE timestamp LIKE ?", [f'{today}%'])
        n = cur.fetchone()[0]
        conn.close()
        return n
    except sqlite3.Error:
        return None


def build_status_report():
    h = get_health()
    last_det = _read_last_detection()
    today_events = _count_today_events(DATABASE)

    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        loc_row = conn.execute("SELECT location FROM SystemSettings WHERE id = 1").fetchone()
        st_row = conn.execute("SELECT status FROM DeviceStatus WHERE device_id = ?", [DEVICE_ID]).fetchone()
        conn.close()
        location = loc_row['location'] if loc_row else '미설정'
        device_status = st_row['status'] if st_row else 'normal'
    except sqlite3.Error:
        location, device_status = '미설정', 'unknown'

    cpu_temp = h.get('cpu_temp_c')
    cpu_temp_str = f"{cpu_temp:.1f}°C" if cpu_temp is not None else 'N/A'
    mic = '✅' if h.get('mic_ok') else ('❌' if h.get('mic_ok') is False else '–')
    ts_ip = _get_tailscale_ip() or '–'
    local_ip = _get_local_ip() or '–'

    last_det_str = (
        f"{last_det['timestamp']} ({last_det['stage']})" if last_det else '없음'
    )

    lines = [
        f"📊 <b>가드이어 상태 보고</b>",
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"📍 위치: {location}",
        f"🚦 장치: {'🚨 ALERT' if device_status == 'alert' else '🟢 정상'}",
        f"🎤 마이크: {mic}",
        f"🌡️ CPU: {cpu_temp_str}",
        f"🌐 IP: {local_ip} / TS: {ts_ip}",
        f"📝 마지막 감지: {last_det_str}",
        f"🔥 오늘 이벤트: {today_events if today_events is not None else '–'}건",
    ]
    return '\n'.join(lines)


def _send_dashboard_url(reason='boot'):
    """대시보드 접속 URL을 Telegram으로 알림. reason: 'boot' | 'ip_changed'."""
    tg = telegram_notify.load_telegram_settings()
    if not (tg.get('telegram_bot_token') and tg.get('telegram_chat_id')):
        return None
    ip = _get_local_ip()
    if not ip:
        return None
    url = f'http://{ip}:5000'
    if reason == 'boot':
        msg = (
            f"🟢 <b>가드이어 시작됨</b>\n"
            f"📱 대시보드: <a href=\"{url}\">{url}</a>\n"
            f"같은 Wi-Fi에서 위 주소로 접속하세요."
        )
    else:
        msg = (
            f"♻️ <b>IP 변경 감지</b>\n"
            f"📱 새 대시보드 주소: <a href=\"{url}\">{url}</a>"
        )
    ok, err = telegram_notify.send_message(msg, settings=tg)
    if not ok:
        print(f"[Telegram URL 알림] 실패: {err}")
    return ip


def status_reporter_loop():
    # 부팅 직후: 네트워크가 잡힐 때까지 잠깐 대기 후 URL 발송
    time.sleep(15)
    last_ip = _send_dashboard_url(reason='boot')

    # IP 변경 감지: 60초 주기로 확인. 30분마다는 정기 상태 보고.
    last_status_t = time.time()
    while True:
        time.sleep(60)
        try:
            cur_ip = _get_local_ip()
            if cur_ip and cur_ip != last_ip:
                print(f"[IP 변경] {last_ip} → {cur_ip}")
                last_ip = _send_dashboard_url(reason='ip_changed') or cur_ip

            if time.time() - last_status_t >= STATUS_REPORT_INTERVAL_SEC:
                tg = telegram_notify.load_telegram_settings()
                if tg.get('telegram_bot_token') and tg.get('telegram_chat_id'):
                    ok, err = telegram_notify.send_message(build_status_report(), settings=tg)
                    if not ok:
                        print(f"[Telegram 상태 보고] 실패: {err}")
                last_status_t = time.time()
        except Exception as e:
            print(f"[Telegram 보고 루프] 예외: {e}")


if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        import database
        database.create_tables()

    threading.Thread(target=status_reporter_loop, daemon=True).start()
    threading.Thread(target=recordings_cleanup_loop, daemon=True).start()
    telegram_notify.start_poll_thread()

    app.run(host='0.0.0.0', port=5000)