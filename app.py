import sqlite3
import os
import atexit
import socket
import subprocess
import threading
import time
import csv
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, g, send_from_directory, abort
from flask_cors import CORS
from notifications import send_notification_task, send_manual_sms_task
from health import get_health
import telegram_notify

STATUS_REPORT_INTERVAL_SEC = 30 * 60  # 30분

DETECTION_LOG = 'logs/detection_log.csv'
RECORDINGS_DIR = os.path.abspath('records')
RECORDING_WINDOW_SEC = 10  # 이벤트 타임스탬프 ±N초의 녹음본을 매칭

DATABASE = 'gard-ear.db'
DEVICE_ID = 'rasp_pi_main'

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
    execute_db("UPDATE DeviceStatus SET status = 'alert' WHERE device_id = ?", [DEVICE_ID])
    settings = query_db("SELECT location FROM SystemSettings WHERE id = 1", one=True)
    location = settings.get('location', '미설정') if settings else '미설정'
    execute_db("INSERT INTO Events (device_id, event_type, location) VALUES (?, 'fire_alarm_detected', ?)", [DEVICE_ID, location])
    
    threading.Thread(target=send_notification_task, args=(location,)).start()
    return jsonify({'message': 'Alert triggered.'}), 201

@app.route('/api/acknowledge', methods=['POST'])
def acknowledge_event():
    execute_db("UPDATE DeviceStatus SET status = 'normal' WHERE device_id = ?", [DEVICE_ID])
    return jsonify({'message': 'Alert acknowledged.'}), 200

@app.route('/api/events', methods=['GET'])
def get_events():
    return jsonify(query_db("SELECT * FROM Events ORDER BY timestamp DESC"))


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

    lo = ev_ts - timedelta(seconds=RECORDING_WINDOW_SEC)
    hi = ev_ts + timedelta(seconds=RECORDING_WINDOW_SEC)

    items = []
    try:
        with open(DETECTION_LOG, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts_str = r.get('timestamp', '').strip()
                if not ts_str:
                    continue
                try:
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
                if not (lo <= ts <= hi):
                    continue
                fname = (r.get('filename') or '').strip()
                if not fname:
                    continue
                exists = os.path.isfile(os.path.join(RECORDINGS_DIR, fname))
                items.append({
                    'filename': fname,
                    'timestamp': ts_str,
                    'stage': r.get('stage'),
                    'rule_score': float(r['rule_score']) if r.get('rule_score') else None,
                    'pred_label': r.get('pred_label') or None,
                    'pred_prob': float(r['pred_prob']) if r.get('pred_prob') else None,
                    'is_fire': r.get('is_fire') == '1',
                    'exists': exists,
                })
    except FileNotFoundError:
        return jsonify({'event_timestamp': row['timestamp'], 'recordings': []})

    return jsonify({'event_timestamp': row['timestamp'], 'recordings': items})


@app.route('/api/detections/recent', methods=['GET'])
def get_recent_detections():
    """최근 N초 내 detection_log 행 + wav 존재 여부."""
    try:
        seconds = int(request.args.get('seconds', 60))
    except (TypeError, ValueError):
        seconds = 60
    seconds = max(1, min(seconds, 600))
    cutoff = datetime.now() - timedelta(seconds=seconds)

    items = []
    try:
        with open(DETECTION_LOG, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts_str = (r.get('timestamp') or '').strip()
                try:
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                fname = (r.get('filename') or '').strip()
                items.append({
                    'filename': fname,
                    'timestamp': ts_str,
                    'stage': r.get('stage'),
                    'rule_score': float(r['rule_score']) if r.get('rule_score') else None,
                    'pred_label': r.get('pred_label') or None,
                    'pred_prob': float(r['pred_prob']) if r.get('pred_prob') else None,
                    'is_fire': r.get('is_fire') == '1',
                    'exists': bool(fname) and os.path.isfile(os.path.join(RECORDINGS_DIR, fname)),
                })
    except FileNotFoundError:
        pass

    items.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'seconds': seconds, 'recordings': items})


@app.route('/api/recordings/<path:filename>', methods=['GET'])
def get_recording(filename):
    # 경로 탈출 방지: 파일명에 슬래시/.. 차단, .wav만 허용
    if '/' in filename or '\\' in filename or '..' in filename or not filename.lower().endswith('.wav'):
        abort(400)
    return send_from_directory(RECORDINGS_DIR, filename, mimetype='audio/wav', conditional=True)

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


def status_reporter_loop():
    # 부팅 직후 30초 뒤 첫 보고 → 그 다음부터 30분 주기
    time.sleep(30)
    while True:
        try:
            tg = telegram_notify.load_telegram_settings()
            if tg.get('telegram_bot_token') and tg.get('telegram_chat_id'):
                ok, err = telegram_notify.send_message(build_status_report(), settings=tg)
                if not ok:
                    print(f"[Telegram 상태 보고] 실패: {err}")
        except Exception as e:
            print(f"[Telegram 상태 보고] 예외: {e}")
        time.sleep(STATUS_REPORT_INTERVAL_SEC)


if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        import database
        database.create_tables()

    threading.Thread(target=status_reporter_loop, daemon=True).start()

    app.run(host='0.0.0.0', port=5000)