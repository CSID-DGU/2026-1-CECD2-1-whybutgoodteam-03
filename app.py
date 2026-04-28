import sqlite3
import os
import atexit
import threading
import time
import csv
from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS
from notifications import send_notification_task, send_manual_sms_task
from health import get_health

DETECTION_LOG = 'logs/detection_log.csv'

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
                      'solapi_api_key', 'solapi_api_secret', 'solapi_sender_number'):
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
            'solapi_sender_number_preview': row.get('solapi_sender_number') or '',
            'gmail_user_set': bool(row.get('gmail_user')),
            'gmail_password_set': bool(row.get('gmail_password')),
            'solapi_api_key_set': bool(row.get('solapi_api_key')),
            'solapi_api_secret_set': bool(row.get('solapi_api_secret')),
            'solapi_sender_number_set': bool(row.get('solapi_sender_number')),
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

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        import database
        database.create_tables()
    app.run(host='0.0.0.0', port=5000)