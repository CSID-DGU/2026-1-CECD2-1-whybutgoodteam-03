import sqlite3
import os
import atexit
import threading
from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS
from notifications import send_notification_task

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
            
        # 발신자 정보 업데이트
        execute_db("""
            UPDATE SystemSettings 
            SET gmail_user=?, gmail_password=?, 
                nhn_app_key=?, nhn_secret_key=?, nhn_sender_number=?
            WHERE id=1
        """, [d['gmail_user'], d['gmail_password'], d['nhn_app_key'], d['nhn_secret_key'], d['nhn_sender_number']])
        
        return jsonify({'success': True})

    # GET 요청: 현재 설정값 반환 (보안상 비밀번호는 마스킹하거나 빈값으로 줄 수도 있음)
    row = query_db("SELECT * FROM SystemSettings WHERE id = 1", one=True)
    return jsonify(row)

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