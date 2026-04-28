import smtplib
import requests
import sqlite3
import time
import hmac
import hashlib
import secrets
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SOLAPI_ENDPOINT = "https://api.solapi.com/messages/v4/send"

# --- DB 연결 ---
DATABASE = 'gard-ear.db'

def load_sender_settings():
    """SystemSettings에서 Gmail/Solapi 발신자 설정을 읽어옵니다."""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT gmail_user, gmail_password, solapi_api_key, solapi_api_secret, solapi_sender_number "
            "FROM SystemSettings WHERE id = 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except sqlite3.Error as e:
        print(f"!!! 발신자 설정 로드 실패: {e}")
        return {}

def log_notification_start(recipient_name, type, message):
    """
    [1단계] 발송 시작 전 'pending'(대기 중) 상태로 로그를 생성하고 ID를 반환합니다.
    """
    print(f"[Log] {recipient_name}에게 {type} 발송 시작 (Pending)...")
    try:
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO NotificationLogs (recipient_name, type, status, message) VALUES (?, ?, 'pending', ?)",
            (recipient_name, type, message)
        )
        conn.commit()
        log_id = cur.lastrowid # 생성된 로그의 ID
        conn.close()
        return log_id
    except sqlite3.Error as e:
        print(f"!!! DB 로그 시작 실패: {e}")
        return None

def update_notification_result(log_id, status, error_message=None):
    """
    [2단계] 발송 완료 후 결과를 업데이트합니다.
    """
    if not log_id: return
    print(f"[Log] 로그 ID {log_id} 상태 업데이트 -> {status}")
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute(
            "UPDATE NotificationLogs SET status = ?, error_message = ? WHERE id = ?",
            (status, error_message, log_id)
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"!!! DB 로그 업데이트 실패: {e}")


# --- 이메일 발송 (Gmail) ---
def send_email(recipient_name, recipient_email, message_body, log_id, settings=None):
    """
    Gmail 발송 후 결과를 DB에 업데이트합니다. 설정은 SystemSettings(DB)에서 로드.
    """
    if settings is None:
        settings = load_sender_settings()
    gmail_user = settings.get('gmail_user')
    gmail_password = settings.get('gmail_password')

    if not gmail_user or not gmail_password:
        update_notification_result(log_id, 'failed', "Gmail 발신자 설정 미입력 (관리자 설정에서 등록 필요)")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = recipient_email
        msg['Subject'] = "[긴급] 가드이어 화재 경보"
        msg.attach(MIMEText(message_body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()

        update_notification_result(log_id, 'success')

    except Exception as e:
        update_notification_result(log_id, 'failed', str(e))

# --- SMS 발송 (Solapi) ---
def _solapi_auth_header(api_key, api_secret):
    date = datetime.now(timezone.utc).isoformat()
    salt = secrets.token_hex(32)
    signature = hmac.new(
        api_secret.encode('utf-8'),
        (date + salt).encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return f"HMAC-SHA256 apiKey={api_key}, date={date}, salt={salt}, signature={signature}"

def send_sms(recipient_name, recipient_phone, message_body, log_id, settings=None):
    """
    Solapi로 SMS 발송 후 결과를 DB에 업데이트합니다. 설정은 SystemSettings(DB)에서 로드.
    """
    if settings is None:
        settings = load_sender_settings()
    api_key = settings.get('solapi_api_key')
    api_secret = settings.get('solapi_api_secret')
    sender_number = settings.get('solapi_sender_number')

    if not api_key or not api_secret or not sender_number:
        update_notification_result(log_id, 'failed', "Solapi 발신자 설정 미입력 (관리자 설정에서 등록 필요)")
        return

    headers = {
        'Authorization': _solapi_auth_header(api_key, api_secret),
        'Content-Type': 'application/json;charset=UTF-8'
    }
    data = {
        "message": {
            "to": recipient_phone,
            "from": sender_number,
            "text": message_body
        }
    }

    try:
        response = requests.post(SOLAPI_ENDPOINT, headers=headers, json=data, timeout=10)
        result = response.json()

        # Solapi 성공 statusCode: "2000" (요청 접수 성공)
        if response.ok and result.get("statusCode", "").startswith("2"):
            update_notification_result(log_id, 'success')
        else:
            error_msg = result.get("statusMessage") or result.get("errorMessage") or f"HTTP {response.status_code}"
            update_notification_result(log_id, 'failed', error_msg)

    except Exception as e:
        update_notification_result(log_id, 'failed', str(e))


# --- 메인 알림 작업 ---
def send_notification_task(location):
    print(f"[알림 작업 시작] 위치: {location}")
    
    now = time.localtime()
    date_str = time.strftime('%Y-%m-%d', now)
    time_str = time.strftime('%H:%M:%S', now)
    message = f"[{date_str}, {location}, {time_str}] 화재 경보가 감지되었습니다. 즉시 확인하세요."
    
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM NotificationContacts WHERE is_active = 1")
        contacts = [dict(row) for row in cur.fetchall()]
        cur.close()
        conn.close()
    except sqlite3.Error as e:
        print(f"!!! 알림 작업 실패: DB에서 연락처를 가져올 수 없습니다. {e}")
        return

    if not contacts:
        print("[알림 작업] 발송할 연락처 없음")
        return

    sender_settings = load_sender_settings()

    for contact in contacts:
        name = contact['name']
        phone = contact.get('phone')
        email = contact.get('email')

        # 1. SMS 발송 프로세스
        if phone:
            log_id = log_notification_start(name, 'sms', message)
            send_sms(name, phone, message, log_id, settings=sender_settings)

        # 2. 이메일 발송 프로세스
        if email:
            log_id = log_notification_start(name, 'email', message)
            send_email(name, email, message, log_id, settings=sender_settings)
            
    print("[알림 작업 완료]")