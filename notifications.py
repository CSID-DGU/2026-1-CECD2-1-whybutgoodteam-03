import os
import smtplib
import requests
import sqlite3
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 환경 변수 로드 ---
GMAIL_USER = os.environ.get('GMAIL_USER')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')

NHN_APP_KEY = os.environ.get('NHN_APP_KEY')
NHN_SECRET_KEY = os.environ.get('NHN_SECRET_KEY')
NHN_SENDER_NUMBER = os.environ.get('NHN_SENDER_NUMBER')

# --- DB 연결 ---
DATABASE = 'gard-ear.db'

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
def send_email(recipient_name, recipient_email, message_body, log_id):
    """
    Gmail 발송 후 결과를 DB에 업데이트합니다.
    """
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        update_notification_result(log_id, 'failed', "GMAIL 환경 변수 미설정")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = recipient_email
        msg['Subject'] = "[긴급] 가드이어 화재 경보"
        msg.attach(MIMEText(message_body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        update_notification_result(log_id, 'success')

    except Exception as e:
        update_notification_result(log_id, 'failed', str(e))

# --- SMS 발송 (NHN Cloud) ---
def send_sms(recipient_name, recipient_phone, message_body, log_id):
    """
    SMS 발송 후 결과를 DB에 업데이트합니다.
    """
    if not NHN_APP_KEY or not NHN_SECRET_KEY or not NHN_SENDER_NUMBER:
        update_notification_result(log_id, 'failed', "NHN Cloud 환경 변수 미설정")
        return

    url = f"https://api-sms.cloud.toast.com/sms/v3.0/appKeys/{NHN_APP_KEY}/sender/sms"
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'X-Secret-Key': NHN_SECRET_KEY
    }
    data = {
        "body": message_body,
        "sendNo": NHN_SENDER_NUMBER,
        "recipientList": [{"recipientNo": recipient_phone}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        if result.get("header", {}).get("isSuccessful", False):
            update_notification_result(log_id, 'success')
        else:
            error_msg = result.get("header", {}).get("resultMessage", "Unknown NHN Error")
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

    for contact in contacts:
        name = contact['name']
        phone = contact.get('phone')
        email = contact.get('email')
        
        # 1. SMS 발송 프로세스
        if phone:
            # (A) 대기 상태로 먼저 기록
            log_id = log_notification_start(name, 'sms', message)
            # (B) 실제 발송 (내부에서 성공/실패 업데이트)
            send_sms(name, phone, message, log_id)
        
        # 2. 이메일 발송 프로세스
        if email:
            log_id = log_notification_start(name, 'email', message)
            send_email(name, email, message, log_id)
            
    print("[알림 작업 완료]")