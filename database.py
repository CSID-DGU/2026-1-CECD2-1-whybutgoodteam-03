import sqlite3
import os

DATABASE = 'gard-ear.db'

def create_tables():
    """
    모든 테이블을 생성합니다.
    v6 변경 사항: SystemSettings에 admin_password + 발신자(Gmail/Solapi) 설정 컬럼 추가
    """
    if os.path.exists(DATABASE):
        print(f"기존 {DATABASE} 파일을 삭제합니다.")
        os.remove(DATABASE)

    print(f"{DATABASE} 파일을 새로 생성합니다...")
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # 1. 장치 상태 테이블
    c.execute('''
    CREATE TABLE DeviceStatus (
        device_id TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'normal' CHECK(status IN ('normal', 'alert'))
    )
    ''')
    c.execute("INSERT INTO DeviceStatus (device_id, status) VALUES ('rasp_pi_main', 'normal')")

    # 2. 화재 이벤트 로그 테이블
    c.execute('''
    CREATE TABLE Events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        location TEXT,
        timestamp DATETIME DEFAULT (datetime('now','localtime'))
    )
    ''')

    # 3. 알림 연락처 테이블
    c.execute('''
    CREATE TABLE NotificationContacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT,
        email TEXT,
        is_active BOOLEAN NOT NULL DEFAULT 1
    )
    ''')

    # 4. 알림 발송 내역 테이블
    c.execute('''
    CREATE TABLE NotificationLogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient_name TEXT NOT NULL,
        type TEXT NOT NULL CHECK(type IN ('email', 'sms')),
        status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'pending')),
        message TEXT,
        error_message TEXT,
        timestamp DATETIME DEFAULT (datetime('now','localtime'))
    )
    ''')

    # 5. 시스템 설정 테이블 (v7: 관리자 비밀번호 + 발신자 정보 + Telegram)
    c.execute('''
    CREATE TABLE SystemSettings (
        id INTEGER PRIMARY KEY CHECK(id = 1),
        location TEXT DEFAULT '미설정',
        admin_password TEXT DEFAULT '1234',
        gmail_user TEXT DEFAULT '',
        gmail_password TEXT DEFAULT '',
        solapi_api_key TEXT DEFAULT '',
        solapi_api_secret TEXT DEFAULT '',
        solapi_sender_number TEXT DEFAULT '',
        telegram_bot_token TEXT DEFAULT '',
        telegram_chat_id TEXT DEFAULT ''
    )
    ''')
    c.execute("INSERT INTO SystemSettings (id) VALUES (1)")

    conn.commit()
    conn.close()
    print(f"데이터베이스 테이블 생성이 완료되었습니다. ({DATABASE})")

if __name__ == '__main__':
    create_tables()
