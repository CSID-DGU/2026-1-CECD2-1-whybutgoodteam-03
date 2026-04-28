"""Telegram 봇 — 다중 수신자 broadcast + /start /stop /status /help 자동 등록 폴러."""
import os
import sqlite3
import time
import wave
import glob
import threading
import requests

DATABASE = 'gard-ear.db'
TG_API = 'https://api.telegram.org/bot{token}/{method}'
RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'records')


# ============================================================
#  설정 / 수신자 DB
# ============================================================

def load_telegram_settings():
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT telegram_bot_token, telegram_chat_id, telegram_update_offset "
            "FROM SystemSettings WHERE id = 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except sqlite3.Error as e:
        print(f"[Telegram] 설정 로드 실패: {e}")
        return {}


def _get_token():
    return load_telegram_settings().get('telegram_bot_token') or ''


def list_active_subscribers():
    try:
        conn = sqlite3.connect(DATABASE)
        rows = conn.execute(
            "SELECT chat_id, name, chat_type FROM TelegramSubscribers WHERE is_active = 1"
        ).fetchall()
        conn.close()
        return [{'chat_id': r[0], 'name': r[1], 'chat_type': r[2]} for r in rows]
    except sqlite3.Error as e:
        print(f"[Telegram] 수신자 조회 실패: {e}")
        return []


def upsert_subscriber(chat_id, name, chat_type):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute(
            "INSERT INTO TelegramSubscribers (chat_id, name, chat_type, is_active) "
            "VALUES (?, ?, ?, 1) "
            "ON CONFLICT(chat_id) DO UPDATE SET name=excluded.name, chat_type=excluded.chat_type, is_active=1",
            (chat_id, name, chat_type),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"[Telegram] 수신자 등록 실패: {e}")
        return False


def deactivate_subscriber(chat_id):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute(
            "UPDATE TelegramSubscribers SET is_active = 0 WHERE chat_id = ?",
            (chat_id,),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"[Telegram] 수신자 해제 실패: {e}")
        return False


def _get_offset():
    try:
        conn = sqlite3.connect(DATABASE)
        row = conn.execute("SELECT telegram_update_offset FROM SystemSettings WHERE id = 1").fetchone()
        conn.close()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error:
        return 0


def _save_offset(offset):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute("UPDATE SystemSettings SET telegram_update_offset = ? WHERE id = 1", (offset,))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[Telegram] offset 저장 실패: {e}")


# ============================================================
#  메시지 발송 (단일 / broadcast)
# ============================================================

def send_message_to(chat_id, text, parse_mode='HTML', token=None):
    tok = token or _get_token()
    if not tok:
        return False, 'token 미설정'
    try:
        r = requests.post(
            TG_API.format(token=tok, method='sendMessage'),
            data={'chat_id': chat_id, 'text': text, 'parse_mode': parse_mode,
                  'disable_web_page_preview': True},
            timeout=10,
        )
        ok = r.ok and r.json().get('ok')
        return ok, '' if ok else r.text[:200]
    except Exception as e:
        return False, str(e)


def send_audio_to(chat_id, file_path, caption='', token=None):
    tok = token or _get_token()
    if not tok:
        return False, 'token 미설정'
    if not os.path.isfile(file_path):
        return False, f'파일 없음: {file_path}'
    try:
        with open(file_path, 'rb') as f:
            r = requests.post(
                TG_API.format(token=tok, method='sendAudio'),
                data={'chat_id': chat_id, 'caption': caption},
                files={'audio': (os.path.basename(file_path), f, 'audio/wav')},
                timeout=60,
            )
        ok = r.ok and r.json().get('ok')
        return ok, '' if ok else r.text[:200]
    except Exception as e:
        return False, str(e)


def send_message(text, settings=None, parse_mode='HTML'):
    """모든 활성 수신자에게 broadcast. (호환을 위해 (ok, err) 리턴: 한 명이라도 성공하면 ok=True)"""
    tok = (settings or {}).get('telegram_bot_token') or _get_token()
    if not tok:
        return False, 'token 미설정'
    subs = list_active_subscribers()
    if not subs:
        return False, '수신자 없음'
    any_ok = False
    errs = []
    for s in subs:
        ok, err = send_message_to(s['chat_id'], text, parse_mode=parse_mode, token=tok)
        if ok:
            any_ok = True
        else:
            errs.append(f"{s['chat_id']}:{err}")
    return any_ok, ';'.join(errs)[:300]


def send_audio(file_path, caption='', settings=None):
    tok = (settings or {}).get('telegram_bot_token') or _get_token()
    if not tok:
        return False, 'token 미설정'
    subs = list_active_subscribers()
    if not subs:
        return False, '수신자 없음'
    any_ok = False
    errs = []
    for s in subs:
        ok, err = send_audio_to(s['chat_id'], file_path, caption=caption, token=tok)
        if ok:
            any_ok = True
        else:
            errs.append(f"{s['chat_id']}:{err}")
    return any_ok, ';'.join(errs)[:300]


# ============================================================
#  최근 1분 녹음 모음 (변동 없음)
# ============================================================

def collect_recent_audio(out_path, target_seconds=60):
    files = sorted(
        glob.glob(os.path.join(RECORDS_DIR, '*.wav')),
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        return False, '녹음 파일 없음'
    n = max(1, target_seconds // 4)
    needed = files[: n * 2 : 2][::-1]
    try:
        with wave.open(needed[0], 'rb') as w:
            params = w.getparams()
        with wave.open(out_path, 'wb') as out:
            out.setparams(params)
            for p in needed:
                try:
                    with wave.open(p, 'rb') as w:
                        out.writeframes(w.readframes(w.getnframes()))
                except Exception:
                    continue
        return True, f'{len(needed)}개 파일, {len(needed) * 4}초'
    except Exception as e:
        return False, str(e)


# ============================================================
#  봇 명령어 폴러 (/start /stop /status /help)
# ============================================================

HELP_TEXT = (
    "📖 <b>가드이어 봇 명령어</b>\n"
    "/start — 알림 받기\n"
    "/stop — 알림 끄기\n"
    "/status — 현재 시스템 상태\n"
    "/help — 도움말"
)


def _chat_label(chat):
    if chat.get('title'):
        return chat['title']
    name = f"{chat.get('first_name','')} {chat.get('last_name','')}".strip()
    return name or f"chat:{chat.get('id')}"


def _strip_bot_mention(cmd):
    return cmd.split('@', 1)[0]


def _process_message(msg):
    chat = msg.get('chat') or {}
    chat_id = chat.get('id')
    if chat_id is None:
        return
    text = (msg.get('text') or '').strip()
    chat_type = chat.get('type', 'private')
    name = _chat_label(chat)

    # 명령어 추출
    cmd = ''
    if text and text.startswith('/'):
        cmd = _strip_bot_mention(text.split()[0]).lower()

    if cmd == '/start':
        upsert_subscriber(chat_id, name, chat_type)
        send_message_to(
            chat_id,
            f"✅ <b>{name}</b> 알림 등록 완료\n"
            "화재 경보·시스템 상태가 이 채팅으로 전송됩니다.\n\n" + HELP_TEXT,
        )
    elif cmd == '/stop':
        deactivate_subscriber(chat_id)
        send_message_to(chat_id, "🔕 알림 해제됨. 다시 받으려면 /start")
    elif cmd == '/status':
        try:
            from app import build_status_report  # 지연 import (순환 방지)
            send_message_to(chat_id, build_status_report())
        except Exception as e:
            send_message_to(chat_id, f"상태 조회 실패: {e}")
    elif cmd == '/help':
        send_message_to(chat_id, HELP_TEXT)
    # 그 외 메시지는 무시


def _process_my_chat_member(upd):
    """봇이 그룹에 추가/제거되었을 때 자동 등록/해제."""
    new_member = upd.get('new_chat_member') or {}
    status = new_member.get('status')
    chat = upd.get('chat') or {}
    chat_id = chat.get('id')
    if chat_id is None:
        return
    if status in ('member', 'administrator'):
        upsert_subscriber(chat_id, _chat_label(chat), chat.get('type', 'group'))
        send_message_to(chat_id, f"✅ 그룹 <b>{_chat_label(chat)}</b> 알림 등록\n" + HELP_TEXT)
    elif status in ('left', 'kicked', 'restricted'):
        deactivate_subscriber(chat_id)


def _process_update(upd):
    if 'message' in upd:
        _process_message(upd['message'])
    elif 'my_chat_member' in upd:
        _process_my_chat_member(upd['my_chat_member'])


def poll_loop():
    """Telegram getUpdates long-poll. 봇 토큰이 없으면 루프 진입 안 함."""
    print("[Telegram poll] 시작")
    while True:
        tok = _get_token()
        if not tok:
            time.sleep(30)
            continue
        offset = _get_offset()
        try:
            r = requests.get(
                TG_API.format(token=tok, method='getUpdates'),
                params={
                    'timeout': 25,
                    'offset': offset,
                    'allowed_updates': '["message","my_chat_member"]',
                },
                timeout=30,
            )
            data = r.json()
            if not data.get('ok'):
                print(f"[Telegram poll] getUpdates 실패: {data}")
                time.sleep(5)
                continue
            for u in data.get('result', []):
                try:
                    _process_update(u)
                except Exception as e:
                    print(f"[Telegram poll] update 처리 오류: {e}")
                _save_offset(u['update_id'] + 1)
        except requests.RequestException as e:
            print(f"[Telegram poll] 네트워크 오류: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"[Telegram poll] 예외: {e}")
            time.sleep(5)


def start_poll_thread():
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    return t
