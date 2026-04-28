"""Telegram bot: 화재 경보 알림 + 30분 주기 상태 보고."""
import os
import sqlite3
import wave
import glob
import requests

DATABASE = 'gard-ear.db'
TG_API = 'https://api.telegram.org/bot{token}/{method}'
RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'records')


def load_telegram_settings():
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT telegram_bot_token, telegram_chat_id FROM SystemSettings WHERE id = 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except sqlite3.Error as e:
        print(f"[Telegram] 설정 로드 실패: {e}")
        return {}


def _is_configured(s):
    return bool(s.get('telegram_bot_token') and s.get('telegram_chat_id'))


def send_message(text, settings=None, parse_mode='HTML'):
    s = settings if settings is not None else load_telegram_settings()
    if not _is_configured(s):
        return False, 'telegram 미설정'
    try:
        r = requests.post(
            TG_API.format(token=s['telegram_bot_token'], method='sendMessage'),
            data={'chat_id': s['telegram_chat_id'], 'text': text, 'parse_mode': parse_mode},
            timeout=10,
        )
        ok = r.ok and r.json().get('ok')
        return ok, '' if ok else r.text[:200]
    except Exception as e:
        return False, str(e)


def send_audio(file_path, caption='', settings=None):
    s = settings if settings is not None else load_telegram_settings()
    if not _is_configured(s):
        return False, 'telegram 미설정'
    if not os.path.isfile(file_path):
        return False, f'파일 없음: {file_path}'
    try:
        with open(file_path, 'rb') as f:
            r = requests.post(
                TG_API.format(token=s['telegram_bot_token'], method='sendAudio'),
                data={'chat_id': s['telegram_chat_id'], 'caption': caption},
                files={'audio': (os.path.basename(file_path), f, 'audio/wav')},
                timeout=60,
            )
        ok = r.ok and r.json().get('ok')
        return ok, '' if ok else r.text[:200]
    except Exception as e:
        return False, str(e)


def collect_recent_audio(out_path, target_seconds=60):
    """최근 녹음을 이어붙여 ~target_seconds 분량 wav 생성.
    각 파일은 4초 길이, 2초 슬라이딩 → 인접 파일은 2초씩 겹친다.
    겹침을 피하려고 짝수 인덱스만 골라 사용한다 (4초 × N개 = 4N초).
    """
    files = sorted(
        glob.glob(os.path.join(RECORDS_DIR, '*.wav')),
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        return False, '녹음 파일 없음'

    # 최근 2N개 중 짝수 인덱스(0,2,4,...) → N개. 시간 정방향(오래된 → 최신)으로 정렬.
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
