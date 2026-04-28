from notifications import send_email, load_sender_settings, log_notification_start

def test_gmail():
    """
    Gmail 발송 기능만 독립적으로 테스트합니다. 발신자 정보는 SystemSettings(DB)에서 로드.
    """
    print("--- 가드이어 Gmail 발송 테스트 ---")

    settings = load_sender_settings()
    gmail_user = settings.get('gmail_user')

    if not gmail_user or not settings.get('gmail_password'):
        print("\n[오류] DB에 Gmail 발신자 정보가 저장되어 있지 않습니다.")
        print("대시보드 관리자 설정 화면에서 Gmail 주소와 앱 비밀번호를 등록해주세요.")
        return

    print(f"\n[확인] 발신자 계정: {gmail_user}")

    recipient_email = input("테스트 메일을 받을 이메일 주소를 입력하세요: ").strip()
    if not recipient_email:
        print("이메일이 입력되지 않아 테스트를 종료합니다.")
        return

    print(f"\n{recipient_email} (으)로 테스트 메일을 발송합니다...")

    body = "이 메일이 성공적으로 수신되었다면, Gmail SMTP 연동에 성공한 것입니다."
    log_id = log_notification_start("테스트 수신자", "email", body)
    send_email("테스트 수신자", recipient_email, body, log_id, settings=settings)
    print(f"\n발송 시도 완료. 로그 ID {log_id} (NotificationLogs 테이블에서 결과 확인)")

if __name__ == "__main__":
    test_gmail()
