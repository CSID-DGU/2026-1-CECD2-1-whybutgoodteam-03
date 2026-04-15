import os
from notifications import send_email # notifications.py 파일에서 send_email 함수 가져오기

def test_gmail():
    """
    Gmail 발송 기능만 독립적으로 테스트합니다.
    """
    print("--- 가드이어 Gmail 발송 테스트 ---")

    # 1. 환경 변수에서 Gmail 정보 읽기
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_pass = os.environ.get('GMAIL_APP_PASSWORD')

    if not gmail_user or not gmail_pass:
        print("\n[오류] GMAIL_USER 또는 GMAIL_APP_PASSWORD 환경 변수가 설정되지 않았습니다.")
        print("README.md 파일을 참고하여 'set' 명령어로 환경 변수를 먼저 설정해주세요.")
        print("예: set GMAIL_USER=\"your-email@gmail.com\"")
        return

    print(f"\n[확인] 발신자 계정: {gmail_user}")

    # 2. 테스트 메일을 받을 이메일 주소 입력받기
    recipient_email = input("테스트 메일을 받을 이메일 주소를 입력하세요 (예: my-test@gmail.com): ")
    if not recipient_email:
        print("이메일이 입력되지 않아 테스트를 종료합니다.")
        return

    # 3. 테스트 메일 발송
    print(f"\n{recipient_email} (으)로 테스트 메일을 발송합니다...")
    
    subject = "[가드이어] Gmail 연동 테스트"
    body = "이 메일이 성공적으로 수신되었다면, Gmail SMTP 연동에 성공한 것입니다."
    
    status = send_email(recipient_email, subject, body)
    
    if status == 'success':
        print("\n[성공] 테스트 메일이 성공적으로 발송되었습니다!")
        print("입력하신 이메일의 받은 편지함을 확인해주세요.")
    else:
        print("\n[실패] 메일 발송에 실패했습니다. 터미널의 오류 메시지를 확인하세요.")
        print("(오류 예: 2단계 인증, 앱 비밀번호, GMAIL_USER 환경 변수 설정 등)")

if __name__ == "__main__":
    test_gmail()