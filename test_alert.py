import requests
import time

# app.py 서버가 실행 중인 주소
# 127.0.0.1 (localhost)는 이 스크립트를 서버와 같은 장치에서 실행할 때 사용합니다.
SERVER_URL = "http://127.0.0.1:5000"
API_ENDPOINT = f"{SERVER_URL}/api/events"

def send_test_alert():
    """
    app.py 서버의 /api/events 엔드포인트로 '화재 감지' POST 요청을 보냅니다.
    """
    print(f"--- 가드이어 화재 감지 테스트 ---")
    print(f"'{API_ENDPOINT}' 주소로 '화재 감지' 신호를 보냅니다...")

    try:
        # 이 부분이 감지팀이 나중에 구현할 '신호 전송'의 핵심입니다.
        response = requests.post(API_ENDPOINT, timeout=10)

        # 서버로부터의 응답 확인
        if response.status_code == 201:
            print("\n[성공] 서버가 신호를 성공적으로 수신했습니다.")
            print("서버가 알림(이메일 등) 발송을 시도합니다.")
            print("지금 바로 받은 편지함과 대시보드 UI를 확인하세요!")
        
        elif response.status_code == 400:
            print(f"\n[실패] 서버가 요청을 거부했습니다 (400 Bad Request)")
            print(f"서버 응답: {response.json()}")
            print("힌트: '설정' 탭에서 '설치 위치'를 먼저 저장했는지 확인하세요.")

        else:
            print(f"\n[실패] 서버에서 오류가 발생했습니다 (Status Code: {response.status_code})")
            print(f"서버 응답: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"\n[오류] '{SERVER_URL}'에 연결할 수 없습니다.")
        print("힌트: app.py 서버가 다른 터미널에서 실행 중인지 확인하세요.")
    
    except Exception as e:
        print(f"\n[알 수 없는 오류] {e}")

if __name__ == "__main__":
    send_test_alert()