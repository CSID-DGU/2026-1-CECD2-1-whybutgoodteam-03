# 가드이어 (Guard-Ear)

마이크로 화재 경보음을 듣고 관리자에게 문자 및 이메일로 알려주는 라즈베리파이 기반 시스템입니다.

전북개발공사 공공임대주택의 노후 화재수신기를 직접 손대지 않고 비접촉으로 경보음만 인식해서 알림을 띄우는 게 목적이고, 2026년 4월 30일에 군산 금광지구 행복주택에 1차 설치를 끝냈습니다. 동국대 컴퓨터AI학부 종합설계 2 왜인진모르겠지만잘하는팀으로 진행 중이고, 같은 결과물을 정리해서 KCC2026 학부생/주니어 논문경진대회에 제출했습니다.

## 어떻게 동작하나요

마이크로 4초씩 오디오를 모아서 다음 단계를 거칩니다.

1. 룰 프리필터 — dB + 600~2500Hz 대역 에너지 비율 + 자기상관 주기성, 셋을 0.3 / 0.2 / 0.5 가중치로 합쳐 점수가 임계값 이상이면 다음 단계로 보냅니다. 조용하거나 화재 경보음 대역이 아닌 구간은 여기서 끝나서, 평소엔 딥러닝을 거의 안 돌립니다.
2. 딥러닝 분류 — ESResNeXt-fbsp 백본으로 `other / emergency / fire_alarm` 3-class softmax를 뽑습니다.
3. outdoor 사후 룰 — fire_alarm으로 찍힌 구간 중 가스경보·백업알람·전자 차임벨처럼 단일 초고주파 순음인 경우는 별도 룰(welch 스펙트럼에서 peak > 3500Hz, tone_ratio > 0.95, flatness < 0.005)로 걸러내서 `outdoor`로 재라벨합니다. 실질적으로 4-class로 동작하는 셈입니다.
4. 2-of-3 의결 — 위 결과를 시간 순으로 큐에 쌓아두고, 최근 3개 중 2개 이상이 fire_alarm이면 그때 실제 화재 이벤트로 확정합니다. 단발성 잡음 한 번에 알림이 나가는 걸 막기 위함입니다.
5. 알림 발송 — Flask 서버에 이벤트가 들어오면 Solapi SMS, Gmail SMTP 두 채널로 동시에 알립니다.

## 알림 정책

현장에 배포하면서 실무자 분들과 협의해서 정한 규칙입니다.

- 화재 확정 직후 10초 간격으로 2번 발송하고, 그 다음엔 5분 쿨타임을 둡니다. 같은 화재 한 건에 문자가 수십 통 가는 사고를 막기 위함입니다.
- Emergency 로 분류되더라도 알림은 안 나갑니다. 학습 단계에서는 화재 경보음과 변별하라고 Emergency를 따로 학습시키지만, 운영 단계에서 매번 알림을 보내면 허위 출동 부담이 너무 큽니다.
- 녹음 파일은 디스크 부담 때문에 최근 1~2시간 분량만 남기고, 화재로 확정된 이벤트의 음원만 영구 보관합니다.

## 파일 구조

```
.
├── app.py                  # Flask 서버 + 대시보드 + REST API (포트 5000)
├── main_detector.py        # 마이크 입력 → 추론 루프
├── pipeline_mul.py         # 룰 프리필터 + 모델 추론 통합
├── prefilter.py            # Stage 0 룰 (dB / band / periodicity)
├── notifications.py        # Gmail + Solapi SMS 발송
├── telegram_notify.py      # Telegram 봇 (다중 수신자, /start /stop 등 자동 등록)
├── common_raw_audio.py     # 오디오 로드/리샘플 유틸
├── database.py             # SQLite 스키마 초기화
├── index.html              # 대시보드 SPA
├── esresnext/              # ESResNeXt-fbsp 모델 정의
├── test_email.py           # Gmail 발송 단독 테스트
├── test_alert.py           # /api/events 직접 트리거
├── benchmark_prefilter.py  # 프리필터 ON/OFF 비교 벤치
└── benchmark_postprocessor.py  # 2-of-3 ON/OFF 비교 벤치
```

## 모델 파일

용량 때문에 Git에 넣지 않았습니다. Releases 페이지에서 받아서 프로젝트 루트에 두면 됩니다.

| 파일 | 용량 | 용도 |
| --- | --- | --- |
| `best_3class_esresnext_tuned.pt` | 120MB | 메인 분류기 |
| `best_3class_rf.pkl` | 8MB | Random Forest 백업 |
| `yamnet_transfer_classifier.keras` | 13MB | YAMNet Transfer |
| `yamnet_mlp_best.pt` | 2.6MB | YAMNet MLP |

YAMNet hub 모델까지 쓰려면 다음 한 줄을 추가로 받아두시길 바랍니다.

```bash
mkdir -p yamnet_local
wget -O yamnet_local/yamnet.tar.gz https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed
tar -xvf yamnet_local/yamnet.tar.gz -C yamnet_local
```

## 실행 방법

`requirements.txt`는 Flask 쪽 최소만 들어있어서, 추론까지 돌리려면 `torch`, `numpy`, `scipy`, `librosa`, `soundfile`, `pyaudio`, `scikit-learn` 정도는 별도 설치가 필요합니다.

```bash
pip install -r requirements.txt
python database.py       # gard-ear.db 생성 (있으면 덮어씀)
python app.py            # Flask 서버 (포트 5000)
python main_detector.py  # 다른 터미널에서 감지기 실행
```

브라우저로 `http://127.0.0.1:5000` 또는 같은 네트워크의 다른 PC에서 `http://<라즈베리파이_IP>:5000` 으로 들어가서 관리자 로그인(초기 비밀번호 `1234`)한 다음, 발신자 정보 탭에서 Gmail 앱 비밀번호 / Solapi API Key·Secret·발신번호를 입력하면 됩니다. 저장하면 바로 반영됩니다.

발송이 잘 되는지 보고 싶으면 `python test_email.py`, `python test_alert.py` 로 단독 테스트 가능합니다.

## 하드웨어

현장 배포용 BOM은 대략 이 정도입니다.

- 라즈베리파이 4 Model B (4GB)
- APC BE400-KR UPS 멀티탭 정전 시 안전 종료용. 처음엔 DFROBOT UPS HAT + 리튬 1250mAh를 썼는데 운영 시간이 40분 정도밖에 안 나오고 신뢰성도 애매해서 APC 멀티탭으로 바꿨습니다.
- Coms USB 핀 마이크 (ICF1902)
- SanDisk MicroSD 32GB
- 5V 3A C타입 어댑터

한 대 기준 약 30만원 정도이고, 좀 더 큰 규모 보급을 가정해서 ESP32 + 외부 서버 구조도 PoC를 따로 검증해놨습니다. 다만 ESP32 안은 현장 운영은 아직입니다.

## 대시보드

- 로컬: `http://127.0.0.1:5000`
- 같은 네트워크: `http://<서버_IP>:5000`
- 초기 관리자 비밀번호: `1234` (반드시 바꾸세요)

이벤트 로그, 발송 내역, 디바이스 상태, 발신자/수신자 설정, 최근 녹음 재생까지 다 한 페이지에서 됩니다.

## 보안 관련 주의

내부망 프로토타입이라 보안은 좀 느슨하게 짜져 있습니다. 공개 인터넷에 그대로 노출하면 안 됩니다.

- 초기 비밀번호 `1234`는 첫 접속 후 바로 바꾸기
- `admin_password`는 DB에 평문으로 저장됨 
- `/api/admin/settings` 엔드포인트에는 인증 미들웨어가 없음
- Gmail 앱 비밀번호, Solapi API 키, Telegram 토큰 전부 `SystemSettings` 테이블에 평문으로 들어감 → DB 파일을 잘 관리해야 합니다
- `gard-ear.db`, `records/`, `logs/` 는 `.gitignore` 로 이미 빠져있으니 절대 커밋하지 말 것

## API

| Method | Path | 설명 |
| --- | --- | --- |
| GET | `/` | 대시보드 |
| GET | `/api/status` | 장치 상태 |
| POST | `/api/events` | 화재 이벤트 (감지기가 호출) |
| GET | `/api/events` | 이벤트 내역 |
| POST | `/api/acknowledge` | 경보 확인/해제 |
| GET | `/api/logs` | 알림 발송 내역 |
| GET/POST | `/api/system-settings` | 설치 위치 |
| GET/POST | `/api/notification-contacts` | 수신자 목록/추가 |
| PATCH/DELETE | `/api/notification-contacts/<id>` | 수신자 수정/삭제 |
| POST | `/api/admin/login` | 관리자 로그인 |
| GET/POST | `/api/admin/settings` | 관리자 설정 |

## 팀 / 라이선스

- 동국대학교 컴퓨터AI학부 종합설계 2 
- 왜인진모르겠지만잘하는팀 — 황재형, 김범주, 안소희, 임다희
- 지도교수: 석문기 / 산업체 멘토: 전북개발공사 김정중 차장
- 내부 연구·프로토타입 용도로만 사용합니다.
