# 🔥 가드이어 (Gard-Ear) — AI 화재 경보 감지 시스템

마이크로 화재 경보음을 실시간 감지하여 **이메일/SMS 알림**을 전송하는 로컬 서버.
라즈베리파이 + Flask + PyTorch(ESResNeXtFBSP) 기반.

## ✨ 주요 기능
- 🎤 **연속 오디오 감지** — 4초 녹음, 2초 슬라이딩 윈도우
- 🧠 **2-Stage 파이프라인** — 룰 프리필터 → 딥러닝 분류 (ESResNeXt / YAMNet / RandomForest 선택)
- 🗳️ **2-of-3 확정 로직** — 최근 3회 중 2회 FIRE 판정 시 확정 경보 (오탐 억제)
- 📧 **자동 알림** — Gmail SMTP + NHN Cloud SMS
- 🖥️ **웹 대시보드** — 상태 모니터링, 이벤트/발송 내역, 연락처 관리

## 📂 프로젝트 구조
```
webserverfile/
├── app.py                  # Flask 서버 (포트 5000, API + 대시보드)
├── main_detector.py        # 마이크 실시간 감지 루프
├── pipeline_mul.py         # 2-Stage 추론 파이프라인
├── prefilter.py            # Stage 0: 룰 기반 프리필터
├── common_raw_audio.py     # 오디오 로드/전처리 유틸
├── database.py             # SQLite 스키마 생성
├── notifications.py        # Gmail / NHN SMS 발송
├── index.html              # 관리자 대시보드 UI
├── test_email.py           # Gmail 발송 테스트
├── test_alert.py           # /api/events 트리거 테스트
├── check_error.py          # 라이브러리 설치 점검 스크립트
├── esresnext/              # ESResNeXtFBSP 모델 정의 (파이썬 모듈)
├── .env.example            # 환경 변수 템플릿
├── .gitignore
├── requirements.txt
└── README.md
```

## 📥 모델 파일 다운로드 및 배치

⚠️ **모델 파일은 용량 문제로 저장소에 포함되어 있지 않습니다.**
GitHub Releases에서 받아 아래 구조로 배치해야 `main_detector.py` / `pipeline_mul.py`가 정상 동작합니다.

### 1) Release에서 받을 4개 파일

👉 [**Releases 페이지에서 다운로드**](../../releases/latest) (태그 `v1.0`)

| 파일 | 용량 | 용도 |
|---|---|---|
| `best_3class_esresnext_tuned.pt` | 120 MB | 메인 분류기 (기본값, `ACTIVE_MODEL="esresnext"`) |
| `best_3class_rf.pkl` | 8 MB | RandomForest (`ACTIVE_MODEL="rf"`) |
| `yamnet_transfer_classifier.keras` | 13 MB | YAMNet Transfer (`ACTIVE_MODEL="yamnet"`) |
| `yamnet_mlp_best.pt` | 2.6 MB | YAMNet MLP (옵션) |

### 2) 배치 위치 — **프로젝트 루트에 그대로 복사**

`app.py`, `README.md`가 있는 폴더(= `webserverfile/`)에 **모두 같은 레벨로** 놓으세요.
하위 폴더를 만들지 말고 그대로 떨어뜨리면 됩니다.

### 3) 최종 디렉토리 구조 (모델 배치 후)

```
webserverfile/                              ← 여기가 "프로젝트 루트"
├── app.py
├── main_detector.py
├── pipeline_mul.py
├── ... (기타 .py, index.html 등)
├── best_3class_esresnext_tuned.pt          ← ✅ 루트 직속
├── best_3class_rf.pkl                      ← ✅ 루트 직속
├── yamnet_transfer_classifier.keras        ← ✅ 루트 직속
├── yamnet_mlp_best.pt                      ← ✅ 루트 직속
├── yamnet_local/                           ← ✅ YAMNet 사용 시 (아래 4번 참고)
│   ├── assets/
│   ├── variables/
│   └── saved_model.pb
├── records/                                ← 실행 시 자동 생성 (녹음 저장)
└── logs/                                   ← 실행 시 자동 생성 (CSV 로그)
```

### 4) YAMNet Hub 모델 (선택 — `yamnet` 모델을 쓸 때만)

```bash
# 프로젝트 루트에서 실행
mkdir -p yamnet_local
wget -O yamnet_local/yamnet.tar.gz https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed
tar -xvf yamnet_local/yamnet.tar.gz -C yamnet_local
```

### 5) 배치 확인

루트에서 다음 명령어로 4개 파일이 보이면 OK:

```bash
# Linux / macOS
ls -lh *.pt *.pkl *.keras

# Windows (PowerShell)
Get-ChildItem *.pt, *.pkl, *.keras
```

> 📌 `pipeline_mul.py`의 `ESRESNEXT_MODEL_PATH = "./best_3class_esresnext_tuned.pt"`처럼
> 모든 모델 경로가 **상대경로(`./`) + 루트 기준**으로 하드코딩되어 있습니다. 반드시 루트에 두세요.

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```
> `requirements.txt`에는 Flask 관련 최소 라이브러리만 포함되어 있습니다.
> 추론을 돌리려면 `torch`, `numpy`, `scipy`, `librosa`, `soundfile`, `pyaudio`, `scikit-learn` 등이 추가로 필요합니다.

### 2. 환경 변수 설정
`.env.example`을 참고하여 값을 채운 뒤 터미널에 등록합니다.

**Linux / macOS:**
```bash
export GMAIL_USER="your-email@gmail.com"
export GMAIL_APP_PASSWORD="your-16-digit-app-password"
export NHN_APP_KEY="..."
export NHN_SECRET_KEY="..."
export NHN_SENDER_NUMBER="01012345678"
```

**Windows (CMD):**
```cmd
set GMAIL_USER=your-email@gmail.com
set GMAIL_APP_PASSWORD=your-16-digit-app-password
```

**Windows (PowerShell):**
```powershell
$env:GMAIL_USER = "your-email@gmail.com"
$env:GMAIL_APP_PASSWORD = "your-16-digit-app-password"
```

> ⚠️ 환경 변수는 **현재 터미널 세션에만 적용**됩니다.
> 반드시 서버를 실행할 터미널에서 동일하게 등록하세요.

### 3. 데이터베이스 생성
```bash
python database.py
```
> 기존 `gard-ear.db`가 있으면 삭제되고 새로 생성됩니다.

### 4. (선택) 이메일 발송 테스트
```bash
python test_email.py
```

### 5. Flask 서버 실행
```bash
python app.py
```
→ `http://127.0.0.1:5000` 접속

### 6. 감지기 실행 (별도 터미널)
```bash
python main_detector.py
```

## 🌐 대시보드 접속
- 로컬: http://127.0.0.1:5000
- 같은 네트워크: http://[서버_IP]:5000
- 초기 관리자 비밀번호: **`1234`** (설정 탭에서 변경 가능)

## 🔐 보안 주의사항
이 프로젝트는 **로컬/내부망 프로토타입**으로 설계되었습니다. 공개 인터넷에 노출 시:

- ⚠️ **초기 비밀번호 `1234`를 반드시 변경**하세요
- ⚠️ `admin_password`가 DB에 **평문 저장**됩니다 (실서비스라면 해시 필요)
- ⚠️ `/api/admin/settings` 엔드포인트에 **인증 미들웨어가 없습니다** (프로토타입 한계)
- ⚠️ Gmail 앱 비밀번호, NHN API 키를 DB UI로 입력하면 평문 보관됩니다 → 가능하면 **환경 변수**로만 관리하세요
- ⚠️ `gard-ear.db`, `records/`, `logs/`는 `.gitignore`로 제외되어 있습니다. 절대 커밋하지 마세요
- ⚠️ `.env` 파일 역시 제외되어 있으니 `.env.example`만 공유하세요

## 🧪 감지 알고리즘

### 2-Stage 파이프라인
```
오디오(4초) → Stage 0: 룰 프리필터
              ↓ (dB + 600-2500Hz 대역비 + 주기성, score ≥ 0.07)
              Stage 1: 딥러닝 분류 (3-class)
              ↓
              other / emergency / fire_alarm
```

### 2-of-3 확정 로직
2초 간격 슬라이딩 추론 → 최근 3회 중 **2회 이상 FIRE**면 확정 경보 → `/api/events` POST → 이메일+SMS 발송

## 📡 API 엔드포인트
| Method | Path | 설명 |
|---|---|---|
| GET | `/` | 대시보드 |
| GET | `/api/status` | 장치 상태 조회 |
| POST | `/api/events` | 화재 이벤트 발생 (감지기가 호출) |
| GET | `/api/events` | 이벤트 내역 |
| POST | `/api/acknowledge` | 경보 확인/해제 |
| GET | `/api/logs` | 알림 발송 내역 |
| GET/POST | `/api/system-settings` | 설치 위치 조회/변경 |
| GET/POST | `/api/notification-contacts` | 연락처 목록/추가 |
| PATCH/DELETE | `/api/notification-contacts/<id>` | 연락처 수정/삭제 |
| POST | `/api/admin/login` | 관리자 로그인 |
| GET/POST | `/api/admin/settings` | 관리자 설정 조회/변경 |

## 📄 라이선스
내부 연구/프로토타입용.
