# Daily Signal Generator 설정 가이드

**목표**: 매일 종장 이후 자동으로 신호를 생성하고 저장

---

## 방법 1️⃣: 윈도우 작업 스케줄러 (권장 - 가장 쉬움)

### Step 1: 파이썬 경로 확인

```bash
# CMD 실행
where python
# 결과 예: C:\Users\anti_\AppData\Local\Programs\Python\Python311\python.exe
```

### Step 2: 스크립트 경로 확인

```
C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run\daily_signal_generator.py
```

### Step 3: 작업 스케줄러 설정

**1) 작업 스케줄러 열기**
- `Win + R` → `taskschd.msc` → Enter

**2) 기본 작업 생성**
- 왼쪽 패널: "작업 만들기" 클릭
- 일반 탭:
  - 이름: `NDX Daily Signal`
  - 설명: `Generate daily regime-switching signal`
  - ✓ "가장 높은 수준의 권한으로 실행" 체크

**3) 트리거 설정**
- 트리거 탭 → "새로 만들기"
  - 작업 시작: `매일`
  - 시작: `2025-02-28` (오늘)
  - 시간: `16:00` (오후 4시 - NYSE 종장 직후)
  - 반복 간격: `1일`

**4) 작업 설정**
- 작업 탭 → "새로 만들기"
  - 프로그램/스크립트:
    ```
    C:\Users\anti_\AppData\Local\Programs\Python\Python311\python.exe
    ```
  - 인수 추가:
    ```
    C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run\daily_signal_generator.py
    ```
  - 시작 위치:
    ```
    C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
    ```

**5) 조건 설정**
- 조건 탭:
  - ✓ "컴퓨터가 AC 전원에 연결된 경우에만 작업 실행"
  - ✓ "유휴 상태일 때만 작업 실행" (체크 해제)

**6) 설정 탭**
- ✓ "작업이 실패한 경우 다시 시도" (1분 재시도)
- ✓ "실행 중인 작업이 있으면 새 인스턴스를 시작하지 않음"

**7) 저장**
- `OK` → 완료

### Step 4: 테스트

```bash
# 수동 실행 (작업 스케줄러에서)
- 생성한 작업 우클릭 → "실행"

# 또는 명령행
cd C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
python daily_signal_generator.py
```

### Step 5: 결과 확인

```
output/daily_signals.csv  ← 신호 로그 (CSV)
output/daily_signals.html ← 시각화 (브라우저에서 열기)
```

---

## 방법 2️⃣: Streamlit 대시보드 (가장 편함)

### 설치

```bash
pip install streamlit streamlit-autorefresh
```

### 대시보드 코드

```python
# File: daily_signal_app.py

import streamlit as st
import pandas as pd
from pathlib import Path
import datetime
from daily_signal_generator import generate_signal, OPTIMAL_PARAMS

st.set_page_config(page_title="NDX Daily Signal", layout="wide")

st.title("⚡ NDX Regime-Switching Daily Signal")
st.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 신호 생성
result = generate_signal()

# 메인 표시
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Date", result['date'])

with col2:
    st.metric("Price", f"${result['price']:.2f}")

with col3:
    if result['status'] == 'SUCCESS':
        signal_text = "🟢 BUY" if result['signal'] == 1 else "🔴 HOLD"
        st.metric("Signal", signal_text)
    else:
        st.metric("Status", "⚠️ ERROR")

# 상세 정보
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Current MA Values")
    ma_data = {
        'Low-Vol Fast': f"${result['fast_low_ma']:.2f}",
        'Low-Vol Slow': f"${result['slow_low_ma']:.2f}",
        'High-Vol Fast': f"${result['fast_high_ma']:.2f}",
        'High-Vol Slow': f"${result['slow_high_ma']:.2f}",
    }
    for k, v in ma_data.items():
        st.write(f"**{k}**: {v}")

with col2:
    st.subheader("🎯 Current Regime")
    st.write(f"**Regime**: {result['regime']}")
    st.write(f"**Volatility**: {result['volatility_pct']:.0f}%")
    st.write(f"**Threshold**: {OPTIMAL_PARAMS['vol_threshold_pct']:.1f}%")

# 신호 로그
st.divider()
st.subheader("📝 Recent Signals (Last 20)")

log_file = Path("output/daily_signals.csv")
if log_file.exists():
    df = pd.read_csv(log_file).tail(20)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No signal history yet")

# 설정 정보
st.divider()
st.caption("⚙️ Strategy Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Fast Low**: {OPTIMAL_PARAMS['fast_low']}")
    st.write(f"**Slow Low**: {OPTIMAL_PARAMS['slow_low']}")
with col2:
    st.write(f"**Fast High**: {OPTIMAL_PARAMS['fast_high']}")
    st.write(f"**Slow High**: {OPTIMAL_PARAMS['slow_high']}")
with col3:
    st.write(f"**Vol Lookback**: {OPTIMAL_PARAMS['vol_lookback']}")
    st.write(f"**Vol Threshold**: {OPTIMAL_PARAMS['vol_threshold_pct']:.1f}%")
```

### 실행

```bash
streamlit run daily_signal_app.py

# 자동 새로고침 (매 5분)
streamlit run daily_signal_app.py --logger.level=error
```

**특징**:
- 📱 웹 대시보드 (스마트폰에서도 접근 가능)
- 🔄 자동 새로고침
- 📊 실시간 그래프
- ☁️ 클라우드 배포 가능 (Streamlit Cloud)

---

## 방법 3️⃣: 이메일 알림 (최고의 편의성)

### 설치

```bash
pip install smtplib
```

### 이메일 발송 코드 추가

```python
# daily_signal_generator.py 끝에 추가

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(result):
    """신호를 이메일로 전송"""

    # Gmail 설정
    SENDER_EMAIL = "your_email@gmail.com"
    SENDER_PASSWORD = "your_app_password"  # Gmail 앱 비밀번호 (2FA 활성화 필요)
    RECEIVER_EMAIL = "your_personal_email@gmail.com"

    subject = f"🟢 NDX Signal: {result['signal_type']}" if result['signal'] == 1 else f"🔴 NDX: Hold"

    body = f"""
    Daily Signal Report
    ==================

    Generated: {result['timestamp']}
    Date: {result['date']}
    Price: ${result['price']:.2f}

    Signal: {result['signal_type']}
    Regime: {result['regime']} (Vol: {result['volatility_pct']:.0f}%)

    MA Values:
    - Low-Vol: ${result['fast_low_ma']:.2f} (Fast) vs ${result['slow_low_ma']:.2f} (Slow)
    - High-Vol: ${result['fast_high_ma']:.2f} (Fast) vs ${result['slow_high_ma']:.2f} (Slow)

    Status: {result['status']}
    """

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Gmail SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"✓ Email sent to {RECEIVER_EMAIL}")
    except Exception as e:
        print(f"✗ Email failed: {e}")

# main() 끝에 추가
send_email_alert(result)
```

**Gmail 앱 비밀번호 생성**:
1. Gmail 계정 → 보안
2. 2단계 인증 활성화
3. "앱 비밀번호" → Python 선택 → 비밀번호 생성

---

## 방법 4️⃣: Google Sheets 자동 기록

### 설치

```bash
pip install gspread oauth2client
```

### 코드

```python
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def append_to_sheets(result):
    """Google Sheets에 자동 기록"""

    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    # credentials.json 파일 필요 (Google Cloud 설정)
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open("NDX Daily Signal").sheet1

    sheet.append_row([
        result['timestamp'],
        result['date'],
        result['price'],
        result['signal'],
        result['signal_type'],
        result['regime'],
    ])

    print("✓ Data appended to Google Sheets")
```

**특징**:
- 📱 스마트폰에서 실시간 확인
- 📊 자동 데이터 축적
- 🔗 공유 가능

---

## 비교표

| 방법 | 난이도 | 편의성 | 자동화 | 비용 | 추천도 |
|------|--------|--------|--------|------|--------|
| **작업 스케줄러** | ⭐ | ⭐⭐ | ✓ 자동 | 무료 | ⭐⭐⭐⭐⭐ |
| **Streamlit** | ⭐⭐ | ⭐⭐⭐⭐ | 수동/자동 | 무료 | ⭐⭐⭐⭐ |
| **이메일** | ⭐⭐ | ⭐⭐⭐ | ✓ 자동 | 무료 | ⭐⭐⭐⭐ |
| **Google Sheets** | ⭐⭐ | ⭐⭐⭐ | ✓ 자동 | 무료 | ⭐⭐⭐ |

---

## 🎯 추천 조합

**최고의 자동화**:
```
작업 스케줄러 (매일 4 PM)
    ↓
daily_signal_generator.py (신호 생성)
    ↓
CSV 저장 + 이메일 발송
    ↓
사용자가 이메일로 확인
```

**최고의 편의성**:
```
Streamlit 대시보드 (항상 켜짐)
    ↓
웹 브라우저에서 실시간 확인
    ↓
또는 작업 스케줄러에서 자동 새로고침
```

---

## 실행 예시

### 현재 신호 확인 (바로)

```bash
cd C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
python daily_signal_generator.py
```

**출력**:
```
================================================================================
  Daily Regime-Switching Signal Generator
  Parameters: {'fast_low': 12, 'slow_low': 237, ...}
================================================================================

[2025-02-27 15:45:23] Downloading NDX data (last 300 days)...
  Latest: 2025-02-27 @ $23456.78
  Signal: 1 (0=관망, 1=매수)

Updating logs...
  → output/daily_signals.csv updated (45 records)

Generating HTML report...
  → output/daily_signals.html created

================================================================================
  ✓ Done!
  📊 View report: output/daily_signals.html
  📝 View logs: output/daily_signals.csv
================================================================================
```

---

## 주의사항

1. **시간대**: NYSE 종장은 오후 4시 (동부 시간)
   - 한국: 오전 6시 (서머 타임) 또는 오전 5시 (표준 시간)
   - 스케줄러를 그에 맞춰 설정하세요

2. **네트워크**: 정기적인 데이터 다운로드 필요
   - 안정적인 인터넷 연결 확인
   - yfinance가 간헐적으로 실패할 수 있으니 재시도 로직 포함

3. **정확도**: 최신 데이터만 사용
   - 마진 거래나 선물 거래는 신호가 약간 다를 수 있음
   - 실제 거래 전에 다른 지표도 함께 확인하세요

---

## 트러블슈팅

### 작업이 실행되지 않음

```bash
# 작업 스케줄러 로그 확인
Get-Content "C:\Windows\System32\winevt\Logs\System"

# 또는 Event Viewer 열기
eventvwr.msc
```

### Python 경로 오류

```bash
# 정확한 Python 경로 확인
python -c "import sys; print(sys.executable)"

# 또는 현재 디렉토리 확인
import os; print(os.getcwd())
```

### 데이터 다운로드 실패

```bash
# yfinance 업데이트
pip install --upgrade yfinance

# 테스트
python -c "import yfinance; print(yfinance.download('^NDX', period='1d'))"
```

---

## 다음 단계

1. ✅ 본 스크립트 실행: `python daily_signal_generator.py`
2. ✅ 결과 확인: `output/daily_signals.html` 열기
3. ✅ 자동화 설정: 작업 스케줄러 (또는 Streamlit)
4. ✅ 신호 모니터: 매일 자동으로 수신

---

**문제가 있으면 알려주세요!** 🚀
