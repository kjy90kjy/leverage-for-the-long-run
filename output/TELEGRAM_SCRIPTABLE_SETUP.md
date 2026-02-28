# 텔레그램 + Scriptable 위젯 설정 가이드

**목표**: 매일 종장 후 텔레그램 + iOS 위젯으로 신호 수신

---

## 🎯 시스템 구조

```
매일 4 PM (NYSE 종장)
    ↓
Windows 작업 스케줄러
    ↓
daily_signal_telegram.py
    ↓
┌─────────────────────┬─────────────────────┐
│  Telegram Bot       │  JSON 파일 / 웹     │
│  (메시지 발송)      │  (데이터 저장)      │
└─────────────────────┴─────────────────────┘
    ↓                       ↓
📱 핸드폰 (알림)      📱 iOS 위젯 (실시간)
```

---

## Step 1️⃣: 텔레그램 봇 설정 (5분)

### 1.1) 봇 생성

**핸드폰에서**:
1. 텔레그램 앱 열기
2. `@BotFather` 검색
3. `/start` 입력
4. `/newbot` 입력
5. 봇 이름: `NDX Daily Signal`
6. 봇 유저명: `ndx_signal_bot` (고유해야 함)

**응답**:
```
Done! Congratulations on your new bot. You will find it at t.me/ndx_signal_bot.
You can now add a description, about section and profile picture for your bot,
see /help for a list of commands.

Use this token to access the HTTP API:
1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk

Keep your token secure and store it safely, it can be used by anyone to control your bot.
```

**Token 저장** ← 이게 중요!

### 1.2) Chat ID 얻기

**Step A**: 방금 만든 봇 찾기 (t.me/ndx_signal_bot)
- `/start` 입력

**Step B**: 이 URL을 브라우저에서 열기
```
https://api.telegram.org/bot<TOKEN>/getUpdates
```
예시:
```
https://api.telegram.org/bot1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk/getUpdates
```

**응답** (JSON):
```json
{
  "ok": true,
  "result": [
    {
      "update_id": 123456789,
      "message": {
        "message_id": 1,
        "date": 1700000000,
        "chat": {
          "id": 987654321,  ← 이 값!
          "type": "private"
        },
        ...
      }
    }
  ]
}
```

**Chat ID 저장** ← 987654321

---

## Step 2️⃣: 파이썬 스크립트 설정

### 2.1) 텔레그램 정보 입력

파일 열기: `daily_signal_telegram.py`

이 부분을 수정:
```python
# ⚠️ 설정: 아래 값들을 본인 값으로 변경하세요
TELEGRAM_TOKEN = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"  # 변경!
TELEGRAM_CHAT_ID = "987654321"   # 변경!
```

### 2.2) 필요한 패키지 설치

```bash
pip install requests
```

### 2.3) 테스트 실행

```bash
cd C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
python daily_signal_telegram.py
```

**예상 출력**:
```
================================================================================
  Daily Signal with Telegram
================================================================================
[2026-02-27 15:35:01] Downloading NDX data...
✓ Signal generated: 2026-02-26 @ $25034.37
  Signal: 0 (관망)

📈 Prediction:
📈 내일 $25,050 이상이면 LOW-VOL 매수 신호 발생!
   필요 상승: $16 (+0.06%)

📱 Sending Telegram message...
✓ Telegram message sent successfully

================================================================================
  ✓ Done!
================================================================================
```

**핸드폰 확인**: 텔레그램 봇에서 메시지 도착!

---

## Step 3️⃣: 자동화 설정

### 3.1) 윈도우 작업 스케줄러

`Win + R` → `taskschd.msc` → Enter

**기본 작업 생성**:

| 항목 | 값 |
|------|-----|
| 이름 | NDX Daily Signal Telegram |
| 설명 | Send daily signal via Telegram |

**트리거**:
```
매일
시간: 16:00 (NYSE 종장 직후)
```

**작업**:
```
프로그램: C:\Users\anti_\AppData\Local\Programs\Python\Python311\python.exe
인수: daily_signal_telegram.py
시작위치: C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
```

---

## Step 4️⃣: Scriptable 위젯 (iOS)

### 4.1) Scriptable 앱 설치

App Store에서 **Scriptable** 검색 & 설치

### 4.2) 신호 데이터를 웹에서 접근 가능하게 만들기

**Option A: Google Sheets (추천 - 가장 쉬움)**

1. Google Sheets에서 새 시트 생성
2. Python 스크립트 수정:

```python
# daily_signal_telegram.py 끝에 추가

def save_to_google_sheets(result, prediction):
    """Google Sheets에 데이터 저장 (Apps Script 통해)"""
    import requests

    # Google Apps Script 웹앱 URL
    GAS_URL = "https://script.google.com/macros/d/YOUR_SCRIPT_ID/userweb"

    data = {
        'signal': result['signal'],
        'signal_type': result['signal_type'],
        'price': result['price'],
        'date': result['date'],
        'regime': result['regime'],
        'prediction_text': prediction['prediction_text'],
        'timestamp': result['timestamp'],
    }

    try:
        requests.post(GAS_URL, json=data)
    except:
        pass
```

**Google Apps Script 설정**:
1. [script.google.com](https://script.google.com) 열기
2. 새 프로젝트
3. 코드:

```javascript
function doPost(e) {
  const sheet = SpreadsheetApp.openById("YOUR_SHEET_ID").getSheetByName("Signal");
  const data = JSON.parse(e.postData.contents);

  sheet.appendRow([
    new Date(),
    data.signal,
    data.signal_type,
    data.price,
    data.date,
    data.regime,
    data.prediction_text
  ]);

  return ContentService.createTextOutput(JSON.stringify({status: 'ok'}));
}
```

4. 배포 → 새 배포 → 유형: 웹앱
5. "누구나" 접근 가능하게
6. Deploy ID 복사

**Google Sheets에서 JSON 내보내기**:

```javascript
// Apps Script 추가
function doGet() {
  const sheet = SpreadsheetApp.openById("YOUR_SHEET_ID").getSheetByName("Signal");
  const data = sheet.getRange(sheet.getLastRow(), 1, 1, 7).getValues()[0];

  const json = {
    signal: data[1],
    signal_type: data[2],
    price: data[3],
    date: data[4],
    regime: data[5],
    prediction_text: data[6],
    timestamp: data[0]
  };

  return ContentService
    .createTextOutput(JSON.stringify(json))
    .setMimeType(ContentService.MimeType.JSON);
}
```

배포 후 URL:
```
https://script.google.com/macros/d/YOUR_SCRIPT_ID/exec
```

---

**Option B: 자신의 웹서버** (고급)

Vercel, Heroku, AWS Lambda 등에 배포

간단한 Node.js 예시:
```javascript
// app.js
const express = require('express');
const fs = require('fs');
const app = express();

app.get('/ndx_signal.json', (req, res) => {
  const data = JSON.parse(fs.readFileSync('./signal.json', 'utf-8'));
  res.json(data);
});

app.post('/ndx_signal', express.json(), (req, res) => {
  fs.writeFileSync('./signal.json', JSON.stringify(req.body));
  res.json({status: 'ok'});
});

app.listen(3000);
```

---

### 4.3) Scriptable 스크립트 설정

**iOS에서**:
1. Scriptable 앱 열기
2. "+" 누르기
3. 코드 입력: `scriptable_widget.js` 내용 복사 & 붙여넣기
4. 이름: `NDX Daily Signal`
5. 저장

**스크립트 수정**:
```javascript
const DATA_URL = "https://script.google.com/macros/d/YOUR_SCRIPT_ID/exec";
// 또는
const DATA_URL = "https://your-server.com/ndx_signal.json";
```

### 4.4) 위젯 추가

**홈화면에 추가**:
1. 홈화면에서 길게 누르기
2. "+" 누르기
3. Scriptable 검색 & 선택
4. "스크립트 선택" 누르기
5. `NDX Daily Signal` 선택
6. 위젯 추가 (Small/Medium/Large)

**결과**:
```
┌──────────────────┐
│  🟢 BUY          │
│  Low-Vol Entry   │
│                  │
│  $25,034         │
│  2026-02-26      │
│                  │
│  ❄️ Low Vol      │
│                  │
│  📈 내일 $25,050 │
│    이상이면 신호 │
└──────────────────┘
```

---

## Step 5️⃣: 완전 자동화

### 5.1) Python 스크립트 최종 버전

```python
# daily_signal_telegram.py 수정

def main():
    # 1. 신호 생성
    result, ndx = generate_signal()

    if result['status'] != 'SUCCESS':
        return

    # 2. 예측 계산
    prediction = calculate_prediction(ndx, result)

    # 3. 텔레그램 발송
    message = format_telegram_message(result, prediction)
    send_telegram_message(message)

    # 4. Google Sheets/웹에 저장 (위젯용)
    save_to_web(result, prediction)
```

### 5.2) 매일 자동 실행

**작업 스케줄러 설정** (Step 3 참고):
```
매일 16:00 (NYSE 종장 후)
python daily_signal_telegram.py 실행
```

---

## 🎯 완성된 시스템

```
매일 4 PM (NYSE 종장)
    ↓
자동으로 daily_signal_telegram.py 실행
    ↓
┌─────────────────────────┬─────────────────────┐
│ 📱 텔레그램             │ 📱 iOS 위젯          │
│ (5초 후 메시지 도착)    │ (홈화면에서 확인)    │
│                         │                     │
│ "내일 $25,050          │ 🟢 BUY              │
│  이상이면 신호 발생"    │ Low-Vol Entry       │
│                         │ $25,034             │
│                         │ 내일 신호 조건...   │
└─────────────────────────┴─────────────────────┘
```

---

## 📝 체크리스트

- [ ] 1. 텔레그램 봇 생성
  - [ ] Token 저장
  - [ ] Chat ID 저장

- [ ] 2. Python 스크립트 설정
  - [ ] Token & Chat ID 입력
  - [ ] requests 패키지 설치
  - [ ] 테스트 실행

- [ ] 3. 자동화
  - [ ] 작업 스케줄러 설정
  - [ ] 매일 4 PM 실행 확인

- [ ] 4. Scriptable 위젯
  - [ ] Scriptable 앱 설치
  - [ ] 스크립트 입력
  - [ ] 웹 URL 연결
  - [ ] iOS 홈화면에 위젯 추가

- [ ] 5. 완성!
  - [ ] 텔레그램에서 매일 신호 수신
  - [ ] iOS 위젯에서 실시간 확인

---

## 🚨 트러블슈팅

### 텔레그램 메시지가 안 옴

```bash
# Token & Chat ID 다시 확인
https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>&text=test

# 응답이 ok: true인지 확인
```

### Scriptable 위젯이 데이터를 못 받음

```javascript
// Scriptable에서 콘솔 확인
console.log("Fetching from: " + DATA_URL);

// URL이 공개되어 있는지 확인
// 브라우저에서 직접 URL 열기 → JSON이 표시되나?
```

### 작업 스케줄러가 실행 안 됨

```bash
# 파이썬 경로 다시 확인
where python
# C:\Users\anti_\AppData\Local\Programs\Python\Python311\python.exe

# 권한 설정 확인 (높은 권한으로 실행)
```

---

## 🎉 축하합니다!

이제 매일:
- 📱 **텔레그램**: "내일 $25,050 이상이면 신호 발생!" 메시지
- 🏠 **iOS 위젯**: 홈화면에서 현재 신호 + 예측 실시간 확인

**운영 팁**:
1. 주말/공휴일은 자동으로 건너뜀 (NYSE 휴장)
2. 신호 변경되면 즉시 알림
3. 위젯은 자동으로 매 5분마다 새로고침
4. 스크린타임에 방해하지 않음 (조용한 알림)

Happy trading! 🚀
