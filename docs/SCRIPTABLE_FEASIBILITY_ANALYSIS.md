# Scriptable iOS 구현 가능성 분석

**분석일**: 2026-03-03  
**대상**: Daily Regime-Switching Signal Generator를 iOS Scriptable으로 구현할 수 있는지 검토

---

## 📊 현재 아키텍처

```
┌──────────────────────────────────────────┐
│ Python: daily_signal_generator.py        │
│ - yfinance로 NDX 데이터 다운로드         │
│ - MA 계산, 변동성 계산                   │
│ - 신호 생성 (Regime-Switching)          │
│ - JSON / CSV / HTML 저장                │
└──────────┬───────────────────────────────┘
           │ (매일 4PM, Task Scheduler)
           ↓
┌──────────────────────────────────────────┐
│ 결과 파일 (JSON / CSV)                   │
└──────────┬───────────────────────────────┘
           │ (HTTP 또는 로컬 파일)
           ↓
┌──────────────────────────────────────────┐
│ Scriptable Widget (scriptable_widget.js) │
│ - JSON 호출                              │
│ - UI 렌더링                              │
│ - iOS 홈화면 위젯                        │
└──────────────────────────────────────────┘
```

---

## ✅ Scriptable에서 가능한 것

### 1. JSON 데이터 호출
```javascript
// ✅ 가능
const req = new Request("https://SERVER/signal.json");
const data = await req.loadJSON();
```

### 2. 로컬 파일 접근
```javascript
// ✅ 가능 (Mac/iCloud Drive)
const fm = FileManager.iCloud();
const data = fm.readString(filePath);
```

### 3. UI 렌더링
```javascript
// ✅ 가능 (이미 구현됨)
const w = new ListWidget();
w.backgroundColor = new Color("#1abc9c");
w.addText("🟢 BUY");
```

### 4. HTTP 요청
```javascript
// ✅ 가능
const response = await new Request(url).load();
```

---

## ❌ Scriptable에서 어려운 것

### 1. Yahoo Finance 데이터 다운로드

**문제**: 
- Scriptable의 HTTP 요청은 JSON/HTML만 처리
- yfinance는 공식 API 없음 (웹 스크래핑 필요)
- 정규식으로 HTML 파싱은 불안정

**해결책**:
```javascript
// ❌ 안정적이지 않음
const html = await new Request("https://finance.yahoo.com/quote/^NDX").loadString();
const priceMatch = html.match(/regularMarketPrice":{.*?"raw":([0-9.]+)/);
```

### 2. 복잡한 신호 계산

**필요한 계산**:
- SMA (Simple Moving Average) × 4개
- 변동성 계산 (표본 표준편차 × 252)
- 변동성 백분위 (rolling percentile)
- Regime 판정 (HIGH/LOW)

**Scriptable 코드 예시** (부분):
```javascript
// SMA 계산
function calcSMA(prices, window) {
  if (prices.length < window) return NaN;
  let sum = 0;
  for (let i = prices.length - window; i < prices.length; i++) {
    sum += prices[i];
  }
  return sum / window;
}

// 변동성 계산
function calcVolatility(prices, lookback) {
  let returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push(Math.log(prices[i] / prices[i-1]));
  }
  
  let mean = 0;
  let recentReturns = returns.slice(-lookback);
  for (let r of recentReturns) mean += r;
  mean /= recentReturns.length;
  
  let variance = 0;
  for (let r of recentReturns) {
    variance += Math.pow(r - mean, 2);
  }
  variance /= recentReturns.length;
  
  return Math.sqrt(variance) * Math.sqrt(252);
}
```

**평가**: 
- 이론적으로는 가능 (JavaScript 수학)
- 실무적으로는 300일 × 4개 MA = 부하 높음
- 버그 가능성 높음 (Python과 다른 부동소수점 처리)

### 3. leverage_rotation.py 의존성

**Python 코드**:
```python
from leverage_rotation import signal_regime_switching_dual_ma
```

**Scriptable 대체**: 전체 로직을 JavaScript로 재구현 필요
- 약 200줄 이상의 코드
- 테스트 & 검증 필요
- 유지보수 어려움

---

## 📋 구현 옵션 평가

### 옵션 1: Python + Scriptable (현재) ⭐⭐⭐ 권장

**구조**:
```
Python (daily_signal_generator.py) → JSON 저장
                ↓
Scriptable (JSON 호출) → 위젯 표시
```

**장점**:
- ✅ 신뢰도 높음 (Python은 검증된 코드)
- ✅ 유지보수 용이 (로직 중앙화)
- ✅ 복잡한 계산 가능 (NumPy, Pandas)
- ✅ 이미 구현됨 (scriptable_widget.js 완성)
- ✅ 성능 우수 (Python 최적화)

**단점**:
- ❌ Windows Task Scheduler 설정 필요
- ❌ 서버/클라우드 배포 필요 (또는 로컬 파일 공유)

**예상 시간**: 30분 (Task Scheduler 설정)

---

### 옵션 2: 순수 Scriptable ❌ 비권장

**구조**:
```
Scriptable
  → Yahoo Finance 데이터 (웹 스크래핑)
  → 신호 계산 (JavaScript)
  → 위젯 표시
```

**장점**:
- ✅ 서버 불필요
- ✅ 완전 자동화 (iOS만)
- ✅ 오프라인 가능

**단점**:
- ❌ 복잡도 매우 높음 (300줄+ JavaScript)
- ❌ 신뢰도 낮음 (웹 스크래핑)
- ❌ 성능 문제 (JavaScript 느림)
- ❌ 유지보수 어려움
- ❌ 테스트 불가능

**예상 시간**: 3-5시간 (개발 + 테스트)

**추가 이슈**:
```javascript
// Yahoo Finance HTML 파싱 예시 (불안정)
const html = await new Request(
  "https://finance.yahoo.com/quote/^NDX"
).loadString();

// 정규식으로 가격 추출 (언제든 깨질 수 있음)
const match = html.match(
  /regularMarketPrice[^}]*?"raw":([0-9.]+)/
);
```

---

### 옵션 3: 하이브리드 (부분 Scriptable)

**구조**:
```
Scriptable (매일 4PM)
  → Python 스크립트 트리거 (SSH 또는 HTTP)
  → Python 신호 계산
  → JSON 저장
  → Scriptable 위젯 표시
```

**평가**: 복잡도 높음, 이득 없음 (옵션 1 사용)

---

## 🎯 최종 권장안

### 구현 순서

#### Phase 1: Python 자동화 (기본)
```bash
# 1. calibrate_tqqq.py 실행
python core/calibrate_tqqq.py

# 2. daily_signal_generator.py 실행
python signals/daily_signal_generator.py

# 3. 결과 확인
# output/daily_signals.json
# output/daily_signals.html
```

**시간**: 5분

#### Phase 2: 자동 실행 설정

**Windows Task Scheduler**:
```
트리거: 매일 오후 4시 (NYSE 종장)
프로그램: C:\Python312\python.exe
인수: C:\...\signals\daily_signal_generator.py
```

**시간**: 15분

#### Phase 3: Scriptable 위젯 연결

**방법 A: 로컬 파일** (권장)
```javascript
// iCloud Drive에 JSON 저장
const fm = FileManager.iCloud();
const data = fm.readString("/LRS/daily_signal.json");
```

**방법 B: HTTP 호출**
```javascript
// 웹서버 또는 GitHub에 JSON 업로드
const data = await new Request(
  "https://YOUR_SERVER/daily_signal.json"
).loadJSON();
```

**시간**: 10분 (scriptable_widget.js 수정)

---

## 📊 비교 표

| 항목 | Python + Scriptable | 순수 Scriptable |
|------|:---:|:---:|
| 신뢰도 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 복잡도 | ⭐⭐ (낮음) | ⭐⭐⭐⭐⭐ (매우 높음) |
| 성능 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 유지보수 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 구현 시간 | 30분 | 3-5시간 |
| 서버 필요 | 선택 | 불필요 |
| 웹 스크래핑 | 불필요 | 필수 |
| 테스트 가능 | ✅ | ❌ |
| 자동 업데이트 | ✅ | ⚠️ |

---

## 💡 최종 결론

### ✅ Python + Scriptable 구조 추천

**이유**:
1. **현재 구현 완료**: scriptable_widget.js가 이미 존재
2. **신뢰도 높음**: Python은 검증된 금융 계산 엔진
3. **확장성 우수**: 신호 알고리즘 개선 용이
4. **유지보수 용이**: 로직 중앙화
5. **성능 우수**: NumPy 최적화

### 구현 로드맵

```
Week 1:
  - [x] Python 스크립트 완성 (daily_signal_generator.py)
  - [ ] Windows Task Scheduler 설정 (15분)
  
Week 2:
  - [ ] JSON 서버 배포 (선택, GitHub Pages / AWS S3)
  - [ ] Scriptable 위젯 최종 테스트
  - [ ] iOS 홈화면 위젯 배포
  
Week 3:
  - [ ] 모니터링 & 버그 수정
  - [ ] Telegram 알림 통합 (signals/daily_signal_telegram.py)
```

---

## 📌 다음 단계

### 즉시 실행 (30분)

```bash
# 1. Python 신호 생성
python signals/daily_signal_generator.py
# → output/daily_signals.json 생성

# 2. JSON 파일을 웹서버/iCloud에 동기화
# → macOS: iCloud Drive 폴더에 복사
# → 또는 GitHub Pages에 배포

# 3. Scriptable 위젯 설정
# → scriptable_widget.js의 DATA_URL 수정
# → iOS 홈화면에 위젯 추가
```

### 자동화 (15분)

```bash
# Windows Task Scheduler 또는 macOS Automator 설정
# → 매일 4PM (NYSE 종장) 자동 실행
```

---

## 🔗 관련 파일

- `signals/daily_signal_generator.py` - Python 신호 생성
- `signals/daily_signal_telegram.py` - Telegram 알림
- `scriptable_widget.js` - iOS 위젯
- `docs/DAILY_SIGNAL_SETUP.md` - 설정 가이드
- `docs/TELEGRAM_SCRIPTABLE_SETUP.md` - iOS 알림 설정

---

**Status**: ✅ Python + Scriptable 구조 권장  
**Recommendation**: 현재 구조 유지, Task Scheduler 자동화만 추가

