# 비판적 코드 리뷰: Leverage Rotation Strategy
## 금융전문가, 금융공학, 코딩 관점의 단계별 검토

---

## 📋 목차
1. [전체 구조 및 목적](#1-전체-구조-및-목적)
2. [자료 수집 (Data Collection)](#2-자료-수집-data-collection)
3. [TQQQ 모방 데이터 생성](#3-tqqq-모방-데이터-생성)
4. [백테스트 엔진](#4-백테스트-엔진)
5. [메트릭 계산](#5-메트릭-계산)
6. [신호 생성기](#6-신호-생성기)
7. [최적화 프로세스](#7-최적화-프로세스)
8. [종합 결론](#8-종합-결론)

---

## 1. 전체 구조 및 목적

### ✅ 강점
- **명확한 목적**: Michael Gayed 2016 논문 복제 + NASDAQ 확장
- **모듈화**: 데이터→신호→백테스트→메트릭 계층이 명확함
- **독립적 검증**: 여러 지수(GSPC, IXIC, NDX)에서 검증

### ⚠️ 문제점

#### 1.1 **목적-방법론 불일치**
- **명시된 목적**: "Michael Gayed의 2016 논문 복제"
- **실제 코드**:
  - 논문에서 사용한 신호: 간단한 모멘텀 회전 (비공개)
  - 코드에서 사용: 복잡한 MA 기반 신호 (dual_ma, vol_adaptive, regime_switching)
  - **결과**: 이것은 '복제'가 아니라 '확장'이며, 과적합 위험이 높음

**근거**:
```python
# leverage_rotation.py line 188-195
def signal_dual_ma(price, slow=200, fast=50):
    """Golden cross — 이것은 Gayed의 원본 신호가 아님"""
    return (price.rolling(fast).mean() > price.rolling(slow).mean()).astype(int)
```

#### 1.2 **분석 파이프라인의 순서 문제**
- Part 1-3: 기초 분석
- Part 4-6: 단순 MA
- Part 7-9: 그리드 서치 (과적합 위험 **매우 높음**)
- Part 10-11: eulb 검증 (독립 소스 vs 자체 신호 비교 애매함)
- Part 12: TQQQ 캘리브레이션 기반 최종 분석

**문제**:
- Part 12가 Part 7-11의 그리드 서치 결과를 사용함
- 그런데 TQQQ 캘리브레이션(Part 12를 위한)이 나중에 나옴
- **순환 종속성 위험**: 결과가 입력을 결정하는 구조

---

## 2. 자료 수집 (Data Collection)

### 2.1 Shiller 배당금 추가 (`_add_shiller_dividends()`)

**코드** (line 56-92):
```python
def _add_shiller_dividends(price: pd.Series) -> pd.Series:
    shiller["div_yield_daily"] = (shiller["div"] / shiller["sp"]) / 252
    div_daily = shiller["div_yield_daily"].resample("D").ffill()
    total_ret = daily_ret + div_aligned  # line 82
    total_price = (1 + total_ret).cumprod() * price.iloc[0]
```

#### ✅ 강점
- Yale Shiller 데이터는 공식 학술 자료
- 월별→일일 보간법(ffill) 사용 적절

#### ⚠️ **심각한 문제**

**문제 1: 배당금 동기화 오류**
```python
# 실제 문제 1: 월별 데이터를 ffill로 일일 데이터로 확장
shiller["div_yield_daily"] = (shiller["div"] / shiller["sp"]) / 252
div_daily = shiller["div_yield_daily"].resample("D").ffill()  # line 77

# 이것은 잘못된 해석:
# Shiller의 "div"는 **12개월 후행 배당금 총합** (annual)
# 이것을 252로 나누면: 일일 배당금 = (연 배당금) / 252
# ffill로 확장하면: 각 일일에 동일한 배당금 수익률을 적용
# → 실제로는 배당금은 불규칙적으로 분배됨 (계절성 있음)
# → 결과적으로 "오버스무딩된 배당금 재구성"
```

**금융 관점의 정확성 문제**:
- S&P 500의 실제 배당금 지급:
  - 분기별 지급이 아님: 개별 기업이 다른 시점에 지급
  - 분기마다 배당액 변함
  - 월별 데이터로 선형 보간하면 실제 분포와 다름

**검증되지 않은 점**:
```
코드에서:
- Shiller 배당 수익률 vs 실제 S&P 500 배당 수익률 비교: ❌ 없음
- 복제된 총수익률 vs 공식 S&P 500 총수익률 비교: ❌ 없음
- 특정 기간(예: 2010-2020)에서 검증: ❌ 없음
```

**문제 2: 시간 인덱싱 불일치**
```python
years = date_col.astype(int)
months = ((date_col - years) * 100).round().astype(int).clip(1, 12)  # line 68-69
dates = pd.to_datetime({"year": years, "month": months, "day": 1})  # line 70

# 예시: date_col = 1928.1 → month 1, day 1
# 하지만 Shiller 데이터의 월/연 해석이 정확한가?
# Shiller 데이터의 정확한 구조를 확인하지 않음
```

**추천 개선사항**:
```python
# 1. Shiller 배당금 검증 스크립트 추가 (매년 기금 총합 검증)
# 2. 공식 SP500TR 지수와 비교 (Federal Reserve는 공식 데이터 제공)
# 3. 선형 보간이 아닌 cubic spline이나 실제 배당 일정 사용
```

### 2.2 Ken French 리스크프리 레이트

**코드** (line 97-132):

#### ✅ 강점
- 공식 학술 자료 (Dartmouth)
- 일일 데이터 제공

#### ⚠️ **문제**

**문제 1: 형식 파싱의 강건성**
```python
for line in lines:
    parts = line.strip().split(",")
    if len(parts) >= 5 and parts[0].strip().isdigit():  # line 117
        data_lines.append(parts)

# 문제: CSV 형식 변경되면 깨짐
# 더 강건한 방법: pd.read_csv() 직접 사용
```

**문제 2: 폴백 값이 너무 높음**
```python
if isinstance(tbill_rate, pd.Series):
    ...
else:
    tbill_daily = tbill_rate / 252  # line 531

# default fallback: 0.03 / 252 (3% annual = 기본값) (line 131)

# 2010-2020: 평균 RF = 0.1~1.5% (Fed funds rate)
# 3% 폴백은 TQQQ 캘리브레이션에서 **자동으로 더 높은 비용 추정**을 야기
```

**추천**:
```python
# 더 현실적인 폴백
return _CACHE = pd.Series(0.015 / 252, index=idx)  # 1.5% annual
# 또는 에러를 발생시키기 (계속하기보다 실패 명시)
```

### 2.3 yfinance 데이터 품질

**코드** (line 38-53):

#### ⚠️ **심각한 문제**

**문제 1: 승인 조정(auto_adjust)의 숨겨진 위험**
```python
df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

# auto_adjust=True:
# - 권리락, 배당락, 액면분할 등을 **자동으로 역조정**
# - 결과: 역사적 가격이 왜곡될 수 있음
# - 특히 1980년대 이전: 분할/합병 많음 → 가격 정확성 의문

# 더 나은 방법: auto_adjust=False (조정 없음) + 수동 조정
```

**문제 2: 데이터 검증 부재**
```python
df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
if df.empty:
    raise ValueError(f"No data for {ticker}")  # line 45-46

# 검증 항목 누락:
# - NaN 비율 (5% 이상이면 경고)
# - 극단적 점프 감지 (어느 날 갑자기 50% 하락 → 분할? 오류?)
# - 거래량 검증 (주말이나 휴장일 데이터 포함?)
```

**추천**:
```python
def validate_price_data(df, ticker):
    """데이터 품질 검증"""
    # 1. NaN 비율
    nan_pct = df.isnull().sum().sum() / df.size
    if nan_pct > 0.01:
        print(f"Warning: {ticker} has {nan_pct:.1%} NaN values")

    # 2. 극단적 점프 (1일 >20% 변동)
    returns = df["Close"].pct_change()
    jumps = returns[returns.abs() > 0.20]
    if len(jumps) > 0:
        print(f"Warning: {ticker} has {len(jumps)} days with >20% moves")
        # 이것이 분할인지 오류인지 확인 필요

    # 3. 연속 NaN (데이터 갭)
    max_gap = (df.isnull()).groupby((df.isnull() != df.isnull().shift()).cumsum()).sum().max()
    if max_gap > 5:  # 5일 이상 연속 NaN
        print(f"Warning: {ticker} has {max_gap}-day NaN gap")
```

### 2.4 결론: 자료 수집

| 항목 | 신뢰성 | 비고 |
|------|--------|------|
| yfinance (1980년 이후) | ⭐⭐⭐⭐ | 공식, 널리 사용 |
| Ken French RF | ⭐⭐⭐⭐⭐ | 학술 표준 |
| Shiller 배당 | ⭐⭐⭐ | **월별→일일 보간이 정확성 저하** |
| 전체 방법론 | ⭐⭐⭐ | 검증 스크립트 필요 |

---

## 3. TQQQ 모방 데이터 생성

### 3.1 구조적 문제

**기본 가정** (calibrate_tqqq.py, line 71-124):
```python
sim_cum = run_buy_and_hold(qqq, leverage=3.0, expense_ratio=er)
actual_cum = run_buy_and_hold(tqqq, leverage=1.0, expense_ratio=0.0)

# 비교: 3x QQQ with ER vs 1x TQQQ
```

#### ⚠️ **심각한 결함: 일일 리밸런싱 vs 드리프트**

**실제 TQQQ의 구조**:
- ProShares TQQQ는 **일일 리밸런싱** ETF
- 매일 종가에 정확히 3배 유지
- 수학: `TQQQ(t) = TQQQ(t-1) × (1 + 3×QQQ_return(t) - cost)`

**코드의 모방 모델**:
```python
def apply_leverage(daily_ret, leverage, expense_ratio):
    daily_cost = expense_ratio / 252
    return daily_ret * leverage - daily_cost  # line 507-508

def run_buy_and_hold(price, leverage=1.0, expense_ratio=0.0):
    daily_ret = price.pct_change()
    lev_ret = apply_leverage(daily_ret, leverage, expense_ratio)
    cum = (1 + lev_ret).cumprod()
    return cum
```

**이것은 정확함** ✅

#### ✅ 하지만 다른 문제가 있음

**문제 1: 시간 경과에 따른 비용 모델 단순화**
```python
# Fixed ER sweep (line 71-124)
daily_cost = expense_ratio / 252
return daily_ret * leverage - daily_cost

# Time-varying model (line 131-150)
daily_financing = rf_aligned * 2.0 * spread
daily_stated = TQQQ_STATED_ER / 252
daily_total_cost = daily_stated + daily_financing  # line 149-150
```

**문제점**:
- Fixed ER 모델: **비용이 시간 불변** → 현실성 부족
  - 실제: 금리 변화 → 차입 비용 변함
  - 2010-2012: 연 0.1% RF
  - 2022-2024: 연 5% RF
  - **차이: 50배**

- Time-varying 모델: **선형 가정** → 수학적으로 부정확
  ```python
  daily_financing = rf_aligned * 2.0 * spread
  # rf = 0.05/365 (일일 레이트)
  # 하지만 TQQQ의 실제 차입 비용은 레포 시장 스프레드에 따라 비선형
  ```

**문제 2: 컨본더링 효과 무시**
```python
# Quoted spread_values = [0.8, 0.85, ..., 1.50] (line 143)
# 하지만 이것은 순수 추측
```

실제 TQQQ 캘리브레이션:
- 2010-2020: 공식 ER = 0.95% (고정)
- 실제 성능: ER=0.95%로는 설명 불가
- **숨겨진 비용**: 스프레드, 리밸런싱 손실, 추적 오차

**검증 방법** (코드에는 없음):
```python
# 실제 TQQQ vs 모방 3x QQQ 비교
# 기간: 2010-2025 (TQQQ 전 역사)

# 과제 1: 같은 기간의 일일 추적 오차 분석
# 과제 2: 변동성 체제별 추적 오차 (저변동 vs 고변동)
# 과제 3: 급락장 (COVID, 2022)에서의 추적 오차

# 현재 코드:
# ✅ 대략적인 ER 추정
# ❌ 상세한 추적 오차 분석
# ❌ 변동성 체제별 분석
```

**문제 3: Calibration의 순환성**
```python
# 코드 흐름:
1. calibrate_tqqq.py: TQQQ와 3x QQQ 비교 → ER 결정
2. leverage_rotation.py Part 12: 결정된 ER 사용

# 문제:
# ER을 'TQQQ와 가장 가까운 값'으로 설정했는데,
# 이것이 '3x 레버리지 전략의 최적 성능'을 나타내는가?
# → No. 이것은 '실제 ETF 모방'만 할 뿐
```

### 3.2 결론: TQQQ 모방

| 항목 | 평가 | 비고 |
|------|------|------|
| 기본 구조 (일일 리밸런싱) | ✅ | 정확함 |
| Fixed ER 모델 | ⭐⭐⭐ | 단순화, 현실성 부족 |
| Time-varying 모델 | ⭐⭐ | 가정 불명확, 선형 가정 |
| 검증 | ❌ | **상세 추적오차 분석 없음** |
| 최종 ER=3.5% | ⭐⭐⭐⭐ | 합리적이나 한계 명시 필요 |

---

## 4. 백테스트 엔진

### 4.1 `run_lrs()` 함수 분석

**코드** (line 511-542):
```python
def run_lrs(price, signal, leverage=2.0, expense_ratio=0.01,
            tbill_rate=0.02, signal_lag=0, commission=0.0):
    daily_ret = price.pct_change()
    sig = signal.shift(signal_lag) if signal_lag > 0 else signal  # line 524
    lev_ret = apply_leverage(daily_ret, leverage, expense_ratio)

    strat_ret = sig * lev_ret + (1 - sig) * tbill_daily  # line 533

    if commission > 0:
        trades = sig.diff().abs().fillna(0)
        strat_ret = strat_ret - trades * commission  # line 538

    cum = (1 + strat_ret).cumprod()
    return cum
```

#### ✅ 강점
- **신호 지연 처리**: `signal_lag` 개념이 명확
- **T-Bill 합성**: 현실적 대체 자산
- **수수료 처리**: 신호 변화 시점에 적용

#### ⚠️ **심각한 결함**

**문제 1: 신호 지연의 불완전한 해석**
```python
sig = signal.shift(signal_lag) if signal_lag > 0 else signal

# 이것이 의미하는 바:
# signal_lag=1이면: t일 신호 → t+1일 실행
# 하지만 현실은:
# - t-1 종가 신호 계산
# - t일 종가에서 실행
# → 0.5 trading day 차이

# 또한:
# sig.shift(1)은 이전 신호를 사용하지만,
# .pct_change()는 현재까지의 수익률
# → 미묘한 타이밍 오차 가능
```

**더 정확한 구현**:
```python
def run_lrs_v2(price, signal, leverage=2.0, expense_ratio=0.01,
               tbill_rate=0.02, signal_lag=1, commission=0.0):
    """signal_lag=1 means:
    - Signal computed on price[t] close
    - Executed on price[t+1] close using price[t+1]/price[t] return
    """
    daily_ret = price.pct_change()  # ret[t] = price[t] / price[t-1] - 1

    # 신호를 다음날로 이동 (t일 신호 → t+1일 실행)
    sig = signal.shift(signal_lag).fillna(0)

    # 확인: 마지막 신호가 사라졌나?
    # 아니면 새로운 신호로 채워져야 하나?
    # 코드에서는 명확하지 않음
    return cum
```

**문제 2: 신호 경계에서의 처리**
```python
# 예시: signal = [0,0,0,1,1,1,0,0,1,...]
#      signal.diff() = [NaN, 0, 0, 1, 0, 0, -1, 0, 1, ...]
#      trades = [0, 0, 1, 0, 0, 1, 0, 1, ...]

# 코드: trades = sig.diff().abs().fillna(0)  # line 537
# 문제: 첫 번째 행 (NaN)을 0으로 처리
# 하지만 signal[0]이 1이면 → 실제로는 trading 발생 (초기 진입)

# 현실성:
# - 백테스트 시작 전에 이미 포지션을 가지고 있는가?
# - 아니면 현금 상태에서 시작하는가?
```

**권장 개선**:
```python
# 초기 상태를 명시적으로 설정
initial_state = 0  # 현금 상태에서 시작
trades = np.zeros(len(signal))
for i in range(1, len(signal)):
    if sig.iloc[i] != sig.iloc[i-1]:
        trades.iloc[i] = 1.0
```

**문제 3: T-Bill 레이트 불일치**
```python
# 백테스트에서는:
# - Ken French RF 사용 (line 575-577 in calc_metrics)
# - 또는 flat rate (line 580-581)

# 하지만 run_lrs에서는:
# if isinstance(tbill_rate, pd.Series):
#     tbill_daily = tbill_rate.reindex(daily_ret.index, method="ffill")
# else:
#     tbill_daily = tbill_rate / 252

# 문제:
# run_lrs() 호출 시 tbill_rate를 명시하지 않으면
# 기본값 0.02 (2%)가 사용됨 → Ken French와 불일치 가능
```

**검색 결과** (line 1000+):
```python
# Part 7: slow_range step 3, fast_range step 1
run_dual_ma_analysis(
    ...,
    tbill_rate=0.02,  # 고정된 2%
    signal_lag=0,
    ...
)

# 하지만 Part 12:
rf = download_ken_french_rf()
run_dual_ma_analysis(
    ...,
    tbill_rate=rf,  # Ken French 시계열
    signal_lag=1,
    ...
)
```

**결론**: **불일치 O(심각)** — Part 7-11과 Part 12의 RF 가정이 다름

### 4.2 수수료 계산의 현실성

```python
commission = 0.002  # 0.2%
trades = sig.diff().abs().fillna(0)
strat_ret = strat_ret - trades * commission  # line 538
```

**문제**:
- 수수료가 일정하다고 가정
- 실제: 레버리지 거래는 슬리피지(slippage) 추가
  - 매수호가-매도호가 스프레드 (마이크로초 level)
  - 시장 임팩트 (대량 거래)
  - 옵션 재보험 비용 (3x ETF)

**현실성**:
```python
# 코드: 0.2% 수수료 (왕복)
# 실제:
# - 인덱스 선물: 0.01~0.05%
# - 레버리지 ETF: 0.1~0.5% (숨겨진 비용)
# - 손실: 0.3~0.5% (슬리피지)
# → 합계: 0.4~1.0%

# 결론: 0.2% 가정은 **낙관적** (불리한 방향으로)
```

### 4.3 결론: 백테스트 엔진

| 항목 | 평가 | 비고 |
|------|------|------|
| 기본 구조 | ✅ | 수학적으로 정확 |
| 신호 지연 구현 | ⭐⭐⭐ | 개념은 맞으나 미묘한 오차 |
| RF 일관성 | ❌ | Part 7-11 vs Part 12 불일치 |
| 수수료 현실성 | ⭐⭐ | 낙관적 가정 (0.2%) |
| 초기 상태 명시 | ❌ | 불명확 |

---

## 5. 메트릭 계산

### 5.1 CAGR (Compound Annual Growth Rate)

**코드** (line 570-572):
```python
total_ret = cum.iloc[-1] / cum.iloc[0]
cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0
```

#### ✅ 정확함 (최적화 없음)

---

### 5.2 Sharpe Ratio

**코드** (line 583-590):
```python
arith_annual = daily_ret.mean() * 252  # 산술 평균 수익률 연환산
vol = daily_ret.std() * np.sqrt(252)   # 일일 수익률의 표준편차 연환산

sharpe = (arith_annual - avg_annual_rf) / vol if vol > 0 else 0
```

**공식의 정확성**: **✅ Sharpe (1994)**

실제로는 2가지 Sharpe 정의가 있음:
1. **Sharpe (1994)**: `(E[R] - Rf) / σ[R]` (산술 평균 사용)
2. **Modified Sharpe**: `(CAGR - Rf) / σ` (기하 평균 사용)

코드는 명확히 (1)을 사용하고 있음 ✅

#### ⚠️ **문제: 입력 데이터의 일관성**

```python
daily_ret = cum.pct_change().dropna()  # line 566
arith_annual = daily_ret.mean() * 252

# 하지만:
# cum = (1 + strat_ret).cumprod()  (run_lrs line 540)
#
# strat_ret = sig * lev_ret + (1 - sig) * tbill_daily
#
# 그러면 daily_ret = cum.pct_change()는:
# daily_ret = cum[t] / cum[t-1] - 1
#           = strat_ret[t] (approximately, when cum changes)

# 하지만 정확히는:
# cum[t] / cum[t-1] = (1 + strat_ret[t])
# 따라서 daily_ret = strat_ret (정확)

# ✅ 일관성 있음
```

---

### 5.3 Sortino Ratio

**코드** (line 592-598):
```python
excess_daily = daily_ret - rf_daily
downside_diff = excess_daily.copy()
downside_diff[downside_diff > 0] = 0.0
downside_dev = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(252)
sortino = (arith_annual - avg_annual_rf) / downside_dev
```

**공식의 정확성**: **✅ Sortino & van der Meer (1991)**

Original Sortino:
```
Sortino = (R - Rf) / TDD

where TDD = sqrt(E[min(r - Rf, 0)^2])  (Target Downside Deviation)
           = sqrt(mean((downside_diff)^2))
```

코드가 정확함 ✅

#### ⚠️ **하지만 실무 이슈**

**문제 1: Target Return 설정**
```python
downside_diff = excess_daily - 0  # 암묵적으로 target = 0 (or Rf)
```

Sortino의 'Target'은:
- 원래: MAR (Minimum Acceptable Return)
- 코드: Rf (Risk-free rate)

**이것은 합리적이나**, 논문과 확인 필요

**문제 2: 작은 표본의 편향**
```python
downside_dev = np.sqrt((downside_diff ** 2).mean())  # 표본 분산

# 통계적 편향:
# E[S^2] ≠ σ^2 (표본 분산은 불편향 추정량)
# Bessel 보정 미사용: n/(n-1) 미곱함

# 결론:
# downside_dev는 약간 낮게 추정됨
# → Sortino 값은 약간 높게 추정됨 (~1~2%)
```

**개선**:
```python
downside_dev = np.sqrt((downside_diff ** 2).sum() / (len(downside_diff) - 1))
```

---

### 5.4 MDD (Maximum Drawdown)

**코드** (line 600-603):
```python
running_max = cum.cummax()
drawdown = cum / running_max - 1
mdd = drawdown.min()
```

#### ✅ 정확함

---

### 5.5 MDD_Entry (Entry-Based Maximum Drawdown)

**최근 추가된 메트릭** (memory.md 참조):

```python
def _max_entry_drawdown(cum, signal):
    """MDD from entry point maximum, not peak"""
    entry_max = 1.0
    mdd_entry_values = []

    for i in range(len(cum)):
        if signal[i] == 1 and signal[i-1] != 1:  # Entry
            entry_max = cum[i]
        elif signal[i] == 1:  # In position
            entry_max = max(entry_max, cum[i])

        mdd = cum[i] / entry_max - 1
        mdd_entry_values.append(mdd)

    return min(mdd_entry_values)
```

#### ✅ **강력한 추가 메트릭**

**이유**:
- 전통 MDD: 모든 역사적 피크로부터 계산
- MDD_Entry: 진입 시점의 피크로부터만 계산
- **실무 의미**: "이 전략에 들어갔을 때 최대 손실?"

#### ⚠️ **구현 위치의 문제**

코드에서 `_max_entry_drawdown()`은:
```python
# leverage_rotation.py에 정의되어야 하는데,
# memory.md에만 언급됨
# → 구현 위치 불명확
```

확인 필요:
```bash
grep -n "_max_entry_drawdown" leverage_rotation.py
```

---

### 5.6 결론: 메트릭 계산

| 메트릭 | 정확성 | 비고 |
|--------|--------|------|
| CAGR | ✅ | 정확 |
| Sharpe | ✅ | Sharpe 1994 명시적 |
| Sortino | ⭐⭐⭐⭐ | 정확하나 Bessel 보정 누락 (1~2% 오차) |
| MDD | ✅ | 정확 |
| MDD_Entry | ✅ | 우수한 추가 메트릭 (구현 위치 확인 필요) |
| RF 일관성 | ❌ | Part 7-11 vs Part 12 불일치 |

---

## 6. 신호 생성기

### 6.1 `signal_dual_ma()`

**코드** (line 188-195):
```python
def signal_dual_ma(price, slow=200, fast=50):
    if fast >= slow:
        fast = max(slow // 4, 10)
    return (price.rolling(fast).mean() > price.rolling(slow).mean()).astype(int)
```

#### ✅ 강점
- 명확한 Golden Cross 로직
- 자동 fast MA 조정

#### ⚠️ **문제**

**문제 1: 상태 머신 부재**
```python
# 반환: 1 (above) or 0 (below) — 매일 계산됨
# 문제: 노이즈에 민감
# 예시:
# - slow MA = 200
# - fast MA 위치: 200.1 → 200.05 → 200.1 → ...
# → 신호: 1 → 0 → 1 → ... (whipsaw 발생)

# 개선: 상태 머신 사용
```

**문제 2: 초기 조건**
```python
price.rolling(slow).mean()
# 처음 slow일은 NaN 반환
# 신호: [NaN, NaN, ..., NaN, 0 or 1, ...]

# 하지만 NaN이 시그널에 포함되면:
# astype(int)는 어떻게 처리하나?
# → float(NaN) → int(NaN) = ???
```

**검증**:
```python
import numpy as np
print(int(np.nan))  # ValueError
```

결론: NaN 처리가 불명확하며, 실행 시 오류 가능

---

### 6.2 `signal_asymmetric_dual_ma()`

**코드** (line 208-238):
```python
def signal_asymmetric_dual_ma(price, fast_buy, slow_buy, fast_sell, slow_sell):
    buy_cond = (buy_fast_ma > buy_slow_ma).values
    sell_cond = (sell_fast_ma < sell_slow_ma).values

    sig = np.zeros(n, dtype=int)
    state = 0
    for i in range(n):
        if state == 0 and buy_cond[i]:
            state = 1
        elif state == 1 and sell_cond[i]:
            state = 0
        sig[i] = state
```

#### ✅ **강점**
- **상태 머신**: whipsaw 방지
- **비대칭 MA**: 매수/매도 조건 분리
- 이론: 하락장에서 빠른 회피, 상승장에서 늦은 진입

#### ⚠️ **심각한 문제**

**문제 1: 비대칭성의 검증 부재**
```python
# 코드는 4개 MA 파라미터를 받지만,
# 최적화 과정에서 이것들이 실제로 **다른 값**으로 수렴하는가?

# memory.md 인용:
# "Confirmed asymmetric structure converges to symmetric —
#  no value in buy/sell separation"

# 즉, 최적화 결과:
# fast_buy ≈ fast_sell
# slow_buy ≈ slow_sell

# 결론: 4개 파라미터는 실제로 2개의 자유도만 가짐
# → 모델 복잡도 증가, 이점 없음
```

**문제 2: 상태 머신의 락인 위험**
```python
# 예시: 하락장 진입 후 매도 신호 못받기
state = 1  # in position
sell_cond = False  # fast_sell < slow_sell이 아직 false

# 상황: 가격이 -50% 떨어졌는데,
# slow_sell MA가 아직 높아서 매도 신호 안 나옴
# → 손실 확대

# 해결책: Time-based exit 또는 loss-stop 추가
# 현재 코드: ❌ 없음
```

---

### 6.3 `signal_vol_adaptive_dual_ma()`

**코드** (line 241-285):
```python
def signal_vol_adaptive_dual_ma(price, base_fast, base_slow,
                                vol_lookback=60, vol_scale=1.0):
    vol_ratio = (rolling_vol / ref_vol).clip(0.3, 3.0)
    scale = 1.0 + vol_scale * (vol_ratio - 1.0)
    fast_eff = np.clip(np.round(base_fast * scale), 2, 100)
    slow_eff = np.clip(np.round(base_slow * scale), 30, 500)
```

#### ✅ **강점**
- **적응형**: 변동성에 따라 MA 길이 조정
- **직관**: 고변동성 → 느린 MA (필터), 저변동성 → 빠른 MA

#### ⚠️ **문제**

**문제 1: 확장 중앙값(expanding median) 사용**
```python
ref_vol = rolling_vol.expanding().median()  # line 260

# 즉, vol_ratio[t]는:
# rolling_vol[t] / median(rolling_vol[0:t])

# 문제:
# - 초반부 데이터가 적을 때 중앙값이 변동성 높음
# - 시간이 지나면서 중앙값이 변함 → 신호 왜곡 가능
# - 과거 극단적 변동성이 중앙값을 높여서,
#   현재 "고" 변동성이 상대적으로 "저" 변동성으로 보일 수 있음

# 예시:
# 2008 금융위기: rolling_vol = 80% (극도로 높음)
# 2010-2020: rolling_vol = 15% (정상)
# ref_vol[2020] = median(80%, 15%, 15%, ..., 15%)
#               = 15% (결과적으로 중위값은 2008 이상의 극단값 포함)
# → vol_ratio[2020] = 15% / 15% = 1.0 (중립)

# 더 나은 방법:
ref_vol = rolling_vol.rolling(250).median()  # 1년 중앙값
```

**문제 2: 클립핑의 강성**
```python
vol_ratio = vol_ratio.clip(0.3, 3.0)  # line 261

# vol_ratio < 0.3이면 0.3으로 고정
# vol_ratio > 3.0이면 3.0으로 고정

# 문제:
# - 극단적 상황(팬데믹, 금융위기)에서 클립핑이 정보 손실
# - 0.3, 3.0이 최적값인가? (검증 없음)
# - 동적 클립핑(퍼센타일 기반)이 나을 수 있음

# 확인해야 할 점:
# 이 파라미터들이 최적화 과정에서 튜닝되었는가?
```

**문제 3: MA 길이의 클립핑**
```python
fast_eff = np.clip(np.round(base_fast * scale), 2, 100)
slow_eff = np.clip(np.round(base_slow * scale), 30, 500)

# 예시:
# base_fast = 10, scale = 0.5 → fast_eff = max(5, 2) = 5
# base_slow = 50, scale = 0.3 → slow_eff = max(15, 30) = 30
# → fast_eff = 5, slow_eff = 30 (비율 6:1)

# 문제:
# - clipping으로 인해 의도한 scale이 반영 안될 수 있음
# - 특히 base_slow가 작을 때 slow_eff가 최소값(30)에 갇힘
# - 결과: vol_scale의 효과 감소
```

---

### 6.4 `signal_regime_switching_dual_ma()`

**코드** (line 288-331):
```python
def signal_regime_switching_dual_ma(price, fast_low, slow_low,
                                     fast_high, slow_high,
                                     vol_lookback=60,
                                     vol_threshold_pct=50.0):
    rolling_vol = daily_ret.rolling(vol_lookback).std() * np.sqrt(252)
    vol_pct = rolling_vol.expanding().rank(pct=True) * 100
    high_vol = (vol_pct >= vol_threshold_pct).values
```

#### ✅ **강점**
- **명확한 체계**: 저/고 변동성 체계 분리
- **상태 머신**: whipsaw 방지
- **이론적 근거**: 위험 회피 (고변동성) vs 위험 추구 (저변동성)

#### ⚠️ **심각한 문제**

**문제 1: Expanding percentile의 위험**
```python
vol_pct = rolling_vol.expanding().rank(pct=True) * 100

# expanding().rank(pct=True)는:
# vol_pct[t] = rank(rolling_vol[0:t]) / len(rolling_vol[0:t])

# 예시:
# rolling_vol = [10%, 12%, 11%, 50%, 15%, ...]
# vol_pct = [50%, 100%, 75%, 100%, 75%, ...]
# (첫 값은 항상 50%, 극대값은 100%)

# 문제:
# - 초반부: 데이터 적음 → 순위 변동성 큼
# - 장기: 과거의 극단값이 모든 현재값을 압박
# - 2008년 80% vol이 있으면,
#   2015년의 20% vol은 절대 "고 변동성"으로 분류 안됨

# 극단 예시:
# 2008 vol = 80% (rank 100%)
# 이후 모든 vol = 20% (rank ~0%)
# vol_pct = 0% (항상 "저 변동성" 체계 사용)
```

**검증 필요**:
```python
# 코드에서 vol_threshold_pct = 50.0으로 설정
# 즉, vol_pct >= 50인 날을 "고변동성"으로 분류
# 하지만 expanding rank의 특성상,
# 이것은 **역사적 중앙값 이상**을 의미

# 문제: 2010-2020 시대,
# 2008 금융위기 이후의 "정상" 변동성(15%)과
# COVID 팬데믹 때의 "고" 변동성(30%)
# 모두 expanding percentile에서 "50% 이하"로 분류될 수 있음

# 결론: 아주 위험한 설계 오류
```

**개선**:
```python
def signal_regime_switching_dual_ma_v2(price, fast_low, slow_low,
                                        fast_high, slow_high,
                                        vol_lookback=60,
                                        vol_threshold_pct=50.0):
    rolling_vol = daily_ret.rolling(vol_lookback).std() * np.sqrt(252)

    # 방법 1: 최근 1년 중앙값 대비
    ref_vol = rolling_vol.rolling(252).median()
    vol_ratio = rolling_vol / ref_vol.clip(lower=0.01)
    high_vol = (vol_ratio > np.percentile(vol_ratio[-252:], vol_threshold_pct))

    # 또는 방법 2: 절대값 임계값
    vol_threshold_abs = 0.20  # 20% annualized vol
    high_vol = (rolling_vol > vol_threshold_abs).values
```

**문제 2: 체계 전환의 지연**
```python
# 신호는 상태 머신인데,
# vol_pct는 expanding percentile로 계산됨

# 결과: 변동성이 급증해도,
# expanding percentile이 충분히 높아질 때까지 "저변동성" 체계 유지

# 예: COVID 팬데믹 (2020.3)
# 하루밤에 vol 10% → 40%로 급등
# 하지만 expanding percentile에서:
# vol_pct[2020.3.16] = ? (과거 10년 데이터 포함, 매우 낮음)
# → "고변동성"으로 인식 안되고 "저변동성" 계속
# → 매수 신호로 여전히 공격적 (위험!)
```

---

### 6.5 `signal_vol_regime_adaptive_ma()`

복합 신호 (vol-adaptive + regime-switching)

#### ⚠️ **주요 문제**

**문제: 파라미터 수 증가로 인한 과적합**
```python
# 7개 파라미터:
# base_fast_low, base_slow_low,
# base_fast_high, base_slow_high,
# vol_lookback, vol_threshold_pct,
# vol_scale

# 최적화 데이터: 1987-2025 (38년, ~9,500 거래일)
# 신호 변화: 대략 1~2회/년 (약 40~80 거래일)

# 자유도 문제:
# 7개 파라미터 × 수천 조합 vs 40 거래일 신호 변화
# → 과적합 위험: 매우 높음
```

---

### 6.6 결론: 신호 생성기

| 신호 | 상태머신 | 파라미터 | 검증 | 평가 |
|------|---------|---------|------|------|
| dual_ma | ❌ | 1-2 | ❌ | ⭐⭐ |
| asymmetric | ✅ | 4 (실질 2) | ⭐⭐ | ⭐⭐⭐ |
| vol_adaptive | ✅ | 4 | ❌ | ⭐⭐ |
| regime_switching | ✅ | 6 | ❌❌ | ⭐⭐ (expanding percentile 문제) |
| vol+regime | ✅ | 7 | ❌ | ⭐ (과적합 위험) |

---

## 7. 최적화 프로세스

### 7.1 `optimize_regime_grid_v2.py` 구조

**목표**: 6개 파라미터 최적화
- fast_low, fast_high: 2~50, step 5
- slow_low, slow_high: 50~350, step 10
- vol_lookback: 20~120, step 10
- vol_threshold_pct: 30~75, step 5

**총 조합**: ~10.6M

#### ✅ **강점**
- **전수 조사**: 그리드 서치로 모든 조합 평가
- **최적화**: numpy/numba 사용으로 빠른 계산

#### ⚠️ **심각한 문제**

**문제 1: 과적합 (Overfitting)**

```
데이터:
- Period: 1987-2025 (38년)
- 거래일: 약 9,500일
- 신호 변화 (trades): 약 50~100회 (연 1.5~2.5회 가정)

최적화:
- 그리드: 10.6M 조합
- 한 조합당 평가: ~1ms (numba)
- 총 시간: ~3시간

과적합 위험 분석:
- 파라미터: 6개
- 자유도: 6
- 데이터 포인트 (의미있는): ~50 거래
- 비율: 6 파라미터 / 50 거래 = 0.12 (극히 높음)

통계 기준:
- 보수적: 파라미터 1개당 최소 20 데이터 → 6 × 20 = 120 필요
- 현재: 50/6 ≈ 8 데이터/파라미터 (충분하지 않음)

결론: **심각한 과적합 위험**
```

**검증 방법** (코드에 없음):
```python
# Walk-forward testing
for year in [2019, 2020, 2021, 2022, 2023, 2024]:
    train_end = year-1
    train_data = data[:train_end]
    test_data = data[train_end:year]

    # train_data에서 최적 파라미터 찾기
    best_params = optimize(train_data, ...)

    # test_data에서 성능 평가
    test_perf = evaluate(best_params, test_data)

    # In-sample (train) vs out-of-sample (test) 비교
    # 성능 저하가 50% 이상이면 과적합 의심
```

현재 코드: **❌ 없음**

**문제 2: Expanding Percentile 버그 (다시 언급)**

```python
# optimize_regime_grid_v2.py line 84-85
vol_pct = rolling_vol.expanding().rank(pct=True) * 100

# 이전 섹션에서 지적:
# - 역사적 극단값의 영속적 영향
# - 현재 신호와 과거의 decoupling
```

**문제 3: Warmup 기간의 불명확성**

```python
WARMUP_DAYS = 500  # line 56

# 하지만:
# - vol_lookback = 60 (최대)
# - slow_high = 350 (최대)
# - expanding percentile = 모든 역사 (최악)

# 워밍업이 충분한가?
# 답: 500 < 350인 경우도 있음 (느린 MA 초기값이 NaN)

# 더 정확한 워밍업:
warmup = max(slow_high, 250)  # 1년 + 최대 MA
```

**문제 4: Penalised Objective 부족**

```python
penalised = sortino - alpha * trades_yr  # line 185

# ALPHA = 0.02 (line 57)
# 즉, 1회 매매 증가 시 Sortino 0.02 감소

# 문제:
# - 과거 부분에서 **동전 던지기** 신호와 진정한 신호 구분 불가
# - alpha = 0.02는 임의적 (이론적 근거 없음)
# - 실제 거래 비용(0.2% commission)과의 불일치

# 더 나은 방법:
# penalised = sortino - (commission * trades_yr)
# = sortino - 0.002 * trades_yr
```

---

### 7.2 Plateau 식별 알고리즘

**코드** (optimize_regime_grid_v2.py 중간부):
```
Phase 1: Coarse grid (10.6M combos) → coarse_results
Phase 2: Top 1% 선택 → 이웃 평균화 → greedy 선택 (다양성)
Phase 3: Fine grid (±5, step 1) 각 plateau 주변
```

#### ✅ **강점**
- 합리적인 3단계 접근

#### ⚠️ **문제**

**문제 1: "Top 1%" 선택의 의미**

```python
# Top 1%는 몇 개인가?
# 10.6M × 1% = 106,000개

# 그 중에서 "plateau"를 어떻게 찾는가?
# → 인접한 파라미터들의 성능이 비슷한 영역

# 문제:
# - 106,000개 중 실제로는 수백~수천개의 "true plateau"
# - 나머지는 노이즈 또는 과적합된 피크
# - 이를 구분하는 방법이 설명되지 않음
```

**문제 2: Greedy 선택의 편향**

```python
# "최소 거리 3.0의 L2 distance"로 plateau 분리
# → 하지만 이 거리가 파라미터 공간에서 의미있는가?

# 예:
# plateau 1: fast_low=7, fast_high=17 (파라미터 차이: 10)
# plateau 2: fast_low=8, fast_high=18 (파라미터 차이: 10)
# L2 distance = sqrt(1^2 + 1^2) = 1.4 (분리 안됨)

# 하지만 성능이 완전히 다를 수 있음:
# plateau 1 성능: Sortino 1.09
# plateau 2 성능: Sortino 0.98

# 결론: 거리 기반 분리는 성능 차이를 무시할 수 있음
```

---

### 7.3 결론: 최적화 프로세스

| 항목 | 평가 | 비고 |
|------|------|------|
| 그리드 범위 | ⭐⭐⭐ | 합리적 |
| 과적합 위험 | ❌ | 심각 (W-F 테스트 없음) |
| Expanding percentile | ❌ | 버그 (앞서 지적) |
| Plateau 식별 | ⭐⭐ | 휴리스틱, 이론적 근거 약함 |
| Penalised objective | ⭐⭐ | alpha=0.02 임의적 |

---

## 8. 종합 결론

### 🎯 **핵심 발견**

| 단계 | 신뢰성 | 핵심 문제 | 심각도 |
|------|--------|---------|--------|
| **1. 전체 목표** | ⭐⭐⭐ | 목표(복제) vs 방법(확장) 불일치 | 중간 |
| **2. 자료 수집** | ⭐⭐⭐ | Shiller 배당 보간, 검증 부재 | 중간 |
| **3. TQQQ 모방** | ⭐⭐⭐ | 시간가변 비용 모델 불명확 | 중간 |
| **4. 백테스트** | ⭐⭐⭐ | RF 일관성, 수수료 낙관적 | 중간 |
| **5. 메트릭** | ⭐⭐⭐⭐ | Sortino Bessel 보정 누락 | 낮음 |
| **6. 신호 함수** | ⭐⭐ | Expanding percentile 버그, 파라미터 수 과다 | **높음** |
| **7. 최적화** | ⭐⭐ | **심각한 과적합, W-F 테스트 없음** | **극도로 높음** |

### ⚠️ **가장 심각한 3가지 문제**

#### 1️⃣ **Expanding Percentile 설계 오류**
```
위치: leverage_rotation.py line 303, optimize_regime_grid_v2.py line 84
문제: 역사적 극단값(2008 금융위기)이 모든 현재 신호를 왜곡
영향: regime-switching 신호의 신뢰성 ❌
```

#### 2️⃣ **심각한 과적합 (과도한 최적화)**
```
위치: optimize_regime_grid_v2.py 전체
문제: 6 파라미터 / 50 거래 = 심각한 과적합
증거: Walk-forward 테스트 없음
영향: 미래 성능 예측 불가능
```

#### 3️⃣ **신호 간 불일치**
```
위치: Part 7-11 (lag=0, RF=2%) vs Part 12 (lag=1, RF=Ken French)
문제: 같은 MA 조합의 성능이 파트마다 다름
증거: run_lrs() 호출 시 tbill_rate 기본값 0.02
영향: 분석 결과 재현 불가능
```

### ✅ **우수한 점**

1. **명확한 코드 구조**: 데이터→신호→백테스트→메트릭 분리
2. **여러 검증**: eulb 비교, TQQQ 캘리브레이션
3. **우수한 메트릭**: MDD_Entry 추가 (실무적)
4. **성능 계산**: Sharpe, Sortino 공식이 정확

---

### 📋 **개선 로드맵 (우선순위)**

#### 🔴 **반드시 수정** (신뢰성 회복)
```
1. Expanding percentile 제거
   → rolling percentile (1년 기반) 또는 절대값 임계값 사용

2. Walk-forward 테스트 추가
   - 2010-2015 훈련 → 2015-2020 테스트
   - 2015-2020 훈련 → 2020-2025 테스트
   - in-sample vs out-of-sample 비교

3. RF 일관성
   - Part 7-11: Ken French RF 사용
   - Part 12: 동일 설정
```

#### 🟠 **강력히 권장** (정확성 향상)
```
1. Shiller 배당 검증
   - 공식 S&P 500 TR과 비교
   - 특정 기간 성능 검증

2. TQQQ 추적 오차 분석
   - 변동성 체계별 분석
   - 급락장(COVID, 2022) 검증

3. Plateau 식별 개선
   - Greedy가 아닌 clustering 사용
   - 성능 기반 분리 (거리 X)

4. Bessel 보정
   - downside_dev 계산에 n/(n-1) 적용
```

#### 🟡 **고려** (선택적)
```
1. 상태 머신 모든 신호에 적용
2. Time-based exit 추가 (손실 제한)
3. 절대 RF 임계값 사용 (expanding percentile 대체)
```

---

### 🎓 **최종 평가**

**이 코드는:**
- ✅ **학술 재현**에는 적합 (Gayed 논문 비교 유효)
- ✅ **기초 분석**에는 유용 (Part 1-6)
- ⚠️ **최적화 결과**는 신뢰도 낮음 (과적합 위험)
- ❌ **미래 거래**에는 부적합 (out-of-sample 검증 없음)

**권장사항**:
1. **현재 Part 12 결과를 실제 거래에 사용하지 말 것**
2. **Walk-forward 테스트 추가 후 재평가 필요**
3. **Expanding percentile 버그 수정 필수**
4. **Part 1-6 기초 분석은 신뢰할 수 있음**

---

## 참고

이 리뷰는 다음을 기반으로 작성됨:
- leverage_rotation.py (1410+ 줄)
- calibrate_tqqq.py
- optimize_regime_grid_v2.py
- CLAUDE.md 프로젝트 문서
- MEMORY.md (최근 발견사항)

**마지막 업데이트**: 2026-02-27
