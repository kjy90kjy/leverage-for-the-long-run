# 시그널 최적화 로드맵

## 현재 시스템 한계

현재 레버리지 로테이션 전략(LRS)은 **대칭 시그널**을 사용한다:
- 하나의 dual MA 크로스오버 `(fast, slow)`가 매수와 매도를 동시에 결정
- fast MA > slow MA → 매수 (레버리지), fast MA < slow MA → 매도 (T-Bill)
- 파라미터 공간: 2차원 `(fast, slow)`

**문제점:**
- 최적의 진입 타이밍 ≠ 최적의 탈출 타이밍
- 시장 변동성이 높을 때와 낮을 때 같은 MA를 쓰는 것이 최적인지 의문
- 대칭 그리드 서치에서 나온 최적 파라미터가 진입/탈출 중 어느 쪽에 편향되었는지 알 수 없음

---

## Phase 1: 비대칭 매수/매도 시그널 (4 파라미터)

### 개념

매수와 매도에 각각 다른 dual MA 조건을 사용:
- **매수 조건**: `fast_buy` MA > `slow_buy` MA
- **매도 조건**: `fast_sell` MA < `slow_sell` MA

State machine (hysteresis) 방식으로 동작:
- 현재 cash(0) → 매수 조건만 체크
- 현재 invested(1) → 매도 조건만 체크

### 파라미터 공간

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `fast_buy` | 2~50 | 매수용 단기 MA |
| `slow_buy` | 50~350 | 매수용 장기 MA |
| `fast_sell` | 2~50 | 매도용 단기 MA |
| `slow_sell` | 50~350 | 매도용 장기 MA |

### 구현: `optimize_asymmetric.py`

- Optuna TPE sampler, 2000 trials, Sortino 최대화
- Walk-forward: 1985-2014 (train) / 2015-2025 (test)
- 대칭 baseline: (3,161), (3,200), (10,150)

### 결과

Train 최적: `(2, 89, 17, 346)` — "빨리 사고 천천히 판다"
- **Train**: Sortino 1.05, CAGR 30.2%
- **Test OOS**: Sortino 열위 — COVID 늦은 탈출, 2022 whipsaw 31회

### 결론

비대칭 구조 자체가 대칭 대비 유의미한 개선을 만들지 못함. 특히 전체 기간 최적화에서 Optuna가 `(50,346,50,345)` — 사실상 대칭 (50,345)로 수렴. 비대칭 buy/sell 분리보다 시장 국면 적응이 더 효과적.

---

## Phase 2: 적응형 시그널 최적화 (구현 완료)

Phase 1의 한계를 극복하기 위해 4가지 새로운 접근법을 구현.
**모든 최적화는 전체 기간(1987-2025, 38.3년)에서 실행.**

### 공통 조건

| 항목 | 값 |
|------|-----|
| 데이터 | ^NDX (1985-10-01 ~ 2025-12-30) |
| 평가 기간 | 1987-09-23 ~ 2025-12-30 (warmup=500일 이후) |
| 레버리지 | 3x (TQQQ 시뮬레이션) |
| 비용 | ER=3.5%, commission=0.2%, signal_lag=1 |
| 최적화 | Optuna TPE, 2000 trials |
| 목적함수 | Sortino - α × Trades/Year (과거래 페널티) |

### Approach C: Whipsaw 페널티 (`optimize_penalized_full.py`)

비대칭 4 파라미터 + 거래빈도 페널티:
```
목적함수: Sortino - 0.02 × Trades/Year
```
- 결과: `(50,346,50,345)` → 대칭으로 수렴, Sortino 0.965
- **결론**: 비대칭 구조 자체가 무의미함을 재확인

### Approach B: 변동성 적응형 MA (`optimize_all_full.py` Plan 4)

**핵심 아이디어**: 변동성이 높으면 MA를 늘려 whipsaw 방어, 낮으면 줄여 빠른 반응.

시그널 함수: `signal_vol_adaptive_dual_ma()`

```
rolling_vol = daily_ret.rolling(vol_lookback).std() × √252
ref_vol = rolling_vol.expanding().median()     # look-ahead 방지
vol_ratio = clip(rolling_vol / ref_vol, 0.3, 3.0)
scale = 1 + vol_scale × (vol_ratio - 1)
fast_eff = clip(round(base_fast × scale), 2, 100)
slow_eff = clip(round(base_slow × scale), 30, 500)
```

날마다 가변 window MA → cumsum trick으로 O(n) 최적화.

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `base_fast` | 2~50 | 기본 단기 MA |
| `base_slow` | 50~350 | 기본 장기 MA |
| `vol_lookback` | 20~120 | 변동성 계산 window |
| `vol_scale` | 0.0~2.0 | 스케일링 강도 (0=고정, 1=비례, 2=과민) |

- **결과**: `(45, 158, lb=73, scale=0.396)`, Sortino **1.085**, CAGR 34.6%
- 회복기간 1891일(최단), 연 1.2회 매매

### Approach D: 레짐 전환 MA (`optimize_all_full.py` Plan 5)

**핵심 아이디어**: 저변동 레짐과 고변동 레짐에서 다른 고정 MA pair 사용.

시그널 함수: `signal_regime_switching_dual_ma()`

```
레짐 판별: expanding percentile(threshold) of rolling vol
  → look-ahead 없이 현재까지 변동성 분포에서 위치 판단
저변동: SMA(fast_low) > SMA(slow_low) → 매수
고변동: SMA(fast_high) > SMA(slow_high) → 매수
```

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `fast_low` | 2~50 | 저변동 레짐 단기 MA |
| `slow_low` | 50~350 | 저변동 레짐 장기 MA |
| `fast_high` | 2~50 | 고변동 레짐 단기 MA |
| `slow_high` | 50~350 | 고변동 레짐 장기 MA |
| `vol_lookback` | 20~120 | 변동성 계산 window |
| `vol_threshold_pct` | 30~70 | 레짐 분류 백분위 임계값 |

- **결과**: lo=(48,323), hi=(15,229), lb=49, threshold=57.3%
- Sortino **1.088**, CAGR 34.9%, MDD_Entry **-39.2%** (가장 양호)
- **해석**: 저변동(추세장)에서 빠른 MA(48,323)로 추세 추종, 고변동에서 느린 MA(15,229)로 whipsaw 방어

### Approach B+D: Vol-Adaptive + Regime (`optimize_all_full.py` Plan 6)

**핵심 아이디어**: 레짐별 base MA 선택 + 변동성 비례 스케일링. 가장 복잡한 모델.

시그널 함수: `signal_vol_regime_adaptive_ma()`

```
1. 레짐 판별 (Approach D와 동일)
2. 레짐별 base MA pair 선택: (base_fast_low/slow_low) 또는 (base_fast_high/slow_high)
3. vol_scale 적용 (Approach B와 동일): 변동성에 비례해 MA 길이 조절
```

7개 파라미터, mild penalty α=0.01 (과적합 방어).

- **결과**: lo=(49,143), hi=(13,204), lb=84, threshold=49.1%, scale=0.588
- Sortino **1.103** (최고), CAGR **35.3%** (최고)

---

## 전체 기간 최종 비교 (1987-09-23 ~ 2025-12-30)

| 전략 | CAGR | Sharpe | Sortino | MDD | MDD_Entry | 회복(d) | Trd/Yr | Vol |
|------|------|--------|---------|-----|-----------|---------|--------|-----|
| P1 Asym (50,350,26,345) | 29.9% | 0.690 | 0.989 | -84.5% | -40.0% | 3409 | 6.8 | 60.2% |
| **P4 VolAdapt (45,158,lb73,s0.4)** | 34.6% | 0.754 | 1.085 | -79.9% | -58.8% | **1891** | 1.2 | 57.9% |
| **P5 Regime (lo48,323\|hi15,229)** | 34.9% | 0.762 | 1.088 | -79.9% | **-39.2%** | 2452 | 1.7 | 57.1% |
| **P6 VolRegime (lo49,143\|hi13,204)** | **35.3%** | **0.766** | **1.103** | -84.5% | -44.3% | 2456 | 1.6 | 57.3% |
| Sym best (11,237) | 30.4% | 0.703 | 1.002 | -79.9% | -41.2% | 1891 | 1.7 | 56.7% |
| Sym (3,161) eulb | 23.7% | 0.613 | 0.861 | -96.9% | -87.2% | 4964 | 4.2 | 53.9% |
| B&H 3x | 14.2% | 0.529 | 0.758 | -100.0% | -100.0% | 6479 | 0.0 | 79.1% |

---

## 핵심 발견

### 1. 비대칭 매수/매도 분리는 무의미
Phase 1(비대칭)과 Approach C(페널티 비대칭) 모두 전체 기간에서 Optuna가 대칭으로 수렴. 매수/매도 조건을 분리하는 것보다 시장 국면에 적응하는 것이 더 효과적.

### 2. 적응형 전략이 고정 MA를 일관되게 이김
P4/P5/P6 모두 대칭 최적 (11,237) 대비 Sortino +8~10%, CAGR +4~5%p 개선.
- Vol-Adaptive: 고변동 시 MA 자동 연장 → 닷컴/GFC에서 whipsaw 감소
- Regime-Switching: 고변동 시 느린 MA(15,229) 명시적 전환 → MDD_Entry -39% (최저)
- Vol+Regime: 두 효과 결합 → Sortino 최고

### 3. 3x 레버리지의 구조적 한계
어떤 MA 전략이든 전통 MDD -80% 이상은 피할 수 없음. 닷컴 같은 장기 하락장에서 MA 시그널이 베어마켓 랠리에 재진입하는 구조적 문제. MDD_Entry(매수가 대비 낙폭)가 실제 체감 리스크에 더 적합한 지표.

### 4. 최적 전략은 연 1~2회 매매
상위 전략 모두 연간 1~2회 거래. 고빈도 트레이딩이 아니라 장기 추세를 따라가되, 위기 시 한 번 빠지는 전략이 최적.

---

## 오버피팅 방어 전략 (공통)

### 1. 거래빈도 페널티
- 목적함수: `Sortino - α × Trades/Year`
- α=0.02 (4~6 params), α=0.01 (7 params)
- 과거래에 의한 noise fitting 방지

### 2. 파라미터 안정성 (Robustness)
- Top-20 trials의 파라미터 분포:
  - IQR이 전체 범위의 20% 이하 → robust (파라미터 고원 존재)
  - IQR이 전체 범위의 50% 이상 → flat landscape → 의미 있는 최적점 없음
- 최적 파라미터 주변 ±10% 변동 시 성능 변화가 작아야 함

### 3. Look-ahead 방지 (적응형 전략)
- `ref_vol = rolling_vol.expanding().median()` — 미래 데이터 사용 안 함
- `vol_pct = rolling_vol.expanding().rank(pct=True)` — 과거 분포 기준 백분위
- warmup=500일로 초기 불안정 구간 제거

### 4. 경제적 합리성
- Vol-Adaptive: "위기 시 더 보수적" — 직관적으로 합리적
- Regime-Switching: "저변동=추세장, 고변동=위기" — 시장 구조와 일치
- 설명 불가능한 파라미터 조합은 noise fitting일 가능성 높음

---

## 스크립트 목록

| 스크립트 | 접근법 | 파라미터 수 | 런타임 |
|---------|--------|------------|--------|
| `optimize_asymmetric.py` | Phase 1 비대칭 (train/test) | 4 | ~7분 |
| `optimize_penalized_full.py` | Approach C 페널티 (전체기간) | 4 | ~7분 |
| `optimize_all_full.py` | Plan 4/5/6 통합 (전체기간) | 4/6/7 | ~15분 |
| `optimize_common.py` | 공통 유틸리티 | — | — |

레거시 (train/test 분할 방식, 참고용):
- `optimize_penalized.py`, `optimize_walkforward.py`, `optimize_wf_penalized.py`
- `optimize_vol_adaptive.py`, `optimize_regime.py`, `optimize_vol_regime.py`
