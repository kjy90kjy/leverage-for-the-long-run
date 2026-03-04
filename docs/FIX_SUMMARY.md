# Expanding Percentile 버그 수정 완료

## 수정 내용

### 1. leverage_rotation.py

#### signal_regime_switching_dual_ma() (L303)
```python
# 이전 (버그)
vol_pct = rolling_vol.expanding().rank(pct=True) * 100

# 수정됨
vol_pct = rolling_vol.rolling(252, min_periods=1).rank(pct=True) * 100
```

#### signal_vol_regime_adaptive_ma() (L352-354)
```python
# 이전 (버그)
ref_vol = rolling_vol.expanding().median()
vol_pct = rolling_vol.expanding().rank(pct=True) * 100

# 수정됨
ref_vol = rolling_vol.rolling(252, min_periods=1).median()  # 1-year rolling
vol_pct = rolling_vol.rolling(252, min_periods=1).rank(pct=True) * 100
```

#### signal_macro_regime_dual_ma() (L418 + L444)
```python
# 이전 (버그)
vol_pct = rolling_vol.expanding().rank(pct=True) * 100
cs_pct = cs.expanding().rank(pct=True).values * 100

# 수정됨
vol_pct = rolling_vol.rolling(252, min_periods=1).rank(pct=True) * 100
cs_pct = cs.rolling(252, min_periods=1).rank(pct=True).values * 100
```

### 2. optimize_regime_grid_v2.py

#### precompute_vol_regimes() (L85)
```python
# 이전 (버그)
vol_pct = rolling_vol.expanding().rank(pct=True) * 100

# 수정됨
vol_pct = rolling_vol.rolling(252, min_periods=1).rank(pct=True) * 100
```

---

## 변경의 의미

### 문제점 (이전)
- **Expanding percentile**: 1987년부터 현재까지의 모든 변동성 데이터를 사용
- **영향**: 2008년 금융위기의 80% 변동성이 2010년 이후의 모든 신호를 왜곡
- **결과**: 2010-2020년의 "정상" 변동성(15%)도 상대적으로 "저변동성"으로 분류

### 해결책 (수정됨)
- **Rolling percentile (252-day window = 1년)**: 최근 1년의 변동성만 비교
- **이점**:
  1. 과거 극단값이 현재 신호를 왜곡하지 않음
  2. COVID/2008/2022 같은 각 위기가 로컬 극값으로 평가됨
  3. 변동성 체계가 더 **역동적**으로 변함
  4. **min_periods=1**: 초기 데이터 부족 기간에도 계산 가능

---

## 영향받는 분석

### 🔴 재실행 필요
- Part 12: TQQQ-calibrated NDX grid search
  - optimize_regime_grid_v2.py 사용 → 신호 변경
- Part 5: macro regime layer 사용하는 경우
- test_macro_regime.py, analyze_crises.py 등

### ✅ 영향 없음
- Part 1-11: regime-switching 신호 미사용 또는 dual_ma만 사용

---

## 다음 단계

1. ✅ 코드 수정 완료
2. ⬜ 테스트: Part 12 재실행
3. ⬜ 비교: 이전 vs 신규 신호 동작 분석
4. ⬜ 다른 버그 수정 (Walk-forward 테스트 추가, RF 일관성 등)

---

## 기술 상세

### Rolling percentile의 수학
```
rolling_vol.rolling(252, min_periods=1).rank(pct=True)

예시 (5일 윈도우):
Date  Vol   Rank(pct)
----  ---   ---------
1     10%   20%   (5개 중 1위)
2     15%   40%   (5개 중 2위)
3     12%   60%   (5개 중 3위)
4     20%   80%   (5개 중 4위)
5     18%   100%  (5개 중 5위)
6     22%   100%  (최근 5개: 15,12,20,18,22 → 22% = 5위)
7     11%   20%   (최근 5개: 12,20,18,22,11 → 11% = 1위)

→ 과거의 80% vol이 미래에 영향 없음
```

---

## 검증 스크립트

```bash
# Part 12 재실행
python run_part12_only.py

# 신호 비교 (신규)
python test_vol_percentile_fix.py  # (다음 생성)
```

---

**수정 완료**: 2026-02-27
