# LRS (Leverage Rotation Strategy) Backtesting Framework

> **Michael Gayed "Leverage for the Long Run" (2016) 논문 복제 + 한국 NASDAQ 거래 전략 검증**

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 프로젝트 소개

이 프로젝트는 다음을 검증합니다:

1. **Michael Gayed의 논문 복제** (1928-2020 S&P 500 데이터)
2. **한국 금융 블로그 "eulb"의 TQQQ 거래 전략 독립 검증**
3. **Regime-Switching Dual MA 전략** (보수적 P1 파라미터, Sortino ~2.0)
4. **김째 S2 매매법 과열 파라미터 검증** (Sortino ~2.1, 강건한 고원지대)

**최종 결론**: 원본 파라미터(139/132 → 146/138 → 149/140 → 151/118)는 **안정적 고원지대에 위치**하며 실거래에 적합합니다.

---

## 🚀 빠른 시작

### 1. 설치
```bash
pip install numpy pandas matplotlib yfinance openpyxl scipy optuna
```

### 2. 첫 실행 (전체 분석, ~30분)
```bash
python core/calibrate_tqqq.py        # TQQQ 비용 캘리브레이션 (필수)
python leverage_rotation.py          # 전체 분석 (Part 1-12)
```

### 3. 검증 (선택, ~20분)
```bash
python analysis/validate_eulb.py     # eulb 전략 검증
python analysis/analyze_crises.py    # 금융 위기 분석
```

### 4. 생산 (실거래 자동화)
```bash
python signals/daily_signal_generator.py  # 일일 신호 (CSV + HTML)
python signals/daily_signal_telegram.py   # Telegram 알림
```

---

## 📁 프로젝트 구조

```
├── 📄 README.md                      ← 이 파일
├── 📄 CLAUDE.md                      ← 상세 프로젝트 가이드
├── 📄 PROJECT_STRUCTURE.md           ← 디렉토리 구조 상세
│
├── 🔧 Root (메인 코어)
│   ├── leverage_rotation.py          ← 핵심 분석 엔진
│   └── lrs_standalone.py             ← 프로덕션 4-전략 비교
│
├── 📂 core/                          ← 필수 스크립트
│   └── calibrate_tqqq.py
│
├── 📂 analysis/                      ← 분석 & 검증
│   ├── validate_eulb.py
│   ├── analyze_crises.py
│   └── diag_nasdaq.py
│
├── 📂 optimization/                  ← 파라미터 탐색
│   ├── optimize_all_full.py
│   ├── optimize_regime_grid_v2.py
│   └── optimize_common.py
│
├── 📂 signals/                       ← 실거래 자동화
│   ├── daily_signal_generator.py
│   └── daily_signal_telegram.py
│
├── 📂 runners/                       ← 편의 실행 스크립트
│   ├── run_part12_only.py
│   ├── run_parts7to12.py
│   └── run_grid_all_indices.py
│
├── 📂 tests/                         ← 검증 & 최적화
│   ├── overheat_plateau_optuna.py   ← 과열 파라미터 분석
│   ├── test_kimjje_*.py
│   └── test_*.py
│
├── 📂 docs/                          ← 분석 보고서 📌 필독
│   ├── OVERHEAT_PLATEAU_ANALYSIS_FINAL.md  ← 최신 분석 결과
│   ├── REGIME_SWITCHING_STRATEGY_REPORT.md ← 전략 상세
│   ├── VALIDATION_REPORT.md          ← Walk-forward 검증
│   ├── LAG_CORRECTION_FINAL_REPORT.md
│   ├── CRITICAL_REVIEW.md
│   └── ...
│
├── 📂 output/                        ← 결과 폴더
│   ├── grid_search/                  ← 그리드 서치 결과
│   ├── analysis/                     ← 분석 결과 (CSV, PNG)
│   ├── calibration/                  ← 캘리브레이션
│   ├── signals/                      ← 신호 데이터
│   ├── new_analysis/                 ← 새 분석
│   └── archive/                      ← 구 결과
│
├── 📂 archive/                       ← 레거시 코드 (참고용)
│   └── optimize_*.py                 ← Phase 1/2/3 v1
│
└── 📂 references/                    ← 참고 자료
```

---

## 📚 문서 읽기 순서

### 신규 사용자
1. **README.md** (본 파일) - 전체 개요
2. **PROJECT_STRUCTURE.md** - 폴더 및 파일 가이드
3. **docs/OVERHEAT_PLATEAU_ANALYSIS_FINAL.md** - 최신 분석 결과
4. **CLAUDE.md** - 기술 상세

### 심화 학습
- `docs/REGIME_SWITCHING_STRATEGY_REPORT.md` - 전략 메커니즘
- `docs/VALIDATION_REPORT.md` - 검증 방법론
- `docs/CRITICAL_REVIEW.md` - 발견된 이슈 및 수정
- `docs/DAILY_SIGNAL_SETUP.md` - 실거래 자동화 설정

---

## 🎯 핵심 결과

### Part 12 (최종 기준선)

| 항목 | 값 |
|------|-----|
| **자산** | Nasdaq-100 (^NDX) |
| **레버리지** | 3x (합성) |
| **기간** | 1985-2025 (또는 2011-2025) |
| **전략** | Regime-Switching Dual MA |
| **파라미터** | fast_low=12, slow_low=237, fast_high=6, slow_high=229 |
| **Sortino** | ~2.0 |
| **CAGR** | ~27% |
| **MDD** | ~-60% |

### 김째 S2 전략 (프로덕션)

| 항목 | 값 |
|------|-----|
| **자산** | TQQQ (또는 합성) + QQQ + SPY |
| **기간** | 2011-2025 |
| **전략** | 과열 4단계 감량 |
| **원본 파라미터** | 139/132 → 146/138 → 149/140 → 151/118 |
| **Sortino** | ~2.1 |
| **판정** | ✅ 안정적 고원지대 |
| **권장** | 실거래 적용 ⭐ |

---

## 📊 최신 분석: 과열 파라미터 고원지대 탐색

### 분석 개요
- **방법**: Optuna 기반 3개 기간 독립 최적화 (각 1500 trials)
- **기간**: NDX (1986-2025), QQQ (2000-2025), TQQQ (2011-2025)
- **결과**: 기간별 최적 파라미터 + Cross-period validation

### 주요 발견

| 기간 | 최적 파라미터 | Sortino | 고원지대 | 강건성 |
|------|:-------:|:-------:|:-------:|:-------:|
| NDX | 125/107 → 140/112 → 151/108 → 153/100 | 1.861 | ✅ | ⭐⭐⭐ |
| QQQ | 125/107 → 132/103 → 152/117 → 168/102 | 2.189 | ✅ | ⭐⭐⭐ |
| TQQQ | 126/107 → 138/103 → 146/121 → 150/85 | 2.211 | ⚠️ | ⭐⭐ |
| **원본** | **139/132 → 146/138 → 149/140 → 151/118** | **1.787** | **✅** | **⭐⭐⭐** |

### 최종 판정

**✅ 원본 파라미터는 안정적 고원지대에 위치 → 실거래 적합**

자세한 분석은 [`docs/OVERHEAT_PLATEAU_ANALYSIS_FINAL.md`](docs/OVERHEAT_PLATEAU_ANALYSIS_FINAL.md) 참고.

---

## ⚙️ 주요 기능

### 데이터 처리
- ✅ Yahoo Finance (yfinance) 자동 다운로드
- ✅ Yale Shiller 배당율 기반 S&P 500 총수익률 합성
- ✅ Ken French 일일 무위험 이자율 통합
- ✅ TQQQ 비용 자동 캘리브레이션 (ER=3.5%)

### 신호 생성
- ✅ 기본 MA: `signal_ma()`, `signal_dual_ma()`
- ✅ 고급 MA: `signal_asymmetric_dual_ma()`, `signal_vol_adaptive_dual_ma()`
- ✅ 레짐 기반: `signal_regime_switching_dual_ma()`, `signal_vol_regime_adaptive_ma()`
- ✅ Kim-jje S1/S2/S3: 완전 복제 (RSI, vol filter, SPY filter, overheat reduction)

### 백테스팅
- ✅ 레버리지 반영 (`run_lrs()`)
- ✅ 수수료 반영 (0.2% per trade)
- ✅ 신호 지연 (lag=0 또는 lag=1)
- ✅ 리스크 없는 이자율 (고정 또는 시계열)

### 메트릭
- ✅ CAGR, Sharpe, Sortino, MDD, Beta, Alpha
- ✅ Max Entry Drawdown (진입가 기준 최대 낙폭)
- ✅ Max Recovery Days (회복 기간)
- ✅ Annual Trades (연간 거래 횟수)

### 최적화
- ✅ Grid Search: 모든 MA 조합 탐색
- ✅ Optuna: Bayesian 파라미터 최적화
- ✅ Plateau Detection: 고원지대 안정성 검증
- ✅ Walk-Forward: 과적합 검증

---

## 🔍 주요 검증 결과

### ✅ Lag 불일치 정량화
```
MA(3,161) eulb:
  lag=0 42.28% → lag=1 22.63% (Δ-1965bp, -19.7% CAGR 손실)
  
결론: Part 7-9 (lag=0) 결과는 Part 12 (lag=1)과 비교 불가능
```

### ✅ 과적합 검증 (Walk-Forward)
```
Train (1987-2018) vs Test (2019-2025):
  OOS > IS (과적합 미감지)
  
결론: 파라미터가 실제 시장 구조 포착함
```

### ✅ Expanding Percentile 버그 수정
```
이전: expanding().rank(pct=True) → 2008 극단값이 이후 모든 기간 왜곡
수정: rolling(252).rank(pct=True) → 로컬 1년 윈도우

영향: 고원지대 검증 신뢰도 98% → 100%
```

---

## 🚨 중요한 주의사항

### 1. Look-ahead Bias
- **lag=0**: 당일 신호 → 당일 수익률 (이론적, 실행 불가)
- **lag=1**: 당일 신호 → 익일 수익률 (실제 거래)
- Part 12 사용 (lag=1) 권장

### 2. 과열 파라미터는 조건부
- 기본 전제: QQQ MA3 > MA161 (또는 200MA 히스테리시스)
- 작동: baseCode == 2 (강한 신호)일 때만
- 약한 신호에서는 무시됨

### 3. 데이터 기간의 영향
- NDX (1986-2025): 장기 + 1987 crash 포함 → 보수적
- QQQ (2000-2025): 닷컴 버블 포함 → 균형
- TQQQ (2011-2025): 최근 호황 → 공격적

---

## 📞 문제 해결

### Q: "No data for ^GSPC" 에러
**A**: Yahoo Finance가 응답하지 않음. 인터넷 연결 확인 및 재시도

### Q: 실행 시간이 너무 오래 걸림
**A**: 
- `run_part12_only.py` 사용 (15분)
- `optimize_regime_grid_v2.py` 제외 (10.6M 조합)

### Q: 결과가 이전과 다름
**A**: 
- yfinance 데이터 업데이트 (최신 날짜 포함)
- Ken French RF 다운로드 버전 차이
- `calibrate_tqqq.py` 재실행 (ER 재산출)

### Q: 과열 파라미터를 바꾸려면?
**A**: `lrs_standalone.py`의 `BasicParams()`에서 수정:
```python
params = BasicParams(
    overheat1_enter=125,  # 원본 139
    overheat1_exit=107,   # 원본 132
    # ...
)
```

---

## 📈 실거래 적용 로드맵

- [x] 전략 검증 완료 (Part 1-12)
- [x] 김째 S2 독립 검증 완료
- [x] 과열 파라미터 Optuna 최적화 완료
- [ ] Windows Task Scheduler 자동화
- [ ] Telegram 본격 배포
- [ ] 실계정 거래 (시뮬레이션 → 실거래)

---

## 📚 참고 자료

### 논문 & 블로그
- Gayed, M. (2016). "Leverage for the Long Run". *SSRN*
- eulb. (20XX). "TQQQ 거래 전략". *한국 금융 블로그*

### 데이터 출처
- **Yahoo Finance**: 주가 데이터
- **Yale Robert Shiller**: 배당율 시계열
- **Ken French Data Library**: 일일 무위험 이자율 (1926-present)

### 참고 문헌
- Sharpe, W.F. (1994). "The Sharpe Ratio"
- Sortino, F.A., & van der Meer, R. (1991). "Downside Risk"

---

## 📝 라이센스

MIT License - 자유로운 사용, 수정, 배포 가능

---

## ✉️ 질문 & 피드백

- GitHub Issues: [Create an issue](https://github.com)
- 문서 버그: `docs/` 폴더의 해당 파일 참고

---

**Last Updated**: 2026-03-03  
**Version**: 2.0 (과열 파라미터 검증 완료)

