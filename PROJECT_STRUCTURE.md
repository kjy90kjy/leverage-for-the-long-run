# 프로젝트 구조

## 📁 디렉토리 가이드

### Root (메인 스크립트)
- **leverage_rotation.py** - 핵심 분석 엔진 (Part 1-12)
- **lrs_standalone.py** - 프로덕션 4-전략 비교
- **calibrate_tqqq.py** - TQQQ 비용 캘리브레이션

### Core Scripts (1단계 - 필수 실행)
**위치**: `/core/`
- calibrate_tqqq.py (먼저 실행, ER=3.5% 산출)
- leverage_rotation.py (전체 분석)

### Analysis Scripts (2단계 - 검증 & 분석)
**위치**: `/analysis/`
- validate_eulb.py - eulb 블로그 결과 검증
- analyze_crises.py - 9개 금융 위기 분석
- diag_nasdaq.py - NASDAQ 데이터 품질 진단

### Optimization Scripts (3단계 - 파라미터 탐색)
**위치**: `/optimization/`
- optimize_all_full.py - Phase 2: 전기간 최적화
- optimize_regime_grid_v2.py - Phase 3: 고밀도 그리드 서치
- optimize_common.py - 공유 최적화 라이브러리

### Daily Signals (4단계 - 실거래)
**위치**: `/signals/`
- daily_signal_generator.py - 일일 신호 생성 (CSV + HTML)
- daily_signal_telegram.py - Telegram 알림 + 예측

### Tests (검증)
**위치**: `/tests/`
- test_*.py - 각종 검증 테스트
- optimize_kimjje_*.py - 김째 전략 파라미터 최적화
- overheat_plateau_optuna.py - 과열 고원지대 탐색

### Convenience Scripts (편의)
**위치**: `/runners/`
- run_part12_only.py - Part 12 단독 실행
- run_parts7to12.py - Parts 7-12 재실행
- run_grid_all_indices.py - 3-지수 캘리브레이션 그리드

### Archive (폐기 & 참고용)
**위치**: `/archive/`
- Phase 1/2/3 v1 최적화 (현재 사용 안함)
- 분석 보조 도구들

---

## 📊 Output 폴더 구조

### Grid Search Results
- GSPC_TR_3x_grid_results.csv (S&P 500 3x 그리드)
- IXIC_3x_grid_results.csv (NASDAQ Composite 3x)
- NDX_calibrated_grid_results.csv (Nasdaq-100 3x, Part 12)

### Analysis Results
- crisis_analysis_detail.csv (9개 위기 분석)
- crisis_mdd_breakdown.csv (위기별 MDD 분해)
- kimjje_*.csv (김째 전략 분석)
- overheat_plateau_*.csv (과열 파라미터 Optuna)

### Calibration
- Part7_lag_correction_table.csv (Lag 보정)
- Part8_lag_correction_table.csv (Lag 보정)
- daily_signals.csv (오늘의 신호)

### Archive (보관)
- `archive/output/` - 이전 실행 결과

---

## 📚 Docs 폴더 구조

### Core Documentation
- CLAUDE.md - 프로젝트 가이드 (필수 읽기)
- PROJECT_STRUCTURE.md (본 파일)

### Analysis Reports
- **Validation**
  - VALIDATION_REPORT.md - Walk-forward 검증
  - LAG_CORRECTION_FINAL_REPORT.md - 신호 lag 보정
  
- **Strategy**
  - REGIME_SWITCHING_STRATEGY_REPORT.md - 최종 전략
  - OVERHEAT_PLATEAU_*.md - 김째 과열 파라미터 분석
  
- **Implementation**
  - DAILY_SIGNAL_SETUP.md - 일일 신호 자동화
  - TELEGRAM_SCRIPTABLE_SETUP.md - iOS 알림 설정

### Issue Tracking
- COMPLETION_SUMMARY.md - 완료 사항
- CRITICAL_REVIEW.md - 발견된 문제 및 수정 사항
- FIX_SUMMARY.md - 수정 이력
- PRIORITY4_STATUS.md - 진행 상황

### Archive (이전 버전)
- `archive/docs/` - 구버전 분석 보고서

---

## 🔄 실행 순서

### 신규 사용자
```bash
# 1. 데이터 검증
python diag_nasdaq.py

# 2. TQQQ 비용 캘리브레이션
python calibrate_tqqq.py          # 5분

# 3. 전체 분석 (필수)
python leverage_rotation.py        # 25-30분

# 4. 검증 (선택)
python validate_eulb.py            # 5분
python analyze_crises.py           # 10분
```

### 재실행 (파라미터 변경 후)
```bash
# Part 12만 빠르게 재실행
python run_part12_only.py         # 15분

# Parts 7-12 재실행
python run_parts7to12.py          # 45분
```

### 최적화 (고급)
```bash
# Phase 3 고밀도 그리드 (12분)
python optimize_regime_grid_v2.py

# 김째 전략 파라미터 최적화 (30분)
python tests/optimize_kimjje_overheat.py
```

### 실거래 자동화
```bash
# 일일 신호 생성
python daily_signal_generator.py

# Telegram 알림 포함
python daily_signal_telegram.py
```

---

## 🎯 주요 결과

### Part 12 (최종 기준)
- **자산**: Nasdaq-100 (^NDX)
- **레버리지**: 3x (합성)
- **기간**: 1985-2025 (또는 2011-2025)
- **전략**: Regime-Switching Dual MA
- **파라미터**: fast_low=12, slow_low=237, fast_high=6, slow_high=229
- **결과**: Sortino ~2.0, CAGR ~27%

### 김째 S2 전략 (프로덕션)
- **자산**: TQQQ + QQQ + SPY
- **기간**: 2011-2025
- **전략**: 과열 4단계 감량 (S1~S4)
- **원본 파라미터**: 139/132 → 146/138 → 149/140 → 151/118
- **검증**: Sortino ~2.1, 강건한 고원지대

---

## ✅ 체크리스트

- [x] 데이터 품질 검증 (NASDAQ data ok)
- [x] Lag 불일치 정량화 (Part 7-9 vs Part 12)
- [x] 과적합 검증 (Walk-forward OOS > IS)
- [x] Expanding percentile 버그 수정
- [x] Part 12 최종 기준 확립
- [x] 김째 전략 독립 검증
- [x] 과열 파라미터 Optuna 최적화
- [ ] 실거래 자동화 배포

