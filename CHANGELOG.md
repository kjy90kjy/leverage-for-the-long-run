# 변경 이력 (CHANGELOG)

## [v0.4.0] — 2026-02-26
### 추가
- **Phase 3 Regime Grid Search**: 레짐 전환 6-파라미터 다중 해상도 그리드 서치 (`optimize_regime_grid.py`)
  - Coarse grid (step 10, ~720k 조합) → Plateau 식별 (top 1% → neighbour-average → greedy 선택) → Fine grid (step 1, ±5)
  - MA dict 사전계산, vol-regime boolean 캐시, numpy-only `fast_eval()` — Optuna 대비 대규모 파라미터 공간 탐색
  - 5개 플래토 발견: P4 peak (47,124,50,197) Sortino=1.094 최고 성능, P5 Optuna MDD_Entry=-39.2% 최저 낙폭
  - 출력: `regime_grid_coarse_top.csv`, `regime_grid_final.csv`, `regime_grid_plateaus.png`, `regime_grid_comparison.png`
- `CLAUDE.md`: Phase 3 스크립트 문서 추가

### 수정
- `validate_eulb.py`: 하드코딩 경로를 `os.path.dirname(__file__)` 기반 상대 경로로 수정

### 정리
- `Source/` 디렉터리를 `references/`로 이동 (PDF 원본, 프로토타입 코드)
- `gen_diag.py`, `ssrn-2741701.pdf` → `references/`로 이동

## [v0.3.0] — 2026-02-25
### 추가
- **3지수 캘리브레이션 그리드 서치**: S&P 500 TR, IXIC, NDX 3개 지수에 대해 TQQQ 캘리브레이션 파라미터(`ER=3.5%`, `lag=1`, `comm=0.2%`) 적용한 그리드 서치
- `run_grid_all_indices.py`: 3지수 캘리브레이션 그리드 서치 러너 스크립트
- 4-클러스터 비교 차트 (`cluster_ABCD_comparison.png`)

### 수정
- **Warm-up 기간 수정**: MA warm-up 기간 동안 cash-start 로직 적용 (기존: 투자 상태 시작 → 수정: 현금 상태 시작)
- NDX 캘리브레이션 그리드 `slow_range` step을 3→1로 세분화

## [v0.2.0] — 2026-02-24
### 추가
- `calibrate_tqqq.py`: TQQQ 비용 캘리브레이션 스크립트 (합성 3x QQQ vs 실제 TQQQ 비교)
- **Part 12**: TQQQ 캘리브레이션 기반 NDX 그리드 서치 (`ER=3.5%`, `lag=1`, `comm=0.2%`)
- `run_part12_only.py`, `run_parts7to12.py`: 부분 실행 러너 스크립트
- **MDD_Entry 메트릭**: 진입 시점 기준 최대 낙폭 (equity curve peak이 아닌 running max of entry-point equity)
- **Sortino 비율**: TDD 기반 (Sortino & van der Meer 1991)
- 6-패널 복합 히트맵 (`plot_composite_heatmap()`)
- 클러스터 A/B 비교 차트
- `CLAUDE.md`: 프로젝트 가이드 문서

### 수정
- **Sharpe 비율 계산 수정**: 산술 평균 기반 (Sharpe 1994)으로 통일
- `references/backtesting_prototype.py`: eulb 원본 코드 참고용 보존

## [v0.1.0] — 2026-02-24
### 최초 릴리스
- **논문 재현**: Michael Gayed "Leverage for the Long Run" (2016) Table 8 재현
  - S&P 500 Total Return (Shiller 배당 데이터 합성), 1928–2020
  - Ken French 무위험이자율 사용
- **NASDAQ 분석 확장**: IXIC (1971~), NDX (1985~)
- **Dual MA 그리드 서치**: S&P 500, IXIC, NDX 3개 지수 × (slow, fast) MA 조합 × 레버리지 배율
- **eulb 검증**: eulb의 TQQQ 이동평균선 매매법 (포스트 1, 5) 독립 검증
- **ETF 비교**: SPY/SSO/UPRO, QQQ/TQQQ 실제 ETF 성과 비교
- `leverage_rotation.py`: 메인 분석 스크립트 (Part 1~11)
- `diag_nasdaq.py`: NASDAQ 데이터 품질 진단
- `validate_eulb.py`: eulb 결과 검증 스크립트
