# Leverage Rotation Strategy (LRS) — 레버리지 ETF 회전 전략 백테스팅

Michael Gayed의 2016 Dow Award 논문 "[Leverage for the Long Run](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2741701)"을 재현하고, NASDAQ/NDX에 확장 적용한 프로젝트.

eulb(에펨코리아)의 TQQQ 이동평균선 매매법 시리즈 결과를 독립 검증하고, TQQQ 실제 비용을 캘리브레이션하여 현실적인 레버리지 전략을 탐색하는 것이 주요 목적.

## 핵심 발견

### Look-ahead Bias 수정
- **`signal_lag=0`은 `backtesting.py`의 `trade_on_close=True`와 같지 않다.**
- `backtesting.py`의 `trade_on_close=True`: 종가에 시그널 → 종가에 매수 → **다음 날부터 수익** (= `signal_lag=1`)
- 단순 벡터화 백테스트의 `signal_lag=0`: 종가로 시그널 계산 + **같은 날 수익을 소급 적용** → look-ahead bias
- 진입일(상승분 소급 적용)과 청산일(하락분 회피) 양쪽 모두 유리하게 작동 → CAGR 크게 과대 추정
- **`fast=1`(가격 자체)은 극심한 look-ahead를 야기**하여 grid search 기본값에서 제외 (`fast_range` 기본 2~50)

### TQQQ 비용 캘리브레이션
- 합성 3x QQQ 모델과 실제 TQQQ를 비교하여 최적 `expense_ratio` = 3.5% 도출
- 금리 환경별(ZIRP, 금리 인상기, COVID, 고금리) 분석 포함

### eulb 결과 재현 (Part 10, 11)
`signal_lag=1` + `commission=0.2%`로 보정 후, eulb 5편 추천 **(3, 161)이 최적 영역에 위치함을 확인**.

## 폴더 구조

```
├── leverage_rotation.py        메인 분석 스크립트 (Part 1~12)
├── calibrate_tqqq.py           TQQQ 비용 캘리브레이션
├── diag_nasdaq.py              NASDAQ 데이터 품질 진단
├── validate_eulb.py            eulb 결과 검증
├── run_part12_only.py          Part 12 단독 실행 (~15분)
├── run_parts7to12.py           Part 7~12 실행 (~45분)
├── run_grid_all_indices.py     3지수 캘리브레이션 그리드 서치
├── output/                     생성된 히트맵·차트·CSV
├── references/                 참고 자료
│   ├── ssrn-2741701.pdf        Gayed 원논문
│   ├── backtesting_prototype.py  eulb 원본 코드 (참고용)
│   └── *.pdf                   eulb 원문 포스트 (4개)
├── CLAUDE.md                   AI 어시스턴트 가이드
├── CHANGELOG.md                변경 이력
└── .gitignore
```

## 분석 파트 구성

| Part | 내용 | 데이터 | signal_lag |
|------|------|--------|------------|
| 1 | 논문 Table 8 재현 (1928–2020) | ^GSPC Total Return (Shiller) | 0 |
| 1.5 | NASDAQ Composite 장기 (1971–현재) | ^IXIC | 0 |
| 2 | 현대 분석 (1990–현재) | ^GSPC | 0 |
| 3 | 확장 MA (50, 100, 200) | ^GSPC | 0 |
| 4 | Nasdaq-100 | ^NDX | 0 |
| 5 | ETF 비교 (SPY/SSO/UPRO/QQQ/TQQQ) | 실제 ETF | 0 |
| 6 | Dual MA 시그널 예시 | ^GSPC | 0 |
| 7 | Dual MA Grid Search — S&P 500 (1928–2020, TR) | ^GSPC | 0 |
| 8 | Dual MA Grid Search — NASDAQ Composite (1971–2025) | ^IXIC | 0 |
| 9 | Dual MA Grid Search — Nasdaq-100 (1985–2025) | ^NDX | 0 |
| 10 | eulb 1편 재현 — ^NDX (1985–2025) | ^NDX | 1 |
| 11 | eulb 5편 재현 — ^NDX (2006–2024) | ^NDX | 1 |
| 12 | TQQQ 캘리브레이션 NDX 그리드 (1985–2025) | ^NDX | 1 |

## 실행 방법

### 사전 준비
```bash
pip install numpy pandas matplotlib yfinance openpyxl scipy
```

### 실행
```bash
# 1. TQQQ 비용 캘리브레이션 (먼저 실행, ~5분)
python calibrate_tqqq.py

# 2. 전체 분석 (Part 1~12, ~25-30분)
python leverage_rotation.py

# 3. (선택) 부분 실행
python run_part12_only.py       # Part 12만 (~15분)
python run_parts7to12.py        # Part 7~12 (~45분)
python run_grid_all_indices.py  # 3지수 캘리브레이션 그리드

# 4. (선택) 보조 스크립트
python diag_nasdaq.py           # NASDAQ 데이터 품질 진단
python validate_eulb.py         # eulb 결과 검증
```

결과는 `output/` 디렉토리에 PNG 차트와 CSV 파일로 저장됩니다.

## 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `signal_lag` | 시그널 지연 (0=동일일, 1=익일) | Part별 상이 |
| `expense_ratio` | 연간 비용 비율 | 0.01 (1%) |
| `commission` | 매매당 수수료 (0.002 = 0.2%) | 0.0 (Part 10/11/12: 0.002) |
| `fast_range` | 단기 MA 탐색 범위 | range(2, 51) |
| `slow_range` | 장기 MA 탐색 범위 | range(50, 351, 3) |
| `tbill_rate` | T-Bill 금리: 스칼라 또는 `"ken_french"` (일별) | Part별 상이 |

## Importable API

```python
from leverage_rotation import (
    download, signal_ma, signal_dual_ma, run_lrs, run_buy_and_hold,
    calc_metrics, signal_trades_per_year, download_ken_french_rf,
    run_dual_ma_analysis,
    run_eulb1_comparison, run_eulb5_spotcheck, run_part12_comparison,
)
```

## 외부 데이터 소스

- **Yahoo Finance** (yfinance): ^GSPC, ^IXIC, ^NDX, SPY, SSO, UPRO, QQQ, TQQQ
- **Yale Robert Shiller**: S&P 500 과거 배당수익률 (총수익 합성용)
- **Ken French Data Library**: 일별 무위험이자율 (1926–현재)

## 참고 자료

- Gayed, M. (2016). *Leverage for the Long Run — A Systematic Approach to Managing Risk and Magnifying Returns in Stocks*. SSRN 2741701.
- eulb, "TQQQ 이동평균선 매매법 최적화" 시리즈 1~5편. 에펨코리아.
