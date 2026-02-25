# Leverage for the Long Run — Leveraged ETF Rotation Strategy

Michael Gayed의 2016 Dow Award 논문 "[Leverage for the Long Run](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2741701)"을 재현하고, NASDAQ/NDX에 확장 적용한 프로젝트.

eulb(에펨코리아)의 TQQQ 이동평균선 매매법 시리즈 결과를 독립 검증하는 것이 주요 목적.

## 핵심 발견

### Look-ahead Bias 수정
- **`signal_lag=0`은 `backtesting.py`의 `trade_on_close=True`와 같지 않다.**
- `backtesting.py`의 `trade_on_close=True`: 종가에 시그널 → 종가에 매수 → **다음 날부터 수익** (= `signal_lag=1`)
- 단순 벡터화 백테스트의 `signal_lag=0`: 종가로 시그널 계산 + **같은 날 수익을 소급 적용** → look-ahead bias
- 진입일(상승분 소급 적용)과 청산일(하락분 회피) 양쪽 모두 유리하게 작동 → CAGR 크게 과대 추정
- **`fast=1`(가격 자체)은 극심한 look-ahead를 야기**하여 grid search 기본값에서 제외 (`fast_range` 기본 2~50)

### eulb 결과 재현 (Part 10, 11)
`signal_lag=1` + `commission=0.2%`로 보정 후, eulb 5편 추천 **(3, 161)이 최적 영역에 위치함을 확인**.

| 설정 | (3, 161) CAGR | Best Sharpe 조합 | (3, 161) Sharpe 순위 |
|------|--------------|-----------------|---------------------|
| Part 11 (2006-2024) | 34.47% | (182, 3) = 0.769 | Top 4 (0.760) |

## 구성

```
leverage_rotation.py    메인 분석 스크립트 (Part 1~11)
output/                 생성된 히트맵·차트 (71개 PNG)
Source/                 eulb 원문 PDF
ssrn-2741701.pdf       Gayed 원논문
```

## 분석 파트 구성

| Part | 내용 | 데이터 |
|------|------|--------|
| 1 | 논문 Table 8 재현 (1928-2020) | ^GSPC Total Return (Shiller) |
| 1.5 | NASDAQ Composite 장기 (1971-현재) | ^IXIC |
| 2 | 현대 분석 (1990-현재) | ^GSPC |
| 3 | 확장 MA (50, 100, 200) | ^GSPC |
| 4 | Nasdaq-100 | ^NDX |
| 5 | ETF 비교 (SPY/SSO/UPRO/QQQ/TQQQ) | 실제 ETF |
| 6 | Dual MA 시그널 예시 | ^GSPC |
| 7 | **Dual MA Grid Search** — S&P 500 (1928-2020, TR) | ^GSPC |
| 8 | **Dual MA Grid Search** — NASDAQ Composite (1971-2025) | ^IXIC |
| 9 | **Dual MA Grid Search** — Nasdaq-100 (1985-2025) | ^NDX |
| 10 | **eulb 1편 재현** — ^NDX (1985-2025), lag=1, comm=0.2% | ^NDX |
| 11 | **eulb 5편 재현** — ^NDX (2006-2024), lag=1, comm=0.2% | ^NDX |

## 실행

```bash
pip install numpy pandas matplotlib yfinance openpyxl
python leverage_rotation.py
```

전체 실행에 약 20~30분 소요 (grid search 포함). 결과는 `output/` 디렉토리에 저장.

## 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `signal_lag` | 시그널 지연 (0=동일일, 1=익일) | Part별 상이 |
| `expense_ratio` | 연간 비용 비율 | 0.01 (1%) |
| `commission` | 매매당 수수료 | 0.0 (Part 10/11: 0.002) |
| `fast_range` | 단기 MA 탐색 범위 | range(2, 51) |
| `slow_range` | 장기 MA 탐색 범위 | range(50, 351, 3) |
| `tbill_rate` | T-Bill 금리 (스칼라 or Ken French 일별) | Part별 상이 |

## 참고 자료
- Gayed, M. (2016). *Leverage for the Long Run — A Systematic Approach to Managing Risk and Magnifying Returns in Stocks*. SSRN 2741701.
- eulb, "TQQQ 이동평균선 매매법 최적화" 시리즈 1~5편. 에펨코리아.
