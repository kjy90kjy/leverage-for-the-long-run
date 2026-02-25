# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Leverage Rotation Strategy (LRS) backtesting framework that replicates Michael Gayed's 2016 paper "Leverage for the Long Run" and extends analysis to NASDAQ indices. Also independently validates Korean financial blog "eulb"'s TQQQ moving-average trading strategies.

## Running Scripts

```bash
pip install numpy pandas matplotlib yfinance openpyxl scipy

python calibrate_tqqq.py       # TQQQ cost calibration (~5 min) — run first
python leverage_rotation.py   # Full analysis (~25-30 min, includes grid searches + Part 12)
python diag_nasdaq.py          # NASDAQ data quality diagnostics
python validate_eulb.py        # Validate against eulb's published results
```

All scripts are standalone — no build system or test framework. Output PNGs go to `output/`.

All scripts use a Windows UTF-8 boilerplate at the top (`sys.stdout` wrapping + `warnings.filterwarnings("ignore")`) — preserve this pattern when creating new scripts. Matplotlib uses `Agg` backend (headless, file-only output — no interactive windows).

## Architecture

### leverage_rotation.py (main script, ~1410 lines)

Organized as a pipeline with these layers:

1. **Data layer** (lines ~34-144): `download()` fetches from yfinance; `_add_shiller_dividends()` synthesizes S&P 500 total returns using Yale Shiller dividend data; `download_ken_french_rf()` gets daily risk-free rates
2. **Signal layer** (lines ~151-175): `signal_ma()` (price vs SMA), `signal_dual_ma()` (golden cross / fast vs slow SMA), `signal_rsi()`
3. **Strategy engine** (lines ~181-229): `run_lrs()` is the core backtest loop — applies leverage when signal=1, T-Bill returns when signal=0, with configurable signal lag and per-trade commission. `run_buy_and_hold()` for benchmarks
4. **Metrics** (lines ~235-315): `calc_metrics()` computes CAGR, Sharpe (arithmetic mean, Sharpe 1994), Sortino (TDD per Sortino & van der Meer 1991), MDD, Beta, Alpha. `_max_entry_drawdown()` computes MDD from running max of entry-point equity (not equity curve peak). `_max_recovery_days()` computes longest peak-to-recovery span.
5. **Visualization** (lines ~318-400): cumulative returns, drawdowns, volatility bars, rolling excess
6. **Dual MA grid search** (lines ~658-1050): `run_dual_ma_grid()` tests all (slow, fast) MA combos × leverage levels; `plot_heatmap()` generates 2D metric heatmaps; `plot_composite_heatmap()` generates 6-panel composite; `run_dual_ma_analysis()` is the high-level orchestrator that runs a full grid search, generates all heatmaps, and saves full results to CSV

The main block (lines ~477-1413) runs 12 analysis parts sequentially, each with a config dict specifying ticker, date range, signal parameters, and lag settings.

### Analysis Parts

| Part | Description | Data |
|------|-------------|------|
| 1 | Paper Table 8 replication (1928-2020) | ^GSPC Total Return (Shiller) |
| 1.5 | NASDAQ Composite long-run (1971-present) | ^IXIC |
| 2 | Modern analysis (1990-present) | ^GSPC |
| 3 | Extended MA (50, 100, 200) | ^GSPC |
| 4 | Nasdaq-100 | ^NDX |
| 5 | ETF comparison (SPY/SSO/UPRO/QQQ/TQQQ) | Real ETFs |
| 6 | Dual MA signal example | ^GSPC |
| 7 | Dual MA Grid Search — S&P 500 (1928-2020, TR) | ^GSPC |
| 8 | Dual MA Grid Search — NASDAQ Composite (1971-2025) | ^IXIC |
| 9 | Dual MA Grid Search — Nasdaq-100 (1985-2025) | ^NDX |
| 10 | eulb post 1 replication — ^NDX, lag=1, comm=0.2% | ^NDX |
| 11 | eulb post 5 replication — ^NDX (2006-2024), lag=1, comm=0.2% | ^NDX |
| 12 | TQQQ-calibrated NDX grid search (1985-2025), lag=1, comm=0.2% | ^NDX |

### Key config presets

- **PAPER_CONFIG**: ^GSPC total return, 1928-2020, Ken French RF, lag=0 (paper replication)
- **NASDAQ_LONGRUN_CONFIG**: ^IXIC price-only, 1971-present, lag=0
- **DEFAULT_CONFIG**: ^GSPC, 1990-present, 2% T-Bill, lag=0

### Runner scripts

- `run_part12_only.py`: Runs Part 12 only (TQQQ-calibrated NDX grid, slow step 1). ~15 min.
- `run_parts7to12.py`: Runs Parts 7-12 (all grid searches). ~45 min.
- `run_grid_all_indices.py`: Runs 3x calibrated grid search for all three indices (S&P 500 TR, IXIC, NDX) with `CALIBRATED_ER=0.035`, `lag=1`, `comm=0.2%`, `slow_range step=1`.

### Script dependencies

`diag_nasdaq.py`, `validate_eulb.py`, and `calibrate_tqqq.py` all import from `leverage_rotation.py`. `validate_eulb.py` uses `sys.path.insert(0, script_dir)` relative to its own location — no manual path editing needed.

### calibrate_tqqq.py

TQQQ cost calibration: compares synthetic 3x QQQ (our model) vs actual TQQQ to determine the optimal `expense_ratio` parameter. Sweeps fixed ER (0.5%-3.5%), tests time-varying financing cost model (Ken French RF-based), and analyzes performance across interest rate regimes (ZIRP, rate hike, COVID, high rate). Outputs recommended ER value and three charts. The calibrated ER is used by Part 12 in `leverage_rotation.py`.

### diag_nasdaq.py

Data quality checks: NaN gaps, suspicious jumps, return characteristics, MA signal behavior, dot-com bubble analysis.

### validate_eulb.py

Runs identical backtests with eulb's parameters comparing lag=0 vs lag=1, validates relative ranking of MA combos like (3,220), (1,200), (4,80).

## Critical Domain Concept: Look-Ahead Bias

The `signal_lag` parameter is the most important correctness control:
- **`signal_lag=0`**: Signal computed at close → applied to same-day return → **look-ahead bias** (theoretical only)
- **`signal_lag=1`**: Signal computed at close → applied to next-day return → **realistic execution**
- **`fast=1` MA is excluded** from grid searches because price vs price creates extreme look-ahead bias

This differs from `backtesting.py` where `trade_on_close=True` still uses next-day returns internally.

## External Data Sources

- **Yahoo Finance** (yfinance): ^GSPC, ^IXIC, ^NDX, SPY, SSO, UPRO, QQQ, TQQQ
- **Yale Robert Shiller**: S&P 500 historical dividend yields for total return synthesis
- **Ken French Data Library**: Daily risk-free rates (1926-present)

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `signal_lag` | Signal delay (0=same-day, 1=next-day) | Varies by Part |
| `expense_ratio` | Annual cost ratio | 0.01 (1%) |
| `commission` | Per-trade cost as fraction (0.002 = 0.2%) | 0.0 (Part 10/11/12: 0.002) |
| `fast_range` | Short MA search range | range(2, 51) |
| `slow_range` | Long MA search range | range(50, 351, 3) |
| `tbill_rate` | T-Bill rate: scalar, or `"ken_french"` for daily series | Varies by Part |

## Importable API

```python
from leverage_rotation import (
    download, signal_ma, signal_dual_ma, run_lrs, run_buy_and_hold,
    calc_metrics, signal_trades_per_year, download_ken_french_rf,
    run_dual_ma_analysis,
    run_eulb1_comparison, run_eulb5_spotcheck, run_part12_comparison,
)
```
