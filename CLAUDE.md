# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Leverage Rotation Strategy (LRS) backtesting framework that replicates Michael Gayed's 2016 paper "Leverage for the Long Run" and extends analysis to NASDAQ indices. Also independently validates Korean financial blog "eulb"'s TQQQ moving-average trading strategies.

## Running Scripts

```bash
pip install numpy pandas matplotlib yfinance openpyxl scipy optuna

python calibrate_tqqq.py       # TQQQ cost calibration (~5 min) — run first
python leverage_rotation.py   # Full analysis (~25-30 min, includes grid searches + Part 12)
python diag_nasdaq.py          # NASDAQ data quality diagnostics
python validate_eulb.py        # Validate against eulb's published results
python optimize_asymmetric.py  # Phase 1: Asymmetric buy/sell optimization (~7 min)
python optimize_penalized_full.py   # Phase 2 Approach C: penalized asymmetric, full period (~7 min)
python optimize_all_full.py         # Phase 2 Plans 4/5/6: vol-adaptive, regime, vol+regime (~15 min)
python optimize_regime_grid.py      # Phase 3: Multi-resolution grid search for regime-switching (~10-15 min)
python optimize_regime_grid_v2.py   # Phase 3 v2: Dense grid search (step 5/10/5, ~10.6M combos, ~12 min)
```

All scripts are standalone — no build system or test framework. Output PNGs go to `output/`.

All scripts use a Windows UTF-8 boilerplate at the top (`sys.stdout` wrapping + `warnings.filterwarnings("ignore")`) — preserve this pattern when creating new scripts. Matplotlib uses `Agg` backend (headless, file-only output — no interactive windows).

## Architecture

### leverage_rotation.py (main script, ~1410 lines)

Organized as a pipeline with these layers:

1. **Data layer** (lines ~34-144): `download()` fetches from yfinance; `_add_shiller_dividends()` synthesizes S&P 500 total returns using Yale Shiller dividend data; `download_ken_french_rf()` gets daily risk-free rates
2. **Signal layer** (lines ~151-340): `signal_ma()` (price vs SMA), `signal_dual_ma()` (golden cross / fast vs slow SMA), `signal_rsi()`, `signal_asymmetric_dual_ma()` (separate buy/sell MA pairs with hysteresis state machine), `signal_vol_adaptive_dual_ma()` (MA lengths scale with realized vol, cumsum O(n)), `signal_regime_switching_dual_ma()` (different MA pairs for high/low vol regimes, expanding percentile), `signal_vol_regime_adaptive_ma()` (regime-dependent base MA + vol-adaptive scaling)
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

### optimize_asymmetric.py

Phase 1 asymmetric signal optimization: uses Optuna (TPE sampler, 2000 trials) to find optimal `(fast_buy, slow_buy, fast_sell, slow_sell)` parameters for `signal_asymmetric_dual_ma()`. Walk-forward validation with 1985-2014 train / 2015-2025 test split. Compares against symmetric baselines and B&H 3x. Outputs trial CSV, cumulative return charts, and parameter stability box plots. Uses Part 12 conditions (TQQQ-calibrated ER=3.5%, Ken French RF, lag=1, comm=0.2%). Full roadmap: `references/signal_optimization_plan.md`.

### Phase 2 Signal Optimization Scripts

All Phase 2 scripts use full-period optimization (1987-2025) with penalized objective: `Sortino - α × Trades/Year`. Common infrastructure in `optimize_common.py`.

- **`optimize_penalized_full.py`** (Approach C): Asymmetric 4-param + whipsaw penalty on full period. Confirmed asymmetric structure converges to symmetric — no value in buy/sell separation. ~7 min.
- **`optimize_all_full.py`** (Plans 4/5/6): Runs vol-adaptive, regime-switching, and vol+regime optimizations sequentially. Single data download, final comparison table and chart. ~15 min. Key results:
  - Plan 4 Vol-Adaptive (`signal_vol_adaptive_dual_ma`): 4 params, Sortino 1.085
  - Plan 5 Regime-Switching (`signal_regime_switching_dual_ma`): 6 params, Sortino 1.088, MDD_Entry -39.2% (best)
  - Plan 6 Vol+Regime (`signal_vol_regime_adaptive_ma`): 7 params, Sortino 1.103 (best), CAGR 35.3%
- **`optimize_common.py`**: Shared utilities — `apply_warmup()`, `trim_warmup()`, `run_backtest()`, `get_symmetric_baselines()`, `print_comparison_table()`, `plot_cumulative_comparison()`, `stitch_oos_segments()`, `plot_param_stability()`, `download_ndx_and_rf()`. Common constants: `CALIBRATED_ER=0.035, LEVERAGE=3.0, SIGNAL_LAG=1, COMMISSION=0.002`.

### optimize_regime_grid.py (Phase 3)

Multi-resolution grid search for regime-switching 6-parameter plateau exploration. Three phases:
1. **Coarse grid** (step 10): ~720k combos, all MAs precomputed via cumsum, pure numpy backtest (no pandas overhead per trial)
2. **Plateau identification**: top 1% → neighbour-average Sortino → diverse plateau centres via greedy selection
3. **Fine grid** (step 1, ±5): ~15k trials per plateau centre, vol params fixed

Key optimisations: MA dict precomputation, vol-regime boolean array cache, `fast_eval()` with numpy-only Sortino/trades calculation. Full backtest comparison via `run_lrs()` + `calc_metrics()` at end.

Outputs: `regime_grid_coarse_top.csv`, `regime_grid_final.csv`, `regime_grid_plateaus.png`, `regime_grid_comparison.png`.

### optimize_regime_grid_v2.py (Phase 3 v2 — Dense Grid)

Denser grid search to capture Optuna sweet spots that v1's coarse step missed (e.g., fast_high=15, vol_threshold=57.3%, vol_lookback=49). Same 3-phase structure as v1 with denser grid:

| Dimension | v1 | v2 |
|-----------|----|----|
| fast MA | step 10, 6 vals | step 5, 10 vals [2,7,12,17,...,47] |
| slow MA | step 10, 31 vals | step 10, 31 vals (unchanged) |
| vol_lookback | step 20, 6 vals | step 10, 11 vals [20,30,...,120] |
| vol_threshold | step 10, 5 vals | step 5, 10 vals [30,35,...,75] |

Total: ~10.6M combos (~12 min at ~18k trial/s). Plateau identification step sizes adjusted accordingly.

Outputs: `regime_grid_v2_coarse_top.csv`, `regime_grid_v2_final.csv`, `regime_grid_v2_plateaus.png`, `regime_grid_v2_comparison.png`.

Legacy train/test split versions (kept for reference): `optimize_penalized.py`, `optimize_walkforward.py`, `optimize_wf_penalized.py`, `optimize_vol_adaptive.py`, `optimize_regime.py`, `optimize_vol_regime.py`, `compare_all_strategies.py`.

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
    download, signal_ma, signal_dual_ma, signal_asymmetric_dual_ma,
    signal_vol_adaptive_dual_ma, signal_regime_switching_dual_ma,
    signal_vol_regime_adaptive_ma,
    run_lrs, run_buy_and_hold,
    calc_metrics, signal_trades_per_year, download_ken_french_rf,
    _max_entry_drawdown, _max_recovery_days,
    run_dual_ma_analysis,
    run_eulb1_comparison, run_eulb5_spotcheck, run_part12_comparison,
)
```
