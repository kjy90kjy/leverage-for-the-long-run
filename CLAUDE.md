# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Leverage Rotation Strategy (LRS) backtesting framework that replicates Michael Gayed's 2016 paper "Leverage for the Long Run" and extends analysis to NASDAQ indices. Also independently validates Korean financial blog "eulb"'s TQQQ moving-average trading strategies.

## Quick Start

```bash
pip install numpy pandas matplotlib yfinance openpyxl scipy optuna

python calibrate_tqqq.py       # TQQQ cost calibration — run first (~5 min)
python leverage_rotation.py    # Full analysis, Parts 1-12 (~25-30 min)
```

## Running Scripts

All scripts are standalone Python — no build system, no test framework. Output PNGs/CSVs go to `output/`.

**Main analysis:**
```bash
python leverage_rotation.py              # Full pipeline (Parts 1-12)
python calibrate_tqqq.py                 # TQQQ cost calibration (run first, ER=3.5%)
python diag_nasdaq.py                    # NASDAQ data quality diagnostics
python validate_eulb.py                  # Validate against eulb's published results
```

**Phase 2 optimization (full-period, Optuna-based):**
```bash
python optimize_all_full.py              # Plans 4/5/6: Vol-Adaptive, Regime, Vol+Regime (~15 min)
```

**Phase 3 regime grid search:**
```bash
python optimize_regime_grid_v2.py        # Dense grid (step 5/10/5, ~10.6M combos, ~12 min)
```

**Crisis analysis & signal experiments:**
```bash
python analyze_crises.py                 # 9 major NDX crises, 5 strategies
python tests/test_vote_hysteresis.py     # Vote gate (AND/Hysteresis/OR) comparison
python tests/test_macro_regime.py        # Macro regime layer testing
python tests/test_hybrid_entry_exit.py   # Hybrid entry/exit testing
```

**Standalone strategy comparison (production):**
```bash
python lrs_standalone.py                 # 4-strategy comparison across 3 data modes (real TQQQ, synthetic QQQ, synthetic NDX)
```

**Kim-jje (김째) strategy analysis:**
```bash
python tests/test_kimjje_overheat.py     # Overheat stage analysis — traces history, compares variants
python tests/test_kimjje_sensitivity.py  # Parameter sensitivity sweeps, plateau width, walk-forward validation
python tests/optimize_kimjje_overheat.py # Multi-period grid search + plateau detection for overheat params (~30 min)
python tests/test_plateau_solo.py        # Plateau-center solo strategy comparison
python tests/test_plateau_vote.py        # Plateau-center vote strategy comparison
python tests/test_spmo_comparison.py     # SPMO (S&P momentum) strategy comparison
```

**Validation scripts:**
```bash
python tests/test_lag_comparison.py      # Quantify look-ahead bias (~2 min)
python tests/test_walk_forward.py        # In-sample vs out-of-sample comparison (~3 min)
python tests/test_vol_percentile_fix.py  # Verify expanding→rolling percentile fix
python tests/fix_part79_lag_mismatch.py  # Generate Part 7-9 correction tables (~10 min)
python tests/run_part7_8_corrected.py    # Re-run Part 7-8 with lag=1 + Ken French RF (~40 min)
```

**Daily signal generation (production):**
```bash
python daily_signal_generator.py         # Generate today's regime-switching signal → CSV + HTML
python daily_signal_telegram.py          # Same + Telegram notification with price predictions
```

**Convenience runners:**
```bash
python run_part12_only.py                # Part 12 only (~15 min)
python run_parts7to12.py                 # Parts 7-12 (~45 min)
python run_grid_all_indices.py           # 3-index calibrated grid (~45 min)
```

## Code Conventions

- All scripts use a Windows UTF-8 boilerplate at the top (`sys.stdout` wrapping + `warnings.filterwarnings("ignore")`) — preserve this pattern when creating new scripts.
- Matplotlib uses `Agg` backend (headless, file-only output — no interactive windows).
- `calibrate_tqqq.py` must run before `leverage_rotation.py` Part 12 (uses calibrated ER=3.5%).

## Architecture

### leverage_rotation.py (core engine, ~1430 lines)

All other scripts import from this. Organized as a pipeline:

1. **Data layer** (~34-144): `download()` fetches from yfinance; `_add_shiller_dividends()` synthesizes S&P 500 total returns via Yale Shiller dividend data; `download_ken_french_rf()` gets daily risk-free rates
2. **Signal layer** (~151-340):
   - Basic: `signal_ma()` (price vs SMA), `signal_dual_ma()` (golden cross), `signal_rsi()`
   - Optimized: `signal_asymmetric_dual_ma()` (separate buy/sell MA pairs, hysteresis state machine)
   - Adaptive: `signal_vol_adaptive_dual_ma()` (MA lengths scale with realized vol, cumsum O(n))
   - Regime-based: `signal_regime_switching_dual_ma()` (different MA pairs for high/low vol regimes), `signal_vol_regime_adaptive_ma()` (regime-dependent base MA + vol-adaptive scaling)
3. **Strategy engine** (~181-229): `run_lrs()` — applies leverage when signal=1, T-Bill returns when signal=0, with configurable signal lag and per-trade commission. `run_buy_and_hold()` for benchmarks.
4. **Metrics** (~235-315): `calc_metrics()` computes CAGR, Sharpe (arithmetic mean, Sharpe 1994), Sortino (TDD per Sortino & van der Meer 1991), MDD, Beta, Alpha. `_max_entry_drawdown()` = MDD from running max of entry-point equity (not equity curve peak). `_max_recovery_days()` = longest peak-to-recovery span.
5. **Visualization** (~318-400): cumulative returns, drawdowns, volatility bars, rolling excess, crisis annotations
6. **Grid search orchestrators**: `run_dual_ma_grid()` tests all (slow, fast) MA combos x leverage levels; `run_dual_ma_analysis()` generates 2D/6-panel heatmaps + CSV

The main block (~477-1413) runs 12 analysis parts sequentially, each with a config dict.

### Analysis Parts

| Part | Description | Data | lag |
|------|-------------|------|-----|
| 1 | Paper Table 8 replication (1928-2020) | ^GSPC Total Return (Shiller) | 0 |
| 1.5 | NASDAQ Composite long-run (1971-present) | ^IXIC | 0 |
| 2 | Modern analysis (1990-present) | ^GSPC | 0 |
| 3 | Extended MA (50, 100, 200) | ^GSPC | 0 |
| 4 | Nasdaq-100 | ^NDX | 0 |
| 5 | ETF comparison (SPY/SSO/UPRO/QQQ/TQQQ) | Real ETFs | 0 |
| 6 | Dual MA signal example | ^GSPC | 0 |
| 7 | Dual MA Grid — S&P 500 (1928-2020, TR) | ^GSPC | 0 |
| 8 | Dual MA Grid — NASDAQ Composite (1971-2025) | ^IXIC | 0 |
| 9 | Dual MA Grid — Nasdaq-100 (1985-2025) | ^NDX | 0 |
| 10 | eulb post 1 replication | ^NDX | 1 |
| 11 | eulb post 5 replication (2006-2024) | ^NDX | 1 |
| 12 | TQQQ-calibrated NDX grid (production baseline) | ^NDX | 1 |

Config presets: `PAPER_CONFIG` (^GSPC TR, 1928-2020, Ken French RF, lag=0), `NASDAQ_LONGRUN_CONFIG` (^IXIC, 1971+, lag=0), `DEFAULT_CONFIG` (^GSPC, 1990+, 2% T-Bill, lag=0).

### optimize_common.py (shared Phase 2 infrastructure)

Shared utilities for optimization scripts: `apply_warmup()`, `trim_warmup()`, `run_backtest()`, `get_symmetric_baselines()`, `print_comparison_table()`, `plot_cumulative_comparison()`, `download_ndx_and_rf()`. Common constants: `CALIBRATED_ER=0.035, LEVERAGE=3.0, SIGNAL_LAG=1, COMMISSION=0.002`.

### optimize_regime_grid_v2.py (Phase 3 — production grid search)

3-phase multi-resolution grid search for regime-switching 6-parameter space:
1. **Coarse grid**: ~10.6M combos, all MAs precomputed via cumsum, `fast_eval()` with numpy-only Sortino/trades (no pandas per trial)
2. **Plateau identification**: top 1% by penalised objective → neighbour-average Sortino → greedy selection of diverse centres (L2 distance >= 3.0 in normalized space)
3. **Fine grid** (step 1, +/-5): ~15k trials per plateau centre

Also runs a separate post-dotcom (2003+) grid search. Outputs: `regime_grid_v2_*.csv`, `regime_grid_v2_*.png`.

### Daily signal pipeline

- **`daily_signal_generator.py`**: Downloads latest NDX, computes regime-switching signal using "Conservative P1" params, outputs `daily_signals.csv` + `daily_signals.html`. Designed for Windows Task Scheduler (4 PM EST trigger).
- **`daily_signal_telegram.py`**: Extends above with Telegram notifications + price prediction (binary search to find exact crossover price for signal flip).
- **`scriptable_widget.js`**: iOS Scriptable widget that displays the daily signal on iPhone home screen, consuming JSON from the signal generator.

### Script dependency graph

```
leverage_rotation.py (core — all scripts import from here)
  ├── optimize_common.py (shared Phase 2 infra, imports from above)
  │     ├── optimize_all_full.py
  │     └── (archived Phase 1/2 scripts)
  ├── optimize_regime_grid_v2.py (Phase 3, standalone fast_eval)
  ├── calibrate_tqqq.py
  ├── diag_nasdaq.py
  ├── validate_eulb.py
  ├── analyze_crises.py
  ├── daily_signal_generator.py
  ├── daily_signal_telegram.py
  ├── lrs_standalone.py               (standalone 4-strategy comparison, self-contained)
  ├── tests/                          (validation, fix & strategy analysis scripts)
  │     ├── test_*.py, fix_*.py, run_part7_8_corrected.py
  │     ├── apply_lag_correction.py
  │     ├── optimize_kimjje_overheat.py  (overheat param grid search + plateau detection)
  │     ├── test_kimjje_overheat.py      (overheat stage analysis)
  │     ├── test_kimjje_sensitivity.py   (parameter sensitivity & walk-forward)
  │     ├── test_plateau_solo.py, test_plateau_vote.py
  │     └── test_spmo_comparison.py
  └── docs/                           (analysis reports & guides)
```

### lrs_standalone.py (standalone strategy comparison)

Self-contained production strategy comparison script (does not import from `leverage_rotation.py`). Compares 4 strategies across 3 data modes (real TQQQ, synthetic QQQ-based, synthetic NDX-based):
- **Regime-Switching Dual MA**: Conservative P1 parameters from grid search
- **Kim-jje S1**: Simple TQQQ 200-day MA crossover
- **Kim-jje S2**: Full system with vol filter, SPY filter, overheat detection, stop-loss, and 10% take-profit
- **Kim-jje S3**: S2 variant with `spy_bear_cap=10%`

Contains all required functions internally: `download_all_data()`, `backtest_kimjje()`, `calc_metrics()`, `BasicParams` dataclass.

### archive/ directory

Legacy scripts moved here — superseded by full-period optimization or kept for reference only:
- Phase 1 train/test split: `optimize_asymmetric.py`, `optimize_penalized.py`, `optimize_walkforward.py`, `optimize_wf_penalized.py`
- Phase 2 individual: `optimize_vol_adaptive.py`, `optimize_regime.py`, `optimize_vol_regime.py`, `optimize_penalized_full.py`
- Phase 3 v1: `optimize_regime_grid.py`
- Analysis helpers: `compare_all_strategies.py`, `clean_metrics.py`, `regime_switching_ensemble.py`, `crisis_mdd_breakdown.py`, `debug_signals.py`, etc.

## Critical Domain Concepts

### Look-Ahead Bias & Signal Lag

The `signal_lag` parameter is the most important correctness control:
- **`signal_lag=0`**: Signal at day-T close → applied to day-T return → **look-ahead bias** (theoretical only)
- **`signal_lag=1`**: Signal at day-T close → applied to day-T+1 return → **realistic execution**
- **`fast=1` MA is excluded** from grid searches — price vs itself creates extreme look-ahead bias

Implementation: `signal.shift(signal_lag)` in `run_lrs()`.

**Part 7-9 use lag=0; Part 10-12 use lag=1.** Part 7-9 results are inflated by 1.46-1.66x. Correction tables exist in `output/Part7_lag_correction_table.csv` and `output/Part8_lag_correction_table.csv`. Part 12 (NDX, lag=1, Ken French RF) is the recommended production baseline. See `docs/LAG_CORRECTION_FINAL_REPORT.md` for full analysis.

### Vol Percentile Fix (critical correctness)

Regime-switching signals use a volatility percentile to classify high/low vol regimes. Original code used `expanding().rank(pct=True)` which caused 2008's extreme volatility to permanently warp all future percentiles (COVID 2020 appeared as low-vol because 2008 dominated the expanding window). Fixed to `rolling(252, min_periods=1).rank(pct=True)` — 1-year rolling window so each crisis is recognized as a local extreme. Validated by `tests/test_vol_percentile_fix.py`.

### Production Trading Constants

```python
CALIBRATED_ER = 0.035   # 3.5% annual expense ratio (from calibrate_tqqq.py)
LEVERAGE = 3.0
SIGNAL_LAG = 1          # next-day execution (mandatory for KRX, recommended for US)
COMMISSION = 0.002      # 0.2% per trade
```

## External Data Sources

- **Yahoo Finance** (yfinance): ^GSPC, ^IXIC, ^NDX, SPY, SSO, UPRO, QQQ, TQQQ
- **Yale Robert Shiller**: S&P 500 historical dividend yields for total return synthesis
- **Ken French Data Library**: Daily risk-free rates (1926-present)

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `signal_lag` | Signal delay (0=same-day, 1=next-day) | Varies by Part |
| `expense_ratio` | Annual cost ratio | 0.01 (Part 12: 0.035) |
| `commission` | Per-trade cost fraction | 0.0 (Part 10-12: 0.002) |
| `fast_range` | Short MA search range | range(2, 51) |
| `slow_range` | Long MA search range | range(50, 351, 3) |
| `tbill_rate` | T-Bill rate: scalar, or `"ken_french"` for daily series | Varies by Part |

## Importable API

```python
from leverage_rotation import (
    # Data
    download, download_ken_french_rf,
    # Signals
    signal_ma, signal_dual_ma, signal_asymmetric_dual_ma,
    signal_vol_adaptive_dual_ma, signal_regime_switching_dual_ma,
    signal_vol_regime_adaptive_ma,
    # Backtesting
    run_lrs, run_buy_and_hold,
    # Metrics
    calc_metrics, signal_trades_per_year,
    _max_entry_drawdown, _max_recovery_days,
    # Orchestrators
    run_dual_ma_analysis,
    run_eulb1_comparison, run_eulb5_spotcheck, run_part12_comparison,
)
```

## Documentation References

- `docs/CRITICAL_REVIEW.md` — Detailed code review from financial engineering perspective
- `docs/VALIDATION_REPORT.md` — Walk-forward and lag comparison validation results
- `docs/LAG_CORRECTION_FINAL_REPORT.md` — Part 7-9 correction factor analysis
- `docs/COMPLETION_SUMMARY.md` — Summary of validation work completed
- `references/signal_optimization_plan.md` — Signal optimization roadmap
- `docs/REGIME_SWITCHING_STRATEGY_REPORT.md` — Regime-switching strategy report
- `docs/DAILY_SIGNAL_SETUP.md` — Daily signal automation setup guide
- `docs/TELEGRAM_SCRIPTABLE_SETUP.md` — Telegram + iOS widget setup guide
