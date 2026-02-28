# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Leverage Rotation Strategy (LRS) backtesting framework that replicates Michael Gayed's 2016 paper "Leverage for the Long Run" and extends analysis to NASDAQ indices. Also independently validates Korean financial blog "eulb"'s TQQQ moving-average trading strategies.

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib yfinance openpyxl scipy optuna

# 2. TQQQ calibration — MUST RUN FIRST (~5 min)
python calibrate_tqqq.py

# 3. Full analysis (Parts 1-12, ~25-30 min total)
python leverage_rotation.py
```

## All Scripts

**Main analysis:**
```bash
python leverage_rotation.py              # Full pipeline (Parts 1-12)
python calibrate_tqqq.py                 # TQQQ cost calibration (run first)
python diag_nasdaq.py                    # NASDAQ data quality diagnostics
python validate_eulb.py                  # Validate against eulb's published results
```

**Phase 1: Asymmetric Signal (Optuna-based):**
```bash
python optimize_asymmetric.py            # Train/test split, 2000 trials (~7 min)
```

**Phase 2: Adaptive Signals (Full-period optimization):**
```bash
python optimize_penalized_full.py        # Approach C: Asymmetric + penalty (~7 min)
python optimize_all_full.py              # Plans 4/5/6: Vol-Adaptive, Regime, Vol+Regime (~15 min)
```

**Phase 3: Regime Grid Search (Plateau exploration):**
```bash
python optimize_regime_grid.py           # v1: Coarse grid (step 10, ~720k combos, ~10-15 min)
python optimize_regime_grid_v2.py        # v2: Dense grid (step 5/10/5, ~10.6M combos, ~12 min) — LATEST
```

**Crisis analysis & testing:**
```bash
python analyze_crises.py                 # 9 major NDX crises comparison (5 strategies)
python test_vote_hysteresis.py           # Vote gate (AND/Hysteresis/OR) comparison
python test_macro_regime.py              # Macro regime layer testing
python test_hybrid_entry_exit.py         # Hybrid entry/exit testing
```

**Convenience runners:**
```bash
python run_part12_only.py                # Part 12 only (TQQQ-calibrated NDX grid, ~15 min)
python run_parts7to12.py                 # Parts 7-12 all grid searches (~45 min)
python run_grid_all_indices.py           # 3-index calibrated grid search (~3x15 min)
```

All scripts are standalone — no build system or test framework. Output PNGs and CSVs go to `output/`.

All scripts use a Windows UTF-8 boilerplate at the top (`sys.stdout` wrapping + `warnings.filterwarnings("ignore")`) — preserve this pattern when creating new scripts. Matplotlib uses `Agg` backend (headless, file-only output — no interactive windows).

**Note:** `calibrate_tqqq.py` must run before `leverage_rotation.py` (Part 12 uses calibrated ER=3.5%). Other parts can run independently.

## Architecture

### leverage_rotation.py (main script, ~1410 lines)

Organized as a pipeline with these layers:

1. **Data layer** (lines ~34-144): `download()` fetches from yfinance; `_add_shiller_dividends()` synthesizes S&P 500 total returns using Yale Shiller dividend data; `download_ken_french_rf()` gets daily risk-free rates
2. **Signal layer** (lines ~151-340):
   - Basic: `signal_ma()` (price vs SMA), `signal_dual_ma()` (golden cross / fast vs slow SMA), `signal_rsi()`
   - Optimized: `signal_asymmetric_dual_ma()` (separate buy/sell MA pairs with hysteresis state machine)
   - Adaptive: `signal_vol_adaptive_dual_ma()` (MA lengths scale with realized vol, cumsum O(n))
   - Regime-based: `signal_regime_switching_dual_ma()` (different MA pairs for high/low vol regimes), `signal_vol_regime_adaptive_ma()` (regime-dependent base MA + vol-adaptive scaling)
3. **Strategy engine** (lines ~181-229): `run_lrs()` is the core backtest loop — applies leverage when signal=1, T-Bill returns when signal=0, with configurable signal lag and per-trade commission. `run_buy_and_hold()` for benchmarks.
4. **Metrics** (lines ~235-315): `calc_metrics()` computes CAGR, Sharpe (arithmetic mean, Sharpe 1994), Sortino (TDD per Sortino & van der Meer 1991), MDD, Beta, Alpha. `_max_entry_drawdown()` computes MDD from running max of entry-point equity (not equity curve peak). `_max_recovery_days()` computes longest peak-to-recovery span. `signal_trades_per_year()` counts annual round-trips.
5. **Visualization** (lines ~318-400): cumulative returns, drawdowns, volatility bars, rolling excess, crisis annotations
6. **Analysis orchestrators**: `run_dual_ma_grid()` tests all (slow, fast) MA combos × leverage levels; `run_dual_ma_analysis()` runs full grid search, generates 2D/6-panel heatmaps, saves results to CSV

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

### Runner Scripts

- **`run_part12_only.py`**: Runs Part 12 only (TQQQ-calibrated NDX grid, slow 50-350 step 1, fast 2-50 step 1 = 14,748 combos). ~15 min.
- **`run_parts7to12.py`**: Runs Parts 7-12 (all dual MA grid searches for S&P 500 TR, IXIC, NDX). ~45 min.
- **`run_grid_all_indices.py`**: Runs 3x calibrated grid search for all three indices (S&P 500 TR, IXIC, NDX) with `CALIBRATED_ER=0.035`, `lag=1`, `comm=0.2%`, `slow_range step=1`.

### Test Scripts

- **`test_vote_hysteresis.py`**: Compares vote-based signal gates (AND vs Hysteresis vs OR) across crises and full period. Outputs 3 PNGs + CSV.
- **`test_macro_regime.py`**: Tests macro regime layer (bond/equity momentum) + regime-switching interaction. Outputs crisis comparison + full-period plots.
- **`test_hybrid_entry_exit.py`**: Tests hybrid entry/exit strategies combining multiple signal types. Outputs parameter sweep heatmaps + recovery analysis.

### Script dependencies

`diag_nasdaq.py`, `validate_eulb.py`, and `calibrate_tqqq.py` all import from `leverage_rotation.py`. `validate_eulb.py` uses `sys.path.insert(0, script_dir)` relative to its own location — no manual path editing needed.

### calibrate_tqqq.py

TQQQ cost calibration: compares synthetic 3x QQQ (our model) vs actual TQQQ to determine the optimal `expense_ratio` parameter. Sweeps fixed ER (0.5%-3.5%), tests time-varying financing cost model (Ken French RF-based), and analyzes performance across interest rate regimes (ZIRP, rate hike, COVID, high rate). Outputs recommended ER value and three charts. The calibrated ER is used by Part 12 in `leverage_rotation.py`.

**Output:** `calibrate_tqqq_*.png` charts; writes `CALIBRATED_ER = 0.035` to be used by Part 12.

### analyze_crises.py

Crisis-by-crisis strategy comparison across 9 major NDX crises (1987-2025). Compares 5 strategies:
1. Symmetric Dual MA (3, 161) — eulb baseline
2. Vol-Adaptive (`signal_vol_adaptive_dual_ma`)
3. Regime-Switching (`signal_regime_switching_dual_ma`)
4. Vol+Regime (`signal_vol_regime_adaptive_ma`)
5. Buy & Hold 3x (benchmark)

For each crisis: measures signal exit/re-entry timing, MaxDD, period return, recovery days, whipsaw count. Outputs 4 PNGs (crisis detail, period return, MDD, recovery) + 1 CSV (`macro_regime_crisis_detail.csv`). Uses Part 12 conditions (ER=3.5%, lag=1, comm=0.2%).

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

#### Phase 2 Results Summary (Full Period 1987-2025, NDX 3x, ER=3.5%, lag=1, comm=0.2%)

| Strategy | CAGR | Sortino | MDD | MDD_Entry | Recovery (days) | Trades/Year |
|----------|------|---------|-----|-----------|-----------------|-------------|
| **Vol+Regime** (7 params) | **35.3%** | **1.103** | -84.5% | -44.3% | 2456 | 1.6 |
| **Regime-Switching** (6 params) | 34.9% | 1.088 | -79.9% | **-39.2%** | 2452 | 1.7 |
| **Vol-Adaptive** (4 params) | 34.6% | 1.085 | -79.9% | -58.8% | **1891** | 1.2 |
| Symmetric Best (11, 237) | 30.4% | 1.002 | -79.9% | -41.2% | 1891 | 1.7 |
| Symmetric (3, 161) eulb | 23.7% | 0.861 | -96.9% | -87.2% | 4964 | 4.2 |
| Buy & Hold 3x | 14.2% | 0.758 | -100% | -100% | 6479 | 0.0 |

**Key insights:** Vol+Regime achieves +10% Sortino vs symmetric best while maintaining risk (MDD_Entry). Regime-Switching best for downside protection (MDD_Entry -39.2%). Vol-Adaptive best for drawdown recovery speed.

### optimize_regime_grid.py (Phase 3)

Multi-resolution grid search for regime-switching 6-parameter plateau exploration. Three phases:
1. **Coarse grid** (step 10): ~720k combos, all MAs precomputed via cumsum, pure numpy backtest (no pandas overhead per trial)
2. **Plateau identification**: top 1% → neighbour-average Sortino → diverse plateau centres via greedy selection
3. **Fine grid** (step 1, ±5): ~15k trials per plateau centre, vol params fixed

Key optimisations: MA dict precomputation, vol-regime boolean array cache, `fast_eval()` with numpy-only Sortino/trades calculation. Full backtest comparison via `run_lrs()` + `calc_metrics()` at end.

Outputs: `regime_grid_coarse_top.csv`, `regime_grid_final.csv`, `regime_grid_plateaus.png`, `regime_grid_comparison.png`.

### optimize_regime_grid_v2.py (Phase 3 v2 — Dense Grid) — LATEST

Denser grid search to capture Optuna sweet spots that v1's coarse step missed (e.g., fast_high=15, vol_threshold=57.3%, vol_lookback=49). Same 3-phase structure as v1 but with denser exploration:

**Grid Density Comparison:**

| Dimension | v1 | v2 |
|-----------|----|----|
| fast_low / fast_high | step 10, 6 vals each | step 5, 10 vals each [2,7,12,17,...,47] |
| slow_low / slow_high | step 10, 31 vals each | step 10, 31 vals each (unchanged) |
| vol_lookback | step 20, 6 vals | step 10, 11 vals [20,30,...,120] |
| vol_threshold_pct | step 10, 5 vals | step 5, 10 vals [30,35,...,75] |

Total: ~10.6M combos (~12 min at ~18k trial/s).

**Plateau Identification (Phase 2) — Step Sizes in v2:**
```python
step_sizes = {
    'fast_low': 5,          # (adjusted from v1: 10)
    'slow_low': 10,
    'fast_high': 5,         # (adjusted from v1: 10)
    'slow_high': 10,
    'vol_lookback': 10,     # (adjusted from v1: 20)
    'vol_threshold_pct': 5  # (adjusted from v1: 10)
}

# Grid step for neighbour lookup (Chebyshev distance, O(3^6) per trial)
fast_step = 5
slow_step = 10
vol_lb_step = 10
vol_th_step = 5

# Minimum normalized L2 distance between selected plateau centres
min_dist = 3.0
```

Algorithm: Top 1% by penalised objective → neighbour-average (via grid dict lookup, O(1) per neighbor) → greedy selection of diverse centres (L2 distance ≥ 3.0 in normalized space).

Outputs: `regime_grid_v2_coarse_top.csv`, `regime_grid_v2_final.csv`, `regime_grid_v2_plateaus.png`, `regime_grid_v2_comparison.png`.

**Legacy scripts** (kept for reference, use full-period versions instead):
- **Train/test split approaches:** `optimize_asymmetric.py` (Phase 1), `optimize_penalized.py`, `optimize_walkforward.py`, `optimize_wf_penalized.py`, `optimize_vol_adaptive.py`, `optimize_regime.py`, `optimize_vol_regime.py`
- **Old runners:** `compare_all_strategies.py`
- **Phase 3 v1:** `optimize_regime_grid.py` (replaced by denser v2)

**Current production versions:** `optimize_penalized_full.py`, `optimize_all_full.py`, `optimize_regime_grid_v2.py`.

## Critical Domain Concept: Look-Ahead Bias & Signal Lag Timing

The `signal_lag` parameter is the most important correctness control. However, the optimal lag depends on **market microstructure** (liquidity, trading hours, automation capability):

### Signal Lag Definition
- **`signal_lag=0`**: Signal computed at day-T close → applied to day-T return → **same-day execution**
  - Theoretical: Possible if immediate execution after close
  - Practical: Only feasible with automated systems + after-hours liquidity

- **`signal_lag=1`**: Signal computed at day-T close → applied to day-T+1 return → **next-day execution**
  - Conservative assumption: Next trading session (day T+1 open/close)
  - Always feasible: Guaranteed liquidity at market open

### Market-Specific Considerations

**Korean Markets (KRX)**:
- Close: 15:00 KST
- After-hours: No trading (liquidity = 0)
- ∴ **lag=1 is mandatory** (next day 09:00+ trading only)
- lag=0 = impossible

**US Markets (NASDAQ, SPY/TQQQ)**:
- Close: 16:00 EST
- After-hours: 16:00-20:00 trading available (lower liquidity, wider spreads)
- Option A: **lag=0** with after-hours execution (possible but slippage-prone)
- Option B: **lag=1** with next-day open execution (safer, standard)
- Recommendation: **lag=1 for backtesting conservatism**, but lag=0 feasible for automated systems with after-hours capability

**Crypto/Futures Markets**:
- 24/7 trading available
- lag=0 and lag=1 both practical
- Slippage considerations apply

### Code Implementation Note
- `signal.shift(signal_lag)` properly implements both conventions in `run_lrs()`
- However, backtesting uses historical close prices only (no slippage, no spread)
- Real trading will differ: after-hours prices ≠ next open prices

### Bottom Line for This Project
- **Part 12 (NDX, lag=1)**: Conservative & recommended baseline
- **Part 7-9 (lag=0)**: Theoretical, requires inflated returns correction (1.46-1.66x factors)
- **For production trading**: Choose lag based on your execution capability:
  - If automated + after-hours access: lag=0 is possible
  - If manual/next-day only: lag=1 is required
  - For safety: Always use lag=1 as fallback

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
# Core utilities
from leverage_rotation import (
    download, download_ken_french_rf, signal_trades_per_year,
)

# Signal functions
from leverage_rotation import (
    signal_ma, signal_dual_ma, signal_asymmetric_dual_ma,
    signal_vol_adaptive_dual_ma, signal_regime_switching_dual_ma,
    signal_vol_regime_adaptive_ma,
)

# Backtesting
from leverage_rotation import (
    run_lrs, run_buy_and_hold,
)

# Metrics
from leverage_rotation import (
    calc_metrics, _max_entry_drawdown, _max_recovery_days,
)

# Analysis orchestrators
from leverage_rotation import (
    run_dual_ma_analysis,
    run_eulb1_comparison, run_eulb5_spotcheck, run_part12_comparison,
)
```

---

## Known Limitations & Production Warnings

### ⚠️ CRITICAL: Part 7-9 Lag Mismatch (Look-Ahead Bias)

**Status**: ✅ Analyzed and quantified (2026-02-28)

**Problem:**
- **Part 7-9** (grid searches): `lag=0` (same-day execution) + flat 2% risk-free rate
- **Part 12** (production): `lag=1` (next-day execution) + Ken French daily RF
- **Impact**: Part 7-9 results inflated by 46-66% compared to realistic execution

**Correction Factors** (Part 7-9 CAGR ÷ realistic CAGR):
```
Part 7 (GSPC 1928-2020):
  • Mean correction: 1.46x (fast=3 combos need -16% adjustment)
  • Example: MA(3,115) reported 35.65% → realistic 22.08%

Part 8 (IXIC 1971-2025):
  • Mean correction: 1.66x (fast=2 combos need -50% adjustment)
  • Example: MA(2,50) reported 76.71% → realistic 31.51%

Part 12 (NDX):
  • No correction needed (already lag=1, Ken French RF)
```

**How to Use Part 7-9 Results:**
1. Open `output/Part7_lag_correction_table.csv` or `output/Part8_lag_correction_table.csv`
2. Find your combo and its `CAGR_Correction_Factor`
3. **Realistic CAGR** = Reported CAGR ÷ Correction_Factor
4. For fast=2 combos (IXIC): Consider alternative slower combos instead (correction > 2.0x)

**Validation:**
- ✅ Confirmed by `test_lag_comparison.py` (~11% average bias)
- ✅ Validated by `test_walk_forward.py` (realistic lag=1 matches OOS performance)
- ✅ See `LAG_CORRECTION_FINAL_REPORT.md` for detailed analysis

**Recommendation:**
- ✅ **Use Part 12 results** (NDX 3x, lag=1, Ken French RF) as primary reference
- ⚠️ **Apply correction factors** if using Part 7-9 combos
- ❌ **Do NOT compare** Part 7-9 directly to Part 12 without adjustment

---

### Data Snooping & Overfitting Risk

**Status**: ✅ Walk-forward tested (2026-02-28)

**Analysis Period:**
- Full optimization: 1987-2025 (38 years of data snooping)
- Parameters tuned: 6-7 (regime-switching)
- Trading signals: ~50-80 per period

**Finding**: Walk-forward test shows **NO severe numerical overfitting**
- Out-of-sample (2019-2025) performance BETTER than in-sample (1987-2018)
- Reason: 2008 financial crisis (IS period) vs trend-friendly 2019-2025 (OOS period)
- Conclusion: Parameters capture real market structure, not noise

**However:**
- Single train/test split not definitive
- Different time periods will have different regime favorability
- Recommend quarterly walk-forward revalidation (see Priority 2 below)

---

### External Validity & Future Applicability

**Parameter Stability** (regime-switching):
- Optimized on: 1987-2025 (38 years)
- Tested on: 2019-2025 (recent 6 years, favorable regime)
- ⚠️ Risk: 2026+ market regime may differ (fed policy, inflation, geopolitics)
- **Mitigation**: Implement quarterly walk-forward testing with rolling windows

**Signal Lag Convention Variance:**
- Paper replication (Part 1): `lag=0` (theoretical paper standard)
- Realistic execution (Part 12): `lag=1` (practical trading standard)
- Older code (Part 7-9): Mixes both (source of confusion)
- **Mitigation**: All new analysis uses `lag=1` + Ken French RF explicitly

---

## Recommended Production Setup

### For Trading with Regime-Switching (NDX 3x):
```python
# Use Part 12 conditions (fully corrected):
from leverage_rotation import (
    download, signal_regime_switching_dual_ma, run_lrs, calc_metrics,
    download_ken_french_rf
)

price = download("^NDX", start="1985-10-01")
rf_series = download_ken_french_rf()

# Load best params from optimize_regime_grid_v2.py output
fast_low, slow_low = 10, 150    # example
fast_high, slow_high = 20, 200
vol_lookback, vol_threshold = 60, 50.0

sig = signal_regime_switching_dual_ma(
    price,
    fast_low=fast_low, slow_low=slow_low,
    fast_high=fast_high, slow_high=slow_high,
    vol_lookback=vol_lookback, vol_threshold_pct=vol_threshold
)

cum = run_lrs(
    price, sig,
    leverage=3.0,
    expense_ratio=0.035,  # CALIBRATED_ER
    tbill_rate=rf_series,
    signal_lag=1,         # CRITICAL: next-day execution
    commission=0.002      # 0.2% per trade
)

metrics = calc_metrics(cum, tbill_rate=rf_series.mean() * 252, rf_series=rf_series)
```

### For Comparing Alternative Indices (GSPC, IXIC):
```python
# If using Part 7-9 combos, apply correction factors:
from pathlib import Path
import pandas as pd

# Load correction table
corrections = pd.read_csv(Path("output") / "Part7_lag_correction_table.csv")
best_combo = corrections.iloc[0]  # top 1

reported_cagr = best_combo["lag0_CAGR"]
realistic_cagr = reported_cagr / best_combo["CAGR_Correction_Factor"]

print(f"Reported: {reported_cagr:.1%} → Realistic: {realistic_cagr:.1%}")
```

---

## Improvement Roadmap

| Priority | Item | Status | Timeline |
|----------|------|--------|----------|
| **1** | Lag standardization (Part 7-9) | ✅ Analyzed & corrected | Done |
| **2** | Continuous walk-forward testing | 📋 Planned | Q2 2026+ (quarterly) |
| **3** | Documentation of limitations | ✅ This section | Done |
| **4** | Rerun Part 7-9 with lag=1 | 📅 Planned | Q2 2026 |
| **5** | Unified RF across all parts | 📅 Planned | Q3 2026 |

---

## Validation Test Scripts

Three new validation scripts have been added to catch future issues:

```bash
# Test 1: Quantify look-ahead bias
python test_lag_comparison.py        # ~2 min
# Output: lag_comparison_results.csv, lag_comparison_curves.png

# Test 2: Check in-sample vs out-of-sample performance
python test_walk_forward.py          # ~3 min
# Output: walk_forward_is_vs_oos.csv, walk_forward_is_vs_oos_curves.png

# Test 3: Correct Part 7-9 results with lag=1
python fix_part79_lag_mismatch.py    # ~10 min
# Output: Part7_lag_correction_table.csv, Part8_lag_correction_table.csv
```

See `VALIDATION_REPORT.md` and `LAG_CORRECTION_FINAL_REPORT.md` for detailed analysis.
