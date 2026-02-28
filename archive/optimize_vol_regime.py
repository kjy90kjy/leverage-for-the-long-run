"""
Phase 2 — Plan 6: Vol-Adaptive + Regime-Switching (Approach B+D)

Combined: regime-dependent base MA pairs + vol-adaptive scaling.
Mild Sortino penalty to discourage overfitting with 7 params.

Optuna params (7): base_fast_low, base_slow_low, base_fast_high, base_slow_high,
                    vol_lookback, vol_threshold_pct, vol_scale.

NDX 1985-2025, 3x leverage, TQQQ-calibrated costs.
Train: 1985-2014 | Test: 2015-2025.

Usage:
    python optimize_vol_regime.py
"""

import sys
import io
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from leverage_rotation import (
    signal_vol_regime_adaptive_ma, signal_vol_adaptive_dual_ma,
    signal_regime_switching_dual_ma, signal_trades_per_year,
    run_buy_and_hold, calc_metrics,
)
from optimize_common import (
    OUT_DIR, CALIBRATED_ER, LEVERAGE, SIGNAL_LAG, COMMISSION, SEED,
    apply_warmup, trim_warmup, run_backtest,
    get_symmetric_baselines, print_comparison_table, plot_cumulative_comparison,
    plot_param_stability, download_ndx_and_rf,
)

TRAIN_END = "2014-12-31"
TEST_START = "2015-01-01"
N_TRIALS = 1500  # reduced from 2000 for runtime
WARMUP_DAYS = 500
MILD_ALPHA = 0.01  # mild penalty for 7 params


def make_objective(train_price, rf_series, warmup_days):
    """Maximize Sortino - mild penalty * trades_per_year."""
    def objective(trial):
        base_fast_low = trial.suggest_int("base_fast_low", 2, 50)
        base_slow_low = trial.suggest_int("base_slow_low", 50, 350)
        base_fast_high = trial.suggest_int("base_fast_high", 2, 50)
        base_slow_high = trial.suggest_int("base_slow_high", 50, 350)
        vol_lookback = trial.suggest_int("vol_lookback", 20, 120)
        vol_threshold_pct = trial.suggest_float("vol_threshold_pct", 30.0, 70.0)
        vol_scale = trial.suggest_float("vol_scale", 0.0, 2.0)

        sig_raw = signal_vol_regime_adaptive_ma(
            train_price, base_fast_low, base_slow_low,
            base_fast_high, base_slow_high,
            vol_lookback, vol_threshold_pct, vol_scale)
        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)
        tpy = signal_trades_per_year(sig_raw)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("Sortino", m["Sortino"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"] - MILD_ALPHA * tpy
    return objective


def main():
    print("=" * 70)
    print("  Phase 2 — Plan 6: Vol-Adaptive + Regime-Switching (B+D)")
    print(f"  NDX 1985-2025, 3x leverage, TQQQ-calibrated costs")
    print(f"  Optuna TPE, {N_TRIALS} trials, 7 params, mild penalty α={MILD_ALPHA}")
    print("=" * 70)

    print("\n[1/6] Downloading data...")
    ndx_price, rf_series = download_ndx_and_rf()

    train_price = ndx_price.loc[:TRAIN_END]
    test_price = ndx_price.loc[TEST_START:]
    test_warmup = min(WARMUP_DAYS, len(test_price) - 1)

    print(f"  Train: {len(train_price)} days | Test: {len(test_price)} days")

    # ── Optimize ──
    print(f"\n[2/6] Running Optuna ({N_TRIALS} trials)...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="vol_regime_adaptive")
    obj = make_objective(train_price, rf_series, WARMUP_DAYS)
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    bp = best.params
    print(f"\n[3/6] Best parameters (train):")
    print(f"  Low vol:  base_fast={bp['base_fast_low']}, base_slow={bp['base_slow_low']}")
    print(f"  High vol: base_fast={bp['base_fast_high']}, base_slow={bp['base_slow_high']}")
    print(f"  vol_lookback={bp['vol_lookback']}, threshold={bp['vol_threshold_pct']:.1f}%, "
          f"vol_scale={bp['vol_scale']:.3f}")
    print(f"  Sortino={best.user_attrs['Sortino']:.3f}, CAGR={best.user_attrs['CAGR']:.2%}, "
          f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}, "
          f"Penalized={best.value:.3f}")

    # ── OOS evaluation ──
    print("\n[4/6] Out-of-sample validation...")

    sig_test = signal_vol_regime_adaptive_ma(
        test_price, bp["base_fast_low"], bp["base_slow_low"],
        bp["base_fast_high"], bp["base_slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"], bp["vol_scale"])
    cum_test, m_test = run_backtest(test_price, rf_series, sig_test, test_warmup)
    m_test["Trades/Yr"] = signal_trades_per_year(sig_test)

    sig_train = signal_vol_regime_adaptive_ma(
        train_price, bp["base_fast_low"], bp["base_slow_low"],
        bp["base_fast_high"], bp["base_slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"], bp["vol_scale"])
    cum_train, m_train = run_backtest(train_price, rf_series, sig_train, WARMUP_DAYS)
    m_train["Trades/Yr"] = signal_trades_per_year(sig_train)

    # Decomposition: vol-only and regime-only for comparison
    # Vol-only: use average of low/high base params
    avg_fast = (bp["base_fast_low"] + bp["base_fast_high"]) // 2
    avg_slow = (bp["base_slow_low"] + bp["base_slow_high"]) // 2
    sig_vol_only = signal_vol_adaptive_dual_ma(
        test_price, avg_fast, avg_slow,
        bp["vol_lookback"], bp["vol_scale"])
    cum_vol_only, m_vol_only = run_backtest(test_price, rf_series, sig_vol_only, test_warmup)
    m_vol_only["Trades/Yr"] = signal_trades_per_year(sig_vol_only)

    # Regime-only: no vol scaling
    sig_regime_only = signal_regime_switching_dual_ma(
        test_price, bp["base_fast_low"], bp["base_slow_low"],
        bp["base_fast_high"], bp["base_slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"])
    cum_regime_only, m_regime_only = run_backtest(test_price, rf_series, sig_regime_only, test_warmup)
    m_regime_only["Trades/Yr"] = signal_trades_per_year(sig_regime_only)

    # Baselines
    test_baselines = get_symmetric_baselines(test_price, rf_series, test_warmup)
    rf_scalar = rf_series.mean() * 252
    bh3x_test = run_buy_and_hold(test_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_test = trim_warmup(bh3x_test, test_warmup, test_price.index)
    m_bh3x = calc_metrics(bh3x_test, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh3x["Trades/Yr"] = 0

    for k, v in test_baselines.items():
        v["metrics"]["Trades/Yr"] = v["trades_yr"]

    label_combo = f"B+D Combined"
    label_vol = f"Vol-Only (avg base)"
    label_regime = f"Regime-Only (no scale)"

    print_comparison_table("TEST Period (2015-2025)", {
        label_combo: m_test,
        label_vol: m_vol_only,
        label_regime: m_regime_only,
        **{f"Symmetric {k}": v["metrics"] for k, v in test_baselines.items()},
        "B&H 3x": m_bh3x,
    })

    degradation = (m_train["Sortino"] - m_test["Sortino"]) / m_train["Sortino"] * 100 if m_train["Sortino"] > 0 else 0
    print(f"\n  Sortino degradation: {degradation:.1f}%")

    # ── Save results ──
    print("\n[5/6] Saving results...")
    records = []
    for t in study.trials:
        if t.value is not None:
            records.append({
                "trial": t.number,
                **t.params,
                "Sortino": t.user_attrs.get("Sortino"),
                "Penalized": t.value,
                "CAGR": t.user_attrs.get("CAGR"),
                "Sharpe": t.user_attrs.get("Sharpe"),
                "MDD": t.user_attrs.get("MDD"),
                "Trades_Yr": t.user_attrs.get("Trades_Yr"),
            })
    pd.DataFrame(records).to_csv(OUT_DIR / "vol_regime_results.csv", index=False)
    print(f"  → saved vol_regime_results.csv ({len(records)} trials)")

    # ── Charts ──
    print("\n[6/6] Generating charts...")

    # OOS equity
    curves = {
        label_combo: cum_test,
        "Symmetric (3,161)": test_baselines["(3,161) eulb"]["cum"],
        "B&H 3x": bh3x_test,
    }
    plot_cumulative_comparison(curves,
                               "Vol-Regime Adaptive vs Baselines — OOS (2015-2025)",
                               "vol_regime_test.png")

    # Decomposition: 3-way comparison
    curves_decomp = {
        label_combo: cum_test,
        label_vol: cum_vol_only,
        label_regime: cum_regime_only,
        "Symmetric (3,161)": test_baselines["(3,161) eulb"]["cum"],
    }
    plot_cumulative_comparison(curves_decomp,
                               "Decomposition: Combined vs Vol-Only vs Regime-Only (OOS)",
                               "vol_regime_decomposition.png")

    # Parameter stability
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True)[:20]
    params = ["base_fast_low", "base_slow_low", "base_fast_high", "base_slow_high",
              "vol_lookback", "vol_threshold_pct", "vol_scale"]
    param_data = {p: [t.params[p] for t in trials_sorted] for p in params}
    param_ranges = {
        "base_fast_low": (2, 50), "base_slow_low": (50, 350),
        "base_fast_high": (2, 50), "base_slow_high": (50, 350),
        "vol_lookback": (20, 120), "vol_threshold_pct": (30, 70),
        "vol_scale": (0, 2),
    }
    plot_param_stability(param_data, param_ranges,
                         "Vol-Regime Parameter Stability (Top 20)",
                         "vol_regime_stability.png")

    print("\n" + "=" * 70)
    print("  Done! Results saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
