"""
Phase 2 — Plan 5: Regime-Switching MA Optimization (Approach D)

Different MA pairs for high/low volatility regimes.
Regime detection: expanding percentile of rolling vol (no look-ahead).

Optuna params (6): fast_low, slow_low, fast_high, slow_high, vol_lookback, vol_threshold_pct.

NDX 1985-2025, 3x leverage, TQQQ-calibrated costs.
Train: 1985-2014 | Test: 2015-2025.

Usage:
    python optimize_regime.py
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
import matplotlib.dates as mdates

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from leverage_rotation import (
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
N_TRIALS = 2000
WARMUP_DAYS = 500


def make_objective(train_price, rf_series, warmup_days):
    """Maximize Sortino with regime-switching signal."""
    def objective(trial):
        fast_low = trial.suggest_int("fast_low", 2, 50)
        slow_low = trial.suggest_int("slow_low", 50, 350)
        fast_high = trial.suggest_int("fast_high", 2, 50)
        slow_high = trial.suggest_int("slow_high", 50, 350)
        vol_lookback = trial.suggest_int("vol_lookback", 20, 120)
        vol_threshold_pct = trial.suggest_float("vol_threshold_pct", 30.0, 70.0)

        sig_raw = signal_regime_switching_dual_ma(
            train_price, fast_low, slow_low, fast_high, slow_high,
            vol_lookback, vol_threshold_pct)
        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)
        tpy = signal_trades_per_year(sig_raw)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"]
    return objective


def compute_regime_timeline(price, vol_lookback, vol_threshold_pct):
    """Compute regime classification over time."""
    daily_ret = price.pct_change()
    rolling_vol = daily_ret.rolling(vol_lookback).std() * np.sqrt(252)
    vol_pct = rolling_vol.expanding().rank(pct=True) * 100
    high_vol = (vol_pct >= vol_threshold_pct).astype(int)
    return high_vol, rolling_vol


def main():
    print("=" * 70)
    print("  Phase 2 — Plan 5: Regime-Switching MA Optimization (Approach D)")
    print(f"  NDX 1985-2025, 3x leverage, TQQQ-calibrated costs")
    print(f"  Optuna TPE, {N_TRIALS} trials, 6 params")
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
                                study_name="regime_switching")
    obj = make_objective(train_price, rf_series, WARMUP_DAYS)
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    bp = best.params
    print(f"\n[3/6] Best parameters (train):")
    print(f"  Low vol:  fast={bp['fast_low']}, slow={bp['slow_low']}")
    print(f"  High vol: fast={bp['fast_high']}, slow={bp['slow_high']}")
    print(f"  vol_lookback={bp['vol_lookback']}, threshold={bp['vol_threshold_pct']:.1f}%")
    print(f"  Sortino={best.value:.3f}, CAGR={best.user_attrs['CAGR']:.2%}, "
          f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}")

    # ── OOS evaluation ──
    print("\n[4/6] Out-of-sample validation...")

    sig_test = signal_regime_switching_dual_ma(
        test_price, bp["fast_low"], bp["slow_low"],
        bp["fast_high"], bp["slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"])
    cum_test, m_test = run_backtest(test_price, rf_series, sig_test, test_warmup)
    m_test["Trades/Yr"] = signal_trades_per_year(sig_test)

    sig_train = signal_regime_switching_dual_ma(
        train_price, bp["fast_low"], bp["slow_low"],
        bp["fast_high"], bp["slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"])
    cum_train, m_train = run_backtest(train_price, rf_series, sig_train, WARMUP_DAYS)
    m_train["Trades/Yr"] = signal_trades_per_year(sig_train)

    # Baselines
    test_baselines = get_symmetric_baselines(test_price, rf_series, test_warmup)
    rf_scalar = rf_series.mean() * 252
    bh3x_test = run_buy_and_hold(test_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_test = trim_warmup(bh3x_test, test_warmup, test_price.index)
    m_bh3x = calc_metrics(bh3x_test, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh3x["Trades/Yr"] = 0

    for k, v in test_baselines.items():
        v["metrics"]["Trades/Yr"] = v["trades_yr"]

    label = (f"Regime ({bp['fast_low']},{bp['slow_low']}|"
             f"{bp['fast_high']},{bp['slow_high']})")
    print_comparison_table("TEST Period (2015-2025)", {
        label: m_test,
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
                "Sortino": t.value,
                "CAGR": t.user_attrs.get("CAGR"),
                "Sharpe": t.user_attrs.get("Sharpe"),
                "MDD": t.user_attrs.get("MDD"),
                "Trades_Yr": t.user_attrs.get("Trades_Yr"),
            })
    pd.DataFrame(records).to_csv(OUT_DIR / "regime_results.csv", index=False)
    print(f"  → saved regime_results.csv ({len(records)} trials)")

    # ── Charts ──
    print("\n[6/6] Generating charts...")

    # OOS equity
    curves = {
        label: cum_test,
        "Symmetric (3,161)": test_baselines["(3,161) eulb"]["cum"],
        "B&H 3x": bh3x_test,
    }
    plot_cumulative_comparison(curves,
                               "Regime-Switching vs Baselines — OOS (2015-2025)",
                               "regime_test.png")

    # Regime bands — full period
    sig_full = signal_regime_switching_dual_ma(
        ndx_price, bp["fast_low"], bp["slow_low"],
        bp["fast_high"], bp["slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"])
    regime, rolling_vol = compute_regime_timeline(
        ndx_price, bp["vol_lookback"], bp["vol_threshold_pct"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax1 = axes[0]
    ax1.plot(ndx_price.index, ndx_price.values, color="black", linewidth=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel("NDX Price (log)")
    ax1.set_title("Regime Classification Over Time", fontsize=14)

    # Color regime bands
    regime_vals = regime.values
    idx = ndx_price.index
    i = 0
    while i < len(idx):
        j = i
        while j < len(idx) and regime_vals[j] == regime_vals[i]:
            j += 1
        color = "#ffcccc" if regime_vals[i] == 1 else "#ccffcc"
        ax1.axvspan(idx[i], idx[min(j, len(idx)-1)], alpha=0.3, color=color)
        i = j

    # Legend patches
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor="#ccffcc", alpha=0.5, label="Low Vol Regime"),
        Patch(facecolor="#ffcccc", alpha=0.5, label="High Vol Regime"),
    ], fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(rolling_vol.index, rolling_vol.values, color="#e74c3c", linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("Rolling Vol (ann.)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "regime_bands.png", dpi=150)
    plt.close(fig)
    print(f"  → saved regime_bands.png")

    # Parameter stability
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True)[:20]
    params = ["fast_low", "slow_low", "fast_high", "slow_high", "vol_lookback", "vol_threshold_pct"]
    param_data = {p: [t.params[p] for t in trials_sorted] for p in params}
    param_ranges = {"fast_low": (2, 50), "slow_low": (50, 350),
                    "fast_high": (2, 50), "slow_high": (50, 350),
                    "vol_lookback": (20, 120), "vol_threshold_pct": (30, 70)}
    plot_param_stability(param_data, param_ranges,
                         "Regime-Switching Parameter Stability (Top 20)",
                         "regime_stability.png")

    print("\n" + "=" * 70)
    print("  Done! Results saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
