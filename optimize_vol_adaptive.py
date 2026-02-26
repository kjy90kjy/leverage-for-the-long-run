"""
Phase 2 — Plan 4: Volatility-Adaptive MA Optimization (Approach B)

MA lengths scale with realized volatility: high vol → longer MAs (slower),
low vol → shorter MAs (faster reaction).

Optuna params: base_fast, base_slow, vol_lookback, vol_scale.

NDX 1985-2025, 3x leverage, TQQQ-calibrated costs.
Train: 1985-2014 | Test: 2015-2025.

Usage:
    python optimize_vol_adaptive.py
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
    signal_vol_adaptive_dual_ma, signal_trades_per_year,
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
WARMUP_DAYS = 500  # max slow 350 + max vol_lookback 120 + margin


def make_objective(train_price, rf_series, warmup_days):
    """Maximize Sortino with vol-adaptive signal."""
    def objective(trial):
        base_fast = trial.suggest_int("base_fast", 2, 50)
        base_slow = trial.suggest_int("base_slow", 50, 350)
        vol_lookback = trial.suggest_int("vol_lookback", 20, 120)
        vol_scale = trial.suggest_float("vol_scale", 0.0, 2.0)

        sig_raw = signal_vol_adaptive_dual_ma(train_price, base_fast, base_slow,
                                               vol_lookback, vol_scale)
        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)
        tpy = signal_trades_per_year(sig_raw)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"]
    return objective


def compute_effective_ma_timeline(price, base_fast, base_slow, vol_lookback, vol_scale):
    """Compute the effective MA lengths over time for visualization."""
    prices = price.values.astype(float)
    n = len(prices)
    daily_ret = np.empty(n)
    daily_ret[0] = 0.0
    daily_ret[1:] = prices[1:] / prices[:-1] - 1.0

    rolling_vol = pd.Series(daily_ret, index=price.index).rolling(vol_lookback).std() * np.sqrt(252)
    ref_vol = rolling_vol.expanding().median()
    vol_ratio = (rolling_vol / ref_vol).clip(0.3, 3.0).fillna(1.0).values

    scale = 1.0 + vol_scale * (vol_ratio - 1.0)
    fast_eff = np.clip(np.round(base_fast * scale).astype(int), 2, 100)
    slow_eff = np.clip(np.round(base_slow * scale).astype(int), 30, 500)

    return pd.Series(fast_eff, index=price.index), pd.Series(slow_eff, index=price.index)


def main():
    print("=" * 70)
    print("  Phase 2 — Plan 4: Vol-Adaptive MA Optimization (Approach B)")
    print(f"  NDX 1985-2025, 3x leverage, TQQQ-calibrated costs")
    print(f"  Optuna TPE, {N_TRIALS} trials, 4 params")
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
                                study_name="vol_adaptive")
    obj = make_objective(train_price, rf_series, WARMUP_DAYS)
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    bp = best.params
    print(f"\n[3/6] Best parameters (train):")
    print(f"  base_fast={bp['base_fast']}, base_slow={bp['base_slow']}")
    print(f"  vol_lookback={bp['vol_lookback']}, vol_scale={bp['vol_scale']:.3f}")
    print(f"  Sortino={best.value:.3f}, CAGR={best.user_attrs['CAGR']:.2%}, "
          f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}")

    # ── OOS evaluation ──
    print("\n[4/6] Out-of-sample validation...")

    # Test period
    sig_test = signal_vol_adaptive_dual_ma(test_price, bp["base_fast"], bp["base_slow"],
                                            bp["vol_lookback"], bp["vol_scale"])
    cum_test, m_test = run_backtest(test_price, rf_series, sig_test, test_warmup)
    m_test["Trades/Yr"] = signal_trades_per_year(sig_test)

    # Train period
    sig_train = signal_vol_adaptive_dual_ma(train_price, bp["base_fast"], bp["base_slow"],
                                             bp["vol_lookback"], bp["vol_scale"])
    cum_train, m_train = run_backtest(train_price, rf_series, sig_train, WARMUP_DAYS)
    m_train["Trades/Yr"] = signal_trades_per_year(sig_train)

    # Full period
    sig_full = signal_vol_adaptive_dual_ma(ndx_price, bp["base_fast"], bp["base_slow"],
                                            bp["vol_lookback"], bp["vol_scale"])
    cum_full, m_full = run_backtest(ndx_price, rf_series, sig_full, WARMUP_DAYS)
    m_full["Trades/Yr"] = signal_trades_per_year(sig_full)

    # Baselines
    test_baselines = get_symmetric_baselines(test_price, rf_series, test_warmup)
    rf_scalar = rf_series.mean() * 252
    bh3x_test = run_buy_and_hold(test_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_test = trim_warmup(bh3x_test, test_warmup, test_price.index)
    m_bh3x = calc_metrics(bh3x_test, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh3x["Trades/Yr"] = 0

    for k, v in test_baselines.items():
        v["metrics"]["Trades/Yr"] = v["trades_yr"]

    # Comparison tables
    label = f"VolAdaptive ({bp['base_fast']},{bp['base_slow']},lb={bp['vol_lookback']},s={bp['vol_scale']:.2f})"
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
    pd.DataFrame(records).to_csv(OUT_DIR / "vol_adaptive_results.csv", index=False)
    print(f"  → saved vol_adaptive_results.csv ({len(records)} trials)")

    # ── Charts ──
    print("\n[6/6] Generating charts...")

    # OOS equity
    curves = {
        label: cum_test,
        "Symmetric (3,161)": test_baselines["(3,161) eulb"]["cum"],
        "B&H 3x": bh3x_test,
    }
    plot_cumulative_comparison(curves,
                               "Vol-Adaptive MA vs Baselines — OOS (2015-2025)",
                               "vol_adaptive_test.png")

    # MA timeline
    fast_eff, slow_eff = compute_effective_ma_timeline(
        ndx_price, bp["base_fast"], bp["base_slow"],
        bp["vol_lookback"], bp["vol_scale"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(fast_eff.index, fast_eff.values, color="#3498db", linewidth=0.5, alpha=0.7)
    ax1.axhline(bp["base_fast"], color="#3498db", linestyle="--", alpha=0.5,
                label=f"base_fast={bp['base_fast']}")
    ax1.set_ylabel("Effective Fast MA")
    ax1.set_title("Effective MA Lengths Over Time (Vol-Adaptive)", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Highlight crisis periods
    for ax in axes:
        for start, end, label_crisis in [
            ("2000-03-01", "2002-10-01", "Dot-com"),
            ("2007-10-01", "2009-03-01", "GFC"),
            ("2020-02-01", "2020-04-01", "COVID"),
            ("2022-01-01", "2022-10-01", "2022 Bear"),
        ]:
            try:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                          alpha=0.1, color="red")
            except Exception:
                pass

    ax2 = axes[1]
    ax2.plot(slow_eff.index, slow_eff.values, color="#e74c3c", linewidth=0.5, alpha=0.7)
    ax2.axhline(bp["base_slow"], color="#e74c3c", linestyle="--", alpha=0.5,
                label=f"base_slow={bp['base_slow']}")
    ax2.set_ylabel("Effective Slow MA")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "vol_adaptive_ma_timeline.png", dpi=150)
    plt.close(fig)
    print(f"  → saved vol_adaptive_ma_timeline.png")

    # Parameter stability
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True)[:20]
    param_data = {p: [t.params[p] for t in trials_sorted]
                  for p in ["base_fast", "base_slow", "vol_lookback", "vol_scale"]}
    param_ranges = {"base_fast": (2, 50), "base_slow": (50, 350),
                    "vol_lookback": (20, 120), "vol_scale": (0, 2)}
    plot_param_stability(param_data, param_ranges,
                         "Vol-Adaptive Parameter Stability (Top 20)",
                         "vol_adaptive_stability.png")

    print("\n" + "=" * 70)
    print("  Done! Results saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
