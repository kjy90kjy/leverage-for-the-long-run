"""
Phase 2 — Plan 2: Walk-Forward Optimization (Approach A)

10 rolling windows, 500 Optuna trials per window.
Maximize Sortino on train, evaluate OOS, stitch segments.

NDX 1985-2025, 3x leverage, TQQQ-calibrated costs.

Usage:
    python optimize_walkforward.py
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
    signal_asymmetric_dual_ma, signal_trades_per_year,
    run_buy_and_hold, calc_metrics,
)
from optimize_common import (
    OUT_DIR, CALIBRATED_ER, LEVERAGE, SIGNAL_LAG, COMMISSION, SEED,
    apply_warmup, trim_warmup, run_backtest,
    get_symmetric_baselines, print_comparison_table, plot_cumulative_comparison,
    stitch_oos_segments, download_ndx_and_rf,
)

TRIALS_PER_WINDOW = 500
WARMUP_DAYS = 350

# Walk-forward windows: (train_start, train_end, test_start, test_end)
WINDOWS = [
    ("1987-01-01", "1996-12-31", "1997-01-01", "1999-12-31"),
    ("1990-01-01", "1999-12-31", "2000-01-01", "2002-12-31"),
    ("1993-01-01", "2002-12-31", "2003-01-01", "2005-12-31"),
    ("1996-01-01", "2005-12-31", "2006-01-01", "2008-12-31"),
    ("1999-01-01", "2008-12-31", "2009-01-01", "2011-12-31"),
    ("2002-01-01", "2011-12-31", "2012-01-01", "2014-12-31"),
    ("2005-01-01", "2014-12-31", "2015-01-01", "2017-12-31"),
    ("2008-01-01", "2017-12-31", "2018-01-01", "2020-12-31"),
    ("2011-01-01", "2020-12-31", "2021-01-01", "2023-12-31"),
    ("2014-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
]


def make_objective(train_price, rf_series, warmup_days):
    """Maximize Sortino on train set."""
    def objective(trial):
        fast_buy = trial.suggest_int("fast_buy", 2, 50)
        slow_buy = trial.suggest_int("slow_buy", 50, 350)
        fast_sell = trial.suggest_int("fast_sell", 2, 50)
        slow_sell = trial.suggest_int("slow_sell", 50, 350)

        sig_raw = signal_asymmetric_dual_ma(train_price, fast_buy, slow_buy,
                                            fast_sell, slow_sell)
        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)
        tpy = signal_trades_per_year(sig_raw)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"]
    return objective


def run_single_window(i, ndx_price, rf_series):
    """Run optimization for a single walk-forward window."""
    train_start, train_end, test_start, test_end = WINDOWS[i]

    # Include warmup before train_start
    warmup_start = ndx_price.index[0]
    train_with_warmup = ndx_price.loc[warmup_start:train_end]
    train_price = ndx_price.loc[train_start:train_end]

    # Use full data from warmup through train for signal computation
    # but warmup_days is relative to train_with_warmup start
    actual_warmup = len(train_with_warmup) - len(train_price)
    actual_warmup = max(actual_warmup, WARMUP_DAYS)

    # For signal computation, we need enough history
    # Use data from earliest available through train_end
    full_train = train_with_warmup

    test_price = ndx_price.loc[test_start:test_end]
    if len(test_price) < 10:
        print(f"    Window {i}: test period too short, skipping")
        return None

    print(f"  Window {i}: train {train_start}→{train_end}, test {test_start}→{test_end}")
    print(f"    Train: {len(full_train)} days, Test: {len(test_price)} days")

    # Optimize on train
    sampler = optuna.samplers.TPESampler(seed=SEED + i)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"wf_window_{i}")
    obj = make_objective(full_train, rf_series, actual_warmup)
    study.optimize(obj, n_trials=TRIALS_PER_WINDOW, show_progress_bar=False)

    best = study.best_trial
    bp = best.params
    print(f"    Best: ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"
          f" Sortino={best.value:.3f}, Trades/Yr={best.user_attrs['Trades_Yr']:.1f}")

    # OOS evaluation — use data from warmup start through test_end for signal
    full_test_data = ndx_price.loc[:test_end]
    sig_raw_test = signal_asymmetric_dual_ma(full_test_data,
                                              bp["fast_buy"], bp["slow_buy"],
                                              bp["fast_sell"], bp["slow_sell"])
    # Extract test period signal
    sig_test = sig_raw_test.loc[test_start:test_end]

    # Run backtest on test period only (signal already computed with full history)
    from leverage_rotation import run_lrs
    sig_full_test = pd.Series(0, index=test_price.index)
    sig_full_test.loc[sig_test.index] = sig_test
    cum_test = run_lrs(test_price, sig_full_test, leverage=LEVERAGE,
                       expense_ratio=CALIBRATED_ER, tbill_rate=rf_series,
                       signal_lag=SIGNAL_LAG, commission=COMMISSION)
    cum_test = cum_test / cum_test.iloc[0]

    rf_scalar = rf_series.mean() * 252
    m_test = calc_metrics(cum_test, tbill_rate=rf_scalar, rf_series=rf_series)
    tpy_test = signal_trades_per_year(sig_test)
    m_test["Trades/Yr"] = tpy_test

    return {
        "window": i,
        "train_start": train_start, "train_end": train_end,
        "test_start": test_start, "test_end": test_end,
        "params": bp,
        "train_sortino": best.value,
        "train_cagr": best.user_attrs["CAGR"],
        "train_sharpe": best.user_attrs["Sharpe"],
        "train_mdd": best.user_attrs["MDD"],
        "train_trades_yr": best.user_attrs["Trades_Yr"],
        "test_metrics": m_test,
        "cum_test": cum_test,
    }


def main():
    print("=" * 70)
    print("  Phase 2 — Plan 2: Walk-Forward Optimization (Approach A)")
    print(f"  NDX 1985-2025, 3x leverage, TQQQ-calibrated costs")
    print(f"  {len(WINDOWS)} windows × {TRIALS_PER_WINDOW} trials = "
          f"{len(WINDOWS) * TRIALS_PER_WINDOW} total trials")
    print("=" * 70)

    # ── Data ──
    print("\n[1/5] Downloading data...")
    ndx_price, rf_series = download_ndx_and_rf()

    # ── Walk-forward ──
    print("\n[2/5] Running walk-forward optimization...")
    window_results = []
    for i in range(len(WINDOWS)):
        res = run_single_window(i, ndx_price, rf_series)
        if res is not None:
            window_results.append(res)

    # ── Save window results ──
    print("\n[3/5] Saving results...")
    records = []
    for r in window_results:
        records.append({
            "window": r["window"],
            "train_period": f"{r['train_start']}→{r['train_end']}",
            "test_period": f"{r['test_start']}→{r['test_end']}",
            "fast_buy": r["params"]["fast_buy"],
            "slow_buy": r["params"]["slow_buy"],
            "fast_sell": r["params"]["fast_sell"],
            "slow_sell": r["params"]["slow_sell"],
            "train_Sortino": r["train_sortino"],
            "train_CAGR": r["train_cagr"],
            "train_Trades_Yr": r["train_trades_yr"],
            "test_Sortino": r["test_metrics"]["Sortino"],
            "test_CAGR": r["test_metrics"]["CAGR"],
            "test_MDD": r["test_metrics"]["MDD"],
            "test_Trades_Yr": r["test_metrics"]["Trades/Yr"],
        })
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "wf_window_results.csv", index=False)
    print(f"  → saved wf_window_results.csv ({len(df)} windows)")

    # ── Comparison table ──
    print("\n  Per-window results:")
    print(f"  {'Win':>3} {'Train':>10} {'Test':>10} {'Params':>25} "
          f"{'Train Sort':>10} {'Test Sort':>10} {'Test CAGR':>10}")
    print(f"  {'─' * 90}")
    for r in window_results:
        bp = r["params"]
        pstr = f"({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"
        print(f"  {r['window']:>3} {r['train_end'][:4]:>10} {r['test_end'][:4]:>10} "
              f"{pstr:>25} {r['train_sortino']:>10.3f} "
              f"{r['test_metrics']['Sortino']:>10.3f} "
              f"{r['test_metrics']['CAGR']:>10.2%}")

    # ── Stitch OOS ──
    print("\n[4/5] Stitching OOS segments...")
    oos_segments = [r["cum_test"] for r in window_results]
    stitched = stitch_oos_segments(oos_segments)
    print(f"  Stitched curve: {len(stitched)} days "
          f"({stitched.index[0].date()} → {stitched.index[-1].date()})")

    rf_scalar = rf_series.mean() * 252
    m_stitched = calc_metrics(stitched, tbill_rate=rf_scalar, rf_series=rf_series)
    print(f"  Stitched OOS — CAGR: {m_stitched['CAGR']:.2%}, "
          f"Sortino: {m_stitched['Sortino']:.3f}, MDD: {m_stitched['MDD']:.2%}")

    # ── Baselines for stitched period ──
    stitch_start = stitched.index[0]
    stitch_end = stitched.index[-1]
    stitch_price = ndx_price.loc[stitch_start:stitch_end]

    # Symmetric baselines over stitched period
    from leverage_rotation import signal_dual_ma, run_lrs
    baseline_curves = {}
    for label, (fast, slow) in [("Symmetric (3,161)", (3, 161)),
                                 ("Symmetric (3,200)", (3, 200))]:
        # Need full history for signal
        full_for_sig = ndx_price.loc[:stitch_end]
        sig_raw = signal_dual_ma(full_for_sig, slow=slow, fast=fast)
        sig_period = sig_raw.loc[stitch_start:stitch_end]
        cum = run_lrs(stitch_price, sig_period, leverage=LEVERAGE,
                      expense_ratio=CALIBRATED_ER, tbill_rate=rf_series,
                      signal_lag=SIGNAL_LAG, commission=COMMISSION)
        baseline_curves[label] = cum / cum.iloc[0]

    bh3x = run_buy_and_hold(stitch_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    baseline_curves["B&H 3x"] = bh3x / bh3x.iloc[0]

    # ── Charts ──
    print("\n[5/5] Generating charts...")

    # Stitched equity
    curves = {"WF Stitched OOS": stitched}
    curves.update(baseline_curves)
    plot_cumulative_comparison(
        curves,
        f"Walk-Forward Stitched OOS vs Baselines ({stitched.index[0].year}-{stitched.index[-1].year})",
        "wf_stitched_equity.png",
    )

    # Parameter evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    params_list = ["fast_buy", "slow_buy", "fast_sell", "slow_sell"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for ax, param, color in zip(axes.flat, params_list, colors):
        wins = [r["window"] for r in window_results]
        vals = [r["params"][param] for r in window_results]
        ax.plot(wins, vals, "o-", color=color, linewidth=2, markersize=8)
        ax.set_title(param, fontsize=12)
        ax.set_xlabel("Window")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Parameter Evolution Across Walk-Forward Windows", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wf_param_evolution.png", dpi=150)
    plt.close(fig)
    print(f"  → saved wf_param_evolution.png")

    # Train vs Test Sortino bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    wins = [r["window"] for r in window_results]
    train_s = [r["train_sortino"] for r in window_results]
    test_s = [r["test_metrics"]["Sortino"] for r in window_results]
    x = np.arange(len(wins))
    w = 0.35
    ax.bar(x - w/2, train_s, w, label="Train Sortino", color="#3498db", alpha=0.8)
    ax.bar(x + w/2, test_s, w, label="Test Sortino", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w}" for w in wins])
    ax.set_ylabel("Sortino Ratio")
    ax.set_title("Walk-Forward: Train vs Test Sortino per Window", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wf_sortino_per_window.png", dpi=150)
    plt.close(fig)
    print(f"  → saved wf_sortino_per_window.png")

    print("\n" + "=" * 70)
    print("  Done! Results saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
