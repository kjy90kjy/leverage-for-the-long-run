"""
Phase 2 — Plan 3: Walk-Forward + Penalty (Approach A+C)

Walk-forward structure from Plan 2 + C1 penalty objective from Plan 1.
Each window maximizes: Sortino - 0.02 * trades_per_year.

Also runs unpenalized for comparison within each window.

NDX 1985-2025, 3x leverage, TQQQ-calibrated costs.

Usage:
    python optimize_wf_penalized.py
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
    signal_asymmetric_dual_ma, signal_dual_ma, signal_trades_per_year,
    run_lrs, run_buy_and_hold, calc_metrics,
)
from optimize_common import (
    OUT_DIR, CALIBRATED_ER, LEVERAGE, SIGNAL_LAG, COMMISSION, SEED,
    apply_warmup, trim_warmup, run_backtest,
    stitch_oos_segments, plot_cumulative_comparison, download_ndx_and_rf,
)

TRIALS_PER_WINDOW = 500
WARMUP_DAYS = 350
ALPHA = 0.02

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


def make_objective(train_price, rf_series, warmup_days, penalized=True):
    """Create objective. If penalized=True, uses Sortino - ALPHA*trades."""
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
        trial.set_user_attr("Sortino", m["Sortino"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        if penalized:
            return m["Sortino"] - ALPHA * tpy
        return m["Sortino"]
    return objective


def evaluate_oos(ndx_price, params, test_start, test_end, rf_series):
    """Evaluate a parameter set on OOS period."""
    test_price = ndx_price.loc[test_start:test_end]
    if len(test_price) < 10:
        return None, None, None

    full_data = ndx_price.loc[:test_end]
    sig_raw = signal_asymmetric_dual_ma(full_data,
                                         params["fast_buy"], params["slow_buy"],
                                         params["fast_sell"], params["slow_sell"])
    sig_test = sig_raw.loc[test_start:test_end]

    sig_aligned = pd.Series(0, index=test_price.index)
    sig_aligned.loc[sig_test.index] = sig_test
    cum = run_lrs(test_price, sig_aligned, leverage=LEVERAGE,
                  expense_ratio=CALIBRATED_ER, tbill_rate=rf_series,
                  signal_lag=SIGNAL_LAG, commission=COMMISSION)
    cum = cum / cum.iloc[0]

    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    tpy = signal_trades_per_year(sig_test)
    m["Trades/Yr"] = tpy
    return cum, m, tpy


def run_window(i, ndx_price, rf_series, penalized=True):
    """Run optimization for one window."""
    train_start, train_end, test_start, test_end = WINDOWS[i]
    full_train = ndx_price.loc[:train_end]
    train_price = ndx_price.loc[train_start:train_end]
    actual_warmup = max(len(full_train) - len(train_price), WARMUP_DAYS)

    tag = "penalized" if penalized else "unpenalized"
    sampler = optuna.samplers.TPESampler(seed=SEED + i + (100 if not penalized else 0))
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"wfp_{tag}_{i}")
    obj = make_objective(full_train, rf_series, actual_warmup, penalized=penalized)
    study.optimize(obj, n_trials=TRIALS_PER_WINDOW, show_progress_bar=False)

    best = study.best_trial
    bp = best.params
    cum_test, m_test, tpy_test = evaluate_oos(ndx_price, bp, test_start, test_end, rf_series)

    return {
        "window": i,
        "params": bp,
        "train_sortino": best.user_attrs.get("Sortino", best.value),
        "train_trades_yr": best.user_attrs["Trades_Yr"],
        "test_sortino": m_test["Sortino"] if m_test else None,
        "test_cagr": m_test["CAGR"] if m_test else None,
        "test_mdd": m_test["MDD"] if m_test else None,
        "test_trades_yr": tpy_test,
        "cum_test": cum_test,
    }


def main():
    print("=" * 70)
    print("  Phase 2 — Plan 3: Walk-Forward + Penalty (Approach A+C)")
    print(f"  {len(WINDOWS)} windows × {TRIALS_PER_WINDOW} trials, ALPHA={ALPHA}")
    print("=" * 70)

    print("\n[1/5] Downloading data...")
    ndx_price, rf_series = download_ndx_and_rf()

    # ── Run both penalized and unpenalized ──
    print("\n[2/5] Running walk-forward (penalized)...")
    pen_results = []
    for i in range(len(WINDOWS)):
        _, _, ts, te = WINDOWS[i]
        print(f"  Window {i}: test {ts}→{te}")
        res = run_window(i, ndx_price, rf_series, penalized=True)
        pen_results.append(res)
        bp = res["params"]
        print(f"    Pen: ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"
              f" Train Sort={res['train_sortino']:.3f}, Test Sort={res['test_sortino']:.3f}"
              f" Trades/Yr={res['test_trades_yr']:.1f}")

    print("\n[3/5] Running walk-forward (unpenalized)...")
    unpen_results = []
    for i in range(len(WINDOWS)):
        _, _, ts, te = WINDOWS[i]
        print(f"  Window {i}: test {ts}→{te}")
        res = run_window(i, ndx_price, rf_series, penalized=False)
        unpen_results.append(res)
        bp = res["params"]
        print(f"    Unp: ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"
              f" Train Sort={res['train_sortino']:.3f}, Test Sort={res['test_sortino']:.3f}"
              f" Trades/Yr={res['test_trades_yr']:.1f}")

    # ── Save CSV ──
    print("\n[4/5] Saving results...")
    records = []
    for pen, unp in zip(pen_results, unpen_results):
        records.append({
            "window": pen["window"],
            "test_period": f"{WINDOWS[pen['window']][2]}→{WINDOWS[pen['window']][3]}",
            # Penalized
            "pen_params": f"({pen['params']['fast_buy']},{pen['params']['slow_buy']},"
                          f"{pen['params']['fast_sell']},{pen['params']['slow_sell']})",
            "pen_train_sortino": pen["train_sortino"],
            "pen_test_sortino": pen["test_sortino"],
            "pen_test_cagr": pen["test_cagr"],
            "pen_test_trades_yr": pen["test_trades_yr"],
            # Unpenalized
            "unp_params": f"({unp['params']['fast_buy']},{unp['params']['slow_buy']},"
                          f"{unp['params']['fast_sell']},{unp['params']['slow_sell']})",
            "unp_train_sortino": unp["train_sortino"],
            "unp_test_sortino": unp["test_sortino"],
            "unp_test_cagr": unp["test_cagr"],
            "unp_test_trades_yr": unp["test_trades_yr"],
        })
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "wfp_window_results.csv", index=False)
    print(f"  → saved wfp_window_results.csv")

    # ── Stitch and compare ──
    pen_segments = [r["cum_test"] for r in pen_results if r["cum_test"] is not None]
    unp_segments = [r["cum_test"] for r in unpen_results if r["cum_test"] is not None]
    pen_stitched = stitch_oos_segments(pen_segments)
    unp_stitched = stitch_oos_segments(unp_segments)

    # Symmetric baseline over stitched period
    stitch_start = pen_stitched.index[0]
    stitch_end = pen_stitched.index[-1]
    stitch_price = ndx_price.loc[stitch_start:stitch_end]

    full_for_sig = ndx_price.loc[:stitch_end]
    sig_sym = signal_dual_ma(full_for_sig, slow=161, fast=3)
    sig_period = sig_sym.loc[stitch_start:stitch_end]
    cum_sym = run_lrs(stitch_price, sig_period, leverage=LEVERAGE,
                      expense_ratio=CALIBRATED_ER, tbill_rate=rf_series,
                      signal_lag=SIGNAL_LAG, commission=COMMISSION)
    cum_sym = cum_sym / cum_sym.iloc[0]

    # ── Charts ──
    print("\n[5/5] Generating charts...")

    # Stitched equity: penalized vs unpenalized vs symmetric
    curves = {
        "WF Penalized": pen_stitched,
        "WF Unpenalized": unp_stitched,
        "Symmetric (3,161)": cum_sym,
    }
    plot_cumulative_comparison(
        curves,
        f"Walk-Forward Penalized vs Unpenalized ({stitch_start.year}-{stitch_end.year})",
        "wfp_stitched_equity.png",
    )

    # Penalty effect: Sortino comparison per window
    fig, ax = plt.subplots(figsize=(12, 6))
    wins = list(range(len(WINDOWS)))
    pen_s = [r["test_sortino"] if r["test_sortino"] is not None else 0 for r in pen_results]
    unp_s = [r["test_sortino"] if r["test_sortino"] is not None else 0 for r in unpen_results]
    x = np.arange(len(wins))
    w = 0.35
    ax.bar(x - w/2, pen_s, w, label="Penalized", color="#2ecc71", alpha=0.8)
    ax.bar(x + w/2, unp_s, w, label="Unpenalized", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i}" for i in wins])
    ax.set_ylabel("Test Sortino")
    ax.set_title("Penalty Effect: OOS Sortino per Window", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wfp_penalty_effect.png", dpi=150)
    plt.close(fig)
    print(f"  → saved wfp_penalty_effect.png")

    # Trades comparison per window
    fig, ax = plt.subplots(figsize=(12, 6))
    pen_t = [r["test_trades_yr"] if r["test_trades_yr"] is not None else 0 for r in pen_results]
    unp_t = [r["test_trades_yr"] if r["test_trades_yr"] is not None else 0 for r in unpen_results]
    ax.bar(x - w/2, pen_t, w, label="Penalized", color="#2ecc71", alpha=0.8)
    ax.bar(x + w/2, unp_t, w, label="Unpenalized", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i}" for i in wins])
    ax.set_ylabel("Trades / Year")
    ax.set_title("Penalty Effect: Trading Frequency per Window", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wfp_trades_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  → saved wfp_trades_comparison.png")

    # Summary
    rf_scalar = rf_series.mean() * 252
    m_pen = calc_metrics(pen_stitched, tbill_rate=rf_scalar, rf_series=rf_series)
    m_unp = calc_metrics(unp_stitched, tbill_rate=rf_scalar, rf_series=rf_series)
    print(f"\n  Stitched OOS Summary:")
    print(f"    Penalized:   CAGR={m_pen['CAGR']:.2%}, Sortino={m_pen['Sortino']:.3f}, MDD={m_pen['MDD']:.2%}")
    print(f"    Unpenalized: CAGR={m_unp['CAGR']:.2%}, Sortino={m_unp['Sortino']:.3f}, MDD={m_unp['MDD']:.2%}")

    print("\n" + "=" * 70)
    print("  Done! Results saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
