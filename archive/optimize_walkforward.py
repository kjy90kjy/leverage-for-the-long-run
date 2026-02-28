"""
Phase 2 — Walk-Forward Optimization

10 rolling windows (10yr train / 3yr test), 500 Optuna trials per window.
Penalized objective: Sortino - 0.02 * Trades/Year.
OOS segments stitched into full curve (1997-2025).

Comparison: stitched OOS vs Sym best (11,237), (3,161), B&H 3x
with full metrics: CAGR, Sharpe, Sortino, MDD, MDD_Entry, Max_Recovery_Days, Trades/Yr.

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
import matplotlib.ticker as mticker
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from leverage_rotation import (
    signal_asymmetric_dual_ma, signal_dual_ma, signal_trades_per_year,
    run_lrs, run_buy_and_hold, calc_metrics,
    _max_entry_drawdown, _max_recovery_days,
    download, download_ken_french_rf,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002
SEED = 42

TRIALS_PER_WINDOW = 500
WARMUP_DAYS = 500
ALPHA = 0.02

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


def backtest_period(ndx_price, rf_series, sig_fn, start, end):
    """Backtest a signal function over a specific period using full history for signal."""
    full_data = ndx_price.loc[:end]
    test_price = ndx_price.loc[start:end]
    if len(test_price) < 10:
        return None, None, None

    sig_raw = sig_fn(full_data)
    sig_test = sig_raw.loc[start:end]

    sig_aligned = pd.Series(0, index=test_price.index)
    sig_aligned.loc[sig_test.index] = sig_test

    cum = run_lrs(test_price, sig_aligned, leverage=LEVERAGE,
                  expense_ratio=CALIBRATED_ER, tbill_rate=rf_series,
                  signal_lag=SIGNAL_LAG, commission=COMMISSION)
    cum = cum / cum.iloc[0]
    tpy = signal_trades_per_year(sig_test)
    return cum, sig_aligned, tpy


def make_objective(train_data, rf_series, warmup_days):
    """Penalized objective: Sortino - ALPHA * Trades/Year."""
    def objective(trial):
        fast_buy = trial.suggest_int("fast_buy", 2, 50)
        slow_buy = trial.suggest_int("slow_buy", 50, 350)
        fast_sell = trial.suggest_int("fast_sell", 2, 50)
        slow_sell = trial.suggest_int("slow_sell", 50, 350)

        sig_raw = signal_asymmetric_dual_ma(train_data, fast_buy, slow_buy,
                                            fast_sell, slow_sell)
        # Warmup
        sig = sig_raw.copy()
        if warmup_days < len(train_data):
            wd = train_data.index[warmup_days]
            sig.loc[:wd] = 0
            if sig_raw.loc[wd] == 1:
                post = sig_raw.loc[wd:]
                fz = post[post == 0].index
                if len(fz) > 0:
                    sig.loc[wd:fz[0]] = 0

        cum = run_lrs(train_data, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                      tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
        wd = train_data.index[min(warmup_days, len(train_data) - 1)]
        cum = cum.loc[wd:]
        cum = cum / cum.iloc[0]

        rf_scalar = rf_series.mean() * 252
        m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
        tpy = signal_trades_per_year(sig)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("Sortino", m["Sortino"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"] - ALPHA * tpy
    return objective


def run_window(i, ndx_price, rf_series):
    """Optimize one window, return OOS result."""
    train_start, train_end, test_start, test_end = WINDOWS[i]

    # Train data: from earliest available through train_end (for warmup)
    train_data = ndx_price.loc[:train_end]
    actual_warmup = min(WARMUP_DAYS, len(train_data) - 1)

    sampler = optuna.samplers.TPESampler(seed=SEED + i)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"wf_{i}")
    obj = make_objective(train_data, rf_series, actual_warmup)
    study.optimize(obj, n_trials=TRIALS_PER_WINDOW, show_progress_bar=False)

    best = study.best_trial
    bp = best.params

    # OOS evaluation
    sig_fn = lambda p, b=bp: signal_asymmetric_dual_ma(
        p, b["fast_buy"], b["slow_buy"], b["fast_sell"], b["slow_sell"])
    cum_test, sig_test, tpy_test = backtest_period(
        ndx_price, rf_series, sig_fn, test_start, test_end)

    rf_scalar = rf_series.mean() * 252
    m_test = calc_metrics(cum_test, tbill_rate=rf_scalar, rf_series=rf_series) if cum_test is not None else {}

    print(f"  W{i}: train→{train_end[:4]} test {test_start[:4]}-{test_end[:4]} "
          f"({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']}) "
          f"TrainSort={best.user_attrs['Sortino']:.3f} "
          f"TestSort={m_test.get('Sortino', 0):.3f} "
          f"Trd/Yr={tpy_test:.1f}")

    return {
        "window": i,
        "train_end": train_end, "test_start": test_start, "test_end": test_end,
        "params": bp,
        "train_sortino": best.user_attrs["Sortino"],
        "train_cagr": best.user_attrs["CAGR"],
        "train_trades_yr": best.user_attrs["Trades_Yr"],
        "test_sortino": m_test.get("Sortino", 0),
        "test_cagr": m_test.get("CAGR", 0),
        "test_mdd": m_test.get("MDD", 0),
        "test_trades_yr": tpy_test,
        "cum_test": cum_test,
        "sig_test": sig_test,
    }


def stitch_segments(segments):
    """Chain-link OOS equity segments."""
    if not segments:
        return pd.Series(dtype=float)
    stitched = segments[0].copy()
    for seg in segments[1:]:
        scale = stitched.iloc[-1] / seg.iloc[0]
        stitched = pd.concat([stitched, seg.iloc[1:] * scale])
    return stitched


def compute_full_metrics(cum, sig, rf_series):
    """Full metric set."""
    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    if sig is not None:
        m["MDD_Entry"] = _max_entry_drawdown(cum, sig, signal_lag=SIGNAL_LAG)
        m["Trades/Yr"] = signal_trades_per_year(sig)
    else:
        m["MDD_Entry"] = m["MDD"]
        m["Trades/Yr"] = 0.0
    m["Max_Recovery_Days"] = _max_recovery_days(cum)
    return m


def print_table(label, rows):
    """Print comparison table with all metrics."""
    print(f"\n{'=' * 115}")
    print(f"  {label}")
    print(f"{'=' * 115}")
    header = (f"  {'Strategy':<35} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MDD':>8} {'MDD_Ent':>8} {'Recov':>7} {'Trd/Yr':>7} {'Vol':>7}")
    print(header)
    print(f"  {'─' * 108}")
    for name, m in rows.items():
        recov = m.get("Max_Recovery_Days", 0)
        recov_str = f"{recov:>6d}d" if isinstance(recov, (int, np.integer)) else f"{'—':>7}"
        tpy = m.get("Trades/Yr", 0)
        print(f"  {name:<35} {m['CAGR']:>7.1%} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} "
              f"{m['MDD']:>8.1%} {m['MDD_Entry']:>8.1%} {recov_str} {tpy:>7.1f} {m['Volatility']:>7.1%}")
    print(f"{'=' * 115}")


def main():
    print("=" * 70)
    print("  Walk-Forward Optimization (Penalized)")
    print(f"  {len(WINDOWS)} windows x {TRIALS_PER_WINDOW} trials")
    print(f"  Objective: Sortino - {ALPHA} * Trades/Yr")
    print("=" * 70)

    # ── Data ──
    print("\n[1/5] Downloading data...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    rf_series = download_ken_french_rf()
    print(f"  NDX: {ndx_price.index[0].date()} ~ {ndx_price.index[-1].date()} ({len(ndx_price)}d)")

    # ── Walk-forward ──
    print(f"\n[2/5] Running {len(WINDOWS)} windows...")
    results = []
    for i in range(len(WINDOWS)):
        res = run_window(i, ndx_price, rf_series)
        results.append(res)

    # ── Stitch ──
    print("\n[3/5] Stitching OOS segments...")
    oos_segments = [r["cum_test"] for r in results if r["cum_test"] is not None]
    oos_sigs = [r["sig_test"] for r in results if r["sig_test"] is not None]
    stitched = stitch_segments(oos_segments)
    stitched_sig = pd.concat(oos_sigs)

    stitch_start = stitched.index[0]
    stitch_end = stitched.index[-1]
    n_years = (stitch_end - stitch_start).days / 365.25
    print(f"  Stitched: {stitch_start.date()} ~ {stitch_end.date()} ({len(stitched)}d, {n_years:.1f}yr)")

    m_stitched = compute_full_metrics(stitched, stitched_sig, rf_series)

    # ── Baselines over stitched period ──
    print("\n[4/5] Computing baselines...")
    stitch_price = ndx_price.loc[stitch_start:stitch_end]

    all_results = {"WF Stitched OOS": m_stitched}
    all_curves = {"WF Stitched OOS": stitched}

    # Sym best (11,237)
    sig_fn = lambda p: signal_dual_ma(p, slow=237, fast=11)
    cum_sym, sig_sym, _ = backtest_period(ndx_price, rf_series, sig_fn,
                                           str(stitch_start.date()), str(stitch_end.date()))
    m_sym = compute_full_metrics(cum_sym, sig_sym, rf_series)
    all_results["Sym best (11,237)"] = m_sym
    all_curves["Sym best (11,237)"] = cum_sym

    # Sym (3,161)
    sig_fn = lambda p: signal_dual_ma(p, slow=161, fast=3)
    cum_161, sig_161, _ = backtest_period(ndx_price, rf_series, sig_fn,
                                           str(stitch_start.date()), str(stitch_end.date()))
    m_161 = compute_full_metrics(cum_161, sig_161, rf_series)
    all_results["Sym (3,161) eulb"] = m_161
    all_curves["Sym (3,161) eulb"] = cum_161

    # B&H 3x
    bh3x = run_buy_and_hold(stitch_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x = bh3x / bh3x.iloc[0]
    m_bh = compute_full_metrics(bh3x, None, rf_series)
    all_results["B&H 3x"] = m_bh
    all_curves["B&H 3x"] = bh3x

    # ── Results ──
    print_table(f"Stitched OOS ({stitch_start.date()} ~ {stitch_end.date()})", all_results)

    # Per-window table
    print(f"\n  Per-window detail:")
    print(f"  {'W':>2} {'Test':>12} {'Params':>25} {'TrainSort':>10} {'TestSort':>9} {'TestCAGR':>9} {'Trd/Yr':>7}")
    print(f"  {'─' * 85}")
    for r in results:
        bp = r["params"]
        pstr = f"({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"
        print(f"  {r['window']:>2} {r['test_start'][:4]}-{r['test_end'][:4]:>4} "
              f"{pstr:>25} {r['train_sortino']:>10.3f} "
              f"{r['test_sortino']:>9.3f} {r['test_cagr']:>9.1%} {r['test_trades_yr']:>7.1f}")

    # Sortino degradation
    train_avg = np.mean([r["train_sortino"] for r in results])
    test_avg = np.mean([r["test_sortino"] for r in results])
    print(f"\n  Avg Train Sortino: {train_avg:.3f}")
    print(f"  Avg Test Sortino:  {test_avg:.3f}")
    print(f"  Avg Degradation:   {(train_avg - test_avg) / train_avg * 100:.1f}%")

    # ── Save CSV ──
    records = []
    for r in results:
        records.append({
            "window": r["window"],
            "test_period": f"{r['test_start']}~{r['test_end']}",
            "fast_buy": r["params"]["fast_buy"],
            "slow_buy": r["params"]["slow_buy"],
            "fast_sell": r["params"]["fast_sell"],
            "slow_sell": r["params"]["slow_sell"],
            "train_Sortino": r["train_sortino"],
            "train_CAGR": r["train_cagr"],
            "train_Trades_Yr": r["train_trades_yr"],
            "test_Sortino": r["test_sortino"],
            "test_CAGR": r["test_cagr"],
            "test_MDD": r["test_mdd"],
            "test_Trades_Yr": r["test_trades_yr"],
        })
    pd.DataFrame(records).to_csv(OUT_DIR / "wf_window_results.csv", index=False)
    print(f"\n  -> wf_window_results.csv ({len(records)} windows)")

    # ── Charts ──
    print("\n[5/5] Generating charts...")

    # Stitched equity
    fig, ax = plt.subplots(figsize=(16, 8))
    styles = [("-", 2.2, "#e74c3c"), ("--", 1.5, "#3498db"),
              ("--", 1.5, "#2ecc71"), (":", 1.2, "#95a5a6")]
    for (label, s), (ls, lw, c) in zip(all_curves.items(), styles):
        ax.plot(s.index, s.values, label=label, linestyle=ls, linewidth=lw, color=c)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(f"Walk-Forward Stitched OOS vs Baselines ({stitch_start.year}-{stitch_end.year})",
                 fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wf_stitched_equity.png", dpi=150)
    plt.close(fig)
    print(f"  -> wf_stitched_equity.png")

    # Parameter evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    params_list = ["fast_buy", "slow_buy", "fast_sell", "slow_sell"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for ax, param, color in zip(axes.flat, params_list, colors):
        wins = [r["window"] for r in results]
        vals = [r["params"][param] for r in results]
        ax.plot(wins, vals, "o-", color=color, linewidth=2, markersize=8)
        ax.set_title(param, fontsize=12)
        ax.set_xlabel("Window")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Parameter Evolution Across Walk-Forward Windows", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wf_param_evolution.png", dpi=150)
    plt.close(fig)
    print(f"  -> wf_param_evolution.png")

    # Train vs Test Sortino
    fig, ax = plt.subplots(figsize=(12, 6))
    wins = list(range(len(results)))
    train_s = [r["train_sortino"] for r in results]
    test_s = [r["test_sortino"] for r in results]
    x = np.arange(len(wins))
    w = 0.35
    ax.bar(x - w/2, train_s, w, label="Train Sortino", color="#3498db", alpha=0.8)
    ax.bar(x + w/2, test_s, w, label="Test Sortino", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    labels = [f"W{i}\n{results[i]['test_start'][:4]}-{results[i]['test_end'][:4]}" for i in wins]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Sortino Ratio")
    ax.set_title("Walk-Forward: Train vs Test Sortino per Window", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wf_sortino_per_window.png", dpi=150)
    plt.close(fig)
    print(f"  -> wf_sortino_per_window.png")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
