"""
Phase 2 — Plan 1 (Full Period): Whipsaw Penalty Optimization

Train = Test = Full period (1985-2025). No train/test split.
Penalized objective: Sortino - ALPHA * trades_per_year.

Benchmarks:
  1. Symmetric best from Part 12 grid search (by Sortino)
  2. Symmetric (3,161) eulb
  3. B&H 3x

Comparison table: CAGR, MDD, MDD_Entry, Max_Recovery_Days, Sharpe, Sortino, Trades/Yr.

Usage:
    python optimize_penalized_full.py
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
    download, download_ken_french_rf,
    signal_asymmetric_dual_ma, signal_dual_ma,
    signal_trades_per_year,
    run_lrs, run_buy_and_hold, calc_metrics,
    _max_entry_drawdown, _max_recovery_days,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002
SEED = 42

WARMUP_DAYS = 350
N_TRIALS = 2000
ALPHA = 0.02


def full_backtest(price, rf_series, sig_raw, warmup_days):
    """Backtest with warmup, return (cum_trimmed, signal_trimmed)."""
    sig = sig_raw.copy()
    if warmup_days < len(price):
        warmup_date = price.index[warmup_days]
        sig.loc[:warmup_date] = 0
        if sig_raw.loc[warmup_date] == 1:
            post = sig_raw.loc[warmup_date:]
            first_zero = post[post == 0].index
            if len(first_zero) > 0:
                sig.loc[warmup_date:first_zero[0]] = 0

    cum = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)

    warmup_date = price.index[warmup_days]
    cum = cum.loc[warmup_date:]
    cum = cum / cum.iloc[0]
    sig = sig.loc[warmup_date:]
    return cum, sig


def full_metrics(cum, sig, rf_series):
    """Compute all requested metrics."""
    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    m["MDD_Entry"] = _max_entry_drawdown(cum, sig, signal_lag=SIGNAL_LAG)
    m["Max_Recovery_Days"] = _max_recovery_days(cum)
    m["Trades/Yr"] = signal_trades_per_year(sig)
    return m


def make_objective(price, rf_series, warmup_days):
    """C1: Sortino - ALPHA * trades_per_year on full period."""
    def objective(trial):
        fast_buy = trial.suggest_int("fast_buy", 2, 50)
        slow_buy = trial.suggest_int("slow_buy", 50, 350)
        fast_sell = trial.suggest_int("fast_sell", 2, 50)
        slow_sell = trial.suggest_int("slow_sell", 50, 350)

        sig_raw = signal_asymmetric_dual_ma(price, fast_buy, slow_buy,
                                            fast_sell, slow_sell)
        cum, sig = full_backtest(price, rf_series, sig_raw, warmup_days)
        m = full_metrics(cum, sig, rf_series)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("Sortino", m["Sortino"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("MDD_Entry", m["MDD_Entry"])
        trial.set_user_attr("Trades_Yr", m["Trades/Yr"])

        return m["Sortino"] - ALPHA * m["Trades/Yr"]
    return objective


def print_full_table(label, rows):
    """Print comparison table with all metrics."""
    print(f"\n{'=' * 110}")
    print(f"  {label}")
    print(f"{'=' * 110}")
    header = (f"  {'Strategy':<32} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MDD':>8} {'MDD_Ent':>8} {'Recov':>7} {'Trd/Yr':>7} {'Vol':>7}")
    print(header)
    print(f"  {'─' * 105}")
    for name, m in rows.items():
        recov = m.get("Max_Recovery_Days", "")
        recov_str = f"{recov:>7d}" if isinstance(recov, (int, np.integer)) else f"{'—':>7}"
        tpy = m.get("Trades/Yr", 0)
        print(f"  {name:<32} {m['CAGR']:>7.1%} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} "
              f"{m['MDD']:>8.1%} {m['MDD_Entry']:>8.1%} {recov_str} {tpy:>7.1f} {m['Volatility']:>7.1%}")
    print(f"{'=' * 110}")


def main():
    print("=" * 70)
    print("  Penalized Optimization — FULL PERIOD (1985-2025)")
    print(f"  NDX 3x, ER=3.5%, lag=1, comm=0.2%")
    print(f"  Objective: Sortino - {ALPHA} * Trades/Yr")
    print(f"  {N_TRIALS} trials")
    print("=" * 70)

    # ── Data ──
    print("\n[1/4] Downloading data...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    print(f"  NDX: {len(ndx_price)} days ({ndx_price.index[0].date()} -> {ndx_price.index[-1].date()})")
    rf_series = download_ken_french_rf()

    # ── Optimize ──
    print(f"\n[2/4] Running Optuna ({N_TRIALS} trials, full period)...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="penalized_full")
    obj = make_objective(ndx_price, rf_series, WARMUP_DAYS)
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    bp = best.params
    print(f"\n  Best: ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})")
    print(f"  Sortino={best.user_attrs['Sortino']:.3f}, "
          f"CAGR={best.user_attrs['CAGR']:.1%}, "
          f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}, "
          f"Penalized={best.value:.3f}")

    # ── Evaluate best + benchmarks ──
    print("\n[3/4] Computing benchmarks...")

    results = {}

    # 1) Asymmetric penalized best
    sig_raw = signal_asymmetric_dual_ma(ndx_price, bp["fast_buy"], bp["slow_buy"],
                                         bp["fast_sell"], bp["slow_sell"])
    cum_asym, sig_asym = full_backtest(ndx_price, rf_series, sig_raw, WARMUP_DAYS)
    m_asym = full_metrics(cum_asym, sig_asym, rf_series)
    label_asym = f"Asym ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"
    results[label_asym] = m_asym

    # 2) Symmetric best from grid search CSV
    grid_csv = OUT_DIR / "NDX_calibrated_grid_results.csv"
    if grid_csv.exists():
        df_grid = pd.read_csv(grid_csv)
        best_sym_row = df_grid.loc[df_grid["Sortino"].idxmax()]
        sym_fast = int(best_sym_row["fast"])
        sym_slow = int(best_sym_row["slow"])
        sig_raw = signal_dual_ma(ndx_price, slow=sym_slow, fast=sym_fast)
        cum_sym, sig_sym = full_backtest(ndx_price, rf_series, sig_raw, WARMUP_DAYS)
        m_sym = full_metrics(cum_sym, sig_sym, rf_series)
        label_sym = f"Sym best ({sym_fast},{sym_slow})"
        results[label_sym] = m_sym
        print(f"  Grid search best: ({sym_fast},{sym_slow}) Sortino={m_sym['Sortino']:.3f}")
    else:
        print(f"  WARNING: {grid_csv} not found, skipping symmetric best")
        cum_sym = None

    # 3) Symmetric (3,161) eulb
    sig_raw = signal_dual_ma(ndx_price, slow=161, fast=3)
    cum_161, sig_161 = full_backtest(ndx_price, rf_series, sig_raw, WARMUP_DAYS)
    m_161 = full_metrics(cum_161, sig_161, rf_series)
    results["Sym (3,161) eulb"] = m_161

    # 4) B&H 3x
    bh3x = run_buy_and_hold(ndx_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    warmup_date = ndx_price.index[WARMUP_DAYS]
    bh3x = bh3x.loc[warmup_date:]
    bh3x = bh3x / bh3x.iloc[0]
    rf_scalar = rf_series.mean() * 252
    m_bh3x = calc_metrics(bh3x, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh3x["MDD_Entry"] = m_bh3x["MDD"]  # always in position
    m_bh3x["Max_Recovery_Days"] = _max_recovery_days(bh3x)
    m_bh3x["Trades/Yr"] = 0.0
    results["B&H 3x"] = m_bh3x

    # ── Print table ──
    print_full_table("FULL PERIOD (~1987-2025) — Penalized Asym vs Benchmarks", results)

    # ── Top 10 by penalized score ──
    print(f"\n  Top 10 trials by penalized score:")
    print(f"  {'#':>3} {'fb':>3} {'sb':>4} {'fs':>3} {'ss':>4}  {'Sortino':>8} {'Trd/Yr':>7} {'Penalized':>10} {'CAGR':>7} {'MDD':>7}")
    print(f"  {'─' * 65}")
    top10 = sorted([t for t in study.trials if t.value is not None],
                   key=lambda t: t.value, reverse=True)[:10]
    for t in top10:
        p = t.params
        print(f"  {t.number:>3} {p['fast_buy']:>3} {p['slow_buy']:>4} "
              f"{p['fast_sell']:>3} {p['slow_sell']:>4}  "
              f"{t.user_attrs['Sortino']:>8.3f} {t.user_attrs['Trades_Yr']:>7.1f} "
              f"{t.value:>10.3f} {t.user_attrs['CAGR']:>7.1%} {t.user_attrs['MDD']:>7.1%}")

    # ── Save CSV ──
    records = []
    for t in study.trials:
        if t.value is not None:
            records.append({
                "trial": t.number, **t.params,
                "Sortino": t.user_attrs.get("Sortino"),
                "CAGR": t.user_attrs.get("CAGR"),
                "Sharpe": t.user_attrs.get("Sharpe"),
                "MDD": t.user_attrs.get("MDD"),
                "MDD_Entry": t.user_attrs.get("MDD_Entry"),
                "Trades_Yr": t.user_attrs.get("Trades_Yr"),
                "Penalized": t.value,
            })
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "penalized_full_results.csv", index=False)
    print(f"\n  -> saved penalized_full_results.csv ({len(df)} trials)")

    # ── Charts ──
    print("\n[4/4] Generating charts...")

    curves = {label_asym: cum_asym}
    if cum_sym is not None:
        curves[label_sym] = cum_sym
    curves["Sym (3,161) eulb"] = cum_161
    curves["B&H 3x"] = bh3x

    fig, ax = plt.subplots(figsize=(16, 8))
    styles = [
        ("-", 2.0, "#e74c3c"),   # asym best
        ("--", 1.5, "#3498db"),  # sym grid best
        ("--", 1.5, "#2ecc71"),  # (3,161)
        (":", 1.2, "#95a5a6"),   # B&H
    ]
    for (label, s), (ls, lw, color) in zip(curves.items(), styles):
        ax.plot(s.index, s.values, label=label, linestyle=ls, linewidth=lw, color=color)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Penalized Asymmetric (Full Period) vs Benchmarks", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "penalized_full_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> saved penalized_full_comparison.png")

    # Stability box plot
    top20 = sorted([t for t in study.trials if t.value is not None],
                   key=lambda t: t.value, reverse=True)[:20]
    params = ["fast_buy", "slow_buy", "fast_sell", "slow_sell"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    ranges = {"fast_buy": (2, 50), "slow_buy": (50, 350),
              "fast_sell": (2, 50), "slow_sell": (50, 350)}
    for ax, param in zip(axes, params):
        vals = [t.params[param] for t in top20]
        bp_plot = ax.boxplot(vals, widths=0.6, patch_artist=True)
        bp_plot["boxes"][0].set_facecolor("#3498db")
        bp_plot["boxes"][0].set_alpha(0.6)
        ax.set_title(param, fontsize=12)
        ax.set_ylim(ranges[param])
        ax.set_xticklabels([])
        ax.grid(axis="y", alpha=0.3)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        full_r = ranges[param][1] - ranges[param][0]
        pct = iqr / full_r * 100 if full_r > 0 else 0
        ax.text(0.5, 0.02, f"IQR: {iqr:.0f} ({pct:.0f}%)",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="red" if pct > 50 else "green")
    fig.suptitle("Full-Period Penalized: Parameter Stability (Top 20)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "penalized_full_stability.png", dpi=150)
    plt.close(fig)
    print(f"  -> saved penalized_full_stability.png")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
