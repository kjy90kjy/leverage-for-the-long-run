"""
Phase 2 — Plans 4/5/6 Full Period Optimization + Final Comparison

All optimizations on full period (1987-2025, warmup=500).
Penalized objective: Sortino - ALPHA * trades_per_year.

Plan 4: Vol-Adaptive MA (4 params)
Plan 5: Regime-Switching (6 params)
Plan 6: Vol+Regime (7 params, mild penalty)

Final comparison: all above + Plan 1 best (from CSV) + Sym best + (3,161) + B&H 3x.

Usage:
    python optimize_all_full.py
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
    signal_vol_adaptive_dual_ma, signal_regime_switching_dual_ma,
    signal_vol_regime_adaptive_ma,
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

WARMUP_DAYS = 500  # max slow 350 + vol_lookback 120 + margin
N_TRIALS = 2000
ALPHA = 0.02
MILD_ALPHA = 0.01  # for 7-param Plan 6


# ── Shared helpers ──

def backtest_and_trim(price, rf_series, sig_raw, warmup_days):
    """Backtest with warmup, return (cum_trimmed, sig_trimmed)."""
    sig = sig_raw.copy()
    if warmup_days < len(price):
        wd = price.index[warmup_days]
        sig.loc[:wd] = 0
        if sig_raw.loc[wd] == 1:
            post = sig_raw.loc[wd:]
            fz = post[post == 0].index
            if len(fz) > 0:
                sig.loc[wd:fz[0]] = 0

    cum = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
    wd = price.index[warmup_days]
    cum = cum.loc[wd:]
    cum = cum / cum.iloc[0]
    sig = sig.loc[wd:]
    return cum, sig


def compute_metrics(cum, sig, rf_series):
    """Full metric set: CAGR, Sharpe, Sortino, Vol, MDD, MDD_Entry, Recovery, Trades/Yr."""
    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    m["MDD_Entry"] = _max_entry_drawdown(cum, sig, signal_lag=SIGNAL_LAG)
    m["Max_Recovery_Days"] = _max_recovery_days(cum)
    m["Trades/Yr"] = signal_trades_per_year(sig)
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


def save_trials(study, name, extra_attrs=None):
    """Save trial CSV."""
    records = []
    for t in study.trials:
        if t.value is not None:
            row = {"trial": t.number, **t.params, "Penalized": t.value}
            for attr in ["Sortino", "CAGR", "Sharpe", "MDD", "MDD_Entry", "Trades_Yr"]:
                row[attr] = t.user_attrs.get(attr)
            records.append(row)
    df = pd.DataFrame(records)
    fname = f"{name}_full_results.csv"
    df.to_csv(OUT_DIR / fname, index=False)
    print(f"  -> {fname} ({len(df)} trials)")
    return df


# ── Plan 4: Vol-Adaptive ──

def run_plan4(price, rf_series):
    print("\n" + "─" * 70)
    print("  Plan 4: Vol-Adaptive MA (4 params)")
    print("─" * 70)

    def objective(trial):
        base_fast = trial.suggest_int("base_fast", 2, 50)
        base_slow = trial.suggest_int("base_slow", 50, 350)
        vol_lookback = trial.suggest_int("vol_lookback", 20, 120)
        vol_scale = trial.suggest_float("vol_scale", 0.0, 2.0)

        sig_raw = signal_vol_adaptive_dual_ma(price, base_fast, base_slow,
                                               vol_lookback, vol_scale)
        cum, sig = backtest_and_trim(price, rf_series, sig_raw, WARMUP_DAYS)
        m = compute_metrics(cum, sig, rf_series)

        for k in ["CAGR", "Sharpe", "Sortino", "MDD", "MDD_Entry"]:
            trial.set_user_attr(k, m[k])
        trial.set_user_attr("Trades_Yr", m["Trades/Yr"])
        return m["Sortino"] - ALPHA * m["Trades/Yr"]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="vol_adaptive_full")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    bp = best.params
    print(f"  Best: base_fast={bp['base_fast']}, base_slow={bp['base_slow']}, "
          f"vol_lb={bp['vol_lookback']}, vol_scale={bp['vol_scale']:.3f}")
    print(f"  Sortino={best.user_attrs['Sortino']:.3f}, "
          f"CAGR={best.user_attrs['CAGR']:.1%}, "
          f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}")

    save_trials(study, "vol_adaptive")

    sig_raw = signal_vol_adaptive_dual_ma(price, bp["base_fast"], bp["base_slow"],
                                           bp["vol_lookback"], bp["vol_scale"])
    cum, sig = backtest_and_trim(price, rf_series, sig_raw, WARMUP_DAYS)
    m = compute_metrics(cum, sig, rf_series)

    label = f"VolAdapt ({bp['base_fast']},{bp['base_slow']},lb{bp['vol_lookback']},s{bp['vol_scale']:.1f})"
    return label, cum, m, study


# ── Plan 5: Regime-Switching ──

def run_plan5(price, rf_series):
    print("\n" + "─" * 70)
    print("  Plan 5: Regime-Switching (6 params)")
    print("─" * 70)

    def objective(trial):
        fast_low = trial.suggest_int("fast_low", 2, 50)
        slow_low = trial.suggest_int("slow_low", 50, 350)
        fast_high = trial.suggest_int("fast_high", 2, 50)
        slow_high = trial.suggest_int("slow_high", 50, 350)
        vol_lookback = trial.suggest_int("vol_lookback", 20, 120)
        vol_threshold_pct = trial.suggest_float("vol_threshold_pct", 30.0, 70.0)

        sig_raw = signal_regime_switching_dual_ma(
            price, fast_low, slow_low, fast_high, slow_high,
            vol_lookback, vol_threshold_pct)
        cum, sig = backtest_and_trim(price, rf_series, sig_raw, WARMUP_DAYS)
        m = compute_metrics(cum, sig, rf_series)

        for k in ["CAGR", "Sharpe", "Sortino", "MDD", "MDD_Entry"]:
            trial.set_user_attr(k, m[k])
        trial.set_user_attr("Trades_Yr", m["Trades/Yr"])
        return m["Sortino"] - ALPHA * m["Trades/Yr"]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="regime_full")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    bp = best.params
    print(f"  Best: lo=({bp['fast_low']},{bp['slow_low']}), hi=({bp['fast_high']},{bp['slow_high']})")
    print(f"  vol_lb={bp['vol_lookback']}, threshold={bp['vol_threshold_pct']:.1f}%")
    print(f"  Sortino={best.user_attrs['Sortino']:.3f}, "
          f"CAGR={best.user_attrs['CAGR']:.1%}, "
          f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}")

    save_trials(study, "regime")

    sig_raw = signal_regime_switching_dual_ma(
        price, bp["fast_low"], bp["slow_low"], bp["fast_high"], bp["slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"])
    cum, sig = backtest_and_trim(price, rf_series, sig_raw, WARMUP_DAYS)
    m = compute_metrics(cum, sig, rf_series)

    label = f"Regime (lo{bp['fast_low']},{bp['slow_low']}|hi{bp['fast_high']},{bp['slow_high']})"
    return label, cum, m, study


# ── Plan 6: Vol+Regime ──

def run_plan6(price, rf_series):
    print("\n" + "─" * 70)
    print("  Plan 6: Vol-Adaptive + Regime-Switching (7 params)")
    print("─" * 70)

    def objective(trial):
        bfl = trial.suggest_int("base_fast_low", 2, 50)
        bsl = trial.suggest_int("base_slow_low", 50, 350)
        bfh = trial.suggest_int("base_fast_high", 2, 50)
        bsh = trial.suggest_int("base_slow_high", 50, 350)
        vol_lookback = trial.suggest_int("vol_lookback", 20, 120)
        vol_threshold_pct = trial.suggest_float("vol_threshold_pct", 30.0, 70.0)
        vol_scale = trial.suggest_float("vol_scale", 0.0, 2.0)

        sig_raw = signal_vol_regime_adaptive_ma(
            price, bfl, bsl, bfh, bsh,
            vol_lookback, vol_threshold_pct, vol_scale)
        cum, sig = backtest_and_trim(price, rf_series, sig_raw, WARMUP_DAYS)
        m = compute_metrics(cum, sig, rf_series)

        for k in ["CAGR", "Sharpe", "Sortino", "MDD", "MDD_Entry"]:
            trial.set_user_attr(k, m[k])
        trial.set_user_attr("Trades_Yr", m["Trades/Yr"])
        return m["Sortino"] - MILD_ALPHA * m["Trades/Yr"]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="vol_regime_full")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    bp = best.params
    print(f"  Best: lo=({bp['base_fast_low']},{bp['base_slow_low']}), "
          f"hi=({bp['base_fast_high']},{bp['base_slow_high']})")
    print(f"  vol_lb={bp['vol_lookback']}, threshold={bp['vol_threshold_pct']:.1f}%, "
          f"vol_scale={bp['vol_scale']:.3f}")
    print(f"  Sortino={best.user_attrs['Sortino']:.3f}, "
          f"CAGR={best.user_attrs['CAGR']:.1%}, "
          f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}")

    save_trials(study, "vol_regime")

    sig_raw = signal_vol_regime_adaptive_ma(
        price, bp["base_fast_low"], bp["base_slow_low"],
        bp["base_fast_high"], bp["base_slow_high"],
        bp["vol_lookback"], bp["vol_threshold_pct"], bp["vol_scale"])
    cum, sig = backtest_and_trim(price, rf_series, sig_raw, WARMUP_DAYS)
    m = compute_metrics(cum, sig, rf_series)

    label = f"VolRegime (lo{bp['base_fast_low']},{bp['base_slow_low']}|hi{bp['base_fast_high']},{bp['base_slow_high']})"
    return label, cum, m, study


# ── Main ──

def main():
    print("=" * 70)
    print("  Phase 2 — Full Period Optimization: Plans 4/5/6 + Comparison")
    print(f"  NDX 3x, ER=3.5%, lag=1, comm=0.2%")
    print(f"  Period: ~1987-2025 (warmup={WARMUP_DAYS}d)")
    print(f"  Trials: {N_TRIALS} per plan")
    print("=" * 70)

    # ── Data ──
    print("\n[1/6] Downloading data...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    rf_series = download_ken_french_rf()
    eval_start = ndx_price.index[WARMUP_DAYS]
    print(f"  NDX: {ndx_price.index[0].date()} ~ {ndx_price.index[-1].date()}")
    print(f"  Eval: {eval_start.date()} ~ {ndx_price.index[-1].date()} "
          f"({len(ndx_price) - WARMUP_DAYS}d, {(len(ndx_price) - WARMUP_DAYS)/252:.1f}yr)")

    # ── Run Plans ──
    print("\n[2/6] Plan 4: Vol-Adaptive...")
    l4, c4, m4, s4 = run_plan4(ndx_price, rf_series)

    print("\n[3/6] Plan 5: Regime-Switching...")
    l5, c5, m5, s5 = run_plan5(ndx_price, rf_series)

    print("\n[4/6] Plan 6: Vol+Regime...")
    l6, c6, m6, s6 = run_plan6(ndx_price, rf_series)

    # ── Benchmarks ──
    print("\n[5/6] Computing benchmarks...")

    all_results = {}
    all_curves = {}

    # Plan 1 penalized best (from CSV)
    csv_p1 = OUT_DIR / "penalized_full_results.csv"
    if csv_p1.exists():
        df_p1 = pd.read_csv(csv_p1)
        best_p1 = df_p1.loc[df_p1["Sortino"].idxmax()]
        fb, sb = int(best_p1["fast_buy"]), int(best_p1["slow_buy"])
        fs, ss = int(best_p1["fast_sell"]), int(best_p1["slow_sell"])
        sig_raw = signal_asymmetric_dual_ma(ndx_price, fb, sb, fs, ss)
        cum, sig = backtest_and_trim(ndx_price, rf_series, sig_raw, WARMUP_DAYS)
        m = compute_metrics(cum, sig, rf_series)
        l1 = f"P1 Asym ({fb},{sb},{fs},{ss})"
        all_results[l1] = m
        all_curves[l1] = cum
        print(f"  P1 loaded: {l1}")

    # Plans 4/5/6
    all_results[l4] = m4; all_curves[l4] = c4
    all_results[l5] = m5; all_curves[l5] = c5
    all_results[l6] = m6; all_curves[l6] = c6

    # Sym grid best
    grid_csv = OUT_DIR / "NDX_calibrated_grid_results.csv"
    if grid_csv.exists():
        df_g = pd.read_csv(grid_csv)
        best_g = df_g.loc[df_g["Sortino"].idxmax()]
        sf, ss_g = int(best_g["fast"]), int(best_g["slow"])
        sig_raw = signal_dual_ma(ndx_price, slow=ss_g, fast=sf)
        cum, sig = backtest_and_trim(ndx_price, rf_series, sig_raw, WARMUP_DAYS)
        m = compute_metrics(cum, sig, rf_series)
        l_sym = f"Sym best ({sf},{ss_g})"
        all_results[l_sym] = m
        all_curves[l_sym] = cum

    # Sym (3,161)
    sig_raw = signal_dual_ma(ndx_price, slow=161, fast=3)
    cum, sig = backtest_and_trim(ndx_price, rf_series, sig_raw, WARMUP_DAYS)
    m = compute_metrics(cum, sig, rf_series)
    all_results["Sym (3,161) eulb"] = m
    all_curves["Sym (3,161) eulb"] = cum

    # B&H 3x
    bh3x = run_buy_and_hold(ndx_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    wd = ndx_price.index[WARMUP_DAYS]
    bh3x = bh3x.loc[wd:]; bh3x = bh3x / bh3x.iloc[0]
    rf_scalar = rf_series.mean() * 252
    m_bh = calc_metrics(bh3x, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh["MDD_Entry"] = m_bh["MDD"]
    m_bh["Max_Recovery_Days"] = _max_recovery_days(bh3x)
    m_bh["Trades/Yr"] = 0.0
    all_results["B&H 3x"] = m_bh
    all_curves["B&H 3x"] = bh3x

    # ── Final table ──
    print_table(f"FULL PERIOD ({eval_start.date()} ~ {ndx_price.index[-1].date()}) — All Strategies",
                all_results)

    # ── Save CSV ──
    records = []
    for name, m in all_results.items():
        records.append({
            "strategy": name,
            "CAGR": m["CAGR"], "Sharpe": m["Sharpe"], "Sortino": m["Sortino"],
            "Volatility": m["Volatility"], "MDD": m["MDD"],
            "MDD_Entry": m["MDD_Entry"],
            "Max_Recovery_Days": m.get("Max_Recovery_Days", ""),
            "Trades_Yr": m.get("Trades/Yr", ""),
        })
    pd.DataFrame(records).to_csv(OUT_DIR / "phase2_full_comparison.csv", index=False)
    print(f"\n  -> phase2_full_comparison.csv")

    # ── Charts ──
    print("\n[6/6] Generating charts...")

    fig, ax = plt.subplots(figsize=(16, 9))
    cmap = plt.cm.tab10
    n = len(all_curves)
    for i, (label, s) in enumerate(all_curves.items()):
        is_baseline = label.startswith("Sym") or label.startswith("B&H")
        ls = "--" if is_baseline else "-"
        lw = 1.8 if is_baseline else 2.2
        ax.plot(s.index, s.values, label=label, linestyle=ls, linewidth=lw,
                color=cmap(i / max(n - 1, 1)))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(f"Phase 2 Full Period: All Strategies ({eval_start.year}-{ndx_price.index[-1].year})",
                 fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_full_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> phase2_full_comparison.png")

    # Stability plots for each plan
    for plan_name, study, params, ranges in [
        ("P4 Vol-Adaptive", s4,
         ["base_fast", "base_slow", "vol_lookback", "vol_scale"],
         {"base_fast": (2, 50), "base_slow": (50, 350), "vol_lookback": (20, 120), "vol_scale": (0, 2)}),
        ("P5 Regime", s5,
         ["fast_low", "slow_low", "fast_high", "slow_high", "vol_lookback", "vol_threshold_pct"],
         {"fast_low": (2, 50), "slow_low": (50, 350), "fast_high": (2, 50), "slow_high": (50, 350),
          "vol_lookback": (20, 120), "vol_threshold_pct": (30, 70)}),
        ("P6 VolRegime", s6,
         ["base_fast_low", "base_slow_low", "base_fast_high", "base_slow_high",
          "vol_lookback", "vol_threshold_pct", "vol_scale"],
         {"base_fast_low": (2, 50), "base_slow_low": (50, 350),
          "base_fast_high": (2, 50), "base_slow_high": (50, 350),
          "vol_lookback": (20, 120), "vol_threshold_pct": (30, 70), "vol_scale": (0, 2)}),
    ]:
        top20 = sorted([t for t in study.trials if t.value is not None],
                       key=lambda t: t.value, reverse=True)[:20]
        np_params = len(params)
        fig, axes = plt.subplots(1, np_params, figsize=(3.5 * np_params, 5))
        if np_params == 1:
            axes = [axes]
        for ax, param in zip(axes, params):
            vals = [t.params[param] for t in top20]
            bp_plot = ax.boxplot(vals, widths=0.6, patch_artist=True)
            bp_plot["boxes"][0].set_facecolor("#3498db")
            bp_plot["boxes"][0].set_alpha(0.6)
            ax.set_title(param, fontsize=10)
            if param in ranges:
                ax.set_ylim(ranges[param])
            ax.set_xticklabels([])
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{plan_name}: Parameter Stability (Top 20)", fontsize=13)
        fig.tight_layout()
        safe_name = plan_name.lower().replace(" ", "_").replace("+", "")
        fig.savefig(OUT_DIR / f"{safe_name}_full_stability.png", dpi=150)
        plt.close(fig)
        print(f"  -> {safe_name}_full_stability.png")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
