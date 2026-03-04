"""
Plateau Solo Comparison: Sym(3,161) vs P5 Optuna vs 5 Plateau Peaks (단독)

Vote 없이 각 고원 peak 단독 전략끼리 비교.
플롯: full period, post dot-com, crisis grid.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leverage_rotation import (
    download, download_ken_french_rf,
    signal_dual_ma, signal_regime_switching_dual_ma,
    run_lrs, calc_metrics, signal_trades_per_year,
    _max_entry_drawdown, _max_recovery_days,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002

PLATEAUS = {
    "P1": dict(fast_low=48, slow_low=107, fast_high=49, slow_high=205, vol_lookback=120, vol_threshold_pct=70),
    "P2": dict(fast_low=47, slow_low=124, fast_high=24, slow_high=251, vol_lookback=60,  vol_threshold_pct=70),
    "P3": dict(fast_low=45, slow_low=120, fast_high=50, slow_high=243, vol_lookback=80,  vol_threshold_pct=70),
    "P4": dict(fast_low=47, slow_low=124, fast_high=50, slow_high=197, vol_lookback=60,  vol_threshold_pct=70),
    "P5": dict(fast_low=48, slow_low=298, fast_high=49, slow_high=205, vol_lookback=120, vol_threshold_pct=70),
}

P5_OPTUNA = dict(fast_low=48, slow_low=323, fast_high=15, slow_high=229,
                 vol_lookback=49, vol_threshold_pct=57.3)

CRISES = [
    ("1987 Black Monday",   "1987-10-02", "1987-12-04", "1988-06-30"),
    ("1998 LTCM",           "1998-07-20", "1998-10-08", "1999-03-31"),
    ("2000-02 Dot-com",     "2000-03-10", "2002-10-09", "2004-03-31"),
    ("2007-09 GFC",         "2007-10-31", "2009-03-09", "2011-03-31"),
    ("2018 Q4",             "2018-10-01", "2018-12-24", "2019-06-30"),
    ("2020 COVID",          "2020-02-19", "2020-03-23", "2021-03-31"),
    ("2022 Bear",           "2021-11-19", "2022-10-13", "2024-03-31"),
    ("2024 Aug Dip",        "2024-07-10", "2024-08-05", "2024-12-31"),
    ("2025 Liberation Day", "2025-02-19", "2025-04-08", "2025-12-31"),
]

STRAT_NAMES = ["Sym(3,161)", "P5 Optuna", "P1", "P2", "P3", "P4", "P5"]

STYLES = {
    "Sym(3,161)":  {"color": "#2ecc71", "ls": "--", "lw": 2.0, "zorder": 3},
    "P5 Optuna":   {"color": "#95a5a6", "ls": "--", "lw": 1.5, "zorder": 2},
    "P1":          {"color": "#2980b9", "ls": "-",  "lw": 1.8, "zorder": 5},
    "P2":          {"color": "#8e44ad", "ls": "-",  "lw": 1.8, "zorder": 6},
    "P3":          {"color": "#e67e22", "ls": "-",  "lw": 2.2, "zorder": 9},
    "P4":          {"color": "#16a085", "ls": "-",  "lw": 1.8, "zorder": 7},
    "P5":          {"color": "#c0392b", "ls": "-",  "lw": 1.5, "zorder": 4},
}


def find_exit_reentry(sig_sub, peak_date):
    after_peak = sig_sub.loc[peak_date:]
    if len(after_peak) == 0:
        return None, None
    exits = after_peak[after_peak == 0]
    if len(exits) == 0:
        return None, None
    exit_date = exits.index[0]
    reentries = after_peak.loc[exit_date:][after_peak.loc[exit_date:] == 1]
    reentry_date = reentries.index[0] if len(reentries) > 0 else None
    return exit_date, reentry_date


def plot_fullperiod(equities, ndx, title_suffix, fname, start_date="1988-01-01"):
    fig, ax = plt.subplots(figsize=(16, 8))

    for sname in STRAT_NAMES:
        sty = STYLES[sname]
        eq = equities[sname].loc[start_date:]
        if len(eq) == 0:
            continue
        eq_norm = eq / eq.iloc[0]
        ax.plot(eq_norm.index, eq_norm.values, label=sname,
                color=sty["color"], linestyle=sty["ls"],
                linewidth=sty["lw"], zorder=sty["zorder"])

    for cname, peak, trough, wend in CRISES:
        if pd.Timestamp(trough) < pd.Timestamp(start_date):
            continue
        ax.axvspan(pd.Timestamp(peak), pd.Timestamp(trough),
                   alpha=0.12, color="#e74c3c", zorder=0)
        mid = pd.Timestamp(peak) + (pd.Timestamp(trough) - pd.Timestamp(peak)) / 2
        ymax = ax.get_ylim()[1]
        ax.text(mid, ymax * 0.7, cname.split()[0], fontsize=6,
                ha="center", va="top", color="#c0392b", alpha=0.7, rotation=90)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(f"Plateau Solo Comparison {title_suffix}\n"
                 f"(NDX 3x, ER=3.5%, lag=1, comm=0.2%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")


def plot_crisis_grid(equities, ndx):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (cname, peak, trough, wend) in enumerate(CRISES):
        ax = axes[idx]
        for sname in STRAT_NAMES:
            sty = STYLES[sname]
            eq = equities[sname]
            eq_sub = eq.loc[peak:wend]
            if len(eq_sub) < 2:
                continue
            eq_norm = eq_sub / eq_sub.iloc[0]
            days = np.arange(len(eq_norm))
            ax.plot(days, eq_norm.values, label=sname,
                    color=sty["color"], linestyle=sty["ls"],
                    linewidth=sty["lw"], zorder=sty["zorder"])

        trough_idx = len(ndx.loc[peak:trough]) - 1
        ax.axvline(trough_idx, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

        ndx_sub = ndx.loc[peak:trough]
        if len(ndx_sub) > 1:
            dd = ndx_sub.iloc[-1] / ndx_sub.iloc[0] - 1
            ax.set_title(f"{cname}\n(NDX {dd:+.0%})", fontsize=11, fontweight="bold")
        else:
            ax.set_title(cname, fontsize=11, fontweight="bold")

        ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Trading days from peak", fontsize=8)
        ax.set_ylabel("Equity (peak=1.0)", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Plateau Solo Comparison: Sym(3,161) vs Optuna vs Grid Peaks\n"
                 "(NDX 3x, ER=3.5%, lag=1, comm=0.2%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "plateau_solo_crisis_grid.png", dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / 'plateau_solo_crisis_grid.png'}")


def main():
    print("=" * 70)
    print("  Plateau Solo Comparison")
    print("  Sym(3,161) vs P5 Optuna vs 5 Grid Peaks")
    print("  NDX 3x | ER=3.5% | lag=1 | comm=0.2%")
    print("=" * 70)

    print("\n  Downloading data...")
    ndx = download("^NDX", start="1985-10-01")
    rf = download_ken_french_rf()
    print(f"  NDX: {len(ndx)} days ({ndx.index[0].date()} -> {ndx.index[-1].date()})")

    # Build signals
    print("  Building signals...")
    strats = {}
    strats["Sym(3,161)"] = signal_dual_ma(ndx, fast=3, slow=161)
    strats["P5 Optuna"]  = signal_regime_switching_dual_ma(ndx, **P5_OPTUNA)
    for pname, params in PLATEAUS.items():
        strats[pname] = signal_regime_switching_dual_ma(ndx, **params)

    # Backtests
    print("  Running backtests...")
    equities = {}
    for label, sig in strats.items():
        equities[label] = run_lrs(ndx, sig, leverage=LEVERAGE,
                                   expense_ratio=CALIBRATED_ER,
                                   tbill_rate=rf, signal_lag=SIGNAL_LAG,
                                   commission=COMMISSION)

    # Metrics
    rf_scalar = rf.mean() * 252

    def print_metrics(title, start_date=None):
        print(f"\n{'=' * 120}")
        print(f"  {title}")
        print(f"{'=' * 120}")
        print(f"  {'Strategy':<16} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MDD':>9} {'MDD_Entry':>10} {'Trades/Yr':>10} {'MaxRecov':>9}")
        print(f"  {'-' * 115}")
        for label in STRAT_NAMES:
            sig = strats[label]
            eq = equities[label]
            if start_date:
                eq = eq.loc[start_date:]
                if len(eq) < 2:
                    continue
                eq = eq / eq.iloc[0]
                sig_sub = sig.loc[start_date:]
                rf_sub = rf.reindex(eq.index, method="ffill")
                rf_sc = rf_sub.mean() * 252
                m = calc_metrics(eq, tbill_rate=rf_sc, rf_series=rf_sub)
                tpy = signal_trades_per_year(sig_sub)
                mdd_e = _max_entry_drawdown(eq, sig_sub, SIGNAL_LAG)
                mrd = _max_recovery_days(eq)
            else:
                m = calc_metrics(eq, tbill_rate=rf_scalar, rf_series=rf)
                tpy = signal_trades_per_year(sig)
                mdd_e = _max_entry_drawdown(eq, sig, SIGNAL_LAG)
                mrd = _max_recovery_days(eq)
            print(f"  {label:<16} {m['CAGR']:>8.2%} {m['Sharpe']:>8.3f} "
                  f"{m['Sortino']:>9.3f} {m['MDD']:>9.2%} {mdd_e:>10.2%} "
                  f"{tpy:>10.1f} {mrd:>8d}d")

    print_metrics("FULL PERIOD (1985-2025)")
    print_metrics("POST DOT-COM (2003-2025)", start_date="2003-01-01")
    print_metrics("MODERN ERA (2010-2025)",   start_date="2010-01-01")

    # Crisis detail
    print(f"\n\n{'#' * 70}")
    print(f"  CRISIS-BY-CRISIS")
    print(f"{'#' * 70}")

    for cname, peak, trough, wend in CRISES:
        ndx_sub = ndx.loc[peak:trough]
        ndx_dd = ndx_sub.iloc[-1] / ndx_sub.iloc[0] - 1 if len(ndx_sub) > 1 else 0.0

        print(f"\n{'=' * 115}")
        print(f"  {cname}  (NDX {ndx_dd:+.0%})")
        print(f"{'=' * 115}")
        print(f"  {'Strategy':<16} {'Pk->Tr':>8} {'MaxDD':>8} {'MDD_Entry':>10} "
              f"{'WinRet':>8} {'D_Exit':>7} {'D_Cash':>7} {'Whip':>5}")
        print(f"  {'-' * 110}")

        for sname in STRAT_NAMES:
            sig = strats[sname]
            eq = equities[sname]
            eq_w = eq.loc[peak:wend]
            if len(eq_w) < 2:
                continue
            eq_norm = eq_w / eq_w.iloc[0]
            maxdd = (eq_norm / eq_norm.cummax() - 1).min()

            sig_w = sig.loc[peak:wend]
            mdd_e = _max_entry_drawdown(eq_norm, sig_w, SIGNAL_LAG)

            eq_tr = eq.loc[peak:trough]
            pk2tr = eq_tr.iloc[-1] / eq_tr.iloc[0] - 1 if len(eq_tr) > 1 else 0.0
            win_ret = eq_norm.iloc[-1] - 1

            exit_d, reentry_d = find_exit_reentry(sig_w, peak)
            d_exit = len(ndx.loc[peak:exit_d]) - 1 if exit_d else None
            d_cash = len(ndx.loc[exit_d:reentry_d]) - 1 if exit_d and reentry_d else None
            whip = int(sig_w.diff().abs().sum())

            de = f"{d_exit:>6d}d" if d_exit is not None else "  stay"
            dc = f"{d_cash:>6d}d" if d_cash is not None else "     -"

            print(f"  {sname:<16} {pk2tr:>+7.1%} {maxdd:>7.1%} {mdd_e:>10.2%} "
                  f"{win_ret:>+7.1%} {de} {dc} {whip:>5d}")

    # Charts
    print("\n  Generating charts...")
    plot_fullperiod(equities, ndx, "Full Period (1988-2025)",
                    "plateau_solo_fullperiod.png", start_date="1988-01-01")
    plot_fullperiod(equities, ndx, "Post Dot-com (2003-2025)",
                    "plateau_solo_postdotcom.png", start_date="2003-01-01")
    plot_crisis_grid(equities, ndx)

    print("\n  Done.")


if __name__ == "__main__":
    main()
