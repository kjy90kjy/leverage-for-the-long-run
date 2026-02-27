"""
Vote-based Signal Test: P5 AND (3,161)

AND gate: invested only when BOTH P5 Regime and Sym(3,161) agree.
- Exit: whichever is faster to go 0
- Entry: whichever is slower to go 1

Charts: full period, post-dotcom, per-crisis equity grid.
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

from leverage_rotation import (
    download, download_ken_french_rf,
    signal_dual_ma, signal_regime_switching_dual_ma,
    run_lrs, calc_metrics, signal_trades_per_year,
    _max_entry_drawdown,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002

P5_PARAMS = dict(fast_low=48, slow_low=323, fast_high=15, slow_high=229,
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

CONFIRM_DAYS = [0, 1, 2, 3, 5, 10]

BASELINES = ["Sym(3,161)", "P5 Regime"]
BASELINE_STYLES = {
    "Sym(3,161)":  {"color": "#2ecc71", "ls": "--", "lw": 1.5, "zorder": 2},
    "P5 Regime":   {"color": "#e74c3c", "ls": "--", "lw": 1.5, "zorder": 3},
}
CONFIRM_COLORS = ["#2980b9", "#8e44ad", "#e67e22", "#1abc9c", "#c0392b", "#34495e"]


def signal_and_confirmed(sig_a, sig_b, confirm_days=0):
    """AND gate with entry confirmation.

    Exit: immediate when either signal goes 0.
    Entry: AND must be 1 for confirm_days consecutive days.
    confirm_days=0 is plain AND gate.
    """
    and_sig = ((sig_a + sig_b) == 2).astype(int)
    if confirm_days <= 0:
        return and_sig

    n = len(and_sig)
    result = np.zeros(n, dtype=int)
    state = 0          # 0=cash, 1=invested
    consec_ones = 0    # consecutive days AND==1

    for i in range(n):
        if and_sig.iloc[i] == 1:
            consec_ones += 1
        else:
            consec_ones = 0

        if state == 0:
            if consec_ones > confirm_days:  # N full days confirmed
                state = 1
        elif state == 1:
            if and_sig.iloc[i] == 0:
                state = 0

        result[i] = state

    return pd.Series(result, index=sig_a.index)


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


def get_style(sname):
    """Return style dict for a strategy name."""
    if sname in BASELINE_STYLES:
        return BASELINE_STYLES[sname]
    # Confirmation variants
    for i, n in enumerate(CONFIRM_DAYS):
        if sname == f"AND cf={n}":
            return {"color": CONFIRM_COLORS[i % len(CONFIRM_COLORS)],
                    "ls": "-", "lw": 2.0 if n in (0, 3, 5) else 1.5,
                    "zorder": 5 + i}
    return {"color": "#333", "ls": "-", "lw": 1.5, "zorder": 1}


def plot_fullperiod(equities, strat_names, ndx, title_suffix, fname,
                    start_date="1988-01-01"):
    """Full cumulative returns on log scale with crisis shading."""
    fig, ax = plt.subplots(figsize=(16, 8))

    for sname in strat_names:
        sty = get_style(sname)
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
    ax.set_title(f"Cumulative Returns {title_suffix}\n(NDX 3x, ER=3.5%, lag=1, comm=0.2%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")


def plot_crisis_grid(equities, strat_names, ndx):
    """3x3 per-crisis equity grid."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (cname, peak, trough, wend) in enumerate(CRISES):
        ax = axes[idx]
        for sname in strat_names:
            sty = get_style(sname)
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
            ax.legend(fontsize=6, loc="lower left")

    fig.suptitle("P5 AND (3,161) Entry Confirmation Sweep\n"
                 "(NDX 3x, ER=3.5%, lag=1, comm=0.2%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "vote_crisis_grid.png", dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / 'vote_crisis_grid.png'}")


def main():
    print("=" * 70)
    print("  P5 AND (3,161) + Entry Confirmation Sweep")
    print("  Confirm N = " + str(CONFIRM_DAYS))
    print("  NDX 3x | ER=3.5% | lag=1 | comm=0.2%")
    print("=" * 70)

    print("\n  Downloading data...")
    ndx = download("^NDX", start="1985-10-01")
    rf = download_ken_french_rf()
    print(f"  NDX: {len(ndx)} days ({ndx.index[0].date()} -> {ndx.index[-1].date()})")

    # Build signals
    print("  Building signals...")
    sig_sym3 = signal_dual_ma(ndx, fast=3, slow=161)
    sig_p5   = signal_regime_switching_dual_ma(ndx, **P5_PARAMS)

    strats = {"Sym(3,161)": sig_sym3, "P5 Regime": sig_p5}
    STRAT_NAMES = list(strats.keys())
    for n in CONFIRM_DAYS:
        label = f"AND cf={n}"
        strats[label] = signal_and_confirmed(sig_p5, sig_sym3, confirm_days=n)
        STRAT_NAMES.append(label)

    # Run backtests
    print("  Running backtests...")
    equities = {}
    for label, sig in strats.items():
        equities[label] = run_lrs(ndx, sig, leverage=LEVERAGE,
                                   expense_ratio=CALIBRATED_ER,
                                   tbill_rate=rf, signal_lag=SIGNAL_LAG,
                                   commission=COMMISSION)

    # Full-period metrics
    rf_scalar = rf.mean() * 252

    def print_metrics(title, start_date=None):
        print(f"\n{'=' * 105}")
        print(f"  {title}")
        print(f"{'=' * 105}")
        print(f"  {'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MDD':>9} {'MDD_Entry':>10} {'Trades/Yr':>10}")
        print(f"  {'─' * 100}")
        for label in STRAT_NAMES:
            sig = strats[label]
            eq = equities[label]
            if start_date:
                eq = eq.loc[start_date:]
                eq = eq / eq.iloc[0]
                sig_sub = sig.loc[start_date:]
                rf_sub = rf.reindex(eq.index, method="ffill")
                rf_sc = rf_sub.mean() * 252
                m = calc_metrics(eq, tbill_rate=rf_sc, rf_series=rf_sub)
                tpy = signal_trades_per_year(sig_sub)
                mdd_e = _max_entry_drawdown(eq, sig_sub, SIGNAL_LAG)
            else:
                m = calc_metrics(eq, tbill_rate=rf_scalar, rf_series=rf)
                tpy = signal_trades_per_year(sig)
                mdd_e = _max_entry_drawdown(eq, sig, SIGNAL_LAG)
            print(f"  {label:<20} {m['CAGR']:>8.2%} {m['Sharpe']:>8.3f} "
                  f"{m['Sortino']:>9.3f} {m['MDD']:>9.2%} {mdd_e:>10.2%} {tpy:>10.1f}")

    print_metrics("FULL PERIOD (1985-2025)")
    print_metrics("POST DOT-COM (2003-2025)", start_date="2003-01-01")
    print_metrics("MODERN ERA (2010-2025)",   start_date="2010-01-01")

    # Crisis comparison
    print(f"\n\n{'#' * 70}")
    print(f"  CRISIS-BY-CRISIS: MDD & MDD_Entry")
    print(f"{'#' * 70}")

    for cname, peak, trough, wend in CRISES:
        ndx_sub = ndx.loc[peak:trough]
        ndx_dd = ndx_sub.iloc[-1] / ndx_sub.iloc[0] - 1 if len(ndx_sub) > 1 else 0.0

        print(f"\n{'=' * 100}")
        print(f"  {cname}  (NDX {ndx_dd:+.0%})")
        print(f"{'=' * 100}")
        print(f"  {'Strategy':<20} {'Pk->Tr':>8} {'MaxDD':>8} {'MDD_Entry':>10} "
              f"{'WinRet':>8} {'D_Exit':>7} {'D_Cash':>7} {'Whip':>5}")
        print(f"  {'─' * 95}")

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

            print(f"  {sname:<20} {pk2tr:>+7.1%} {maxdd:>7.1%} {mdd_e:>10.2%} "
                  f"{win_ret:>+7.1%} {de} {dc} {whip:>5d}")

    # Charts
    print("\n  Generating charts...")

    plot_fullperiod(equities, STRAT_NAMES, ndx, "Full Period (1988-2025)",
                    "vote_fullperiod.png", start_date="1988-01-01")

    plot_fullperiod(equities, STRAT_NAMES, ndx, "Post Dot-com (2003-2025)",
                    "vote_post_dotcom.png", start_date="2003-01-01")

    plot_crisis_grid(equities, STRAT_NAMES, ndx)

    print("\n  Done.")


if __name__ == "__main__":
    main()
