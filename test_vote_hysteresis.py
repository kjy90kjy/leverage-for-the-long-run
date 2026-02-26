"""
Vote Hysteresis Test: P5 + Sym(3,161)

Compare three vote modes:
  - AND gate:    sum==2 → invest, else → cash  (previous test)
  - Hysteresis:  sum==2 → invest, sum==0 → cash, sum==1 → hold previous
  - Baselines:   Sym(3,161), P5 Regime

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

STRAT_NAMES = ["Sym(3,161)", "P5 Regime", "AND", "Hyst(2in/0out)", "OR(1in/0out)"]
STYLES = {
    "Sym(3,161)":     {"color": "#2ecc71", "ls": "--", "lw": 1.5, "zorder": 2},
    "P5 Regime":      {"color": "#e74c3c", "ls": "--", "lw": 1.5, "zorder": 3},
    "AND":            {"color": "#2980b9", "ls": "-",  "lw": 2.0, "zorder": 5},
    "Hyst(2in/0out)": {"color": "#8e44ad", "ls": "-",  "lw": 2.0, "zorder": 6},
    "OR(1in/0out)":   {"color": "#e67e22", "ls": "-",  "lw": 2.0, "zorder": 7},
}


def signal_and_gate(sig_a, sig_b):
    """AND gate: sum==2 → 1, else → 0."""
    return ((sig_a + sig_b) == 2).astype(int)


def signal_hysteresis(sig_a, sig_b):
    """Hysteresis vote: sum==2 → enter, sum==0 → exit, sum==1 → hold."""
    vote_sum = sig_a + sig_b
    n = len(vote_sum)
    result = np.zeros(n, dtype=int)
    state = 0  # 0=cash, 1=invested

    for i in range(n):
        v = vote_sum.iloc[i]
        if v == 2:
            state = 1
        elif v == 0:
            state = 0
        # v == 1: keep state
        result[i] = state

    return pd.Series(result, index=sig_a.index)


def signal_or_hysteresis(sig_a, sig_b):
    """OR-like vote: sum>=1 → enter, sum==0 → exit, sum==2 → hold."""
    vote_sum = sig_a + sig_b
    n = len(vote_sum)
    result = np.zeros(n, dtype=int)
    state = 0

    for i in range(n):
        v = vote_sum.iloc[i]
        if v >= 1:
            state = 1
        elif v == 0:
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

    fig.suptitle("Vote Gate Comparison: P5 + Sym(3,161)\n"
                 "(NDX 3x, ER=3.5%, lag=1, comm=0.2%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "hysteresis_crisis_grid.png", dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / 'hysteresis_crisis_grid.png'}")


def main():
    print("=" * 70)
    print("  Vote Gate Comparison: P5 + Sym(3,161)")
    print("  AND:            sum==2 → invest, else → cash")
    print("  Hyst(2in/0out): sum==2 → invest, sum==0 → cash, sum==1 → hold")
    print("  OR(1in/0out):   sum>=1 → invest, sum==0 → cash")
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

    strats = {
        "Sym(3,161)":     sig_sym3,
        "P5 Regime":      sig_p5,
        "AND":            signal_and_gate(sig_p5, sig_sym3),
        "Hyst(2in/0out)": signal_hysteresis(sig_p5, sig_sym3),
        "OR(1in/0out)":   signal_or_hysteresis(sig_p5, sig_sym3),
    }

    # Run backtests
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
        print(f"\n{'=' * 110}")
        print(f"  {title}")
        print(f"{'=' * 110}")
        print(f"  {'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MDD':>9} {'MDD_Entry':>10} {'Trades/Yr':>10}")
        print(f"  {'─' * 105}")
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

        print(f"\n{'=' * 105}")
        print(f"  {cname}  (NDX {ndx_dd:+.0%})")
        print(f"{'=' * 105}")
        print(f"  {'Strategy':<20} {'Pk->Tr':>8} {'MaxDD':>8} {'MDD_Entry':>10} "
              f"{'WinRet':>8} {'D_Exit':>7} {'D_Cash':>7} {'Whip':>5}")
        print(f"  {'─' * 100}")

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

    plot_fullperiod(equities, ndx, "Full Period (1988-2025)",
                    "hysteresis_fullperiod.png", start_date="1988-01-01")

    plot_fullperiod(equities, ndx, "Post Dot-com (2003-2025)",
                    "hysteresis_post_dotcom.png", start_date="2003-01-01")

    plot_crisis_grid(equities, ndx)

    print("\n  Done.")


if __name__ == "__main__":
    main()
