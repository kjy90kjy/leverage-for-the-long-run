"""
SPMO vs Leveraged Strategies Comparison (since SPMO inception 2015-10)

Strategies:
  - SPMO Buy & Hold (S&P 500 Momentum ETF, 1x)
  - Sym(3,161) — NDX 3x
  - P5 Regime — NDX 3x
  - AND gate (P5 AND Sym(3,161)) — NDX 3x

Charts: full period since SPMO inception, per-crisis equity grid.
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
    run_lrs, run_buy_and_hold, calc_metrics, signal_trades_per_year,
    _max_entry_drawdown,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002

P5_PARAMS = dict(fast_low=48, slow_low=323, fast_high=15, slow_high=229,
                 vol_lookback=49, vol_threshold_pct=57.3)

CRISES = [
    ("2018 Q4",             "2018-10-01", "2018-12-24", "2019-06-30"),
    ("2020 COVID",          "2020-02-19", "2020-03-23", "2021-03-31"),
    ("2022 Bear",           "2021-11-19", "2022-10-13", "2024-03-31"),
    ("2024 Aug Dip",        "2024-07-10", "2024-08-05", "2024-12-31"),
    ("2025 Liberation Day", "2025-02-19", "2025-04-08", "2025-12-31"),
]

STRAT_NAMES = ["SPMO B&H", "Sym(3,161)", "P5 Regime", "AND"]
STYLES = {
    "SPMO B&H":    {"color": "#f39c12", "ls": "-",  "lw": 2.0, "zorder": 4},
    "Sym(3,161)":  {"color": "#2ecc71", "ls": "--", "lw": 1.5, "zorder": 2},
    "P5 Regime":   {"color": "#e74c3c", "ls": "--", "lw": 1.5, "zorder": 3},
    "AND":         {"color": "#2980b9", "ls": "-",  "lw": 2.0, "zorder": 5},
}


def signal_and_gate(sig_a, sig_b):
    return ((sig_a + sig_b) == 2).astype(int)


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


def plot_fullperiod(equities, title_suffix, fname, start_date):
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
        ax.text(mid, ymax * 0.7, cname.split()[0], fontsize=7,
                ha="center", va="top", color="#c0392b", alpha=0.7, rotation=90)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(f"Cumulative Returns {title_suffix}\n"
                 f"(NDX 3x strategies vs SPMO 1x, ER=3.5%, lag=1, comm=0.2%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")


def plot_crisis_grid(equities, ndx):
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 11))
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

    # Hide unused subplot
    if len(CRISES) < nrows * ncols:
        for j in range(len(CRISES), nrows * ncols):
            axes[j].set_visible(False)

    fig.suptitle("SPMO vs Leveraged Strategies: Per-Crisis Comparison\n"
                 "(NDX 3x strategies vs SPMO 1x)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "spmo_crisis_grid.png", dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / 'spmo_crisis_grid.png'}")


def main():
    print("=" * 70)
    print("  SPMO vs Leveraged Strategies (since SPMO inception)")
    print("  NDX 3x | ER=3.5% | lag=1 | comm=0.2%")
    print("=" * 70)

    print("\n  Downloading data...")
    ndx = download("^NDX", start="1985-10-01")
    spmo = download("SPMO", start="2015-10-01")
    rf = download_ken_french_rf()

    spmo_start = spmo.index[0]
    # Skip first few days with zero volume (illiquid)
    spmo_valid = spmo[spmo.pct_change().abs() > 0]
    if len(spmo_valid) > 0:
        spmo_start = spmo_valid.index[0]

    print(f"  NDX:  {len(ndx)} days ({ndx.index[0].date()} -> {ndx.index[-1].date()})")
    print(f"  SPMO: {len(spmo)} days ({spmo.index[0].date()} -> {spmo.index[-1].date()})")
    print(f"  Comparison start: {spmo_start.date()}")

    # Build signals
    print("  Building signals...")
    sig_sym3 = signal_dual_ma(ndx, fast=3, slow=161)
    sig_p5   = signal_regime_switching_dual_ma(ndx, **P5_PARAMS)
    sig_and  = signal_and_gate(sig_p5, sig_sym3)

    # Run backtests
    print("  Running backtests...")

    eq_sym3 = run_lrs(ndx, sig_sym3, leverage=LEVERAGE,
                       expense_ratio=CALIBRATED_ER,
                       tbill_rate=rf, signal_lag=SIGNAL_LAG,
                       commission=COMMISSION)
    eq_p5   = run_lrs(ndx, sig_p5, leverage=LEVERAGE,
                       expense_ratio=CALIBRATED_ER,
                       tbill_rate=rf, signal_lag=SIGNAL_LAG,
                       commission=COMMISSION)
    eq_and  = run_lrs(ndx, sig_and, leverage=LEVERAGE,
                       expense_ratio=CALIBRATED_ER,
                       tbill_rate=rf, signal_lag=SIGNAL_LAG,
                       commission=COMMISSION)

    # SPMO equity (buy and hold, 1x, no leverage)
    eq_spmo = spmo / spmo.iloc[0]

    equities_full = {
        "SPMO B&H":    eq_spmo,
        "Sym(3,161)":  eq_sym3,
        "P5 Regime":   eq_p5,
        "AND":         eq_and,
    }

    # Align to common period
    start_str = str(spmo_start.date())

    strats_sig = {
        "Sym(3,161)": sig_sym3,
        "P5 Regime":  sig_p5,
        "AND":        sig_and,
    }

    # Metrics
    def print_metrics(title, start_date):
        print(f"\n{'=' * 115}")
        print(f"  {title}")
        print(f"{'=' * 115}")
        print(f"  {'Strategy':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MDD':>9} {'MDD_Entry':>10} {'Trades/Yr':>10}")
        print(f"  {'─' * 108}")

        for label in STRAT_NAMES:
            eq = equities_full[label]
            eq_sub = eq.loc[start_date:]
            if len(eq_sub) < 10:
                continue
            eq_norm = eq_sub / eq_sub.iloc[0]
            rf_sub = rf.reindex(eq_norm.index, method="ffill")
            rf_sc = rf_sub.mean() * 252
            m = calc_metrics(eq_norm, tbill_rate=rf_sc, rf_series=rf_sub)

            if label in strats_sig:
                sig_sub = strats_sig[label].loc[start_date:]
                tpy = signal_trades_per_year(sig_sub)
                mdd_e = _max_entry_drawdown(eq_norm, sig_sub, SIGNAL_LAG)
            else:
                tpy = 0.0
                mdd_e = m["MDD"]  # B&H: MDD_Entry == MDD

            print(f"  {label:<20} {m['CAGR']:>8.2%} {m['Sharpe']:>8.3f} "
                  f"{m['Sortino']:>9.3f} {m['MDD']:>9.2%} {mdd_e:>10.2%} {tpy:>10.1f}")

    print_metrics(f"SINCE SPMO INCEPTION ({spmo_start.date()})", start_date=start_str)
    print_metrics("SINCE 2020", start_date="2020-01-01")

    # Crisis comparison
    print(f"\n\n{'#' * 70}")
    print(f"  CRISIS-BY-CRISIS: MDD & MDD_Entry")
    print(f"{'#' * 70}")

    for cname, peak, trough, wend in CRISES:
        ndx_sub = ndx.loc[peak:trough]
        ndx_dd = ndx_sub.iloc[-1] / ndx_sub.iloc[0] - 1 if len(ndx_sub) > 1 else 0.0

        print(f"\n{'=' * 110}")
        print(f"  {cname}  (NDX {ndx_dd:+.0%})")
        print(f"{'=' * 110}")
        print(f"  {'Strategy':<20} {'Pk->Tr':>8} {'MaxDD':>8} {'MDD_Entry':>10} "
              f"{'WinRet':>8} {'D_Exit':>7} {'D_Cash':>7} {'Whip':>5}")
        print(f"  {'─' * 105}")

        for sname in STRAT_NAMES:
            eq = equities_full[sname]
            eq_w = eq.loc[peak:wend]
            if len(eq_w) < 2:
                continue
            eq_norm = eq_w / eq_w.iloc[0]
            maxdd = (eq_norm / eq_norm.cummax() - 1).min()

            if sname in strats_sig:
                sig = strats_sig[sname]
                sig_w = sig.loc[peak:wend]
                mdd_e = _max_entry_drawdown(eq_norm, sig_w, SIGNAL_LAG)
                exit_d, reentry_d = find_exit_reentry(sig_w, peak)
                whip = int(sig_w.diff().abs().sum())
            else:
                mdd_e = maxdd
                exit_d, reentry_d = None, None
                whip = 0

            eq_tr = eq.loc[peak:trough]
            pk2tr = eq_tr.iloc[-1] / eq_tr.iloc[0] - 1 if len(eq_tr) > 1 else 0.0
            win_ret = eq_norm.iloc[-1] - 1

            d_exit = len(ndx.loc[peak:exit_d]) - 1 if exit_d else None
            d_cash = len(ndx.loc[exit_d:reentry_d]) - 1 if exit_d and reentry_d else None

            if sname in strats_sig:
                de = f"{d_exit:>6d}d" if d_exit is not None else "  stay"
                dc = f"{d_cash:>6d}d" if d_cash is not None else "     -"
            else:
                de = "  hold"
                dc = "     -"

            print(f"  {sname:<20} {pk2tr:>+7.1%} {maxdd:>7.1%} {mdd_e:>10.2%} "
                  f"{win_ret:>+7.1%} {de} {dc} {whip:>5d}")

    # Charts
    print("\n  Generating charts...")
    plot_fullperiod(equities_full, f"Since SPMO Inception ({spmo_start.date()})",
                    "spmo_fullperiod.png", start_date=start_str)
    plot_crisis_grid(equities_full, ndx)

    print("\n  Done.")


if __name__ == "__main__":
    main()
