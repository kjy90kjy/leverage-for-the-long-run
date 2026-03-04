"""
Crisis-by-crisis strategy comparison analysis.

Compares 5 strategies across 9 major NDX crises (1987-2025):
  - Signal exit/re-entry timing
  - MaxDD, period return, recovery
  - Whipsaw count

Outputs: 4 PNGs + 1 CSV to output/
"""

import sys
import io
import warnings

# Force UTF-8 output on Windows
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
    signal_dual_ma, signal_vol_adaptive_dual_ma,
    signal_regime_switching_dual_ma, signal_vol_regime_adaptive_ma,
    run_lrs,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Constants (Part 12 baseline) ──
CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002


# ── Crisis definitions ──
# (name, peak_date, trough_date, window_end)
# window_end: recovery point or reasonable analysis end
CRISES = [
    ("1987 Black Monday",  "1987-10-02", "1987-12-04", "1988-06-30"),
    ("1998 LTCM",          "1998-07-20", "1998-10-08", "1999-03-31"),
    ("2000-02 Dot-com",    "2000-03-10", "2002-10-09", "2004-03-31"),
    ("2007-09 GFC",        "2007-10-31", "2009-03-09", "2011-03-31"),
    ("2018 Q4",            "2018-10-01", "2018-12-24", "2019-06-30"),
    ("2020 COVID",         "2020-02-19", "2020-03-23", "2021-03-31"),
    ("2022 Bear",          "2021-11-19", "2022-10-13", "2024-03-31"),
    ("2024 Aug Dip",       "2024-07-10", "2024-08-05", "2024-12-31"),
    ("2025 Liberation Day","2025-02-19", "2025-04-08", "2025-12-31"),
]

# ── Strategy definitions ──
STRATEGY_STYLES = {
    "Sym(11,237)":  {"color": "#7f8c8d", "ls": "--",  "lw": 1.8, "zorder": 2},
    "Sym(3,161)":   {"color": "#2ecc71", "ls": "--",  "lw": 1.8, "zorder": 3},
    "P4 VolAdapt":  {"color": "#3498db", "ls": "-",   "lw": 2.0, "zorder": 4},
    "P5 Regime":    {"color": "#e74c3c", "ls": "-",   "lw": 2.0, "zorder": 5},
    "P6 VolRegime": {"color": "#9b59b6", "ls": (0,(4,1.5)), "lw": 2.0, "zorder": 6},
}
STRATEGY_COLORS = {k: v["color"] for k, v in STRATEGY_STYLES.items()}


def build_signals(price):
    """Generate signals for all 5 strategies."""
    return {
        "Sym(11,237)": signal_dual_ma(price, fast=11, slow=237),
        "Sym(3,161)":  signal_dual_ma(price, fast=3, slow=161),
        "P4 VolAdapt": signal_vol_adaptive_dual_ma(
            price, base_fast=45, base_slow=158, vol_lookback=73, vol_scale=0.396),
        "P5 Regime":   signal_regime_switching_dual_ma(
            price, fast_low=48, slow_low=323, fast_high=15, slow_high=229,
            vol_lookback=49, vol_threshold_pct=57.3),
        "P6 VolRegime": signal_vol_regime_adaptive_ma(
            price, base_fast_low=49, base_slow_low=143,
            base_fast_high=13, base_slow_high=204,
            vol_lookback=84, vol_threshold_pct=49.1, vol_scale=0.588),
    }


def build_equities(price, signals, rf_series):
    """Run LRS for all strategies, return equity curves."""
    equities = {}
    for name, sig in signals.items():
        equities[name] = run_lrs(price, sig, leverage=LEVERAGE,
                                  expense_ratio=CALIBRATED_ER,
                                  tbill_rate=rf_series,
                                  signal_lag=SIGNAL_LAG,
                                  commission=COMMISSION)
    return equities


def find_exit_reentry(sig_sub, peak_date):
    """Find first exit (1->0) after peak, and first re-entry (0->1) after exit.

    Returns (exit_date, reentry_date) or (None, None).
    """
    after_peak = sig_sub.loc[peak_date:]
    if len(after_peak) == 0:
        return None, None

    # Find first 0 (exit)
    exits = after_peak[after_peak == 0]
    if len(exits) == 0:
        return None, None  # never exited

    exit_date = exits.index[0]

    # Find first 1 after exit (re-entry)
    after_exit = after_peak.loc[exit_date:]
    reentries = after_exit[after_exit == 1]
    reentry_date = reentries.index[0] if len(reentries) > 0 else None

    return exit_date, reentry_date


def find_recovery_date(eq_sub, peak_date):
    """Find date when equity recovers to peak level after trough.

    Returns (recovery_date, trading_days_from_peak) or (None, None).
    """
    peak_value = eq_sub.loc[:peak_date].iloc[-1]
    after_peak = eq_sub.loc[peak_date:]
    # Find first date where equity >= peak value (after initial drop)
    # Skip first day (peak itself)
    dropped = False
    for i, (date, val) in enumerate(after_peak.items()):
        if val < peak_value:
            dropped = True
        if dropped and val >= peak_value:
            days = i
            return date, days
    return None, None


def count_whipsaws(sig_sub):
    """Count number of signal flips in a window."""
    return int(sig_sub.diff().abs().sum())


def analyze_single_crisis(name, peak_date, trough_date, window_end,
                           signals, equities, ndx_price):
    """Analyze one crisis for all strategies. Returns list of dicts."""
    results = []

    # NDX drawdown
    ndx_sub = ndx_price.loc[peak_date:trough_date]
    if len(ndx_sub) > 1:
        ndx_dd = ndx_sub.iloc[-1] / ndx_sub.iloc[0] - 1
    else:
        ndx_dd = 0.0

    for sname in signals:
        sig = signals[sname]
        eq = equities[sname]

        sig_window = sig.loc[peak_date:window_end]
        eq_window = eq.loc[peak_date:window_end]

        if len(eq_window) < 2:
            continue

        # Normalize equity to 1.0 at peak
        eq_norm = eq_window / eq_window.iloc[0]

        # MaxDD in window
        maxdd = (eq_norm / eq_norm.cummax() - 1).min()

        # Period return (peak to trough)
        eq_trough = eq.loc[peak_date:trough_date]
        if len(eq_trough) > 1:
            peak_to_trough = eq_trough.iloc[-1] / eq_trough.iloc[0] - 1
        else:
            peak_to_trough = 0.0

        # Full window return
        window_return = eq_norm.iloc[-1] - 1

        # Signal at peak
        sig_at_peak = int(sig.loc[:peak_date].iloc[-1]) if peak_date in sig.index or len(sig.loc[:peak_date]) > 0 else -1

        # Exit / re-entry
        exit_date, reentry_date = find_exit_reentry(sig.loc[peak_date:window_end], peak_date)

        # Days to exit (from peak)
        if exit_date is not None:
            days_to_exit = len(ndx_price.loc[peak_date:exit_date]) - 1
        else:
            days_to_exit = None

        # Days in cash
        if exit_date is not None and reentry_date is not None:
            days_in_cash = len(ndx_price.loc[exit_date:reentry_date]) - 1
        else:
            days_in_cash = None

        # Recovery
        recovery_date, days_to_recovery = find_recovery_date(eq, peak_date)

        # Whipsaws
        whipsaws = count_whipsaws(sig_window)

        results.append({
            "crisis": name,
            "strategy": sname,
            "NDX_drawdown": ndx_dd,
            "peak_to_trough": peak_to_trough,
            "window_MaxDD": maxdd,
            "window_return": window_return,
            "sig_at_peak": sig_at_peak,
            "exit_date": exit_date.strftime("%Y-%m-%d") if exit_date else "no exit",
            "reentry_date": reentry_date.strftime("%Y-%m-%d") if reentry_date else "N/A",
            "days_to_exit": days_to_exit,
            "days_in_cash": days_in_cash,
            "recovery_date": recovery_date.strftime("%Y-%m-%d") if recovery_date else "not recovered",
            "days_to_recovery": days_to_recovery,
            "whipsaws": whipsaws,
        })

    return results


# ── Chart 1: Per-crisis equity grid (3x3) ──

def plot_crisis_equity_grid(all_results, signals, equities, ndx_price):
    """3x3 subplot grid: normalized equity during each crisis."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (cname, peak, trough, wend) in enumerate(CRISES):
        ax = axes[idx]

        for sname, sty in STRATEGY_STYLES.items():
            eq = equities[sname]
            eq_sub = eq.loc[peak:wend]
            if len(eq_sub) < 2:
                continue
            eq_norm = eq_sub / eq_sub.iloc[0]
            # Convert to trading days from peak
            days = np.arange(len(eq_norm))
            ax.plot(days, eq_norm.values, label=sname,
                    color=sty["color"], linestyle=sty["ls"],
                    linewidth=sty["lw"], zorder=sty["zorder"])

        # Mark trough date
        trough_idx = len(ndx_price.loc[peak:trough]) - 1
        ax.axvline(trough_idx, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

        # NDX drawdown annotation
        ndx_sub = ndx_price.loc[peak:trough]
        if len(ndx_sub) > 1:
            ndx_dd = ndx_sub.iloc[-1] / ndx_sub.iloc[0] - 1
            ax.set_title(f"{cname}\n(NDX {ndx_dd:+.0%})", fontsize=11, fontweight="bold")
        else:
            ax.set_title(cname, fontsize=11, fontweight="bold")

        ax.axhline(1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Trading days from peak", fontsize=8)
        ax.set_ylabel("Equity (peak=1.0)", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

        if idx == 0:
            ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Strategy Equity During Major Crises (NDX 3x, ER=3.5%, lag=1)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "crisis_equity_grid.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")


# ── Chart 2: MaxDD heatmap ──

def plot_maxdd_heatmap(df):
    """Heatmap: rows=crises, columns=strategies, cells=MaxDD."""
    pivot = df.pivot_table(index="crisis", columns="strategy",
                           values="window_MaxDD", aggfunc="first")
    # Reorder rows by crisis order
    crisis_order = [c[0] for c in CRISES]
    strat_order = list(STRATEGY_COLORS.keys())
    pivot = pivot.reindex(index=crisis_order, columns=strat_order)

    fig, ax = plt.subplots(figsize=(12, 8))
    data = pivot.values * 100  # to percent

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-100, vmax=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("MaxDD (%)", fontsize=11)

    # Cell annotations
    for i in range(len(crisis_order)):
        for j in range(len(strat_order)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val < -60 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(strat_order)))
    ax.set_xticklabels(strat_order, fontsize=10, rotation=15, ha="right")
    ax.set_yticks(range(len(crisis_order)))
    ax.set_yticklabels(crisis_order, fontsize=10)
    ax.set_title("MaxDD by Crisis and Strategy (%)", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fname = "crisis_maxdd_heatmap.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")


# ── Chart 3: Exit/recovery timeline ──

def plot_timeline(df):
    """Horizontal bar chart: days to exit, days in cash, days to recovery."""
    crisis_order = [c[0] for c in CRISES]
    strat_order = list(STRATEGY_COLORS.keys())
    n_strats = len(strat_order)
    n_crises = len(crisis_order)

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    metrics = [
        ("days_to_exit", "Days to Exit (peak -> signal=0)", "Exit Speed"),
        ("days_in_cash", "Days in Cash (exit -> re-entry)", "Cash Duration"),
        ("days_to_recovery", "Days to Recovery (peak -> equity recovers)", "Recovery"),
    ]

    for ax, (col, title, short) in zip(axes, metrics):
        y_pos = np.arange(n_crises)
        bar_height = 0.15

        for si, sname in enumerate(strat_order):
            vals = []
            for cname in crisis_order:
                row = df[(df["crisis"] == cname) & (df["strategy"] == sname)]
                if len(row) > 0 and row.iloc[0][col] is not None and not pd.isna(row.iloc[0][col]):
                    vals.append(row.iloc[0][col])
                else:
                    vals.append(0)

            offset = (si - n_strats / 2 + 0.5) * bar_height
            bars = ax.barh(y_pos + offset, vals, height=bar_height,
                          color=STRATEGY_COLORS[sname], label=sname if ax == axes[0] else "",
                          alpha=0.85, edgecolor="white", linewidth=0.3)

            # Value labels for non-zero bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                            f"{int(val)}", va="center", fontsize=7, color="#555")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(crisis_order, fontsize=9)
        ax.set_xlabel("Trading Days", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Crisis Response Timing: Exit, Cash, and Recovery",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "crisis_timeline.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")


# ── Chart 4: Full period with crisis shading ──

def plot_fullperiod_shaded(equities, ndx_price):
    """Full cumulative returns on log scale with crisis periods shaded."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Normalize all equity curves from a common start (1987-09-23 warmup end approx)
    common_start = "1988-01-01"
    for sname, sty in STRATEGY_STYLES.items():
        eq = equities[sname]
        eq_sub = eq.loc[common_start:]
        if len(eq_sub) > 0:
            eq_norm = eq_sub / eq_sub.iloc[0]
            ax.plot(eq_norm.index, eq_norm.values, label=sname,
                    color=sty["color"], linestyle=sty["ls"],
                    linewidth=sty["lw"], zorder=sty["zorder"])

    # Shade crisis periods
    for cname, peak, trough, wend in CRISES:
        ax.axvspan(pd.Timestamp(peak), pd.Timestamp(trough),
                   alpha=0.12, color="#e74c3c", zorder=0)
        # Label at top
        mid = pd.Timestamp(peak) + (pd.Timestamp(trough) - pd.Timestamp(peak)) / 2
        ymax = ax.get_ylim()[1]
        ax.text(mid, ymax * 0.7, cname.split()[0], fontsize=6,
                ha="center", va="top", color="#c0392b", alpha=0.7, rotation=90)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Full Period Cumulative Returns with Crisis Periods (NDX 3x, 1988-2025)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "crisis_fullperiod_shaded.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")


# ── Console output ──

def print_crisis_table(crisis_name, crisis_results):
    """Print formatted table for one crisis."""
    print(f"\n{'=' * 110}")
    print(f"  {crisis_name}")
    print(f"{'=' * 110}")
    print(f"  {'Strategy':<18} {'Pk->Tr':>8} {'MaxDD':>8} {'WinRet':>8} "
          f"{'Exit':>12} {'Re-entry':>12} {'D_Exit':>7} {'D_Cash':>7} "
          f"{'Recovery':>12} {'D_Recv':>7} {'Whip':>5}")
    print(f"  {'─' * 105}")

    for r in crisis_results:
        d_exit = f"{r['days_to_exit']:>6d}d" if r['days_to_exit'] is not None else "  stay"
        d_cash = f"{r['days_in_cash']:>6d}d" if r['days_in_cash'] is not None else "     -"
        d_recv = f"{r['days_to_recovery']:>6d}d" if r['days_to_recovery'] is not None else "   N/R"
        print(f"  {r['strategy']:<18} {r['peak_to_trough']:>+7.1%} {r['window_MaxDD']:>7.1%} "
              f"{r['window_return']:>+7.1%}  {r['exit_date']:>12} {r['reentry_date']:>12} "
              f"{d_exit} {d_cash}  {r['recovery_date']:>12} {d_recv} {r['whipsaws']:>5d}")


# ── Main ──

def main():
    print("=" * 70)
    print("  Crisis-by-Crisis Strategy Comparison")
    print("  NDX 3x | ER=3.5% | lag=1 | comm=0.2%")
    print("=" * 70)

    # Download data
    print("\n  Downloading data...")
    ndx_price = download("^NDX", start="1985-10-01")
    rf_series = download_ken_french_rf()
    print(f"  NDX: {len(ndx_price)} days ({ndx_price.index[0].date()} -> {ndx_price.index[-1].date()})")

    # Build signals & equity curves
    print("  Building signals...")
    signals = build_signals(ndx_price)
    print("  Running backtests...")
    equities = build_equities(ndx_price, signals, rf_series)

    # Analyze each crisis
    all_results = []
    for cname, peak, trough, wend in CRISES:
        crisis_results = analyze_single_crisis(
            cname, peak, trough, wend, signals, equities, ndx_price)
        all_results.extend(crisis_results)
        print_crisis_table(cname, crisis_results)

    # Build DataFrame
    df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = OUT_DIR / "crisis_analysis_detail.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  -> saved {csv_path}")

    # Charts
    print("\n  Generating charts...")
    plot_crisis_equity_grid(all_results, signals, equities, ndx_price)
    plot_maxdd_heatmap(df)
    plot_timeline(df)
    plot_fullperiod_shaded(equities, ndx_price)

    print("\n  Done.")


if __name__ == "__main__":
    main()
