"""
Phase 3: Macro Regime Layer Test

Tests macro-regime-augmented signal vs P5 Regime (vol-only) across 9 crises.
Downloads FRED macro data (Fed Funds, Yield Curve, Credit Spread) and
diagnoses macro state at each crisis peak.

No Optuna optimization — fixed rules based on economic intuition.

Outputs:
  - Console: macro state at each crisis + strategy comparison table
  - output/macro_state_at_crises.csv
  - output/macro_regime_crisis_comparison.png
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
    download, download_ken_french_rf, download_fred_series,
    signal_dual_ma, signal_regime_switching_dual_ma,
    signal_macro_regime_dual_ma,
    run_lrs, calc_metrics, signal_trades_per_year,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Constants (Part 12 baseline) ──
CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002

# ── Crisis definitions (same as analyze_crises.py) ──
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

# ── P5 Regime parameters (from Phase 2 optimization) ──
P5_PARAMS = dict(fast_low=48, slow_low=323, fast_high=15, slow_high=229,
                 vol_lookback=49, vol_threshold_pct=57.3)


# ── Helpers ──

def find_exit_reentry(sig_sub, peak_date):
    """Find first exit (1->0) after peak, and first re-entry (0->1) after exit."""
    after_peak = sig_sub.loc[peak_date:]
    if len(after_peak) == 0:
        return None, None
    exits = after_peak[after_peak == 0]
    if len(exits) == 0:
        return None, None
    exit_date = exits.index[0]
    after_exit = after_peak.loc[exit_date:]
    reentries = after_exit[after_exit == 1]
    reentry_date = reentries.index[0] if len(reentries) > 0 else None
    return exit_date, reentry_date


def analyze_crisis(name, peak_date, trough_date, window_end,
                   strategies, ndx_price):
    """Analyze one crisis for all strategies. Returns list of dicts."""
    results = []

    ndx_sub = ndx_price.loc[peak_date:trough_date]
    ndx_dd = ndx_sub.iloc[-1] / ndx_sub.iloc[0] - 1 if len(ndx_sub) > 1 else 0.0

    for sname, (sig, eq) in strategies.items():
        eq_window = eq.loc[peak_date:window_end]
        if len(eq_window) < 2:
            continue
        eq_norm = eq_window / eq_window.iloc[0]
        maxdd = (eq_norm / eq_norm.cummax() - 1).min()

        eq_trough = eq.loc[peak_date:trough_date]
        pk_to_tr = eq_trough.iloc[-1] / eq_trough.iloc[0] - 1 if len(eq_trough) > 1 else 0.0

        window_return = eq_norm.iloc[-1] - 1

        exit_date, reentry_date = find_exit_reentry(
            sig.loc[peak_date:window_end], peak_date)

        days_to_exit = None
        if exit_date is not None:
            days_to_exit = len(ndx_price.loc[peak_date:exit_date]) - 1

        days_in_cash = None
        if exit_date is not None and reentry_date is not None:
            days_in_cash = len(ndx_price.loc[exit_date:reentry_date]) - 1

        # Whipsaws
        sig_window = sig.loc[peak_date:window_end]
        whipsaws = int(sig_window.diff().abs().sum())

        results.append({
            "crisis": name, "strategy": sname,
            "NDX_dd": ndx_dd, "pk_to_tr": pk_to_tr,
            "MaxDD": maxdd, "win_ret": window_return,
            "exit_date": exit_date.strftime("%Y-%m-%d") if exit_date else "stay",
            "reentry": reentry_date.strftime("%Y-%m-%d") if reentry_date else "N/A",
            "d_exit": days_to_exit, "d_cash": days_in_cash,
            "whipsaws": whipsaws,
        })
    return results


def print_crisis_table(crisis_name, crisis_results):
    """Print formatted comparison table for one crisis."""
    print(f"\n{'=' * 115}")
    print(f"  {crisis_name}")
    print(f"{'=' * 115}")
    print(f"  {'Strategy':<22} {'Pk->Tr':>8} {'MaxDD':>8} {'WinRet':>8} "
          f"{'Exit':>12} {'Re-entry':>12} {'D_Exit':>7} {'D_Cash':>7} {'Whip':>5}")
    print(f"  {'─' * 110}")

    for r in crisis_results:
        d_exit = f"{r['d_exit']:>6d}d" if r['d_exit'] is not None else "  stay"
        d_cash = f"{r['d_cash']:>6d}d" if r['d_cash'] is not None else "     -"
        print(f"  {r['strategy']:<22} {r['pk_to_tr']:>+7.1%} {r['MaxDD']:>7.1%} "
              f"{r['win_ret']:>+7.1%}  {r['exit_date']:>12} {r['reentry']:>12} "
              f"{d_exit} {d_cash} {r['whipsaws']:>5d}")


# ── Main ──

def main():
    print("=" * 70)
    print("  Phase 3: Macro Regime Layer Test")
    print("  NDX 3x | ER=3.5% | lag=1 | comm=0.2%")
    print("=" * 70)

    # ── 1. Download data ──
    print("\n  Downloading NDX data...")
    ndx_price = download("^NDX", start="1985-10-01")
    rf_series = download_ken_french_rf()
    print(f"  NDX: {len(ndx_price)} days ({ndx_price.index[0].date()} -> "
          f"{ndx_price.index[-1].date()})")

    print("\n  Downloading FRED macro data...")
    fed_funds = download_fred_series("DFF")       # Daily Fed Funds Rate
    yield_curve = download_fred_series("T10Y2Y")  # 10Y-2Y Spread
    credit_spread = download_fred_series("BAA10Y") # BAA-10Y Spread

    macro_ok = all(len(s) > 0 for s in [fed_funds, yield_curve, credit_spread])
    if not macro_ok:
        print("  [ERROR] FRED download failed — cannot run macro regime test.")
        return

    # ── 2. Diagnose macro state at each crisis peak ──
    print("\n" + "=" * 90)
    print("  MACRO STATE AT CRISIS PEAKS")
    print("=" * 90)
    print(f"  {'Crisis':<22} {'Fed Funds':>10} {'10Y-2Y':>10} {'BAA-10Y':>10} "
          f"{'YC Inv?':>8} {'Hiking?':>8} {'Cr Pct':>8} {'Diagnosis':<20}")
    print(f"  {'─' * 87}")

    macro_rows = []
    for cname, peak, trough, wend in CRISES:
        pk = pd.Timestamp(peak)

        # Get most recent value on or before peak
        ff_val = fed_funds.loc[:pk].iloc[-1] if len(fed_funds.loc[:pk]) > 0 else np.nan
        yc_val = yield_curve.loc[:pk].iloc[-1] if len(yield_curve.loc[:pk]) > 0 else np.nan
        cs_val = credit_spread.loc[:pk].iloc[-1] if len(credit_spread.loc[:pk]) > 0 else np.nan

        # Yield curve inverted?
        yc_inverted = (not np.isnan(yc_val)) and yc_val < 0

        # Fed hiking? (60d avg > 120d avg)
        ff_aligned = fed_funds.reindex(ndx_price.index, method="ffill")
        ff_at_peak = ff_aligned.loc[:pk]
        if len(ff_at_peak) >= 60:
            ff_60 = ff_at_peak.iloc[-60:].mean()
            ff_120 = ff_at_peak.iloc[-120:].mean() if len(ff_at_peak) >= 120 else ff_at_peak.mean()
            hiking = ff_60 > ff_120
        else:
            hiking = False

        # Credit spread percentile (expanding, up to peak)
        cs_aligned = credit_spread.reindex(ndx_price.index, method="ffill").loc[:pk]
        if len(cs_aligned) > 30:
            cs_pct = cs_aligned.rank(pct=True).iloc[-1] * 100
        else:
            cs_pct = np.nan

        # Diagnosis
        if not np.isnan(yc_val) and yc_inverted:
            if hiking:
                diag = "Tightening"
            else:
                diag = "YC Inverted"
        elif cs_pct >= 70:
            diag = "Credit Stress"
        elif hiking:
            diag = "Rate Hiking"
        else:
            diag = "Normal/Accommodative"

        ff_str = f"{ff_val:.2f}%" if not np.isnan(ff_val) else "N/A"
        yc_str = f"{yc_val:+.2f}%" if not np.isnan(yc_val) else "N/A"
        cs_str = f"{cs_val:.2f}%" if not np.isnan(cs_val) else "N/A"
        cp_str = f"{cs_pct:.0f}%" if not np.isnan(cs_pct) else "N/A"

        print(f"  {cname:<22} {ff_str:>10} {yc_str:>10} {cs_str:>10} "
              f"{'YES' if yc_inverted else 'no':>8} {'YES' if hiking else 'no':>8} "
              f"{cp_str:>8} {diag:<20}")

        macro_rows.append({
            "crisis": cname, "peak_date": peak,
            "fed_funds": ff_val, "yield_curve_10y2y": yc_val,
            "baa_10y_spread": cs_val,
            "yc_inverted": yc_inverted, "fed_hiking": hiking,
            "credit_pct": cs_pct, "diagnosis": diag,
        })

    # Save macro state CSV
    macro_df = pd.DataFrame(macro_rows)
    csv_path = OUT_DIR / "macro_state_at_crises.csv"
    macro_df.to_csv(csv_path, index=False)
    print(f"\n  -> saved {csv_path}")

    # ── 3. Build strategies ──
    print("\n  Building signals...")

    # Baseline: Sym(11,237)
    sig_sym = signal_dual_ma(ndx_price, fast=11, slow=237)

    # P5 Regime (vol-only)
    sig_p5 = signal_regime_switching_dual_ma(ndx_price, **P5_PARAMS)

    # Macro Regime: use P5's MA parameters + macro layers
    sig_macro = signal_macro_regime_dual_ma(
        ndx_price,
        fast_low=P5_PARAMS["fast_low"], slow_low=P5_PARAMS["slow_low"],
        fast_high=P5_PARAMS["fast_high"], slow_high=P5_PARAMS["slow_high"],
        vol_lookback=P5_PARAMS["vol_lookback"],
        vol_threshold_pct=P5_PARAMS["vol_threshold_pct"],
        fed_funds=fed_funds, yield_curve=yield_curve,
        credit_spread=credit_spread,
        credit_threshold_pct=70.0,
    )

    # Macro Regime variant 2: lower credit threshold (more sensitive)
    sig_macro_60 = signal_macro_regime_dual_ma(
        ndx_price,
        fast_low=P5_PARAMS["fast_low"], slow_low=P5_PARAMS["slow_low"],
        fast_high=P5_PARAMS["fast_high"], slow_high=P5_PARAMS["slow_high"],
        vol_lookback=P5_PARAMS["vol_lookback"],
        vol_threshold_pct=P5_PARAMS["vol_threshold_pct"],
        fed_funds=fed_funds, yield_curve=yield_curve,
        credit_spread=credit_spread,
        credit_threshold_pct=60.0,
    )

    # Run backtests
    print("  Running backtests...")
    strategies = {}
    for label, sig in [("Sym(11,237)", sig_sym),
                       ("P5 Regime", sig_p5),
                       ("Macro(cr70)", sig_macro),
                       ("Macro(cr60)", sig_macro_60)]:
        eq = run_lrs(ndx_price, sig, leverage=LEVERAGE,
                     expense_ratio=CALIBRATED_ER, tbill_rate=rf_series,
                     signal_lag=SIGNAL_LAG, commission=COMMISSION)
        strategies[label] = (sig, eq)

    # ── 4. Full-period metrics ──
    print("\n" + "=" * 95)
    print("  FULL PERIOD METRICS (NDX 3x, 1985-2025)")
    print("=" * 95)
    print(f"  {'Strategy':<22} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} "
          f"{'MDD':>9} {'Trades/Yr':>10}")
    print(f"  {'─' * 90}")

    rf_scalar = rf_series.mean() * 252
    for label, (sig, eq) in strategies.items():
        m = calc_metrics(eq, tbill_rate=rf_scalar, rf_series=rf_series)
        tpy = signal_trades_per_year(sig)
        print(f"  {label:<22} {m['CAGR']:>8.2%} {m['Sharpe']:>8.3f} "
              f"{m['Sortino']:>9.3f} {m['MDD']:>9.2%} {tpy:>10.1f}")

    # ── 5. Crisis-by-crisis comparison ──
    print("\n\n" + "#" * 70)
    print("  CRISIS-BY-CRISIS COMPARISON")
    print("#" * 70)

    all_results = []
    for cname, peak, trough, wend in CRISES:
        crisis_results = analyze_crisis(
            cname, peak, trough, wend, strategies, ndx_price)
        all_results.extend(crisis_results)
        print_crisis_table(cname, crisis_results)

    # ── 6. Summary: P5 vs Macro at key crises ──
    print("\n\n" + "=" * 80)
    print("  KEY COMPARISON: P5 Regime vs Macro Regime")
    print("  Focus: COVID, Liberation Day (V-shaped) vs Dot-com, GFC (prolonged)")
    print("=" * 80)

    focus_crises = ["2020 COVID", "2025 Liberation Day", "2000-02 Dot-com", "2007-09 GFC"]
    focus_strats = ["P5 Regime", "Macro(cr70)", "Macro(cr60)"]
    for cname in focus_crises:
        rows = [r for r in all_results if r["crisis"] == cname and r["strategy"] in focus_strats]
        if rows:
            print(f"\n  {cname}:")
            for r in rows:
                d_exit = f"{r['d_exit']}d" if r['d_exit'] is not None else "stay"
                print(f"    {r['strategy']:<22} MaxDD={r['MaxDD']:>7.1%}  "
                      f"WinRet={r['win_ret']:>+7.1%}  Exit={r['exit_date']}  "
                      f"D_exit={d_exit}")

    # ── 7. Chart: P5 vs Macro per-crisis equity grid ──
    print("\n  Generating chart...")

    STYLES = {
        "Sym(11,237)":  {"color": "#7f8c8d", "ls": "--", "lw": 1.5},
        "P5 Regime":    {"color": "#e74c3c", "ls": "-",  "lw": 2.0},
        "Macro(cr70)":  {"color": "#2980b9", "ls": "-",  "lw": 2.0},
        "Macro(cr60)":  {"color": "#27ae60", "ls": "--", "lw": 1.8},
    }

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (cname, peak, trough, wend) in enumerate(CRISES):
        ax = axes[idx]
        for sname, sty in STYLES.items():
            _, eq = strategies[sname]
            eq_sub = eq.loc[peak:wend]
            if len(eq_sub) < 2:
                continue
            eq_norm = eq_sub / eq_sub.iloc[0]
            days = np.arange(len(eq_norm))
            ax.plot(days, eq_norm.values, label=sname,
                    color=sty["color"], linestyle=sty["ls"],
                    linewidth=sty["lw"])

        trough_idx = len(ndx_price.loc[peak:trough]) - 1
        ax.axvline(trough_idx, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

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

    fig.suptitle("P5 Regime vs Macro Regime: Crisis Equity Comparison\n"
                 "(NDX 3x, ER=3.5%, lag=1, comm=0.2%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = "macro_regime_crisis_comparison.png"
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> saved {OUT_DIR / fname}")

    # ── 8. Save full results CSV ──
    results_df = pd.DataFrame(all_results)
    csv2 = OUT_DIR / "macro_regime_crisis_detail.csv"
    results_df.to_csv(csv2, index=False)
    print(f"  -> saved {csv2}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
