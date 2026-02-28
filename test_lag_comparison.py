"""
Lag Comparison Test: Quantify impact of look-ahead bias (lag=0 vs lag=1)

Tests whether same-day execution (lag=0, look-ahead bias) significantly outperforms
next-day execution (lag=1, realistic trading) to validate if Part 7-9 results are inflated.

Usage:
    python test_lag_comparison.py

Output:
    - Console: Formatted comparison table (% differences)
    - PNG: Cumulative curves (2 columns × N rows subplots)
    - CSV: Full results
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
    signal_dual_ma,
    run_lrs, run_buy_and_hold, calc_metrics, signal_trades_per_year,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Config (Part 12 conditions) ──
TICKER = "^NDX"
START = "1985-10-01"
END = "2025-12-31"
CALIBRATED_ER = 0.035
LEVERAGE = 3.0
COMMISSION = 0.002

# ── Test MA combinations ──
# eulb baseline, grid best regions, and overall best from regime grid
MA_COMBOS = [
    (3, 161, "eulb baseline"),
    (7, 60, "grid region A (volatile)"),
    (20, 315, "grid region B (calm)"),
    (10, 200, "symmetric baseline"),
]


def run_lag_test():
    """Compare lag=0 vs lag=1 for each MA combo."""
    print("\n" + "=" * 80)
    print("  LAG COMPARISON TEST: look-ahead bias quantification")
    print("=" * 80)

    # Download data
    print(f"\n  Downloading {TICKER} ({START} -> {END})...")
    price = download(TICKER, start=START, end=END)
    rf_series = download_ken_french_rf()
    rf_scalar = rf_series.mean() * 252

    print(f"  Downloaded {len(price)} trading days")

    # Buy & Hold 3x baseline
    bh_3x = run_buy_and_hold(price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)

    results = []
    curves = {}

    for fast, slow, label in MA_COMBOS:
        print(f"\n  Testing MA({fast}, {slow}): {label}")
        sig = signal_dual_ma(price, slow=slow, fast=fast)
        trades_yr = signal_trades_per_year(sig)

        # lag=0 (look-ahead bias, same-day execution)
        lrs_lag0 = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                           tbill_rate=rf_series, signal_lag=0, commission=COMMISSION)
        m_lag0 = calc_metrics(lrs_lag0, benchmark_cum=bh_3x,
                              tbill_rate=rf_scalar, rf_series=rf_series)

        # lag=1 (realistic, next-day execution)
        lrs_lag1 = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                           tbill_rate=rf_series, signal_lag=1, commission=COMMISSION)
        m_lag1 = calc_metrics(lrs_lag1, benchmark_cum=bh_3x,
                              tbill_rate=rf_scalar, rf_series=rf_series)

        # Calculate differences
        cagr_diff = m_lag0["CAGR"] - m_lag1["CAGR"]
        sharpe_diff = m_lag0["Sharpe"] - m_lag1["Sharpe"]
        sortino_diff = m_lag0["Sortino"] - m_lag1["Sortino"]
        mdd_diff = m_lag0["MDD"] - m_lag1["MDD"]

        results.append({
            "MA": f"({fast},{slow})",
            "Label": label,
            "CAGR_lag0": m_lag0["CAGR"],
            "CAGR_lag1": m_lag1["CAGR"],
            "CAGR_diff_bp": cagr_diff * 10000,  # basis points
            "Sharpe_lag0": m_lag0["Sharpe"],
            "Sharpe_lag1": m_lag1["Sharpe"],
            "Sharpe_diff": sharpe_diff,
            "Sortino_lag0": m_lag0["Sortino"],
            "Sortino_lag1": m_lag1["Sortino"],
            "Sortino_diff": sortino_diff,
            "MDD_lag0": m_lag0["MDD"],
            "MDD_lag1": m_lag1["MDD"],
            "MDD_diff_bp": mdd_diff * 10000,
            "Trades_yr": trades_yr,
        })

        # Store curves for plotting
        curves[f"({fast},{slow}) lag=0"] = lrs_lag0
        curves[f"({fast},{slow}) lag=1"] = lrs_lag1

    # Print comparison table
    print("\n" + "=" * 100)
    print("  COMPARISON TABLE: lag=0 vs lag=1")
    print("=" * 100)
    print(f"  {'MA Combo':<12} {'Label':<25} {'CAGR_0':<10} {'CAGR_1':<10} {'Diff(bp)':<10} "
          f"{'Sharpe_0':<10} {'Sharpe_1':<10} {'S_Diff':<10} {'Trades/yr':<10}")
    print("  " + "─" * 96)

    for r in results:
        print(f"  {r['MA']:<12} {r['Label']:<25} {r['CAGR_lag0']:>9.2%} {r['CAGR_lag1']:>9.2%} "
              f"{r['CAGR_diff_bp']:>9.0f} "
              f"{r['Sharpe_lag0']:>9.3f} {r['Sharpe_lag1']:>9.3f} "
              f"{r['Sharpe_diff']:>9.3f} {r['Trades_yr']:>9.1f}")

    print("\n" + "=" * 100)
    print("  INTERPRETATION")
    print("=" * 100)
    max_cagr_diff = max(abs(r["CAGR_diff_bp"]) for r in results)
    if max_cagr_diff > 500:  # >5%
        print(f"  ⚠️  SIGNIFICANT look-ahead bias detected (max {max_cagr_diff:.0f} bp)")
        print(f"  → Part 7-9 results (lag=0) may be inflated by {max_cagr_diff/100:.1f}%")
    else:
        print(f"  ✓ Minimal look-ahead bias (max {max_cagr_diff:.0f} bp)")
        print(f"  → Part 7-9 and Part 12 results reasonably comparable")

    # Save CSV
    df_results = pd.DataFrame(results)
    csv_path = OUT_DIR / "lag_comparison_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")

    # Plot cumulative curves
    n_combos = len(MA_COMBOS)
    fig, axes = plt.subplots(n_combos, 2, figsize=(14, 5 * n_combos))
    if n_combos == 1:
        axes = axes.reshape(1, -1)

    for idx, (fast, slow, label) in enumerate(MA_COMBOS):
        # lag=0
        ax = axes[idx, 0]
        c_lag0 = curves[f"({fast},{slow}) lag=0"]
        c_lag1 = curves[f"({fast},{slow}) lag=1"]
        ax.plot(c_lag0.index, c_lag0.values, label="lag=0 (look-ahead)", linewidth=1.5, color="#e74c3c")
        ax.plot(c_lag1.index, c_lag1.values, label="lag=1 (realistic)", linewidth=1.5, color="#3498db")
        ax.set_yscale("log")
        ax.set_title(f"MA({fast}, {slow}): {label}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Growth of $1 (log)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

        # Outperformance ratio
        ax2 = axes[idx, 1]
        ratio = c_lag0 / c_lag1
        ax2.plot(ratio.index, ratio.values, label="lag=0 / lag=1", linewidth=1.5, color="#2ecc71")
        ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.set_title(f"Outperformance Ratio: lag=0 / lag=1", fontsize=12)
        ax2.set_ylabel("Cumulative Outperformance (ratio)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Add text annotation
        final_ratio = ratio.iloc[-1]
        ax2.text(0.98, 0.05, f"Final: {final_ratio:.2%} better with lag=0",
                transform=ax2.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    fig.suptitle("Lag Comparison: lag=0 (look-ahead bias) vs lag=1 (realistic)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    png_path = OUT_DIR / "lag_comparison_curves.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {png_path}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    run_lag_test()
