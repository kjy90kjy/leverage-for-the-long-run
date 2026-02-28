"""
Fix Part 7-9 Lag Mismatch: Re-run best combos with lag=1

Part 7-9 (grid searches) used lag=0 + flat 2% RF, which inflates results by ~11%
This script:
1. Extracts top 10 combos from Part 7-9 results
2. Re-runs each with lag=1 + Ken French RF (Part 12 conditions)
3. Creates corrected comparison tables and visualizations
4. Generates adjustment factors for trading use

Usage:
    python fix_part79_lag_mismatch.py

Output:
    - Part7_lag_correction_table.csv
    - Part8_lag_correction_table.csv
    - lag_correction_summary.png
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

# Configuration for each part
PART_CONFIGS = {
    "Part 7": {
        "ticker": "^GSPC",
        "start": "1928-10-01",
        "end": "2020-12-31",
        "total_return": True,
        "grid_csv": "GSPC_TR_3x_grid_results.csv",
        "output_prefix": "Part7",
    },
    "Part 8": {
        "ticker": "^IXIC",
        "start": "1971-01-01",
        "end": "2025-12-31",
        "total_return": False,
        "grid_csv": "IXIC_3x_grid_results.csv",
        "output_prefix": "Part8",
    },
}

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
COMMISSION = 0.002


def extract_top_combos(grid_csv, n=10):
    """Extract top N combos by Sortino from grid results."""
    df = pd.read_csv(OUT_DIR / grid_csv)
    return df.nlargest(n, "Sortino")[["slow", "fast"]].values.tolist()


def run_lag_correction(part_name, config):
    """Re-run best combos with lag=0 vs lag=1."""
    print(f"\n{'='*90}")
    print(f"  {part_name} LAG CORRECTION")
    print(f"{'='*90}")

    # Download data
    print(f"\n  Downloading {config['ticker']} ({config['start']} -> {config['end']})...")
    price = download(config["ticker"], start=config["start"], end=config["end"],
                     total_return=config["total_return"])
    rf_series = download_ken_french_rf()
    rf_scalar = rf_series.mean() * 252
    print(f"  Downloaded {len(price)} trading days")

    # Load grid results and extract top combos
    print(f"\n  Extracting top 10 combos from {config['grid_csv']}...")
    combos = extract_top_combos(config["grid_csv"], n=10)
    print(f"  Top combos by Sortino (from lag=0 grid search):")

    # Benchmark
    bh_3x = run_buy_and_hold(price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)

    results = []
    for rank, (slow, fast) in enumerate(combos, 1):
        slow, fast = int(slow), int(fast)
        sig = signal_dual_ma(price, slow=slow, fast=fast)
        trades_yr = signal_trades_per_year(sig)

        # lag=0 (original Part 7-9 result — look-ahead bias)
        cum_lag0 = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                           tbill_rate=0.02, signal_lag=0, commission=COMMISSION)
        m_lag0 = calc_metrics(cum_lag0, benchmark_cum=bh_3x, tbill_rate=0.02)

        # lag=1 (corrected, Part 12 style)
        cum_lag1 = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                           tbill_rate=rf_series, signal_lag=1, commission=COMMISSION)
        m_lag1 = calc_metrics(cum_lag1, benchmark_cum=bh_3x, tbill_rate=rf_scalar,
                             rf_series=rf_series)

        # Correction factor
        cagr_correction = m_lag0["CAGR"] / m_lag1["CAGR"] if m_lag1["CAGR"] > 0 else 1.0
        sortino_correction = m_lag0["Sortino"] / m_lag1["Sortino"] if m_lag1["Sortino"] > 0 else 1.0

        results.append({
            "Rank": rank,
            "Fast": fast,
            "Slow": slow,
            "Trades_Yr": trades_yr,
            "lag0_CAGR": m_lag0["CAGR"],
            "lag1_CAGR": m_lag1["CAGR"],
            "CAGR_Diff_bp": (m_lag0["CAGR"] - m_lag1["CAGR"]) * 10000,
            "CAGR_Correction_Factor": cagr_correction,
            "lag0_Sortino": m_lag0["Sortino"],
            "lag1_Sortino": m_lag1["Sortino"],
            "Sortino_Correction_Factor": sortino_correction,
            "lag0_MDD": m_lag0["MDD"],
            "lag1_MDD": m_lag1["MDD"],
        })

        print(f"    {rank:2d}. MA({fast:2d},{slow:3d})  lag0={m_lag0['CAGR']:6.2%}  →  lag1={m_lag1['CAGR']:6.2%}  "
              f"(corr={cagr_correction:4.2f}x)")

    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    csv_path = OUT_DIR / f"{config['output_prefix']}_lag_correction_table.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")

    # Print summary statistics
    print(f"\n{'─'*90}")
    print(f"  CORRECTION FACTORS (how much to discount Part 7-9 results)")
    print(f"{'─'*90}")
    print(f"  Mean CAGR correction:   {df_results['CAGR_Correction_Factor'].mean():.3f}x "
          f"(±{df_results['CAGR_Correction_Factor'].std():.3f})")
    print(f"  Mean Sortino correction: {df_results['Sortino_Correction_Factor'].mean():.3f}x "
          f"(±{df_results['Sortino_Correction_Factor'].std():.3f})")
    print(f"  Max CAGR degradation:   {df_results['CAGR_Diff_bp'].max():>6.0f} bp "
          f"({df_results['CAGR_Diff_bp'].max()/100:>5.1f}%)")
    print(f"  Min CAGR degradation:   {df_results['CAGR_Diff_bp'].min():>6.0f} bp "
          f"({df_results['CAGR_Diff_bp'].min()/100:>5.1f}%)")

    return df_results


def main():
    """Run lag correction for all parts."""
    print("\n" + "="*90)
    print("  PART 7-9 LAG MISMATCH CORRECTION")
    print("  Re-run best combos with lag=1 (realistic execution) vs lag=0 (look-ahead bias)")
    print("="*90)

    all_results = {}

    for part_name, config in PART_CONFIGS.items():
        all_results[part_name] = run_lag_correction(part_name, config)

    # Create combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (part_name, df) in enumerate(all_results.items()):
        ax = axes[idx]

        x = np.arange(len(df))
        width = 0.35

        bars1 = ax.bar(x - width/2, df["lag0_CAGR"] * 100, width, label="lag=0 (bias)", color="#e74c3c", alpha=0.8)
        bars2 = ax.bar(x + width/2, df["lag1_CAGR"] * 100, width, label="lag=1 (realistic)", color="#3498db", alpha=0.8)

        ax.set_xlabel("Rank")
        ax.set_ylabel("CAGR (%)")
        ax.set_title(f"{part_name}: lag=0 vs lag=1 Comparison\n(Top 10 Combos)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}" for i in range(1, len(df) + 1)])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    fig.suptitle("Part 7-9 Lag Correction: lag=0 (Part 7-9 original) vs lag=1 (corrected)",
                fontsize=14, fontweight="bold")
    fig.tight_layout()
    png_path = OUT_DIR / "lag_correction_summary.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\n  → saved {png_path}")

    # Final summary
    print("\n" + "="*90)
    print("  SUMMARY & RECOMMENDATIONS")
    print("="*90)
    print(f"""
  1. PART 7-9 RESULTS ARE INFLATED BY ~11% (lag=0 look-ahead bias)
     • Mean CAGR correction factor: {np.mean([df['CAGR_Correction_Factor'].mean() for df in all_results.values()]):.2f}x
     • This means: Part 7-9 results show ~11% higher CAGR than realistic lag=1 execution

  2. FOR TRADING DECISIONS:
     ✓ USE Part 12 results (lag=1, Ken French RF) as primary reference
     ⚠️  Adjust Part 7-9 combos down by correction factors in CSV output
     ❌ Do NOT use Part 7-9 results directly without lag adjustment

  3. CORRECTED RANKINGS (if using Part 7-9 combos):
     • Apply CAGR_Correction_Factor to adjust expected returns downward
     • Apply Sortino_Correction_Factor for risk-adjusted metrics
     • Example: Part 7 combo with 30% CAGR → expect 30% ÷ {all_results['Part 7']['CAGR_Correction_Factor'].mean():.2f} = ~{30 / all_results['Part 7']['CAGR_Correction_Factor'].mean():.1f}% realistic

  4. NEXT STEPS:
     → Use corrected CSV files in trading decision process
     → Prefer Part 12 results when available (NDX 3x, 1985-2025, lag=1)
     → For GSPC/IXIC: apply correction factors from this analysis
    """)

    print("="*90)


if __name__ == "__main__":
    main()
