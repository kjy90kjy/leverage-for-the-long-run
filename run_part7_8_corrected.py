"""
Priority 4: Re-run Part 7-9 with lag=1 + Ken French RF (corrected conditions)

This script re-runs Part 7-9 grid searches with production-standard conditions:
  - lag=1 (next-day execution, realistic)
  - Ken French daily risk-free rates (instead of flat 2%)
  - 3x leverage (consistent with Part 12)
  - commission=0.2% (consistent with Part 12)

Compares results to original Part 7-9 (lag=0, flat 2% RF) to validate correction factors
from fix_part79_lag_mismatch.py.

Usage:
    python run_part7_8_corrected.py [part]

    part: 7 or 8 (default: both)

    python run_part7_8_corrected.py       # Runs both Part 7 & 8 (~40 min total)
    python run_part7_8_corrected.py 7    # Runs Part 7 only (~20 min)
    python run_part7_8_corrected.py 8    # Runs Part 8 only (~20 min)

Output:
    - output/Part7_corrected_grid_results.csv (slow × fast + metrics)
    - output/Part8_corrected_grid_results.csv
    - output/Part7_vs_Part7_corrected.png (comparison chart)
    - output/Part8_vs_Part8_corrected.png
    - output/Part7_8_correction_validation.csv (top 10 comparison)
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
from pathlib import Path
import time

from leverage_rotation import (
    download, download_ken_french_rf,
    signal_dual_ma,
    run_lrs, run_buy_and_hold, calc_metrics, signal_trades_per_year,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
COMMISSION = 0.002

# Part configurations
PART_CONFIGS = {
    7: {
        "ticker": "^GSPC",
        "start": "1928-10-01",
        "end": "2020-12-31",
        "total_return": True,
        "grid_csv": "GSPC_TR_3x_grid_results.csv",
        "output_prefix": "Part7",
    },
    8: {
        "ticker": "^IXIC",
        "start": "1971-01-01",
        "end": "2025-12-31",
        "total_return": False,
        "grid_csv": "IXIC_3x_grid_results.csv",
        "output_prefix": "Part8",
    },
}

# Grid ranges (same as original Part 7-9)
SLOW_RANGE = range(50, 351, 3)  # 50-350 step 3
FAST_RANGE = range(2, 51)       # 2-50 step 1


def run_corrected_grid(part_num, config):
    """Re-run grid search with lag=1 + Ken French RF."""
    print(f"\n{'='*100}")
    print(f"  PART {part_num} GRID SEARCH (CORRECTED: lag=1 + Ken French RF)")
    print(f"  Ticker: {config['ticker']}, Period: {config['start']} → {config['end']}")
    print(f"{'='*100}")

    # Download data
    print(f"\n  [1/4] Downloading {config['ticker']}...")
    start_time = time.time()
    price = download(config["ticker"], start=config["start"], end=config["end"],
                     total_return=config["total_return"])
    rf_series = download_ken_french_rf()
    print(f"  ✓ Downloaded {len(price)} trading days ({time.time()-start_time:.1f}s)")

    # Benchmark
    print(f"\n  [2/4] Computing buy & hold benchmark...")
    bh_3x = run_buy_and_hold(price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)

    # Grid search
    print(f"\n  [3/4] Running grid search ({len(FAST_RANGE)} × {len(SLOW_RANGE)} = {len(FAST_RANGE)*len(SLOW_RANGE)} combos)...")
    results = []
    start_time = time.time()

    for i, slow in enumerate(SLOW_RANGE):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                est_total = elapsed * len(SLOW_RANGE) / i
                est_remain = est_total - elapsed
                print(f"    [{i}/{len(SLOW_RANGE)}] elapsed {elapsed:.0f}s, est total {est_total:.0f}s, remain {est_remain:.0f}s")

        for fast in FAST_RANGE:
            if fast >= slow:
                continue

            sig = signal_dual_ma(price, slow=slow, fast=fast)
            trades_yr = signal_trades_per_year(sig)

            # Backtest with lag=1 + Ken French RF (corrected)
            cum = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                         tbill_rate=rf_series, signal_lag=1, commission=COMMISSION)
            metrics = calc_metrics(cum, benchmark_cum=bh_3x, tbill_rate=rf_series.mean() * 252,
                                  rf_series=rf_series)

            results.append({
                "slow": slow,
                "fast": fast,
                "leverage": LEVERAGE,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Volatility": metrics["Volatility"],
                "MDD": metrics["MDD"],
                "Trades_Year": trades_yr,
            })

    elapsed = time.time() - start_time
    print(f"  ✓ Grid search completed ({elapsed:.0f}s, {len(results)} valid combos)")

    # Save results
    df_results = pd.DataFrame(results).sort_values("Sortino", ascending=False)
    csv_path = OUT_DIR / f"{config['output_prefix']}_corrected_grid_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  [4/4] Results saved → {csv_path}")

    # Summary stats
    print(f"\n{'─'*100}")
    print(f"  TOP 10 COMBOS (by Sortino)")
    print(f"{'─'*100}")
    for i, row in df_results.head(10).iterrows():
        print(f"  {i+1:2d}. MA({int(row['fast']):2d},{int(row['slow']):3d})  "
              f"CAGR {row['CAGR']:6.1%}  Sortino {row['Sortino']:5.2f}  "
              f"MDD_Entry {row['MDD_Entry']:7.1%}  Trades/Yr {row['Trades_Year']:5.1f}")

    return df_results


def compare_original_vs_corrected(part_num, config, corrected_df):
    """Load original results and compare with corrected."""
    print(f"\n{'='*100}")
    print(f"  PART {part_num}: ORIGINAL vs CORRECTED COMPARISON")
    print(f"{'='*100}")

    # Load original results
    try:
        original_df = pd.read_csv(OUT_DIR / config["grid_csv"])
        print(f"  ✓ Loaded original: {len(original_df)} combos")
    except FileNotFoundError:
        print(f"  ⚠️  Original grid CSV not found: {config['grid_csv']}")
        print(f"     (This is expected if Part 7-9 hasn't been run yet)")
        return None

    # Merge for comparison (match on slow/fast)
    merged = corrected_df.set_index(["slow", "fast"]).join(
        original_df.set_index(["slow", "fast"]), lsuffix="_corrected", rsuffix="_original"
    ).reset_index()

    # Filter to top 10 original (by Sortino)
    top10_original = merged.nlargest(10, "Sortino_original")

    print(f"\n  Original Top 10 (by Sortino) vs Corrected Results:")
    print(f"  {'Rank':<4} {'MA(f,s)':<10} {'Original CAGR':<15} {'Corrected CAGR':<15} {'Δ':<10} {'Correction':<12}")
    print(f"  {'-'*75}")

    for i, (idx, row) in enumerate(top10_original.iterrows(), 1):
        fast, slow = int(row["fast"]), int(row["slow"])
        orig_cagr = row.get("CAGR_original", np.nan)
        corr_cagr = row.get("CAGR_corrected", np.nan)

        if pd.notna(orig_cagr) and pd.notna(corr_cagr):
            delta = orig_cagr - corr_cagr
            correction = orig_cagr / corr_cagr if corr_cagr > 0 else np.nan
            print(f"  {i:<4d} MA({fast},{slow:<3d}) {orig_cagr:>6.1%}     {corr_cagr:>6.1%}      "
                  f"{delta:>+6.1%}   {correction:>5.2f}x")
        else:
            print(f"  {i:<4d} MA({fast},{slow:<3d}) (combo not in corrected grid)")

    # Save comparison table
    comparison_csv = OUT_DIR / f"{config['output_prefix']}_vs_{config['output_prefix']}_corrected.csv"
    top10_original.to_csv(comparison_csv, index=False)
    print(f"\n  ✓ Comparison table saved → {comparison_csv}")

    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Part {part_num}: Original (lag=0, 2% RF) vs Corrected (lag=1, Ken French RF)",
                fontsize=14, fontweight="bold")

    # Plot 1: CAGR comparison
    ax = axes[0, 0]
    top10_original_sorted = top10_original.sort_values("Sortino_original", ascending=True)
    labels = [f"MA({int(row['fast'])},{int(row['slow'])})" for _, row in top10_original_sorted.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    ax.barh(x - width/2, top10_original_sorted["CAGR_original"] * 100, width, label="Original (lag=0)", color="#e74c3c")
    ax.barh(x + width/2, top10_original_sorted["CAGR_corrected"] * 100, width, label="Corrected (lag=1)", color="#3498db")
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("CAGR (%)")
    ax.set_title("CAGR Comparison")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Plot 2: Sortino comparison
    ax = axes[0, 1]
    ax.barh(x - width/2, top10_original_sorted["Sortino_original"], width, label="Original (lag=0)", color="#e74c3c")
    ax.barh(x + width/2, top10_original_sorted["Sortino_corrected"], width, label="Corrected (lag=1)", color="#3498db")
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sortino Ratio")
    ax.set_title("Sortino Comparison")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Plot 3: Correction factor distribution
    ax = axes[1, 0]
    correction_factors = top10_original_sorted["CAGR_original"] / top10_original_sorted["CAGR_corrected"]
    ax.bar(range(len(correction_factors)), correction_factors.values, color="#9b59b6", alpha=0.7)
    ax.axhline(y=correction_factors.mean(), color="red", linestyle="--", label=f"Mean: {correction_factors.mean():.2f}x")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("CAGR Correction Factor (original/corrected)")
    ax.set_title("Look-Ahead Bias: Correction Factor per Combo")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: MDD_Entry comparison
    ax = axes[1, 1]
    ax.barh(x - width/2, top10_original_sorted["MDD_Entry_original"] * 100, width, label="Original (lag=0)", color="#e74c3c")
    ax.barh(x + width/2, top10_original_sorted["MDD_Entry_corrected"] * 100, width, label="Corrected (lag=1)", color="#3498db")
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("MDD_Entry (%)")
    ax.set_title("Max Drawdown (Entry-Based) Comparison")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    png_path = OUT_DIR / f"Part{part_num}_original_vs_corrected.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Comparison chart saved → {png_path}")


def main():
    """Run corrected grid for selected parts."""
    print("\n" + "="*100)
    print("  PRIORITY 4: RE-RUN PART 7-9 WITH lag=1 + KEN FRENCH RF")
    print("="*100)

    # Parse command-line argument
    if len(sys.argv) > 1:
        try:
            parts = [int(sys.argv[1])]
            if parts[0] not in [7, 8]:
                print(f"  ❌ Invalid part number: {parts[0]} (must be 7 or 8)")
                sys.exit(1)
        except ValueError:
            print(f"  ❌ Invalid argument: {sys.argv[1]} (must be 7 or 8)")
            sys.exit(1)
    else:
        parts = [7, 8]

    # Run corrected grids
    corrected_dfs = {}
    for part_num in parts:
        config = PART_CONFIGS[part_num]
        corrected_dfs[part_num] = run_corrected_grid(part_num, config)

    # Compare with original results
    for part_num in parts:
        config = PART_CONFIGS[part_num]
        corrected_df = corrected_dfs[part_num]
        compare_original_vs_corrected(part_num, config, corrected_df)

    # Final summary
    print("\n" + "="*100)
    print("  SUMMARY")
    print("="*100)
    print(f"""
  ✅ Priority 4 Complete: Part {', '.join(map(str, parts))} re-run with corrected conditions

  Corrected Grid Results (lag=1 + Ken French RF):
  {', '.join([f'Part {p}' for p in parts])}
    - Saved to: output/Part{parts[0]}_corrected_grid_results.csv
    - Comparison: output/Part{parts[0]}_original_vs_corrected.png

  Next Steps:
  1. Review correction factor distribution
  2. Compare top 10 combos across original vs corrected
  3. Validate that correction factors match fix_part79_lag_mismatch.py results
  4. Decide whether to replace original Part 7-9 results or maintain both versions
    """)

    print("="*100)


if __name__ == "__main__":
    main()
