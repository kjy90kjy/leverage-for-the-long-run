"""
Walk-Forward Test (Simplified): Quantify overfitting (IS vs OOS performance)

Validates whether regime-switching parameters optimized on 1987-2018 generalize
to out-of-sample (2019-2025) data.

Data split:
  Train:  1987-01-01 ~ 2018-12-31  (32 years, in-sample optimization period)
  Test:   2019-01-01 ~ 2025-12-31  (6 years, out-of-sample evaluation period)

Workflow:
  1. Select representative regime-switching params (from grid_v2 analysis)
  2. Evaluate on Train period (in-sample)
  3. Evaluate on Test period (out-of-sample)
  4. Compare IS vs OOS performance degradation
  5. Check if OOS results are materially worse (>30% degradation = overfitting)

Usage:
    python test_walk_forward.py

Output:
    - Console: IS/OOS comparison table
    - PNG: IS vs OOS equity curves
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
    signal_regime_switching_dual_ma, signal_dual_ma,
    run_lrs, run_buy_and_hold, calc_metrics, signal_trades_per_year,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Config ──
TICKER = "^NDX"
FULL_START = "1985-10-01"
FULL_END = "2025-12-31"
TRAIN_START = "1987-01-01"
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2025-12-31"

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002

# Representative regime-switching parameter combos to test
# (fast_low, slow_low, fast_high, slow_high, vol_lookback, vol_threshold_pct, label)
TEST_PARAMS = [
    (7, 60, 15, 90, 60, 50.0, "Regime A: Volatile-favoring"),
    (10, 150, 20, 200, 60, 50.0, "Regime B: Moderate"),
    (3, 161, 3, 161, 60, 50.0, "Regime C: Symmetric (control)"),
    (12, 250, 25, 300, 50, 45.0, "Regime D: Conservative"),
]


def run_walk_forward_test():
    """Compare IS vs OOS performance for regime-switching params."""
    print("\n" + "=" * 100)
    print("  WALK-FORWARD TEST: In-Sample vs Out-of-Sample Validation")
    print("=" * 100)

    # Download full data
    print(f"\n  Downloading {TICKER} ({FULL_START} -> {FULL_END})...")
    price_full = download(TICKER, start=FULL_START, end=FULL_END)
    rf_series = download_ken_french_rf()
    rf_scalar = rf_series.mean() * 252
    print(f"  Downloaded {len(price_full)} trading days")

    # Split by date
    price_train = price_full[TRAIN_START:TRAIN_END]
    price_test = price_full[TEST_START:TEST_END]
    print(f"\n  Train period: {TRAIN_START} -> {TRAIN_END} ({len(price_train)} days = {len(price_train)/252:.1f} years)")
    print(f"  Test period:  {TEST_START} -> {TEST_END} ({len(price_test)} days = {len(price_test)/252:.1f} years)")

    # Buy & Hold baselines
    bh_train = run_buy_and_hold(price_train, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh_test = run_buy_and_hold(price_test, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)

    results = []
    curves_is = {}
    curves_oos = {}

    print(f"\n  Evaluating {len(TEST_PARAMS)} regime-switching param combos...")

    for fast_low, slow_low, fast_high, slow_high, vol_lb, vol_th, label in TEST_PARAMS:
        print(f"\n  Testing: {label}")
        print(f"    Low regime:  MA({fast_low}, {slow_low})")
        print(f"    High regime: MA({fast_high}, {slow_high})")
        print(f"    Vol trigger: lb={vol_lb}, threshold={vol_th}%")

        # ── IN-SAMPLE (Train) ──
        sig_train = signal_regime_switching_dual_ma(
            price_train,
            fast_low=fast_low, slow_low=slow_low,
            fast_high=fast_high, slow_high=slow_high,
            vol_lookback=vol_lb, vol_threshold_pct=vol_th
        )
        cum_train = run_lrs(price_train, sig_train, leverage=LEVERAGE,
                            expense_ratio=CALIBRATED_ER,
                            tbill_rate=rf_series, signal_lag=SIGNAL_LAG,
                            commission=COMMISSION)
        m_train = calc_metrics(cum_train, benchmark_cum=bh_train,
                              tbill_rate=rf_scalar, rf_series=rf_series)
        trades_yr_train = signal_trades_per_year(sig_train)

        # ── OUT-OF-SAMPLE (Test) ──
        sig_test = signal_regime_switching_dual_ma(
            price_test,
            fast_low=fast_low, slow_low=slow_low,
            fast_high=fast_high, slow_high=slow_high,
            vol_lookback=vol_lb, vol_threshold_pct=vol_th
        )
        cum_test = run_lrs(price_test, sig_test, leverage=LEVERAGE,
                           expense_ratio=CALIBRATED_ER,
                           tbill_rate=rf_series, signal_lag=SIGNAL_LAG,
                           commission=COMMISSION)
        m_test = calc_metrics(cum_test, benchmark_cum=bh_test,
                             tbill_rate=rf_scalar, rf_series=rf_series)
        trades_yr_test = signal_trades_per_year(sig_test)

        # Calculate degradation
        cagr_degrad = (m_train["CAGR"] - m_test["CAGR"]) / m_train["CAGR"] * 100 if m_train["CAGR"] > 0 else 0
        sortino_degrad = (m_train["Sortino"] - m_test["Sortino"]) / m_train["Sortino"] * 100 if m_train["Sortino"] > 0 else 0

        results.append({
            "Label": label,
            "Fast_Low": fast_low,
            "Slow_Low": slow_low,
            "Fast_High": fast_high,
            "Slow_High": slow_high,
            "Vol_Lookback": vol_lb,
            "Vol_Threshold": vol_th,
            "IS_CAGR": m_train["CAGR"],
            "OOS_CAGR": m_test["CAGR"],
            "CAGR_Degrad_pct": cagr_degrad,
            "IS_Sortino": m_train["Sortino"],
            "OOS_Sortino": m_test["Sortino"],
            "Sortino_Degrad_pct": sortino_degrad,
            "IS_MDD": m_train["MDD"],
            "OOS_MDD": m_test["MDD"],
            "IS_Trades_Yr": trades_yr_train,
            "OOS_Trades_Yr": trades_yr_test,
        })

        curves_is[label] = cum_train
        curves_oos[label] = cum_test

        # Print summary for this param
        print(f"    IS  CAGR: {m_train['CAGR']:>7.2%}  Sortino: {m_train['Sortino']:>6.3f}  MDD: {m_train['MDD']:>7.2%}  Trades/Yr: {trades_yr_train:>5.1f}")
        print(f"    OOS CAGR: {m_test['CAGR']:>7.2%}  Sortino: {m_test['Sortino']:>6.3f}  MDD: {m_test['MDD']:>7.2%}  Trades/Yr: {trades_yr_test:>5.1f}")
        print(f"    Degrad:   CAGR {cagr_degrad:>6.1f}%  Sortino {sortino_degrad:>6.1f}%")

    # Print comparison table
    print("\n" + "=" * 130)
    print("  IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("=" * 130)
    print(f"  {'Label':<32} {'IS CAGR':>10} {'OOS CAGR':>10} {'Degrad %':>10} "
          f"{'IS Sort':>10} {'OOS Sort':>10} {'S.Degrad%':>10}")
    print("  " + "─" * 125)

    for r in results:
        degrad_marker = "✓" if r["CAGR_Degrad_pct"] < 20 else "⚠" if r["CAGR_Degrad_pct"] < 40 else "✗"
        print(f"  {r['Label']:<32} {r['IS_CAGR']:>9.2%} {r['OOS_CAGR']:>9.2%} "
              f"{degrad_marker}{r['CAGR_Degrad_pct']:>8.1f}% {r['IS_Sortino']:>9.3f} {r['OOS_Sortino']:>9.3f} "
              f"{r['Sortino_Degrad_pct']:>9.1f}%")

    # Overall assessment
    print("\n" + "=" * 130)
    print("  OVERFITTING ASSESSMENT")
    print("=" * 130)

    mean_cagr_degrad = np.mean([r["CAGR_Degrad_pct"] for r in results])
    max_cagr_degrad = np.max([r["CAGR_Degrad_pct"] for r in results])

    print(f"  Mean CAGR degradation:  {mean_cagr_degrad:>6.1f}%")
    print(f"  Max CAGR degradation:   {max_cagr_degrad:>6.1f}%")

    if mean_cagr_degrad > 50:
        print(f"\n  ✗ SEVERE OVERFITTING: OOS {mean_cagr_degrad:.1f}% worse than IS")
        print(f"  → Full-period optimization NOT reliable for live trading")
    elif mean_cagr_degrad > 30:
        print(f"\n  ⚠ MODERATE OVERFITTING: OOS {mean_cagr_degrad:.1f}% worse than IS")
        print(f"  → Use with caution; recommend walk-forward approach")
    elif mean_cagr_degrad > 10:
        print(f"\n  ~ MILD OVERFITTING: OOS {mean_cagr_degrad:.1f}% worse than IS")
        print(f"  → Reasonable generalization; normal market regime shift")
    else:
        print(f"\n  ✓ MINIMAL OVERFITTING: OOS only {mean_cagr_degrad:.1f}% worse than IS")
        print(f"  → Parameters generalize very well to OOS period")

    # Save results
    df_results = pd.DataFrame(results)
    csv_path = OUT_DIR / "walk_forward_is_vs_oos.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")

    # Plot IS vs OOS comparison
    fig, axes = plt.subplots(2, len(TEST_PARAMS) // 2 + (len(TEST_PARAMS) % 2),
                             figsize=(6 * (len(TEST_PARAMS) // 2 + 1), 10))
    axes = axes.flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        label = r["Label"]

        ax.plot(curves_is[label].index, curves_is[label].values,
               label="In-Sample (Train 1987-2018)", linewidth=2, color="#3498db")
        ax.plot(curves_oos[label].index, curves_oos[label].values,
               label="Out-of-Sample (Test 2019-2025)", linewidth=2, color="#e74c3c")

        ax.set_yscale("log")
        ax.set_title(f"{label}\n"
                    f"IS: {r['IS_CAGR']:.1%} | OOS: {r['OOS_CAGR']:.1%} | Degrad: {r['CAGR_Degrad_pct']:.0f}%",
                    fontsize=10)
        ax.set_ylabel("Growth of $1 (log)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("In-Sample vs Out-of-Sample: Regime-Switching Parameter Validation",
                fontsize=14, fontweight="bold")
    fig.tight_layout()
    png_path = OUT_DIR / "walk_forward_is_vs_oos_curves.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {png_path}")

    print("\n" + "=" * 130)


if __name__ == "__main__":
    run_walk_forward_test()
