"""
Phase 2 — Final Comparison: All Strategies

Loads best results from each Plan's CSV output, runs OOS backtests,
and produces a unified comparison chart + table.

Requires all Plan scripts to have been run first.

Usage:
    python compare_all_strategies.py
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
    signal_asymmetric_dual_ma, signal_dual_ma,
    signal_vol_adaptive_dual_ma, signal_regime_switching_dual_ma,
    signal_vol_regime_adaptive_ma,
    signal_trades_per_year, run_buy_and_hold, calc_metrics,
)
from optimize_common import (
    OUT_DIR, CALIBRATED_ER, LEVERAGE, SIGNAL_LAG, COMMISSION,
    apply_warmup, trim_warmup, run_backtest,
    print_comparison_table, download_ndx_and_rf,
)

TEST_START = "2015-01-01"
WARMUP_DAYS = 500


def load_best_from_csv(csv_path, sortino_col="Sortino"):
    """Load CSV and return best trial row by Sortino."""
    df = pd.read_csv(csv_path)
    if sortino_col not in df.columns:
        print(f"  WARNING: {sortino_col} not in {csv_path.name}")
        return None
    return df.loc[df[sortino_col].idxmax()]


def evaluate_strategy(label, sig_fn, price, rf_series, warmup_days):
    """Run backtest and return (cum, metrics)."""
    sig_raw = sig_fn(price)
    cum, m = run_backtest(price, rf_series, sig_raw, warmup_days)
    m["Trades/Yr"] = signal_trades_per_year(sig_raw)
    return cum, m


def main():
    print("=" * 70)
    print("  Phase 2 — Final Comparison: All Strategies")
    print("=" * 70)

    print("\n[1/4] Downloading data...")
    ndx_price, rf_series = download_ndx_and_rf()
    test_price = ndx_price.loc[TEST_START:]
    full_price = ndx_price
    test_warmup = min(WARMUP_DAYS, len(test_price) - 1)
    full_warmup = WARMUP_DAYS

    strategies = {}  # name -> {signal_fn, csv_path}

    # ── Load best params from each Plan ──
    print("\n[2/4] Loading best parameters from CSV outputs...")

    # Plan 1 C1 — Penalized
    csv_c1 = OUT_DIR / "penalized_C1_results.csv"
    if csv_c1.exists():
        best = load_best_from_csv(csv_c1)
        if best is not None:
            strategies["C1 Penalized"] = {
                "fn": lambda p, b=best: signal_asymmetric_dual_ma(
                    p, int(b["fast_buy"]), int(b["slow_buy"]),
                    int(b["fast_sell"]), int(b["slow_sell"])),
                "params": f"({int(best['fast_buy'])},{int(best['slow_buy'])},"
                          f"{int(best['fast_sell'])},{int(best['slow_sell'])})",
            }
            print(f"  C1: ({int(best['fast_buy'])},{int(best['slow_buy'])},"
                  f"{int(best['fast_sell'])},{int(best['slow_sell'])}) Sortino={best['Sortino']:.3f}")
    else:
        print(f"  C1: {csv_c1} not found — skipping")

    # Plan 1 C3 — Hard cap
    csv_c3 = OUT_DIR / "penalized_C3_results.csv"
    if csv_c3.exists():
        best = load_best_from_csv(csv_c3)
        if best is not None:
            strategies["C3 Hard Cap"] = {
                "fn": lambda p, b=best: signal_asymmetric_dual_ma(
                    p, int(b["fast_buy"]), int(b["slow_buy"]),
                    int(b["fast_sell"]), int(b["slow_sell"])),
                "params": f"({int(best['fast_buy'])},{int(best['slow_buy'])},"
                          f"{int(best['fast_sell'])},{int(best['slow_sell'])})",
            }
            print(f"  C3: ({int(best['fast_buy'])},{int(best['slow_buy'])},"
                  f"{int(best['fast_sell'])},{int(best['slow_sell'])})")
    else:
        print(f"  C3: {csv_c3} not found — skipping")

    # Plan 4 — Vol-Adaptive
    csv_vol = OUT_DIR / "vol_adaptive_results.csv"
    if csv_vol.exists():
        best = load_best_from_csv(csv_vol)
        if best is not None:
            strategies["Vol-Adaptive"] = {
                "fn": lambda p, b=best: signal_vol_adaptive_dual_ma(
                    p, int(b["base_fast"]), int(b["base_slow"]),
                    int(b["vol_lookback"]), float(b["vol_scale"])),
                "params": f"(bf={int(best['base_fast'])},bs={int(best['base_slow'])},"
                          f"lb={int(best['vol_lookback'])},s={best['vol_scale']:.2f})",
            }
            print(f"  Vol-Adaptive: bf={int(best['base_fast'])}, bs={int(best['base_slow'])}, "
                  f"lb={int(best['vol_lookback'])}, s={best['vol_scale']:.2f}")
    else:
        print(f"  Vol-Adaptive: {csv_vol} not found — skipping")

    # Plan 5 — Regime
    csv_regime = OUT_DIR / "regime_results.csv"
    if csv_regime.exists():
        best = load_best_from_csv(csv_regime)
        if best is not None:
            strategies["Regime-Switch"] = {
                "fn": lambda p, b=best: signal_regime_switching_dual_ma(
                    p, int(b["fast_low"]), int(b["slow_low"]),
                    int(b["fast_high"]), int(b["slow_high"]),
                    int(b["vol_lookback"]), float(b["vol_threshold_pct"])),
                "params": f"(lo={int(best['fast_low'])},{int(best['slow_low'])}|"
                          f"hi={int(best['fast_high'])},{int(best['slow_high'])})",
            }
            print(f"  Regime: lo=({int(best['fast_low'])},{int(best['slow_low'])}), "
                  f"hi=({int(best['fast_high'])},{int(best['slow_high'])})")
    else:
        print(f"  Regime: {csv_regime} not found — skipping")

    # Plan 6 — Vol+Regime
    csv_vr = OUT_DIR / "vol_regime_results.csv"
    if csv_vr.exists():
        best = load_best_from_csv(csv_vr)
        if best is not None:
            strategies["Vol+Regime"] = {
                "fn": lambda p, b=best: signal_vol_regime_adaptive_ma(
                    p, int(b["base_fast_low"]), int(b["base_slow_low"]),
                    int(b["base_fast_high"]), int(b["base_slow_high"]),
                    int(b["vol_lookback"]), float(b["vol_threshold_pct"]),
                    float(b["vol_scale"])),
                "params": "combined",
            }
            print(f"  Vol+Regime: loaded")
    else:
        print(f"  Vol+Regime: {csv_vr} not found — skipping")

    if not strategies:
        print("\n  ERROR: No strategy CSVs found. Run Plan scripts first.")
        return

    # ── Evaluate all on test period ──
    print(f"\n[3/4] Evaluating {len(strategies)} strategies on test period...")

    test_curves = {}
    test_metrics = {}
    full_curves = {}
    full_metrics = {}

    for name, info in strategies.items():
        # Test period
        cum_t, m_t = evaluate_strategy(name, info["fn"], test_price, rf_series, test_warmup)
        test_curves[name] = cum_t
        test_metrics[name] = m_t

        # Full period
        cum_f, m_f = evaluate_strategy(name, info["fn"], full_price, rf_series, full_warmup)
        full_curves[name] = cum_f
        full_metrics[name] = m_f

    # Baselines
    for label, (fast, slow) in [("Symmetric (3,161)", (3, 161))]:
        sig_fn = lambda p, f=fast, s=slow: signal_dual_ma(p, slow=s, fast=f)
        cum_t, m_t = evaluate_strategy(label, sig_fn, test_price, rf_series, test_warmup)
        test_curves[label] = cum_t
        test_metrics[label] = m_t
        cum_f, m_f = evaluate_strategy(label, sig_fn, full_price, rf_series, full_warmup)
        full_curves[label] = cum_f
        full_metrics[label] = m_f

    # B&H 3x
    rf_scalar = rf_series.mean() * 252
    bh3x_test = run_buy_and_hold(test_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_test = trim_warmup(bh3x_test, test_warmup, test_price.index)
    m_bh3x_test = calc_metrics(bh3x_test, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh3x_test["Trades/Yr"] = 0
    test_curves["B&H 3x"] = bh3x_test
    test_metrics["B&H 3x"] = m_bh3x_test

    bh3x_full = run_buy_and_hold(full_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_full = trim_warmup(bh3x_full, full_warmup, full_price.index)
    m_bh3x_full = calc_metrics(bh3x_full, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh3x_full["Trades/Yr"] = 0
    full_curves["B&H 3x"] = bh3x_full
    full_metrics["B&H 3x"] = m_bh3x_full

    # ── Walk-forward results (stitched, not single-param) ──
    wf_csv = OUT_DIR / "wf_window_results.csv"
    wfp_csv = OUT_DIR / "wfp_window_results.csv"
    has_wf = wf_csv.exists()
    has_wfp = wfp_csv.exists()

    # ── Comparison tables ──
    print_comparison_table("TEST Period (2015-2025) — All Strategies", test_metrics)
    print_comparison_table("FULL Period (~1987-2025) — All Strategies", full_metrics)

    if has_wf:
        print(f"\n  Note: Walk-forward results in {wf_csv.name} — stitched OOS, not directly comparable")
    if has_wfp:
        print(f"  Note: WF+Penalty results in {wfp_csv.name}")

    # ── Save table ──
    records = []
    for period_label, metrics_dict in [("test", test_metrics), ("full", full_metrics)]:
        for name, m in metrics_dict.items():
            records.append({
                "period": period_label,
                "strategy": name,
                "CAGR": m["CAGR"],
                "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "Volatility": m["Volatility"],
                "MDD": m["MDD"],
                "Trades_Yr": m.get("Trades/Yr", ""),
            })
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "final_strategy_table.csv", index=False)
    print(f"\n  → saved final_strategy_table.csv ({len(df)} rows)")

    # ── Charts ──
    print("\n[4/4] Generating charts...")

    # Test period comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_curves)))
    for (label, s), color in zip(test_curves.items(), colors):
        lw = 2.0 if label in ("Symmetric (3,161)", "B&H 3x") else 1.3
        ls = "--" if label in ("Symmetric (3,161)", "B&H 3x") else "-"
        ax.plot(s.index, s.values, label=label, linewidth=lw, linestyle=ls, color=color)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Phase 2: All Strategies — OOS Comparison (2015-2025)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_strategy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  → saved final_strategy_comparison.png")

    # Full period comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    for (label, s), color in zip(full_curves.items(), colors):
        lw = 2.0 if label in ("Symmetric (3,161)", "B&H 3x") else 1.3
        ls = "--" if label in ("Symmetric (3,161)", "B&H 3x") else "-"
        ax.plot(s.index, s.values, label=label, linewidth=lw, linestyle=ls, color=color)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Phase 2: All Strategies — Full Period (~1987-2025)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_strategy_comparison_full.png", dpi=150)
    plt.close(fig)
    print(f"  → saved final_strategy_comparison_full.png")

    print("\n" + "=" * 70)
    print("  Done! Final comparison saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
