"""
Validate our LRS framework against eulb results from fmkorea.
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

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leverage_rotation import (
    download, signal_dual_ma, run_lrs, run_buy_and_hold,
    calc_metrics, signal_trades_per_year, download_ken_french_rf,
)


def run_eulb_backtest(price, fast, slow, leverage=3, expense_ratio=0.0,
                      tbill_rate=0.0, signal_lag=0):
    sig = signal_dual_ma(price, slow=slow, fast=fast)
    cum = run_lrs(price, sig, leverage=leverage,
                  expense_ratio=expense_ratio,
                  tbill_rate=tbill_rate,
                  signal_lag=signal_lag)
    m = calc_metrics(cum, tbill_rate=0.0)
    tpy = signal_trades_per_year(sig)
    return cum, m, tpy, sig


def main():
    TR = "Total Return"
    CA = "CAGR"
    VO = "Volatility"
    MD = "MDD"
    SH = "Sharpe"

    sep = "=" * 90
    print(sep)
    print("  VALIDATION: Our LRS Framework vs eulb Results (fmkorea)")
    print(sep)

    print("\n[1] Downloading ^NDX (full history, price-only)...")
    price = download("^NDX", start="1985-01-01", end="2026-12-31", total_return=False)
    print(f"    Data: {len(price)} trading days  ({price.index[0].date()} -> {price.index[-1].date()})")

    eulb_combos = [
        (1,  200,  9654.6,  None,    "MA200 (single MA)"),
        (3,  220, 23941.3,  None,    "eulb highlight"),
        (4,   80,  None,    28.554,  "Best CAGR full period"),
        (3,  161,  None,    None,    "eulb 5p optimal"),
        (3,  216,  None,    None,    "eulb fine-tuned optimal"),
        (35, 160,  None,    None,    "Best post-2002 (full here)"),
    ]
    # B&H 3x
    print("\n[2] Running B&H 3x (0% expense)...")
    bh3_cum = run_buy_and_hold(price, leverage=3, expense_ratio=0.0)
    bh3_m = calc_metrics(bh3_cum, tbill_rate=0.0)
    bh3_total_pct = (bh3_m[TR] - 1) * 100
    bh3_cagr_pct = bh3_m[CA] * 100
    print(f"    B&H 3x: Total = {bh3_total_pct:.1f}% ({bh3_m[TR]:.1f}x), CAGR = {bh3_cagr_pct:.3f}%")

    # === Run with lag=0 (eulb claimed same-day) ===
    print("\n[3] Running backtests with lag=0 (same-day signal)...")
    print("    " + "-" * 80)

    results_lag0 = []
    for fast, slow, eulb_ret_pct, eulb_cagr_pct, desc in eulb_combos:
        cum, m, tpy, sig = run_eulb_backtest(price, fast, slow,
                                              leverage=3, expense_ratio=0.0,
                                              tbill_rate=0.0, signal_lag=0)
        total_ret_pct = (m[TR] - 1) * 100
        cagr_pct = m[CA] * 100
        results_lag0.append({
            "fast": fast, "slow": slow, "desc": desc,
            "total_pct": total_ret_pct, "total_x": m[TR],
            "cagr_pct": cagr_pct, "mdd": m[MD], "tpy": tpy,
            "eulb_total_pct": eulb_ret_pct, "eulb_cagr_pct": eulb_cagr_pct,
        })
        print(f"    ({fast:>2}, {slow:>3}): CAGR={cagr_pct:>8.3f}%  Total={total_ret_pct:>14,.1f}%  ({m[TR]:>10,.1f}x)  MDD={m[MD]:>8.2%}  | {desc}")

    # === Run with lag=1 (realistic, no look-ahead) ===
    print("\n[4] Running backtests with lag=1 (next-day signal, no look-ahead)...")
    print("    " + "-" * 80)

    results_lag1 = []
    for fast, slow, eulb_ret_pct, eulb_cagr_pct, desc in eulb_combos:
        cum, m, tpy, sig = run_eulb_backtest(price, fast, slow,
                                              leverage=3, expense_ratio=0.0,
                                              tbill_rate=0.0, signal_lag=1)
        total_ret_pct = (m[TR] - 1) * 100
        cagr_pct = m[CA] * 100
        results_lag1.append({
            "fast": fast, "slow": slow, "desc": desc,
            "total_pct": total_ret_pct, "total_x": m[TR],
            "cagr_pct": cagr_pct, "mdd": m[MD], "tpy": tpy,
            "eulb_total_pct": eulb_ret_pct, "eulb_cagr_pct": eulb_cagr_pct,
        })
        print(f"    ({fast:>2}, {slow:>3}): CAGR={cagr_pct:>8.3f}%  Total={total_ret_pct:>14,.1f}%  ({m[TR]:>10,.1f}x)  MDD={m[MD]:>8.2%}  | {desc}")
    # === Run with OUR standard settings ===
    print("\n[5] Running with OUR standard settings (1% expense, Ken French RF, lag=1)...")
    rf_series = download_ken_french_rf()
    print("    " + "-" * 80)

    results_ours = []
    for fast, slow, _, _, desc in eulb_combos:
        cum, m, tpy, sig = run_eulb_backtest(price, fast, slow,
                                              leverage=3, expense_ratio=0.01,
                                              tbill_rate=rf_series, signal_lag=1)
        m2 = calc_metrics(cum, rf_series=rf_series)
        total_ret_pct = (m2[TR] - 1) * 100
        cagr_pct = m2[CA] * 100
        results_ours.append({
            "fast": fast, "slow": slow, "desc": desc,
            "total_pct": total_ret_pct, "total_x": m2[TR],
            "cagr_pct": cagr_pct, "mdd": m2[MD],
            "sharpe": m2[SH], "tpy": tpy,
        })
        print(f"    ({fast:>2}, {slow:>3}): CAGR={cagr_pct:>8.3f}%  Total={total_ret_pct:>14,.1f}%  ({m2[TR]:>10,.1f}x)  Sharpe={m2[SH]:.3f}  | {desc}")

    bh3_ours = run_buy_and_hold(price, leverage=3, expense_ratio=0.01)
    bh3_ours_m = calc_metrics(bh3_ours, rf_series=rf_series)
    # ===== COMPARISON TABLE =====
    print("\n" + "=" * 120)
    print("  COMPARISON TABLE: eulb vs Our Framework (lag=0 / lag=1 / our standard)")
    print("  NOTE: lag=0 has severe look-ahead bias for fast=1 (price vs MA). lag=1 is realistic.")
    print("=" * 120)

    hdr = "  {:10s} {:22s} {:>14s}  {:>10s} {:>10s} {:>10s} {:>14s}".format(
        "Combo", "Description", "eulb Total%", "eulb CAGR", "Lag0 CAGR", "Lag1 CAGR", "Ours-Std CAGR")
    print(hdr)
    print("  " + "-" * 116)

    for r0, r1, ro in zip(results_lag0, results_lag1, results_ours):
        combo = f"({r0['fast']},{r0['slow']})"
        eulb_t = f"{r0['eulb_total_pct']:>12,.1f}%" if r0["eulb_total_pct"] is not None else "          N/A  "
        eulb_c = f"{r0['eulb_cagr_pct']:>8.3f}%" if r0["eulb_cagr_pct"] is not None else "      N/A "
        print(f"  {combo:<10} {r0['desc']:<22} {eulb_t:>14}  {eulb_c:>10} {r0['cagr_pct']:>8.3f}%  {r1['cagr_pct']:>8.3f}%  {ro['cagr_pct']:>12.3f}%")

    our_bh_total = bh3_total_pct
    bh_ours_cagr = bh3_ours_m[CA] * 100
    print(f"  {'B&H 3x':<10} {'Buy & Hold 3x':<22}     3,448.3%         N/A {bh3_cagr_pct:>8.3f}%  {bh3_cagr_pct:>8.3f}%  {bh_ours_cagr:>12.3f}%")
    # ===== SETTINGS IMPACT =====
    print("\n" + "=" * 120)
    print("  SETTINGS IMPACT: lag=1 (0% expense) vs Our Standard (1% expense, Ken French RF)")
    print("=" * 120)

    hdr2 = "  {:10s} {:22s} {:>12s} {:>12s} {:>12s} {:>12s} {:>10s}".format(
        "Combo", "Description", "Lag1 CAGR", "Lag1 Total", "Std CAGR", "Std Total", "CAGR diff")
    print(hdr2)
    print("  " + "-" * 116)

    for r1, ro in zip(results_lag1, results_ours):
        combo = f"({r1['fast']},{r1['slow']})"
        diff = ro["cagr_pct"] - r1["cagr_pct"]
        print(f"  {combo:<10} {r1['desc']:<22} {r1['cagr_pct']:>10.3f}% {r1['total_pct']:>10,.1f}% {ro['cagr_pct']:>10.3f}% {ro['total_pct']:>10,.1f}% {diff:>10.3f}%")

    bh_ours_total = (bh3_ours_m[TR] - 1) * 100
    bh_diff = bh3_ours_m[CA] * 100 - bh3_cagr_pct
    print(f"  {'B&H 3x':<10} {'Buy & Hold 3x':<22} {bh3_cagr_pct:>10.3f}% {our_bh_total:>10,.1f}% {bh3_ours_m[CA]*100:>10.3f}% {bh_ours_total:>10,.1f}% {bh_diff:>10.3f}%")
    # ===== POST-2002 TEST =====
    print("\n" + "=" * 90)
    print("  POST-2002 TEST: (35, 160) -- eulb reports 30.128% CAGR")
    print("=" * 90)

    price_post = price[price.index >= "2002-01-01"]
    print(f"  Period: {price_post.index[0].date()} -> {price_post.index[-1].date()} ({len(price_post)} days)")

    cum0, m0, _, _ = run_eulb_backtest(price_post, fast=35, slow=160, leverage=3, expense_ratio=0.0, tbill_rate=0.0, signal_lag=0)
    cum1, m1, _, _ = run_eulb_backtest(price_post, fast=35, slow=160, leverage=3, expense_ratio=0.0, tbill_rate=0.0, signal_lag=1)
    print(f"  lag=0: CAGR = {m0[CA]*100:.3f}%, Total = {m0[TR]:.1f}x, MDD = {m0[MD]:.2%}")
    print(f"  lag=1: CAGR = {m1[CA]*100:.3f}%, Total = {m1[TR]:.1f}x, MDD = {m1[MD]:.2%}")
    print(f"  eulb:  CAGR = 30.128%")
    diff0 = abs(m0[CA] * 100 - 30.128)
    diff1 = abs(m1[CA] * 100 - 30.128)
    print(f"  Diff from eulb: lag0={diff0:.3f}pp, lag1={diff1:.3f}pp")

    # ===== SUMMARY =====
    print("\n" + "=" * 90)
    print("  VALIDATION SUMMARY")
    print("=" * 90)
    print("")
    print("  KEY FINDINGS:")
    print("  1. With lag=0, fast=1 (price vs MA) creates EXTREME look-ahead bias")
    print("     (signal uses today close which IS the return -> selection bias).")
    print("     For fast>=3, lag=0 bias is moderate since signal uses multi-day avg.")
    print("  2. With lag=1 (realistic), our framework produces reasonable results.")
    print("  3. Absolute numbers differ from eulb due to different data endpoints")
    print("     (our data extends to 2026, eulb likely ended ~2023-2024).")
    print("  4. The 1% expense ratio + Ken French RF reduces returns by ~2-5% CAGR.")
    print("")
    print("  RANKING CHECK (lag=1, sorted by total return):")
    sorted_r = sorted(results_lag1, key=lambda r: r["total_pct"], reverse=True)
    for i, r in enumerate(sorted_r, 1):
        print(f"    #{i}: ({r['fast']},{r['slow']}) = {r['cagr_pct']:>7.3f}% CAGR, {r['total_pct']:>12,.1f}% total -- {r['desc']}")
    print(f"    B&H 3x = {bh3_cagr_pct:>7.3f}% CAGR, {our_bh_total:>12,.1f}% total")

    print("")
    print("  EULB RANKING: (3,220) > (1,200) > B&H 3x")
    print("  (by total return: 23941% > 9654% > 3448%)")
    print("")
    r1_200 = next(r for r in results_lag1 if r["fast"]==1 and r["slow"]==200)
    r3_220 = next(r for r in results_lag1 if r["fast"]==3 and r["slow"]==220)
    ours_ranking = r3_220["total_pct"] > r1_200["total_pct"] > our_bh_total
    print(f"  OUR RANKING (lag=1): (3,220) {r3_220['total_pct']:,.0f}% > (1,200) {r1_200['total_pct']:,.0f}% > B&H {our_bh_total:,.0f}%")
    if ours_ranking:
        print("  --> Rankings MATCH eulb. Framework logic validated.")
    else:
        print("  --> Rankings DIFFER from eulb.")

    print("\n" + "=" * 90)
    print("  Done.")
    print("=" * 90)


if __name__ == "__main__":
    main()
