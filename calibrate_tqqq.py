"""
TQQQ Cost Calibration — Synthetic 3x NDX vs Actual TQQQ

Calibrates the expense_ratio parameter by comparing:
1. Synthetic 3x leveraged QQQ (our model) vs actual TQQQ
2. Fixed expense_ratio sweep (0.5% ~ 3.5%)
3. Time-varying financing cost model (Ken French RF-based)
4. Interest rate regime sub-analysis

Output:
    output/tqqq_calibration_cumulative.png
    output/tqqq_calibration_landscape.png
    output/tqqq_calibration_rolling_te.png

Usage:
    python calibrate_tqqq.py
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
    download, run_buy_and_hold, calc_metrics, download_ken_french_rf,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

TQQQ_INCEPTION = "2010-02-11"
TQQQ_STATED_ER = 0.0086  # ProShares TQQQ stated expense ratio (0.86%)


# ──────────────────────────────────────────────
# Section 1: Data Download
# ──────────────────────────────────────────────

def load_data():
    """Download QQQ and TQQQ, align to common trading dates."""
    print("\n[1] Downloading QQQ and TQQQ...")
    qqq = download("QQQ", start=TQQQ_INCEPTION, end="2026-12-31")
    tqqq = download("TQQQ", start=TQQQ_INCEPTION, end="2026-12-31")

    common = qqq.index.intersection(tqqq.index)
    qqq = qqq.loc[common]
    tqqq = tqqq.loc[common]

    print(f"    QQQ:  {len(qqq)} days  ({qqq.index[0].date()} -> {qqq.index[-1].date()})")
    print(f"    TQQQ: {len(tqqq)} days ({tqqq.index[0].date()} -> {tqqq.index[-1].date()})")
    return qqq, tqqq


# ──────────────────────────────────────────────
# Section 2: Fixed Expense Ratio Sweep
# ──────────────────────────────────────────────

def fixed_er_sweep(qqq: pd.Series, tqqq: pd.Series):
    """Sweep expense_ratio from 0.5% to 3.5% and compare with actual TQQQ."""
    print("\n[2] Fixed expense_ratio sweep (0.5% ~ 3.5%, step 0.1%)...")

    actual_cum = run_buy_and_hold(tqqq, leverage=1.0, expense_ratio=0.0)
    actual_metrics = calc_metrics(actual_cum)
    actual_cagr = actual_metrics["CAGR"]
    actual_daily_ret = tqqq.pct_change().dropna()

    print(f"    Actual TQQQ: CAGR = {actual_cagr:.4%}")

    er_values = np.arange(0.005, 0.036, 0.001)
    results = []

    for er in er_values:
        sim_cum = run_buy_and_hold(qqq, leverage=3.0, expense_ratio=er)
        sim_metrics = calc_metrics(sim_cum)
        sim_cagr = sim_metrics["CAGR"]

        # Daily tracking error
        sim_daily_ret = sim_cum.pct_change().dropna()
        common_idx = actual_daily_ret.index.intersection(sim_daily_ret.index)
        te_daily = (sim_daily_ret.loc[common_idx] - actual_daily_ret.loc[common_idx]).std() * np.sqrt(252)

        # Cumulative divergence at end
        cum_div = (sim_cum.iloc[-1] / actual_cum.iloc[-1]) - 1

        results.append({
            "ER": er,
            "Sim_CAGR": sim_cagr,
            "Actual_CAGR": actual_cagr,
            "CAGR_diff": sim_cagr - actual_cagr,
            "CAGR_abs_diff": abs(sim_cagr - actual_cagr),
            "Tracking_Error": te_daily,
            "Cum_Divergence": cum_div,
        })

    df = pd.DataFrame(results)

    # Find optimal ER (minimize |CAGR_sim - CAGR_actual|)
    best_idx = df["CAGR_abs_diff"].idxmin()
    best_er = df.loc[best_idx, "ER"]
    best_diff = df.loc[best_idx, "CAGR_diff"]

    print(f"\n    {'ER':>6s}  {'Sim CAGR':>10s}  {'CAGR diff':>10s}  {'Track.Err':>10s}  {'Cum div':>10s}")
    print(f"    {'-'*52}")
    for _, r in df.iterrows():
        marker = " <-- best" if abs(r["ER"] - best_er) < 0.0001 else ""
        print(f"    {r['ER']:>5.1%}  {r['Sim_CAGR']:>10.3%}  {r['CAGR_diff']:>+10.3%}  "
              f"{r['Tracking_Error']:>10.3%}  {r['Cum_Divergence']:>+10.2%}{marker}")

    print(f"\n    Optimal fixed ER: {best_er:.2%} (CAGR diff = {best_diff:+.4%})")

    return df, best_er


# ──────────────────────────────────────────────
# Section 3: Time-Varying Financing Cost Model
# ──────────────────────────────────────────────

def time_varying_model(qqq: pd.Series, tqqq: pd.Series, rf_series: pd.Series):
    """Model: daily_cost = stated_ER/252 + rf_daily * (leverage-1) * spread_multiplier."""
    print("\n[3] Time-varying financing cost model (Ken French RF-based)...")

    actual_cum = run_buy_and_hold(tqqq, leverage=1.0, expense_ratio=0.0)
    actual_metrics = calc_metrics(actual_cum)
    actual_cagr = actual_metrics["CAGR"]

    qqq_daily_ret = qqq.pct_change()
    tqqq_daily_ret = tqqq.pct_change()
    rf_aligned = rf_series.reindex(qqq.index, method="ffill").fillna(0)

    spread_values = np.arange(0.8, 1.55, 0.05)
    results = []

    for spread in spread_values:
        # Synthetic 3x with time-varying cost
        daily_financing = rf_aligned * 2.0 * spread  # (leverage-1) = 2
        daily_stated = TQQQ_STATED_ER / 252
        daily_total_cost = daily_stated + daily_financing

        sim_ret = qqq_daily_ret * 3.0 - daily_total_cost
        sim_cum = (1 + sim_ret).cumprod()
        sim_cum.iloc[0] = 1.0

        sim_metrics = calc_metrics(sim_cum)
        sim_cagr = sim_metrics["CAGR"]

        # Tracking error
        common_idx = tqqq_daily_ret.dropna().index.intersection(sim_ret.dropna().index)
        te = (sim_ret.loc[common_idx] - tqqq_daily_ret.loc[common_idx]).std() * np.sqrt(252)

        cum_div = (sim_cum.iloc[-1] / actual_cum.iloc[-1]) - 1

        # Average annualized total cost
        avg_annual_cost = daily_total_cost.mean() * 252

        results.append({
            "Spread": spread,
            "Sim_CAGR": sim_cagr,
            "CAGR_diff": sim_cagr - actual_cagr,
            "CAGR_abs_diff": abs(sim_cagr - actual_cagr),
            "Tracking_Error": te,
            "Cum_Divergence": cum_div,
            "Avg_Annual_Cost": avg_annual_cost,
        })

    df = pd.DataFrame(results)
    best_idx = df["CAGR_abs_diff"].idxmin()
    best_spread = df.loc[best_idx, "Spread"]
    best_diff = df.loc[best_idx, "CAGR_diff"]
    best_avg_cost = df.loc[best_idx, "Avg_Annual_Cost"]

    print(f"\n    {'Spread':>7s}  {'Sim CAGR':>10s}  {'CAGR diff':>10s}  {'Track.Err':>10s}  {'Avg Cost':>10s}")
    print(f"    {'-'*55}")
    for _, r in df.iterrows():
        marker = " <-- best" if abs(r["Spread"] - best_spread) < 0.001 else ""
        print(f"    {r['Spread']:>6.2f}x  {r['Sim_CAGR']:>10.3%}  {r['CAGR_diff']:>+10.3%}  "
              f"{r['Tracking_Error']:>10.3%}  {r['Avg_Annual_Cost']:>10.2%}{marker}")

    print(f"\n    Optimal spread multiplier: {best_spread:.2f}x")
    print(f"    Average annualized cost at optimum: {best_avg_cost:.2%}")
    print(f"    CAGR diff at optimum: {best_diff:+.4%}")

    return df, best_spread


# ──────────────────────────────────────────────
# Section 4: Interest Rate Regime Sub-Analysis
# ──────────────────────────────────────────────

def regime_analysis(qqq: pd.Series, tqqq: pd.Series, rf_series: pd.Series,
                    best_fixed_er: float, best_spread: float):
    """Compare fixed-ER and time-varying models across interest rate regimes."""
    print("\n[4] Interest rate regime sub-analysis...")

    regimes = {
        "ZIRP (2010-2015)":        ("2010-02-11", "2015-12-31"),
        "Rate Hike (2016-2019)":   ("2016-01-01", "2019-12-31"),
        "COVID ZIRP (2020-2021)":  ("2020-01-01", "2021-12-31"),
        "High Rate (2022-present)": ("2022-01-01", "2026-12-31"),
    }

    qqq_ret = qqq.pct_change()
    tqqq_ret = tqqq.pct_change()
    rf_aligned = rf_series.reindex(qqq.index, method="ffill").fillna(0)

    print(f"\n    {'Regime':<28s}  {'Avg RF':>7s}  {'Fixed TE':>10s}  {'TV TE':>10s}  "
          f"{'Fixed CAGR diff':>15s}  {'TV CAGR diff':>15s}")
    print(f"    {'-'*95}")

    regime_results = []

    for name, (s, e) in regimes.items():
        mask = (qqq.index >= s) & (qqq.index <= e)
        if mask.sum() < 50:
            continue

        idx = qqq.index[mask]
        q_sub = qqq.loc[idx]
        t_sub = tqqq.loc[idx]
        rf_sub = rf_aligned.loc[idx]

        # Actual TQQQ
        actual_cum = run_buy_and_hold(t_sub, leverage=1.0, expense_ratio=0.0)
        actual_m = calc_metrics(actual_cum)
        actual_ret = t_sub.pct_change().dropna()

        # Fixed ER model
        fixed_cum = run_buy_and_hold(q_sub, leverage=3.0, expense_ratio=best_fixed_er)
        fixed_m = calc_metrics(fixed_cum)
        fixed_ret = fixed_cum.pct_change().dropna()

        # Time-varying model
        daily_financing = rf_sub * 2.0 * best_spread
        daily_stated = TQQQ_STATED_ER / 252
        tv_ret = q_sub.pct_change() * 3.0 - daily_stated - daily_financing
        tv_cum = (1 + tv_ret).cumprod()
        tv_cum.iloc[0] = 1.0
        tv_m = calc_metrics(tv_cum)

        # Tracking errors
        common_f = fixed_ret.index.intersection(actual_ret.index)
        te_fixed = (fixed_ret.loc[common_f] - actual_ret.loc[common_f]).std() * np.sqrt(252)
        common_t = tv_ret.dropna().index.intersection(actual_ret.index)
        te_tv = (tv_ret.loc[common_t] - actual_ret.loc[common_t]).std() * np.sqrt(252)

        avg_rf = rf_sub.mean() * 252

        regime_results.append({
            "Regime": name,
            "Avg_RF": avg_rf,
            "Fixed_TE": te_fixed,
            "TV_TE": te_tv,
            "Fixed_CAGR_diff": fixed_m["CAGR"] - actual_m["CAGR"],
            "TV_CAGR_diff": tv_m["CAGR"] - actual_m["CAGR"],
        })

        print(f"    {name:<28s}  {avg_rf:>6.2%}  {te_fixed:>10.3%}  {te_tv:>10.3%}  "
              f"{fixed_m['CAGR'] - actual_m['CAGR']:>+14.3%}  {tv_m['CAGR'] - actual_m['CAGR']:>+14.3%}")

    return pd.DataFrame(regime_results)


# ──────────────────────────────────────────────
# Section 5: Visualizations
# ──────────────────────────────────────────────

def plot_cumulative_comparison(qqq: pd.Series, tqqq: pd.Series,
                                best_fixed_er: float, best_spread: float,
                                rf_series: pd.Series):
    """Plot actual TQQQ vs fixed-ER vs time-varying models."""
    print("\n[5] Generating visualizations...")

    # Actual TQQQ
    actual_cum = run_buy_and_hold(tqqq, leverage=1.0, expense_ratio=0.0)

    # Fixed ER
    fixed_cum = run_buy_and_hold(qqq, leverage=3.0, expense_ratio=best_fixed_er)

    # Time-varying
    rf_aligned = rf_series.reindex(qqq.index, method="ffill").fillna(0)
    daily_financing = rf_aligned * 2.0 * best_spread
    daily_stated = TQQQ_STATED_ER / 252
    tv_ret = qqq.pct_change() * 3.0 - daily_stated - daily_financing
    tv_cum = (1 + tv_ret).cumprod()
    tv_cum.iloc[0] = 1.0

    # Naive (1% ER, current default)
    naive_cum = run_buy_and_hold(qqq, leverage=3.0, expense_ratio=0.01)

    # --- Cumulative chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual_cum.index, actual_cum.values, label="TQQQ (actual)", linewidth=2, color="black")
    ax.plot(fixed_cum.index, fixed_cum.values, label=f"Synthetic 3x (fixed ER={best_fixed_er:.2%})",
            linewidth=1.5, color="#2ecc71", linestyle="--")
    ax.plot(tv_cum.index, tv_cum.values, label=f"Synthetic 3x (TV, spread={best_spread:.2f}x)",
            linewidth=1.5, color="#3498db", linestyle="-.")
    ax.plot(naive_cum.index, naive_cum.values, label="Synthetic 3x (naive ER=1.00%)",
            linewidth=1.0, color="#e74c3c", linestyle=":", alpha=0.7)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("TQQQ Calibration: Actual vs Synthetic 3x Models", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tqqq_calibration_cumulative.png", dpi=150)
    plt.close(fig)
    print(f"    -> saved {OUT_DIR / 'tqqq_calibration_cumulative.png'}")

    return actual_cum, fixed_cum, tv_cum


def plot_er_landscape(fixed_df: pd.DataFrame, tv_df: pd.DataFrame):
    """Plot CAGR error landscape for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Fixed ER landscape
    ax1.bar(fixed_df["ER"] * 100, fixed_df["CAGR_abs_diff"] * 100,
            width=0.08, color="#3498db", alpha=0.8)
    best_fixed = fixed_df.loc[fixed_df["CAGR_abs_diff"].idxmin()]
    ax1.axvline(best_fixed["ER"] * 100, color="red", linestyle="--", linewidth=1.5,
                label=f'Optimal: {best_fixed["ER"]:.2%}')
    ax1.set_xlabel("Expense Ratio (%)")
    ax1.set_ylabel("|CAGR_sim - CAGR_actual| (%p)")
    ax1.set_title("Fixed ER: CAGR Error Landscape")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Time-varying spread landscape
    ax2.bar(tv_df["Spread"], tv_df["CAGR_abs_diff"] * 100,
            width=0.04, color="#2ecc71", alpha=0.8)
    best_tv = tv_df.loc[tv_df["CAGR_abs_diff"].idxmin()]
    ax2.axvline(best_tv["Spread"], color="red", linestyle="--", linewidth=1.5,
                label=f'Optimal: {best_tv["Spread"]:.2f}x')
    ax2.set_xlabel("Spread Multiplier")
    ax2.set_ylabel("|CAGR_sim - CAGR_actual| (%p)")
    ax2.set_title("Time-Varying: CAGR Error Landscape")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "tqqq_calibration_landscape.png", dpi=150)
    plt.close(fig)
    print(f"    -> saved {OUT_DIR / 'tqqq_calibration_landscape.png'}")


def plot_rolling_tracking_error(tqqq: pd.Series, fixed_cum: pd.Series,
                                 tv_cum: pd.Series, window: int = 252):
    """Rolling 1-year annualized tracking error."""
    actual_ret = tqqq.pct_change().dropna()
    fixed_ret = fixed_cum.pct_change().dropna()
    tv_ret = tv_cum.pct_change().dropna()

    common = actual_ret.index.intersection(fixed_ret.index).intersection(tv_ret.index)
    diff_fixed = (fixed_ret.loc[common] - actual_ret.loc[common])
    diff_tv = (tv_ret.loc[common] - actual_ret.loc[common])

    roll_te_fixed = diff_fixed.rolling(window).std() * np.sqrt(252)
    roll_te_tv = diff_tv.rolling(window).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(roll_te_fixed.index, roll_te_fixed.values,
            label="Fixed ER model", linewidth=1.2, color="#2ecc71")
    ax.plot(roll_te_tv.index, roll_te_tv.values,
            label="Time-varying model", linewidth=1.2, color="#3498db")
    ax.set_title("Rolling 1Y Annualized Tracking Error vs Actual TQQQ", fontsize=14)
    ax.set_ylabel("Tracking Error (annualized)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tqqq_calibration_rolling_te.png", dpi=150)
    plt.close(fig)
    print(f"    -> saved {OUT_DIR / 'tqqq_calibration_rolling_te.png'}")


# ──────────────────────────────────────────────
# Section 6: Summary & Recommendation
# ──────────────────────────────────────────────

def print_recommendation(fixed_df: pd.DataFrame, tv_df: pd.DataFrame,
                          regime_df: pd.DataFrame, best_fixed_er: float,
                          best_spread: float):
    """Print final calibration recommendation."""
    print("\n" + "=" * 70)
    print("  CALIBRATION RESULTS & RECOMMENDATION")
    print("=" * 70)

    best_fixed = fixed_df.loc[fixed_df["CAGR_abs_diff"].idxmin()]
    best_tv = tv_df.loc[tv_df["CAGR_abs_diff"].idxmin()]

    print(f"\n  Fixed ER model:")
    print(f"    Optimal ER:        {best_fixed_er:.2%}")
    print(f"    CAGR difference:   {best_fixed['CAGR_diff']:+.4%}")
    print(f"    Tracking error:    {best_fixed['Tracking_Error']:.3%}")

    print(f"\n  Time-varying model (stated_ER + rf * 2 * spread):")
    print(f"    Stated ER:         {TQQQ_STATED_ER:.2%}")
    print(f"    Optimal spread:    {best_spread:.2f}x")
    print(f"    Avg annual cost:   {best_tv['Avg_Annual_Cost']:.2%}")
    print(f"    CAGR difference:   {best_tv['CAGR_diff']:+.4%}")
    print(f"    Tracking error:    {best_tv['Tracking_Error']:.3%}")

    # Decision
    fixed_te = best_fixed["Tracking_Error"]
    tv_te = best_tv["Tracking_Error"]
    fixed_cagr_err = best_fixed["CAGR_abs_diff"]
    tv_cagr_err = best_tv["CAGR_abs_diff"]

    print(f"\n  Comparison:")
    print(f"    Tracking error:  Fixed {fixed_te:.3%} vs TV {tv_te:.3%}")
    print(f"    CAGR error:      Fixed {fixed_cagr_err:.4%} vs TV {tv_cagr_err:.4%}")

    # Check regime stability for fixed ER
    if not regime_df.empty:
        max_regime_fixed = regime_df["Fixed_CAGR_diff"].abs().max()
        max_regime_tv = regime_df["TV_CAGR_diff"].abs().max()
        print(f"    Max regime CAGR diff: Fixed {max_regime_fixed:.3%} vs TV {max_regime_tv:.3%}")

    if fixed_cagr_err < 0.005:  # < 0.5%
        print(f"\n  RECOMMENDATION: Use FIXED ER = {best_fixed_er:.2%}")
        print(f"    CAGR error < 0.5% — fixed model is sufficient for grid search ranking.")
        print(f"    Time-varying model adds complexity without significant ranking improvement.")
        recommended_er = best_fixed_er
    else:
        print(f"\n  NOTE: Fixed ER CAGR error = {fixed_cagr_err:.3%}")
        if tv_cagr_err < fixed_cagr_err * 0.7:
            print(f"    Time-varying model is notably better — consider for high-precision work.")
        print(f"    For grid search ranking purposes, fixed ER = {best_fixed_er:.2%} is still usable.")
        recommended_er = best_fixed_er

    print(f"\n  For leverage_rotation.py Part 12:")
    print(f"    expense_ratio = {recommended_er:.4f}  # TQQQ-calibrated")
    print(f"    tbill_rate = rf_series  # Ken French daily RF (signal=0 periods)")
    print(f"    signal_lag = 1  # realistic execution")
    print(f"    commission = 0.002  # 0.2% per trade")

    return recommended_er


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TQQQ Cost Calibration")
    print("  Synthetic 3x QQQ vs Actual TQQQ")
    print("=" * 70)

    # Section 1: Data
    qqq, tqqq = load_data()

    # Section 2: Fixed ER sweep
    fixed_df, best_fixed_er = fixed_er_sweep(qqq, tqqq)

    # Section 3: Time-varying model
    rf_series = download_ken_french_rf()
    tv_df, best_spread = time_varying_model(qqq, tqqq, rf_series)

    # Section 4: Regime analysis
    regime_df = regime_analysis(qqq, tqqq, rf_series, best_fixed_er, best_spread)

    # Section 5: Visualizations
    actual_cum, fixed_cum, tv_cum = plot_cumulative_comparison(
        qqq, tqqq, best_fixed_er, best_spread, rf_series)
    plot_er_landscape(fixed_df, tv_df)
    plot_rolling_tracking_error(tqqq, fixed_cum, tv_cum)

    # Section 6: Recommendation
    recommended_er = print_recommendation(fixed_df, tv_df, regime_df,
                                           best_fixed_er, best_spread)

    print("\n" + "=" * 70)
    print(f"  Done! Recommended ER = {recommended_er:.4f}")
    print(f"  Charts saved in: {OUT_DIR}")
    print("=" * 70)

    return recommended_er


if __name__ == "__main__":
    main()
