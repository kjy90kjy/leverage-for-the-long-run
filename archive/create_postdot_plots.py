"""
Post-Dotcom Analysis Visualizations
- Metrics comparison table as image
- Post-dotcom only backtest comparison chart
"""

import sys
import io
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

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
    _max_entry_drawdown, _max_recovery_days,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002
WARMUP_DAYS = 500
POSTDOT_START = "2003-01-01"

def full_backtest(price, rf_series, fl, sl, fh, sh, vl, vt):
    """Run full backtest using leverage_rotation API."""
    sig_raw = signal_regime_switching_dual_ma(price, fl, sl, fh, sh, vl, vt)

    sig = sig_raw.copy()
    if WARMUP_DAYS < len(price):
        wd = price.index[WARMUP_DAYS]
        sig.loc[:wd] = 0
        if sig_raw.loc[wd] == 1:
            post = sig_raw.loc[wd:]
            fz = post[post == 0].index
            if len(fz) > 0:
                sig.loc[wd:fz[0]] = 0

    cum = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
    wd = price.index[WARMUP_DAYS]
    cum = cum.loc[wd:]
    cum = cum / cum.iloc[0]
    sig = sig.loc[wd:]

    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    m["MDD_Entry"] = _max_entry_drawdown(cum, sig, signal_lag=SIGNAL_LAG)
    m["Max_Recovery_Days"] = _max_recovery_days(cum)
    m["Trades/Yr"] = signal_trades_per_year(sig)
    return cum, sig, m

print("=" * 70)
print("  Post-Dotcom Analysis Visualizations")
print("=" * 70)

print("\n[1/3] Downloading data...")
ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
rf_series = download_ken_french_rf()

# Full period
print(f"  Full period: {ndx_price.index[0].date()} ~ {ndx_price.index[-1].date()}")

# Post-dotcom
ndx_postdot = ndx_price.loc[POSTDOT_START:]
print(f"  Post-dotcom: {ndx_postdot.index[0].date()} ~ {ndx_postdot.index[-1].date()}")

print("\n[2/3] Running backtests for visualization...")

# Best strategies (from grid search results)
strategies = {
    "P5 Optuna": {"params": (48, 323, 15, 229, 49, 57.3), "price_range": ndx_price},
    "P1 peak": {"params": (43, 125, 39, 265, 70, 75), "price_range": ndx_price},
    "P2 peak": {"params": (43, 125, 13, 248, 50, 75), "price_range": ndx_price},
    "P3 peak": {"params": (43, 125, 22, 261, 70, 75), "price_range": ndx_price},
    "Sym (3,161)": {"params": None, "price_range": ndx_price},
    "B&H 3x": {"params": None, "price_range": ndx_price},
}

all_results_full = {}
all_results_postdot = {}
all_curves_postdot = {}

# Full period
for name, config in strategies.items():
    if name == "Sym (3,161)":
        sig_raw = signal_dual_ma(ndx_price, slow=161, fast=3)
        sig = sig_raw.copy()
        wd = ndx_price.index[WARMUP_DAYS]
        sig.loc[:wd] = 0
        if sig_raw.loc[wd] == 1:
            post = sig_raw.loc[wd:]
            fz = post[post == 0].index
            if len(fz) > 0:
                sig.loc[wd:fz[0]] = 0
        cum = run_lrs(ndx_price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                      tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
        cum = cum.loc[wd:]; cum = cum / cum.iloc[0]
        sig = sig.loc[wd:]
        rf_scalar = rf_series.mean() * 252
        m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
        m["MDD_Entry"] = _max_entry_drawdown(cum, sig, signal_lag=SIGNAL_LAG)
        m["Max_Recovery_Days"] = _max_recovery_days(cum)
        m["Trades/Yr"] = signal_trades_per_year(sig)
        all_results_full[name] = m
    elif name == "B&H 3x":
        bh3x = run_buy_and_hold(ndx_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
        wd = ndx_price.index[WARMUP_DAYS]
        bh3x = bh3x.loc[wd:]; bh3x = bh3x / bh3x.iloc[0]
        rf_scalar = rf_series.mean() * 252
        m = calc_metrics(bh3x, tbill_rate=rf_scalar, rf_series=rf_series)
        m["MDD_Entry"] = m["MDD"]
        m["Max_Recovery_Days"] = _max_recovery_days(bh3x)
        m["Trades/Yr"] = 0.0
        all_results_full[name] = m
    else:
        fl, sl, fh, sh, vl, vt = config["params"]
        cum, sig, m = full_backtest(ndx_price, rf_series, fl, sl, fh, sh, vl, vt)
        all_results_full[name] = m

# Post-dotcom
for name, config in strategies.items():
    if name == "Sym (3,161)":
        sig_raw = signal_dual_ma(ndx_postdot, slow=161, fast=3)
        sig = sig_raw.copy()
        wd = ndx_postdot.index[WARMUP_DAYS]
        sig.loc[:wd] = 0
        if sig_raw.loc[wd] == 1:
            post = sig_raw.loc[wd:]
            fz = post[post == 0].index
            if len(fz) > 0:
                sig.loc[wd:fz[0]] = 0
        cum = run_lrs(ndx_postdot, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                      tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
        cum = cum.loc[wd:]; cum = cum / cum.iloc[0]
        sig = sig.loc[wd:]
        rf_scalar = rf_series.mean() * 252
        m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
        m["MDD_Entry"] = _max_entry_drawdown(cum, sig, signal_lag=SIGNAL_LAG)
        m["Max_Recovery_Days"] = _max_recovery_days(cum)
        m["Trades/Yr"] = signal_trades_per_year(sig)
        all_results_postdot[name] = m
        all_curves_postdot[name] = cum
    elif name == "B&H 3x":
        bh3x = run_buy_and_hold(ndx_postdot, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
        wd = ndx_postdot.index[WARMUP_DAYS]
        bh3x = bh3x.loc[wd:]; bh3x = bh3x / bh3x.iloc[0]
        rf_scalar = rf_series.mean() * 252
        m = calc_metrics(bh3x, tbill_rate=rf_scalar, rf_series=rf_series)
        m["MDD_Entry"] = m["MDD"]
        m["Max_Recovery_Days"] = _max_recovery_days(bh3x)
        m["Trades/Yr"] = 0.0
        all_results_postdot[name] = m
        all_curves_postdot[name] = bh3x
    else:
        fl, sl, fh, sh, vl, vt = config["params"]
        cum, sig, m = full_backtest(ndx_postdot, rf_series, fl, sl, fh, sh, vl, vt)
        all_results_postdot[name] = m
        all_curves_postdot[name] = cum

print("\n[3/3] Generating visualizations...")

# ══════════════════════════════════════════════════════════════
# 1. METRICS TABLE AS IMAGE
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('tight')
ax.axis('off')

# Prepare data for table
table_data = []
for name in all_results_full.keys():
    m_full = all_results_full[name]
    m_pd = all_results_postdot[name]

    table_data.append([
        name,
        f"{m_full['CAGR']:.1%}",
        f"{m_pd['CAGR']:.1%}",
        f"{m_full['Sharpe']:.3f}",
        f"{m_pd['Sharpe']:.3f}",
        f"{m_full['Sortino']:.3f}",
        f"{m_pd['Sortino']:.3f}",
        f"{m_full['MDD']:.1%}",
        f"{m_pd['MDD']:.1%}",
        f"{m_full['MDD_Entry']:.1%}",
        f"{m_pd['MDD_Entry']:.1%}",
        f"{m_full['Volatility']:.1%}",
        f"{m_pd['Volatility']:.1%}",
    ])

columns = [
    "Strategy",
    "CAGR\n(Full)", "CAGR\n(PD)",
    "Sharpe\n(Full)", "Sharpe\n(PD)",
    "Sortino\n(Full)", "Sortino\n(PD)",
    "MDD\n(Full)", "MDD\n(PD)",
    "MDD_Entry\n(Full)", "MDD_Entry\n(PD)",
    "Vol\n(Full)", "Vol\n(PD)",
]

table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                colWidths=[0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(len(columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#F2F2F2')

plt.title("NDX 3x Regime-Switching Strategies: Full Period vs Post-Dotcom Comparison\n(1987-2025 vs 2003-2025)",
          fontsize=14, fontweight='bold', pad=20)

fig.savefig(OUT_DIR / "regime_grid_v2_metrics_table.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  -> regime_grid_v2_metrics_table.png")

# ══════════════════════════════════════════════════════════════
# 2. POST-DOTCOM ONLY COMPARISON CHART
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(16, 9))
cmap = plt.cm.tab10
n_curves = len(all_curves_postdot)

for i, (label, cum) in enumerate(all_curves_postdot.items()):
    is_baseline = "Sym" in label or "B&H" in label or "Optuna" in label
    ls = "--" if is_baseline else "-"
    lw = 1.5 if is_baseline else 2.0
    ax.plot(cum.index, cum.values, label=label, linestyle=ls, linewidth=lw,
            color=cmap(i / max(n_curves - 1, 1)))

ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.set_title(f"Post-Dotcom Backtest Comparison (2003-2025)\nNDX 3x, ER=3.5%, lag=1, comm=0.2%",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Growth of $1 (log scale)", fontsize=11)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig.savefig(OUT_DIR / "regime_grid_v2_comparison_postdot.png", dpi=150)
plt.close(fig)
print("  -> regime_grid_v2_comparison_postdot.png")

print("\n" + "=" * 70)
print("  Done! Visualizations created successfully")
print("=" * 70)
