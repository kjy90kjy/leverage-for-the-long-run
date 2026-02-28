"""
Full Post-Dotcom Comparison Chart (모든 전략 포함)
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
print("  Full Post-Dotcom Comparison (모든 전략 포함)")
print("=" * 70)

print("\n[1/2] Downloading data...")
ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
rf_series = download_ken_french_rf()
ndx_postdot = ndx_price.loc[POSTDOT_START:]

print(f"  Post-dotcom: {ndx_postdot.index[0].date()} ~ {ndx_postdot.index[-1].date()}")

print("\n[2/2] Running all strategies for post-dotcom period...")

all_curves_postdot = {}

# 모든 전략 (peak + robust 포함)
strategies = {
    "P5 Optuna": (48, 323, 15, 229, 49, 57.3),
    "P1 peak": (43, 125, 39, 265, 70, 75),
    "P1 robust": (46, 130, 46, 259, 70, 75),
    "P2 peak": (43, 125, 13, 248, 50, 75),
    "P2 robust": (46, 134, 22, 252, 50, 75),
    "P3 peak": (43, 125, 22, 261, 70, 75),
    "P3 robust": (46, 131, 22, 258, 70, 75),
    "P4 peak": (44, 122, 2, 303, 110, 75),
    "P4 robust": (42, 123, 6, 300, 110, 75),
    "P5 peak": (43, 125, 30, 295, 50, 75),
    "P5 robust": (46, 134, 27, 292, 50, 75),
    "Sym (3,161)": None,
    "B&H 3x": None,
}

for name, params in strategies.items():
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
        all_curves_postdot[name] = cum
    elif name == "B&H 3x":
        bh3x = run_buy_and_hold(ndx_postdot, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
        wd = ndx_postdot.index[WARMUP_DAYS]
        bh3x = bh3x.loc[wd:]; bh3x = bh3x / bh3x.iloc[0]
        all_curves_postdot[name] = bh3x
    else:
        fl, sl, fh, sh, vl, vt = params
        cum, sig, m = full_backtest(ndx_postdot, rf_series, fl, sl, fh, sh, vl, vt)
        all_curves_postdot[name] = cum

# Plotting
fig, ax = plt.subplots(figsize=(16, 9))
cmap = plt.cm.tab20
n_curves = len(all_curves_postdot)

for i, (label, cum) in enumerate(all_curves_postdot.items()):
    is_baseline = "Sym" in label or "B&H" in label or "Optuna" in label
    ls = "--" if is_baseline else "-"
    lw = 1.5 if is_baseline else 2.0
    ax.plot(cum.index, cum.values, label=label, linestyle=ls, linewidth=lw,
            color=cmap(i / max(n_curves - 1, 1)))

ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.set_title(f"Post-Dotcom Backtest Comparison - All Strategies (2003-2025)\nNDX 3x, ER=3.5%, lag=1, comm=0.2%",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Growth of $1 (log scale)", fontsize=11)
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig.savefig(OUT_DIR / "regime_grid_v2_comparison_postdot_full.png", dpi=150)
plt.close(fig)
print("  -> regime_grid_v2_comparison_postdot_full.png")

print("\n" + "=" * 70)
print("  Done!")
print("=" * 70)
