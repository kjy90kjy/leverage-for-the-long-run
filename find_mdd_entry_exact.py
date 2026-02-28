"""
Find exact date and context when P1/P2 hit MDD_Entry
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
from pathlib import Path

from leverage_rotation import (
    download, download_ken_french_rf,
    signal_regime_switching_dual_ma,
    run_lrs,
)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002
WARMUP_DAYS = 500

# 전략 정의
STRATEGIES = {
    "P1 peak": (43, 125, 39, 265, 70, 75),
    "P2 peak": (43, 125, 13, 248, 50, 75),
    "P3 peak": (43, 125, 22, 261, 70, 75),
}

print("=" * 90)
print("  Finding exact dates when MDD_Entry was hit")
print("=" * 90)

print("\n[1/2] Downloading data...")
ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
rf_series = download_ken_french_rf()

print("\n[2/2] Tracing MDD_Entry for each strategy...\n")

for strat_name, params in STRATEGIES.items():
    print(f"\n{'='*90}")
    print(f"  {strat_name}: {params}")
    print(f"{'='*90}")

    fl, sl, fh, sh, vl, vt = params

    # Generate signal
    sig_raw = signal_regime_switching_dual_ma(ndx_price, fl, sl, fh, sh, vl, vt)
    sig = sig_raw.copy()

    # Apply warmup
    wd = ndx_price.index[WARMUP_DAYS]
    sig.loc[:wd] = 0
    if sig_raw.loc[wd] == 1:
        post = sig_raw.loc[wd:]
        fz = post[post == 0].index
        if len(fz) > 0:
            sig.loc[wd:fz[0]] = 0

    # Run backtest
    cum = run_lrs(ndx_price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)

    # Trim warmup
    cum = cum.loc[wd:]
    cum_norm = cum / cum.iloc[0]
    sig = sig.loc[wd:]

    # Calculate entry-based MDD with tracking
    equity_vals = cum_norm.values
    entry_max = equity_vals.copy()
    entry_date = np.full(len(equity_vals), None, dtype=object)
    entry_date[0] = cum_norm.index[0]

    worst_mdd_entry = 0.0
    worst_date = None
    worst_entry_date = None
    worst_eq_val = None

    for i in range(len(sig)):
        if sig.iloc[i] == 1.0:  # Signal changed to 1 (entered)
            entry_max[i:] = np.maximum(entry_max[i:], equity_vals[i])
            entry_date[i:] = cum_norm.index[i]

        # Calculate current MDD_Entry
        mdd_e = equity_vals[i] / entry_max[i] - 1
        if mdd_e < worst_mdd_entry:
            worst_mdd_entry = mdd_e
            worst_date = cum_norm.index[i]
            worst_entry_date = entry_date[i]
            worst_eq_val = equity_vals[i]

    print(f"\n  Full period MDD_Entry: {worst_mdd_entry:.1%}")
    print(f"  Worst hit date: {worst_date.date()}")
    print(f"  Entry date: {worst_entry_date.date()}")
    print(f"  Entry equity: {entry_max[np.where(cum_norm.index == worst_entry_date)[0][0]]:.4f}")
    print(f"  Worst equity: {worst_eq_val:.4f}")
    print(f"  Period: {worst_entry_date.date()} → {worst_date.date()}")

    # Check surrounding signals
    worst_idx = np.where(cum_norm.index == worst_date)[0][0]
    start_idx = max(0, worst_idx - 30)
    end_idx = min(len(cum_norm), worst_idx + 30)

    print(f"\n  Signal context ({worst_idx-start_idx}d before ~ {end_idx-worst_idx}d after):")
    print(f"  Date         │ Price │ Signal │ MDD_Entry")
    print("  " + "─" * 50)

    for j in range(start_idx, end_idx):
        marker = " ← WORST" if j == worst_idx else ""
        mdd_e = equity_vals[j] / entry_max[j] - 1
        print(f"  {cum_norm.index[j].date()} │ {equity_vals[j]:>6.3f} │  {sig.iloc[j]:>5.0f} │ {mdd_e:>8.1%}{marker}")

print("\n" + "=" * 90)
print("  Done!")
print("=" * 90)
