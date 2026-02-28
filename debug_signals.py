"""
Debug: 신호 시계열 확인 (진입 시점이 정말 역대 최고점인지)
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

STRATEGIES = {
    "P1 peak": (43, 125, 39, 265, 70, 75),
    "P2 peak": (43, 125, 13, 248, 50, 75),
    "P3 peak": (43, 125, 22, 261, 70, 75),
    "P5 peak": (43, 125, 30, 295, 50, 75),
}

print("Debugging Entry Points\n")

ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
rf_series = download_ken_french_rf()

for strat_name, params in sorted(STRATEGIES.items()):
    fl, sl, fh, sh, vl, vt = params

    sig_raw = signal_regime_switching_dual_ma(ndx_price, fl, sl, fh, sh, vl, vt)
    sig = sig_raw.copy()

    wd = ndx_price.index[WARMUP_DAYS]
    sig.loc[:wd] = 0
    if sig_raw.loc[wd] == 1:
        post = sig_raw.loc[wd:]
        fz = post[post == 0].index
        if len(fz) > 0:
            sig.loc[wd:fz[0]] = 0

    # Backtest
    cum = run_lrs(ndx_price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)

    cum = cum.loc[wd:]
    sig = sig.loc[wd:]
    eq = (cum / cum.iloc[0]).values

    # 역대 최고
    running_max = np.maximum.accumulate(eq)
    global_peak = running_max[-1]
    global_peak_idx = np.argmax(running_max == global_peak)

    # 첫 진입
    first_entry_idx = -1
    for i in range(1, len(sig)):
        if sig.iloc[i-1] == 0 and sig.iloc[i] == 1:
            first_entry_idx = i
            break

    if first_entry_idx >= 0:
        first_entry_eq = eq[first_entry_idx]
        print(f"{strat_name}:")
        print(f"  첫 진입: {cum.index[first_entry_idx].date()}, eq={first_entry_eq:.2f}")
        print(f"  역대 최고: {cum.index[global_peak_idx].date()}, eq={running_max[global_peak_idx]:.2f}")
        print(f"  진입이 역대 최고인가? {first_entry_eq >= running_max[global_peak_idx]*0.99}")
        print()
