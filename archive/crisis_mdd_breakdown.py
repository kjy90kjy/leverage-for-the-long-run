"""
Crisis-by-Crisis MDD & MDD_Entry Breakdown for Peak Strategies
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
from pathlib import Path

from leverage_rotation import (
    download, download_ken_french_rf,
    signal_regime_switching_dual_ma,
    run_lrs, _max_entry_drawdown, _max_recovery_days,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002
WARMUP_DAYS = 500

# 크라이시스 정의
CRISES = [
    ("1987 Black Monday", "1987-09-01", "1988-06-30"),
    ("1998 LTCM", "1998-07-01", "1998-12-31"),
    ("2000-02 Dot-com", "2000-01-01", "2003-06-30"),
    ("2007-09 GFC", "2007-07-01", "2009-03-31"),
    ("2018 Q4", "2018-08-01", "2019-01-31"),
    ("2020 COVID", "2020-02-01", "2020-06-30"),
    ("2022 Bear", "2021-11-01", "2023-03-31"),
    ("2024 Aug Dip", "2024-07-01", "2024-12-31"),
    ("2025 Liberation Day", "2025-02-01", "2025-06-30"),
]

# 전략 정의 (P1/P2/P3 peaks)
STRATEGIES = {
    "P1 peak": (43, 125, 39, 265, 70, 75),
    "P2 peak": (43, 125, 13, 248, 50, 75),
    "P3 peak": (43, 125, 22, 261, 70, 75),
    "P5 peak": (43, 125, 30, 295, 50, 75),
}

def calc_crisis_mdd_entry(price_slice, sig_slice):
    """Calculate MDD and MDD_Entry for a price/signal slice."""
    if len(price_slice) < 2:
        return 0.0, 0.0

    # Simple equity curve
    daily_ret = np.diff(np.log(price_slice))
    cum_ret = np.concatenate([[1.0], np.exp(np.cumsum(daily_ret))])

    # MDD (traditional)
    running_max = np.maximum.accumulate(cum_ret)
    mdd = np.min(cum_ret / running_max - 1)

    # MDD_Entry
    entry_max = cum_ret.copy()
    for i in range(len(sig_slice)):
        if sig_slice.iloc[i] == 1.0:  # Entry signal
            entry_max[i:] = np.maximum(entry_max[i:], cum_ret[i])
    mdd_entry = np.min(cum_ret / entry_max - 1)

    return max(mdd, -0.99), max(mdd_entry, -0.99)

print("=" * 80)
print("  Crisis MDD Breakdown: P1/P2/P3/P5 Peak Strategies")
print("=" * 80)

print("\n[1/2] Downloading data...")
ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
rf_series = download_ken_french_rf()

# Prepare results
results = []

print("\n[2/2] Analyzing crises...\n")

for crisis_name, start_date, end_date in CRISES:
    price_slice = ndx_price.loc[start_date:end_date]

    if len(price_slice) < 2:
        continue

    print(f"\n{crisis_name} ({start_date} ~ {end_date})")
    print("─" * 80)
    print(f"  {'Strategy':<12} {'MDD':>10} {'MDD_Entry':>10}")
    print("─" * 80)

    for strat_name, params in STRATEGIES.items():
        fl, sl, fh, sh, vl, vt = params

        # Generate signal
        sig_raw = signal_regime_switching_dual_ma(price_slice, fl, sl, fh, sh, vl, vt)
        sig = sig_raw.copy()

        # Apply warmup
        if WARMUP_DAYS < len(price_slice):
            wd = price_slice.index[min(WARMUP_DAYS, len(price_slice)-1)]
            sig.loc[:wd] = 0
            if sig_raw.loc[wd] == 1:
                post = sig_raw.loc[wd:]
                fz = post[post == 0].index
                if len(fz) > 0:
                    sig.loc[wd:fz[0]] = 0

        # Run backtest
        cum = run_lrs(price_slice, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                      tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)

        # Trim warmup
        if WARMUP_DAYS < len(cum):
            wd = price_slice.index[min(WARMUP_DAYS, len(price_slice)-1)]
            cum = cum.loc[wd:]

        if len(cum) < 2:
            mdd, mdd_entry = 0.0, 0.0
        else:
            cum_norm = cum / cum.iloc[0]

            # MDD
            running_max = np.maximum.accumulate(cum_norm.values)
            mdd = float(np.min(cum_norm.values / running_max - 1))

            # MDD_Entry (from entry signal timing)
            # Align sig and cum_norm by date
            sig_aligned = sig.loc[cum_norm.index]
            entry_max = cum_norm.values.copy()
            for i in range(len(cum_norm)):
                if i < len(sig_aligned) and sig_aligned.iloc[i] == 1.0:
                    entry_max[i:] = np.maximum(entry_max[i:], cum_norm.iloc[i])
            mdd_entry = float(np.min(cum_norm.values / entry_max - 1))

        print(f"  {strat_name:<12} {mdd:>10.1%} {mdd_entry:>10.1%}")
        results.append({
            'Crisis': crisis_name,
            'Strategy': strat_name,
            'MDD': mdd,
            'MDD_Entry': mdd_entry,
        })

# Save to CSV
results_df = pd.DataFrame(results)
csv_path = OUT_DIR / "crisis_mdd_breakdown.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n\n  -> {csv_path.name}")

# Create heatmap
print("\n[3/3] Creating heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, metric in enumerate(['MDD', 'MDD_Entry']):
    pivot = results_df.pivot(index='Strategy', columns='Crisis', values=metric)

    # Reorder columns chronologically
    crisis_order = [c[0] for c in CRISES]
    pivot = pivot[[c for c in crisis_order if c in pivot.columns]]

    ax = axes[idx]
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=-1.0, vmax=0.0)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            text = ax.text(j, i, f'{val:.1%}', ha="center", va="center",
                          color="white" if val < -0.5 else "black", fontsize=8)

    ax.set_title(f'{metric} by Crisis', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, format='%.0%')

plt.suptitle('Peak Strategies: Crisis-by-Crisis MDD Analysis', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT_DIR / "crisis_mdd_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  -> crisis_mdd_heatmap.png")

print("\n" + "=" * 80)
print("  Done!")
print("=" * 80)
