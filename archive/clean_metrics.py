"""
Clean, Single-Source Metrics
모든 지표를 run_lrs() + calc_metrics()로만 계산
시각적 검증 포함
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
import matplotlib.patches as mpatches
from pathlib import Path

from leverage_rotation import (
    download, download_ken_french_rf,
    signal_regime_switching_dual_ma,
    run_lrs, calc_metrics, _max_recovery_days,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

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

print("=" * 90)
print("  Clean Metrics: Single Source (run_lrs + calc_metrics)")
print("=" * 90)

print("\n[1/3] Downloading...")
ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
rf_series = download_ken_french_rf()

print("\n[2/3] Computing metrics (full period 1987-2025)...\n")

all_results = {}
all_curves = {}
worst_dd_info = {}

for strat_name, params in sorted(STRATEGIES.items()):
    fl, sl, fh, sh, vl, vt = params

    # Signal
    sig_raw = signal_regime_switching_dual_ma(ndx_price, fl, sl, fh, sh, vl, vt)
    sig = sig_raw.copy()

    # Warmup
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

    # Trim warmup
    cum = cum.loc[wd:]
    sig = sig.loc[wd:]
    cum_norm = cum / cum.iloc[0]

    # Metrics (from leverage_rotation.py)
    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum_norm, tbill_rate=rf_scalar, rf_series=rf_series)
    m["Max_Recovery_Days"] = _max_recovery_days(cum_norm)

    all_results[strat_name] = m
    all_curves[strat_name] = cum_norm

    # Find worst drawdown (single biggest drop from running max)
    eq = cum_norm.values
    running_max = np.maximum.accumulate(eq)
    drawdown = eq / running_max - 1

    worst_idx = np.argmin(drawdown)
    worst_dd = drawdown[worst_idx]

    # Find entry point (last local max before worst)
    entry_idx = np.argmax(running_max[:worst_idx+1])

    # Calculate Entry MDD (진입 시점 대비 최악 낙폭)
    # 신호가 0→1로 바뀔 때마다 그 시점의 equity를 진입가로 사용
    # 그 이후 최악 낙폭을 각 진입별로 계산
    # 모든 진입 중 최악을 Entry MDD로 사용

    entry_mdd = 0.0
    worst_entry_mdd_idx = -1
    entry_price = 0.0
    entry_idx = -1

    # 각 진입(0→1)별로 그 시점 이후의 최악 낙폭 계산
    for i in range(1, len(sig)):
        if sig.iloc[i-1] == 0 and sig.iloc[i] == 1:  # 신호가 0→1 (진입)
            entry_price = eq[i]
            entry_idx = i

            # 이 진입 이후의 최악 낙폭 찾기
            for j in range(i, len(eq)):
                dd = eq[j] / entry_price - 1
                if dd < entry_mdd:
                    entry_mdd = dd
                    worst_entry_mdd_idx = j

    # Entry Recovery Days: 최악에서 진입가까지 회복되는 일수
    entry_recovery_days = 0
    if worst_entry_mdd_idx >= 0 and entry_price > 0:
        for j in range(worst_entry_mdd_idx + 1, len(eq)):
            if eq[j] >= entry_price:
                entry_recovery_days = (cum_norm.index[j] - cum_norm.index[worst_entry_mdd_idx]).days
                break

    worst_dd_info[strat_name] = {
        'worst_dd': worst_dd,
        'worst_date': cum_norm.index[worst_idx],
        'worst_eq': eq[worst_idx],
        'entry_date': cum_norm.index[entry_idx],
        'entry_eq': eq[entry_idx],
        'days': (cum_norm.index[worst_idx] - cum_norm.index[entry_idx]).days,
        'entry_mdd': entry_mdd,
        'entry_mdd_date': cum_norm.index[worst_entry_mdd_idx] if worst_entry_mdd_idx >= 0 else None,
        'entry_price': entry_price,
        'entry_recovery_days': entry_recovery_days,
    }

    print(f"{strat_name:<15} CAGR={m['CAGR']:>6.1%} | Sharpe={m['Sharpe']:>6.3f} | Sortino={m['Sortino']:>6.3f} | " +
          f"MDD={m['MDD']:>7.1%} | Recovery={m['Max_Recovery_Days']:>5.0f}d")

# Print detailed worst drawdown
print("\n" + "=" * 100)
print("  Drawdown Metrics (계좌 기준)")
print("=" * 100)
print(f"\n{'Strategy':<15} {'MDD':>10} {'Entry MDD':>12} {'Entry Recover':>15} {'MDD Date':>16}")
print("─" * 100)

for name in sorted(worst_dd_info.keys()):
    info = worst_dd_info[name]
    m = all_results[name]
    recover_str = f"{info['entry_recovery_days']}d" if info['entry_recovery_days'] > 0 else "Never"
    print(f"{name:<15} {m['MDD']:>10.1%} {info['entry_mdd']:>12.1%} {recover_str:>15} " +
          f"{info['worst_date'].strftime('%Y-%m-%d'):>16}")

# Ranking
print("\n" + "=" * 90)
print("  Ranking by Key Metrics")
print("=" * 90)

metrics_to_rank = ['CAGR', 'Sharpe', 'Sortino', 'MDD']

for metric in metrics_to_rank:
    is_mdd = (metric == 'MDD')
    ranked = sorted(all_results.items(),
                   key=lambda x: x[1][metric],
                   reverse=not is_mdd)

    print(f"\n  {metric}:")
    for i, (name, m) in enumerate(ranked, 1):
        print(f"    {i}. {name:<15} {m[metric]:>8.1%}" if metric != 'Sharpe' else
              f"    {i}. {name:<15} {m[metric]:>8.3f}")

# CSV output
df = pd.DataFrame(all_results).T
df.to_csv(OUT_DIR / "clean_metrics.csv")
print(f"\n  -> clean_metrics.csv")

print("\n[3/3] Creating visualization...\n")

# Create 4-panel plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Strategy Comparison: Full Period (1987-2025)", fontsize=14, fontweight='bold')

# Panel 1: Equity curves (log scale)
ax = axes[0, 0]
colors = plt.cm.tab10(np.linspace(0, 1, len(all_curves)))
for (name, curve), color in zip(all_curves.items(), colors):
    ax.plot(curve.index, curve.values, label=name, linewidth=2, color=color)
ax.set_yscale('log')
ax.set_ylabel('Growth of $1 (log)', fontsize=10)
ax.set_title('Cumulative Returns (log scale)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Drawdown over time
ax = axes[0, 1]
for (name, curve), color in zip(all_curves.items(), colors):
    eq = curve.values
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1
    ax.plot(curve.index, dd * 100, label=name, linewidth=1.5, color=color)
ax.set_ylabel('Drawdown (%)', fontsize=10)
ax.set_title('Drawdown Over Time', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Panel 3: Risk-Return scatter
ax = axes[1, 0]
for name, color in zip(sorted(all_results.keys()), colors):
    m = all_results[name]
    ax.scatter(m['Volatility'] * 100, m['CAGR'] * 100, s=300, color=color, alpha=0.7,
              edgecolors='black', linewidth=1.5)
    ax.annotate(name, (m['Volatility'] * 100, m['CAGR'] * 100),
               fontsize=9, ha='center', va='center', fontweight='bold')
ax.set_xlabel('Volatility (%)', fontsize=10)
ax.set_ylabel('CAGR (%)', fontsize=10)
ax.set_title('Risk-Return Profile', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 4: Key metrics table
ax = axes[1, 1]
ax.axis('off')

table_data = []
for name in sorted(all_results.keys()):
    m = all_results[name]
    worst = worst_dd_info[name]
    recover_str = f"{worst['entry_recovery_days']:.0f}d" if worst['entry_recovery_days'] > 0 else "Never"
    table_data.append([
        name,
        f"{m['CAGR']:.1%}",
        f"{m['Sharpe']:.3f}",
        f"{m['Sortino']:.3f}",
        f"{m['MDD']:.1%}",
        f"{worst['entry_mdd']:.1%}",
        recover_str,
    ])

cols = ['Strategy', 'CAGR', 'Sharpe', 'Sortino', 'MDD', 'Entry MDD', 'Entry Recovery']
table = ax.table(cellText=table_data, colLabels=cols, cellLoc='center', loc='center',
                colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(len(cols)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data) + 1):
    for j in range(len(cols)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#F2F2F2')

plt.tight_layout()
fig.savefig(OUT_DIR / "clean_metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  -> clean_metrics_comparison.png")

print("\n" + "=" * 90)
print("  Done! All metrics from single source (run_lrs + calc_metrics)")
print("=" * 90)
