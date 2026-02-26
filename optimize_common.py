"""
Shared infrastructure for Phase 2 optimization scripts.

Common utilities, constants, and plotting functions extracted from
optimize_asymmetric.py to avoid duplication.
"""

import sys
import io
import warnings

# Force UTF-8 output on Windows
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from leverage_rotation import (
    download, download_ken_french_rf,
    signal_dual_ma,
    run_lrs, run_buy_and_hold, calc_metrics, signal_trades_per_year,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Common constants (Part 12 baseline) ──
CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002
SEED = 42


def apply_warmup(sig: pd.Series, warmup_days: int, price_index) -> pd.Series:
    """Force signal=0 during warm-up; if signal is already 1 at warm-up end,
    stay in cash until signal goes to 0 first, then follow normally."""
    sig_mod = sig.copy()
    if warmup_days >= len(price_index):
        sig_mod[:] = 0
        return sig_mod

    warmup_date = price_index[warmup_days]
    sig_mod.loc[:warmup_date] = 0

    if sig.loc[warmup_date] == 1:
        orig_post = sig.loc[warmup_date:]
        first_zero = orig_post[orig_post == 0].index
        if len(first_zero) > 0:
            sig_mod.loc[warmup_date:first_zero[0]] = 0
    return sig_mod


def trim_warmup(cum: pd.Series, warmup_days: int, price_index) -> pd.Series:
    """Trim warm-up period and renormalize to 1.0."""
    warmup_date = price_index[warmup_days]
    trimmed = cum.loc[warmup_date:]
    return trimmed / trimmed.iloc[0]


def run_backtest(price, rf_series, sig_raw, warmup_days):
    """Run a single backtest with warmup handling. Returns (cum_trimmed, metrics)."""
    sig = apply_warmup(sig_raw, warmup_days, price.index)
    cum = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
    cum = trim_warmup(cum, warmup_days, price.index)
    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    return cum, m


def get_symmetric_baselines(price, rf_series, warmup_days):
    """Run best symmetric combos as baselines: (3,161), (3,200), (10,150)."""
    baselines = {}
    for label, (fast, slow) in [("(3,161) eulb", (3, 161)),
                                 ("(3,200) classic", (3, 200)),
                                 ("(10,150)", (10, 150))]:
        sig_raw = signal_dual_ma(price, slow=slow, fast=fast)
        cum, m = run_backtest(price, rf_series, sig_raw, warmup_days)
        tpy = signal_trades_per_year(sig_raw)
        baselines[label] = {"cum": cum, "metrics": m, "fast": fast, "slow": slow,
                            "trades_yr": tpy}
    return baselines


def print_comparison_table(label, metrics_dict):
    """Print a formatted comparison table with Trades/Yr column."""
    print(f"\n{'─' * 90}")
    print(f"  {label}")
    print(f"{'─' * 90}")
    header = (f"  {'Strategy':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MDD':>9} {'Trades/Yr':>10}")
    print(header)
    print(f"  {'─' * 85}")
    for name, m in metrics_dict.items():
        tpy = m.get("Trades/Yr", "")
        tpy_str = f"{tpy:>10.1f}" if isinstance(tpy, (int, float)) else f"{'—':>10}"
        print(f"  {name:<35} {m['CAGR']:>8.2%} {m['Sharpe']:>8.3f} "
              f"{m['Sortino']:>9.3f} {m['MDD']:>9.2%} {tpy_str}")


def plot_cumulative_comparison(curves, title, fname):
    """Cumulative return curves on log scale."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, s in curves.items():
        ax.plot(s.index, s.values, label=label, linewidth=1.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")


def stitch_oos_segments(segments):
    """Chain-link OOS equity segments into a continuous curve.

    Parameters
    ----------
    segments : list of pd.Series
        Each Series is a cumulative return curve (starting at 1.0) for an OOS window.
        Segments should be in chronological order with non-overlapping dates.

    Returns
    -------
    pd.Series
        Stitched cumulative return curve starting at 1.0.
    """
    if not segments:
        return pd.Series(dtype=float)

    stitched = segments[0].copy()
    for seg in segments[1:]:
        scale = stitched.iloc[-1] / seg.iloc[0]
        stitched = pd.concat([stitched, seg.iloc[1:] * scale])
    return stitched


def plot_param_stability(param_data, param_ranges, title, fname, top_n=20):
    """Box plot of parameter distributions.

    Parameters
    ----------
    param_data : dict of str -> list of values
    param_ranges : dict of str -> (min, max) for y-axis limits
    title : str
    fname : str
    """
    n_params = len(param_data)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for ax, (param, vals) in zip(axes, param_data.items()):
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][0].set_alpha(0.6)
        ax.set_title(param, fontsize=12)
        if param in param_ranges:
            ax.set_ylim(param_ranges[param])
        ax.set_xticklabels([])
        ax.grid(axis="y", alpha=0.3)

        if len(vals) > 1:
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            r = param_ranges.get(param, (min(vals), max(vals)))
            full_range = r[1] - r[0]
            pct = iqr / full_range * 100 if full_range > 0 else 0
            ax.text(0.5, 0.02, f"IQR: {iqr:.0f} ({pct:.0f}%)",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    color="red" if pct > 50 else "green")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")


def download_ndx_and_rf():
    """Download NDX price data and Ken French RF series."""
    print("  Downloading NDX data...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    print(f"  NDX: {len(ndx_price)} days ({ndx_price.index[0].date()} -> {ndx_price.index[-1].date()})")
    rf_series = download_ken_french_rf()
    return ndx_price, rf_series
