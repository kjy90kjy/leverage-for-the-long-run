"""
Multi-Resolution Grid Search: Regime-Switching 6-Param Plateau Exploration

Phase 1: Coarse grid (step 10) — ~720k combos
Phase 2: Fine grid (step 1, ±5) around top plateaus
Phase 3: Full backtest comparison

Key optimisation: all MAs precomputed, pure numpy backtest, no pandas overhead.

Usage:
    python optimize_regime_grid.py
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

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from itertools import product
import numba

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
ALPHA = 0.02  # penalty coefficient


# ══════════════════════════════════════════════════════════════
# 1. PRECOMPUTATION
# ══════════════════════════════════════════════════════════════

def precompute_mas(prices_arr, max_ma=350):
    """Precompute SMA for all window lengths 1..max_ma. Returns dict[int -> np.ndarray]."""
    n = len(prices_arr)
    cumsum = np.concatenate([[0.0], np.cumsum(prices_arr)])
    mas = {}
    for w in range(1, max_ma + 1):
        ma = np.full(n, np.nan)
        ma[w - 1:] = (cumsum[w:] - cumsum[:-w]) / w
        mas[w] = ma
    return mas


def precompute_vol_regimes(daily_ret_arr, price_index,
                           vol_lookbacks, vol_thresholds):
    """Precompute high_vol boolean array for each (lookback, threshold) combo.
    Returns dict[(lookback, threshold) -> np.ndarray of bool]."""
    regimes = {}
    ret_series = pd.Series(daily_ret_arr, index=price_index)
    for lb in vol_lookbacks:
        rolling_vol = ret_series.rolling(lb).std() * np.sqrt(252)
        vol_pct = rolling_vol.expanding().rank(pct=True) * 100
        vol_pct_vals = vol_pct.values
        for th in vol_thresholds:
            high_vol = np.zeros(len(daily_ret_arr), dtype=bool)
            valid = ~np.isnan(vol_pct_vals)
            high_vol[valid] = vol_pct_vals[valid] >= th
            regimes[(lb, th)] = high_vol
    return regimes


# ══════════════════════════════════════════════════════════════
# 2. FAST SIGNAL + BACKTEST (pure numpy)
# ══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def fast_regime_signal(ma_fast_low, ma_slow_low, ma_fast_high, ma_slow_high,
                       high_vol, warmup):
    """State-machine signal for regime-switching dual MA, pure numpy input/output."""
    n = len(ma_fast_low)
    sig = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(warmup, n):
        if np.isnan(ma_slow_low[i]) or np.isnan(ma_slow_high[i]):
            continue
        if high_vol[i]:
            buy_cond = ma_fast_high[i] > ma_slow_high[i]
            sell_cond = ma_fast_high[i] <= ma_slow_high[i]
        else:
            buy_cond = ma_fast_low[i] > ma_slow_low[i]
            sell_cond = ma_fast_low[i] <= ma_slow_low[i]
        if state == 0 and buy_cond:
            state = 1
        elif state == 1 and sell_cond:
            state = 0
        sig[i] = state
    return sig


@numba.njit(cache=True)
def fast_eval(sig_raw, lev_ret, tbill_daily, commission, alpha):
    """Compute penalised Sortino from raw signal (applies lag=1 internally).
    Returns (sortino, trades_per_year, penalised)."""
    n = len(sig_raw)
    # Apply lag=1: shift signal forward by 1
    sig = np.empty(n, dtype=np.float64)
    sig[0] = 0.0
    for i in range(1, n):
        sig[i] = float(sig_raw[i - 1])

    # Strategy returns
    sum_strat = 0.0
    sum_excess = 0.0
    sum_down_sq = 0.0
    total_trades = 0.0
    prev_sig = 0.0
    for i in range(n):
        trade = abs(sig[i] - prev_sig)
        total_trades += trade
        sr = sig[i] * lev_ret[i] + (1.0 - sig[i]) * tbill_daily[i] - trade * commission
        ex = sr - tbill_daily[i]
        sum_excess += ex
        if ex < 0:
            sum_down_sq += ex * ex
        prev_sig = sig[i]

    dd = np.sqrt(sum_down_sq / n) * np.sqrt(252.0)
    mean_excess_annual = (sum_excess / n) * 252.0
    sortino = mean_excess_annual / dd if dd > 0.0 else 0.0

    trades_yr = total_trades / (n / 252.0)
    penalised = sortino - alpha * trades_yr
    return sortino, trades_yr, penalised


# ══════════════════════════════════════════════════════════════
# 3. COARSE GRID SEARCH
# ══════════════════════════════════════════════════════════════

def run_coarse_grid(prices_arr, daily_ret_arr, price_index,
                    mas_dict, regimes_dict,
                    lev_ret, tbill_daily):
    """Phase 1: coarse grid sweep."""
    fast_vals = list(range(2, 51, 10))   # [2, 12, 22, 32, 42]
    if 50 not in fast_vals:
        fast_vals.append(50)             # ensure 50 included → 6 values
    slow_vals = list(range(50, 351, 10))  # 31 values
    vol_lookbacks = [20, 40, 60, 80, 100, 120]
    vol_thresholds = [30.0, 40.0, 50.0, 60.0, 70.0]

    # Count valid combos (fast < slow)
    total = 0
    combos = []
    for fl, sl, fh, sh in product(fast_vals, slow_vals, fast_vals, slow_vals):
        if fl >= sl or fh >= sh:
            continue
        for vl, vt in product(vol_lookbacks, vol_thresholds):
            combos.append((fl, sl, fh, sh, vl, vt))
            total += 1

    print(f"  Coarse grid: {total:,} combos")

    # Dtype for results
    dt = np.dtype([
        ('fast_low', np.int16), ('slow_low', np.int16),
        ('fast_high', np.int16), ('slow_high', np.int16),
        ('vol_lookback', np.int16), ('vol_threshold_pct', np.float32),
        ('sortino', np.float32), ('trades_yr', np.float32),
        ('penalised', np.float32),
    ])
    results = np.empty(total, dtype=dt)

    t0 = time.time()
    report_interval = max(total // 20, 1)

    for idx, (fl, sl, fh, sh, vl, vt) in enumerate(combos):
        warmup = max(sl, sh, vl)
        sig = fast_regime_signal(
            mas_dict[fl], mas_dict[sl],
            mas_dict[fh], mas_dict[sh],
            regimes_dict[(vl, vt)], warmup
        )
        sortino, tpy, pen = fast_eval(sig, lev_ret, tbill_daily, COMMISSION, ALPHA)

        results[idx] = (fl, sl, fh, sh, vl, vt, sortino, tpy, pen)

        if (idx + 1) % report_interval == 0:
            elapsed = time.time() - t0
            pct = (idx + 1) / total * 100
            speed = (idx + 1) / elapsed
            eta = (total - idx - 1) / speed
            print(f"    {pct:5.1f}% ({idx+1:>7,}/{total:,}) "
                  f"| {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining "
                  f"| {speed:,.0f} trial/s")

    elapsed = time.time() - t0
    print(f"  Coarse grid done: {elapsed:.1f}s ({total/elapsed:,.0f} trial/s)")
    return results


# ══════════════════════════════════════════════════════════════
# 4. PLATEAU IDENTIFICATION
# ══════════════════════════════════════════════════════════════

def identify_plateaus(results, n_plateaus=5):
    """Find plateau centres from coarse grid results.

    1. Top 1% by penalised objective
    2. Compute neighbour-average via dict lookup (O(3^6) per trial, not O(N))
    3. Select top-n diverse plateau centres
    """
    # Sort by penalised descending
    top_pct = max(int(len(results) * 0.01), 100)
    top_idx = np.argpartition(results['penalised'], -top_pct)[-top_pct:]
    top = results[top_idx]
    top = top[np.argsort(-top['penalised'])]

    print(f"\n  Top 1%: {len(top)} trials")
    print(f"  Top penalised range: {top['penalised'][0]:.3f} ~ {top['penalised'][-1]:.3f}")

    # Step sizes for normalised coordinates
    step_sizes = {'fast_low': 10, 'slow_low': 10, 'fast_high': 10, 'slow_high': 10,
                  'vol_lookback': 20, 'vol_threshold_pct': 10}

    # Build dict: grid key → list of penalised values for O(1) neighbour lookup
    print("  Building grid lookup dict...")
    grid_dict = {}
    for i in range(len(results)):
        r = results[i]
        key = (int(r['fast_low']), int(r['slow_low']),
               int(r['fast_high']), int(r['slow_high']),
               int(r['vol_lookback']), int(round(r['vol_threshold_pct'])))
        grid_dict[key] = float(r['penalised'])

    # Grid step values for generating neighbour keys
    fast_step = 10
    slow_step = 10
    vol_lb_step = 20
    vol_th_step = 10

    def get_neighbour_avg(r):
        """Average penalised of all neighbours within 1 grid step (Chebyshev)."""
        fl0, sl0 = int(r['fast_low']), int(r['slow_low'])
        fh0, sh0 = int(r['fast_high']), int(r['slow_high'])
        vl0, vt0 = int(r['vol_lookback']), int(round(r['vol_threshold_pct']))
        total = 0.0
        count = 0
        for dfl in (-fast_step, 0, fast_step):
            for dsl in (-slow_step, 0, slow_step):
                for dfh in (-fast_step, 0, fast_step):
                    for dsh in (-slow_step, 0, slow_step):
                        for dvl in (-vol_lb_step, 0, vol_lb_step):
                            for dvt in (-vol_th_step, 0, vol_th_step):
                                key = (fl0+dfl, sl0+dsl, fh0+dfh, sh0+dsh,
                                       vl0+dvl, vt0+dvt)
                                if key in grid_dict:
                                    total += grid_dict[key]
                                    count += 1
        return total / count if count > 0 else float(r['penalised'])

    # Compute neighbour averages for top trials
    print("  Computing neighbour averages...")
    neighbour_avgs = np.empty(len(top))
    for i in range(len(top)):
        neighbour_avgs[i] = get_neighbour_avg(top[i])

    # Sort by neighbour average (higher = broader plateau)
    order = np.argsort(-neighbour_avgs)
    top = top[order]
    neighbour_avgs = neighbour_avgs[order]

    def param_vec(r):
        return np.array([
            r['fast_low'] / step_sizes['fast_low'],
            r['slow_low'] / step_sizes['slow_low'],
            r['fast_high'] / step_sizes['fast_high'],
            r['slow_high'] / step_sizes['slow_high'],
            r['vol_lookback'] / step_sizes['vol_lookback'],
            r['vol_threshold_pct'] / step_sizes['vol_threshold_pct'],
        ])

    # Select diverse plateaus: greedily pick top, skip if too close to already-selected
    min_dist = 3.0  # minimum normalised L2 distance between plateau centres
    selected = []
    selected_vecs = []
    for i in range(len(top)):
        tv = param_vec(top[i])
        too_close = False
        for sv in selected_vecs:
            if np.linalg.norm(tv - sv) < min_dist:
                too_close = True
                break
        if not too_close:
            selected.append((top[i], neighbour_avgs[i]))
            selected_vecs.append(tv)
            if len(selected) >= n_plateaus:
                break

    print(f"\n  Identified {len(selected)} plateau centres:")
    for i, (r, navg) in enumerate(selected):
        print(f"    Plateau {i+1}: lo=({r['fast_low']},{r['slow_low']}), "
              f"hi=({r['fast_high']},{r['slow_high']}), "
              f"vol_lb={r['vol_lookback']}, th={r['vol_threshold_pct']:.0f}%")
        print(f"      peak penalised={r['penalised']:.3f}, "
              f"neighbour avg={navg:.3f}, sortino={r['sortino']:.3f}")

    return selected


# ══════════════════════════════════════════════════════════════
# 5. FINE GRID SEARCH
# ══════════════════════════════════════════════════════════════

def run_fine_grid(centre, mas_dict, regimes_dict, lev_ret, tbill_daily):
    """Phase 2: fine grid (step 1, ±5) around a plateau centre.
    Vol params fixed at coarse values."""
    r = centre
    fl_c, sl_c = int(r['fast_low']), int(r['slow_low'])
    fh_c, sh_c = int(r['fast_high']), int(r['slow_high'])
    vl, vt = int(r['vol_lookback']), float(r['vol_threshold_pct'])

    # MA ranges: ±5, clamped to valid domain
    fl_range = range(max(2, fl_c - 5), min(50, fl_c + 5) + 1)
    sl_range = range(max(50, sl_c - 5), min(350, sl_c + 5) + 1)
    fh_range = range(max(2, fh_c - 5), min(50, fh_c + 5) + 1)
    sh_range = range(max(50, sh_c - 5), min(350, sh_c + 5) + 1)

    high_vol = regimes_dict[(vl, vt)]

    dt = np.dtype([
        ('fast_low', np.int16), ('slow_low', np.int16),
        ('fast_high', np.int16), ('slow_high', np.int16),
        ('vol_lookback', np.int16), ('vol_threshold_pct', np.float32),
        ('sortino', np.float32), ('trades_yr', np.float32),
        ('penalised', np.float32),
    ])

    combos = []
    for fl, sl, fh, sh in product(fl_range, sl_range, fh_range, sh_range):
        if fl >= sl or fh >= sh:
            continue
        combos.append((fl, sl, fh, sh))

    results = np.empty(len(combos), dtype=dt)
    for idx, (fl, sl, fh, sh) in enumerate(combos):
        warmup = max(sl, sh, vl)
        sig = fast_regime_signal(
            mas_dict[fl], mas_dict[sl],
            mas_dict[fh], mas_dict[sh],
            high_vol, warmup
        )
        sortino, tpy, pen = fast_eval(sig, lev_ret, tbill_daily, COMMISSION, ALPHA)
        results[idx] = (fl, sl, fh, sh, vl, vt, sortino, tpy, pen)

    return results


# ══════════════════════════════════════════════════════════════
# 6. FULL BACKTEST (via leverage_rotation API)
# ══════════════════════════════════════════════════════════════

def full_backtest(price, rf_series, fl, sl, fh, sh, vl, vt):
    """Run full backtest using leverage_rotation API. Returns (cum, sig, metrics)."""
    sig_raw = signal_regime_switching_dual_ma(price, fl, sl, fh, sh, vl, vt)

    # Apply warmup
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


# ══════════════════════════════════════════════════════════════
# 7. VISUALISATION
# ══════════════════════════════════════════════════════════════

def plot_plateau_params(plateaus, fine_results_list, fname):
    """Visualise parameter distributions across plateaus."""
    n_plateaus = len(plateaus)
    param_names = ['fast_low', 'slow_low', 'fast_high', 'slow_high']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, n_plateaus))

    for ax, pname in zip(axes, param_names):
        for i, (fine_res, color) in enumerate(zip(fine_results_list, colors)):
            # Top 10% of fine grid
            top_n = max(int(len(fine_res) * 0.1), 10)
            top_idx = np.argpartition(fine_res['penalised'], -top_n)[-top_n:]
            vals = fine_res[pname][top_idx].astype(float)
            bp = ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True)
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)
        ax.set_title(pname, fontsize=12)
        ax.set_xticks(range(n_plateaus))
        ax.set_xticklabels([f"P{i+1}" for i in range(n_plateaus)])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Regime-Switching Grid: Plateau Parameter Distributions (Top 10% of Fine Grid)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


def print_final_table(label, rows):
    """Print formatted comparison table."""
    print(f"\n{'=' * 120}")
    print(f"  {label}")
    print(f"{'=' * 120}")
    header = (f"  {'Strategy':<42} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MDD':>8} {'MDD_Ent':>8} {'Recov':>7} {'Trd/Yr':>7} {'Vol':>7}")
    print(header)
    print(f"  {'─' * 113}")
    for name, m in rows.items():
        recov = m.get("Max_Recovery_Days", 0)
        recov_str = f"{recov:>6d}d" if isinstance(recov, (int, np.integer)) else f"{'—':>7}"
        tpy = m.get("Trades/Yr", 0)
        print(f"  {name:<42} {m['CAGR']:>7.1%} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} "
              f"{m['MDD']:>8.1%} {m['MDD_Entry']:>8.1%} {recov_str} {tpy:>7.1f} {m['Volatility']:>7.1%}")
    print(f"{'=' * 120}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  Multi-Resolution Grid Search: Regime-Switching 6-Param")
    print(f"  NDX 3x, ER=3.5%, lag=1, comm=0.2%, penalty α={ALPHA}")
    print(f"  Warmup={WARMUP_DAYS}d")
    print("=" * 70)

    # ── Data ──
    print("\n[1/6] Downloading data...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    rf_series = download_ken_french_rf()
    eval_start = ndx_price.index[WARMUP_DAYS]
    print(f"  NDX: {ndx_price.index[0].date()} ~ {ndx_price.index[-1].date()}")
    print(f"  Eval: {eval_start.date()} ~ {ndx_price.index[-1].date()} "
          f"({len(ndx_price) - WARMUP_DAYS}d, {(len(ndx_price) - WARMUP_DAYS)/252:.1f}yr)")

    # ── Precompute ──
    print("\n[2/6] Precomputing MAs and vol regimes...")
    prices_arr = ndx_price.values.astype(np.float64)
    n = len(prices_arr)
    daily_ret_arr = np.empty(n)
    daily_ret_arr[0] = 0.0
    daily_ret_arr[1:] = prices_arr[1:] / prices_arr[:-1] - 1.0

    t0 = time.time()
    mas_dict = precompute_mas(prices_arr, max_ma=350)
    print(f"  MAs (1~350): {time.time() - t0:.1f}s")

    vol_lookbacks = [20, 40, 60, 80, 100, 120]
    vol_thresholds = [30.0, 40.0, 50.0, 60.0, 70.0]

    t0 = time.time()
    regimes_dict = precompute_vol_regimes(daily_ret_arr, ndx_price.index,
                                           vol_lookbacks, vol_thresholds)
    print(f"  Vol regimes ({len(vol_lookbacks)}×{len(vol_thresholds)}={len(regimes_dict)}): "
          f"{time.time() - t0:.1f}s")

    # Precompute leveraged return and tbill arrays
    daily_cost = CALIBRATED_ER / 252
    lev_ret = daily_ret_arr * LEVERAGE - daily_cost

    rf_aligned = rf_series.reindex(ndx_price.index, method="ffill").fillna(0)
    tbill_daily = rf_aligned.values.astype(np.float64)

    # JIT warmup
    print("  JIT warmup...")
    _dummy_hv = np.zeros(n, dtype=np.bool_)
    _dummy_sig = fast_regime_signal(mas_dict[2], mas_dict[50], mas_dict[2], mas_dict[50], _dummy_hv, 50)
    fast_eval(_dummy_sig, lev_ret, tbill_daily, COMMISSION, ALPHA)
    print("  JIT ready")

    # ── Phase 1: Coarse Grid ──
    print("\n[3/6] Phase 1: Coarse grid search (step=10)...")
    coarse_results = run_coarse_grid(
        prices_arr, daily_ret_arr, ndx_price.index,
        mas_dict, regimes_dict,
        lev_ret, tbill_daily
    )

    # Save coarse top 1%
    top_pct = max(int(len(coarse_results) * 0.01), 100)
    top_idx = np.argpartition(coarse_results['penalised'], -top_pct)[-top_pct:]
    top_coarse = coarse_results[top_idx]
    top_coarse = top_coarse[np.argsort(-top_coarse['penalised'])]

    df_top = pd.DataFrame({
        'fast_low': top_coarse['fast_low'],
        'slow_low': top_coarse['slow_low'],
        'fast_high': top_coarse['fast_high'],
        'slow_high': top_coarse['slow_high'],
        'vol_lookback': top_coarse['vol_lookback'],
        'vol_threshold_pct': top_coarse['vol_threshold_pct'],
        'sortino': top_coarse['sortino'],
        'trades_yr': top_coarse['trades_yr'],
        'penalised': top_coarse['penalised'],
    })
    df_top.to_csv(OUT_DIR / "regime_grid_coarse_top.csv", index=False)
    print(f"  -> regime_grid_coarse_top.csv ({len(df_top)} rows)")

    # Best coarse
    best_c = coarse_results[np.argmax(coarse_results['penalised'])]
    print(f"\n  Coarse best: lo=({best_c['fast_low']},{best_c['slow_low']}), "
          f"hi=({best_c['fast_high']},{best_c['slow_high']}), "
          f"vol_lb={best_c['vol_lookback']}, th={best_c['vol_threshold_pct']:.0f}%")
    print(f"    Sortino={best_c['sortino']:.3f}, Trades/Yr={best_c['trades_yr']:.1f}, "
          f"Penalised={best_c['penalised']:.3f}")

    # Check P5 Optuna neighbourhood
    p5_fl, p5_sl, p5_fh, p5_sh = 48, 323, 15, 229
    p5_near = coarse_results[
        (np.abs(coarse_results['fast_low'].astype(int) - p5_fl) <= 10) &
        (np.abs(coarse_results['slow_low'].astype(int) - p5_sl) <= 10) &
        (np.abs(coarse_results['fast_high'].astype(int) - p5_fh) <= 10) &
        (np.abs(coarse_results['slow_high'].astype(int) - p5_sh) <= 10)
    ]
    if len(p5_near) > 0:
        best_p5n = p5_near[np.argmax(p5_near['penalised'])]
        rank = (coarse_results['penalised'] > best_p5n['penalised']).sum() + 1
        print(f"\n  P5 Optuna neighbourhood ({len(p5_near)} points):")
        print(f"    Best near P5: lo=({best_p5n['fast_low']},{best_p5n['slow_low']}), "
              f"hi=({best_p5n['fast_high']},{best_p5n['slow_high']})")
        print(f"    Penalised={best_p5n['penalised']:.3f}, "
              f"Rank={rank}/{len(coarse_results)}")
    else:
        print("\n  P5 Optuna params not found in coarse grid neighbourhood")

    # ── Phase 1.5: Plateau Identification ──
    print("\n[4/6] Identifying plateaus...")
    plateaus = identify_plateaus(coarse_results, n_plateaus=5)

    # ── Phase 2: Fine Grid ──
    print("\n[5/6] Phase 2: Fine grid search (step=1, ±5) per plateau...")
    fine_results_list = []
    plateau_metrics = []

    for i, (centre, navg) in enumerate(plateaus):
        t0 = time.time()
        fine_res = run_fine_grid(centre, mas_dict, regimes_dict, lev_ret, tbill_daily)
        elapsed = time.time() - t0

        best_f = fine_res[np.argmax(fine_res['penalised'])]
        avg_pen = fine_res['penalised'].mean()
        peak_pen = best_f['penalised']

        # 95% breadth: how many trials have sortino ≥ peak × 95%
        peak_sort = best_f['sortino']
        breadth_95 = (fine_res['sortino'] >= peak_sort * 0.95).sum()

        # Robust: best neighbour-average within fine grid
        # (simple: average of top quartile)
        top_q = fine_res[fine_res['penalised'] >= np.percentile(fine_res['penalised'], 75)]
        robust_idx = np.argmin(np.abs(fine_res['penalised'] - np.median(top_q['penalised'])))
        robust = fine_res[robust_idx]

        fine_results_list.append(fine_res)
        plateau_metrics.append({
            'peak': best_f,
            'robust': robust,
            'avg_penalised': avg_pen,
            'peak_penalised': peak_pen,
            'breadth_95': breadth_95,
            'total_fine': len(fine_res),
            'navg_coarse': navg,
        })

        print(f"\n  Plateau {i+1}: {len(fine_res)} fine combos ({elapsed:.1f}s)")
        print(f"    Peak:   lo=({best_f['fast_low']},{best_f['slow_low']}), "
              f"hi=({best_f['fast_high']},{best_f['slow_high']}) "
              f"| Sortino={best_f['sortino']:.3f}, Pen={peak_pen:.3f}")
        print(f"    Robust: lo=({robust['fast_low']},{robust['slow_low']}), "
              f"hi=({robust['fast_high']},{robust['slow_high']}) "
              f"| Sortino={robust['sortino']:.3f}, Pen={robust['penalised']:.3f}")
        print(f"    Avg penalised={avg_pen:.3f}, 95% breadth={breadth_95}/{len(fine_res)}")

    # ── Phase 3: Full Backtest Comparison ──
    print("\n[6/6] Phase 3: Full backtest comparison...")
    all_results = {}
    all_curves = {}

    # P5 Optuna baseline
    print("  Running P5 Optuna (48,323,15,229,49,57.3)...")
    cum, sig, m = full_backtest(ndx_price, rf_series, 48, 323, 15, 229, 49, 57.3)
    all_results["P5 Optuna (48,323,15,229)"] = m
    all_curves["P5 Optuna (48,323,15,229)"] = cum

    # Each plateau's peak and robust
    for i, pm in enumerate(plateau_metrics):
        pk = pm['peak']
        rb = pm['robust']

        fl, sl = int(pk['fast_low']), int(pk['slow_low'])
        fh, sh = int(pk['fast_high']), int(pk['slow_high'])
        vl, vt = int(pk['vol_lookback']), float(pk['vol_threshold_pct'])

        label_pk = f"P{i+1} peak ({fl},{sl},{fh},{sh})"
        print(f"  Running {label_pk}...")
        cum, sig, m = full_backtest(ndx_price, rf_series, fl, sl, fh, sh, vl, vt)
        all_results[label_pk] = m
        all_curves[label_pk] = cum

        fl_r, sl_r = int(rb['fast_low']), int(rb['slow_low'])
        fh_r, sh_r = int(rb['fast_high']), int(rb['slow_high'])
        vl_r, vt_r = int(rb['vol_lookback']), float(rb['vol_threshold_pct'])

        if (fl_r, sl_r, fh_r, sh_r) != (fl, sl, fh, sh):
            label_rb = f"P{i+1} robust ({fl_r},{sl_r},{fh_r},{sh_r})"
            print(f"  Running {label_rb}...")
            cum, sig, m = full_backtest(ndx_price, rf_series, fl_r, sl_r, fh_r, sh_r, vl_r, vt_r)
            all_results[label_rb] = m
            all_curves[label_rb] = cum

    # Sym (3,161) baseline
    print("  Running Sym (3,161) baseline...")
    sig_raw = signal_dual_ma(ndx_price, slow=161, fast=3)
    sig_sym = sig_raw.copy()
    wd = ndx_price.index[WARMUP_DAYS]
    sig_sym.loc[:wd] = 0
    if sig_raw.loc[wd] == 1:
        post = sig_raw.loc[wd:]
        fz = post[post == 0].index
        if len(fz) > 0:
            sig_sym.loc[wd:fz[0]] = 0
    cum = run_lrs(ndx_price, sig_sym, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
    cum = cum.loc[wd:]; cum = cum / cum.iloc[0]
    sig_sym = sig_sym.loc[wd:]
    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    m["MDD_Entry"] = _max_entry_drawdown(cum, sig_sym, signal_lag=SIGNAL_LAG)
    m["Max_Recovery_Days"] = _max_recovery_days(cum)
    m["Trades/Yr"] = signal_trades_per_year(sig_sym)
    all_results["Sym (3,161) baseline"] = m
    all_curves["Sym (3,161) baseline"] = cum

    # B&H 3x
    bh3x = run_buy_and_hold(ndx_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x = bh3x.loc[wd:]; bh3x = bh3x / bh3x.iloc[0]
    m_bh = calc_metrics(bh3x, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh["MDD_Entry"] = m_bh["MDD"]
    m_bh["Max_Recovery_Days"] = _max_recovery_days(bh3x)
    m_bh["Trades/Yr"] = 0.0
    all_results["B&H 3x"] = m_bh
    all_curves["B&H 3x"] = bh3x

    # ── Final table ──
    print_final_table(
        f"REGIME GRID SEARCH — Full Backtest ({eval_start.date()} ~ {ndx_price.index[-1].date()})",
        all_results
    )

    # ── Plateau summary table ──
    print(f"\n{'─' * 80}")
    print(f"  Plateau Quality Summary")
    print(f"{'─' * 80}")
    print(f"  {'Plateau':<10} {'Peak Pen':>9} {'Avg Pen':>9} {'95% Breadth':>12} "
          f"{'Coarse NAvg':>12} {'Fine Trials':>12}")
    print(f"  {'─' * 75}")
    for i, pm in enumerate(plateau_metrics):
        print(f"  {'P' + str(i+1):<10} {pm['peak_penalised']:>9.3f} {pm['avg_penalised']:>9.3f} "
              f"{pm['breadth_95']:>6d}/{pm['total_fine']:<5d} "
              f"{pm['navg_coarse']:>12.3f} {pm['total_fine']:>12d}")

    # ── Save final CSV ──
    records = []
    for name, m in all_results.items():
        records.append({
            "strategy": name,
            "CAGR": m["CAGR"], "Sharpe": m["Sharpe"], "Sortino": m["Sortino"],
            "Volatility": m["Volatility"], "MDD": m["MDD"],
            "MDD_Entry": m["MDD_Entry"],
            "Max_Recovery_Days": m.get("Max_Recovery_Days", ""),
            "Trades_Yr": m.get("Trades/Yr", ""),
        })
    pd.DataFrame(records).to_csv(OUT_DIR / "regime_grid_final.csv", index=False)
    print(f"\n  -> regime_grid_final.csv")

    # ── Charts ──
    print("\n  Generating charts...")

    # Plateau parameter distributions
    plot_plateau_params(plateaus, fine_results_list, "regime_grid_plateaus.png")

    # Cumulative return comparison
    fig, ax = plt.subplots(figsize=(16, 9))
    cmap = plt.cm.tab10
    n_curves = len(all_curves)
    for i, (label, s) in enumerate(all_curves.items()):
        is_baseline = "Sym" in label or "B&H" in label or "Optuna" in label
        ls = "--" if is_baseline else "-"
        lw = 1.5 if is_baseline else 2.0
        ax.plot(s.index, s.values, label=label, linestyle=ls, linewidth=lw,
                color=cmap(i / max(n_curves - 1, 1)))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(f"Regime Grid Search: Full Backtest Comparison "
                 f"({eval_start.year}-{ndx_price.index[-1].year})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regime_grid_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> regime_grid_comparison.png")

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Done! Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
