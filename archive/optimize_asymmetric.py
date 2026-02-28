"""
Phase 1: Asymmetric Buy/Sell Signal Optimization

Uses Optuna (TPE sampler) to find optimal asymmetric dual MA parameters
for TQQQ-calibrated NDX backtest.

Walk-forward: train 1985-2014, test 2015-2025.

Usage:
    pip install optuna
    python optimize_asymmetric.py
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
    signal_asymmetric_dual_ma, signal_dual_ma,
    run_lrs, run_buy_and_hold, calc_metrics, signal_trades_per_year,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Configuration (Part 12 baseline) ──
CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002

TRAIN_END = "2014-12-31"
TEST_START = "2015-01-01"

N_TRIALS = 2000
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


def make_objective(train_price, rf_series, warmup_days):
    """Create Optuna objective function (closure over data)."""

    def objective(trial):
        fast_buy = trial.suggest_int("fast_buy", 2, 50)
        slow_buy = trial.suggest_int("slow_buy", 50, 350)
        fast_sell = trial.suggest_int("fast_sell", 2, 50)
        slow_sell = trial.suggest_int("slow_sell", 50, 350)

        sig_raw = signal_asymmetric_dual_ma(train_price, fast_buy, slow_buy,
                                            fast_sell, slow_sell)
        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)

        # Store extra metrics for analysis
        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("MDD", m["MDD"])

        return m["Sortino"]

    return objective


def get_symmetric_baseline(price, rf_series, warmup_days):
    """Run best symmetric combos from Part 12 grid search as baselines."""
    baselines = {}
    # eulb recommended: (3, 161)
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
    """Print a formatted comparison table."""
    print(f"\n{'─' * 80}")
    print(f"  {label}")
    print(f"{'─' * 80}")
    header = f"  {'Strategy':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} {'MDD':>9}"
    print(header)
    print(f"  {'─' * 75}")
    for name, m in metrics_dict.items():
        print(f"  {name:<35} {m['CAGR']:>8.2%} {m['Sharpe']:>8.3f} "
              f"{m['Sortino']:>9.3f} {m['MDD']:>9.2%}")


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


def plot_param_stability(study, top_n=20):
    """Box plot of top-N trials' parameter distributions."""
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1e9,
                    reverse=True)[:top_n]

    params = ["fast_buy", "slow_buy", "fast_sell", "slow_sell"]
    data = {p: [t.params[p] for t in trials] for p in params}

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    ranges = {"fast_buy": (2, 50), "slow_buy": (50, 350),
              "fast_sell": (2, 50), "slow_sell": (50, 350)}

    for ax, param in zip(axes, params):
        vals = data[param]
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][0].set_alpha(0.6)
        ax.set_title(param, fontsize=12)
        ax.set_ylim(ranges[param])
        ax.set_xticklabels([])
        ax.grid(axis="y", alpha=0.3)

        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        full_range = ranges[param][1] - ranges[param][0]
        pct = iqr / full_range * 100
        ax.text(0.5, 0.02, f"IQR: {iqr:.0f} ({pct:.0f}%)",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="red" if pct > 50 else "green")

    fig.suptitle(f"Parameter Stability (Top {top_n} Trials)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "asymmetric_param_stability.png", dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / 'asymmetric_param_stability.png'}")


def main():
    print("=" * 70)
    print("  Asymmetric Buy/Sell Signal Optimization (Phase 1)")
    print(f"  NDX 1985-2025, 3x leverage, TQQQ-calibrated costs")
    print(f"  Train: 1985-{TRAIN_END[:4]} | Test: {TEST_START[:4]}-2025")
    print(f"  Optuna TPE, {N_TRIALS} trials, maximize Sortino")
    print("=" * 70)

    # ── 1. Download data ──
    print("\n[1/8] Downloading data...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    print(f"  NDX: {len(ndx_price)} days ({ndx_price.index[0].date()} -> {ndx_price.index[-1].date()})")

    rf_series = download_ken_french_rf()

    # ── 2. Train/Test split ──
    print("\n[2/8] Splitting train/test...")
    train_price = ndx_price.loc[:TRAIN_END]
    test_price = ndx_price.loc[TEST_START:]
    print(f"  Train: {len(train_price)} days ({train_price.index[0].date()} -> {train_price.index[-1].date()})")
    print(f"  Test:  {len(test_price)} days ({test_price.index[0].date()} -> {test_price.index[-1].date()})")

    warmup_days = 350  # max possible slow MA

    # ── 3. Run Optuna ──
    print(f"\n[3/8] Running Optuna optimization ({N_TRIALS} trials)...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="asymmetric_dual_ma")

    objective = make_objective(train_price, rf_series, warmup_days)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ── 4. Best parameters ──
    best = study.best_trial
    print(f"\n[4/8] Best parameters (train):")
    print(f"  fast_buy={best.params['fast_buy']}, slow_buy={best.params['slow_buy']}")
    print(f"  fast_sell={best.params['fast_sell']}, slow_sell={best.params['slow_sell']}")
    print(f"  Sortino={best.value:.3f}, CAGR={best.user_attrs['CAGR']:.2%}, "
          f"Sharpe={best.user_attrs['Sharpe']:.3f}, MDD={best.user_attrs['MDD']:.2%}")

    # ── 5. Out-of-sample validation ──
    print("\n[5/8] Out-of-sample validation...")
    bp = best.params

    # Asymmetric best — on test set
    test_warmup = min(warmup_days, len(test_price) - 1)
    sig_test = signal_asymmetric_dual_ma(test_price,
                                          bp["fast_buy"], bp["slow_buy"],
                                          bp["fast_sell"], bp["slow_sell"])
    cum_test_asym, m_test_asym = run_backtest(test_price, rf_series, sig_test, test_warmup)

    # Asymmetric best — on train set (for comparison)
    sig_train = signal_asymmetric_dual_ma(train_price,
                                           bp["fast_buy"], bp["slow_buy"],
                                           bp["fast_sell"], bp["slow_sell"])
    cum_train_asym, m_train_asym = run_backtest(train_price, rf_series, sig_train, warmup_days)

    # ── 6. Symmetric baselines ──
    print("\n[6/8] Computing symmetric baselines...")
    train_baselines = get_symmetric_baseline(train_price, rf_series, warmup_days)
    test_baselines = get_symmetric_baseline(test_price, rf_series, test_warmup)

    # B&H 3x
    bh3x_train = run_buy_and_hold(train_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_train = trim_warmup(bh3x_train, warmup_days, train_price.index)
    rf_scalar = rf_series.mean() * 252
    m_bh3x_train = calc_metrics(bh3x_train, tbill_rate=rf_scalar, rf_series=rf_series)

    bh3x_test = run_buy_and_hold(test_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_test = trim_warmup(bh3x_test, test_warmup, test_price.index)
    m_bh3x_test = calc_metrics(bh3x_test, tbill_rate=rf_scalar, rf_series=rf_series)

    # Full period — asymmetric best
    sig_full = signal_asymmetric_dual_ma(ndx_price,
                                          bp["fast_buy"], bp["slow_buy"],
                                          bp["fast_sell"], bp["slow_sell"])
    cum_full_asym, m_full_asym = run_backtest(ndx_price, rf_series, sig_full, warmup_days)

    # Full period — symmetric baselines
    full_baselines = get_symmetric_baseline(ndx_price, rf_series, warmup_days)
    bh3x_full = run_buy_and_hold(ndx_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_full = trim_warmup(bh3x_full, warmup_days, ndx_price.index)
    m_bh3x_full = calc_metrics(bh3x_full, tbill_rate=rf_scalar, rf_series=rf_series)

    # ── Print comparison tables ──
    print_comparison_table("TRAIN Period (1985-2014)", {
        f"Asymmetric ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})": m_train_asym,
        **{f"Symmetric {k}": v["metrics"] for k, v in train_baselines.items()},
        "B&H 3x": m_bh3x_train,
    })

    print_comparison_table("TEST Period (2015-2025) — Out-of-Sample", {
        f"Asymmetric ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})": m_test_asym,
        **{f"Symmetric {k}": v["metrics"] for k, v in test_baselines.items()},
        "B&H 3x": m_bh3x_test,
    })

    print_comparison_table("FULL Period (1985-2025)", {
        f"Asymmetric ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})": m_full_asym,
        **{f"Symmetric {k}": v["metrics"] for k, v in full_baselines.items()},
        "B&H 3x": m_bh3x_full,
    })

    # Sortino degradation analysis
    sortino_train = m_train_asym["Sortino"]
    sortino_test = m_test_asym["Sortino"]
    degradation = (sortino_train - sortino_test) / sortino_train * 100 if sortino_train > 0 else float("nan")
    print(f"\n  Sortino degradation (train→test): {degradation:.1f}%")
    if degradation > 50:
        print("  ⚠ WARNING: >50% degradation suggests overfitting")
    elif degradation > 30:
        print("  ⚠ CAUTION: 30-50% degradation — moderate overfitting risk")
    else:
        print("  ✓ <30% degradation — reasonable generalization")

    # ── 7. Save results ──
    print("\n[7/8] Saving results...")

    # Full trial results CSV
    records = []
    for t in study.trials:
        if t.value is not None:
            records.append({
                "trial": t.number,
                "fast_buy": t.params["fast_buy"],
                "slow_buy": t.params["slow_buy"],
                "fast_sell": t.params["fast_sell"],
                "slow_sell": t.params["slow_sell"],
                "Sortino": t.value,
                "CAGR": t.user_attrs.get("CAGR"),
                "Sharpe": t.user_attrs.get("Sharpe"),
                "MDD": t.user_attrs.get("MDD"),
            })
    df_results = pd.DataFrame(records)
    df_results.to_csv(OUT_DIR / "asymmetric_optimization_results.csv", index=False)
    print(f"  → saved {OUT_DIR / 'asymmetric_optimization_results.csv'} ({len(df_results)} trials)")

    # ── 8. Charts ──
    print("\n[8/8] Generating charts...")

    # Cumulative comparison — test period
    test_curves = {
        f"Asymmetric ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})": cum_test_asym,
        **{f"Symmetric {k}": v["cum"] for k, v in test_baselines.items()},
        "B&H 3x": bh3x_test,
    }
    plot_cumulative_comparison(
        test_curves,
        "Asymmetric vs Symmetric — Out-of-Sample (2015-2025)",
        "asymmetric_best_vs_baseline_test.png",
    )

    # Cumulative comparison — full period
    full_curves = {
        f"Asymmetric ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})": cum_full_asym,
        **{f"Symmetric {k}": v["cum"] for k, v in full_baselines.items()},
        "B&H 3x": bh3x_full,
    }
    plot_cumulative_comparison(
        full_curves,
        "Asymmetric vs Symmetric — Full Period (1985-2025)",
        "asymmetric_best_vs_baseline.png",
    )

    # Parameter stability
    plot_param_stability(study, top_n=20)

    print("\n" + "=" * 70)
    print("  Done! Results saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
