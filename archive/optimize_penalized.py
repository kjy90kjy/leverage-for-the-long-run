"""
Phase 2 — Plan 1: Whipsaw Penalty Optimization (Approach C)

Three objective variants:
  C1: Sortino - ALPHA * trades_per_year  (ALPHA=0.02)
  C2: Multi-objective Pareto (Sortino ↑, trades ↓)
  C3: Hard cap (trades_per_year > 15 → prune)

NDX 1985-2025, 3x leverage, TQQQ-calibrated costs.
Train: 1985-2014 | Test: 2015-2025.

Usage:
    python optimize_penalized.py
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from leverage_rotation import signal_asymmetric_dual_ma, signal_trades_per_year
from optimize_common import (
    OUT_DIR, CALIBRATED_ER, LEVERAGE, SIGNAL_LAG, COMMISSION, SEED,
    apply_warmup, trim_warmup, run_backtest,
    get_symmetric_baselines, print_comparison_table, plot_cumulative_comparison,
    plot_param_stability, download_ndx_and_rf,
)
from leverage_rotation import run_buy_and_hold, calc_metrics

TRAIN_END = "2014-12-31"
TEST_START = "2015-01-01"
N_TRIALS = 2000
ALPHA = 0.02  # penalty coefficient for C1


def make_objective_c1(train_price, rf_series, warmup_days):
    """C1: Sortino - ALPHA * trades_per_year."""
    def objective(trial):
        fast_buy = trial.suggest_int("fast_buy", 2, 50)
        slow_buy = trial.suggest_int("slow_buy", 50, 350)
        fast_sell = trial.suggest_int("fast_sell", 2, 50)
        slow_sell = trial.suggest_int("slow_sell", 50, 350)

        sig_raw = signal_asymmetric_dual_ma(train_price, fast_buy, slow_buy,
                                            fast_sell, slow_sell)
        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)
        tpy = signal_trades_per_year(sig_raw)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("Sortino", m["Sortino"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"] - ALPHA * tpy
    return objective


def make_objective_c2(train_price, rf_series, warmup_days):
    """C2: Multi-objective Pareto (maximize Sortino, minimize trades)."""
    def objective(trial):
        fast_buy = trial.suggest_int("fast_buy", 2, 50)
        slow_buy = trial.suggest_int("slow_buy", 50, 350)
        fast_sell = trial.suggest_int("fast_sell", 2, 50)
        slow_sell = trial.suggest_int("slow_sell", 50, 350)

        sig_raw = signal_asymmetric_dual_ma(train_price, fast_buy, slow_buy,
                                            fast_sell, slow_sell)
        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)
        tpy = signal_trades_per_year(sig_raw)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("Sortino", m["Sortino"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"], tpy  # maximize Sortino, minimize trades
    return objective


def make_objective_c3(train_price, rf_series, warmup_days):
    """C3: Hard cap — prune if trades_per_year > 15."""
    def objective(trial):
        fast_buy = trial.suggest_int("fast_buy", 2, 50)
        slow_buy = trial.suggest_int("slow_buy", 50, 350)
        fast_sell = trial.suggest_int("fast_sell", 2, 50)
        slow_sell = trial.suggest_int("slow_sell", 50, 350)

        sig_raw = signal_asymmetric_dual_ma(train_price, fast_buy, slow_buy,
                                            fast_sell, slow_sell)
        tpy = signal_trades_per_year(sig_raw)
        if tpy > 15:
            raise optuna.TrialPruned()

        _, m = run_backtest(train_price, rf_series, sig_raw, warmup_days)

        trial.set_user_attr("CAGR", m["CAGR"])
        trial.set_user_attr("Sharpe", m["Sharpe"])
        trial.set_user_attr("Sortino", m["Sortino"])
        trial.set_user_attr("MDD", m["MDD"])
        trial.set_user_attr("Trades_Yr", tpy)

        return m["Sortino"]
    return objective


def run_variant(name, train_price, test_price, rf_series, warmup_days, test_warmup):
    """Run a single optimization variant and return results."""
    print(f"\n  --- Variant {name} ---")

    if name == "C1":
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler,
                                    study_name=f"penalized_{name}")
        obj = make_objective_c1(train_price, rf_series, warmup_days)
        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)
        best = study.best_trial
        print(f"  Best: fb={best.params['fast_buy']}, sb={best.params['slow_buy']}, "
              f"fs={best.params['fast_sell']}, ss={best.params['slow_sell']}")
        print(f"  Sortino={best.user_attrs['Sortino']:.3f}, "
              f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}, "
              f"Penalized={best.value:.3f}")

    elif name == "C2":
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(
            directions=["maximize", "minimize"], sampler=sampler,
            study_name=f"penalized_{name}")
        obj = make_objective_c2(train_price, rf_series, warmup_days)
        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)
        # Select Pareto-best by highest Sortino among low-trade solutions
        pareto = study.best_trials
        # Filter to trades < 15, then pick max Sortino
        candidates = [t for t in pareto if t.values[1] <= 15]
        if not candidates:
            candidates = sorted(pareto, key=lambda t: t.values[1])[:5]
        best = max(candidates, key=lambda t: t.values[0])
        print(f"  Pareto front: {len(pareto)} trials")
        print(f"  Selected: fb={best.params['fast_buy']}, sb={best.params['slow_buy']}, "
              f"fs={best.params['fast_sell']}, ss={best.params['slow_sell']}")
        print(f"  Sortino={best.values[0]:.3f}, Trades/Yr={best.values[1]:.1f}")

    elif name == "C3":
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler,
                                    study_name=f"penalized_{name}")
        obj = make_objective_c3(train_price, rf_series, warmup_days)
        study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)
        best = study.best_trial
        print(f"  Best: fb={best.params['fast_buy']}, sb={best.params['slow_buy']}, "
              f"fs={best.params['fast_sell']}, ss={best.params['slow_sell']}")
        print(f"  Sortino={best.user_attrs['Sortino']:.3f}, "
              f"Trades/Yr={best.user_attrs['Trades_Yr']:.1f}")

    # OOS evaluation
    bp = best.params
    sig_test = signal_asymmetric_dual_ma(test_price,
                                          bp["fast_buy"], bp["slow_buy"],
                                          bp["fast_sell"], bp["slow_sell"])
    cum_test, m_test = run_backtest(test_price, rf_series, sig_test, test_warmup)
    tpy_test = signal_trades_per_year(sig_test)
    m_test["Trades/Yr"] = tpy_test

    sig_train = signal_asymmetric_dual_ma(train_price,
                                           bp["fast_buy"], bp["slow_buy"],
                                           bp["fast_sell"], bp["slow_sell"])
    cum_train, m_train = run_backtest(train_price, rf_series, sig_train, warmup_days)
    tpy_train = signal_trades_per_year(sig_train)
    m_train["Trades/Yr"] = tpy_train

    # Save trial CSV
    records = []
    for t in study.trials:
        if name == "C2":
            if t.values is not None and len(t.values) == 2:
                records.append({
                    "trial": t.number,
                    **t.params,
                    "Sortino": t.values[0],
                    "Trades_Yr": t.values[1],
                    "CAGR": t.user_attrs.get("CAGR"),
                    "Sharpe": t.user_attrs.get("Sharpe"),
                    "MDD": t.user_attrs.get("MDD"),
                })
        else:
            if t.value is not None:
                records.append({
                    "trial": t.number,
                    **t.params,
                    "Sortino": t.user_attrs.get("Sortino"),
                    "Trades_Yr": t.user_attrs.get("Trades_Yr"),
                    "CAGR": t.user_attrs.get("CAGR"),
                    "Sharpe": t.user_attrs.get("Sharpe"),
                    "MDD": t.user_attrs.get("MDD"),
                    "Penalized_Score": t.value,
                })
    df = pd.DataFrame(records)
    csv_name = f"penalized_{name}_results.csv"
    df.to_csv(OUT_DIR / csv_name, index=False)
    print(f"  → saved {csv_name} ({len(df)} trials)")

    return {
        "study": study,
        "best": best,
        "params": bp,
        "cum_test": cum_test,
        "m_test": m_test,
        "cum_train": cum_train,
        "m_train": m_train,
    }


def plot_pareto_scatter(study_c2):
    """Sortino vs Trades/Yr scatter with Pareto front highlighted."""
    fig, ax = plt.subplots(figsize=(10, 7))

    sortinos, trades = [], []
    for t in study_c2.trials:
        if t.values is not None and len(t.values) == 2:
            sortinos.append(t.values[0])
            trades.append(t.values[1])

    ax.scatter(trades, sortinos, alpha=0.2, s=10, c="#95a5a6", label="All trials")

    # Pareto front
    pareto = study_c2.best_trials
    p_sortinos = [t.values[0] for t in pareto]
    p_trades = [t.values[1] for t in pareto]
    # Sort by trades for line
    order = np.argsort(p_trades)
    p_trades_sorted = [p_trades[i] for i in order]
    p_sortinos_sorted = [p_sortinos[i] for i in order]

    ax.plot(p_trades_sorted, p_sortinos_sorted, "r-o", markersize=4,
            linewidth=1.5, label=f"Pareto front ({len(pareto)} trials)")
    ax.scatter(p_trades, p_sortinos, c="red", s=30, zorder=5)

    ax.set_xlabel("Trades / Year", fontsize=12)
    ax.set_ylabel("Sortino Ratio (Train)", fontsize=12)
    ax.set_title("C2 Multi-Objective: Sortino vs Trading Frequency", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "penalized_pareto_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  → saved penalized_pareto_scatter.png")


def main():
    print("=" * 70)
    print("  Phase 2 — Plan 1: Whipsaw Penalty Optimization (Approach C)")
    print(f"  NDX 1985-2025, 3x leverage, TQQQ-calibrated costs")
    print(f"  Variants: C1 (penalty), C2 (Pareto), C3 (hard cap)")
    print(f"  {N_TRIALS} trials each, ALPHA={ALPHA}")
    print("=" * 70)

    # ── Data ──
    print("\n[1/5] Downloading data...")
    ndx_price, rf_series = download_ndx_and_rf()

    train_price = ndx_price.loc[:TRAIN_END]
    test_price = ndx_price.loc[TEST_START:]
    warmup_days = 350
    test_warmup = min(warmup_days, len(test_price) - 1)

    print(f"  Train: {len(train_price)} days | Test: {len(test_price)} days")

    # ── Run all variants ──
    print("\n[2/5] Running optimization variants...")
    results = {}
    for variant in ["C1", "C2", "C3"]:
        results[variant] = run_variant(variant, train_price, test_price,
                                       rf_series, warmup_days, test_warmup)

    # ── Baselines ──
    print("\n[3/5] Computing baselines...")
    test_baselines = get_symmetric_baselines(test_price, rf_series, test_warmup)

    rf_scalar = rf_series.mean() * 252
    bh3x_test = run_buy_and_hold(test_price, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER)
    bh3x_test = trim_warmup(bh3x_test, test_warmup, test_price.index)
    m_bh3x_test = calc_metrics(bh3x_test, tbill_rate=rf_scalar, rf_series=rf_series)
    m_bh3x_test["Trades/Yr"] = 0

    # Add trades/yr to baselines
    for k, v in test_baselines.items():
        v["metrics"]["Trades/Yr"] = v["trades_yr"]

    # ── Comparison table ──
    print("\n[4/5] Results comparison...")
    table = {}
    for variant in ["C1", "C2", "C3"]:
        r = results[variant]
        bp = r["params"]
        label = f"{variant} ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"
        table[label] = r["m_test"]
    for k, v in test_baselines.items():
        table[f"Symmetric {k}"] = v["metrics"]
    table["B&H 3x"] = m_bh3x_test

    print_comparison_table("TEST Period (2015-2025) — Out-of-Sample", table)

    # ── Charts ──
    print("\n[5/5] Generating charts...")

    # OOS equity comparison
    curves = {}
    for variant in ["C1", "C2", "C3"]:
        r = results[variant]
        bp = r["params"]
        curves[f"{variant} ({bp['fast_buy']},{bp['slow_buy']},{bp['fast_sell']},{bp['slow_sell']})"] = r["cum_test"]
    curves["Symmetric (3,161)"] = test_baselines["(3,161) eulb"]["cum"]
    curves["B&H 3x"] = bh3x_test

    plot_cumulative_comparison(
        curves,
        "Penalized Optimization — OOS Comparison (2015-2025)",
        "penalized_test_comparison.png",
    )

    # C2 Pareto scatter
    plot_pareto_scatter(results["C2"]["study"])

    # C1 stability plot
    study_c1 = results["C1"]["study"]
    trials_sorted = sorted(
        [t for t in study_c1.trials if t.value is not None],
        key=lambda t: t.value, reverse=True)[:20]
    param_data = {p: [t.params[p] for t in trials_sorted]
                  for p in ["fast_buy", "slow_buy", "fast_sell", "slow_sell"]}
    param_ranges = {"fast_buy": (2, 50), "slow_buy": (50, 350),
                    "fast_sell": (2, 50), "slow_sell": (50, 350)}
    plot_param_stability(param_data, param_ranges,
                         "C1 Parameter Stability (Top 20 Trials)",
                         "penalized_stability.png")

    print("\n" + "=" * 70)
    print("  Done! Results saved in:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
