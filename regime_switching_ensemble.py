"""
Regime-Switching Ensemble: 과최적화 위험 감소 전략

단일 최적 파라미터 대신 여러 좋은 파라미터 조합의 가중 평균을 사용.
→ 분산 감소, 일반화 성능 개선

실행:
    python regime_switching_ensemble.py
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
    run_lrs, calc_metrics, _max_entry_drawdown, _max_recovery_days,
    signal_trades_per_year,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

CALIBRATED_ER = 0.035
LEVERAGE = 3.0
SIGNAL_LAG = 1
COMMISSION = 0.002
WARMUP_DAYS = 500

# ══════════════════════════════════════════════════════════════
# Candidate Parameters: Top performers (Phase 3v2 결과)
# ══════════════════════════════════════════════════════════════

CANDIDATES = [
    {
        'name': 'P1 Optimal (Phase 3v2)',
        'params': {'fast_low': 12, 'slow_low': 237, 'fast_high': 6, 'slow_high': 229,
                   'vol_lookback': 49, 'vol_threshold_pct': 57.3},
        'weight': 0.40,
        'note': '최고 Sortino 1.088, MDD_Entry -39.2%'
    },
    {
        'name': 'P2 Conservative',
        'params': {'fast_low': 11, 'slow_low': 240, 'fast_high': 6, 'slow_high': 235,
                   'vol_lookback': 50, 'vol_threshold_pct': 60},
        'weight': 0.30,
        'note': '더 느린 신호, MDD_Entry 우수'
    },
    {
        'name': 'P3 Balanced',
        'params': {'fast_low': 13, 'slow_low': 230, 'fast_high': 7, 'slow_high': 225,
                   'vol_lookback': 48, 'vol_threshold_pct': 55},
        'weight': 0.20,
        'note': '중간값 조합'
    },
    {
        'name': 'P4 Adaptive',
        'params': {'fast_low': 12, 'slow_low': 237, 'fast_high': 6, 'slow_high': 220,
                   'vol_lookback': 50, 'vol_threshold_pct': 60},
        'weight': 0.10,
        'note': '낮은 변동성 우시 더 빠른 신호'
    },
]


def create_ensemble_signal(price, candidates):
    """
    각 파라미터로 신호를 생성 후 가중 평균.

    Returns: pd.Series (0~1 범위)
    """
    n = len(price)
    ensemble = np.zeros(n)

    for cand in candidates:
        p = cand['params']
        sig = signal_regime_switching_dual_ma(
            price,
            fast_low=p['fast_low'], slow_low=p['slow_low'],
            fast_high=p['fast_high'], slow_high=p['slow_high'],
            vol_lookback=p['vol_lookback'], vol_threshold_pct=p['vol_threshold_pct']
        )
        ensemble += sig.values * cand['weight']

    return pd.Series(ensemble, index=price.index)


def ensemble_to_binary_signal(ensemble_signal, threshold=0.5):
    """
    연속 앙상블 신호 (0~1)를 이진 신호 (0/1)로 변환.
    threshold=0.5: 50% 이상이면 매수
    """
    return (ensemble_signal >= threshold).astype(int)


def backtest_ensemble(price, rf_series, ensemble_signal_continuous):
    """
    앙상블 신호로 백테스트.
    """
    # 이진 신호로 변환 (threshold=0.5)
    sig = ensemble_to_binary_signal(ensemble_signal_continuous, threshold=0.5)
    sig = sig.astype(float)

    # Warmup
    wd = price.index[WARMUP_DAYS]
    sig.loc[:wd] = 0
    if ensemble_signal_continuous.loc[wd] < 0.5:
        post = ensemble_signal_continuous.loc[wd:]
        cross_up = post[post >= 0.5].index
        if len(cross_up) > 0:
            sig.loc[wd:cross_up[0]] = 0

    # Backtest
    cum = run_lrs(price, sig, leverage=LEVERAGE, expense_ratio=CALIBRATED_ER,
                  tbill_rate=rf_series, signal_lag=SIGNAL_LAG, commission=COMMISSION)
    cum = cum.loc[wd:]
    cum = cum / cum.iloc[0]
    sig = sig.loc[wd:]

    # Metrics
    rf_scalar = rf_series.mean() * 252
    m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
    m["MDD_Entry"] = _max_entry_drawdown(cum, sig, signal_lag=SIGNAL_LAG)
    m["Max_Recovery_Days"] = _max_recovery_days(cum)
    m["Trades/Yr"] = signal_trades_per_year(sig)

    return cum, sig, m


def main():
    print("=" * 80)
    print("  Regime-Switching Ensemble Strategy")
    print("  과최적화 위험 감소를 위한 다중 파라미터 접근")
    print("=" * 80)

    # 데이터 다운로드
    print("\n[1/4] Downloading data...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    rf_series = download_ken_french_rf()
    eval_start = ndx_price.index[WARMUP_DAYS]
    print(f"  NDX: {ndx_price.index[0].date()} ~ {ndx_price.index[-1].date()}")
    print(f"  Eval: {eval_start.date()} ~ {ndx_price.index[-1].date()} "
          f"({len(ndx_price) - WARMUP_DAYS}d, {(len(ndx_price) - WARMUP_DAYS)/252:.1f}yr)")

    # 후보 파라미터 정보 출력
    print("\n[2/4] Parameter Candidates:")
    for cand in CANDIDATES:
        print(f"\n  {cand['name']} (weight={cand['weight']:.0%})")
        p = cand['params']
        print(f"    MA pairs: low({p['fast_low']},{p['slow_low']}), "
              f"high({p['fast_high']},{p['slow_high']})")
        print(f"    Vol: lookback={p['vol_lookback']}, threshold={p['vol_threshold_pct']:.1f}%")
        print(f"    Note: {cand['note']}")

    # 앙상블 신호 생성
    print("\n[3/4] Creating ensemble signal...")
    ensemble_continuous = create_ensemble_signal(ndx_price, CANDIDATES)

    # 신호 통계
    ens_stats = ensemble_continuous.describe()
    print(f"  Ensemble signal range: {ens_stats['min']:.3f} ~ {ens_stats['max']:.3f}")
    print(f"  Mean: {ens_stats['mean']:.3f}, Std: {ens_stats['std']:.3f}")

    # 앙상블 백테스트
    print("\n[4/4] Backtesting ensemble strategy...")
    cum_ens, sig_ens, metrics_ens = backtest_ensemble(ndx_price, rf_series, ensemble_continuous)

    # 개별 파라미터 백테스트 (비교용)
    print("\n  Individual parameter backtests...")
    individual_results = {}
    individual_curves = {}
    for cand in CANDIDATES:
        p = cand['params']
        sig = signal_regime_switching_dual_ma(
            ndx_price,
            fast_low=p['fast_low'], slow_low=p['slow_low'],
            fast_high=p['fast_high'], slow_high=p['slow_high'],
            vol_lookback=p['vol_lookback'], vol_threshold_pct=p['vol_threshold_pct']
        )

        # Warmup 적용
        wd = ndx_price.index[WARMUP_DAYS]
        sig_test = sig.copy()
        sig_test.loc[:wd] = 0
        if sig.loc[wd] == 1:
            post = sig.loc[wd:]
            zero_cross = post[post == 0].index
            if len(zero_cross) > 0:
                sig_test.loc[wd:zero_cross[0]] = 0

        cum = run_lrs(ndx_price, sig_test, leverage=LEVERAGE,
                     expense_ratio=CALIBRATED_ER, tbill_rate=rf_series,
                     signal_lag=SIGNAL_LAG, commission=COMMISSION)
        cum = cum.loc[wd:]
        cum = cum / cum.iloc[0]
        sig_test = sig_test.loc[wd:]

        rf_scalar = rf_series.mean() * 252
        m = calc_metrics(cum, tbill_rate=rf_scalar, rf_series=rf_series)
        m["MDD_Entry"] = _max_entry_drawdown(cum, sig_test, signal_lag=SIGNAL_LAG)
        m["Max_Recovery_Days"] = _max_recovery_days(cum)
        m["Trades/Yr"] = signal_trades_per_year(sig_test)

        individual_results[cand['name']] = m
        individual_curves[cand['name']] = cum

    # 결과 테이블
    print("\n" + "=" * 130)
    print(f"  Comparison: Individual vs. Ensemble")
    print("=" * 130)
    header = (f"  {'Strategy':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MDD':>9} {'MDD_Ent':>9} {'Recov':>8} {'Trd/Yr':>8} {'Vol':>8}")
    print(header)
    print(f"  {'─' * 125}")

    for name, m in individual_results.items():
        recov = m.get("Max_Recovery_Days", 0)
        recov_str = f"{recov:>7d}d" if isinstance(recov, (int, np.integer)) else "—"
        tpy = m.get("Trades/Yr", 0)
        print(f"  {name:<35} {m['CAGR']:>8.1%} {m['Sharpe']:>8.3f} {m['Sortino']:>9.3f} "
              f"{m['MDD']:>9.1%} {m['MDD_Entry']:>9.1%} {recov_str} {tpy:>8.1f} {m['Volatility']:>8.1%}")

    print(f"  {'─' * 125}")
    recov = metrics_ens.get("Max_Recovery_Days", 0)
    recov_str = f"{recov:>7d}d" if isinstance(recov, (int, np.integer)) else "—"
    tpy = metrics_ens.get("Trades/Yr", 0)
    print(f"  {'ENSEMBLE (Weighted Avg):':<35} {metrics_ens['CAGR']:>8.1%} {metrics_ens['Sharpe']:>8.3f} "
          f"{metrics_ens['Sortino']:>9.3f} {metrics_ens['MDD']:>9.1%} {metrics_ens['MDD_Entry']:>9.1%} "
          f"{recov_str} {tpy:>8.1f} {metrics_ens['Volatility']:>8.1%}")
    print("=" * 130)

    # 개선 효과 분석
    print("\n" + "=" * 80)
    print("  Ensemble Benefits Analysis")
    print("=" * 80)

    best_individual_sortino = max([m['Sortino'] for m in individual_results.values()])
    worst_individual_sortino = min([m['Sortino'] for m in individual_results.values()])

    ens_sortino = metrics_ens['Sortino']

    print(f"\n  Sortino Range (Individual):")
    print(f"    Best:  {best_individual_sortino:.3f}")
    print(f"    Worst: {worst_individual_sortino:.3f}")
    print(f"    Range: {best_individual_sortino - worst_individual_sortino:.3f}")

    print(f"\n  Ensemble Sortino: {ens_sortino:.3f}")
    print(f"    Distance from best: {best_individual_sortino - ens_sortino:+.3f}")
    print(f"    vs. worst: {ens_sortino - worst_individual_sortino:+.3f} ({(ens_sortino - worst_individual_sortino) / (best_individual_sortino - worst_individual_sortino) * 100:.1f}% of range)")

    # MDD_Entry 비교
    best_mdd_entry = max([m['MDD_Entry'] for m in individual_results.values()])  # 가장 작은 손실 (최고 값)
    worst_mdd_entry = min([m['MDD_Entry'] for m in individual_results.values()])  # 가장 큰 손실

    print(f"\n  MDD_Entry Range (Individual):")
    print(f"    Best (least loss):  {best_mdd_entry:.1%}")
    print(f"    Worst (most loss):  {worst_mdd_entry:.1%}")
    print(f"    Range: {best_mdd_entry - worst_mdd_entry:.1%}")

    print(f"\n  Ensemble MDD_Entry: {metrics_ens['MDD_Entry']:.1%}")
    print(f"    Risk mitigation: {abs(metrics_ens['MDD_Entry']) - abs(worst_mdd_entry):.1%} (worst 대비)")

    # Volatility 비교 (평활화 효과)
    vol_avg = np.mean([m['Volatility'] for m in individual_results.values()])
    print(f"\n  Volatility Smoothing:")
    print(f"    Individual average: {vol_avg:.1%}")
    print(f"    Ensemble: {metrics_ens['Volatility']:.1%}")
    print(f"    Reduction: {vol_avg - metrics_ens['Volatility']:+.1%}")

    # 거래 빈도 비교
    trades_avg = np.mean([m.get("Trades/Yr", 0) for m in individual_results.values()])
    print(f"\n  Trade Frequency:")
    print(f"    Individual average: {trades_avg:.1f}/year")
    print(f"    Ensemble: {metrics_ens.get('Trades/Yr', 0):.1f}/year")
    print(f"    Reduction: {trades_avg - metrics_ens.get('Trades/Yr', 0):+.1f}/year ({(1 - metrics_ens.get('Trades/Yr', 0) / trades_avg if trades_avg > 0 else 0) * 100:.0f}%)")

    # 차트 1: 누적 수익 비교
    print("\n[Generating charts...]")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 누적 수익
    ax1.plot(cum_ens.index, cum_ens.values, linewidth=2.5, label='Ensemble (Weighted Avg)',
            color='#2E7D32', alpha=0.9)

    for cand in CANDIDATES:
        cum_ind = individual_curves[cand['name']]
        ax1.plot(cum_ind.index, cum_ind.values, linewidth=1, alpha=0.4,
                label=cand['name'].split('(')[0].strip())

    ax1.set_yscale('log')
    ax1.set_ylabel('Growth of $1 (log scale)')
    ax1.set_title('Regime-Switching Ensemble vs. Individual Parameters (1987-2025)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 신호 히스토그램 (앙상블)
    ax2.hist(ensemble_continuous, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Binary threshold (0.5)')
    ax2.set_xlabel('Ensemble Signal Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Ensemble Signal (Continuous)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ensemble_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> ensemble_comparison.png")

    # 차트 2: 파라미터별 성과 비교
    fig, ax = plt.subplots(figsize=(12, 7))

    names = [cand['name'].split('(')[0].strip() for cand in CANDIDATES] + ['Ensemble']
    sortinos = [individual_results[cand['name']]['Sortino'] for cand in CANDIDATES] + [metrics_ens['Sortino']]
    cagrs = [individual_results[cand['name']]['CAGR'] for cand in CANDIDATES] + [metrics_ens['CAGR']]
    mdd_entries = [individual_results[cand['name']]['MDD_Entry'] for cand in CANDIDATES] + [metrics_ens['MDD_Entry']]

    x = np.arange(len(names))
    width = 0.25

    bars1 = ax.bar(x - width, sortinos, width, label='Sortino', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x, cagrs, width, label='CAGR', alpha=0.8, color='orange')

    ax.set_ylabel('Sortino Ratio', fontsize=11)
    ax2.set_ylabel('CAGR (%)', fontsize=11)
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_title('Regime-Switching: Individual vs. Ensemble Performance', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Sortino=1.0')

    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ensemble_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  -> ensemble_metrics.png")

    # CSV 저장
    results_df = pd.DataFrame([
        {
            'Strategy': cand['name'],
            'CAGR': individual_results[cand['name']]['CAGR'],
            'Sharpe': individual_results[cand['name']]['Sharpe'],
            'Sortino': individual_results[cand['name']]['Sortino'],
            'MDD': individual_results[cand['name']]['MDD'],
            'MDD_Entry': individual_results[cand['name']]['MDD_Entry'],
            'Volatility': individual_results[cand['name']]['Volatility'],
            'Trades_Yr': individual_results[cand['name']].get('Trades/Yr', 0),
            'Weight': cand['weight'],
        }
        for cand in CANDIDATES
    ] + [
        {
            'Strategy': 'ENSEMBLE (Weighted Average)',
            'CAGR': metrics_ens['CAGR'],
            'Sharpe': metrics_ens['Sharpe'],
            'Sortino': metrics_ens['Sortino'],
            'MDD': metrics_ens['MDD'],
            'MDD_Entry': metrics_ens['MDD_Entry'],
            'Volatility': metrics_ens['Volatility'],
            'Trades_Yr': metrics_ens.get('Trades/Yr', 0),
            'Weight': 1.0,
        }
    ])

    results_df.to_csv(OUT_DIR / "ensemble_results.csv", index=False)
    print(f"  -> ensemble_results.csv")

    print("\n" + "=" * 80)
    print("  Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
