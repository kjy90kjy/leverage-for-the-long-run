"""
김째 S2 과열 감량 구조 심층 분석

1. 과열 단계별 발동 이력 추적
2. 과열 감량 유/무 비교 (구조적 기여 측정)
3. 라운드 넘버 대체 테스트 (과최적화 검증)
4. 단계 수 변형 (1단계/2단계/4단계)
5. 다른 자산 대용 테스트 (보편성 검증)

실행:
    python tests/test_kimjje_overheat.py
"""

import sys
import io
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrs_standalone import (
    BasicParams, download_all_data, backtest_kimjje, calc_metrics,
    count_trades, compute_kimjje_strategy, sample_stdev,
    COMPARE_START, OUT_DIR,
)

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
# 1. 과열 단계 발동 이력 추적
# ══════════════════════════════════════════════════════════════

def trace_overheat_history(tqqq, qqq, spy):
    """과열 단계를 날짜별로 추적하고, 각 단계 진입/탈출 이벤트 기록."""
    params = BasicParams(spy_bear_cap=0.0)
    tqqq_arr = tqqq.values.astype(float)
    t_ma200 = pd.Series(tqqq_arr).rolling(200, min_periods=200).mean().to_numpy()

    # dist200 계산
    dist200 = np.where(
        np.isnan(tqqq_arr) | np.isnan(t_ma200) | (t_ma200 == 0),
        np.nan, (tqqq_arr / t_ma200) * 100.0
    )

    # 전체 전략 실행 (codes 추출)
    codes, weights = compute_kimjje_strategy(
        qqq.values.astype(float),
        tqqq_arr,
        spy.values.astype(float),
        params,
    )

    df = pd.DataFrame({
        "date": tqqq.index,
        "tqqq": tqqq_arr,
        "dist200": dist200,
        "code": codes,
        "weight": weights,
    })

    return df


def print_overheat_events(df):
    """과열 이벤트 요약 출력."""
    print("\n" + "=" * 70)
    print("과열 이력: TQQQ 200일선 이격도 139%+ 도달 시점")
    print("=" * 70)

    # dist200 >= 139 인 구간 찾기
    mask = df["dist200"] >= 139.0
    if mask.sum() == 0:
        print("  과열 구간 없음")
        return

    # 연속 구간 그룹핑
    groups = (mask != mask.shift()).cumsum()
    hot_groups = df[mask].groupby(groups[mask])

    print(f"\n{'#':>3s} {'시작':>12s} {'종료':>12s} {'일수':>5s} {'최고 이격도':>12s} {'비중 변화':>20s}")
    print("-" * 70)

    for idx, (gid, grp) in enumerate(hot_groups, 1):
        start = grp["date"].iloc[0].date()
        end = grp["date"].iloc[-1].date()
        days = len(grp)
        peak_dist = grp["dist200"].max()

        # 비중 변화 추적
        w_enter = grp["weight"].iloc[0]
        w_min = grp["weight"].min()
        w_exit = grp["weight"].iloc[-1]
        w_str = f"{w_enter:.0%} → {w_min:.0%} → {w_exit:.0%}"

        print(f"{idx:3d} {start} {end} {days:5d} {peak_dist:11.1f}% {w_str:>20s}")

    # dist200 분포 통계
    valid = df["dist200"].dropna()
    print(f"\n  전체 이격도 통계:")
    for pct in [90, 95, 99, 99.5]:
        print(f"    {pct}th percentile: {np.percentile(valid, pct):.1f}%")
    print(f"    최대: {valid.max():.1f}%")
    print(f"    139%+ 일수: {mask.sum()}일 ({mask.sum()/len(valid)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# 2. 과열 감량 변형 비교
# ══════════════════════════════════════════════════════════════

def run_overheat_variants(tqqq, qqq, spy, rf, compare_start):
    """다양한 과열 설정으로 비교."""
    print("\n" + "=" * 70)
    print("과열 감량 변형 비교")
    print("=" * 70)

    start_mask = tqqq.index >= compare_start
    rf_t = rf.loc[start_mask]

    variants = {
        # 기본: 과열 없음
        "과열 없음": BasicParams(
            spy_bear_cap=0.0,
            use_overheat_split=False,
        ),

        # 1단계: 150% 이상이면 전량 청산
        "1단계 (150→0%)": BasicParams(
            spy_bear_cap=0.0,
            overheat1_enter=150.0, overheat1_exit=130.0,
            overheat2_enter=999.0, overheat3_enter=999.0, overheat4_enter=999.0,
        ),

        # 2단계: 라운드 넘버
        "2단계 (140→90%, 150→0%)": BasicParams(
            spy_bear_cap=0.0,
            overheat1_enter=140.0, overheat1_exit=130.0,
            overheat2_enter=150.0, overheat2_exit=135.0,
            overheat3_enter=999.0, overheat4_enter=999.0,
        ),

        # 원본 4단계
        "4단계 원본 (139/146/149/151)": BasicParams(
            spy_bear_cap=0.0,
        ),

        # 4단계 라운드 넘버
        "4단계 라운드 (140/150/155/160)": BasicParams(
            spy_bear_cap=0.0,
            overheat1_enter=140.0, overheat1_exit=130.0,
            overheat2_enter=150.0, overheat2_exit=140.0,
            overheat3_enter=155.0, overheat3_exit=145.0,
            overheat4_enter=160.0, overheat4_exit=120.0,
        ),

        # 공격적: 더 빨리 감량
        "4단계 공격적 (130/140/145/150)": BasicParams(
            spy_bear_cap=0.0,
            overheat1_enter=130.0, overheat1_exit=120.0,
            overheat2_enter=140.0, overheat2_exit=130.0,
            overheat3_enter=145.0, overheat3_exit=135.0,
            overheat4_enter=150.0, overheat4_exit=115.0,
        ),

        # 보수적: 더 늦게 감량
        "4단계 보수적 (150/160/165/170)": BasicParams(
            spy_bear_cap=0.0,
            overheat1_enter=150.0, overheat1_exit=140.0,
            overheat2_enter=160.0, overheat2_exit=150.0,
            overheat3_enter=165.0, overheat3_exit=155.0,
            overheat4_enter=170.0, overheat4_exit=125.0,
        ),
    }

    header = f"{'변형':35s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} {'Trades/Y':>9s}"
    print(f"\n{header}")
    print("-" * len(header))

    results_for_plot = {}

    for name, params in variants.items():
        cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
        c = cum.loc[start_mask]
        c = c / c.iloc[0]
        m = calc_metrics(c, rf_t)
        trades = count_trades(w.loc[start_mask], is_binary=False)

        print(f"{name:35s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} {m['Sortino']:>8.2f} "
              f"{m['MDD']:>7.1%} {trades:>8.1f}")

        results_for_plot[name] = c

    return results_for_plot


# ══════════════════════════════════════════════════════════════
# 3. 과열 구간 시각화
# ══════════════════════════════════════════════════════════════

def plot_overheat_analysis(df, curves, compare_start, fname):
    """이격도 + 비중 + equity 비교 3-panel."""
    mask = df["date"] >= compare_start

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12),
                                          height_ratios=[2, 1, 3],
                                          sharex=True,
                                          gridspec_kw={"hspace": 0.08})

    dates = df.loc[mask, "date"]
    dist = df.loc[mask, "dist200"]
    weights = df.loc[mask, "weight"]

    # Panel 1: 이격도 + 과열 구간 표시
    ax1.plot(dates, dist, color="#333", linewidth=0.8, alpha=0.8)
    ax1.axhline(139, color="#FF9800", linestyle="--", alpha=0.5, label="Stage 1 (139%)")
    ax1.axhline(146, color="#FF5722", linestyle="--", alpha=0.5, label="Stage 2 (146%)")
    ax1.axhline(149, color="#F44336", linestyle="--", alpha=0.5, label="Stage 3 (149%)")
    ax1.axhline(151, color="#B71C1C", linestyle="--", alpha=0.5, label="Stage 4 (151%)")
    ax1.axhline(100, color="#9E9E9E", linestyle=":", alpha=0.3)
    ax1.fill_between(dates, dist, 139, where=(dist >= 139), alpha=0.2, color="#FF5722")
    ax1.set_ylabel("TQQQ 200일선 이격도 (%)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("김째 S2 과열 감량 구조 분석", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.2)

    # Panel 2: 비중
    ax2.fill_between(dates, weights, 0, alpha=0.5, color="#2196F3")
    ax2.set_ylabel("TQQQ 비중")
    ax2.set_ylim(-0.05, 1.1)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.grid(True, alpha=0.2)

    # Panel 3: Equity curves 비교
    colors = ["#9E9E9E", "#8BC34A", "#FF9800", "#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for i, (name, cum) in enumerate(curves.items()):
        lw = 2.0 if "원본" in name else 1.2
        alpha = 1.0 if "원본" in name or "없음" in name else 0.7
        ax3.plot(cum.index, cum.values, label=name, color=colors[i % len(colors)],
                 linewidth=lw, alpha=alpha)

    ax3.set_yscale("log")
    ax3.set_ylabel("누적 수익률 (log)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {path}")


# ══════════════════════════════════════════════════════════════
# 4. 합성 데이터 (닷컴 포함)에서 과열 분석
# ══════════════════════════════════════════════════════════════

def run_overheat_synthetic(compare_start="2000-01-01"):
    """합성 3x QQQ 데이터에서 과열 감량 효과 검증."""
    print("\n" + "=" * 70)
    print(f"합성 3x QQQ 과열 변형 비교 ({compare_start}~)")
    print("=" * 70)

    data = download_all_data(mode="synth")
    tqqq, qqq, spy, rf = data["tqqq"], data["qqq"], data["spy"], data["rf"]

    start_mask = tqqq.index >= compare_start
    rf_t = rf.loc[start_mask]

    variants = {
        "과열 없음": BasicParams(spy_bear_cap=0.0, use_overheat_split=False),
        "2단계 라운드 (140/150)": BasicParams(
            spy_bear_cap=0.0,
            overheat1_enter=140.0, overheat1_exit=130.0,
            overheat2_enter=150.0, overheat2_exit=135.0,
            overheat3_enter=999.0, overheat4_enter=999.0,
        ),
        "4단계 원본": BasicParams(spy_bear_cap=0.0),
        "4단계 라운드 (140/150/155/160)": BasicParams(
            spy_bear_cap=0.0,
            overheat1_enter=140.0, overheat1_exit=130.0,
            overheat2_enter=150.0, overheat2_exit=140.0,
            overheat3_enter=155.0, overheat3_exit=145.0,
            overheat4_enter=160.0, overheat4_exit=120.0,
        ),
    }

    header = f"{'변형':35s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, params in variants.items():
        cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
        c = cum.loc[start_mask]
        c = c / c.iloc[0]
        m = calc_metrics(c, rf_t)
        print(f"{name:35s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} {m['Sortino']:>8.2f} {m['MDD']:>7.1%}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  김째 S2 과열 감량 구조 심층 분석                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # 실제 TQQQ 데이터
    data = download_all_data(mode="real")
    tqqq, qqq, spy, rf = data["tqqq"], data["qqq"], data["spy"], data["rf"]

    # 1) 과열 이력 추적
    print("\n[1/4] 과열 발동 이력")
    df = trace_overheat_history(tqqq, qqq, spy)
    print_overheat_events(df)

    # 2) 과열 변형 비교 (실제 TQQQ)
    print("\n[2/4] 과열 변형 비교 (실제 TQQQ, 2011~)")
    curves = run_overheat_variants(tqqq, qqq, spy, rf, COMPARE_START)

    # 3) 시각화
    print("\n[3/4] 시각화")
    plot_overheat_analysis(df, curves, COMPARE_START, "kimjje_overheat_analysis.png")

    # 4) 합성 데이터 검증
    print("\n[4/4] 합성 데이터 검증")
    run_overheat_synthetic("2000-01-01")

    print("\n완료!")
