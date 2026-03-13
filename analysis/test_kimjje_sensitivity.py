"""
김째매매법 S2 파라미터 감도 분석 + 축소 모델 + Walk-Forward 검증

핵심 질문: 현재 파라미터가 넓은 고원(plateau) 위인가, 뾰족한 봉우리(peak) 위인가?

실행:
    python analysis/test_kimjje_sensitivity.py
"""

import sys
import io
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import time
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
    count_trades, COMPARE_START, OUT_DIR,
)

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
# 1. 파라미터 감도 분석
# ══════════════════════════════════════════════════════════════

PARAM_RANGES = {
    "vol_threshold":      (0.035, 0.085, 0.005,  0.059,  "변동성 락 (%)"),
    "spy_exit":           (94.0,  100.0, 0.5,    97.75,  "SPY 탈출선 (%)"),
    "principal_stop_pct": (0.90,  0.97,  0.005,  0.941,  "손절 기준 (×진입가)"),
    "rsi_reentry_thr":    (25.0,  60.0,  2.5,    43.0,   "RSI 재진입 임계"),
    "dist200_enter":      (98.0,  105.0, 0.5,    101.0,  "200일선 진입 이격도"),
    "overheat1_enter":    (125.0, 155.0, 2.0,    139.0,  "과열1 진입 이격도"),
    "slope_thr":          (0.02,  0.22,  0.02,   0.11,   "기울기 임계값"),
}


def run_sensitivity(tqqq, qqq, spy, rf, compare_start):
    """각 파라미터를 변동시키며 CAGR/Sharpe/MDD 기록."""
    start_mask = tqqq.index >= compare_start
    rf_t = rf.loc[start_mask]

    results = {}
    base_params = BasicParams(spy_bear_cap=0.0)

    total = sum(
        len(np.arange(lo, hi + step / 2, step))
        for lo, hi, step, _, _ in PARAM_RANGES.values()
    )
    done = 0
    t0 = time.time()

    for param_name, (lo, hi, step, default, label) in PARAM_RANGES.items():
        values = np.arange(lo, hi + step / 2, step)
        records = []

        for val in values:
            done += 1
            params = replace(base_params, **{param_name: val})
            cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)

            c = cum.loc[start_mask]
            c = c / c.iloc[0]
            m = calc_metrics(c, rf_t)
            m["Trades/Yr"] = count_trades(w.loc[start_mask], is_binary=False)
            m["value"] = val
            records.append(m)

            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"\r  [{done}/{total}] {param_name}={val:.4f}  "
                  f"CAGR={m['CAGR']:.1%}  Sharpe={m['Sharpe']:.2f}  "
                  f"(ETA {eta:.0f}s)", end="", flush=True)

        results[param_name] = (values, records, default, label)

    print()
    return results


def plot_sensitivity(results, fname):
    """파라미터별 감도 차트 (CAGR + Sharpe 이축)."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), squeeze=False)

    for idx, (param_name, (values, records, default, label)) in enumerate(results.items()):
        ax1 = axes[idx, 0]

        cagrs = [r["CAGR"] for r in records]
        sharpes = [r["Sharpe"] for r in records]
        mdds = [r["MDD"] for r in records]

        # CAGR (왼쪽 축)
        color_cagr = "#2196F3"
        ax1.plot(values, cagrs, "o-", color=color_cagr, linewidth=2, markersize=4,
                 label="CAGR")
        ax1.set_ylabel("CAGR", color=color_cagr)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax1.tick_params(axis="y", labelcolor=color_cagr)

        # Sharpe (오른쪽 축)
        ax2 = ax1.twinx()
        color_sharpe = "#FF5722"
        ax2.plot(values, sharpes, "s--", color=color_sharpe, linewidth=1.5, markersize=3,
                 label="Sharpe")
        ax2.set_ylabel("Sharpe", color=color_sharpe)
        ax2.tick_params(axis="y", labelcolor=color_sharpe)

        # 현재 값 표시
        ax1.axvline(default, color="#333", linestyle=":", linewidth=2, alpha=0.7)
        ax1.set_title(f"{label}  ({param_name}={default})", fontsize=11, fontweight="bold")
        ax1.set_xlabel(param_name)
        ax1.grid(True, alpha=0.2)

        # 범위 표시 (±20%)
        lo20 = default * 0.8
        hi20 = default * 1.2
        ax1.axvspan(lo20, hi20, alpha=0.08, color="green")

        # 범례
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {path}")


def print_plateau_analysis(results):
    """각 파라미터에 대해 plateau 폭 분석."""
    print("\n" + "=" * 70)
    print("파라미터 고원(Plateau) 분석")
    print("=" * 70)

    # 기준: default의 Sharpe 90% 이상인 구간 = plateau
    for param_name, (values, records, default, label) in results.items():
        sharpes = np.array([r["Sharpe"] for r in records])
        cagrs = np.array([r["CAGR"] for r in records])
        default_idx = np.argmin(np.abs(values - default))
        default_sharpe = sharpes[default_idx]
        default_cagr = cagrs[default_idx]

        # Sharpe 90% 이상인 영역
        threshold = default_sharpe * 0.9
        plateau_mask = sharpes >= threshold
        plateau_values = values[plateau_mask]

        if len(plateau_values) > 0:
            plateau_lo = plateau_values[0]
            plateau_hi = plateau_values[-1]
            plateau_width = plateau_hi - plateau_lo
            total_range = values[-1] - values[0]
            plateau_pct = plateau_width / total_range * 100

            # 판정
            if plateau_pct >= 60:
                verdict = "넓은 고원 (robust)"
            elif plateau_pct >= 30:
                verdict = "중간 고원 (moderate)"
            else:
                verdict = "좁은 봉우리 (fragile!)"
        else:
            plateau_lo = plateau_hi = default
            plateau_pct = 0
            verdict = "고원 없음 (극도 취약)"

        print(f"\n  {label} ({param_name})")
        print(f"    현재값: {default}  →  CAGR={default_cagr:.1%}, Sharpe={default_sharpe:.2f}")
        print(f"    Sharpe 90%+ 구간: [{plateau_lo:.3f} ~ {plateau_hi:.3f}]  "
              f"({plateau_pct:.0f}% of range)")
        print(f"    판정: {verdict}")


# ══════════════════════════════════════════════════════════════
# 2. 축소 모델 비교
# ══════════════════════════════════════════════════════════════

def run_reduced_models(tqqq, qqq, spy, rf, compare_start):
    """S2 Full / S2-Lite / S2-Minimal 비교."""
    print("\n" + "=" * 70)
    print("축소 모델 비교")
    print("=" * 70)

    start_mask = tqqq.index >= compare_start
    rf_t = rf.loc[start_mask]

    models = {
        "S2 Full (27p)": BasicParams(spy_bear_cap=0.0),

        "S2-Lite (11p)": BasicParams(
            spy_bear_cap=0.0,
            use_tp10=False,             # TP10 제거
            use_slope_boost=False,      # Slope Boost 제거
            # 과열 2단계로 축소 (stage 3,4 = stage 2)
            overheat3_enter=999.0,      # 비활성화
            overheat4_enter=999.0,
        ),

        "S2-Simple (8p)": BasicParams(
            spy_bear_cap=0.0,
            use_tp10=False,
            use_slope_boost=False,
            use_overheat_split=False,   # 과열 전체 제거
            # SPY/Vol/손절만 유지 (라운드 넘버)
            vol_threshold=0.06,         # 5.9% → 6%
            spy_exit=98.0,              # 97.75 → 98
            principal_stop_pct=0.94,    # 0.941 → 0.94
            rsi_reentry_thr=40.0,       # 43 → 40
        ),

        "S2-Minimal (5p)": BasicParams(
            spy_bear_cap=0.0,
            use_tp10=False,
            use_slope_boost=False,
            use_overheat_split=False,
            use_principal_stop=False,   # 손절도 제거
            use_spy_filter=False,       # SPY 필터도 제거
            # 200MA + Vol Lock만
            vol_threshold=0.06,
        ),
    }

    header = f"{'모델':20s} {'Params':>6s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} {'Trades/Y':>9s}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, params in models.items():
        cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
        c = cum.loc[start_mask]
        c = c / c.iloc[0]
        m = calc_metrics(c, rf_t)
        trades = count_trades(w.loc[start_mask], is_binary=False)

        # 파라미터 수 추출
        pcount = name.split("(")[1].rstrip(")")
        print(f"{name:20s} {pcount:>6s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} "
              f"{m['Sortino']:>8.2f} {m['MDD']:>7.1%} {trades:>8.1f}")


# ══════════════════════════════════════════════════════════════
# 3. Walk-Forward 검증
# ══════════════════════════════════════════════════════════════

def run_walk_forward(tqqq, qqq, spy, rf):
    """In-sample vs Out-of-sample 성과 비교."""
    print("\n" + "=" * 70)
    print("Walk-Forward 검증 (S2 기본 파라미터 고정)")
    print("=" * 70)

    splits = [
        ("2011-01-01", "2016-01-01", "2016-01-01", "2019-01-01"),
        ("2011-01-01", "2019-01-01", "2019-01-01", "2022-01-01"),
        ("2011-01-01", "2022-01-01", "2022-01-01", "2026-12-31"),
    ]

    params = BasicParams(spy_bear_cap=0.0)

    header = f"{'구간':30s} {'기간':>12s} {'CAGR':>8s} {'Sharpe':>8s} {'MDD':>8s}"
    print(f"\n{header}")
    print("-" * len(header))

    for is_start, is_end, os_start, os_end in splits:
        for label, s, e in [("IS", is_start, is_end), ("OS", os_start, os_end)]:
            mask = (tqqq.index >= s) & (tqqq.index < e)
            if mask.sum() < 50:
                continue
            cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
            c = cum.loc[mask]
            if len(c) < 2:
                continue
            c = c / c.iloc[0]
            rf_sub = rf.loc[mask]
            m = calc_metrics(c, rf_sub)

            period = f"{s[:4]}~{e[:4]}"
            tag = f"[{label}] {period}"
            print(f"{tag:30s} {period:>12s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} {m['MDD']:>7.1%}")
        print()

    print("  IS = In-Sample (학습 기간), OS = Out-of-Sample (검증 기간)")
    print("  OS Sharpe가 IS의 50% 이하면 과최적화 우려")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  김째 S2 파라미터 감도 분석 / 축소 모델 / Walk-Forward     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    data = download_all_data(mode="real")

    # 1) 파라미터 감도 분석
    print("\n[1/3] 파라미터 감도 분석")
    print("-" * 60)
    results = run_sensitivity(
        data["tqqq"], data["qqq"], data["spy"], data["rf"],
        compare_start=COMPARE_START,
    )
    plot_sensitivity(results, "kimjje_sensitivity.png")
    print_plateau_analysis(results)

    # 2) 축소 모델 비교
    print("\n[2/3] 축소 모델 비교")
    run_reduced_models(
        data["tqqq"], data["qqq"], data["spy"], data["rf"],
        compare_start=COMPARE_START,
    )

    # 3) Walk-Forward
    print("\n[3/3] Walk-Forward 검증")
    run_walk_forward(data["tqqq"], data["qqq"], data["spy"], data["rf"])

    print("\n완료!")
