"""
김째 S2 과열 파라미터 최적화 — Multi-Period Grid Search + Plateau Detection

안티-오버피팅 3중 장치:
  1. Multi-period 최소 Sortino (NDX 40yr, QQQ 27yr, TQQQ 15yr 중 worst)
  2. Neighbor-averaging (넓은 고원 중심 선택)
  3. Walk-Forward 검증 (IS/OS 분할)

Model A: 2-Stage (4 params) — ~384 combos
Model B: 4-Stage Structured (5 params) — ~11,520 combos

실행:
    python optimization/optimize_kimjje_overheat.py
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
import optuna
from optuna.samplers import TPESampler

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
# [1] 구조적 파라미터 생성
# ══════════════════════════════════════════════════════════════

def generate_2stage_params():
    """Model A: 2-Stage 과열 파라미터 생성.

    s1_enter: 130~160 (step 2)
    s1_exit:  115~140 (step 5)
    gap:      5~20 (step 5) — s2_enter = s1_enter + gap
    s2_exit:  s1_exit + 5

    Returns: list of (grid_key_dict, BasicParams)
    """
    results = []
    for s1_enter in range(130, 161, 2):
        for s1_exit in range(115, 141, 5):
            if s1_exit >= s1_enter:
                continue
            for gap in range(5, 21, 5):
                s2_enter = s1_enter + gap
                s2_exit = s1_exit + 5

                if s2_exit >= s2_enter:
                    continue

                params = replace(
                    BasicParams(spy_bear_cap=0.0),
                    overheat1_enter=float(s1_enter),
                    overheat1_exit=float(s1_exit),
                    overheat2_enter=float(s2_enter),
                    overheat2_exit=float(s2_exit),
                    overheat3_enter=999.0,
                    overheat3_exit=900.0,
                    overheat4_enter=999.0,
                    overheat4_exit=900.0,
                )

                key = {
                    "s1_enter": s1_enter,
                    "s1_exit": s1_exit,
                    "gap": gap,
                }
                results.append((key, params))

    return results


def generate_4stage_params():
    """Model B: 4-Stage Structured 과열 파라미터 생성.

    base_enter: 130~160 (step 2)
    spread:     2~10 (step 2) — 단계 간 진입 간격
    s4_gap:     0~6 (step 2) — 4단계 추가 간격
    base_exit:  115~140 (step 5)
    exit_decay: 3~8 (step 1) — 탈출선 상승 간격

    Returns: list of (grid_key_dict, BasicParams)
    """
    results = []
    for base_enter in range(130, 161, 2):
        for spread in range(2, 11, 2):
            for s4_gap in range(0, 7, 2):
                for base_exit in range(115, 141, 5):
                    if base_exit >= base_enter:
                        continue
                    for exit_decay in range(3, 9, 1):
                        s1_enter = float(base_enter)
                        s2_enter = float(base_enter + spread)
                        s3_enter = float(base_enter + 2 * spread)
                        s4_enter = float(base_enter + 3 * spread + s4_gap)

                        s1_exit = float(base_exit)
                        s2_exit = float(base_exit + exit_decay)
                        s3_exit = float(base_exit + 2 * exit_decay)
                        s4_exit = float(base_exit - 15)  # 깊은 탈출선

                        # 유효성 검사: exit < enter
                        if s1_exit >= s1_enter:
                            continue
                        if s2_exit >= s2_enter:
                            continue
                        if s3_exit >= s3_enter:
                            continue
                        if s4_exit >= s4_enter:
                            continue

                        params = replace(
                            BasicParams(spy_bear_cap=0.0),
                            overheat1_enter=s1_enter,
                            overheat1_exit=s1_exit,
                            overheat2_enter=s2_enter,
                            overheat2_exit=s2_exit,
                            overheat3_enter=s3_enter,
                            overheat3_exit=s3_exit,
                            overheat4_enter=s4_enter,
                            overheat4_exit=s4_exit,
                        )

                        key = {
                            "base_enter": base_enter,
                            "spread": spread,
                            "s4_gap": s4_gap,
                            "base_exit": base_exit,
                            "exit_decay": exit_decay,
                        }
                        results.append((key, params))

    return results


# ══════════════════════════════════════════════════════════════
# [2] Multi-Period 평가
# ══════════════════════════════════════════════════════════════

def eval_across_periods(params, datasets):
    """3개 기간 중 worst-case Sortino를 반환.

    datasets: dict of {name: (tqqq, qqq, spy, rf, start_date)}
    Returns: (min_sortino, detail_dict)
    """
    details = {}
    sortinos = []
    for name, (tqqq, qqq, spy, rf, start) in datasets.items():
        cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
        mask = cum.index >= start
        c = cum.loc[mask]
        if len(c) < 50:
            sortinos.append(0.0)
            details[name] = {"Sortino": 0.0, "CAGR": 0.0, "Sharpe": 0.0, "MDD": 0.0}
            continue
        c = c / c.iloc[0]
        rf_sub = rf.loc[mask]
        m = calc_metrics(c, rf_sub)
        sortinos.append(m["Sortino"])
        details[name] = m
    return min(sortinos), details


# ══════════════════════════════════════════════════════════════
# [3] Grid Search 실행
# ══════════════════════════════════════════════════════════════

def run_grid(param_list, datasets, label="Grid"):
    """Grid search 실행.

    param_list: list of (grid_key_dict, BasicParams)
    Returns: DataFrame with grid keys + min_sortino + per-period metrics
    """
    total = len(param_list)
    print(f"\n  {label}: {total} combinations × {len(datasets)} datasets")

    records = []
    t0 = time.time()

    for idx, (key, params) in enumerate(param_list):
        min_sort, details = eval_across_periods(params, datasets)

        row = dict(key)
        row["min_sortino"] = min_sort
        for dname, m in details.items():
            row[f"{dname}_sortino"] = m["Sortino"]
            row[f"{dname}_cagr"] = m["CAGR"]
            row[f"{dname}_sharpe"] = m["Sharpe"]
            row[f"{dname}_mdd"] = m["MDD"]
        records.append(row)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"\r    [{idx+1}/{total}]  best min_sortino={max(r['min_sortino'] for r in records):.3f}"
                  f"  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", end="", flush=True)

    print()
    df = pd.DataFrame(records)
    return df


# ══════════════════════════════════════════════════════════════
# [4] Plateau 식별
# ══════════════════════════════════════════════════════════════

def identify_plateaus(results_df, grid_keys, n=3, top_pct=0.05, min_dist=2.0):
    """Top combos 중 neighbor-average가 높은 plateau 중심을 찾는다.

    1. Top top_pct by min_sortino
    2. 각 후보의 이웃(grid_keys 각 차원 ±1 step) 평균 min_sortino 계산
    3. Greedy diverse selection (normalized L2 >= min_dist)

    Returns: list of (grid_key_dict, neighbor_avg_sortino, own_sortino)
    """
    if results_df.empty:
        return []

    # 1. Top 후보
    threshold = results_df["min_sortino"].quantile(1.0 - top_pct)
    top = results_df[results_df["min_sortino"] >= threshold].copy()

    if len(top) < 2:
        top = results_df.nlargest(max(5, len(results_df) // 20), "min_sortino").copy()

    # 2. Neighbor averaging — grid_keys 각 차원의 step 계산
    steps = {}
    for k in grid_keys:
        vals = sorted(results_df[k].unique())
        if len(vals) > 1:
            diffs = np.diff(vals)
            steps[k] = float(np.min(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        else:
            steps[k] = 1.0

    # min_sortino lookup (grid_key tuple → min_sortino)
    lookup = {}
    for _, row in results_df.iterrows():
        key_tuple = tuple(row[k] for k in grid_keys)
        lookup[key_tuple] = row["min_sortino"]

    # 각 top 후보의 neighbor avg 계산
    neighbor_avgs = []
    for _, row in top.iterrows():
        center = [row[k] for k in grid_keys]
        neighbors = []
        for dim_idx, k in enumerate(grid_keys):
            for delta in [-1, 1]:
                nb = list(center)
                nb[dim_idx] = center[dim_idx] + delta * steps[k]
                nb_tuple = tuple(nb)
                if nb_tuple in lookup:
                    neighbors.append(lookup[nb_tuple])
        if neighbors:
            avg = np.mean(neighbors)
        else:
            avg = row["min_sortino"]
        neighbor_avgs.append(avg)

    top = top.copy()
    top["neighbor_avg"] = neighbor_avgs
    top = top.sort_values("neighbor_avg", ascending=False)

    # 3. Greedy diverse selection
    # Normalize grid keys to [0, 1] for distance calculation
    ranges = {}
    for k in grid_keys:
        lo = results_df[k].min()
        hi = results_df[k].max()
        ranges[k] = (lo, hi - lo) if (hi - lo) > 0 else (lo, 1.0)

    selected = []
    for _, row in top.iterrows():
        key_dict = {k: row[k] for k in grid_keys}
        norm = np.array([(row[k] - ranges[k][0]) / ranges[k][1] for k in grid_keys])

        too_close = False
        for _, _, _, prev_norm in selected:
            if np.linalg.norm(norm - prev_norm) < min_dist:
                too_close = True
                break
        if not too_close:
            selected.append((key_dict, row["neighbor_avg"], row["min_sortino"], norm))

        if len(selected) >= n:
            break

    return [(k, na, ms) for k, na, ms, _ in selected]


# ══════════════════════════════════════════════════════════════
# [5] Fine Grid (step=1, ± 범위)
# ══════════════════════════════════════════════════════════════

def run_fine_grid_2stage(centre, datasets):
    """Model A: 2-Stage fine grid around plateau centre."""
    s1e = int(centre["s1_enter"])
    s1x = int(centre["s1_exit"])
    gap = int(centre["gap"])

    results = []
    for s1_enter in range(s1e - 4, s1e + 5):
        for s1_exit in range(s1x - 4, s1x + 5):
            if s1_exit >= s1_enter:
                continue
            for g in range(max(2, gap - 4), gap + 5):
                s2_enter = s1_enter + g
                s2_exit = s1_exit + 5
                if s2_exit >= s2_enter:
                    continue

                params = replace(
                    BasicParams(spy_bear_cap=0.0),
                    overheat1_enter=float(s1_enter),
                    overheat1_exit=float(s1_exit),
                    overheat2_enter=float(s2_enter),
                    overheat2_exit=float(s2_exit),
                    overheat3_enter=999.0,
                    overheat3_exit=900.0,
                    overheat4_enter=999.0,
                    overheat4_exit=900.0,
                )
                key = {"s1_enter": s1_enter, "s1_exit": s1_exit, "gap": g}
                results.append((key, params))

    return run_grid(results, datasets, label="Fine-2S")


def run_fine_grid_4stage(centre, datasets):
    """Model B: 4-Stage fine grid around plateau centre."""
    be = int(centre["base_enter"])
    sp = int(centre["spread"])
    sg = int(centre["s4_gap"])
    bx = int(centre["base_exit"])
    ed = int(centre["exit_decay"])

    results = []
    for base_enter in range(be - 3, be + 4):
        for spread in range(max(1, sp - 2), sp + 3):
            for s4_gap in range(max(0, sg - 2), sg + 3):
                for base_exit in range(bx - 4, bx + 5):
                    if base_exit >= base_enter:
                        continue
                    for exit_decay in range(max(1, ed - 2), ed + 3):
                        s1_enter = float(base_enter)
                        s2_enter = float(base_enter + spread)
                        s3_enter = float(base_enter + 2 * spread)
                        s4_enter = float(base_enter + 3 * spread + s4_gap)
                        s1_exit = float(base_exit)
                        s2_exit = float(base_exit + exit_decay)
                        s3_exit = float(base_exit + 2 * exit_decay)
                        s4_exit = float(base_exit - 15)

                        if s1_exit >= s1_enter or s2_exit >= s2_enter:
                            continue
                        if s3_exit >= s3_enter or s4_exit >= s4_enter:
                            continue

                        params = replace(
                            BasicParams(spy_bear_cap=0.0),
                            overheat1_enter=s1_enter,
                            overheat1_exit=s1_exit,
                            overheat2_enter=s2_enter,
                            overheat2_exit=s2_exit,
                            overheat3_enter=s3_enter,
                            overheat3_exit=s3_exit,
                            overheat4_enter=s4_enter,
                            overheat4_exit=s4_exit,
                        )
                        key = {
                            "base_enter": base_enter,
                            "spread": spread,
                            "s4_gap": s4_gap,
                            "base_exit": base_exit,
                            "exit_decay": exit_decay,
                        }
                        results.append((key, params))

    return run_grid(results, datasets, label="Fine-4S")


# ══════════════════════════════════════════════════════════════
# [6] Walk-Forward 검증
# ══════════════════════════════════════════════════════════════

def walk_forward_validate(params, tqqq, qqq, spy, rf):
    """3-split walk-forward 검증.

    Returns: list of (is_period, os_period, is_sharpe, os_sharpe, os_ratio)
    """
    splits = [
        ("2011-01-01", "2016-01-01", "2016-01-01", "2019-01-01"),
        ("2011-01-01", "2019-01-01", "2019-01-01", "2022-01-01"),
        ("2011-01-01", "2022-01-01", "2022-01-01", "2027-12-31"),
    ]

    results = []
    for is_start, is_end, os_start, os_end in splits:
        row = {}
        for label, s, e in [("IS", is_start, is_end), ("OS", os_start, os_end)]:
            mask = (tqqq.index >= s) & (tqqq.index < e)
            if mask.sum() < 50:
                row[label] = {"Sharpe": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sortino": 0.0}
                continue
            cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
            c = cum.loc[mask]
            if len(c) < 2:
                row[label] = {"Sharpe": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sortino": 0.0}
                continue
            c = c / c.iloc[0]
            rf_sub = rf.loc[mask]
            row[label] = calc_metrics(c, rf_sub)

        is_sharpe = row["IS"]["Sharpe"]
        os_sharpe = row["OS"]["Sharpe"]
        ratio = os_sharpe / is_sharpe if is_sharpe > 0 else 0.0

        period_is = f"{is_start[:4]}~{is_end[:4]}"
        period_os = f"{os_start[:4]}~{os_end[:4]}"
        results.append((period_is, period_os, is_sharpe, os_sharpe, ratio))

    return results


def print_walk_forward(label, wf_results):
    """Walk-forward 결과 출력."""
    print(f"\n  Walk-Forward 검증: {label}")
    header = f"    {'IS 기간':>12s} {'OS 기간':>12s} {'IS Sharpe':>10s} {'OS Sharpe':>10s} {'OS/IS':>8s}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    avg_ratio = 0.0
    for period_is, period_os, is_s, os_s, ratio in wf_results:
        flag = " ⚠" if ratio < 0.5 else ""
        print(f"    {period_is:>12s} {period_os:>12s} {is_s:>10.2f} {os_s:>10.2f} {ratio:>7.0%}{flag}")
        avg_ratio += ratio
    avg_ratio /= max(len(wf_results), 1)
    print(f"    {'평균 OS/IS ratio':>37s}: {avg_ratio:.0%}")
    return avg_ratio


# ══════════════════════════════════════════════════════════════
# [7] 파라미터 → BasicParams 복원 헬퍼
# ══════════════════════════════════════════════════════════════

def key_to_params_2stage(key):
    """2-stage grid key → BasicParams."""
    s1_enter = key["s1_enter"]
    s1_exit = key["s1_exit"]
    gap = key["gap"]
    return replace(
        BasicParams(spy_bear_cap=0.0),
        overheat1_enter=float(s1_enter),
        overheat1_exit=float(s1_exit),
        overheat2_enter=float(s1_enter + gap),
        overheat2_exit=float(s1_exit + 5),
        overheat3_enter=999.0,
        overheat3_exit=900.0,
        overheat4_enter=999.0,
        overheat4_exit=900.0,
    )


def key_to_params_4stage(key):
    """4-stage grid key → BasicParams."""
    be = key["base_enter"]
    sp = key["spread"]
    sg = key["s4_gap"]
    bx = key["base_exit"]
    ed = key["exit_decay"]
    return replace(
        BasicParams(spy_bear_cap=0.0),
        overheat1_enter=float(be),
        overheat1_exit=float(bx),
        overheat2_enter=float(be + sp),
        overheat2_exit=float(bx + ed),
        overheat3_enter=float(be + 2 * sp),
        overheat3_exit=float(bx + 2 * ed),
        overheat4_enter=float(be + 3 * sp + sg),
        overheat4_exit=float(bx - 15),
    )


def params_summary(params):
    """과열 파라미터 한 줄 요약."""
    stages = []
    for i, (e, x) in enumerate([
        (params.overheat1_enter, params.overheat1_exit),
        (params.overheat2_enter, params.overheat2_exit),
        (params.overheat3_enter, params.overheat3_exit),
        (params.overheat4_enter, params.overheat4_exit),
    ], 1):
        if e < 900:
            stages.append(f"S{i}:{e:.0f}/{x:.0f}")
    return " | ".join(stages) if stages else "없음"


# ══════════════════════════════════════════════════════════════
# [8] 시각화
# ══════════════════════════════════════════════════════════════

def plot_results(candidates, datasets, data_real, fname):
    """Equity curve 비교 + plateau heatmap."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1],
                              gridspec_kw={"hspace": 0.15})

    # Panel 1: Equity curves (실제 TQQQ 기준)
    ax1 = axes[0]
    tqqq = data_real["tqqq"]
    qqq = data_real["qqq"]
    spy = data_real["spy"]
    rf = data_real["rf"]
    start_mask = tqqq.index >= COMPARE_START

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0",
              "#00BCD4", "#E91E63", "#8BC34A"]

    # 원본
    orig_params = BasicParams(spy_bear_cap=0.0)
    cum_orig, _ = backtest_kimjje(tqqq, qqq, spy, rf, params=orig_params)
    c_orig = cum_orig.loc[start_mask]
    c_orig = c_orig / c_orig.iloc[0]
    ax1.plot(c_orig.index, c_orig.values, label="원본 (139/146/149/151)",
             color="#333333", linewidth=2.5, linestyle="--")

    # 과열 없음
    no_oh = replace(BasicParams(spy_bear_cap=0.0), use_overheat_split=False)
    cum_no, _ = backtest_kimjje(tqqq, qqq, spy, rf, params=no_oh)
    c_no = cum_no.loc[start_mask]
    c_no = c_no / c_no.iloc[0]
    ax1.plot(c_no.index, c_no.values, label="과열 없음",
             color="#9E9E9E", linewidth=1.2, linestyle=":", alpha=0.7)

    # 후보들
    for i, (label, params, _) in enumerate(candidates):
        cum, _ = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
        c = cum.loc[start_mask]
        c = c / c.iloc[0]
        ax1.plot(c.index, c.values, label=label, color=colors[i % len(colors)],
                 linewidth=1.8)

    ax1.set_yscale("log")
    ax1.set_ylabel("누적 수익률 (log)")
    ax1.set_title("김째 S2 과열 파라미터 최적화 결과 (실제 TQQQ, 2011~)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Panel 2: 후보별 multi-period Sortino 비교
    ax2 = axes[1]
    ds_names = list(datasets.keys())
    x_pos = np.arange(len(candidates) + 1)
    width = 0.8 / len(ds_names)

    # 원본 metrics
    all_labels = ["원본"] + [c[0] for c in candidates]
    all_params = [orig_params] + [c[1] for c in candidates]

    for j, ds_name in enumerate(ds_names):
        tq, qq, sp, r, start = datasets[ds_name]
        sortinos = []
        for params in all_params:
            cum, _ = backtest_kimjje(tq, qq, sp, r, params=params)
            mask = cum.index >= start
            c = cum.loc[mask]
            if len(c) < 50:
                sortinos.append(0)
                continue
            c = c / c.iloc[0]
            m = calc_metrics(c, r.loc[mask])
            sortinos.append(m["Sortino"])

        ax2.bar(x_pos + j * width, sortinos, width, label=ds_name, alpha=0.8)

    ax2.set_xticks(x_pos + width * (len(ds_names) - 1) / 2)
    ax2.set_xticklabels(all_labels, fontsize=8, rotation=15)
    ax2.set_ylabel("Sortino")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.set_title("Multi-Period Sortino 비교", fontsize=11)

    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  차트 저장: {path}")


# ══════════════════════════════════════════════════════════════
# [9] Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  김째 S2 과열 파라미터 최적화                              ║")
    print("║  Multi-Period Grid Search + Plateau Detection              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    t_total = time.time()

    # ── [1] 데이터 준비 (3 datasets) ──
    print("\n[1/7] 데이터 준비")
    print("=" * 60)

    data_idx = download_all_data(mode="index")   # NDX 40yr
    data_synth = download_all_data(mode="synth")  # QQQ 27yr
    data_real = download_all_data(mode="real")    # 실제 TQQQ 15yr

    datasets = {
        "NDX 40yr": (data_idx["tqqq"], data_idx["qqq"], data_idx["spy"],
                     data_idx["rf"], "1986-01-01"),
        "QQQ 27yr": (data_synth["tqqq"], data_synth["qqq"], data_synth["spy"],
                     data_synth["rf"], "2000-01-01"),
        "TQQQ 15yr": (data_real["tqqq"], data_real["qqq"], data_real["spy"],
                      data_real["rf"], COMPARE_START),
    }

    for name, (tq, qq, sp, r, start) in datasets.items():
        mask = tq.index >= start
        print(f"  {name}: {mask.sum()} trading days ({start}~)")

    # ── 원본 baseline ──
    print("\n  원본 파라미터 baseline:")
    orig_params = BasicParams(spy_bear_cap=0.0)
    orig_sort, orig_details = eval_across_periods(orig_params, datasets)
    print(f"    min Sortino = {orig_sort:.3f}")
    for dname, m in orig_details.items():
        print(f"    {dname}: CAGR={m['CAGR']:.1%}  Sharpe={m['Sharpe']:.2f}  "
              f"Sortino={m['Sortino']:.2f}  MDD={m['MDD']:.1%}")

    # ── [2] Model A: 2-Stage Optuna ──
    print("\n[2/7] Model A: 2-Stage Optuna Search (TPE, ~500 trials)")
    print("=" * 60)

    def objective_2s(trial):
        s1_enter = trial.suggest_int("s1_enter", 130, 160, step=2)
        s1_exit = trial.suggest_int("s1_exit", 115, 140, step=5)
        gap = trial.suggest_int("gap", 5, 20, step=5)

        if s1_exit >= s1_enter:
            return 0.0

        params = key_to_params_2stage({
            "s1_enter": s1_enter, "s1_exit": s1_exit, "gap": gap
        })
        min_sort, _ = eval_across_periods(params, datasets)
        return min_sort

    sampler_2s = TPESampler(seed=42, multivariate=True)
    study_2s = optuna.create_study(direction="maximize", sampler=sampler_2s)
    study_2s.optimize(objective_2s, n_trials=500, show_progress_bar=True)

    print(f"  Best min_sortino (2S): {study_2s.best_value:.3f}")
    best_trial_2s = study_2s.best_trial
    plateaus_2s = [(best_trial_2s.params, study_2s.best_value, study_2s.best_value)]

    # ── [3] Model B: 4-Stage Optuna ──
    print("\n[3/7] Model B: 4-Stage Optuna Search (TPE, ~1000 trials)")
    print("=" * 60)

    def objective_4s(trial):
        base_enter = trial.suggest_int("base_enter", 130, 160, step=2)
        spread = trial.suggest_int("spread", 2, 10, step=2)
        s4_gap = trial.suggest_int("s4_gap", 0, 6, step=2)
        base_exit = trial.suggest_int("base_exit", 115, 140, step=5)
        exit_decay = trial.suggest_int("exit_decay", 3, 8)

        s1_enter = float(base_enter)
        s2_enter = float(base_enter + spread)
        s3_enter = float(base_enter + 2 * spread)
        s4_enter = float(base_enter + 3 * spread + s4_gap)

        s1_exit = float(base_exit)
        s2_exit = float(base_exit + exit_decay)
        s3_exit = float(base_exit + 2 * exit_decay)
        s4_exit = float(base_exit - 15)

        if s1_exit >= s1_enter or s2_exit >= s2_enter or \
           s3_exit >= s3_enter or s4_exit >= s4_enter:
            return 0.0

        params = replace(
            BasicParams(spy_bear_cap=0.0),
            overheat1_enter=s1_enter, overheat1_exit=s1_exit,
            overheat2_enter=s2_enter, overheat2_exit=s2_exit,
            overheat3_enter=s3_enter, overheat3_exit=s3_exit,
            overheat4_enter=s4_enter, overheat4_exit=s4_exit,
        )
        min_sort, _ = eval_across_periods(params, datasets)
        return min_sort

    sampler_4s = TPESampler(seed=42, multivariate=True)
    study_4s = optuna.create_study(direction="maximize", sampler=sampler_4s)
    study_4s.optimize(objective_4s, n_trials=1000, show_progress_bar=True)

    print(f"  Best min_sortino (4S): {study_4s.best_value:.3f}")
    best_trial_4s = study_4s.best_trial
    plateaus_4s = [(best_trial_4s.params, study_4s.best_value, study_4s.best_value)]

    # Dummy df for later compatibility
    df_2s = pd.DataFrame([{"min_sortino": study_2s.best_value}])
    df_4s = pd.DataFrame([{"min_sortino": study_4s.best_value}])

    # ── [4] Finalize Best Candidates (Optuna already optimized) ──
    print("\n[4/7] Extracting Best Parameters from Optuna")
    print("=" * 60)

    best_candidates = []  # (label, params, min_sortino)

    # Model A best
    best_2s_params = key_to_params_2stage(best_trial_2s.params)
    best_candidates.append((f"A1 (Optuna-2S): {params_summary(best_2s_params)}",
                           best_2s_params, study_2s.best_value))
    print(f"  Model A: min_sortino={study_2s.best_value:.3f}  {params_summary(best_2s_params)}")

    # Model B best
    best_4s_params = replace(
        BasicParams(spy_bear_cap=0.0),
        overheat1_enter=float(best_trial_4s.params["base_enter"]),
        overheat1_exit=float(best_trial_4s.params["base_exit"]),
        overheat2_enter=float(best_trial_4s.params["base_enter"] + best_trial_4s.params["spread"]),
        overheat2_exit=float(best_trial_4s.params["base_exit"] + best_trial_4s.params["exit_decay"]),
        overheat3_enter=float(best_trial_4s.params["base_enter"] + 2*best_trial_4s.params["spread"]),
        overheat3_exit=float(best_trial_4s.params["base_exit"] + 2*best_trial_4s.params["exit_decay"]),
        overheat4_enter=float(best_trial_4s.params["base_enter"] + 3*best_trial_4s.params["spread"] + best_trial_4s.params["s4_gap"]),
        overheat4_exit=float(best_trial_4s.params["base_exit"] - 15),
    )
    best_candidates.append((f"B1 (Optuna-4S): {params_summary(best_4s_params)}",
                           best_4s_params, study_4s.best_value))
    print(f"  Model B: min_sortino={study_4s.best_value:.3f}  {params_summary(best_4s_params)}")

    # Sort by min_sortino
    best_candidates.sort(key=lambda x: x[2], reverse=True)

    # ── [5] Walk-Forward 검증 ──
    print("\n[5/7] Walk-Forward 검증")
    print("=" * 60)

    tqqq_r, qqq_r, spy_r, rf_r = (data_real["tqqq"], data_real["qqq"],
                                    data_real["spy"], data_real["rf"])

    # 원본
    wf_orig = walk_forward_validate(orig_params, tqqq_r, qqq_r, spy_r, rf_r)
    orig_ratio = print_walk_forward("원본 (139/146/149/151)", wf_orig)

    # 후보들
    wf_ratios = []
    for label, params, ms in best_candidates:
        wf = walk_forward_validate(params, tqqq_r, qqq_r, spy_r, rf_r)
        ratio = print_walk_forward(label, wf)
        wf_ratios.append(ratio)

    # ── [6] 최종 추천 + 원본 비교 ──
    print("\n[6/7] 최종 추천 파라미터 + 원본 비교")
    print("=" * 60)

    header = (f"{'순위':>4s} {'모델':>35s} {'min Sort':>9s} {'WF ratio':>9s}"
              f" {'NDX Sort':>9s} {'QQQ Sort':>9s} {'TQQQ Sort':>9s}")
    print(f"\n{header}")
    print("-" * len(header))

    # 원본 행
    print(f"{'ref':>4s} {'원본 (139/146/149/151)':>35s} {orig_sort:>9.3f} {orig_ratio:>8.0%}", end="")
    for dname in datasets:
        print(f" {orig_details[dname]['Sortino']:>9.2f}", end="")
    print()

    # 후보 행
    for rank, ((label, params, ms), wf_r) in enumerate(
            zip(best_candidates, wf_ratios), 1):
        _, details = eval_across_periods(params, datasets)
        print(f"{rank:>4d} {label:>35s} {ms:>9.3f} {wf_r:>8.0%}", end="")
        for dname in datasets:
            print(f" {details[dname]['Sortino']:>9.2f}", end="")
        print()

    # 전체 성과 테이블
    print("\n  실제 TQQQ (2011~) 상세 비교:")
    detail_header = f"{'':>35s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} {'Trades/Y':>9s}"
    print(f"  {detail_header}")
    print("  " + "-" * len(detail_header))

    start_mask = tqqq_r.index >= COMPARE_START
    rf_t = rf_r.loc[start_mask]

    # 원본
    cum, w = backtest_kimjje(tqqq_r, qqq_r, spy_r, rf_r, params=orig_params)
    c = cum.loc[start_mask] / cum.loc[start_mask].iloc[0]
    m = calc_metrics(c, rf_t)
    trades = count_trades(w.loc[start_mask], is_binary=False)
    print(f"  {'원본 (139/146/149/151)':>35s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} "
          f"{m['Sortino']:>8.2f} {m['MDD']:>7.1%} {trades:>8.1f}")

    # 과열 없음
    cum, w = backtest_kimjje(tqqq_r, qqq_r, spy_r, rf_r, params=replace(orig_params, use_overheat_split=False))
    c = cum.loc[start_mask] / cum.loc[start_mask].iloc[0]
    m = calc_metrics(c, rf_t)
    trades = count_trades(w.loc[start_mask], is_binary=False)
    print(f"  {'과열 없음':>35s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} "
          f"{m['Sortino']:>8.2f} {m['MDD']:>7.1%} {trades:>8.1f}")

    # 후보들
    for label, params, ms in best_candidates:
        cum, w = backtest_kimjje(tqqq_r, qqq_r, spy_r, rf_r, params=params)
        c = cum.loc[start_mask] / cum.loc[start_mask].iloc[0]
        m = calc_metrics(c, rf_t)
        trades = count_trades(w.loc[start_mask], is_binary=False)
        print(f"  {label:>35s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} "
              f"{m['Sortino']:>8.2f} {m['MDD']:>7.1%} {trades:>8.1f}")

    # ── [7] 시각화 ──
    print("\n[7/7] 시각화")
    print("=" * 60)

    # Top 4 후보만 차트에 표시
    plot_candidates = best_candidates[:4]
    plot_results(plot_candidates, datasets, data_real, "kimjje_overheat_optim.png")

    # ── 최종 추천 파라미터 정리 ──
    final_results = []
    for rank, ((label, params, ms), wf_r) in enumerate(
            zip(best_candidates, wf_ratios), 1):
        _, details = eval_across_periods(params, datasets)
        final_results.append({
            "Rank": rank,
            "Label": label,
            "min_Sortino": ms,
            "WF_Ratio": wf_r,
            "NDX_Sortino": details["NDX 40yr"]["Sortino"],
            "QQQ_Sortino": details["QQQ 27yr"]["Sortino"],
            "TQQQ_Sortino": details["TQQQ 15yr"]["Sortino"],
            "overheat1_enter": params.overheat1_enter,
            "overheat1_exit": params.overheat1_exit,
            "overheat2_enter": params.overheat2_enter,
            "overheat2_exit": params.overheat2_exit,
            "overheat3_enter": params.overheat3_enter,
            "overheat3_exit": params.overheat3_exit,
            "overheat4_enter": params.overheat4_enter,
            "overheat4_exit": params.overheat4_exit,
        })

    # CSV 저장
    for name, df in [("2stage", df_2s), ("4stage", df_4s)]:
        csv_path = OUT_DIR / f"kimjje_overheat_{name}_grid.csv"
        df.to_csv(csv_path, index=False)
        print(f"  CSV 저장: {csv_path}")

    # 최종 추천 파라미터 저장
    df_final = pd.DataFrame(final_results)
    final_csv_path = OUT_DIR / "kimjje_overheat_final_recommendations.csv"
    df_final.to_csv(final_csv_path, index=False)
    print(f"  최종 추천 파라미터 저장: {final_csv_path}")

    elapsed_total = time.time() - t_total
    print(f"\n총 실행 시간: {elapsed_total:.0f}초 ({elapsed_total/60:.1f}분)")
    print("\n완료!")
