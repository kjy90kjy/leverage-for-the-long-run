"""
【원본 파라미터 안정성 & 고원지대 검증】

목표: 원본 파라미터(139/132, 146/138, 149/140, 151/118)가
      "뾰족한 피크"인지 "안정적 고원지대"인지 판단

방법: 각 Stage별로 enter/exit 쌍을 ±2% 범위에서 테스트
      3개 지표 Worst-Case Sortino로 평가
      Surface Plot으로 시각화
"""

import sys
import io
import warnings
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrs_standalone import (
    BasicParams, download_all_data, backtest_kimjje, calc_metrics,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TIMESTAMP_FILENAME = datetime.now().strftime("%Y%m%d_%H%M%S")

# 원본 파라미터
ORIGINAL = {
    "S1": {"enter": 139.0, "exit": 132.0},
    "S2": {"enter": 146.0, "exit": 138.0},
    "S3": {"enter": 149.0, "exit": 140.0},
    "S4": {"enter": 151.0, "exit": 118.0},
}


def eval_params(s1e, s1x, s2e, s2x, s3e, s3x, s4e, s4x, datasets):
    """파라미터 조합 평가"""
    params = replace(
        BasicParams(spy_bear_cap=0.0),
        overheat1_enter=float(s1e),
        overheat1_exit=float(s1x),
        overheat2_enter=float(s2e),
        overheat2_exit=float(s2x),
        overheat3_enter=float(s3e),
        overheat3_exit=float(s3x),
        overheat4_enter=float(s4e),
        overheat4_exit=float(s4x),
    )

    sortinos = []
    for data_name, (tqqq, qqq, spy, rf, start) in datasets.items():
        mask = tqqq.index >= start
        if mask.sum() < 50:
            sortinos.append(0.0)
            continue

        tqqq_p = tqqq[mask]
        qqq_p = qqq[mask]
        spy_p = spy[mask]
        rf_p = rf[mask]

        cum, w = backtest_kimjje(tqqq_p, qqq_p, spy_p, rf_p, params=params)
        if len(cum) < 50:
            sortinos.append(0.0)
            continue

        cum = cum / cum.iloc[0]
        m = calc_metrics(cum, rf_p)
        sortinos.append(m.get("Sortino", 0.0))

    return min(sortinos) if sortinos else 0.0


def test_stage(stage_name, s_key, enter_range, exit_range, datasets):
    """한 Stage의 enter/exit 쌍을 테스트"""
    print(f"\n  {stage_name} 테스트 중...")

    results = []
    for enter in enter_range:
        for exit_val in exit_range:
            if exit_val >= enter:
                continue

            # 현재 Stage 변수
            if stage_name == "S1":
                s1e, s1x, s2e, s2x, s3e, s3x, s4e, s4x = (
                    enter, exit_val,
                    ORIGINAL["S2"]["enter"], ORIGINAL["S2"]["exit"],
                    ORIGINAL["S3"]["enter"], ORIGINAL["S3"]["exit"],
                    ORIGINAL["S4"]["enter"], ORIGINAL["S4"]["exit"],
                )
            elif stage_name == "S2":
                s1e, s1x, s2e, s2x, s3e, s3x, s4e, s4x = (
                    ORIGINAL["S1"]["enter"], ORIGINAL["S1"]["exit"],
                    enter, exit_val,
                    ORIGINAL["S3"]["enter"], ORIGINAL["S3"]["exit"],
                    ORIGINAL["S4"]["enter"], ORIGINAL["S4"]["exit"],
                )
            elif stage_name == "S3":
                s1e, s1x, s2e, s2x, s3e, s3x, s4e, s4x = (
                    ORIGINAL["S1"]["enter"], ORIGINAL["S1"]["exit"],
                    ORIGINAL["S2"]["enter"], ORIGINAL["S2"]["exit"],
                    enter, exit_val,
                    ORIGINAL["S4"]["enter"], ORIGINAL["S4"]["exit"],
                )
            else:  # S4
                s1e, s1x, s2e, s2x, s3e, s3x, s4e, s4x = (
                    ORIGINAL["S1"]["enter"], ORIGINAL["S1"]["exit"],
                    ORIGINAL["S2"]["enter"], ORIGINAL["S2"]["exit"],
                    ORIGINAL["S3"]["enter"], ORIGINAL["S3"]["exit"],
                    enter, exit_val,
                )

            worst_sort = eval_params(s1e, s1x, s2e, s2x, s3e, s3x, s4e, s4x, datasets)

            results.append({
                "Enter": enter,
                "Exit": exit_val,
                "Worst_Sortino": worst_sort,
                "Gap": enter - exit_val,
            })

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("【원본 파라미터 안정성 & 고원지대 검증】")
    print(f"검증 시각: {TIMESTAMP}")
    print("=" * 100)

    # 데이터
    print("\n📊 데이터 다운로드 중...")
    data_idx = download_all_data(mode="index")
    data_synth = download_all_data(mode="synth")
    data_real = download_all_data(mode="real")

    datasets = {
        "NDX_40yr": (data_idx["tqqq"], data_idx["qqq"], data_idx["spy"], data_idx["rf"], "1986-01-01"),
        "QQQ_27yr": (data_synth["tqqq"], data_synth["qqq"], data_synth["spy"], data_synth["rf"], "2000-01-01"),
        "TQQQ_15yr": (data_real["tqqq"], data_real["qqq"], data_real["spy"], data_real["rf"], "2010-02-11"),
    }

    # 각 Stage 테스트
    all_results = {}

    print("\n【Stage별 고원지대 분석】")

    for stage_name in ["S1", "S2", "S3", "S4"]:
        # 범위 설정 (±2)
        orig = ORIGINAL[stage_name]
        enter_range = np.arange(orig["enter"] - 2, orig["enter"] + 3, 1)
        exit_range = np.arange(orig["exit"] - 2, orig["exit"] + 3, 1)

        # 테스트
        df_results = test_stage(stage_name, stage_name, enter_range, exit_range, datasets)
        all_results[stage_name] = df_results

        # 결과 출력
        print(f"\n{stage_name} 상위 5개:")
        top5 = df_results.nlargest(5, "Worst_Sortino")
        print(top5[["Enter", "Exit", "Worst_Sortino", "Gap"]].to_string(index=False))

        # 원본
        orig_row = df_results[
            (df_results["Enter"] == orig["enter"]) & (df_results["Exit"] == orig["exit"])
        ]
        if not orig_row.empty:
            orig_sortino = orig_row.iloc[0]["Worst_Sortino"]
            print(f"\n  【{stage_name} 원본】")
            print(f"    파라미터: Enter {orig['enter']:.0f}%, Exit {orig['exit']:.0f}%")
            print(f"    Worst-Case Sortino: {orig_sortino:.4f}")

            # 고원지대 찾기 (원본 ±5%)
            plateau = df_results[
                (df_results["Worst_Sortino"] >= orig_sortino * 0.95) &
                (df_results["Worst_Sortino"] <= orig_sortino * 1.05)
            ]
            print(f"    안정성 영역 (±5%): {len(plateau)}개 조합")
            if len(plateau) > 0:
                print(f"      - 범위: Enter {plateau['Enter'].min():.0f}~{plateau['Enter'].max():.0f}, "
                      f"Exit {plateau['Exit'].min():.0f}~{plateau['Exit'].max():.0f}")

        # 히트맵
        fig, ax = plt.subplots(figsize=(10, 8))

        pivot = df_results.pivot_table(
            values="Worst_Sortino",
            index="Exit",
            columns="Enter",
            aggfunc="first"
        )

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", origin="lower", vmin=0, vmax=pivot.values.max())

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{int(x)}" for x in pivot.columns])
        ax.set_yticklabels([f"{int(x)}" for x in pivot.index])

        ax.set_xlabel(f"{stage_name} Enter (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{stage_name} Exit (%)", fontsize=12, fontweight="bold")
        ax.set_title(f"【{stage_name} 고원지대 검증】\n색이 밝을수록 좋은 성과 (원본: {orig['enter']:.0f}%/{orig['exit']:.0f}%)", fontsize=13, fontweight="bold")

        # 원본 표시
        if orig["enter"] in pivot.columns and orig["exit"] in pivot.index:
            orig_col = list(pivot.columns).index(orig["enter"])
            orig_row_idx = list(pivot.index).index(orig["exit"])
            ax.plot(orig_col, orig_row_idx, "b*", markersize=25, label="원본", markeredgecolor="white", markeredgewidth=2)
            ax.legend(loc="upper left", fontsize=11)

        plt.colorbar(im, ax=ax, label="Worst-Case Sortino")
        plt.tight_layout()
        plot_path = OUT_DIR / f"stability_{stage_name}_{TIMESTAMP_FILENAME}.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        print(f"  ✅ 히트맵 저장: stability_{stage_name}_{TIMESTAMP_FILENAME}.png")

    # 종합 리포트
    print("\n\n" + "=" * 100)
    print("【종합 평가】")
    print("=" * 100)

    md_report = f"""# 원본 파라미터 안정성 & 고원지대 검증

**검증 시각**: {TIMESTAMP}

## 원본 파라미터

| Stage | Enter | Exit | 격차 |
|-------|-------|------|------|
| **S1** | 139% | 132% | 7% |
| **S2** | 146% | 138% | 8% |
| **S3** | 149% | 140% | 9% |
| **S4** | 151% | 118% | 33% |

---

## 각 Stage 분석

"""

    for stage_name in ["S1", "S2", "S3", "S4"]:
        df = all_results[stage_name]
        orig = ORIGINAL[stage_name]
        orig_row = df[(df["Enter"] == orig["enter"]) & (df["Exit"] == orig["exit"])]

        if not orig_row.empty:
            orig_sortino = orig_row.iloc[0]["Worst_Sortino"]
            plateau_95 = df[df["Worst_Sortino"] >= orig_sortino * 0.95]
            is_plateau = len(plateau_95) >= 5

            md_report += f"""### {stage_name}

**원본**: Enter {orig['enter']:.0f}%, Exit {orig['exit']:.0f}%, Worst-Case Sortino: {orig_sortino:.4f}

**고원지대 분석**:
- 원본 수준 유지 (95% 이상): **{len(plateau_95)}개 조합**
- 최고 성과: {df['Worst_Sortino'].max():.4f}
- 평가: **{"🟢 안정적 고원지대" if is_plateau else "🔴 뾰족한 피크"}**

"""

    print("\n📄 상세 보고서 저장 중...")

    md_path = DOCS_DIR / f"STABILITY_PLATEAU_DETECTION_{TIMESTAMP_FILENAME}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"✅ 저장됨: {md_path}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
