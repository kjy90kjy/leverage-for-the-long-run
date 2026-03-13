"""
【원본 김째 매매법 과최적화 검증】

기준: references/new strategy.txt (L250-264)
파라미터: overheat1: 139/132, overheat2: 146/138, overheat3: 149/140, overheat4: 151/118

검증 방법: Walk-Forward Test
- 훈련 기간 (IS): 1985-2018 (33년)
- 테스트 기간 (OOS): 2019-2025 (6년)

과최적화 판단:
- IS > OOS: 과최적화 우려
- IS ≈ OOS: 안정적 (과최적화 아님)
- IS < OOS: 견고함 (과최적화 아님)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrs_standalone import (
    BasicParams, download_all_data, backtest_kimjje, calc_metrics,
    count_trades,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# 현재 타임스탐프
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TIMESTAMP_FILENAME = datetime.now().strftime("%Y%m%d_%H%M%S")


# ══════════════════════════════════════════════════════════════
# 원본 파라미터 (references/new strategy.txt L250-264)
# ══════════════════════════════════════════════════════════════

ORIGINAL_PARAMS = replace(
    BasicParams(spy_bear_cap=0.0),
    overheat1_enter=139.0,
    overheat1_exit=132.0,
    overheat2_enter=146.0,
    overheat2_exit=138.0,
    overheat3_enter=149.0,
    overheat3_exit=140.0,
    overheat4_enter=151.0,
    overheat4_exit=118.0,
)


def eval_period(params, datasets, period_name, period_start):
    """특정 기간의 성과 평가"""
    results = {}

    for data_name, (tqqq, qqq, spy, rf, _) in datasets.items():
        # 기간 필터링
        mask = tqqq.index >= period_start
        if mask.sum() < 50:
            results[data_name] = {
                "Days": 0,
                "Sortino": np.nan,
                "CAGR": np.nan,
                "Sharpe": np.nan,
                "MDD": np.nan,
                "Trades/Year": np.nan,
            }
            continue

        # 데이터 필터링
        tqqq_p = tqqq[mask]
        qqq_p = qqq[mask]
        spy_p = spy[mask]
        rf_p = rf[mask]

        # 백테스트
        cum, w = backtest_kimjje(tqqq_p, qqq_p, spy_p, rf_p, params=params)

        if len(cum) < 50:
            results[data_name] = {
                "Days": len(cum),
                "Sortino": np.nan,
                "CAGR": np.nan,
                "Sharpe": np.nan,
                "MDD": np.nan,
                "Trades/Year": np.nan,
            }
            continue

        # 정규화
        cum = cum / cum.iloc[0]

        # 메트릭
        m = calc_metrics(cum, rf_p)
        trades_per_year = count_trades(w) / (len(w) / 252)

        results[data_name] = {
            "Days": len(cum),
            "Sortino": m.get("Sortino", np.nan),
            "CAGR": m.get("CAGR", np.nan),
            "Sharpe": m.get("Sharpe", np.nan),
            "MDD": m.get("MDD", np.nan),
            "Trades/Year": trades_per_year,
        }

    return results


def main():
    print("=" * 100)
    print("【원본 김째 매매법 과최적화 검증】")
    print(f"검증 시각: {TIMESTAMP}")
    print("=" * 100)

    # 데이터 다운로드
    print("\n📊 데이터 다운로드 중...")
    data_idx = download_all_data(mode="index")   # NDX 40yr (1985~)
    data_synth = download_all_data(mode="synth")  # QQQ 27yr (1999~)
    data_real = download_all_data(mode="real")    # 실제 TQQQ 15yr (2010~)

    datasets = {
        "NDX_40yr": (data_idx["tqqq"], data_idx["qqq"], data_idx["spy"], data_idx["rf"], "1986-01-01"),
        "QQQ_27yr": (data_synth["tqqq"], data_synth["qqq"], data_synth["spy"], data_synth["rf"], "2000-01-01"),
        "TQQQ_15yr": (data_real["tqqq"], data_real["qqq"], data_real["spy"], data_real["rf"], "2010-02-11"),
    }

    # Walk-Forward Test
    print("\n【Walk-Forward Validation】")
    print("─" * 100)
    print(f"훈련 기간 (In-Sample):  1985-2018 (33년)")
    print(f"테스트 기간 (Out-of-Sample): 2019-2025 (6년)")
    print("─" * 100)

    is_results = eval_period(ORIGINAL_PARAMS, datasets, "IS", "1985-01-01")
    oos_results = eval_period(ORIGINAL_PARAMS, datasets, "OOS", "2019-01-01")

    # 결과 출력
    print("\n【훈련 기간 (In-Sample) 성과】")
    print(f"{'지표':<15} {'NDX':<20} {'QQQ':<20} {'TQQQ':<20}")
    print("─" * 75)

    for key in ["Sortino", "CAGR", "Sharpe", "MDD", "Trades/Year"]:
        values = {}
        for data_name in ["NDX_40yr", "QQQ_27yr", "TQQQ_15yr"]:
            v = is_results.get(data_name, {}).get(key)
            if key == "MDD":
                values[data_name] = f"{v*100:.1f}%" if not np.isnan(v) else "N/A"
            elif key in ["CAGR"]:
                values[data_name] = f"{v*100:.1f}%" if not np.isnan(v) else "N/A"
            elif key == "Trades/Year":
                values[data_name] = f"{v:.2f}" if not np.isnan(v) else "N/A"
            else:
                values[data_name] = f"{v:.3f}" if not np.isnan(v) else "N/A"

        print(f"{key:<15} {values['NDX_40yr']:<20} {values['QQQ_27yr']:<20} {values['TQQQ_15yr']:<20}")

    print("\n【테스트 기간 (Out-of-Sample) 성과】")
    print(f"{'지표':<15} {'NDX':<20} {'QQQ':<20} {'TQQQ':<20}")
    print("─" * 75)

    for key in ["Sortino", "CAGR", "Sharpe", "MDD", "Trades/Year"]:
        values = {}
        for data_name in ["NDX_40yr", "QQQ_27yr", "TQQQ_15yr"]:
            v = oos_results.get(data_name, {}).get(key)
            if key == "MDD":
                values[data_name] = f"{v*100:.1f}%" if not np.isnan(v) else "N/A"
            elif key in ["CAGR"]:
                values[data_name] = f"{v*100:.1f}%" if not np.isnan(v) else "N/A"
            elif key == "Trades/Year":
                values[data_name] = f"{v:.2f}" if not np.isnan(v) else "N/A"
            else:
                values[data_name] = f"{v:.3f}" if not np.isnan(v) else "N/A"

        print(f"{key:<15} {values['NDX_40yr']:<20} {values['QQQ_27yr']:<20} {values['TQQQ_15yr']:<20}")

    # 과최적화 진단
    print("\n\n【과최적화 진단】")
    print("─" * 100)

    diagnosis = []
    for data_name in ["NDX_40yr", "QQQ_27yr", "TQQQ_15yr"]:
        is_sort = is_results[data_name].get("Sortino")
        oos_sort = oos_results[data_name].get("Sortino")

        if np.isnan(is_sort) or np.isnan(oos_sort):
            status = "데이터 부족"
            ratio = np.nan
        else:
            ratio = oos_sort / is_sort if is_sort > 0 else np.nan
            if ratio < 0.7:
                status = "🔴 심각한 과최적화"
            elif ratio < 0.85:
                status = "🟡 중간 과최적화"
            elif ratio < 1.0:
                status = "🟢 경미한 과적합"
            else:
                status = "✅ 과최적화 없음 (OOS > IS)"

        diagnosis.append({
            "Dataset": data_name,
            "IS Sortino": is_sort,
            "OOS Sortino": oos_sort,
            "OOS/IS Ratio": ratio,
            "Status": status,
        })
        print(f"{data_name:<15} IS: {is_sort:.3f}  OOS: {oos_sort:.3f}  (OOS/IS: {ratio:.2f})  {status}")

    # 최악값 (Worst-Case Sortino)
    print("\n【최악값 (Worst-Case) 비교】")
    print("─" * 100)
    is_worst = min([r["IS Sortino"] for r in diagnosis if not np.isnan(r["IS Sortino"])])
    oos_worst = min([r["OOS Sortino"] for r in diagnosis if not np.isnan(r["OOS Sortino"])])
    worst_ratio = oos_worst / is_worst if is_worst > 0 else np.nan

    print(f"In-Sample (Worst):      {is_worst:.3f}")
    print(f"Out-of-Sample (Worst):  {oos_worst:.3f}")
    print(f"OOS/IS Ratio:           {worst_ratio:.2f}")

    if worst_ratio < 0.7:
        final_verdict = "🔴 심각한 과최적화 감지 — 재검토 필요"
    elif worst_ratio < 0.85:
        final_verdict = "🟡 중간 수준 과최적화 — 주의 필요"
    elif worst_ratio < 1.0:
        final_verdict = "🟢 경미한 과적합 — 실전 가능 (신중함 권고)"
    else:
        final_verdict = "✅ 과최적화 없음 — 안정적 파라미터"

    print(f"\n최종 판정: {final_verdict}")

    # CSV 저장
    print(f"\n\n📁 결과를 CSV로 저장 중...")
    df_diagnosis = pd.DataFrame(diagnosis)
    csv_path = OUT_DIR / f"original_strategy_validation_{TIMESTAMP_FILENAME}.csv"
    df_diagnosis.to_csv(csv_path, index=False)
    print(f"✅ 저장됨: {csv_path}")

    # 마크다운 보고서 저장
    print(f"📄 상세 보고서를 마크다운으로 저장 중...")

    md_content = f"""# 원본 김째 매매법 과최적화 검증 보고서

**검증 시각**: {TIMESTAMP}

## 파라미터

| 단계 | 진입 | 진출 | 감량 |
|------|------|------|------|
| **S1** | 139% | 132% | 10% |
| **S2** | 146% | 138% | 20% |
| **S3** | 149% | 140% | 95% |
| **S4** | 151% | 118% | 100% |

**출처**: `references/new strategy.txt` (L250-264, L63-67)

---

## Walk-Forward Validation

### 훈련 기간 (In-Sample): 1985-2018 (33년)

| 지표 | NDX (40yr) | QQQ (27yr) | TQQQ (15yr) |
|------|-----------|-----------|-----------|
| **Sortino** | {is_results['NDX_40yr']['Sortino']:.3f} | {is_results['QQQ_27yr']['Sortino']:.3f} | {is_results['TQQQ_15yr']['Sortino']:.3f} |
| **CAGR** | {is_results['NDX_40yr']['CAGR']*100:.1f}% | {is_results['QQQ_27yr']['CAGR']*100:.1f}% | {is_results['TQQQ_15yr']['CAGR']*100:.1f}% |
| **Sharpe** | {is_results['NDX_40yr']['Sharpe']:.3f} | {is_results['QQQ_27yr']['Sharpe']:.3f} | {is_results['TQQQ_15yr']['Sharpe']:.3f} |
| **MDD** | {is_results['NDX_40yr']['MDD']*100:.1f}% | {is_results['QQQ_27yr']['MDD']*100:.1f}% | {is_results['TQQQ_15yr']['MDD']*100:.1f}% |
| **연간매매** | {is_results['NDX_40yr']['Trades/Year']:.2f}회 | {is_results['QQQ_27yr']['Trades/Year']:.2f}회 | {is_results['TQQQ_15yr']['Trades/Year']:.2f}회 |

### 테스트 기간 (Out-of-Sample): 2019-2025 (6년)

| 지표 | NDX (40yr) | QQQ (27yr) | TQQQ (15yr) |
|------|-----------|-----------|-----------|
| **Sortino** | {oos_results['NDX_40yr']['Sortino']:.3f} | {oos_results['QQQ_27yr']['Sortino']:.3f} | {oos_results['TQQQ_15yr']['Sortino']:.3f} |
| **CAGR** | {oos_results['NDX_40yr']['CAGR']*100:.1f}% | {oos_results['QQQ_27yr']['CAGR']*100:.1f}% | {oos_results['TQQQ_15yr']['CAGR']*100:.1f}% |
| **Sharpe** | {oos_results['NDX_40yr']['Sharpe']:.3f} | {oos_results['QQQ_27yr']['Sharpe']:.3f} | {oos_results['TQQQ_15yr']['Sharpe']:.3f} |
| **MDD** | {oos_results['NDX_40yr']['MDD']*100:.1f}% | {oos_results['QQQ_27yr']['MDD']*100:.1f}% | {oos_results['TQQQ_15yr']['MDD']*100:.1f}% |
| **연간매매** | {oos_results['NDX_40yr']['Trades/Year']:.2f}회 | {oos_results['QQQ_27yr']['Trades/Year']:.2f}회 | {oos_results['TQQQ_15yr']['Trades/Year']:.2f}회 |

---

## 과최적화 진단

### 개별 지표별 분석

"""

    for row in diagnosis:
        md_content += f"""
#### {row['Dataset']}

| 메트릭 | 값 |
|--------|-----|
| IS Sortino | {row['IS Sortino']:.3f} |
| OOS Sortino | {row['OOS Sortino']:.3f} |
| OOS/IS 비율 | {row['OOS/IS Ratio']:.2f} |
| **진단** | **{row['Status']}** |

"""

    md_content += f"""

### 최악값 (Worst-Case Sortino) 비교

| 구분 | 값 |
|------|-----|
| In-Sample | {is_worst:.3f} |
| Out-of-Sample | {oos_worst:.3f} |
| **OOS/IS 비율** | **{worst_ratio:.2f}** |

---

## 최종 판정

### {final_verdict}

**해석:**
- **OOS/IS 비율 < 0.70**: 파라미터가 훈련 데이터에 과도하게 최적화됨
- **OOS/IS 비율 0.70~0.85**: 중간 수준의 과적합 존재
- **OOS/IS 비율 0.85~1.00**: 경미한 과적합 (일반적)
- **OOS/IS 비율 ≥ 1.00**: 과최적화 없음 (오히려 안정적)

현재 비율 {worst_ratio:.2f}는 **{('과최적화됨' if worst_ratio < 1.0 else '안정적')}**을 의미합니다.

---

## 권장사항

1. **실전 배포 전에**: 이 파라미터로 2026년 이후 추가 OOS 테스트 필요
2. **정기 재검증**: 분기별 walk-forward 테스트로 모니터링
3. **파라미터 유지**: 현재 파라미터는 {('재검토' if worst_ratio < 0.85 else '유지')} 권장

---

**생성일**: {TIMESTAMP}
"""

    md_path = DOCS_DIR / f"ORIGINAL_STRATEGY_VALIDATION_{TIMESTAMP_FILENAME}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅ 저장됨: {md_path}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
