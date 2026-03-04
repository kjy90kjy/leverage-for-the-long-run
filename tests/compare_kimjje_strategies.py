"""
김째 3대 전략 비교: Current Production vs Optuna-Optimized (4-Stage) vs Optuna-Optimized (2-Stage)

3개 지표(NDX 40yr, QQQ 27yr, TQQQ 15yr) × 3개 전략의 상세 성과 비교
"""

import sys
import io
import warnings

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
    count_trades, COMPARE_START,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 전략 정의
# ══════════════════════════════════════════════════════════════

STRATEGIES = {
    "Current_Production": replace(
        BasicParams(spy_bear_cap=0.0),
        overheat1_enter=139.0,
        overheat1_exit=132.0,
        overheat2_enter=146.0,
        overheat2_exit=138.0,
        overheat3_enter=149.0,
        overheat3_exit=140.0,
        overheat4_enter=151.0,
        overheat4_exit=118.0,
    ),
    "Optuna_4Stage_B1": replace(
        BasicParams(spy_bear_cap=0.0),
        overheat1_enter=134.0,
        overheat1_exit=115.0,
        overheat2_enter=138.0,
        overheat2_exit=123.0,
        overheat3_enter=142.0,
        overheat3_exit=131.0,
        overheat4_enter=152.0,
        overheat4_exit=100.0,
    ),
    "Optuna_2Stage_A1": replace(
        BasicParams(spy_bear_cap=0.0),
        overheat1_enter=142.0,
        overheat1_exit=115.0,
        overheat2_enter=152.0,
        overheat2_exit=120.0,
        overheat3_enter=999.0,
        overheat3_exit=900.0,
        overheat4_enter=999.0,
        overheat4_exit=900.0,
    ),
}


def format_params(params):
    """파라미터를 한눈에 보기 좋은 문자열로 포맷"""
    lines = []
    if params.overheat1_enter < 200:
        lines.append(f"  S1: {params.overheat1_enter:.0f}% → {params.overheat1_exit:.0f}%")
    if params.overheat2_enter < 200:
        lines.append(f"  S2: {params.overheat2_enter:.0f}% → {params.overheat2_exit:.0f}%")
    if params.overheat3_enter < 200:
        lines.append(f"  S3: {params.overheat3_enter:.0f}% → {params.overheat3_exit:.0f}%")
    if params.overheat4_enter < 200:
        lines.append(f"  S4: {params.overheat4_enter:.0f}% → {params.overheat4_exit:.0f}%")
    return "\n".join(lines)


def eval_strategy(strategy_name, params, datasets):
    """한 전략을 3개 지표에 대해 평가"""
    results = {}

    for data_name, (tqqq, qqq, spy, rf, start_date) in datasets.items():
        # 백테스트
        cum, w = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
        mask = cum.index >= start_date
        c = cum.loc[mask]

        if len(c) < 50:
            results[data_name] = {
                "Sortino": np.nan,
                "CAGR": np.nan,
                "Sharpe": np.nan,
                "MDD": np.nan,
                "Trades/Year": np.nan,
            }
            continue

        # 정규화
        c = c / c.iloc[0]
        rf_sub = rf.loc[mask]

        # 메트릭 계산
        m = calc_metrics(c, rf_sub)

        # 거래 횟수
        w_sub = w.loc[mask]
        trades_per_year = count_trades(w_sub) / (len(w_sub) / 252)

        results[data_name] = {
            "Sortino": m.get("Sortino", np.nan),
            "CAGR": m.get("CAGR", np.nan),
            "Sharpe": m.get("Sharpe", np.nan),
            "MDD": m.get("MDD", np.nan),
            "Trades/Year": trades_per_year,
        }

    return results


def main():
    print("=" * 100)
    print("【김째 매매 전략 비교: Current vs Optuna Optimized】")
    print("=" * 100)

    # 데이터 다운로드 (3개 모드)
    print("\n📊 데이터 다운로드 중...")
    data_idx = download_all_data(mode="index")   # NDX 40yr
    data_synth = download_all_data(mode="synth")  # QQQ 27yr
    data_real = download_all_data(mode="real")    # 실제 TQQQ 15yr

    datasets = {
        "NDX_40yr": (data_idx["tqqq"], data_idx["qqq"], data_idx["spy"], data_idx["rf"], "1986-01-01"),
        "QQQ_27yr": (data_synth["tqqq"], data_synth["qqq"], data_synth["spy"], data_synth["rf"], "2000-01-01"),
        "TQQQ_15yr": (data_real["tqqq"], data_real["qqq"], data_real["spy"], data_real["rf"], COMPARE_START),
    }

    # 전략별 평가
    all_results = {}
    for strategy_name, params in STRATEGIES.items():
        print(f"\n⏳ {strategy_name} 평가 중...")
        results = eval_strategy(strategy_name, params, datasets)
        all_results[strategy_name] = results

    # 결과 출력
    print("\n\n" + "=" * 100)
    print("【상세 결과】")
    print("=" * 100)

    for strategy_name, params in STRATEGIES.items():
        results = all_results[strategy_name]

        print(f"\n{'─' * 100}")
        print(f"【전략: {strategy_name}】")
        print(f"{'─' * 100}")

        # 파라미터
        print(f"\n파라미터 (Overheat 단계별 진입/진출 유지%):")
        print(format_params(params))

        # 매매 로직
        print(f"\n매매 로직:")
        print(f"  • TQQQ가 200MA 대비 {params.overheat1_enter:.0f}% 도달 → Stage 1 시작")
        print(f"    - 100% → 90% 감량 (10% 익절)")
        print(f"  • {params.overheat2_enter:.0f}% 도달 → Stage 2 시작")
        print(f"    - 100% → 80% 감량 (20% 익절)")
        print(f"  • {params.overheat3_enter:.0f}% 도달 → Stage 3 시작")
        print(f"    - 5% 유지 (95% 익절)")
        print(f"  • {params.overheat4_enter:.0f}% 도달 → Stage 4 시작")
        print(f"    - 0% (전량 익절)")

        # 성과
        print(f"\n백테스트 결과:")
        print(f"{'지표':<15} {'Sortino':>12} {'CAGR':>12} {'Sharpe':>12} {'MDD':>12} {'연간매매':>12}")
        print(f"{'-' * 75}")

        for data_name in ["NDX_40yr", "QQQ_27yr", "TQQQ_15yr"]:
            if data_name in results:
                m = results[data_name]
                sortino_str = f"{m['Sortino']:.3f}" if not np.isnan(m['Sortino']) else "N/A"
                cagr_str = f"{m['CAGR']*100:.1f}%" if not np.isnan(m['CAGR']) else "N/A"
                sharpe_str = f"{m['Sharpe']:.3f}" if not np.isnan(m['Sharpe']) else "N/A"
                mdd_str = f"{m['MDD']*100:.1f}%" if not np.isnan(m['MDD']) else "N/A"
                trades_str = f"{m['Trades/Year']:.2f}" if not np.isnan(m['Trades/Year']) else "N/A"

                print(f"{data_name:<15} {sortino_str:>12} {cagr_str:>12} {sharpe_str:>12} {mdd_str:>12} {trades_str:>12}")

    # 비교 요약
    print(f"\n\n{'═' * 100}")
    print("【전략 비교 요약 — Worst-Case (최악값)】")
    print(f"{'═' * 100}\n")

    # Worst-Case metrics (3개 지표 중 최악)
    print(f"{'지표':<25} {'Current':>20} {'4-Stage':>20} {'2-Stage':>20}")
    print(f"{'-' * 85}")

    for key in ["Sortino", "CAGR", "Sharpe", "MDD", "Trades/Year"]:
        values = {}
        for strategy_name in STRATEGIES.keys():
            metrics = [all_results[strategy_name].get(d, {}).get(key) for d in ["NDX_40yr", "QQQ_27yr", "TQQQ_15yr"]]
            if key == "MDD":
                # MDD는 최악값(가장 큰 음수)
                valid = [v for v in metrics if not np.isnan(v)]
                value = min(valid) if valid else np.nan
            else:
                # 나머지는 최악값(가장 작은 값)
                valid = [v for v in metrics if not np.isnan(v)]
                value = min(valid) if valid else np.nan
            values[strategy_name] = value

        if key == "MDD":
            format_str = lambda v: f"{v*100:.1f}%" if not np.isnan(v) else "N/A"
        elif key == "CAGR":
            format_str = lambda v: f"{v*100:.1f}%" if not np.isnan(v) else "N/A"
        elif key == "Trades/Year":
            format_str = lambda v: f"{v:.2f}" if not np.isnan(v) else "N/A"
        else:
            format_str = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"

        print(
            f"{key:<25} {format_str(values.get('Current_Production', np.nan)):>20} "
            f"{format_str(values.get('Optuna_4Stage_B1', np.nan)):>20} "
            f"{format_str(values.get('Optuna_2Stage_A1', np.nan)):>20}"
        )

    # CSV 저장
    print(f"\n\n📁 결과를 CSV로 저장 중...")

    records = []
    for strategy_name in STRATEGIES.keys():
        for data_name in ["NDX_40yr", "QQQ_27yr", "TQQQ_15yr"]:
            m = all_results[strategy_name].get(data_name, {})
            records.append({
                "Strategy": strategy_name,
                "Dataset": data_name,
                "Sortino": m.get("Sortino"),
                "CAGR": m.get("CAGR"),
                "Sharpe": m.get("Sharpe"),
                "MDD": m.get("MDD"),
                "Trades_Per_Year": m.get("Trades/Year"),
            })

    df = pd.DataFrame(records)
    csv_path = OUT_DIR / "kimjje_strategy_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ 저장됨: {csv_path}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
