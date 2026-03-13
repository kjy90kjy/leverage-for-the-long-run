"""
김째 매매법 과열 파라미터 고원지대 탐색 (Optuna)

목표:
1. NDX / QQQ / TQQQ 기간 각각 기준으로 독립 Optuna 최적화
2. 각 최적 조합이 고원지대에 위치하는지 확인
3. Cross-validation: 각 기간 최적 조합을 다른 기간에 적용
4. 원본과 비교

실행:
    python optimization/overheat_plateau_optuna.py
"""

import sys
import io
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import replace

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lrs_standalone import (
    BasicParams, download_all_data, backtest_kimjje, calc_metrics,
    COMPARE_START, OUT_DIR,
)

# ══════════════════════════════════════════════════════════════
# [1] 파라미터 공간 정의 & 제약 검증
# ══════════════════════════════════════════════════════════════

def validate_overheat_params(s1_e, s1_x, s2_e, s2_x, s3_e, s3_x, s4_e, s4_x):
    """8개 파라미터 제약 조건 검증."""
    return (
        s1_x < s1_e and
        s2_e > s1_e and
        s2_x < s2_e and
        s3_e > s2_e and
        s3_x < s3_e and
        s4_e > s3_e and
        s4_x < s4_e
    )


def params_from_trial(trial, seed=None):
    """Optuna trial → BasicParams (제약 조건 적용)."""
    s1_enter = trial.suggest_int("s1_enter", 125, 160)

    # S1 exit: < S1 enter 제약
    s1_exit = trial.suggest_int("s1_exit", 105, min(s1_enter - 1, 155))

    # S2 gap: S1_enter + gap_12
    gap_12 = trial.suggest_int("gap_12", 3, 25)
    s2_enter = s1_enter + gap_12

    # S2 exit: < S2 enter 제약
    s2_exit = trial.suggest_int("s2_exit", 100, min(s2_enter - 1, 150))

    # S3 gap: S2_enter + gap_23
    gap_23 = trial.suggest_int("gap_23", 2, 20)
    s3_enter = s2_enter + gap_23

    # S3 exit: < S3 enter 제약
    s3_exit = trial.suggest_int("s3_exit", 100, min(s3_enter - 1, 155))

    # S4 gap: S3_enter + gap_34
    gap_34 = trial.suggest_int("gap_34", 2, 20)
    s4_enter = s3_enter + gap_34

    # S4 exit: < S4 enter 제약
    s4_exit = trial.suggest_int("s4_exit", 80, min(s4_enter - 1, 145))

    # 모든 제약 검증
    if not validate_overheat_params(s1_enter, s1_exit, s2_enter, s2_exit,
                                    s3_enter, s3_exit, s4_enter, s4_exit):
        raise optuna.exceptions.TrialPruned()

    return BasicParams(
        overheat1_enter=s1_enter,
        overheat1_exit=s1_exit,
        overheat2_enter=s2_enter,
        overheat2_exit=s2_exit,
        overheat3_enter=s3_enter,
        overheat3_exit=s3_exit,
        overheat4_enter=s4_enter,
        overheat4_exit=s4_exit,
        spy_bear_cap=0.0,
    )


# ══════════════════════════════════════════════════════════════
# [2] 기간별 데이터 로드 & 백테스트
# ══════════════════════════════════════════════════════════════

def load_data_period(mode):
    """기간별 데이터 로드."""
    print(f"데이터 로드: {mode}...")
    data = download_all_data(mode=mode)
    qqq = data["qqq"]
    tqqq = data["tqqq"]
    spy = data["spy"]
    rf = data["rf"]

    # 기간 제한
    qqq = qqq.loc[qqq.index >= COMPARE_START]
    tqqq = tqqq.loc[tqqq.index >= COMPARE_START]
    spy = spy.loc[spy.index >= COMPARE_START]
    rf = rf.loc[rf.index >= COMPARE_START]

    common = qqq.index.intersection(tqqq.index).intersection(spy.index).intersection(rf.index)
    qqq = qqq.loc[common]
    tqqq = tqqq.loc[common]
    spy = spy.loc[common]
    rf = rf.loc[common]

    print(f"  기간: {common[0].date()} → {common[-1].date()} ({len(common)}일)")
    return qqq, tqqq, spy, rf


def backtest_and_score(qqq, tqqq, spy, rf, params):
    """백테스트 및 Sortino 점수 반환."""
    try:
        cum, _ = backtest_kimjje(tqqq, qqq, spy, rf, params=params)
        metrics = calc_metrics(cum, rf)
        sortino = metrics["Sortino"]
        return sortino, metrics
    except Exception as e:
        print(f"백테스트 실패: {e}")
        return 0.0, None


# ══════════════════════════════════════════════════════════════
# [3] Objective Functions (3개 기간)
# ══════════════════════════════════════════════════════════════

def create_objective(qqq, tqqq, spy, rf, period_name):
    """기간별 objective function 생성."""
    def objective(trial):
        try:
            params = params_from_trial(trial)
        except optuna.exceptions.TrialPruned:
            return 0.0

        sortino, _ = backtest_and_score(qqq, tqqq, spy, rf, params)
        return sortino

    return objective


# ══════════════════════════════════════════════════════════════
# [4] 고원지대 검증 (Optuna Study trials 기반)
# ══════════════════════════════════════════════════════════════

def check_plateau_from_trials(best_trial, study, best_sortino):
    """Optuna study의 trials에서 이웃 파라미터들을 찾아 안정성 평가."""
    best_params = best_trial.params

    s1_e = best_params["s1_enter"]
    gap_12 = best_params["gap_12"]
    gap_23 = best_params["gap_23"]
    gap_34 = best_params["gap_34"]

    # 근처 trials 찾기 (각 파라미터 차이 <= 5)
    neighbor_values = []
    for trial in study.trials:
        if trial.value is None:
            continue

        p = trial.params
        diff = (
            abs(p["s1_enter"] - s1_e) +
            abs(p["s1_exit"] - best_params["s1_exit"]) +
            abs(p["gap_12"] - gap_12) +
            abs(p["s2_exit"] - best_params["s2_exit"]) +
            abs(p["gap_23"] - gap_23) +
            abs(p["s3_exit"] - best_params["s3_exit"]) +
            abs(p["gap_34"] - gap_34) +
            abs(p["s4_exit"] - best_params["s4_exit"])
        )

        # 차이가 10 이내인 이웃 수집
        if diff <= 10:
            neighbor_values.append(trial.value)

    if len(neighbor_values) < 3:
        return 0.0, "⚠️ (이웃 부족)"

    avg_neighbor = np.mean(neighbor_values)

    # 고원지대: 이웃 평균이 최적값의 90% 이상
    if best_sortino > 0 and avg_neighbor / best_sortino >= 0.90:
        return avg_neighbor / best_sortino, "✅"
    else:
        return avg_neighbor / best_sortino, "⚠️"


# ══════════════════════════════════════════════════════════════
# [5] 메인 실행
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("김째 과열 파라미터 고원지대 탐색 (Optuna)")
    print("=" * 70)
    print()

    # 원본 파라미터
    original_params = BasicParams(spy_bear_cap=0.0)
    print(f"【원본】 {original_params.overheat1_enter}/{original_params.overheat1_exit} → "
          f"{original_params.overheat2_enter}/{original_params.overheat2_exit} → "
          f"{original_params.overheat3_enter}/{original_params.overheat3_exit} → "
          f"{original_params.overheat4_enter}/{original_params.overheat4_exit}")
    print()

    # 3개 기간 데이터 로드
    data_ndx = load_data_period("index")
    data_qqq = load_data_period("synth")
    data_real = load_data_period("real")
    print()

    # Phase 1: Optuna 최적화 (3개 Study)
    results = {}

    for period_name, data_tuple in [
        ("NDX", data_ndx),
        ("QQQ", data_qqq),
        ("TQQQ", data_real),
    ]:
        print("=" * 70)
        print(f"【{period_name} 최적화】 {period_name} 기간 기준")
        print("=" * 70)

        qqq, tqqq, spy, rf = data_tuple

        # Optuna Study
        sampler = TPESampler(multivariate=True, seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        objective = create_objective(qqq, tqqq, spy, rf, period_name)

        study.optimize(objective, n_trials=1500, show_progress_bar=False)

        # 최적 trial
        best_trial = study.best_trial
        best_value = best_trial.value

        # 최적 파라미터 추출
        best_params_dict = best_trial.params
        s1_e = best_params_dict["s1_enter"]
        s1_x = best_params_dict["s1_exit"]
        gap_12 = best_params_dict["gap_12"]
        s2_e = s1_e + gap_12
        s2_x = best_params_dict["s2_exit"]
        gap_23 = best_params_dict["gap_23"]
        s3_e = s2_e + gap_23
        s3_x = best_params_dict["s3_exit"]
        gap_34 = best_params_dict["gap_34"]
        s4_e = s3_e + gap_34
        s4_x = best_params_dict["s4_exit"]

        best_params = BasicParams(
            overheat1_enter=s1_e,
            overheat1_exit=s1_x,
            overheat2_enter=s2_e,
            overheat2_exit=s2_x,
            overheat3_enter=s3_e,
            overheat3_exit=s3_x,
            overheat4_enter=s4_e,
            overheat4_exit=s4_x,
            spy_bear_cap=0.0,
        )

        print(f"최적 파라미터: {s1_e}/{s1_x} → {s2_e}/{s2_x} → {s3_e}/{s3_x} → {s4_e}/{s4_x}")
        print(f"Sortino: {best_value:.4f}")

        # 고원지대 검증
        plateau_ratio, plateau_status = check_plateau_from_trials(best_trial, study, best_value)
        print(f"고원지대: {plateau_status} (이웃평균/최적 = {plateau_ratio:.3f})")
        print()

        results[period_name] = {
            "params": best_params,
            "sortino": best_value,
            "plateau_ratio": plateau_ratio,
            "plateau_status": plateau_status,
            "trial": best_trial,
        }

    # Phase 3: Cross-Period Validation
    print("=" * 70)
    print("【Cross-Period Validation】")
    print("=" * 70)

    all_data = {"NDX": data_ndx, "QQQ": data_qqq, "TQQQ": data_real}

    validation_rows = []

    for opt_period in ["NDX", "QQQ", "TQQQ"]:
        row = {"Optimized_On": opt_period}
        opt_params = results[opt_period]["params"]

        for eval_period in ["NDX", "QQQ", "TQQQ"]:
            qqq, tqqq, spy, rf = all_data[eval_period]
            sortino, _ = backtest_and_score(qqq, tqqq, spy, rf, opt_params)
            row[f"{eval_period}_Sortino"] = sortino

        validation_rows.append(row)

    # 원본 파라미터로도 검증
    row = {"Optimized_On": "Original"}
    for eval_period in ["NDX", "QQQ", "TQQQ"]:
        qqq, tqqq, spy, rf = all_data[eval_period]
        sortino, _ = backtest_and_score(qqq, tqqq, spy, rf, original_params)
        row[f"{eval_period}_Sortino"] = sortino
    validation_rows.append(row)

    val_df = pd.DataFrame(validation_rows)
    print(val_df.to_string(index=False))
    print()

    # 출력 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV: 최적 파라미터 + 검증
    output_rows = []
    for opt_period in ["NDX", "QQQ", "TQQQ"]:
        p = results[opt_period]["params"]
        output_rows.append({
            "Period": opt_period,
            "S1_Enter": p.overheat1_enter,
            "S1_Exit": p.overheat1_exit,
            "S2_Enter": p.overheat2_enter,
            "S2_Exit": p.overheat2_exit,
            "S3_Enter": p.overheat3_enter,
            "S3_Exit": p.overheat3_exit,
            "S4_Enter": p.overheat4_enter,
            "S4_Exit": p.overheat4_exit,
            "Sortino": results[opt_period]["sortino"],
            "Plateau_Ratio": results[opt_period]["plateau_ratio"],
            "Plateau_Status": results[opt_period]["plateau_status"],
        })

    output_df = pd.DataFrame(output_rows)
    csv_path = OUT_DIR / f"overheat_plateau_{timestamp}.csv"
    output_df.to_csv(csv_path, index=False)
    print(f"CSV 저장: {csv_path}")

    # MD 보고서
    md_path = OUT_DIR / f"../docs/OVERHEAT_PLATEAU_{timestamp}.md"
    md_path.parent.mkdir(exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 김째 과열 파라미터 고원지대 탐색\n\n")
        f.write(f"**생성일**: {timestamp}\n\n")

        f.write("## 최적 파라미터 (기간별)\n\n")
        f.write("| Period | S1_Enter | S1_Exit | S2_Enter | S2_Exit | S3_Enter | S3_Exit | S4_Enter | S4_Exit | Sortino | Plateau_Ratio | Plateau_Status |\n")
        f.write("|--------|----------|---------|----------|---------|----------|---------|----------|---------|---------|---------------|----------------|\n")
        for _, row in output_df.iterrows():
            f.write(f"| {row['Period']} | {int(row['S1_Enter'])} | {int(row['S1_Exit'])} | {int(row['S2_Enter'])} | {int(row['S2_Exit'])} | {int(row['S3_Enter'])} | {int(row['S3_Exit'])} | {int(row['S4_Enter'])} | {int(row['S4_Exit'])} | {row['Sortino']:.4f} | {row['Plateau_Ratio']:.3f} | {row['Plateau_Status']} |\n")
        f.write("\n\n")

        f.write("## Cross-Period Validation\n\n")
        f.write("| Optimized_On | NDX_Sortino | QQQ_Sortino | TQQQ_Sortino |\n")
        f.write("|---|---|---|---|\n")
        for _, row in val_df.iterrows():
            f.write(f"| {row['Optimized_On']} | {row['NDX_Sortino']:.6f} | {row['QQQ_Sortino']:.6f} | {row['TQQQ_Sortino']:.6f} |\n")
        f.write("\n\n")

        f.write("## 평가\n\n")
        for period in ["NDX", "QQQ", "TQQQ"]:
            p = results[period]["params"]
            f.write(f"### {period} 최적\n\n")
            f.write(f"파라미터: {p.overheat1_enter}/{p.overheat1_exit} → "
                   f"{p.overheat2_enter}/{p.overheat2_exit} → "
                   f"{p.overheat3_enter}/{p.overheat3_exit} → "
                   f"{p.overheat4_enter}/{p.overheat4_exit}\n\n")
            f.write(f"Sortino: {results[period]['sortino']:.4f}\n\n")
            f.write(f"고원지대: {results[period]['plateau_status']} "
                   f"(이웃평균/최적 = {results[period]['plateau_ratio']:.3f})\n\n")

    print(f"MD 보고서 저장: {md_path}")
    print()
    print("=" * 70)
    print("✅ 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
