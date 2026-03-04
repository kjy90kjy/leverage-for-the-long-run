"""
Update NDX Signal JSON for iOS Scriptable Widget — 김째 S2

김째매매법 S2 상태머신을 전체 구간 시뮬레이션하여 마지막 날 상태를 추출,
iOS Scriptable 위젯용 JSON으로 저장한다.

Schedule: Hourly via Windows Task Scheduler (07:00~18:00 ET = 20:00~07:00 KST)
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

import math
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports from lrs_standalone.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from lrs_standalone import (
    BasicParams, code_weight, weight_to_code, is_high_exposure,
    rsi_wilder, sample_stdev, rolling_linreg_slope,
    download, synthesize_tqqq,
)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "ndx_signal.json"


def compute_kimjje_last_state(
    qqq_close: np.ndarray,
    tqqq_close: np.ndarray,
    spy_close: np.ndarray,
    params: BasicParams,
) -> dict:
    """김째 S2 상태머신을 전체 구간 돌려서 마지막 날의 상태를 반환.

    compute_kimjje_strategy()와 동일 로직이지만 마지막 바의 내부 상태를
    모두 dict로 반환한다.
    """
    n = len(tqqq_close)

    # MA 계산
    q_ma3 = pd.Series(qqq_close).rolling(3, min_periods=3).mean().to_numpy()
    q_ma161 = pd.Series(qqq_close).rolling(161, min_periods=161).mean().to_numpy()
    t_ma200 = pd.Series(tqqq_close).rolling(200, min_periods=200).mean().to_numpy()
    spy_ma200 = pd.Series(spy_close).rolling(200, min_periods=200).mean().to_numpy()

    dist200 = np.where(
        np.isnan(tqqq_close) | np.isnan(t_ma200) | (t_ma200 == 0),
        np.nan, (tqqq_close / t_ma200) * 100.0
    )

    t_ret = np.full(n, np.nan, dtype=float)
    t_ret[1:] = (tqqq_close[1:] / tqqq_close[:-1]) - 1.0
    t_v20 = sample_stdev(t_ret, params.vol_len)

    spy_dist200 = np.where(
        np.isnan(spy_close) | np.isnan(spy_ma200) | (spy_ma200 == 0),
        np.nan, (spy_close / spy_ma200) * 100.0
    )

    q_rsi = rsi_wilder(qqq_close, params.rsi_len)
    rsi_reentry_ok = (~np.isnan(q_rsi)) & (q_rsi >= params.rsi_reentry_thr)

    dist_slope = rolling_linreg_slope(dist200, params.slope_len)

    ready = (
        (~np.isnan(q_ma3))
        & (~np.isnan(q_ma161))
        & (~np.isnan(t_v20))
        & (~np.isnan(dist200))
        & ((not params.use_spy_filter) | (~np.isnan(spy_dist200)))
    )

    # 상태 변수
    locked = False
    assetCode = 0
    slopeOkFlag = False
    slopeApplicableFlag = False
    above200Hyst = False
    above200HystInited = False
    overheatStage = 0
    fullEntryClose = math.nan
    fullHighClose = math.nan
    lastPrincipalStopPrice = math.nan
    reentryLock = False
    spyBull = False
    spyInited = False
    spyConfirmCnt = 0
    tp10CycleActive = False
    tp10Reduced = False
    tp10EntryClose = math.nan
    posEntryAvgClose = math.nan
    posEntryAvgClosePrev = math.nan

    # safe values
    volThrSafe = max(params.vol_threshold, 1e-6)
    dist200ExitSafe = params.dist200_exit
    dist200EnterSafe = max(params.dist200_enter, dist200ExitSafe + 0.01)
    spyExitSafe = params.spy_exit
    spyEnterSafe = max(params.spy_enter, spyExitSafe + 0.01)
    spyConfirmSafe = max(int(params.spy_confirm_days), 1)
    spyBearCapSafe = min(max(params.spy_bear_cap, 0.0), 1.0)

    s1Enter = params.overheat1_enter
    s1Exit = min(params.overheat1_exit, s1Enter - 0.01)
    s2Enter = max(params.overheat2_enter, s1Enter + 0.01)
    s2Exit = min(params.overheat2_exit, s2Enter - 0.01)
    s3Enter = max(params.overheat3_enter, s2Enter + 0.01)
    s3Exit = min(params.overheat3_exit, s3Enter - 0.01)
    s4Enter = max(params.overheat4_enter, s3Enter + 0.01)
    s4Exit = min(params.overheat4_exit, s4Enter - 0.01)

    # 루프 내에서 매 바마다 갱신할 baseCode (마지막 바 값을 반환하기 위해)
    baseCode = 0
    stopHit = False

    for i in range(n):
        prevAsset = assetCode
        posEntryAvgClosePrev = posEntryAvgClose

        if not bool(ready[i]):
            assetCode = 0
            locked = False
            slopeOkFlag = False
            slopeApplicableFlag = False
            above200Hyst = False
            above200HystInited = False
            overheatStage = 0
            fullEntryClose = math.nan
            fullHighClose = math.nan
            lastPrincipalStopPrice = math.nan
            reentryLock = False
            spyBull = False
            spyInited = False
            spyConfirmCnt = 0
            tp10CycleActive = False
            tp10Reduced = False
            tp10EntryClose = math.nan
            posEntryAvgClose = math.nan
            posEntryAvgClosePrev = math.nan
            baseCode = 0
            stopHit = False
            continue

        # 1) vol lock
        locked = (t_v20[i] + 1e-10) >= volThrSafe

        # 2) SPY filter
        if not params.use_spy_filter:
            spyBull = True
            spyInited = True
            spyConfirmCnt = 0
        else:
            if not spyInited:
                spyBull = bool(spy_dist200[i] >= spyEnterSafe)
                spyInited = True
                spyConfirmCnt = 0
            else:
                if spyBull:
                    if spy_dist200[i] <= spyExitSafe:
                        spyConfirmCnt += 1
                        if spyConfirmCnt >= spyConfirmSafe:
                            spyBull = False
                            spyConfirmCnt = 0
                    else:
                        spyConfirmCnt = 0
                else:
                    if spy_dist200[i] >= spyEnterSafe:
                        spyConfirmCnt += 1
                        if spyConfirmCnt >= spyConfirmSafe:
                            spyBull = True
                            spyConfirmCnt = 0
                    else:
                        spyConfirmCnt = 0

        spyBearNow = params.use_spy_filter and spyInited and (not spyBull)
        spyForceCash = spyBearNow and (spyBearCapSafe <= 1e-9)

        # 3) reentry block
        reentryBlockedNow = reentryLock and (not bool(rsi_reentry_ok[i]))

        # 4) 200dist hysteresis
        if not above200HystInited:
            above200Hyst = bool(dist200[i] >= dist200EnterSafe)
            above200HystInited = True
        else:
            if above200Hyst and (dist200[i] <= dist200ExitSafe):
                above200Hyst = False
            elif (not above200Hyst) and (dist200[i] >= dist200EnterSafe):
                above200Hyst = True

        # 5) baseCode
        if locked or spyForceCash or reentryBlockedNow:
            baseCode = 0
            slopeOkFlag = False
            slopeApplicableFlag = False
        else:
            base0 = 2 if above200Hyst else (1 if (q_ma3[i] > q_ma161[i]) else 0)
            slopeApplicable = (
                params.use_slope_boost
                and (base0 == 1)
                and (not np.isnan(dist_slope[i]))
                and (dist200[i] <= params.dist_cap)
                and (t_v20[i] <= params.vol_cap)
            )
            slopeApplicableFlag = bool(slopeApplicable)
            slopeOk = bool(slopeApplicable) and (dist_slope[i] >= params.slope_thr)
            slopeOkFlag = bool(slopeOk)
            baseCode = 2 if slopeOk else base0

        # 6) overheat stage
        if (not params.use_overheat_split) or locked or spyForceCash or reentryBlockedNow:
            overheatStage = 0
        else:
            st = overheatStage
            if st == 0:
                if baseCode == 2:
                    if dist200[i] >= s4Enter: st = 4
                    elif dist200[i] >= s3Enter: st = 3
                    elif dist200[i] >= s2Enter: st = 2
                    elif dist200[i] >= s1Enter: st = 1
                    else: st = 0
                else:
                    st = 0
            elif st == 1:
                if dist200[i] >= s4Enter: st = 4
                elif dist200[i] >= s3Enter: st = 3
                elif dist200[i] >= s2Enter: st = 2
                elif dist200[i] <= s1Exit: st = 0
                else: st = 1
            elif st == 2:
                if dist200[i] >= s4Enter: st = 4
                elif dist200[i] >= s3Enter: st = 3
                elif dist200[i] <= s1Exit: st = 0
                elif dist200[i] <= s2Exit: st = 1
                else: st = 2
            elif st == 3:
                if dist200[i] >= s4Enter: st = 4
                elif dist200[i] <= s1Exit: st = 0
                elif dist200[i] <= s2Exit: st = 1
                elif dist200[i] <= s3Exit: st = 2
                else: st = 3
            else:
                st = 0 if (dist200[i] <= s4Exit) else 4
            overheatStage = st

        preStopCode = baseCode
        if params.use_overheat_split and (not locked) and (not spyForceCash) and (not reentryBlockedNow):
            if overheatStage == 4: preStopCode = 0
            elif overheatStage == 3: preStopCode = 0 if (baseCode == 0) else 5
            elif overheatStage == 2: preStopCode = 4 if (baseCode == 2) else baseCode
            elif overheatStage == 1: preStopCode = 3 if (baseCode == 2) else baseCode

        # 7) principal stop
        stopHit = False
        prevHigh = is_high_exposure(prevAsset)

        if params.use_principal_stop and prevHigh and (not math.isnan(fullEntryClose)):
            lossCutPrice = fullEntryClose * params.principal_stop_pct
            lastPrincipalStopPrice = lossCutPrice
            stopHit = bool(tqqq_close[i] <= lossCutPrice)
            if stopHit:
                reentryLock = True

        rawCode = 0 if stopHit else preStopCode
        finalCode = rawCode

        # 8) SPY bear cap
        if spyBearNow:
            if code_weight(finalCode) > (spyBearCapSafe + 1e-9):
                finalCode = weight_to_code(spyBearCapSafe)

        # 9) reentry block
        if reentryBlockedNow:
            finalCode = 0

        # 10) TP10
        wFinalPreTp = code_weight(finalCode)
        inHighNow = wFinalPreTp >= (0.80 - 1e-9)

        if (not params.use_tp10) or (not inHighNow) or locked or stopHit or spyBearNow or reentryBlockedNow:
            tp10CycleActive = False
            tp10Reduced = False
            tp10EntryClose = math.nan
        else:
            entered100 = (finalCode == 2) and (code_weight(prevAsset) < 0.999) and (not tp10CycleActive) and (not tp10Reduced)
            if entered100:
                tp10CycleActive = True
                tp10Reduced = False
                tp10EntryClose = float(tqqq_close[i])

            tpHitNow = (
                tp10CycleActive
                and (not tp10Reduced)
                and (finalCode == 2)
                and (not math.isnan(tp10EntryClose))
                and (tqqq_close[i] >= tp10EntryClose * (1.0 + params.tp10_trigger))
            )
            if tpHitNow:
                tp10Reduced = True

            if tp10Reduced and (finalCode == 2):
                finalCode = weight_to_code(params.tp10_cap)

        # 11) cap 재확인
        if spyBearNow:
            if code_weight(finalCode) > (spyBearCapSafe + 1e-9):
                finalCode = weight_to_code(spyBearCapSafe)
        if reentryBlockedNow:
            finalCode = 0

        assetCode = int(finalCode)

        # 12) reentryLock 해제
        if reentryLock and (code_weight(assetCode) > 1e-9):
            reentryLock = False

        # 13) high exposure tracking
        nowHigh = is_high_exposure(assetCode)
        enteredHigh = nowHigh and (not prevHigh)
        stayedHigh = nowHigh and prevHigh
        exitedHigh = (not nowHigh) and prevHigh

        if enteredHigh:
            fullEntryClose = float(tqqq_close[i])
            fullHighClose = float(tqqq_close[i])
            lastPrincipalStopPrice = float(tqqq_close[i]) * params.principal_stop_pct
        elif stayedHigh:
            if math.isnan(fullHighClose):
                fullHighClose = float(tqqq_close[i])
            else:
                fullHighClose = max(fullHighClose, float(tqqq_close[i]))
            lastPrincipalStopPrice = float(fullEntryClose) * params.principal_stop_pct
        elif exitedHigh:
            fullEntryClose = math.nan
            fullHighClose = math.nan
            lastPrincipalStopPrice = math.nan

        if not nowHigh:
            lastPrincipalStopPrice = math.nan

        # 14) posEntryAvgClose
        prevW = code_weight(prevAsset)
        nowW = code_weight(assetCode)
        epsW = 1e-9
        if nowW <= epsW:
            posEntryAvgClose = math.nan
        else:
            if prevW <= epsW:
                posEntryAvgClose = float(tqqq_close[i])
            elif nowW > prevW + epsW:
                oldEntry = posEntryAvgClosePrev
                if math.isnan(oldEntry):
                    oldEntry = float(tqqq_close[i])
                posEntryAvgClose = (oldEntry * prevW + float(tqqq_close[i]) * (nowW - prevW)) / nowW
            else:
                posEntryAvgClose = posEntryAvgClosePrev

    # ── 마지막 바 상태 반환 ──
    last = n - 1
    return {
        "qqq_price": float(qqq_close[last]),
        "qqq_ma3": float(q_ma3[last]) if not np.isnan(q_ma3[last]) else None,
        "qqq_ma161": float(q_ma161[last]) if not np.isnan(q_ma161[last]) else None,
        "golden_cross": bool(q_ma3[last] > q_ma161[last]) if (not np.isnan(q_ma3[last]) and not np.isnan(q_ma161[last])) else False,

        "tqqq_price": float(tqqq_close[last]),
        "tqqq_ma200": float(t_ma200[last]) if not np.isnan(t_ma200[last]) else None,
        "dist200": float(dist200[last]) if not np.isnan(dist200[last]) else None,
        "above200": above200Hyst,

        "base_code": baseCode,
        "base_code_label": {0: "관망", 1: "골든크로스", 2: "풀진입"}.get(baseCode, f"code{baseCode}"),

        "overheat_stage": overheatStage,
        "final_code": assetCode,
        "final_weight_pct": int(round(code_weight(assetCode) * 100)),

        "vol_locked": locked,
        "vol20": float(t_v20[last]) if not np.isnan(t_v20[last]) else None,
        "spy_bull": spyBull,
        "spy_price": float(spy_close[last]),
        "spy_ma200": float(spy_ma200[last]) if not np.isnan(spy_ma200[last]) else None,
        "spy_dist200": float(spy_dist200[last]) if not np.isnan(spy_dist200[last]) else None,
        "stop_hit": stopHit,
        "stop_price": float(lastPrincipalStopPrice) if not math.isnan(lastPrincipalStopPrice) else None,

        "reentry_lock": reentryLock,
        "tp10_reduced": tp10Reduced,
        "slope_ok": slopeOkFlag,

        # 과열 경계값 (다음 스테이지 전환 가격 계산용)
        "_overheat_bounds": {
            "s1_enter": s1Enter, "s1_exit": s1Exit,
            "s2_enter": s2Enter, "s2_exit": s2Exit,
            "s3_enter": s3Enter, "s3_exit": s3Exit,
            "s4_enter": s4Enter, "s4_exit": s4Exit,
        },
    }


def compute_next_thresholds(state: dict) -> dict:
    """현재 과열 스테이지 기준으로 다음 상승/하락 전환 이격도 및 TQQQ 가격 계산."""
    bounds = state["_overheat_bounds"]
    stage = state["overheat_stage"]
    ma200 = state["tqqq_ma200"]

    if ma200 is None or ma200 <= 0:
        return {"next_overheat_up": None, "next_overheat_down": None}

    # 다음 상승 스테이지
    up_map = {
        0: (1, bounds["s1_enter"]),
        1: (2, bounds["s2_enter"]),
        2: (3, bounds["s3_enter"]),
        3: (4, bounds["s4_enter"]),
        4: (None, None),
    }
    up_stage, up_dist = up_map.get(stage, (None, None))

    next_up = None
    if up_stage is not None and up_dist is not None:
        next_up = {
            "stage": up_stage,
            "dist200": up_dist,
            "tqqq_price": round(ma200 * up_dist / 100, 2),
        }

    # 다음 하락 스테이지
    down_map = {
        0: (None, None),
        1: (0, bounds["s1_exit"]),
        2: (1, bounds["s2_exit"]),
        3: (2, bounds["s3_exit"]),
        4: (0, bounds["s4_exit"]),
    }
    down_stage, down_dist = down_map.get(stage, (None, None))

    next_down = None
    if down_stage is not None and down_dist is not None:
        next_down = {
            "stage": down_stage,
            "dist200": down_dist,
            "tqqq_price": round(ma200 * down_dist / 100, 2),
        }

    return {"next_overheat_up": next_up, "next_overheat_down": next_down}


def compute_qqq_cross_price(qqq_prices: np.ndarray) -> float | None:
    """QQQ MA3 = MA161 교차 가격을 이진 탐색으로 계산.

    내일 종가가 X일 때 MA3(new) == MA161(new)이 되는 X를 찾는다.
    """
    n = len(qqq_prices)
    if n < 161:
        return None

    # MA161: 마지막 160개 + 새 가격 1개 = 161개
    # MA3: 마지막 2개 + 새 가격 1개 = 3개
    last_160 = qqq_prices[-(161 - 1):]  # 160개
    last_2 = qqq_prices[-2:]            # 2개

    # f(x) = MA3(x) - MA161(x)
    # MA3(x)  = (last_2[0] + last_2[1] + x) / 3  -- 단, last_2[1]은 today
    # 실제로는 today가 마지막이므로 virtual = [..., x]
    # MA3: (prices[-2] + prices[-1] + x) / 3  -- 틀림, 오늘이 prices[-1]
    # 이 함수는 "오늘 종가가 바뀌면" MA가 어떻게 변하는지 계산
    # → 실제로는 내일 신호 전환 가격이 아니라 오늘 종가 기준 교차점

    # 간단화: 현재 MA3 vs MA161에서 교차 가격 = 오늘 종가가 X로 바뀌었을 때 MA3==MA161
    sum_ma161_rest = float(last_160[:-1].sum())  # 159개 합 (오늘 제외)
    sum_ma3_rest = float(last_2[0])              # MA3의 나머지 1개 (어제 + 그제 중 2개)

    # MA3 = (qqq[-3] + qqq[-2] + x) / 3,  MA161 = (sum_of_160 + x) / 161
    # 여기서 x = 오늘 종가 변경값
    if n >= 3:
        sum_ma3_rest = float(qqq_prices[-3] + qqq_prices[-2])
    else:
        return None

    sum_ma161_rest = float(qqq_prices[-(161):-1].sum()) if n >= 161 else None
    if sum_ma161_rest is None:
        return None

    # MA3(x) = (sum_ma3_rest + x) / 3
    # MA161(x) = (sum_ma161_rest + x) / 161
    # 교차: (sum_ma3_rest + x)/3 = (sum_ma161_rest + x)/161
    # 161*(sum_ma3_rest + x) = 3*(sum_ma161_rest + x)
    # 161*sum_ma3_rest + 161*x = 3*sum_ma161_rest + 3*x
    # (161-3)*x = 3*sum_ma161_rest - 161*sum_ma3_rest
    # x = (3*sum_ma161_rest - 161*sum_ma3_rest) / 158

    x = (3.0 * sum_ma161_rest - 161.0 * sum_ma3_rest) / 158.0
    if x <= 0:
        return None
    return round(x, 2)


def generate_signal() -> dict | None:
    """김째 S2 마지막 날 상태를 계산하여 JSON용 dict 반환."""
    try:
        # 데이터 다운로드 — 합성 TQQQ 사용 (NDX 기반, 충분한 히스토리)
        print("  데이터 다운로드...")
        qqq = download("QQQ", start="2009-01-01")
        tqqq = download("TQQQ", start="2009-01-01")
        spy = download("SPY", start="2009-01-01")

        # 공통 인덱스
        common = qqq.index.intersection(tqqq.index).intersection(spy.index)
        qqq = qqq.loc[common]
        tqqq = tqqq.loc[common]
        spy = spy.loc[common]

        print(f"  데이터: {common[0].date()} → {common[-1].date()} ({len(common)}일)")

        # 김째 S2 상태머신 실행
        print("  S2 상태머신 실행...")
        params = BasicParams()  # S2 기본값 (spy_bear_cap=0.0)
        state = compute_kimjje_last_state(
            qqq.values.astype(float),
            tqqq.values.astype(float),
            spy.values.astype(float),
            params,
        )

        # 다음 과열 스테이지 전환 가격
        thresholds = compute_next_thresholds(state)

        # QQQ 골든크로스 교차 가격
        qqq_cross = compute_qqq_cross_price(qqq.values.astype(float))

        # JSON 구성
        last_date = common[-1]
        data = {
            "date": last_date.strftime("%Y-%m-%d"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "qqq_price": round(state["qqq_price"], 2),
            "qqq_ma3": round(state["qqq_ma3"], 2) if state["qqq_ma3"] else None,
            "qqq_ma161": round(state["qqq_ma161"], 2) if state["qqq_ma161"] else None,
            "golden_cross": bool(state["golden_cross"]),

            "tqqq_price": round(state["tqqq_price"], 2),
            "tqqq_ma200": round(state["tqqq_ma200"], 2) if state["tqqq_ma200"] else None,
            "dist200": round(state["dist200"], 1) if state["dist200"] else None,
            "above200": bool(state["above200"]),

            "base_code": state["base_code"],
            "base_code_label": state["base_code_label"],

            "overheat_stage": state["overheat_stage"],
            "final_code": state["final_code"],
            "final_weight_pct": state["final_weight_pct"],

            "vol_locked": bool(state["vol_locked"]),
            "spy_bull": bool(state["spy_bull"]),
            "spy_price": round(state["spy_price"], 2),
            "spy_dist200": round(state["spy_dist200"], 1) if state["spy_dist200"] else None,
            "stop_hit": bool(state["stop_hit"]),
            "stop_price": round(state["stop_price"], 2) if state["stop_price"] else None,

            "next_overheat_up": thresholds["next_overheat_up"],
            "next_overheat_down": thresholds["next_overheat_down"],
            "qqq_cross_price": qqq_cross,
        }

        return data

    except Exception as e:
        import traceback
        print(f"  Error: {traceback.format_exc()}")
        return None


def save_json(data: dict) -> Path:
    """JSON 파일 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return OUTPUT_FILE


def main():
    print("=" * 60)
    print("  김째 S2 Signal JSON Generator")
    print("=" * 60)

    data = generate_signal()

    if data is None:
        print("  FAILED")
        return False

    # 상태 출력
    print(f"\n  날짜: {data['date']}")
    print(f"  QQQ:  ${data['qqq_price']}  MA3=${data['qqq_ma3']}  MA161=${data['qqq_ma161']}  GC={'Y' if data['golden_cross'] else 'N'}")
    print(f"  TQQQ: ${data['tqqq_price']}  MA200=${data['tqqq_ma200']}")
    print(f"  이격도: {data['dist200']}%  above200={'Y' if data['above200'] else 'N'}")
    print(f"  baseCode: {data['base_code']} ({data['base_code_label']})")
    print(f"  과열: Stage {data['overheat_stage']}  → finalCode: {data['final_code']}  비중: {data['final_weight_pct']}%")
    print(f"  SPY: ${data['spy_price']}  이격도={data['spy_dist200']}%  Bull={'Y' if data['spy_bull'] else 'N'}")
    print(f"  Vol locked: {'Y' if data['vol_locked'] else 'N'}  Stop hit: {'Y' if data['stop_hit'] else 'N'}")

    if data["next_overheat_up"]:
        nxt = data["next_overheat_up"]
        print(f"  다음 ↑ S{nxt['stage']}: TQQQ ${nxt['tqqq_price']} ({nxt['dist200']}%)")
    if data["next_overheat_down"]:
        nxt = data["next_overheat_down"]
        print(f"  다음 ↓ S{nxt['stage']}: TQQQ ${nxt['tqqq_price']} ({nxt['dist200']}%)")
    if data["qqq_cross_price"]:
        print(f"  QQQ GC 교차가: ${data['qqq_cross_price']}")

    # 저장
    path = save_json(data)
    print(f"\n  JSON 저장: {path}")
    print("=" * 60)
    print("  Done!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
