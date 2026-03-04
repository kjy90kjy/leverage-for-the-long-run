"""
LRS Standalone — 검증된 프로덕션 전략 비교 (단일 파일)

전략 1: Regime-Switching Dual MA (Conservative P1)
전략 2: 김째 S1 — TQQQ 200MA 크로스 (오리지널)
전략 3: 김째 S2 — 변동성+SPY 필터+과열감량+손절+TP10 (spy_bear_cap=0%)
전략 4: 김째 S3 — S2와 동일하되 SPY 약세 시 10% 유지 (spy_bear_cap=10%)

실행:
    python lrs_standalone.py
"""

import sys
import io
import warnings

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import math
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# [2] 상수 & 파라미터
# ══════════════════════════════════════════════════════════════

SIGNAL_LAG = 1          # 익일 집행 (필수)
COMMISSION = 0.002      # 0.2% per trade
CALIBRATED_ER = 0.035   # 합성 TQQQ 연간 비용 (calibrate_tqqq.py 결과)

# Conservative P1 — regime-switching 최적 파라미터
REGIME_PARAMS = dict(
    fast_low=12, slow_low=237,
    fast_high=6, slow_high=229,
    vol_lookback=49, vol_threshold_pct=57.3,
)

# TQQQ 200일선 워밍업 필요 → 2011-01-01부터 비교 시작
COMPARE_START = "2011-01-01"


@dataclass(frozen=True)
class BasicParams:
    """김째매매법 Pine v6 파라미터 (S2 기본값)."""
    rsi_len: int = 14
    rsi_reentry_thr: float = 43.0
    vol_threshold: float = 0.059
    vol_len: int = 20
    dist200_enter: float = 101.00
    dist200_exit: float = 100.00
    use_slope_boost: bool = True
    slope_len: int = 45
    slope_thr: float = 0.1100
    dist_cap: float = 98.8
    vol_cap: float = 0.06
    use_overheat_split: bool = True
    overheat1_enter: float = 139.0
    overheat1_exit: float = 132.0
    overheat2_enter: float = 146.0
    overheat2_exit: float = 138.0
    overheat3_enter: float = 149.0
    overheat3_exit: float = 140.0
    overheat4_enter: float = 151.0
    overheat4_exit: float = 118.0
    use_principal_stop: bool = True
    principal_stop_pct: float = 0.941
    use_spy_filter: bool = True
    spy_enter: float = 100.25
    spy_exit: float = 97.75
    spy_confirm_days: int = 1
    spy_bear_cap: float = 0.0      # S2=0.0 (하락 시 전량 청산)
    use_tp10: bool = True
    tp10_trigger: float = 0.10
    tp10_cap: float = 0.95


# ══════════════════════════════════════════════════════════════
# [3] 데이터 다운로드
# ══════════════════════════════════════════════════════════════

def download(ticker: str, start: str = "2009-01-01", end: str = "2030-12-31") -> pd.Series:
    """yfinance에서 Adj Close 시리즈 다운로드."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    close = df["Close"].squeeze()
    close.name = ticker
    return close


_RF_CACHE = None

def download_ken_french_rf() -> pd.Series:
    """Ken French Data Library에서 일별 무위험 이자율 다운로드.
    Returns: 날짜 인덱스 Series (decimal, e.g. 0.0001/day)."""
    global _RF_CACHE
    if _RF_CACHE is not None:
        return _RF_CACHE

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        fname = z.namelist()[0]
        raw = z.open(fname).read().decode("utf-8")

        lines = raw.strip().split("\n")
        data_lines = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 5 and parts[0].strip().isdigit():
                data_lines.append(parts)

        df = pd.DataFrame(data_lines, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
        df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y%m%d")
        df["RF"] = pd.to_numeric(df["RF"].str.strip()) / 100.0
        rf = df.set_index("date")["RF"]
        _RF_CACHE = rf
        print(f"  [Ken French] RF: {rf.index[0].date()} → {rf.index[-1].date()}")
        return rf
    except Exception as e:
        print(f"  [Warning] Ken French 다운로드 실패: {e}, 연 3% 대체 사용")
        idx = pd.date_range("1926-01-01", "2030-12-31", freq="B")
        _RF_CACHE = pd.Series(0.03 / 252, index=idx)
        return _RF_CACHE


def synthesize_tqqq(qqq: pd.Series, leverage: float = 3.0,
                     expense_ratio: float = CALIBRATED_ER) -> pd.Series:
    """QQQ 수익률로 합성 TQQQ 가격 시리즈 생성.
    daily_ret = qqq_ret * leverage - expense_ratio/252"""
    qqq_ret = qqq.pct_change().fillna(0)
    daily_cost = expense_ratio / 252
    synth_ret = qqq_ret * leverage - daily_cost
    synth_price = (1 + synth_ret).cumprod() * 100  # 임의 시작가 $100
    synth_price.iloc[0] = 100.0
    synth_price.name = "TQQQ_synth"
    return synth_price


def download_all_data(mode: str = "real") -> dict:
    """데이터 다운로드.

    mode:
      "real"  — 실제 QQQ/TQQQ/SPY (2010~)
      "synth" — QQQ(1999~) 기반 합성 TQQQ
      "index" — NDX/^GSPC(1985~) 기반 합성 TQQQ (최장 기간)
    Returns: dict with aligned pd.Series (공통 인덱스)."""
    labels = {"real": "실제 TQQQ", "synth": "합성 3x QQQ (1999~)",
              "index": "합성 3x NDX (1985~)"}
    print("=" * 60)
    print(f"데이터 다운로드 — {labels[mode]}")
    print("=" * 60)

    rf = download_ken_french_rf()

    if mode == "real":
        qqq = download("QQQ", start="2009-01-01")
        tqqq = download("TQQQ", start="2009-01-01")
        spy = download("SPY", start="2009-01-01")
        common = qqq.index.intersection(tqqq.index).intersection(spy.index)
    elif mode == "synth":
        qqq = download("QQQ", start="1998-01-01")
        spy = download("SPY", start="1998-01-01")
        tqqq = synthesize_tqqq(qqq)
        common = qqq.index.intersection(spy.index)
        print(f"  합성 TQQQ: QQQ × 3 − ER {CALIBRATED_ER:.1%}/yr")
    else:  # index
        qqq = download("^NDX", start="1984-01-01")
        spy = download("^GSPC", start="1984-01-01")
        tqqq = synthesize_tqqq(qqq)
        common = qqq.index.intersection(spy.index)
        print(f"  합성 TQQQ: NDX × 3 − ER {CALIBRATED_ER:.1%}/yr")
        print(f"  QQQ 대용: ^NDX, SPY 대용: ^GSPC")

    qqq = qqq.loc[common]
    tqqq = tqqq.reindex(common).loc[common]
    spy = spy.loc[common]
    rf = rf.reindex(common, method="ffill").fillna(0)

    print(f"  공통 기간: {common[0].date()} → {common[-1].date()} ({len(common)}일)")
    return {"qqq": qqq, "tqqq": tqqq, "spy": spy, "rf": rf}


# ══════════════════════════════════════════════════════════════
# [4] Regime-Switching Dual MA 시그널
# ══════════════════════════════════════════════════════════════

def signal_regime_switching_dual_ma(price: pd.Series,
                                     fast_low: int, slow_low: int,
                                     fast_high: int, slow_high: int,
                                     vol_lookback: int = 60,
                                     vol_threshold_pct: float = 50.0) -> pd.Series:
    """Regime-switching dual MA: 변동성 레짐별 다른 MA 쌍 사용.

    Low vol: SMA(fast_low) > SMA(slow_low) → 1
    High vol: SMA(fast_high) > SMA(slow_high) → 1
    """
    daily_ret = price.pct_change()
    rolling_vol = daily_ret.rolling(vol_lookback).std() * np.sqrt(252)

    # Rolling percentile (252일 = 1년) — 과거 극단값 왜곡 방지
    vol_pct = rolling_vol.rolling(252, min_periods=1).rank(pct=True) * 100
    high_vol = (vol_pct >= vol_threshold_pct).values

    # 4개 MA 사전 계산
    ma_fast_low = price.rolling(fast_low).mean().values
    ma_slow_low = price.rolling(slow_low).mean().values
    ma_fast_high = price.rolling(fast_high).mean().values
    ma_slow_high = price.rolling(slow_high).mean().values

    n = len(price)
    sig = np.zeros(n, dtype=int)
    state = 0
    warmup = max(slow_low, slow_high, vol_lookback)
    for i in range(n):
        if i < warmup or np.isnan(ma_slow_low[i]) or np.isnan(ma_slow_high[i]):
            sig[i] = 0
            continue
        if high_vol[i]:
            buy_cond = ma_fast_high[i] > ma_slow_high[i]
            sell_cond = ma_fast_high[i] <= ma_slow_high[i]
        else:
            buy_cond = ma_fast_low[i] > ma_slow_low[i]
            sell_cond = ma_fast_low[i] <= ma_slow_low[i]
        if state == 0 and buy_cond:
            state = 1
        elif state == 1 and sell_cond:
            state = 0
        sig[i] = state
    return pd.Series(sig, index=price.index)


# ══════════════════════════════════════════════════════════════
# [5] 김째매매법 헬퍼 함수
# ══════════════════════════════════════════════════════════════

def code_weight(code: int) -> float:
    """Pine assetCode → 비중 (0.0~1.0)."""
    if code == 2: return 1.00
    if code == 3: return 0.90
    if code == 4: return 0.80
    if code == 1: return 0.10
    if code == 5: return 0.05
    if code == 0: return 0.00
    if code >= 100: return float(min(max(code / 1000.0, 0.0), 1.0))
    return 0.00


def weight_to_code(w: float) -> int:
    """비중 → Pine assetCode."""
    wc = float(min(max(w, 0.0), 1.0))
    eps = 0.0005
    if wc <= eps: return 0
    if abs(wc - 0.05) < eps: return 5
    if abs(wc - 0.10) < eps: return 1
    if abs(wc - 0.80) < eps: return 4
    if abs(wc - 0.90) < eps: return 3
    if abs(wc - 1.00) < eps: return 2
    return int(round(wc * 1000.0))


def is_high_exposure(code: int) -> bool:
    return code_weight(code) >= (0.80 - 1e-9)


def rsi_wilder(close: np.ndarray, length: int) -> np.ndarray:
    """Pine RSI (Wilder): 초기 SMA + Wilder smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan, dtype=float)
    if n == 0:
        return rsi

    ch = np.diff(close, prepend=np.nan)
    up = np.where(np.isnan(ch), np.nan, np.maximum(ch, 0.0))
    dn = np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0.0))

    up_sma = pd.Series(up).rolling(length, min_periods=length).mean().to_numpy()
    dn_sma = pd.Series(dn).rolling(length, min_periods=length).mean().to_numpy()

    au = np.full(n, np.nan, dtype=float)
    ad = np.full(n, np.nan, dtype=float)

    for i in range(1, n):
        if np.isnan(au[i - 1]):
            au[i] = up_sma[i]
            ad[i] = dn_sma[i]
        else:
            au[i] = (au[i - 1] * (length - 1.0) + up[i]) / float(length)
            ad[i] = (ad[i - 1] * (length - 1.0) + dn[i]) / float(length)

        if np.isnan(au[i]) or np.isnan(ad[i]):
            continue
        if au[i] == 0 and ad[i] == 0:
            rsi[i] = 50.0
        elif ad[i] == 0:
            rsi[i] = 100.0
        elif au[i] == 0:
            rsi[i] = 0.0
        else:
            rs = au[i] / ad[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def sample_stdev(series: np.ndarray, length: int) -> np.ndarray:
    """Pine sampleStdev = 표본표준편차 (ddof=1)."""
    return pd.Series(series).rolling(length, min_periods=length).std(ddof=1).to_numpy()


def rolling_linreg_slope(y: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.linreg 기반 기울기."""
    n = len(y)
    out = np.full(n, np.nan, dtype=float)
    if length <= 1:
        return out

    x = np.arange(length, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = length * sum_x2 - (sum_x * sum_x)
    if denom == 0:
        return out

    for i in range(length - 1, n):
        w = y[i - length + 1: i + 1]
        if np.isnan(w).any():
            continue
        sum_y = float(w.sum())
        sum_xy = float((x * w).sum())
        m = (length * sum_xy - sum_x * sum_y) / denom
        out[i] = m
    return out


# ══════════════════════════════════════════════════════════════
# [6] 김째매매법 S2 상태머신
# ══════════════════════════════════════════════════════════════

def compute_kimjje_strategy(
    qqq_close: np.ndarray,
    tqqq_close: np.ndarray,
    spy_close: np.ndarray,
    params: BasicParams,
) -> tuple:
    """김째매매법 S2 — yfinance 데이터용 (MA 내부 계산).

    Returns: (codes, weights) — 각각 np.ndarray (int, float)
    """
    n = len(tqqq_close)
    codes = np.zeros(n, dtype=int)
    weights = np.zeros(n, dtype=float)

    # MA 계산 (QQQ 3일/161일, TQQQ 200일, SPY 200일)
    q_ma3 = pd.Series(qqq_close).rolling(3, min_periods=3).mean().to_numpy()
    q_ma161 = pd.Series(qqq_close).rolling(161, min_periods=161).mean().to_numpy()
    t_ma200 = pd.Series(tqqq_close).rolling(200, min_periods=200).mean().to_numpy()
    spy_ma200 = pd.Series(spy_close).rolling(200, min_periods=200).mean().to_numpy()

    # 파생 지표
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

    # 상태 변수 (Pine var)
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
            codes[i] = assetCode
            weights[i] = code_weight(assetCode)
            continue

        # 1) vol lock
        locked = (t_v20[i] + 1e-10) >= volThrSafe

        # 2) SPY filter state machine
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

        # 11) SPY cap / reentry cap 재확인
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

        # 14) posEntryAvgClose 갱신
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

        codes[i] = assetCode
        weights[i] = code_weight(assetCode)

    return codes, weights


def compute_s1_tqqq_200ma_cross(tqqq_close: np.ndarray) -> tuple:
    """김째 S1: TQQQ 200일선 상/하향 돌파 → 100%/0% 전환.

    Returns: (codes, weights) — 각각 np.ndarray
    """
    n = len(tqqq_close)
    t_ma200 = pd.Series(tqqq_close).rolling(200, min_periods=200).mean().to_numpy()
    code = np.zeros(n, dtype=int)
    weight = np.zeros(n, dtype=float)
    in_pos = False

    for i in range(n):
        if np.isnan(tqqq_close[i]) or np.isnan(t_ma200[i]):
            in_pos = False
            code[i] = 0
            weight[i] = 0.0
            continue

        if i == 0 or np.isnan(tqqq_close[i - 1]) or np.isnan(t_ma200[i - 1]):
            code[i] = 2 if in_pos else 0
            weight[i] = 1.0 if in_pos else 0.0
            continue

        cross_up = (tqqq_close[i - 1] <= t_ma200[i - 1]) and (tqqq_close[i] > t_ma200[i])
        cross_dn = (tqqq_close[i - 1] >= t_ma200[i - 1]) and (tqqq_close[i] < t_ma200[i])

        if (not in_pos) and cross_up:
            in_pos = True
        elif in_pos and cross_dn:
            in_pos = False

        code[i] = 2 if in_pos else 0
        weight[i] = 1.0 if in_pos else 0.0

    return code, weight


# ══════════════════════════════════════════════════════════════
# [7] 백테스트 엔진
# ══════════════════════════════════════════════════════════════

def backtest_regime(tqqq: pd.Series, qqq: pd.Series, rf: pd.Series) -> tuple:
    """Regime-Switching 전략 백테스트 (실제 TQQQ 수익률).

    Returns: (cum, signal) — 누적 수익률, 시그널 시리즈
    """
    sig_raw = signal_regime_switching_dual_ma(qqq, **REGIME_PARAMS)
    sig = sig_raw.shift(SIGNAL_LAG).fillna(0).astype(int)

    tqqq_ret = tqqq.pct_change().fillna(0)
    rf_daily = rf.reindex(tqqq_ret.index, method="ffill").fillna(0)

    strat_ret = sig * tqqq_ret + (1 - sig) * rf_daily

    # 수수료: 시그널 변동 시 0.2%
    trades = sig.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * COMMISSION

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, sig_raw


def backtest_s1(tqqq: pd.Series, rf: pd.Series) -> tuple:
    """김째 S1: TQQQ 200MA 크로스 백테스트.

    Returns: (cum, weight_series)
    """
    _, w = compute_s1_tqqq_200ma_cross(tqqq.values.astype(float))
    weight_series = pd.Series(w, index=tqqq.index)
    weight_lagged = weight_series.shift(SIGNAL_LAG).fillna(0)

    tqqq_ret = tqqq.pct_change().fillna(0)
    rf_daily = rf.reindex(tqqq_ret.index, method="ffill").fillna(0)

    strat_ret = weight_lagged * tqqq_ret + (1 - weight_lagged) * rf_daily
    trades = weight_lagged.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * COMMISSION

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, weight_series


def backtest_kimjje(tqqq: pd.Series, qqq: pd.Series, spy: pd.Series,
                     rf: pd.Series, spy_bear_cap: float = 0.0,
                     params: BasicParams = None) -> tuple:
    """김째매매법 백테스트 (실제 TQQQ 수익률, 비중 비례 수수료).

    spy_bear_cap: 0.0=S2(전량청산), 0.10=S3(10% 유지). params 미지정 시 사용.
    params: 직접 BasicParams를 전달하면 spy_bear_cap 무시.
    Returns: (cum, weights) — 누적 수익률, 비중 시리즈
    """
    if params is None:
        params = BasicParams(spy_bear_cap=spy_bear_cap)
    _, w = compute_kimjje_strategy(
        qqq.values.astype(float),
        tqqq.values.astype(float),
        spy.values.astype(float),
        params,
    )
    weight_series = pd.Series(w, index=tqqq.index)

    # lag=1: 오늘 시그널 → 내일 적용
    weight_lagged = weight_series.shift(SIGNAL_LAG).fillna(0)

    tqqq_ret = tqqq.pct_change().fillna(0)
    rf_daily = rf.reindex(tqqq_ret.index, method="ffill").fillna(0)

    strat_ret = weight_lagged * tqqq_ret + (1 - weight_lagged) * rf_daily

    # 수수료: 비중 변동분 × 0.2%
    weight_change = weight_lagged.diff().abs().fillna(0)
    strat_ret = strat_ret - weight_change * COMMISSION

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, weight_series


def backtest_buy_and_hold(tqqq: pd.Series) -> pd.Series:
    """TQQQ Buy & Hold 누적 수익률."""
    daily_ret = tqqq.pct_change().fillna(0)
    cum = (1 + daily_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum


# ══════════════════════════════════════════════════════════════
# [8] 메트릭 계산
# ══════════════════════════════════════════════════════════════

def calc_metrics(cum: pd.Series, rf: pd.Series = None) -> dict:
    """핵심 성과 지표 계산."""
    daily_ret = cum.pct_change().dropna()
    n_days = len(daily_ret)
    n_years = n_days / 252

    total_ret = cum.iloc[-1] / cum.iloc[0]
    cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0

    if rf is not None:
        rf_aligned = rf.reindex(daily_ret.index, method="ffill").fillna(0)
        avg_annual_rf = rf_aligned.mean() * 252
        rf_daily = rf_aligned
    else:
        avg_annual_rf = 0.03
        rf_daily = avg_annual_rf / 252

    arith_annual = daily_ret.mean() * 252
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (arith_annual - avg_annual_rf) / vol if vol > 0 else 0

    excess_daily = daily_ret - rf_daily
    downside_diff = excess_daily.copy()
    downside_diff[downside_diff > 0] = 0.0
    downside_dev = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(252)
    sortino = (arith_annual - avg_annual_rf) / downside_dev if downside_dev > 0 else 0

    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    mdd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Total Return": total_ret,
    }


def count_trades(sig: pd.Series, is_binary: bool = True) -> float:
    """연간 거래 횟수 계산."""
    if is_binary:
        flips = (sig.diff().abs() > 0).sum()
    else:
        # 비중 전략: 비중 변화가 있는 날
        flips = (sig.diff().abs() > 1e-6).sum()
    n_years = len(sig) / 252
    return flips / n_years if n_years > 0 else 0


# ══════════════════════════════════════════════════════════════
# [9] 시각화
# ══════════════════════════════════════════════════════════════

def plot_comparison(curves: dict, title: str, fname: str):
    """Equity curve + Drawdown 2-panel 비교 차트."""
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    colors = {
        "Regime-Switching": "#2196F3",
        "김째 S1 (200MA)": "#8BC34A",
        "김째 S2": "#FF5722",
        "김째 S3": "#FF9800",
        "TQQQ B&H": "#9E9E9E",
    }
    styles = {
        "TQQQ B&H": {"lw": 1.2, "ls": "--", "alpha": 0.5},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.08})

    # Equity curves (log scale)
    for name, cum in curves.items():
        color = colors.get(name, "#333333")
        s = styles.get(name, {"lw": 2.0, "ls": "-", "alpha": 1.0})
        ax1.plot(cum.index, cum.values, label=name, color=color,
                 linewidth=s["lw"], linestyle=s["ls"], alpha=s["alpha"])

    ax1.set_yscale("log")
    ax1.set_ylabel("누적 수익률 (log)")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Drawdowns
    for name, cum in curves.items():
        color = colors.get(name, "#333333")
        dd = cum / cum.cummax() - 1
        s = styles.get(name, {"lw": 2.0, "ls": "-", "alpha": 0.8})
        a = min(s["alpha"], 0.8)
        ax2.fill_between(dd.index, dd.values, 0, alpha=a * 0.3, color=color)
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.8, alpha=a)

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  차트 저장: {path}")


# ══════════════════════════════════════════════════════════════
# [10] 오늘의 시그널
# ══════════════════════════════════════════════════════════════

def show_todays_signal(data: dict, regime_sig: pd.Series,
                        s1_weights: pd.Series,
                        s2_weights: pd.Series,
                        s3_weights: pd.Series):
    """모든 전략의 최신 시그널 상태를 출력."""
    last_date = data["qqq"].index[-1].date()
    qqq_price = data["qqq"].iloc[-1]
    tqqq_price = data["tqqq"].iloc[-1]
    spy_price = data["spy"].iloc[-1]

    print("\n" + "=" * 60)
    print(f"오늘의 시그널 ({last_date})")
    print("=" * 60)
    print(f"  QQQ:  ${qqq_price:.2f}")
    print(f"  TQQQ: ${tqqq_price:.2f}")
    print(f"  SPY:  ${spy_price:.2f}")

    # Regime-Switching
    r_sig = regime_sig.iloc[-1]
    r_status = "매수 (TQQQ 100%)" if r_sig == 1 else "관망 (현금)"
    print(f"\n  [전략 1] Regime-Switching: {r_status}")

    # 김째 S1
    s1_w = s1_weights.iloc[-1]
    print(f"  [전략 2] 김째 S1 (200MA):  TQQQ {s1_w * 100:.0f}%")

    # 김째 S2
    s2_w = s2_weights.iloc[-1]
    print(f"  [전략 3] 김째 S2:          TQQQ {s2_w * 100:.0f}%")

    # 김째 S3
    s3_w = s3_weights.iloc[-1]
    print(f"  [전략 4] 김째 S3:          TQQQ {s3_w * 100:.0f}%")

    print(f"\n  → 모두 내일 적용 (lag=1)")
    print()


# ══════════════════════════════════════════════════════════════
# [11] Main
# ══════════════════════════════════════════════════════════════

def run_comparison(data: dict, compare_start: str, label: str, chart_fname: str):
    """백테스트 + 메트릭 + 차트 한 세트 실행."""
    print("\n" + "=" * 60)
    print(f"백테스트 — {label}")
    print("=" * 60)

    cum_regime, sig_regime = backtest_regime(data["tqqq"], data["qqq"], data["rf"])
    print("  [1/5] Regime-Switching")

    cum_s1, w_s1 = backtest_s1(data["tqqq"], data["rf"])
    print("  [2/5] 김째 S1 (200MA)")

    cum_s2, w_s2 = backtest_kimjje(data["tqqq"], data["qqq"], data["spy"], data["rf"],
                                    spy_bear_cap=0.0)
    print("  [3/5] 김째 S2")

    cum_s3, w_s3 = backtest_kimjje(data["tqqq"], data["qqq"], data["spy"], data["rf"],
                                    spy_bear_cap=0.10)
    print("  [4/5] 김째 S3")

    cum_bh = backtest_buy_and_hold(data["tqqq"])
    print("  [5/5] TQQQ B&H")

    # 비교 기간 트림
    start_mask = cum_regime.index >= compare_start
    all_cums = {
        "Regime-Switching": cum_regime,
        "김째 S1 (200MA)": cum_s1,
        "김째 S2": cum_s2,
        "김째 S3": cum_s3,
        "TQQQ B&H": cum_bh,
    }
    trimmed = {}
    for name, cum in all_cums.items():
        c = cum.loc[start_mask]
        trimmed[name] = c / c.iloc[0]

    rf_t = data["rf"].loc[start_mask]
    period_str = f"{trimmed['Regime-Switching'].index[0].date()} → {trimmed['Regime-Switching'].index[-1].date()}"
    print(f"\n  비교 기간: {period_str}")

    # 메트릭
    print("\n" + "-" * 60)
    print(f"성과 비교 — {label}")
    print("-" * 60)

    metrics = {}
    for name, cum in trimmed.items():
        metrics[name] = calc_metrics(cum, rf_t)

    all_sigs = {
        "Regime-Switching": (sig_regime.loc[start_mask], True),
        "김째 S1 (200MA)": (w_s1.loc[start_mask], False),
        "김째 S2": (w_s2.loc[start_mask], False),
        "김째 S3": (w_s3.loc[start_mask], False),
        "TQQQ B&H": (None, True),
    }
    for name, (sig, is_bin) in all_sigs.items():
        if sig is not None:
            metrics[name]["Trades/Yr"] = count_trades(sig, is_binary=is_bin)
        else:
            metrics[name]["Trades/Yr"] = 0.0

    header = f"{'':20s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} {'Trades/Y':>9s}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:20s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} {m['Sortino']:>8.2f} "
              f"{m['MDD']:>7.1%} {m.get('Trades/Yr', 0):>8.1f}")

    # 차트
    plot_comparison(trimmed, f"{label} ({period_str})", chart_fname)

    return sig_regime, w_s1, w_s2, w_s3


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  LRS Standalone — 프로덕션 전략 비교                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── 데이터 다운로드 (3종) ──
    data_idx = download_all_data(mode="index")
    data_synth = download_all_data(mode="synth")
    data_real = download_all_data(mode="real")

    # ── A) 최장 기간: NDX만 가능 ──
    run_comparison(data_idx, compare_start="1986-01-01",
                   label="합성 3x NDX (최장 40년)",
                   chart_fname="lrs_standalone_index.png")

    # ── B) 26년 비교: NDX vs QQQ 합성 ──
    run_comparison(data_idx, compare_start="2000-01-01",
                   label="합성 3x NDX (26년)",
                   chart_fname="lrs_standalone_ndx_26y.png")
    run_comparison(data_synth, compare_start="2000-01-01",
                   label="합성 3x QQQ (26년)",
                   chart_fname="lrs_standalone_qqq_26y.png")

    # ── C) 15년 비교: NDX vs QQQ vs 실제 TQQQ ──
    run_comparison(data_idx, compare_start=COMPARE_START,
                   label="합성 3x NDX (15년)",
                   chart_fname="lrs_standalone_ndx_15y.png")
    run_comparison(data_synth, compare_start=COMPARE_START,
                   label="합성 3x QQQ (15년)",
                   chart_fname="lrs_standalone_qqq_15y.png")
    sig_regime, w_s1, w_s2, w_s3 = run_comparison(
        data_real, compare_start=COMPARE_START,
        label="실제 TQQQ",
        chart_fname="lrs_standalone_comparison.png")

    # ── 오늘의 시그널 (실제 데이터 기준) ──
    show_todays_signal(data_real, sig_regime, w_s1, w_s2, w_s3)

    print("완료!")
