"""
LRS Standalone v2 — DFF 파이낸싱 + 미국 세금 + 다단계 TP

v1 대비 변경점:
  - Phase 1: FRED DFF 기반 동적 파이낸싱 비용 (고정 ER 3.5% → 실제 기준금리 연동)
  - Phase 2: 미국 자본이득세 시뮬레이션 (단기 37% / 장기 20%, 손실이월)
  - Phase 3: 다단계 TP 래더 + 트레일링 스탑
  - Phase 4: v1 vs v2 검증 및 비교

실행:
    python lrs_standalone_v2.py
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
from dataclasses import dataclass, field
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
CALIBRATED_ER = 0.035   # 합성 TQQQ 연간 비용 (v1 호환용)

# DFF 파이낸싱 파라미터 (Phase 1)
BORROW_SPREAD = 0.01    # 1% 차입 스프레드
STATED_ER = 0.0095      # TQQQ 표기 ER (0.95%)

# 미국 세금 파라미터 (Phase 2)
# 단기 자본이득 = 일반 소득 합산 (누진세율), 장기 = 별도 우대세율
BASE_ANNUAL_INCOME = 100_000  # 연간 노동소득 가정 ($100k)
INITIAL_PORTFOLIO = 500_000   # 초기 투자금 ($500k) — 실현이익 달러 환산 기준
TAX_ENABLED = True            # on/off 토글

# 2025 Federal Income Tax Brackets (Single)
# 단기 자본이득은 여기에 합산
ORDINARY_BRACKETS = [
    (11_925,   0.10),
    (48_475,   0.12),
    (103_350,  0.22),
    (197_300,  0.24),
    (250_525,  0.32),
    (626_350,  0.35),
    (float("inf"), 0.37),
]

# 2025 Long-Term Capital Gains Brackets (Single)
# 구간 기준은 총 과세소득 (일반소득 + LTCG)
LTCG_BRACKETS = [
    (48_350,   0.00),
    (533_400,  0.15),
    (float("inf"), 0.20),
]

# TQQQ 200일선 워밍업 필요 → 2011-01-01부터 비교 시작
COMPARE_START = "2011-01-01"


@dataclass(frozen=True)
class BasicParams:
    """김째매매법 Pine v6 파라미터 (Kim 전략 기본값)."""
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
    spy_bear_cap: float = 0.0      # 하락 시 전량 청산
    # Phase 3: TP 래더 (기존 TP10과 하위호환)
    use_tp10: bool = True
    tp10_trigger: float = 0.10
    tp10_cap: float = 0.95
    tp_ladder: tuple = ((0.10, 0.05),)  # (trigger_pct, sell_frac)
    trailing_stop_enabled: bool = False
    trailing_stop_drawdown: float = 0.15   # 고점 대비 -15%
    trailing_stop_after_tp: bool = True     # 첫 TP 이후에만 활성화


@dataclass(frozen=True)
class SnowballParams:
    """Snowball 전략 파라미터 (snowball.txt 기준)."""
    short_ma: int = 5
    long_ma: int = 220
    dip_lookback: int = 252
    dip1_limit: float = 10.0       # QQQ 낙폭 -10% → Dip1
    dip2_limit: float = 22.0       # QQQ 낙폭 -22% → Dip2
    max_dip_limit: float = 40.0    # QQQ 낙폭 -40% → 매수 차단
    cooldown_days: int = 5         # 매도 후 5일간 매수 차단
    rsi_bonus_level: float = 35.0  # RSI(14) < 35 → 보너스
    rsi_bonus_pct: float = 10.0    # +10% 추가 매수
    tp1_trigger: float = 15.0      # +15% → 초기의 50% 매도
    tp1_sell_pct: float = 50.0
    tp2_trigger: float = 68.0      # +68% → 초기의 35% 매도
    tp2_sell_pct: float = 35.0
    tp3_trigger: float = 350.0     # +350% → 전량 매도
    safe_factor: float = 0.95      # 매수 시 현금의 95%만 사용
    commission_rate: float = 0.001  # 0.1% 수수료 (시뮬레이션 내부)


@dataclass(frozen=True)
class EnsembleParams:
    """Kim + Snowball 앙상블 파라미터."""
    alpha: float = 0.5
    kim_params: BasicParams = field(default_factory=BasicParams)
    snow_params: SnowballParams = field(default_factory=SnowballParams)



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


# ── Phase 1: FRED DFF 파이낸싱 모델 ──

_DFF_CACHE = None

def download_fred_dff() -> pd.Series:
    """FRED에서 일별 Federal Funds Rate (DFF) 다운로드.
    Returns: pd.Series (date index, 값은 percent, 예: 5.33)"""
    global _DFF_CACHE
    if _DFF_CACHE is not None:
        return _DFF_CACHE

    url = "https://fred.stlouisfed.org/data/DFF.txt"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        raw = resp.read().decode("utf-8")

        dates = []
        values = []
        for line in raw.splitlines():
            line = line.strip()
            # FRED embeds data as "#YYYY-MM-DD|value" inside HTML
            if not line.startswith("#"):
                continue
            content = line[1:]  # strip leading #
            if "|" not in content:
                continue
            try:
                raw_date, raw_value = content.split("|", 1)
                dt = pd.to_datetime(raw_date.strip())
                val = float(raw_value.strip())
                dates.append(dt)
                values.append(val)
            except (ValueError, TypeError):
                continue

        if not dates:
            raise RuntimeError("No DFF data parsed")

        dff = pd.Series(values, index=pd.DatetimeIndex(dates), name="DFF")
        dff = dff.sort_index()
        _DFF_CACHE = dff
        print(f"  [FRED DFF] {dff.index[0].date()} → {dff.index[-1].date()} ({len(dff)}일)")
        return dff
    except Exception as e:
        print(f"  [Warning] FRED DFF 다운로드 실패: {e}, 2.0% fallback 사용")
        idx = pd.date_range("1954-01-01", "2030-12-31", freq="B")
        _DFF_CACHE = pd.Series(2.0, index=idx, name="DFF")
        return _DFF_CACHE


def _calendar_days(index: pd.DatetimeIndex) -> pd.Series:
    """각 거래일 간 실제 경과일수 (주말/공휴일 포함).
    금→월 = 3일, 일반 = 1일. 첫 날은 1."""
    delta = pd.Series(index, index=index).diff().dt.days.fillna(1).astype(float)
    return delta


def synthesize_tqqq(qqq: pd.Series, leverage: float = 3.0,
                     expense_ratio: float = CALIBRATED_ER) -> pd.Series:
    """QQQ 수익률로 합성 TQQQ 가격 시리즈 생성 (고정 ER — v1 호환).
    연속복리 + calendar days + 음수 방어."""
    qqq_ret = qqq.pct_change().fillna(0)
    cal_days = _calendar_days(qqq.index)

    gross = np.maximum(1e-12, 1.0 + leverage * qqq_ret)
    cost_mult = np.exp(-expense_ratio * cal_days / 360.0)
    synth_price = (gross * cost_mult).cumprod() * 100
    synth_price.iloc[0] = 100.0
    synth_price.name = "TQQQ_synth"
    return synth_price


def synthesize_tqqq_dff(qqq: pd.Series, dff: pd.Series,
                         leverage: float = 3.0,
                         borrow_spread: float = BORROW_SPREAD,
                         stated_er: float = STATED_ER) -> pd.Series:
    """DFF 기반 동적 파이낸싱 비용으로 합성 TQQQ 생성.

    연속복리: gross × exp(-annual_rate × cal_days / 360)
    음수 방어: gross = max(1e-12, 1 + lev × ret)
    """
    qqq_ret = qqq.pct_change().fillna(0)
    cal_days = _calendar_days(qqq.index)

    # DFF를 QQQ 인덱스에 맞춤 (ffill — 주말/공휴일 이전 값 사용)
    dff_aligned = dff.reindex(qqq.index, method="ffill").fillna(2.0)

    # 연간 파이낸싱 비율
    annual_rate = (leverage - 1) * (dff_aligned / 100.0 + borrow_spread) + stated_er

    gross = np.maximum(1e-12, 1.0 + leverage * qqq_ret)
    cost_mult = np.exp(-annual_rate * cal_days / 360.0)
    synth_price = (gross * cost_mult).cumprod() * 100
    synth_price.iloc[0] = 100.0
    synth_price.name = "TQQQ_synth_dff"
    return synth_price


def download_all_data(mode: str = "real", use_dff: bool = False) -> dict:
    """데이터 다운로드.

    mode:
      "real"  — 실제 QQQ/TQQQ/SPY (2010~)
      "synth" — QQQ(1999~) 기반 합성 TQQQ
      "index" — NDX/^GSPC(1985~) 기반 합성 TQQQ (최장 기간)
    use_dff: True → DFF 기반 동적 파이낸싱, False → 고정 ER 3.5%
    Returns: dict with aligned pd.Series (공통 인덱스)."""
    labels = {"real": "실제 TQQQ", "synth": "합성 3x QQQ (1999~)",
              "index": "합성 3x NDX (1985~)"}
    dff_label = " [DFF]" if use_dff and mode != "real" else ""
    print("=" * 60)
    print(f"데이터 다운로드 — {labels[mode]}{dff_label}")
    print("=" * 60)

    rf = download_ken_french_rf()
    dff = download_fred_dff() if use_dff else None

    if mode == "real":
        qqq = download("QQQ", start="2009-01-01")
        tqqq = download("TQQQ", start="2009-01-01")
        spy = download("SPY", start="2009-01-01")
        common = qqq.index.intersection(tqqq.index).intersection(spy.index)
    elif mode == "synth":
        qqq = download("QQQ", start="1998-01-01")
        spy = download("SPY", start="1998-01-01")
        if use_dff and dff is not None:
            tqqq = synthesize_tqqq_dff(qqq, dff)
            print(f"  합성 TQQQ: QQQ × 3 − DFF 동적비용 (spread={BORROW_SPREAD:.0%}, ER={STATED_ER:.2%})")
        else:
            tqqq = synthesize_tqqq(qqq)
            print(f"  합성 TQQQ: QQQ × 3 − ER {CALIBRATED_ER:.1%}/yr")
        common = qqq.index.intersection(spy.index)
    else:  # index
        qqq = download("^NDX", start="1984-01-01")
        spy = download("^GSPC", start="1984-01-01")
        if use_dff and dff is not None:
            tqqq = synthesize_tqqq_dff(qqq, dff)
            print(f"  합성 TQQQ: NDX × 3 − DFF 동적비용 (spread={BORROW_SPREAD:.0%}, ER={STATED_ER:.2%})")
        else:
            tqqq = synthesize_tqqq(qqq)
            print(f"  합성 TQQQ: NDX × 3 − ER {CALIBRATED_ER:.1%}/yr")
        print(f"  QQQ 대용: ^NDX, SPY 대용: ^GSPC")

        common = qqq.index.intersection(spy.index)

    qqq = qqq.loc[common]
    tqqq = tqqq.reindex(common).loc[common]
    spy = spy.loc[common]
    rf = rf.reindex(common, method="ffill").fillna(0)

    result = {"qqq": qqq, "tqqq": tqqq, "spy": spy, "rf": rf}
    if dff is not None:
        result["dff"] = dff.reindex(common, method="ffill").fillna(2.0)

    print(f"  공통 기간: {common[0].date()} → {common[-1].date()} ({len(common)}일)")
    return result


# ══════════════════════════════════════════════════════════════
# [4] 김째매매법 헬퍼 함수
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

    # Phase 3: TP 래더 상태 (기존 tp10 변수 대체)
    tp_entry_close = math.nan
    tp_triggered = set()           # 발동된 래더 인덱스
    tp_cycle_active = False
    trailing_active = False
    trailing_peak = math.nan

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
            tp_entry_close = math.nan
            tp_triggered = set()
            tp_cycle_active = False
            trailing_active = False
            trailing_peak = math.nan
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

        # 10) TP 래더 + 트레일링 스탑 (Phase 3)
        wFinalPreTp = code_weight(finalCode)
        inHighNow = wFinalPreTp >= (0.80 - 1e-9)

        if (not params.use_tp10) or (not inHighNow) or locked or stopHit or spyBearNow or reentryBlockedNow:
            # TP 사이클 비활성화
            tp_entry_close = math.nan
            tp_triggered = set()
            tp_cycle_active = False
            trailing_active = False
            trailing_peak = math.nan
        else:
            # 새 100% 진입 감지
            entered100 = (finalCode == 2) and (code_weight(prevAsset) < 0.999) and (not tp_cycle_active)
            if entered100:
                tp_cycle_active = True
                tp_entry_close = float(tqqq_close[i])
                tp_triggered = set()
                trailing_active = False
                trailing_peak = math.nan

            # 래더 순회
            if tp_cycle_active and (not math.isnan(tp_entry_close)) and (finalCode == 2):
                current_weight = code_weight(finalCode)
                for idx_tp, (trigger_pct, sell_frac) in enumerate(params.tp_ladder):
                    if idx_tp in tp_triggered:
                        continue
                    if tqqq_close[i] >= tp_entry_close * (1.0 + trigger_pct):
                        tp_triggered.add(idx_tp)
                        new_weight = current_weight * (1.0 - sell_frac)
                        finalCode = weight_to_code(new_weight)
                        current_weight = code_weight(finalCode)
                        # 첫 TP 발동 시 트레일링 활성화
                        if params.trailing_stop_enabled and params.trailing_stop_after_tp:
                            trailing_active = True
                            trailing_peak = float(tqqq_close[i])

            # 트레일링 스탑 (TP 체크 후)
            if params.trailing_stop_enabled and trailing_active:
                if math.isnan(trailing_peak):
                    trailing_peak = float(tqqq_close[i])
                else:
                    trailing_peak = max(trailing_peak, float(tqqq_close[i]))
                if tqqq_close[i] <= trailing_peak * (1.0 - params.trailing_stop_drawdown):
                    finalCode = 0  # 전량 청산
                    tp_entry_close = math.nan
                    tp_triggered = set()
                    tp_cycle_active = False
                    trailing_active = False
                    trailing_peak = math.nan

            # trailing_stop_after_tp=False이면 포지션 진입 즉시 활성화
            if params.trailing_stop_enabled and (not params.trailing_stop_after_tp) and tp_cycle_active:
                if not trailing_active:
                    trailing_active = True
                    trailing_peak = float(tqqq_close[i])

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


def compute_200ma_cross(tqqq_close: np.ndarray) -> tuple:
    """200MA 전략: 3일 연속 상향돌파 확인 매수, 즉시 하향이탈 매도.

    매수: TQQQ > 200MA가 3거래일 연속 유지 시, 3일차 종가에 매수
    매도: TQQQ < 200MA 즉시 매도 (확인 없음)

    Returns: (codes, weights) — 각각 np.ndarray
    """
    n = len(tqqq_close)
    t_ma200 = pd.Series(tqqq_close).rolling(200, min_periods=200).mean().to_numpy()
    code = np.zeros(n, dtype=int)
    weight = np.zeros(n, dtype=float)
    in_pos = False
    above_count = 0  # 연속 상향돌파 일수

    for i in range(n):
        if np.isnan(tqqq_close[i]) or np.isnan(t_ma200[i]):
            in_pos = False
            above_count = 0
            code[i] = 0
            weight[i] = 0.0
            continue

        above = tqqq_close[i] > t_ma200[i]

        if not in_pos:
            if above:
                above_count += 1
                if above_count >= 3:
                    in_pos = True
            else:
                above_count = 0
        else:
            if not above:
                in_pos = False
                above_count = 0

        code[i] = 2 if in_pos else 0
        weight[i] = 1.0 if in_pos else 0.0

    return code, weight


def compute_snowball_strategy(
    tqqq_close: np.ndarray,
    qqq_close: np.ndarray,
    params: SnowballParams = None,
) -> tuple:
    """Snowball 전략: TQQQ MA5/MA220 크로스 + QQQ 낙폭 단계 매수 + TP 래더.

    가상 포트폴리오(cash/shares) 시뮬레이션 → 일별 weight 출력.
    수수료는 시뮬레이션 내부에서 처리 (외부 backtest에서 COMMISSION=0).

    Returns: (weights,) — np.ndarray (float, 0.0~1.0)
    """
    if params is None:
        params = SnowballParams()

    n = len(tqqq_close)
    weights = np.zeros(n, dtype=float)

    # TQQQ MA 계산
    ma_short = pd.Series(tqqq_close).rolling(params.short_ma, min_periods=params.short_ma).mean().to_numpy()
    ma_long = pd.Series(tqqq_close).rolling(params.long_ma, min_periods=params.long_ma).mean().to_numpy()

    # QQQ 252일 롤링 고가 → 낙폭 시리즈
    qqq_rolling_high = pd.Series(qqq_close).rolling(params.dip_lookback, min_periods=1).max().to_numpy()
    qqq_dd = np.where(qqq_rolling_high > 0,
                      (qqq_close / qqq_rolling_high - 1.0) * 100.0,
                      0.0)  # percent, 예: -15.0

    # RSI(14) — 기존 rsi_wilder 재사용
    rsi_vals = rsi_wilder(tqqq_close, 14)

    # 가상 포트폴리오
    cash = 10000.0
    position_size = 0.0  # 주 수 (float)
    avg_price = 0.0
    stage = 0
    tp3_lock = False
    tp3_exit_pending = False
    initial_position_size = 0.0
    last_exit_idx = -999
    tp_sold = [False, False, False]

    cr = params.commission_rate
    sf = params.safe_factor

    def do_buy(qty_target, close_price):
        nonlocal cash, position_size, avg_price
        max_qty = math.floor(cash / (close_price * (1.0 + cr)))
        fill_qty = min(math.floor(qty_target), max_qty)
        if fill_qty <= 0:
            return 0
        notional = fill_qty * close_price
        commission = notional * cr
        cash -= notional + commission
        if position_size == 0:
            avg_price = close_price
        else:
            avg_price = (position_size * avg_price + fill_qty * close_price) / (position_size + fill_qty)
        position_size += fill_qty
        return fill_qty

    def do_sell(qty_target, close_price):
        nonlocal cash, position_size, avg_price
        fill_qty = min(math.floor(qty_target), math.floor(position_size))
        if fill_qty <= 0:
            return 0
        notional = fill_qty * close_price
        commission = notional * cr
        cash += notional - commission
        position_size -= fill_qty
        if position_size <= 0:
            position_size = 0.0
            avg_price = 0.0
        return fill_qty

    for i in range(n):
        close = tqqq_close[i]
        ms = ma_short[i]
        ml = ma_long[i]
        prev_ms = ma_short[i - 1] if i > 0 else np.nan
        prev_ml = ma_long[i - 1] if i > 0 else np.nan
        dd = qqq_dd[i]
        rsi_val = rsi_vals[i]

        # 크로스 감지
        trend_bullish = (i > 0 and not np.isnan(ms) and not np.isnan(ml)
                         and not np.isnan(prev_ms) and not np.isnan(prev_ml)
                         and ms > ml and prev_ms <= prev_ml)
        trend_bearish = (i > 0 and not np.isnan(ms) and not np.isnan(ml)
                         and not np.isnan(prev_ms) and not np.isnan(prev_ml)
                         and ms < ml and prev_ms >= prev_ml)

        # tp3Lock 활성화
        if position_size == 0 and tp3_exit_pending:
            tp3_lock = True
            tp3_exit_pending = False

        # 포지션 없을 때 상태 리셋
        if position_size == 0:
            if stage > 0 and not trend_bearish:
                stage = 0
            tp_sold = [False, False, False]
            initial_position_size = 0.0

        bars_since_exit = i - last_exit_idx
        cooldown_active = bars_since_exit <= params.cooldown_days
        max_dip_active = dd <= -params.max_dip_limit

        cond_trend = trend_bullish and not cooldown_active and not max_dip_active
        cond_dip1 = dd <= -params.dip1_limit and not cooldown_active and not max_dip_active
        cond_dip2 = dd <= -params.dip2_limit and not cooldown_active and not max_dip_active

        total_equity = cash + position_size * close
        current_cash = cash

        # TP 체크 (포지션 보유 시)
        if position_size > 0:
            profit_pct = (close / avg_price - 1.0) * 100.0
            if initial_position_size == 0:
                initial_position_size = position_size

            tp_list = [
                (params.tp1_trigger, params.tp1_sell_pct),
                (params.tp2_trigger, params.tp2_sell_pct),
                (params.tp3_trigger, 100.0),
            ]
            for j, (trigger, sell_pct) in enumerate(tp_list):
                if tp_sold[j]:
                    continue
                if profit_pct >= trigger:
                    qty = position_size if j == 2 else math.floor(initial_position_size * (sell_pct / 100.0))
                    if do_sell(qty, close) > 0:
                        tp_sold[j] = True
                        if j == 2:
                            tp3_exit_pending = True

        # DC 매도
        if trend_bearish:
            stage = 0
            last_exit_idx = i
            tp_sold = [False, False, False]
            initial_position_size = 0.0
            if position_size > 0:
                do_sell(position_size, close)
        else:
            # RSI 보너스
            bonus_pct = params.rsi_bonus_pct if (not np.isnan(rsi_val) and rsi_val < params.rsi_bonus_level) else 0.0

            if stage == 0:
                if cond_trend and not tp3_lock:
                    eff_pct = min(100.0, 100.0 + bonus_pct)
                    qty = math.floor(current_cash * (eff_pct / 100.0) * sf / close)
                    if do_buy(qty, close) > 0:
                        stage = 3
                elif cond_dip1:
                    eff_pct = min(100.0, 20.0 + bonus_pct)
                    qty = math.floor(total_equity * (eff_pct / 100.0) * sf / close)
                    if do_buy(qty, close) > 0:
                        stage = 1
                        tp3_lock = False
            elif stage == 1:
                if cond_trend:
                    eff_pct = min(100.0, 100.0 + bonus_pct)
                    qty = math.floor(current_cash * (eff_pct / 100.0) * sf / close)
                    if do_buy(qty, close) > 0:
                        stage = 3
                        initial_position_size = position_size
                elif cond_dip2:
                    eff_pct = min(100.0, 70.0 + bonus_pct)
                    target_shares = math.floor(total_equity * (eff_pct / 100.0) * sf / close)
                    qty_need = max(0, target_shares - math.floor(position_size))
                    max_qty = math.floor(current_cash * sf / close)
                    qty = math.floor(min(qty_need, max_qty))
                    if do_buy(qty, close) > 0:
                        stage = 2
                        initial_position_size = position_size
            elif stage == 2:
                if cond_trend:
                    eff_pct = min(100.0, 100.0 + bonus_pct)
                    qty = math.floor(current_cash * (eff_pct / 100.0) * sf / close)
                    if do_buy(qty, close) > 0:
                        stage = 3
                        initial_position_size = position_size

        # weight 기록
        total_eq = cash + position_size * close
        if total_eq > 1e-10:
            weights[i] = (position_size * close) / total_eq
        else:
            weights[i] = 0.0

    return weights


def compute_ensemble_strategy(
    qqq_close: np.ndarray,
    tqqq_close: np.ndarray,
    spy_close: np.ndarray,
    params: EnsembleParams = None,
) -> np.ndarray:
    """Kim + Snowball 앙상블: alpha × Kim + (1-alpha) × Snowball.

    Returns: weights (np.ndarray, 0.0~1.0)
    """
    if params is None:
        params = EnsembleParams()

    _, kim_w = compute_kimjje_strategy(qqq_close, tqqq_close, spy_close,
                                        params.kim_params)
    snow_w = compute_snowball_strategy(tqqq_close, qqq_close, params.snow_params)

    weights = params.alpha * kim_w + (1.0 - params.alpha) * snow_w
    return np.clip(weights, 0.0, 1.0)



# ══════════════════════════════════════════════════════════════
# [7] 백테스트 엔진 (Phase 2: 세금 통합)
# ══════════════════════════════════════════════════════════════

def _calc_progressive_tax(base_income: float, short_gains: float,
                           long_gains: float) -> tuple:
    """누진세율 적용 세금 계산.

    단기 자본이득: 일반 소득에 합산 → 누진세율
    장기 자본이득: 별도 우대세율 (총 과세소득 기준 구간 결정)

    Returns: (tax_on_short, tax_on_long)
    """
    # 1) 단기 자본이득세: base_income 위에 쌓아서 한계세율 적용
    #    base_income에 대한 세금은 이미 낸 것이므로, 추가분만 계산
    taxable_short = max(short_gains, 0.0)
    tax_short = 0.0
    if taxable_short > 0:
        # base_income까지의 세금 (기준선)
        base_tax = _ordinary_tax(base_income)
        # base + short_gains 전체 세금
        total_tax = _ordinary_tax(base_income + taxable_short)
        # 차이 = 단기 이득에 대한 세금
        tax_short = total_tax - base_tax

    # 2) 장기 자본이득세: 총 과세소득 기준으로 LTCG 구간 결정
    taxable_long = max(long_gains, 0.0)
    tax_long = 0.0
    if taxable_long > 0:
        # 일반소득 (base + short gains)이 차지한 후 나머지 구간에 LTCG 적용
        ordinary_total = base_income + max(short_gains, 0.0)
        tax_long = _ltcg_tax(ordinary_total, taxable_long)

    return tax_short, tax_long


def _ordinary_tax(income: float) -> float:
    """누진세율로 일반소득 세금 계산."""
    tax = 0.0
    prev_limit = 0.0
    for limit, rate in ORDINARY_BRACKETS:
        taxable_in_bracket = min(income, limit) - prev_limit
        if taxable_in_bracket <= 0:
            break
        tax += taxable_in_bracket * rate
        prev_limit = limit
    return tax


def _ltcg_tax(ordinary_income: float, ltcg: float) -> float:
    """장기 자본이득세 계산. 총 소득 기준 구간 결정."""
    tax = 0.0
    remaining = ltcg
    # LTCG는 일반소득 '위에' 쌓임
    income_so_far = ordinary_income
    for limit, rate in LTCG_BRACKETS:
        if remaining <= 0:
            break
        room = max(limit - income_so_far, 0.0)
        taxable = min(remaining, room)
        if taxable > 0:
            tax += taxable * rate
            remaining -= taxable
            income_so_far += taxable
    return tax


def _apply_tax(strat_ret: pd.Series, weight_lagged: pd.Series,
               tax_enabled: bool = True,
               base_income: float = BASE_ANNUAL_INCOME,
               initial_portfolio: float = INITIAL_PORTFOLIO) -> tuple:
    """실현손익 추적 + 연간 세금 정산 (누진세율, 달러 기준).

    실현이익을 initial_portfolio 기준 달러로 환산 → 누진세율 적용.
    예: 포트 비율 0.05 이익, $500k 포트 → $25k 실현이익 → 누진세 계산.

    Returns: (strat_ret_after_tax, tax_info_dict)
    """
    if not tax_enabled:
        return strat_ret.copy(), {"total_tax_paid": 0.0, "annual_taxes": {}}

    n = len(strat_ret)
    ret_arr = strat_ret.values.copy()
    dates = strat_ret.index

    # 포트폴리오 가치 추적 (비율 기준, 시작=1.0)
    portfolio = 1.0
    entry_price = math.nan       # 가중평균 진입 시점의 포트 비율
    entry_date_idx = 0           # 진입일 인덱스 (보유기간 계산용)

    # 연간 실현손익 (비율 단위 — 나중에 달러 환산)
    current_year = dates[0].year if n > 0 else 0
    short_gains_ratio = 0.0      # 포트 비율 단위 단기 실현이익
    long_gains_ratio = 0.0       # 포트 비율 단위 장기 실현이익
    loss_cf_ratio = 0.0          # 손실이월 (비율 단위)
    total_tax_paid_dollars = 0.0
    annual_taxes = {}

    for i in range(n):
        yr = dates[i].year

        # 연간 정산: 새해 첫 거래일
        if yr != current_year:
            # 현재 포트 달러 가치로 실현이익 환산
            portfolio_dollars = portfolio * initial_portfolio
            short_dollars = short_gains_ratio * initial_portfolio
            long_dollars = long_gains_ratio * initial_portfolio
            loss_cf_dollars = loss_cf_ratio * initial_portfolio

            # 손실이월 적용: 단기 → 장기 순서로 차감
            net_short_d = short_dollars
            net_long_d = long_dollars

            if loss_cf_dollars > 0:
                deduct = min(loss_cf_dollars, max(net_short_d, 0))
                net_short_d -= deduct
                loss_cf_dollars -= deduct

                deduct = min(loss_cf_dollars, max(net_long_d, 0))
                net_long_d -= deduct
                loss_cf_dollars -= deduct

            # 누진세율 적용 (달러 기준)
            tax_short, tax_long = _calc_progressive_tax(
                base_income, net_short_d, net_long_d)
            tax_due_dollars = tax_short + tax_long

            # 순손실 이월
            net_total_d = net_short_d + net_long_d
            if net_total_d < 0:
                loss_cf_dollars += abs(net_total_d)

            # 달러 세금 → 포트 비율로 역환산하여 차감
            if tax_due_dollars > 0 and portfolio_dollars > 1e-10:
                tax_frac = tax_due_dollars / portfolio_dollars
                ret_arr[i] -= tax_frac
                total_tax_paid_dollars += tax_due_dollars
                annual_taxes[current_year] = tax_due_dollars

            # 이월 손실을 비율 단위로 다시 저장
            loss_cf_ratio = loss_cf_dollars / initial_portfolio if initial_portfolio > 0 else 0.0
            short_gains_ratio = 0.0
            long_gains_ratio = 0.0
            current_year = yr

        # 비중 변화 → 실현 이벤트
        w_now = weight_lagged.iloc[i] if i < len(weight_lagged) else 0.0
        w_prev = weight_lagged.iloc[i - 1] if i > 0 else 0.0
        delta_w = w_prev - w_now  # 양수 = 매도

        if delta_w > 1e-9 and not math.isnan(entry_price) and entry_price > 1e-10:
            # 매도 비중 × (현재 포트 - 진입 포트) = 비율 단위 실현이익
            realized_ratio = delta_w * (portfolio - entry_price)

            holding_days = i - entry_date_idx
            if holding_days >= 252:
                long_gains_ratio += realized_ratio
            else:
                short_gains_ratio += realized_ratio

        # 포트폴리오 업데이트
        portfolio *= (1.0 + ret_arr[i])

        # 진입 추적
        if w_now > 1e-9 and w_prev <= 1e-9:
            entry_price = portfolio
            entry_date_idx = i
        elif w_now <= 1e-9:
            entry_price = math.nan
        elif w_now > w_prev + 1e-9:
            if math.isnan(entry_price):
                entry_price = portfolio
                entry_date_idx = i
            else:
                old_frac = w_prev / w_now
                new_frac = 1.0 - old_frac
                entry_price = entry_price * old_frac + portfolio * new_frac

    # 마지막 해 정산 (미처리분)
    short_dollars = short_gains_ratio * initial_portfolio
    long_dollars = long_gains_ratio * initial_portfolio
    loss_cf_dollars = loss_cf_ratio * initial_portfolio

    net_short_d = short_dollars
    net_long_d = long_dollars
    if loss_cf_dollars > 0:
        deduct = min(loss_cf_dollars, max(net_short_d, 0))
        net_short_d -= deduct
        loss_cf_dollars -= deduct
        deduct = min(loss_cf_dollars, max(net_long_d, 0))
        net_long_d -= deduct
        loss_cf_dollars -= deduct

    tax_short, tax_long = _calc_progressive_tax(base_income, net_short_d, net_long_d)
    final_tax = tax_short + tax_long
    if final_tax > 0:
        total_tax_paid_dollars += final_tax
        annual_taxes[current_year] = final_tax

    result = pd.Series(ret_arr, index=strat_ret.index)
    return result, {"total_tax_paid": total_tax_paid_dollars, "annual_taxes": annual_taxes}


def _adjust_rf_calendar(rf: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Ken French RF(거래일 단위)를 calendar days 기준으로 조정.
    주말 3일이면 3배 이자 적용."""
    rf_aligned = rf.reindex(index, method="ffill").fillna(0)
    cal_days = _calendar_days(index)
    return rf_aligned * cal_days


def backtest_200ma(tqqq: pd.Series, rf: pd.Series,
                    tax_enabled: bool = False) -> tuple:
    """200MA 전략: 3일 확인 매수 백테스트.

    Returns: (cum, weight_series, tax_info)
    """
    _, w = compute_200ma_cross(tqqq.values.astype(float))
    weight_series = pd.Series(w, index=tqqq.index)
    weight_lagged = weight_series.shift(SIGNAL_LAG).fillna(0)

    tqqq_ret = tqqq.pct_change().fillna(0)
    rf_daily = _adjust_rf_calendar(rf, tqqq_ret.index)

    strat_ret = weight_lagged * tqqq_ret + (1 - weight_lagged) * rf_daily
    trades = weight_lagged.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * COMMISSION

    strat_ret, tax_info = _apply_tax(strat_ret, weight_lagged, tax_enabled)

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, weight_series, tax_info


def backtest_dual_ma(tqqq: pd.Series, qqq: pd.Series, rf: pd.Series,
                      fast: int = 3, slow: int = 161,
                      tax_enabled: bool = False) -> tuple:
    """NDX(QQQ) Dual MA 크로스 전략 백테스트.

    Returns: (cum, signal_series, tax_info)
    """
    ma_fast = qqq.rolling(fast, min_periods=fast).mean()
    ma_slow = qqq.rolling(slow, min_periods=slow).mean()
    sig_raw = (ma_fast > ma_slow).astype(int)
    sig = sig_raw.shift(SIGNAL_LAG).fillna(0).astype(int)

    tqqq_ret = tqqq.pct_change().fillna(0)
    rf_daily = _adjust_rf_calendar(rf, tqqq_ret.index)

    strat_ret = sig * tqqq_ret + (1 - sig) * rf_daily
    trades = sig.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * COMMISSION

    strat_ret, tax_info = _apply_tax(strat_ret, sig.astype(float), tax_enabled)

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, sig_raw, tax_info


def backtest_kimjje(tqqq: pd.Series, qqq: pd.Series, spy: pd.Series,
                     rf: pd.Series, spy_bear_cap: float = 0.0,
                     params: BasicParams = None,
                     tax_enabled: bool = False) -> tuple:
    """김째매매법 (Kim) 백테스트.

    Returns: (cum, weights, tax_info)
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
    rf_daily = _adjust_rf_calendar(rf, tqqq_ret.index)

    strat_ret = weight_lagged * tqqq_ret + (1 - weight_lagged) * rf_daily

    # 수수료: 비중 변동분 × 0.2%
    weight_change = weight_lagged.diff().abs().fillna(0)
    strat_ret = strat_ret - weight_change * COMMISSION

    strat_ret, tax_info = _apply_tax(strat_ret, weight_lagged, tax_enabled)

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, weight_series, tax_info


def backtest_snowball(tqqq: pd.Series, qqq: pd.Series, rf: pd.Series,
                       params: SnowballParams = None,
                       tax_enabled: bool = False) -> tuple:
    """Snowball 전략 백테스트.

    수수료는 compute_snowball_strategy() 내부에서 처리 → 외부 COMMISSION=0.

    Returns: (cum, weight_series, tax_info)
    """
    w = compute_snowball_strategy(tqqq.values.astype(float),
                                  qqq.values.astype(float), params)
    weight_series = pd.Series(w, index=tqqq.index)
    weight_lagged = weight_series.shift(SIGNAL_LAG).fillna(0)

    tqqq_ret = tqqq.pct_change().fillna(0)
    rf_daily = _adjust_rf_calendar(rf, tqqq_ret.index)

    strat_ret = weight_lagged * tqqq_ret + (1 - weight_lagged) * rf_daily
    # 수수료: 시뮬레이션 내부에서 이미 처리 → 외부 COMMISSION 적용 안 함

    strat_ret, tax_info = _apply_tax(strat_ret, weight_lagged, tax_enabled)

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, weight_series, tax_info


def backtest_ensemble(tqqq: pd.Series, qqq: pd.Series, spy: pd.Series,
                       rf: pd.Series, params: EnsembleParams = None,
                       tax_enabled: bool = False) -> tuple:
    """Kim+Snowball 앙상블 백테스트.

    Returns: (cum, weight_series, tax_info)
    """
    w = compute_ensemble_strategy(qqq.values.astype(float),
                                   tqqq.values.astype(float),
                                   spy.values.astype(float), params)
    weight_series = pd.Series(w, index=tqqq.index)
    weight_lagged = weight_series.shift(SIGNAL_LAG).fillna(0)

    tqqq_ret = tqqq.pct_change().fillna(0)
    rf_daily = _adjust_rf_calendar(rf, tqqq_ret.index)

    strat_ret = weight_lagged * tqqq_ret + (1 - weight_lagged) * rf_daily
    weight_change = weight_lagged.diff().abs().fillna(0)
    strat_ret = strat_ret - weight_change * COMMISSION

    strat_ret, tax_info = _apply_tax(strat_ret, weight_lagged, tax_enabled)

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum, weight_series, tax_info


def backtest_buy_and_hold(tqqq: pd.Series) -> pd.Series:
    """TQQQ Buy & Hold 누적 수익률."""
    daily_ret = tqqq.pct_change().fillna(0)
    cum = (1 + daily_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum


# ══════════════════════════════════════════════════════════════
# [8] 메트릭 계산
# ══════════════════════════════════════════════════════════════

def calc_metrics(cum: pd.Series, rf: pd.Series = None,
                 tax_info: dict = None) -> dict:
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

    result = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Total Return": total_ret,
    }

    if tax_info is not None:
        result["Total Tax Paid"] = tax_info.get("total_tax_paid", 0.0)

    return result


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
        "200MA": "#2196F3",
        "DualMA(3/161)": "#8BC34A",
        "Kim": "#FF5722",
        "Snowball": "#E91E63",
        "Ensemble": "#9C27B0",

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

def show_todays_signal(data: dict,
                        ma200_weights: pd.Series,
                        dual_sig: pd.Series,
                        kim_weights: pd.Series,
                        snow_weights: pd.Series = None,
                        ens_weights: pd.Series = None):
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

    # 200MA
    ma_w = ma200_weights.iloc[-1]
    print(f"\n  [전략 1] 200MA:            TQQQ {ma_w * 100:.0f}%")

    # DualMA(3/161)
    d_sig = dual_sig.iloc[-1]
    d_status = "매수 (TQQQ 100%)" if d_sig == 1 else "관망 (현금)"
    print(f"  [전략 2] DualMA(3/161):    {d_status}")

    # Kim
    kim_w = kim_weights.iloc[-1]
    print(f"  [전략 3] Kim:              TQQQ {kim_w * 100:.0f}%")

    # Snowball
    if snow_weights is not None:
        snow_w = snow_weights.iloc[-1]
        print(f"  [전략 4] Snowball:         TQQQ {snow_w * 100:.0f}%")

    # Ensemble
    if ens_weights is not None:
        ens_w = ens_weights.iloc[-1]
        print(f"  [전략 5] Ensemble:         TQQQ {ens_w * 100:.0f}%")

    print(f"\n  → 모두 내일 적용 (lag=1)")
    print()


# ══════════════════════════════════════════════════════════════
# [11] Phase 4 검증: DFF 트래킹 에러
# ══════════════════════════════════════════════════════════════

def validate_dff_tracking(data_real: dict):
    """합성 TQQQ vs 실제 TQQQ 트래킹 에러 비교 (DFF 모델 vs 고정 ER)."""
    print("\n" + "=" * 60)
    print("DFF 모델 검증: 합성 TQQQ 트래킹 에러")
    print("=" * 60)

    qqq = data_real["qqq"]
    tqqq_real = data_real["tqqq"]

    # 고정 ER 합성
    tqqq_fixed = synthesize_tqqq(qqq)

    # DFF 합성
    dff = download_fred_dff()
    tqqq_dff = synthesize_tqqq_dff(qqq, dff)

    # 공통 인덱스
    common = tqqq_real.index.intersection(tqqq_fixed.index).intersection(tqqq_dff.index)
    tqqq_real = tqqq_real.loc[common]
    tqqq_fixed = tqqq_fixed.loc[common]
    tqqq_dff = tqqq_dff.loc[common]

    # 일별 수익률 비교
    ret_real = tqqq_real.pct_change().dropna()
    ret_fixed = tqqq_fixed.pct_change().dropna()
    ret_dff = tqqq_dff.pct_change().dropna()

    te_fixed = (ret_real - ret_fixed).std() * np.sqrt(252) * 100
    te_dff = (ret_real - ret_dff).std() * np.sqrt(252) * 100

    # 누적 수익률 상관계수
    cum_real = (1 + ret_real).cumprod()
    cum_fixed = (1 + ret_fixed).cumprod()
    cum_dff = (1 + ret_dff).cumprod()

    corr_fixed = cum_real.corr(cum_fixed)
    corr_dff = cum_real.corr(cum_dff)

    print(f"\n  비교 기간: {common[0].date()} → {common[-1].date()}")
    print(f"\n  {'모델':20s} {'트래킹에러(ann%)':>16s} {'누적상관':>10s}")
    print(f"  {'-' * 48}")
    print(f"  {'고정 ER 3.5%':20s} {te_fixed:>15.2f}% {corr_fixed:>9.4f}")
    print(f"  {'DFF 동적':20s} {te_dff:>15.2f}% {corr_dff:>9.4f}")
    better = "DFF 동적" if te_dff < te_fixed else "고정 ER"
    print(f"\n  → {better} 모델이 실제 TQQQ에 더 가까움")


# ══════════════════════════════════════════════════════════════
# [12] Main
# ══════════════════════════════════════════════════════════════

def run_comparison(data: dict, compare_start: str, label: str, chart_fname: str,
                    tax_enabled: bool = False):
    """백테스트 + 메트릭 + 차트 한 세트 실행."""
    tax_label = " [세후]" if tax_enabled else ""
    print("\n" + "=" * 60)
    print(f"백테스트 — {label}{tax_label}")
    print("=" * 60)

    cum_200ma, w_200ma, tax_200ma = backtest_200ma(data["tqqq"], data["rf"], tax_enabled)
    print("  [1/6] 200MA")

    cum_dual, sig_dual, tax_dual = backtest_dual_ma(
        data["tqqq"], data["qqq"], data["rf"], tax_enabled=tax_enabled)
    print("  [2/6] DualMA(3/161)")

    cum_kim, w_kim, tax_kim = backtest_kimjje(
        data["tqqq"], data["qqq"], data["spy"], data["rf"],
        spy_bear_cap=0.0, tax_enabled=tax_enabled)
    print("  [3/6] Kim")

    cum_snow, w_snow, tax_snow = backtest_snowball(
        data["tqqq"], data["qqq"], data["rf"], tax_enabled=tax_enabled)
    print("  [4/6] Snowball")

    cum_ens, w_ens, tax_ens = backtest_ensemble(
        data["tqqq"], data["qqq"], data["spy"], data["rf"],
        tax_enabled=tax_enabled)
    print("  [5/6] Ensemble")

    cum_bh = backtest_buy_and_hold(data["tqqq"])
    print("  [6/6] TQQQ B&H")

    # 비교 기간 트림
    start_mask = cum_200ma.index >= compare_start
    all_cums = {
        "200MA": (cum_200ma, tax_200ma),
        "DualMA(3/161)": (cum_dual, tax_dual),
        "Kim": (cum_kim, tax_kim),
        "Snowball": (cum_snow, tax_snow),
        "Ensemble": (cum_ens, tax_ens),
        "TQQQ B&H": (cum_bh, None),
    }
    trimmed = {}
    tax_infos = {}
    for name, (cum, ti) in all_cums.items():
        c = cum.loc[start_mask]
        trimmed[name] = c / c.iloc[0]
        tax_infos[name] = ti

    rf_t = data["rf"].loc[start_mask]
    period_str = f"{trimmed['200MA'].index[0].date()} → {trimmed['200MA'].index[-1].date()}"
    print(f"\n  비교 기간: {period_str}")

    # 메트릭
    print("\n" + "-" * 60)
    print(f"성과 비교 — {label}{tax_label}")
    print("-" * 60)

    metrics = {}
    for name, cum in trimmed.items():
        metrics[name] = calc_metrics(cum, rf_t, tax_infos.get(name))

    all_sigs = {
        "200MA": (w_200ma.loc[start_mask], False),
        "DualMA(3/161)": (sig_dual.loc[start_mask], True),
        "Kim": (w_kim.loc[start_mask], False),
        "Snowball": (w_snow.loc[start_mask], False),
        "Ensemble": (w_ens.loc[start_mask], False),
        "TQQQ B&H": (None, True),
    }
    for name, (sig, is_bin) in all_sigs.items():
        if sig is not None:
            metrics[name]["Trades/Yr"] = count_trades(sig, is_binary=is_bin)
        else:
            metrics[name]["Trades/Yr"] = 0.0

    if tax_enabled:
        header = f"{'':20s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} {'Trades/Y':>9s} {'TotalTax':>12s}"
    else:
        header = f"{'':20s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} {'Trades/Y':>9s}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, m in metrics.items():
        base = (f"{name:20s} {m['CAGR']:>7.1%} {m['Sharpe']:>8.2f} {m['Sortino']:>8.2f} "
                f"{m['MDD']:>7.1%} {m.get('Trades/Yr', 0):>8.1f}")
        if tax_enabled:
            base += f" ${m.get('Total Tax Paid', 0):>10,.0f}"
        print(base)

    # 차트
    plot_comparison(trimmed, f"{label}{tax_label} ({period_str})", chart_fname)

    return w_200ma, sig_dual, w_kim, w_snow, w_ens


def run_v1_v2_validation(data: dict, compare_start: str):
    """Phase 4: v1(고정 ER, 세금 없음) vs v2(DFF, 세금) 비교."""
    print("\n" + "=" * 70)
    print("Phase 4 검증: v1 vs v2 모드별 CAGR 비교")
    print("=" * 70)

    start_mask = data["tqqq"].index >= compare_start
    rf_t = data["rf"].loc[start_mask]

    # DFF off, tax off (v1 호환 모드)
    cum_kim_v1, _, _ = backtest_kimjje(
        data["tqqq"], data["qqq"], data["spy"], data["rf"],
        spy_bear_cap=0.0, tax_enabled=False)
    c = cum_kim_v1.loc[start_mask]
    m_v1 = calc_metrics(c / c.iloc[0], rf_t)

    # DFF on (data already has DFF pricing), tax off
    cum_kim_dff, _, _ = backtest_kimjje(
        data["tqqq"], data["qqq"], data["spy"], data["rf"],
        spy_bear_cap=0.0, tax_enabled=False)
    c = cum_kim_dff.loc[start_mask]
    m_dff = calc_metrics(c / c.iloc[0], rf_t)

    # DFF on, tax on
    cum_kim_full, _, tax_full = backtest_kimjje(
        data["tqqq"], data["qqq"], data["spy"], data["rf"],
        spy_bear_cap=0.0, tax_enabled=True)
    c = cum_kim_full.loc[start_mask]
    m_full = calc_metrics(c / c.iloc[0], rf_t, tax_full)

    period_str = f"{data['tqqq'].loc[start_mask].index[0].date()} → {data['tqqq'].loc[start_mask].index[-1].date()}"
    print(f"\n  Kim 전략 비교 ({period_str}):")
    print(f"\n  {'모드':25s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s}")
    print(f"  {'-' * 58}")
    print(f"  {'세전 (v1 호환)':25s} {m_v1['CAGR']:>7.1%} {m_v1['Sharpe']:>8.2f} {m_v1['Sortino']:>8.2f} {m_v1['MDD']:>7.1%}")
    print(f"  {'DFF only':25s} {m_dff['CAGR']:>7.1%} {m_dff['Sharpe']:>8.2f} {m_dff['Sortino']:>8.2f} {m_dff['MDD']:>7.1%}")
    print(f"  {'DFF + 세금':25s} {m_full['CAGR']:>7.1%} {m_full['Sharpe']:>8.2f} {m_full['Sortino']:>8.2f} {m_full['MDD']:>7.1%}")
    print(f"\n  세금 총납부: ${m_full.get('Total Tax Paid', 0):,.0f}")
    print(f"  가정: 투자금 ${INITIAL_PORTFOLIO:,.0f}, 노동소득 ${BASE_ANNUAL_INCOME:,.0f}/yr, 누진세율 (2025 Single)")


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  LRS Standalone v2 — DFF 파이낸싱 + 세금 + TP 래더         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── 데이터 다운로드 ──
    data_idx = download_all_data(mode="index", use_dff=True)
    data_synth = download_all_data(mode="synth", use_dff=True)
    data_real_dff = download_all_data(mode="real", use_dff=False)

    # ── 세후 비교 3기간 ──
    print("\n\n" + "#" * 70)
    print("# 세후 비교 (DFF 파이낸싱 + 미국 세금)")
    print("#" * 70)

    # 1) 최장 40년
    run_comparison(data_idx, compare_start="1986-01-01",
                   label="합성 3x NDX [DFF] (최장 40년)",
                   chart_fname="lrs_v2_index_40y_tax.png",
                   tax_enabled=True)

    # 2) 닷컴 이후
    run_comparison(data_synth, compare_start="2003-01-01",
                   label="합성 3x QQQ [DFF] (닷컴 이후)",
                   chart_fname="lrs_v2_synth_postdot_tax.png",
                   tax_enabled=True)

    # 3) TQQQ 이후
    w_200ma, sig_dual, w_kim, w_snow, w_ens = run_comparison(
        data_real_dff, compare_start=COMPARE_START,
        label="실제 TQQQ",
        chart_fname="lrs_v2_real_15y_tax.png",
        tax_enabled=True)

    # ── 오늘의 시그널 ──
    show_todays_signal(data_real_dff, w_200ma, sig_dual, w_kim, w_snow, w_ens)

    print("완료!")
