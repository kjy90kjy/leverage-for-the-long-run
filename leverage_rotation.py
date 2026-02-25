"""
Leverage for the Long Run  - Leverage Rotation Strategy (LRS)
Based on Michael Gayed's 2016 Dow Award paper.

Usage:
    python leverage_rotation.py
"""

import sys
import io
import warnings

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
import zipfile
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 1. DATA MODULE
# ──────────────────────────────────────────────

def download(ticker: str, start: str = "1990-01-01", end: str = "2025-12-31",
             total_return: bool = False) -> pd.Series:
    """Return Adj Close series for *ticker*.
    If total_return=True and ticker is ^GSPC, synthesize total return
    by adding Shiller dividend yields to daily price returns.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    close = df["Close"].squeeze()
    close.name = ticker

    if total_return and ticker == "^GSPC":
        close = _add_shiller_dividends(close)

    return close


def _add_shiller_dividends(price: pd.Series) -> pd.Series:
    """Synthesize S&P 500 total return by adding Shiller monthly dividend yields
    to daily ^GSPC price returns. Returns a synthetic total-return price series."""
    try:
        url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
        raw = pd.read_excel(url, sheet_name="Data", header=None, skiprows=8)
        raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()

        date_col = raw.iloc[:, 0].astype(float)
        sp_price = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
        dividend = pd.to_numeric(raw.iloc[:, 2], errors="coerce")

        years = date_col.astype(int)
        months = ((date_col - years) * 100).round().astype(int).clip(1, 12)
        dates = pd.to_datetime({"year": years, "month": months, "day": 1})

        shiller = pd.DataFrame({"sp": sp_price.values, "div": dividend.values}, index=dates)
        shiller = shiller.dropna()
        # Annual dividend rate -> daily yield
        shiller["div_yield_daily"] = (shiller["div"] / shiller["sp"]) / 252
        # Resample to daily by forward-fill
        div_daily = shiller["div_yield_daily"].resample("D").ffill()

        # Align with price index
        daily_ret = price.pct_change()
        div_aligned = div_daily.reindex(daily_ret.index, method="ffill").fillna(0)
        total_ret = daily_ret + div_aligned
        # Rebuild price series from total returns
        total_price = (1 + total_ret).cumprod() * price.iloc[0]
        total_price.iloc[0] = price.iloc[0]
        total_price.name = price.name
        print(f"  [Total Return] Added Shiller dividends to {price.name}")
        return total_price
    except Exception as e:
        print(f"  [Warning] Could not add Shiller dividends: {e}")
        print(f"  [Warning] Using price-only returns")
        return price


_RF_CACHE = None

def download_ken_french_rf() -> pd.Series:
    """Download daily risk-free rate from Ken French Data Library.
    Returns a Series indexed by date with daily RF rate (decimal, e.g. 0.0001).
    Source: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    """
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
        df["RF"] = pd.to_numeric(df["RF"].str.strip()) / 100.0  # % -> decimal
        rf = df.set_index("date")["RF"]
        _RF_CACHE = rf
        print(f"  [Ken French] Downloaded daily RF: {rf.index[0].date()} -> {rf.index[-1].date()}")
        return rf
    except Exception as e:
        print(f"  [Warning] Ken French download failed: {e}")
        print(f"  [Warning] Using flat 3% annual RF")
        idx = pd.date_range("1926-01-01", "2030-12-31", freq="B")
        _RF_CACHE = pd.Series(0.03 / 252, index=idx)
        return _RF_CACHE


def get_tbill_rate_from_irx() -> float:
    """Try to fetch the latest 13-week T-Bill rate from ^IRX."""
    try:
        df = yf.download("^IRX", period="5d", progress=False)
        if not df.empty:
            rate = df["Close"].squeeze().iloc[-1] / 100.0
            return float(rate)
    except Exception:
        pass
    return 0.02  # fallback


# ──────────────────────────────────────────────
# 2. SIGNAL FUNCTIONS
# ──────────────────────────────────────────────

def signal_ma(price: pd.Series, period: int = 200) -> pd.Series:
    """1 when price > N-day SMA, else 0."""
    ma = price.rolling(period).mean()
    return (price > ma).astype(int)


def signal_dual_ma(price: pd.Series, slow: int = 200, fast: int = 50) -> pd.Series:
    """1 when fast MA > slow MA (golden cross), else 0.
    When called via the framework, `slow` is the MA period from config.
    `fast` defaults to 50 (or slow//4 if slow < 200).
    """
    if fast >= slow:
        fast = max(slow // 4, 10)
    return (price.rolling(fast).mean() > price.rolling(slow).mean()).astype(int)


def signal_rsi(price: pd.Series, period: int = 14, threshold: int = 50) -> pd.Series:
    """1 when RSI > threshold, else 0."""
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return (rsi > threshold).astype(int)


# ──────────────────────────────────────────────
# 3. STRATEGY ENGINE
# ──────────────────────────────────────────────

def apply_leverage(daily_ret: pd.Series, leverage: float, expense_ratio: float) -> pd.Series:
    """Daily leveraged return: ret × leverage − daily expense."""
    daily_cost = expense_ratio / 252
    return daily_ret * leverage - daily_cost


def run_lrs(price: pd.Series, signal: pd.Series,
            leverage: float = 2.0, expense_ratio: float = 0.01,
            tbill_rate=0.02, signal_lag: int = 0,
            commission: float = 0.0) -> pd.Series:
    """
    Leverage Rotation Strategy.
    signal=1 -> leveraged equity; signal=0 -> T-Bill return.

    tbill_rate: float (annualised, converted to daily) or pd.Series of daily rates.
    signal_lag: 0 = same-day (paper convention), 1 = next-day (realistic trading).
    commission: per-trade cost as fraction (0.002 = 0.2%), applied on each signal flip.
    """
    daily_ret = price.pct_change()
    sig = signal.shift(signal_lag) if signal_lag > 0 else signal
    lev_ret = apply_leverage(daily_ret, leverage, expense_ratio)

    # T-Bill daily return: scalar or series
    if isinstance(tbill_rate, pd.Series):
        tbill_daily = tbill_rate.reindex(daily_ret.index, method="ffill").fillna(0)
    else:
        tbill_daily = tbill_rate / 252

    strat_ret = sig * lev_ret + (1 - sig) * tbill_daily

    # Per-trade commission: deduct on each signal change (buy or sell)
    if commission > 0:
        trades = sig.diff().abs().fillna(0)
        strat_ret = strat_ret - trades * commission

    cum = (1 + strat_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum


def run_buy_and_hold(price: pd.Series, leverage: float = 1.0,
                     expense_ratio: float = 0.0) -> pd.Series:
    """Buy & hold with optional leverage."""
    daily_ret = price.pct_change()
    lev_ret = apply_leverage(daily_ret, leverage, expense_ratio)
    cum = (1 + lev_ret).cumprod()
    cum.iloc[0] = 1.0
    return cum


# ──────────────────────────────────────────────
# 4. PERFORMANCE METRICS
# ──────────────────────────────────────────────

def calc_metrics(cum: pd.Series, benchmark_cum: pd.Series = None,
                 tbill_rate=0.02, rf_series: pd.Series = None) -> dict:
    """Compute key performance metrics from a cumulative return series.

    tbill_rate: scalar annual rate (used for Sharpe if rf_series not given).
    rf_series: daily RF series from Ken French (decimal, e.g. 0.0001/day).
    """
    daily_ret = cum.pct_change().dropna()
    n_days = len(daily_ret)
    n_years = n_days / 252

    # CAGR
    total_ret = cum.iloc[-1] / cum.iloc[0]
    cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0

    # Daily RF for Sharpe/Sortino
    if rf_series is not None:
        rf_aligned = rf_series.reindex(daily_ret.index, method="ffill").fillna(0)
        avg_annual_rf = rf_aligned.mean() * 252
        rf_daily = rf_aligned
    else:
        avg_annual_rf = tbill_rate if isinstance(tbill_rate, (int, float)) else 0.03
        rf_daily = avg_annual_rf / 252

    # Arithmetic annualised mean return (Sharpe 1994: use arithmetic, not geometric)
    arith_annual = daily_ret.mean() * 252

    # Volatility
    vol = daily_ret.std() * np.sqrt(252)

    # Sharpe — arithmetic mean excess / annualised vol
    sharpe = (arith_annual - avg_annual_rf) / vol if vol > 0 else 0

    # Sortino — target downside deviation per Sortino & van der Meer (1991):
    # TDD = sqrt( mean( min(r - rf, 0)^2 ) ) over ALL observations
    excess_daily = daily_ret - rf_daily
    downside_diff = excess_daily.copy()
    downside_diff[downside_diff > 0] = 0.0
    downside_dev = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(252)
    sortino = (arith_annual - avg_annual_rf) / downside_dev if downside_dev > 0 else 0

    # MDD
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    mdd = drawdown.min()

    # Beta & Alpha vs benchmark (weekly returns, per paper footnote 22)
    beta, alpha = np.nan, np.nan
    if benchmark_cum is not None:
        # Resample to weekly for alpha/beta (paper convention)
        strat_weekly = cum.resample("W-FRI").last().pct_change().dropna()
        bench_weekly = benchmark_cum.resample("W-FRI").last().pct_change().dropna()
        aligned = pd.concat([strat_weekly, bench_weekly], axis=1, join="inner").dropna()
        if len(aligned) > 20:
            aligned.columns = ["strat", "bench"]
            cov = np.cov(aligned["strat"], aligned["bench"])
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
            bench_annual = aligned["bench"].mean() * 52
            alpha = cagr - (avg_annual_rf + beta * (bench_annual - avg_annual_rf))

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "Beta": beta,
        "Alpha": alpha,
        "Total Return": total_ret,
    }


def signal_trades_per_year(signal: pd.Series) -> float:
    """Average number of round-trip signal changes per year."""
    flips = (signal.diff().abs() > 0).sum()
    n_years = len(signal) / 252
    return flips / n_years if n_years > 0 else 0


# ──────────────────────────────────────────────
# 5. VISUALIZATION
# ──────────────────────────────────────────────

def plot_cumulative(curves: dict[str, pd.Series], title: str, fname: str):
    """Cumulative return curves on log scale."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, s in curves.items():
        ax.plot(s.index, s.values, label=label, linewidth=1.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")


def plot_drawdowns(curves: dict[str, pd.Series], title: str, fname: str):
    """Rolling drawdown comparison."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, cum in curves.items():
        dd = cum / cum.cummax() - 1
        ax.fill_between(dd.index, dd.values, alpha=0.25, label=label)
        ax.plot(dd.index, dd.values, linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")


def plot_vol_bars(price: pd.Series, signal: pd.Series, ma_period: int, fname: str):
    """Annualised vol when above vs below MA."""
    daily_ret = price.pct_change().dropna()
    sig = signal.reindex(daily_ret.index).fillna(0)
    above_vol = daily_ret[sig == 1].std() * np.sqrt(252)
    below_vol = daily_ret[sig == 0].std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Above MA", "Below MA"], [above_vol, below_vol],
                  color=["#2ecc71", "#e74c3c"], edgecolor="black")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.1%}", ha="center", fontsize=11)
    ax.set_title(f"Annualised Volatility: Above vs Below {ma_period}-day MA", fontsize=13)
    ax.set_ylabel("Annualised Volatility")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")


def plot_rolling_excess(lrs_cum: pd.Series, bh_cum: pd.Series,
                        window_years: int, title: str, fname: str):
    """Rolling N-year excess return of LRS over Buy&Hold."""
    window = window_years * 252
    lrs_roll = (lrs_cum / lrs_cum.shift(window)) ** (1 / window_years) - 1
    bh_roll = (bh_cum / bh_cum.shift(window)) ** (1 / window_years) - 1
    excess = lrs_roll - bh_roll

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(excess.index, excess.values,
                    where=excess >= 0, color="#2ecc71", alpha=0.4, label="LRS outperforms")
    ax.fill_between(excess.index, excess.values,
                    where=excess < 0, color="#e74c3c", alpha=0.4, label="B&H outperforms")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(f"Rolling {window_years}Y Excess CAGR")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")


# ──────────────────────────────────────────────
# 6. REPORTING
# ──────────────────────────────────────────────

def print_table(rows: list[dict], title: str):
    """Pretty-print a metrics table."""
    df = pd.DataFrame(rows).set_index("Strategy")
    fmt = {
        "CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}",
        "Sortino": "{:.2f}", "MDD": "{:.2%}", "Beta": "{:.2f}",
        "Alpha": "{:.2%}", "Total Return": "{:.1f}x",
        "Trades/Year": "{:.1f}",
    }
    for col, f in fmt.items():
        if col in df.columns:
            df[col] = df[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "N/A")
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(df.to_string())
    print()


# ──────────────────────────────────────────────
# 7. MAIN  - PAPER REPLICATION + EXTENSIONS
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ticker": "^GSPC",
    "start": "1990-01-01",
    "end": "2025-12-31",
    "ma_periods": [200],
    "leverage_factors": [1, 2, 3],
    "expense_ratio": 0.01,
    "tbill_rate": 0.02,
    "signal_fn": signal_ma,
    "signal_lag": 0,             # 종가 시그널 → 종가 매매
}

# Paper replication config: Oct 1928 - Dec 2020
# Data: S&P 500 Total Return (Gross Dividends) - synthesized via Shiller
# RF: Ken French daily T-Bill rates
# MA: computed on total return series (per footnote 15)
# Signal: same-day close → same-day execution (종가기준 종가매매)
PAPER_CONFIG = {
    "ticker": "^GSPC",
    "start": "1928-10-01",
    "end": "2020-12-31",
    "ma_periods": [200],
    "leverage_factors": [2, 3],
    "expense_ratio": 0.01,
    "tbill_rate": "ken_french",  # use daily RF series
    "signal_fn": signal_ma,
    "total_return": True,        # Shiller dividends for ^GSPC
    "signal_lag": 0,             # 종가기준 종가매매
}

# NASDAQ Composite long-run config: Feb 1971 - present
# Data: ^IXIC price-only (no free total return source for NASDAQ)
# RF: Ken French daily T-Bill rates
# Same methodology: 200-day MA, signal_lag=0, 1% expense
NASDAQ_LONGRUN_CONFIG = {
    "ticker": "^IXIC",
    "start": "1971-01-01",
    "end": "2025-12-31",
    "ma_periods": [200],
    "leverage_factors": [1, 2, 3],
    "expense_ratio": 0.01,
    "tbill_rate": "ken_french",
    "signal_fn": signal_ma,
    "total_return": False,       # no Shiller equivalent for NASDAQ
    "signal_lag": 0,             # 종가기준 종가매매
}


def run_analysis(config: dict):
    """Run full LRS analysis for a given config."""
    ticker = config["ticker"]
    start = config["start"]
    end = config["end"]
    ma_periods = config["ma_periods"]
    leverage_factors = config["leverage_factors"]
    expense = config["expense_ratio"]
    tbill_cfg = config["tbill_rate"]
    signal_fn = config["signal_fn"]
    total_return = config.get("total_return", False)
    signal_lag = config.get("signal_lag", 0)

    # Resolve T-Bill rate: Ken French daily series or scalar
    rf_series = None
    if tbill_cfg == "ken_french":
        rf_series = download_ken_french_rf()
        tbill_scalar = rf_series.mean() * 252  # for display
        tbill_for_lrs = rf_series  # daily series
    else:
        tbill_scalar = float(tbill_cfg)
        tbill_for_lrs = tbill_scalar

    print(f"\n{'#' * 70}")
    print(f"  Analysing: {ticker}  |  {start} -> {end}")
    print(f"  Signal: {signal_fn.__name__}  |  MA periods: {ma_periods}")
    print(f"  Leverage: {leverage_factors}  |  Expense: {expense:.2%}")
    tbill_label = "Ken French daily" if tbill_cfg == "ken_french" else f"{tbill_scalar:.2%}"
    print(f"  T-Bill: {tbill_label} (avg {tbill_scalar:.2%}/yr)  |  Signal lag: {signal_lag}d")
    if total_return:
        print(f"  Data: Total Return (Shiller dividends)  |  MA on: total return series")
    print(f"{'#' * 70}")

    price = download(ticker, start, end, total_return=total_return)
    print(f"  Downloaded {len(price)} trading days  ({price.index[0].date()} -> {price.index[-1].date()})")

    # MA is computed on total return series (paper footnote 15)
    # So signal uses the same 'price' (which is total return if enabled)
    price_for_signal = price

    # Benchmark: unleveraged buy & hold (no expense)
    bh_1x = run_buy_and_hold(price, leverage=1.0, expense_ratio=0.0)

    for ma_period in ma_periods:
        print(f"\n--- MA Period = {ma_period} ---")
        sig = signal_fn(price_for_signal, ma_period)
        trades_yr = signal_trades_per_year(sig)

        curves = {"Buy&Hold 1x": bh_1x}
        rows = []

        # Buy & Hold 1x (baseline, no leverage)
        m = calc_metrics(bh_1x, tbill_rate=tbill_scalar, rf_series=rf_series)
        m["Strategy"] = "Buy&Hold 1x"
        m["Trades/Year"] = 0
        rows.append(m)

        for lev in leverage_factors:
            # Skip duplicate 1x B&H row
            if lev > 1:
                bh_lev = run_buy_and_hold(price, leverage=lev,
                                           expense_ratio=expense)
                label_bh = f"Buy&Hold {lev}x"
                curves[label_bh] = bh_lev
                m = calc_metrics(bh_lev, benchmark_cum=bh_1x,
                                 tbill_rate=tbill_scalar, rf_series=rf_series)
                m["Strategy"] = label_bh
                m["Trades/Year"] = 0
                rows.append(m)

            # LRS
            lrs = run_lrs(price, sig, leverage=lev, expense_ratio=expense,
                          tbill_rate=tbill_for_lrs, signal_lag=signal_lag)
            label_lrs = f"LRS {lev}x (MA{ma_period})"
            curves[label_lrs] = lrs
            m = calc_metrics(lrs, benchmark_cum=bh_1x,
                             tbill_rate=tbill_scalar, rf_series=rf_series)
            m["Strategy"] = label_lrs
            m["Trades/Year"] = trades_yr
            rows.append(m)

        safe_ticker = ticker.replace("^", "")
        print_table(rows, f"{ticker}  - MA {ma_period}  ({start} to {end})")

        # Charts
        plot_cumulative(curves,
                        f"{ticker}  - Cumulative Returns (MA{ma_period})",
                        f"{safe_ticker}_cumulative_ma{ma_period}.png")
        plot_drawdowns(curves,
                       f"{ticker}  - Drawdowns (MA{ma_period})",
                       f"{safe_ticker}_drawdowns_ma{ma_period}.png")
        plot_vol_bars(price, sig, ma_period,
                      f"{safe_ticker}_vol_bars_ma{ma_period}.png")

        # Rolling 3-year excess for the highest leverage LRS vs B&H 1x
        best_lev = max(leverage_factors)
        lrs_best = run_lrs(price, sig, leverage=best_lev,
                           expense_ratio=expense, tbill_rate=tbill_for_lrs,
                           signal_lag=signal_lag)
        plot_rolling_excess(lrs_best, bh_1x, 3,
                            f"{ticker}  - Rolling 3Y Excess Return: LRS {best_lev}x vs B&H 1x",
                            f"{safe_ticker}_rolling3y_ma{ma_period}.png")


def run_etf_comparison(config: dict):
    """Compare theoretical leverage simulation vs actual leveraged ETFs."""
    print(f"\n{'#' * 70}")
    print(f"  ETF vs Simulated Leverage Comparison")
    print(f"{'#' * 70}")

    pairs = [
        ("SPY", "SSO", 2),   # 2x S&P 500
        ("SPY", "UPRO", 3),  # 3x S&P 500
        ("QQQ", "TQQQ", 3),  # 3x Nasdaq-100
    ]

    tbill_cfg = config.get("tbill_rate", 0.02)
    expense = config.get("expense_ratio", 0.01)
    signal_lag = config.get("signal_lag", 0)

    # Resolve RF
    rf_series = None
    if tbill_cfg == "ken_french":
        rf_series = download_ken_french_rf()
        tbill_scalar = rf_series.mean() * 252
        tbill_for_lrs = rf_series
    else:
        tbill_scalar = float(tbill_cfg)
        tbill_for_lrs = tbill_scalar

    for base_ticker, lev_ticker, lev_factor in pairs:
        try:
            base = download(base_ticker, "2010-01-01", config["end"])
            lev_etf = download(lev_ticker, "2010-01-01", config["end"])
        except ValueError as e:
            print(f"  Skipping {base_ticker}/{lev_ticker}: {e}")
            continue

        common = base.index.intersection(lev_etf.index)
        if len(common) < 100:
            print(f"  Skipping {base_ticker}/{lev_ticker}: not enough overlap")
            continue
        base = base.loc[common]
        lev_etf = lev_etf.loc[common]

        sim = run_buy_and_hold(base, leverage=lev_factor, expense_ratio=expense)
        actual = run_buy_and_hold(lev_etf, leverage=1.0, expense_ratio=0.0)
        sig = signal_ma(base, 200)
        lrs = run_lrs(base, sig, leverage=lev_factor, expense_ratio=expense,
                      tbill_rate=tbill_for_lrs, signal_lag=signal_lag)
        bh_1x = run_buy_and_hold(base, leverage=1.0, expense_ratio=0.0)

        rows = []
        for label, cum in [
            (f"{base_ticker} 1x", bh_1x),
            (f"{lev_ticker} (actual)", actual),
            (f"Simulated {lev_factor}x B&H", sim),
            (f"LRS {lev_factor}x (MA200)", lrs),
        ]:
            m = calc_metrics(cum, benchmark_cum=bh_1x,
                             tbill_rate=tbill_scalar, rf_series=rf_series)
            m["Strategy"] = label
            m["Trades/Year"] = signal_trades_per_year(sig) if "LRS" in label else 0
            rows.append(m)

        print_table(rows, f"{base_ticker} / {lev_ticker}  - ETF Comparison (2010-present)")

        safe = f"{base_ticker}_{lev_ticker}"
        plot_cumulative(
            {f"{base_ticker} 1x": bh_1x,
             f"{lev_ticker} (actual)": actual,
             f"Sim {lev_factor}x B&H": sim,
             f"LRS {lev_factor}x": lrs},
            f"{base_ticker}/{lev_ticker}  - Cumulative Returns",
            f"{safe}_cumulative.png")


# ──────────────────────────────────────────────
# 8. DUAL MA GRID SEARCH + HEATMAP
# ──────────────────────────────────────────────

def _max_entry_drawdown(cum: pd.Series, signal: pd.Series, signal_lag: int) -> float:
    """MDD measured from running max of entry-point equity (not equity curve peak).

    Peak only updates when a new trade is entered (signal 0→1).
    Drawdown measured only while in position (signal=1).
    """
    sig = signal.shift(signal_lag) if signal_lag > 0 else signal
    sig = sig.reindex(cum.index).fillna(0)
    entries = (sig.diff() == 1)
    entry_equity = cum.where(entries).ffill().fillna(cum.iloc[0])
    entry_peak = entry_equity.cummax()
    dd = (cum / entry_peak - 1).where(sig == 1)
    return float(dd.min()) if dd.notna().any() else 0.0


def _max_recovery_days(cum: pd.Series) -> int:
    """Maximum number of trading days to recover from a drawdown peak."""
    running_max = cum.cummax()
    is_at_peak = cum >= running_max
    # Find consecutive non-peak streaks
    max_days = 0
    current = 0
    for at_peak in is_at_peak:
        if at_peak:
            max_days = max(max_days, current)
            current = 0
        else:
            current += 1
    max_days = max(max_days, current)
    return max_days


def run_dual_ma_grid(price: pd.Series, leverage_list: list,
                     expense_ratio: float, tbill_rate, signal_lag: int,
                     slow_range: range, fast_range: range,
                     rf_series: pd.Series = None,
                     commission: float = 0.0) -> pd.DataFrame:
    """Run LRS backtest for all (slow, fast) MA combinations × leverage levels.

    Returns DataFrame with columns:
        slow, fast, leverage, CAGR, Sharpe, Vol, MDD,
        Trades_Year, Total_Trades, Max_Recovery_Days
    Also includes B&H rows (slow=0, fast=0) for each leverage.
    """
    daily_ret = price.pct_change()
    results = []

    # Warm-up: skip first max(slow_range) trading days so all MAs are valid.
    # Signal is forced to 0 during warm-up AND until the first fresh 0→1
    # transition after warm-up (investor starts in cash, waits for new buy signal).
    warmup = max(slow_range)
    warmup_date = price.index[warmup]
    print(f"    warm-up: {warmup} trading days trimmed (metrics start {warmup_date.date()})")

    def _apply_warmup(sig):
        """Force signal=0 during warm-up; if signal is already 1 at warm-up end,
        stay in cash until signal goes to 0 first, then follow normally."""
        sig_mod = sig.copy()
        sig_mod.loc[:warmup_date] = 0
        # If original signal was 1 at warm-up boundary, wait for it to go 0 first
        if sig.loc[warmup_date] == 1:
            orig_post = sig.loc[warmup_date:]
            first_zero = orig_post[orig_post == 0].index
            if len(first_zero) > 0:
                sig_mod.loc[warmup_date:first_zero[0]] = 0
        return sig_mod

    def _trim(cum):
        """Trim warm-up period and renormalize to 1.0."""
        trimmed = cum.loc[warmup_date:]
        return trimmed / trimmed.iloc[0]

    # --- B&H reference for each leverage ---
    for lev in leverage_list:
        bh = run_buy_and_hold(price, leverage=lev,
                              expense_ratio=expense_ratio if lev > 1 else 0.0)
        bh = _trim(bh)
        bh_metrics = calc_metrics(bh, tbill_rate=tbill_rate if isinstance(tbill_rate, (int, float)) else rf_series.mean() * 252 if rf_series is not None else 0.03,
                                  rf_series=rf_series)
        bh_mrd = _max_recovery_days(bh)
        results.append({
            "slow": 0, "fast": 0, "leverage": lev,
            "CAGR": bh_metrics["CAGR"], "Sharpe": bh_metrics["Sharpe"],
            "Sortino": bh_metrics["Sortino"],
            "Vol": bh_metrics["Volatility"], "MDD": bh_metrics["MDD"],
            "MDD_Entry": bh_metrics["MDD"],  # B&H: same as MDD (always in position)
            "Trades_Year": 0.0, "Total_Trades": 0,
            "Max_Recovery_Days": bh_mrd,
        })

    # --- Grid search ---
    combos = [(s, f) for s in slow_range for f in fast_range if f < s]
    total = len(combos) * len(leverage_list)
    done = 0

    for slow, fast in combos:
        sig_raw = signal_dual_ma(price, slow=slow, fast=fast)
        sig = _apply_warmup(sig_raw)

        for lev in leverage_list:
            cum = run_lrs(price, sig, leverage=lev, expense_ratio=expense_ratio,
                          tbill_rate=tbill_rate, signal_lag=signal_lag,
                          commission=commission)
            cum = _trim(cum)
            sig_trimmed = sig.loc[warmup_date:]

            tpy = signal_trades_per_year(sig_trimmed)
            total_flips = int((sig_trimmed.diff().abs() > 0).sum())

            tbill_scalar = (rf_series.mean() * 252 if rf_series is not None
                           else tbill_rate if isinstance(tbill_rate, (int, float)) else 0.03)
            m = calc_metrics(cum, tbill_rate=tbill_scalar, rf_series=rf_series)
            mrd = _max_recovery_days(cum)
            mdd_entry = _max_entry_drawdown(cum, sig, signal_lag)

            results.append({
                "slow": slow, "fast": fast, "leverage": lev,
                "CAGR": m["CAGR"], "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "Vol": m["Volatility"], "MDD": m["MDD"],
                "MDD_Entry": mdd_entry,
                "Trades_Year": tpy, "Total_Trades": total_flips,
                "Max_Recovery_Days": mrd,
            })
            done += 1

        if done % 500 < len(leverage_list):
            print(f"    progress: {done}/{total} ({done*100//total}%)")

    print(f"    progress: {done}/{total} (100%)")
    return pd.DataFrame(results)


def plot_heatmap(grid_df: pd.DataFrame, metric: str, title: str, fname: str,
                 leverage: float, slow_range: range, fast_range: range,
                 reverse_cmap: bool = False):
    """Plot a (slow × fast) heatmap for one metric at one leverage level.

    Markers:
      - White ★: single MA200 baseline (slow=200, fast=1)
      - Red ★: best combo for this metric
      - Dashed line annotation: B&H value
    """
    sub = grid_df[(grid_df["leverage"] == leverage) & (grid_df["slow"] > 0)].copy()
    if sub.empty:
        print(f"  [Warning] No data for heatmap: {fname}")
        return

    # Pivot to 2D
    pivot = sub.pivot_table(index="fast", columns="slow", values=metric, aggfunc="first")

    # Determine best: for MDD and Trades_Year, closer to 0 is better
    higher_is_better = metric not in ("MDD", "Trades_Year", "Total_Trades", "Max_Recovery_Days")
    if higher_is_better:
        best_idx = sub[metric].idxmax()
    else:
        # For MDD (negative), closest to 0 = max; for others, min
        if metric == "MDD":
            best_idx = sub[metric].idxmax()  # MDD is negative, max = closest to 0
        else:
            best_idx = sub[metric].idxmin()
    best_row = sub.loc[best_idx]
    best_slow, best_fast = int(best_row["slow"]), int(best_row["fast"])
    best_val = best_row[metric]

    # Baseline: slow=200, fast=1
    baseline_row = sub[(sub["slow"] == 200) & (sub["fast"] == 1)]
    has_baseline = len(baseline_row) > 0
    if has_baseline:
        baseline_val = baseline_row[metric].iloc[0]

    # B&H reference
    bh_rows = grid_df[(grid_df["leverage"] == leverage) & (grid_df["slow"] == 0)]
    bh_val = bh_rows[metric].iloc[0] if len(bh_rows) > 0 else None

    # Colormap
    cmap = "RdYlGn" if not reverse_cmap else "RdYlGn_r"

    fig, ax = plt.subplots(figsize=(16, 8))

    slow_vals = sorted(pivot.columns)
    fast_vals = sorted(pivot.index)
    data = pivot.reindex(index=fast_vals, columns=slow_vals).values

    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap,
                   extent=[slow_vals[0] - 1.5, slow_vals[-1] + 1.5,
                           fast_vals[0] - 0.5, fast_vals[-1] + 0.5])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    # Format colorbar labels
    if metric in ("CAGR", "MDD", "Vol"):
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    elif metric in ("Sharpe", "Sortino"):
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ★ markers
    if has_baseline:
        ax.plot(200, 1, marker="*", color="white", markersize=18,
                markeredgecolor="black", markeredgewidth=1.0, zorder=10)
    ax.plot(best_slow, best_fast, marker="*", color="red", markersize=18,
            markeredgecolor="black", markeredgewidth=1.0, zorder=10)

    # Annotation text
    fmt = _metric_fmt(metric)
    legend_parts = []
    legend_parts.append(f"★ Best: ({best_slow},{best_fast}) = {fmt(best_val)}")
    if has_baseline:
        legend_parts.append(f"★ MA200: (200,1) = {fmt(baseline_val)}")
    if bh_val is not None:
        legend_parts.append(f"--- B&H {leverage:.0f}x = {fmt(bh_val)}")

    ax.text(0.01, 0.99, "\n".join(legend_parts),
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            fontfamily="monospace")

    ax.set_xlabel("Long MA (slow)", fontsize=12)
    ax.set_ylabel("Short MA (fast)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")


def _metric_fmt(metric: str):
    """Return a formatter function for a metric."""
    if metric in ("CAGR", "MDD", "Vol"):
        return lambda v: f"{v:.2%}"
    elif metric in ("Sharpe", "Sortino"):
        return lambda v: f"{v:.2f}"
    elif metric in ("Trades_Year",):
        return lambda v: f"{v:.1f}"
    elif metric in ("Total_Trades", "Max_Recovery_Days"):
        return lambda v: f"{int(v):,}"
    else:
        return lambda v: f"{v:.4f}"


def plot_composite_heatmap(grid_df: pd.DataFrame, leverage: float,
                           slow_range: range, fast_range: range,
                           title_prefix: str, fname: str):
    """6-panel heatmap: CAGR, MDD, Max Recovery Days, Sharpe, Sortino + Composite Rank.

    Composite rank = average percentile rank across 5 metrics (higher = better).
    """
    sub = grid_df[(grid_df["leverage"] == leverage) & (grid_df["slow"] > 0)].copy()
    if sub.empty:
        print(f"  [Warning] No data for composite heatmap: {fname}")
        return

    # --- Compute composite rank (percentile, 0~1, higher = better) ---
    sub = sub.copy()
    sub["rank_CAGR"] = sub["CAGR"].rank(pct=True)
    sub["rank_Sharpe"] = sub["Sharpe"].rank(pct=True)
    sub["rank_Sortino"] = sub["Sortino"].rank(pct=True)
    sub["rank_MDD"] = sub["MDD"].rank(pct=True)           # MDD is negative, higher = shallower
    sub["rank_Recovery"] = sub["Max_Recovery_Days"].rank(pct=True, ascending=False)  # lower = better
    sub["Composite"] = (sub["rank_CAGR"] + sub["rank_Sharpe"] + sub["rank_Sortino"]
                        + sub["rank_MDD"] + sub["rank_Recovery"]) / 5.0

    # Panel definitions: (metric, label, cmap, fmt_func)
    panels = [
        ("CAGR",               "CAGR",                "RdYlGn",   lambda v: f"{v:.2%}"),
        ("Sharpe",             "Sharpe",               "RdYlGn",   lambda v: f"{v:.2f}"),
        ("Sortino",            "Sortino",              "RdYlGn",   lambda v: f"{v:.2f}"),
        ("MDD",                "Max Drawdown",         "RdYlGn",   lambda v: f"{v:.2%}"),
        ("Max_Recovery_Days",  "Max Recovery (days)",  "RdYlGn_r", lambda v: f"{int(v):,}"),
        ("Composite",          "Composite Rank",       "RdYlGn",   lambda v: f"{v:.3f}"),
    ]

    slow_vals = sorted(sub["slow"].unique())
    fast_vals = sorted(sub["fast"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()

    # B&H reference
    bh_rows = grid_df[(grid_df["leverage"] == leverage) & (grid_df["slow"] == 0)]

    for ax, (metric, label, cmap, fmt) in zip(axes, panels):
        pivot = sub.pivot_table(index="fast", columns="slow", values=metric, aggfunc="first")
        pivot = pivot.reindex(index=fast_vals, columns=slow_vals)
        data = pivot.values

        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap,
                       extent=[slow_vals[0] - 1.5, slow_vals[-1] + 1.5,
                               fast_vals[0] - 0.5, fast_vals[-1] + 0.5])
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        if metric in ("CAGR", "MDD"):
            cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        # Best marker
        higher_is_better = metric not in ("MDD", "Max_Recovery_Days")
        if higher_is_better:
            best_idx = sub[metric].idxmax()
        elif metric == "MDD":
            best_idx = sub[metric].idxmax()  # negative, max = shallowest
        else:
            best_idx = sub[metric].idxmin()
        best_row = sub.loc[best_idx]
        best_slow, best_fast = int(best_row["slow"]), int(best_row["fast"])
        best_val = best_row[metric]

        ax.plot(best_slow, best_fast, marker="*", color="red", markersize=14,
                markeredgecolor="black", markeredgewidth=0.8, zorder=10)

        # Annotation
        parts = [f"★ Best: ({best_slow},{best_fast}) = {fmt(best_val)}"]
        if len(bh_rows) > 0 and metric in bh_rows.columns:
            bh_val = bh_rows[metric].iloc[0]
            parts.append(f"B&H = {fmt(bh_val)}")
        ax.text(0.01, 0.99, "\n".join(parts), transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
                fontfamily="monospace")

        ax.set_xlabel("Slow MA", fontsize=10)
        ax.set_ylabel("Fast MA", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

    fig.suptitle(f"{title_prefix} ({leverage:.0f}x)", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  → saved {OUT_DIR / fname}")

    # Print composite top-10
    top10 = sub.nlargest(10, "Composite")
    print(f"\n  Composite Rank Top 10 ({leverage:.0f}x):")
    print(f"  {'slow':>5} {'fast':>5} {'Comp':>6} {'CAGR':>8} {'Sharpe':>7} {'Sortino':>8} {'MDD':>9} {'Recovery':>9}")
    for _, r in top10.iterrows():
        print(f"  {int(r['slow']):5d} {int(r['fast']):5d} {r['Composite']:6.3f} "
              f"{r['CAGR']:7.2%} {r['Sharpe']:7.3f} {r['Sortino']:8.3f} "
              f"{r['MDD']:8.2%} {int(r['Max_Recovery_Days']):8d}")


def run_dual_ma_analysis(price: pd.Series, label: str, safe_ticker: str,
                         expense_ratio: float, tbill_rate, signal_lag: int,
                         rf_series: pd.Series = None,
                         fast_range=None, slow_range=None, leverage_list=None,
                         commission: float = 0.0):
    """Run full dual MA grid search and generate all heatmaps for one ticker."""
    if slow_range is None:
        slow_range = range(50, 351, 3)   # 50,53,...,350 → 101 values
    if fast_range is None:
        fast_range = range(2, 51, 1)     # 2,3,...,50 → 49 values (fast=1 제외: look-ahead bias)
    if leverage_list is None:
        leverage_list = [1, 3]

    print(f"\n  Grid: slow {slow_range.start}-{slow_range.stop - 1} (step {slow_range.step}), "
          f"fast {fast_range.start}-{fast_range.stop - 1} (step {fast_range.step})")
    print(f"  Leverage levels: {leverage_list}")
    if commission > 0:
        print(f"  Commission: {commission:.2%} per trade")

    grid_df = run_dual_ma_grid(
        price, leverage_list=leverage_list,
        expense_ratio=expense_ratio, tbill_rate=tbill_rate,
        signal_lag=signal_lag, slow_range=slow_range, fast_range=fast_range,
        rf_series=rf_series, commission=commission,
    )

    # Save full grid results to CSV
    csv_path = OUT_DIR / f"{safe_ticker}_grid_results.csv"
    grid_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  → saved {csv_path}")

    # Metric definitions for heatmaps
    metrics = [
        ("Sharpe",             "Sharpe Ratio",       False),
        ("Sortino",            "Sortino Ratio",      False),
        ("CAGR",               "CAGR",               False),
        ("Trades_Year",        "Trades / Year",      True),
        ("Total_Trades",       "Total Trades",       True),
        ("MDD",                "Max Drawdown",       True),
        ("Max_Recovery_Days",  "Max Recovery Days",  True),
    ]

    for lev in leverage_list:
        for metric, metric_label, reverse in metrics:
            title = f"{label} — {metric_label} ({lev}x)"
            fname = f"{safe_ticker}_grid_{metric.lower()}_{lev}x.png"
            plot_heatmap(grid_df, metric, title, fname,
                         leverage=lev, slow_range=slow_range, fast_range=fast_range,
                         reverse_cmap=reverse)

        # Composite 6-panel heatmap
        plot_composite_heatmap(grid_df, leverage=lev,
                               slow_range=slow_range, fast_range=fast_range,
                               title_prefix=label,
                               fname=f"{safe_ticker}_composite_{lev}x.png")

    # --- Summary tables ---
    for lev in leverage_list:
        sub = grid_df[(grid_df["leverage"] == lev) & (grid_df["slow"] > 0)]
        bh = grid_df[(grid_df["leverage"] == lev) & (grid_df["slow"] == 0)]

        top20 = sub.nlargest(20, "Sortino")
        baseline = sub[(sub["slow"] == 200) & (sub["fast"] == 1)]

        print(f"\n{'=' * 90}")
        print(f"  {label} — Top 20 by Sortino ({lev}x)")
        print(f"{'=' * 90}")
        display_cols = ["slow", "fast", "Sortino", "Sharpe", "CAGR", "Vol", "MDD", "MDD_Entry",
                        "Trades_Year", "Max_Recovery_Days"]
        fmt_df = top20[display_cols].copy()
        fmt_df["CAGR"] = fmt_df["CAGR"].map("{:.2%}".format)
        fmt_df["Vol"] = fmt_df["Vol"].map("{:.2%}".format)
        fmt_df["MDD"] = fmt_df["MDD"].map("{:.2%}".format)
        fmt_df["MDD_Entry"] = fmt_df["MDD_Entry"].map("{:.2%}".format)
        fmt_df["Sortino"] = fmt_df["Sortino"].map("{:.3f}".format)
        fmt_df["Sharpe"] = fmt_df["Sharpe"].map("{:.3f}".format)
        fmt_df["Trades_Year"] = fmt_df["Trades_Year"].map("{:.1f}".format)
        fmt_df["Max_Recovery_Days"] = fmt_df["Max_Recovery_Days"].astype(int)
        print(fmt_df.to_string(index=False))

        # Direct comparison: B&H vs MA200 vs Best Sortino
        print(f"\n  --- Comparison ({lev}x) ---")
        if len(bh) > 0:
            b = bh.iloc[0]
            print(f"  B&H {lev}x:       Sortino={b['Sortino']:.3f}  Sharpe={b['Sharpe']:.3f}  CAGR={b['CAGR']:.2%}  MDD={b['MDD']:.2%}  Recovery={int(b['Max_Recovery_Days'])}d")
        if len(baseline) > 0:
            b = baseline.iloc[0]
            print(f"  MA200 (200,1):   Sortino={b['Sortino']:.3f}  Sharpe={b['Sharpe']:.3f}  CAGR={b['CAGR']:.2%}  MDD={b['MDD']:.2%}  Recovery={int(b['Max_Recovery_Days'])}d  Trades/Yr={b['Trades_Year']:.1f}")
        best = sub.loc[sub["Sortino"].idxmax()]
        print(f"  Best Sortino:    ({int(best['slow'])},{int(best['fast'])})  Sortino={best['Sortino']:.3f}  Sharpe={best['Sharpe']:.3f}  CAGR={best['CAGR']:.2%}  MDD={best['MDD']:.2%}  Recovery={int(best['Max_Recovery_Days'])}d  Trades/Yr={best['Trades_Year']:.1f}")


# ──────────────────────────────────────────────
# 8b. EULB COMPARISON HELPERS
# ──────────────────────────────────────────────

def run_eulb1_comparison(ndx_price, commission=0.002):
    """Run eulb 1편 key combo comparison and B&H 3x reference.

    Returns (eulb_combos, bh3x_metrics) for reuse in Part 12.
    """
    print("\n" + "=" * 70)
    print("  eulb 1편 보고값 vs 재현값 비교 (lag=1, commission=0.2%)")
    print("=" * 70)

    eulb_combos = [
        (4, 80,  "eulb 1편 전체기간 최적"),
        (3, 220, "eulb 1편 버블 이후 최적"),
        (3, 216, "eulb 1편 세밀 최적화"),
        (3, 161, "eulb 5편 최종 추천"),
    ]

    for fast, slow, desc in eulb_combos:
        sig = signal_dual_ma(ndx_price, slow=slow, fast=fast)
        cum = run_lrs(ndx_price, sig, leverage=3.0, expense_ratio=0.0,
                      tbill_rate=0.0, signal_lag=1, commission=commission)
        m = calc_metrics(cum, tbill_rate=0.0)
        tpy = signal_trades_per_year(sig)
        print(f"  ({fast:2d}, {slow:3d}) [{desc}]")
        print(f"    CAGR={m['CAGR']:.3%}  Total={m['Total Return']:.1f}x  "
              f"Sharpe={m['Sharpe']:.3f}  MDD={m['MDD']:.2%}  Trades/Yr={tpy:.1f}")

    # B&H 3x reference
    bh3x = run_buy_and_hold(ndx_price, leverage=3.0, expense_ratio=0.0)
    bh3x_m = calc_metrics(bh3x, tbill_rate=0.0)
    print(f"\n  B&H 3x:  CAGR={bh3x_m['CAGR']:.3%}  Total={bh3x_m['Total Return']:.1f}x  "
          f"MDD={bh3x_m['MDD']:.2%}")

    print(f"\n  참고 eulb 보고값:")
    print(f"    (4, 80):   CAGR≈28.554%")
    print(f"    (3, 220):  CAGR≈27.482% (post-2002)")
    print(f"    (3, 216):  Total≈32183%")
    print(f"    (3, 161):  eulb 5편 추천")
    print(f"    B&H 3x:   Total≈3448%")

    return eulb_combos, bh3x_m


def run_eulb5_spotcheck(ndx_eulb5, commission=0.002):
    """Run eulb 5편 (3, 161) spot-check and B&H 3x for the 2006-2024 period."""
    print(f"\n  --- eulb 5편 (3, 161) 직접 확인 (2006-2024, lag=1, comm=0.2%) ---")
    sig_161 = signal_dual_ma(ndx_eulb5, slow=161, fast=3)
    cum_161 = run_lrs(ndx_eulb5, sig_161, leverage=3.0, expense_ratio=0.0,
                      tbill_rate=0.0, signal_lag=1, commission=commission)
    m_161 = calc_metrics(cum_161, tbill_rate=0.0)
    tpy_161 = signal_trades_per_year(sig_161)
    print(f"  (3, 161): CAGR={m_161['CAGR']:.3%}  Total={m_161['Total Return']:.1f}x  "
          f"Sharpe={m_161['Sharpe']:.3f}  MDD={m_161['MDD']:.2%}  Trades/Yr={tpy_161:.1f}")

    # B&H 3x for this period
    bh3x_5 = run_buy_and_hold(ndx_eulb5, leverage=3.0, expense_ratio=0.0)
    bh3x_5m = calc_metrics(bh3x_5, tbill_rate=0.0)
    print(f"  B&H 3x:  CAGR={bh3x_5m['CAGR']:.3%}  Total={bh3x_5m['Total Return']:.1f}x  "
          f"MDD={bh3x_5m['MDD']:.2%}")


def run_part12_comparison(ndx_price, rf_series, eulb_combos, bh3x_m,
                          calibrated_er=0.035, commission=0.002):
    """Part 12 vs Part 10 comparison + TQQQ 2010-present validation."""
    print("\n" + "=" * 70)
    print("  Part 12 vs Part 10: eulb 주요 조합 비교 (calibrated vs eulb 조건)")
    print(f"  Part 12: ER={calibrated_er:.2%}, tbill=Ken French RF, lag=1, comm=0.2%")
    print(f"  Part 10: ER=0%, tbill=0, lag=1, comm=0.2%")
    print("=" * 70)

    for fast, slow, desc in eulb_combos:
        # Part 12 conditions (calibrated)
        sig = signal_dual_ma(ndx_price, slow=slow, fast=fast)
        cum_cal = run_lrs(ndx_price, sig, leverage=3.0,
                          expense_ratio=calibrated_er,
                          tbill_rate=rf_series,
                          signal_lag=1, commission=commission)
        m_cal = calc_metrics(cum_cal, rf_series=rf_series)

        # Part 10 conditions (eulb)
        cum_eulb = run_lrs(ndx_price, sig, leverage=3.0,
                           expense_ratio=0.0, tbill_rate=0.0,
                           signal_lag=1, commission=commission)
        m_eulb = calc_metrics(cum_eulb, tbill_rate=0.0)

        tpy = signal_trades_per_year(sig)
        print(f"  ({fast:2d}, {slow:3d}) [{desc}]")
        print(f"    Calibrated: CAGR={m_cal['CAGR']:.3%}  Total={m_cal['Total Return']:>10.1f}x  Sharpe={m_cal['Sharpe']:.3f}")
        print(f"    eulb cond:  CAGR={m_eulb['CAGR']:.3%}  Total={m_eulb['Total Return']:>10.1f}x  Sharpe={m_eulb['Sharpe']:.3f}")
        print(f"    CAGR diff:  {(m_cal['CAGR'] - m_eulb['CAGR'])*100:+.2f}pp")

    # B&H 3x calibrated
    bh3x_cal = run_buy_and_hold(ndx_price, leverage=3.0, expense_ratio=calibrated_er)
    bh3x_cal_m = calc_metrics(bh3x_cal, rf_series=rf_series)
    print(f"\n  B&H 3x (calibrated): CAGR={bh3x_cal_m['CAGR']:.3%}  "
          f"Total={bh3x_cal_m['Total Return']:.1f}x  MDD={bh3x_cal_m['MDD']:.2%}")
    print(f"  B&H 3x (eulb cond):  CAGR={bh3x_m['CAGR']:.3%}  "
          f"Total={bh3x_m['Total Return']:.1f}x  MDD={bh3x_m['MDD']:.2%}")

    # --- 2010-present validation: B&H 3x calibrated vs actual TQQQ ---
    print(f"\n  --- 2010-present: Calibrated B&H 3x vs TQQQ (sanity check) ---")
    try:
        qqq_val = download("QQQ", start="2010-02-11", end="2025-12-31")
        tqqq_val = download("TQQQ", start="2010-02-11", end="2025-12-31")
        common_val = qqq_val.index.intersection(tqqq_val.index)
        qqq_val = qqq_val.loc[common_val]
        tqqq_val = tqqq_val.loc[common_val]

        sim_cal = run_buy_and_hold(qqq_val, leverage=3.0, expense_ratio=calibrated_er)
        actual_tqqq = run_buy_and_hold(tqqq_val, leverage=1.0, expense_ratio=0.0)
        m_sim = calc_metrics(sim_cal)
        m_actual = calc_metrics(actual_tqqq)
        print(f"  Sim 3x (ER={calibrated_er:.2%}): CAGR={m_sim['CAGR']:.3%}  Total={m_sim['Total Return']:.1f}x")
        print(f"  TQQQ (actual):        CAGR={m_actual['CAGR']:.3%}  Total={m_actual['Total Return']:.1f}x")
        print(f"  CAGR diff:            {(m_sim['CAGR'] - m_actual['CAGR'])*100:+.3f}pp")
    except Exception as e:
        print(f"  [Warning] Validation skipped: {e}")


# ──────────────────────────────────────────────
# 9. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Leverage for the Long Run  - Strategy Replication")
    print("  Based on Michael Gayed (2016 NAAIM Dow Award)")
    print("=" * 70)

    # Try to fetch live T-Bill rate for modern analysis
    tbill_live = get_tbill_rate_from_irx()
    print(f"\n  Live T-Bill rate: {tbill_live:.2%}")
    print(f"  Paper T-Bill rate: Ken French daily series")

    # ══════════════════════════════════════════════
    # Part 1: PAPER REPLICATION (Oct 1928 - Dec 2020)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 1: Paper Table 8 Replication (1928-2020)")
    print("=" * 70)
    run_analysis(PAPER_CONFIG)

    # ══════════════════════════════════════════════
    # Part 1.5: NASDAQ COMPOSITE LONG-RUN (1971 - present)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 1.5: NASDAQ Composite Long-Run (1971-present)")
    print("  NOTE: Price-return only (no total return data for NASDAQ)")
    print("=" * 70)
    run_analysis(NASDAQ_LONGRUN_CONFIG)

    # ══════════════════════════════════════════════
    # Part 2: MODERN ANALYSIS (1990 - present)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 2: Modern Analysis (1990-present)")
    print("=" * 70)
    modern_config = {**DEFAULT_CONFIG, "tbill_rate": tbill_live}
    run_analysis(modern_config)

    # Part 3: Extended MA periods (modern)
    extended_config = {
        **modern_config,
        "ma_periods": [50, 100, 200],
    }
    run_analysis(extended_config)

    # Part 4: Nasdaq-100
    ndx_config = {
        **modern_config,
        "ticker": "^NDX",
        "start": "1990-01-01",
        "ma_periods": [200],
    }
    run_analysis(ndx_config)

    # Part 5: ETF comparison (actual vs simulated)
    run_etf_comparison(modern_config)

    # Part 6: Dual MA signal example
    dual_config = {
        **modern_config,
        "signal_fn": signal_dual_ma,
        "ma_periods": [200],
    }
    run_analysis(dual_config)

    # ══════════════════════════════════════════════
    # Part 7: Dual MA Grid Search - S&P 500 (1928-2020, total return)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 7: Dual MA Grid Search — S&P 500 (1928-2020, Total Return)")
    print("  Grid: slow 50-350 (step 3) × fast 2-50 (step 1) × leverage 1x,3x")
    print("=" * 70)

    rf_series_grid = download_ken_french_rf()
    sp_price = download("^GSPC", "1928-10-01", "2020-12-31", total_return=True)
    print(f"  Downloaded {len(sp_price)} trading days ({sp_price.index[0].date()} -> {sp_price.index[-1].date()})")

    run_dual_ma_analysis(
        price=sp_price,
        label="S&P 500 (1928-2020, TR)",
        safe_ticker="GSPC_TR",
        expense_ratio=0.01,
        tbill_rate=rf_series_grid,
        signal_lag=0,
        rf_series=rf_series_grid,
    )

    # ══════════════════════════════════════════════
    # Part 8: Dual MA Grid Search - NASDAQ Composite ^IXIC (1971-2025)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 8: Dual MA Grid Search — NASDAQ Composite ^IXIC (1971-2025)")
    print("  Grid: slow 50-350 (step 3) × fast 2-50 (step 1) × leverage 1x,3x")
    print("=" * 70)

    nq_price = download("^IXIC", "1971-01-01", "2025-12-31", total_return=False)
    print(f"  Downloaded {len(nq_price)} trading days ({nq_price.index[0].date()} -> {nq_price.index[-1].date()})")

    run_dual_ma_analysis(
        price=nq_price,
        label="NASDAQ Composite ^IXIC (1971-2025)",
        safe_ticker="IXIC",
        expense_ratio=0.01,
        tbill_rate=rf_series_grid,
        signal_lag=0,
        rf_series=rf_series_grid,
    )

    # ══════════════════════════════════════════════
    # Part 9: Dual MA Grid Search - Nasdaq-100 ^NDX (1985-2025)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 9: Dual MA Grid Search — Nasdaq-100 ^NDX (1985-2025)")
    print("  Grid: slow 50-350 (step 3) × fast 2-50 (step 1) × leverage 1x,3x")
    print("  NOTE: QQQ/TQQQ track this index. Compare with ^IXIC (Part 8).")
    print("=" * 70)

    ndx_price = download("^NDX", "1985-01-01", "2025-12-31", total_return=False)
    print(f"  Downloaded {len(ndx_price)} trading days ({ndx_price.index[0].date()} -> {ndx_price.index[-1].date()})")

    run_dual_ma_analysis(
        price=ndx_price,
        label="Nasdaq-100 ^NDX (1985-2025)",
        safe_ticker="NDX",
        expense_ratio=0.01,
        tbill_rate=rf_series_grid,
        signal_lag=0,
        rf_series=rf_series_grid,
    )

    # ══════════════════════════════════════════════
    # Part 10: eulb 1편 조건 재현 — ^NDX (1985-2025), expense=0, tbill=0
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 10: eulb 1편 재현 — ^NDX (1985-2025)")
    print("  조건: expense=0, tbill=0, lag=1, commission=0.2%, 3x only")
    print("  Grid: slow 50-350 (step 3) × fast 2-49 (step 1)")
    print("=" * 70)

    # Reuse ndx_price from Part 9 (already downloaded ^NDX 1985-2025)
    eulb1_fast = range(2, 50)    # eulb grid: n1=1~49, fast=1 제외 → 2~49
    eulb1_slow = range(50, 351, 3)
    eulb_commission = 0.002      # 0.2% per trade (backtesting.py convention)

    run_dual_ma_analysis(
        price=ndx_price,
        label="eulb 1편 재현 — ^NDX (1985-2025, lag=1, comm=0.2%)",
        safe_ticker="NDX_eulb1",
        expense_ratio=0.0,
        tbill_rate=0.0,
        signal_lag=1,
        rf_series=None,
        fast_range=eulb1_fast,
        slow_range=eulb1_slow,
        leverage_list=[3],
        commission=eulb_commission,
    )

    eulb_combos, bh3x_m = run_eulb1_comparison(ndx_price, commission=eulb_commission)

    # ══════════════════════════════════════════════
    # Part 11: eulb 5편 조건 재현 — ^NDX (2006-06-21 ~ 2024-06-09)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 11: eulb 5편 재현 — ^NDX (2006-06-21 ~ 2024-06-09)")
    print("  조건: expense=0, tbill=0, lag=1, commission=0.2%, 3x only")
    print("  이 기간에서 (3, 161) 근처가 최적인지 확인")
    print("=" * 70)

    ndx_eulb5 = download("^NDX", "2006-06-21", "2024-06-09", total_return=False)
    print(f"  Downloaded {len(ndx_eulb5)} trading days ({ndx_eulb5.index[0].date()} -> {ndx_eulb5.index[-1].date()})")

    run_dual_ma_analysis(
        price=ndx_eulb5,
        label="eulb 5편 재현 — ^NDX (2006-2024, lag=1, comm=0.2%)",
        safe_ticker="NDX_eulb5",
        expense_ratio=0.0,
        tbill_rate=0.0,
        signal_lag=1,
        rf_series=None,
        fast_range=eulb1_fast,
        slow_range=eulb1_slow,
        leverage_list=[3],
        commission=eulb_commission,
    )

    run_eulb5_spotcheck(ndx_eulb5, commission=eulb_commission)

    # ══════════════════════════════════════════════
    # Part 12: TQQQ-Calibrated NDX Grid Search (1985-2025)
    # ══════════════════════════════════════════════
    # Calibrated via calibrate_tqqq.py: fixed ER that best matches actual TQQQ.
    CALIBRATED_ER = 0.035

    print("\n" + "=" * 70)
    print("  PART 12: TQQQ-Calibrated NDX Grid Search (1985-2025)")
    print(f"  조건: expense={CALIBRATED_ER:.2%} (calibrated), tbill=Ken French RF, lag=1, comm=0.2%, 3x only")
    print("  Grid: slow 50-350 (step 3) × fast 2-50 (step 1)")
    print("  핵심 변경: expense → calibrated, tbill → Ken French RF, lag → 1")
    print("=" * 70)

    # Reuse ndx_price from Part 9 and rf_series_grid from Part 7
    run_dual_ma_analysis(
        price=ndx_price,
        label="NDX Calibrated (1985-2025, lag=1, TQQQ-calibrated costs)",
        safe_ticker="NDX_calibrated",
        expense_ratio=CALIBRATED_ER,
        tbill_rate=rf_series_grid,
        signal_lag=1,
        rf_series=rf_series_grid,
        fast_range=range(2, 51),
        slow_range=range(50, 351, 3),
        leverage_list=[3],
        commission=0.002,
    )

    run_part12_comparison(ndx_price, rf_series_grid, eulb_combos, bh3x_m,
                          calibrated_er=CALIBRATED_ER, commission=eulb_commission)

    print("\n" + "=" * 70)
    print("  Done! Charts saved in:", OUT_DIR)
    print("=" * 70)
