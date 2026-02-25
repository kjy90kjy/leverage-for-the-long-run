"""
Diagnostic: NASDAQ (^IXIC) data quality vs S&P 500 (^GSPC) for dual MA grid search.
"""
import sys, io, warnings, os
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leverage_rotation import download, signal_ma, signal_dual_ma, signal_trades_per_year

SEP = "=" * 70


def main():
    print(SEP)
    print("1. DOWNLOADING DATA")
    print(SEP)

    print("\n[Downloading ^GSPC (1928-2020, total return via Shiller)...]")
    gspc = download("^GSPC", start="1928-01-01", end="2020-12-31", total_return=True)
    print("  ^GSPC downloaded: {} rows".format(len(gspc)))

    print("\n[Downloading ^IXIC (1971-2025)...]")
    ixic = download("^IXIC", start="1971-01-01", end="2025-12-31", total_return=False)
    print("  ^IXIC downloaded: {} rows".format(len(ixic)))

    print("\n" + SEP)
    print("2. BASIC STATS COMPARISON")
    print(SEP)

    for label, s in [("^GSPC", gspc), ("^IXIC", ixic)]:
        print("\n  {}:".format(label))
        print("    Date range:  {} -> {}".format(s.index[0].date(), s.index[-1].date()))
        print("    Length:      {} trading days ({:.1f} years)".format(len(s), len(s)/252))
        print("    First price: {:.2f}".format(s.iloc[0]))
        print("    Last price:  {:.2f}".format(s.iloc[-1]))
        nan_count = s.isna().sum()
        print("    NaN values:  {}".format(nan_count))
        date_diffs = pd.Series(s.index).diff().dt.days
        big_gaps = date_diffs[date_diffs > 5]
        if len(big_gaps) > 0:
            print("    Gaps > 5 calendar days: {}".format(len(big_gaps)))
            for idx in big_gaps.index[:5]:
                print("      {} -> {} ({:.0f} days)".format(
                    s.index[idx-1].date(), s.index[idx].date(), date_diffs[idx]))
            if len(big_gaps) > 5:
                print("      ... and {} more".format(len(big_gaps)-5))
        else:
            print("    Gaps > 5 calendar days: 0")

    print("\n" + SEP)
    print("3. SUSPICIOUS PRICE JUMPS (|daily return| > 15%)")
    print(SEP)

    for label, s in [("^GSPC", gspc), ("^IXIC", ixic)]:
        daily_ret = s.pct_change().dropna()
        big_moves = daily_ret[daily_ret.abs() > 0.15]
        print("\n  {}: {} days with |return| > 15%".format(label, len(big_moves)))
        if len(big_moves) > 0:
            for date, ret in big_moves.items():
                print("    {}: {:+.2%}".format(date.date(), ret))

    print("\n  --- Moves > 10% ---")
    for label, s in [("^GSPC", gspc), ("^IXIC", ixic)]:
        daily_ret = s.pct_change().dropna()
        big_moves = daily_ret[daily_ret.abs() > 0.10]
        print("\n  {}: {} days with |return| > 10%".format(label, len(big_moves)))
        if len(big_moves) > 0:
            for date, ret in big_moves.items():
                print("    {}: {:+.2%}".format(date.date(), ret))

    print("\n" + SEP)
    print("4. RETURN CHARACTERISTICS COMPARISON")
    print(SEP)

    for label, s in [("^GSPC", gspc), ("^IXIC", ixic)]:
        daily_ret = s.pct_change().dropna()
        n_years = len(daily_ret) / 252
        total = s.iloc[-1] / s.iloc[0]
        cagr = total ** (1 / n_years) - 1
        vol = daily_ret.std() * np.sqrt(252)
        ac1 = daily_ret.autocorr(lag=1)
        ac5 = daily_ret.autocorr(lag=5)
        ac20 = daily_ret.autocorr(lag=20)
        skw = daily_ret.skew()
        kurt = daily_ret.kurtosis()
        print("\n  {} ({} to {}):".format(label, s.index[0].date(), s.index[-1].date()))
        print("    Annualized Return (CAGR): {:.2%}".format(cagr))
        print("    Annualized Volatility:    {:.2%}".format(vol))
        print("    Sharpe (crude, rf=0):     {:.2f}".format(cagr/vol))
        print("    Autocorrelation lag-1:    {:.4f}".format(ac1))
        print("    Autocorrelation lag-5:    {:.4f}".format(ac5))
        print("    Autocorrelation lag-20:   {:.4f}".format(ac20))
        print("    Skewness:                 {:.4f}".format(skw))
        print("    Excess Kurtosis:          {:.4f}".format(kurt))
        print("    Daily mean:               {:.6f}".format(daily_ret.mean()))
        print("    Daily std:                {:.6f}".format(daily_ret.std()))

    print("\n  --- COMMON PERIOD COMPARISON (overlapping dates) ---")
    common_start = max(gspc.index[0], ixic.index[0])
    common_end = min(gspc.index[-1], ixic.index[-1])
    gspc_common = gspc.loc[common_start:common_end]
    ixic_common = ixic.loc[common_start:common_end]

    for label, s in [("^GSPC (common)", gspc_common), ("^IXIC (common)", ixic_common)]:
        daily_ret = s.pct_change().dropna()
        n_years = len(daily_ret) / 252
        total = s.iloc[-1] / s.iloc[0]
        cagr = total ** (1 / n_years) - 1
        vol = daily_ret.std() * np.sqrt(252)
        print("\n  {} ({} to {}):".format(label, s.index[0].date(), s.index[-1].date()))
        print("    CAGR: {:.2%}   Vol: {:.2%}   Sharpe(crude): {:.2f}".format(cagr, vol, cagr/vol))

    print("\n" + SEP)
    print("5. MA SIGNAL BEHAVIOR COMPARISON")
    print(SEP)

    for label, s in [("^GSPC", gspc), ("^IXIC", ixic)]:
        print("\n  {}:".format(label))
        for ma_period in [50, 100, 200]:
            sig = signal_ma(s, period=ma_period)
            sig_valid = sig.dropna()
            pct_above = sig_valid.mean()
            trades = signal_trades_per_year(sig_valid)
            print("    MA{}: {:.1%} time above | {:.1f} signal flips/year".format(
                ma_period, pct_above, trades))

        print("  Dual MA signals:")
        for slow, fast in [(200, 50), (100, 25), (77, 19), (50, 12)]:
            sig = signal_dual_ma(s, slow=slow, fast=fast)
            sig_valid = sig.dropna()
            pct_above = sig_valid.mean()
            trades = signal_trades_per_year(sig_valid)
            print("    DualMA({}/{}): {:.1%} time above | {:.1f} flips/year".format(
                fast, slow, pct_above, trades))

    print("\n" + SEP)
    print("6. PRICE-ONLY vs TOTAL RETURN: DIVIDEND YIELD EFFECT")
    print(SEP)

    print("\n[Downloading ^GSPC price-only for comparison...]")
    gspc_price_only = download("^GSPC", start="1971-01-01", end="2020-12-31", total_return=False)
    gspc_tr = download("^GSPC", start="1971-01-01", end="2020-12-31", total_return=True)

    n_years_po = len(gspc_price_only) / 252
    cagr_po = (gspc_price_only.iloc[-1] / gspc_price_only.iloc[0]) ** (1/n_years_po) - 1
    cagr_tr = (gspc_tr.iloc[-1] / gspc_tr.iloc[0]) ** (1/n_years_po) - 1
    div_effect = cagr_tr - cagr_po

    print("\n  ^GSPC 1971-2020:")
    print("    Price-only CAGR:     {:.2%}".format(cagr_po))
    print("    Total Return CAGR:   {:.2%}".format(cagr_tr))
    print("    Implied div yield:   {:.2%} per year".format(div_effect))

    n_years_ixic = len(ixic) / 252
    cagr_ixic = (ixic.iloc[-1] / ixic.iloc[0]) ** (1/n_years_ixic) - 1
    print("\n  ^IXIC (price-only) CAGR: {:.2%}".format(cagr_ixic))
    print("  NASDAQ historical dividend yield ~0.5-1.5% (tech-heavy, lower dividends)")
    print("  Estimated NASDAQ total-return CAGR: ~{:.2%} to ~{:.2%}".format(
        cagr_ixic + 0.008, cagr_ixic + 0.015))
    print("\n  KEY INSIGHT: ^GSPC uses total return (with Shiller dividends) adding ~{:.1%}/yr,".format(div_effect))
    print("  while ^IXIC is price-only. This affects B&H benchmark but NOT MA signal.")
    print("  MA signals are computed on price levels regardless.")

    print("\n" + SEP)
    print("7. FIRST AND LAST ROWS OF ^IXIC DATA")
    print(SEP)

    print("\n  First 10 rows:")
    for i in range(min(10, len(ixic))):
        print("    {}: {:.2f}".format(ixic.index[i].date(), ixic.iloc[i]))

    print("\n  Last 10 rows:")
    for i in range(max(0, len(ixic)-10), len(ixic)):
        print("    {}: {:.2f}".format(ixic.index[i].date(), ixic.iloc[i]))

    print("\n" + SEP)
    print("8. STARTING DATE VERIFICATION")
    print(SEP)

    expected_start = pd.Timestamp("1971-02-05")
    actual_start = ixic.index[0]
    print("\n  Expected ^IXIC start: around Feb 5, 1971")
    print("  Actual ^IXIC start:   {}".format(actual_start.date()))
    diff_days = abs((actual_start - expected_start).days)
    if diff_days < 30:
        print("  STATUS: OK (within {} days of expected)".format(diff_days))
    else:
        print("  STATUS: WARNING - {} days off from expected start!".format(diff_days))

    print("\n" + SEP)
    print("9. ADDITIONAL: WHY SHORT SLOW MAs MIGHT DOMINATE FOR NASDAQ")
    print(SEP)

    daily_ret_ixic = ixic.pct_change().dropna()

    vol_60d = daily_ret_ixic.rolling(60).std() * np.sqrt(252)
    vol_250d = daily_ret_ixic.rolling(250).std() * np.sqrt(252)

    print("\n  ^IXIC rolling volatility stats:")
    print("    60-day rolling vol:  mean={:.2%}, median={:.2%}".format(vol_60d.mean(), vol_60d.median()))
    print("    250-day rolling vol: mean={:.2%}, median={:.2%}".format(vol_250d.mean(), vol_250d.median()))

    print("\n  Trend persistence (autocorrelation of N-day returns):")
    for n in [5, 10, 20, 50, 100, 200]:
        nday_ret = ixic.pct_change(n).dropna()
        ac = nday_ret.autocorr(lag=n)
        print("    {:3d}-day return autocorr(lag={}): {:.4f}".format(n, n, ac))

    print("\n  MA signal forward return analysis (annualized avg daily return):")
    for label, s in [("^GSPC", gspc), ("^IXIC", ixic)]:
        print("\n    {}:".format(label))
        daily_ret_s = s.pct_change()
        for ma_period in [50, 77, 100, 150, 200, 250]:
            sig = signal_ma(s, period=ma_period)
            sig_valid = sig.reindex(daily_ret_s.index)
            above_ret = daily_ret_s[sig_valid == 1].mean() * 252
            below_ret = daily_ret_s[sig_valid == 0].mean() * 252
            spread = above_ret - below_ret
            n_above = (sig_valid == 1).sum()
            n_below = (sig_valid == 0).sum()
            print("      MA{:3d}: above={:+.2%}(n={}), below={:+.2%}(n={}), spread={:+.2%}".format(
                ma_period, above_ret, n_above, below_ret, n_below, spread))

    print("\n  --- DOT-COM BUBBLE EFFECT ON NASDAQ ---")
    periods = [
        ("Pre-bubble (1971-1995)", "1971-01-01", "1995-12-31"),
        ("Bubble (1996-2000)", "1996-01-01", "2000-03-10"),
        ("Crash (2000-2002)", "2000-03-11", "2002-10-09"),
        ("Recovery (2003-2007)", "2003-01-01", "2007-12-31"),
        ("GFC (2008-2009)", "2008-01-01", "2009-03-09"),
        ("Post-GFC (2010-2019)", "2010-01-01", "2019-12-31"),
        ("COVID+ (2020-2025)", "2020-01-01", "2025-12-31"),
    ]

    for pname, pstart, pend in periods:
        mask = (ixic.index >= pstart) & (ixic.index <= pend)
        sub = ixic[mask]
        if len(sub) < 20:
            continue
        n_yr = len(sub) / 252
        cagr_p = (sub.iloc[-1] / sub.iloc[0]) ** (1/n_yr) - 1
        vol_p = sub.pct_change().dropna().std() * np.sqrt(252)
        print("    {:30s}: CAGR={:+.2%}, Vol={:.2%}, start={:.0f}, end={:.0f}".format(
            pname, cagr_p, vol_p, sub.iloc[0], sub.iloc[-1]))

    print("\n  --- TIME BELOW MA (months) DURING DOT-COM CRASH (2000-2003) ---")
    crash_mask = (ixic.index >= "2000-01-01") & (ixic.index <= "2003-12-31")
    ixic_crash = ixic[crash_mask]
    for ma_period in [50, 77, 100, 150, 200, 250]:
        sig = signal_ma(ixic, period=ma_period)
        sig_crash = sig[crash_mask]
        pct_below = (sig_crash == 0).mean()
        months_below = pct_below * len(sig_crash) / 21
        print("    MA{:3d}: {:.1%} of time below ({:.1f} months out of ~48)".format(
            ma_period, pct_below, months_below))

    print("\n  --- DUAL MA SIGNAL DURING DOT-COM CRASH (2000-2003) ---")
    for slow, fast in [(200, 50), (150, 37), (100, 25), (77, 19), (50, 12)]:
        sig = signal_dual_ma(ixic, slow=slow, fast=fast)
        sig_crash = sig[crash_mask]
        pct_risk_off = (sig_crash == 0).mean()
        print("    DualMA({:2d}/{:3d}): {:.1%} of time risk-off during crash".format(
            fast, slow, pct_risk_off))

    print("\n  --- SAMPLE SIZE EFFECT ---")
    print("    ^GSPC: {:.0f} years of data".format(len(gspc)/252))
    print("    ^IXIC: {:.0f} years of data".format(len(ixic)/252))
    print("    ^GSPC has {:.0f} more years of data".format(len(gspc)/252 - len(ixic)/252))
    print("    Shorter history = fewer regime changes = higher overfitting risk")
    print("    NASDAQ history is dominated by tech bubble + recovery pattern")

    print("\n" + SEP)
    print("SUMMARY OF FINDINGS")
    print(SEP)
    print("")
    print("Key questions answered:")
    print("")
    print("1. DATA QUALITY: Check output above for NaN gaps, suspicious jumps,")
    print("   and starting date.")
    print("")
    print("2. PRICE-ONLY vs TOTAL RETURN: ^GSPC uses Shiller total-return synthesis,")
    print("   while ^IXIC is price-only. This affects B&H benchmark but NOT the")
    print("   MA signal computation (signals are based on price levels).")
    print("")
    print("3. WHY SHORT SLOW MAs FOR NASDAQ? Likely explanations:")
    print("   a) NASDAQ is more volatile (~higher beta) -> faster MAs react sooner")
    print("   b) Dot-com bubble: slow MAs (200+) were late to exit and re-enter late")
    print("   c) NASDAQ may have stronger momentum at shorter timescales")
    print("   d) Only ~54 years of data -> grid search overfits to dot-com pattern")
    print("   e) Short slow MAs capture the right timescale for NASDAQ faster trends")
    print("")
    print("4. The most important check: Is the data itself wrong? -> See sections 3, 7, 8.")
    print("")


if __name__ == "__main__":
    main()
