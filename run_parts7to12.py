"""Run Parts 7-12 only — grid searches with Sortino + composite heatmaps."""

import sys
import io
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from leverage_rotation import (
    download, signal_dual_ma, run_lrs, run_buy_and_hold,
    calc_metrics, signal_trades_per_year, download_ken_french_rf,
    run_dual_ma_analysis,
)

if __name__ == "__main__":

    # Shared data
    print("Downloading Ken French RF...")
    rf_series_grid = download_ken_french_rf()

    # ══════════════════════════════════════════════
    # Part 7: Dual MA Grid Search — S&P 500 (1928-2020, Total Return)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 7: Dual MA Grid Search — S&P 500 (1928-2020, Total Return)")
    print("  Grid: slow 50-350 (step 3) × fast 2-50 (step 1) × leverage 1x,3x")
    print("=" * 70)

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
    # Part 8: Dual MA Grid Search — NASDAQ Composite ^IXIC (1971-2025)
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
    # Part 9: Dual MA Grid Search — Nasdaq-100 ^NDX (1985-2025)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 9: Dual MA Grid Search — Nasdaq-100 ^NDX (1985-2025)")
    print("  Grid: slow 50-350 (step 3) × fast 2-50 (step 1) × leverage 1x,3x")
    print("  NOTE: QQQ/TQQQ track this index.")
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
    # Part 10: eulb 1편 재현 — ^NDX (1985-2025)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 10: eulb 1편 재현 — ^NDX (1985-2025)")
    print("  조건: expense=0, tbill=0, lag=1, commission=0.2%, 3x only")
    print("  Grid: slow 50-350 (step 3) × fast 2-49 (step 1)")
    print("=" * 70)

    eulb1_fast = range(2, 50)
    eulb1_slow = range(50, 351, 3)
    eulb_commission = 0.002

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

    # --- eulb 보고값 vs 재현값 비교 ---
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
                      tbill_rate=0.0, signal_lag=1, commission=eulb_commission)
        m = calc_metrics(cum, tbill_rate=0.0)
        tpy = signal_trades_per_year(sig)
        print(f"  ({fast:2d}, {slow:3d}) [{desc}]")
        print(f"    CAGR={m['CAGR']:.3%}  Total={m['Total Return']:.1f}x  "
              f"Sharpe={m['Sharpe']:.3f}  MDD={m['MDD']:.2%}  Trades/Yr={tpy:.1f}")

    bh3x = run_buy_and_hold(ndx_price, leverage=3.0, expense_ratio=0.0)
    bh3x_m = calc_metrics(bh3x, tbill_rate=0.0)
    print(f"\n  B&H 3x:  CAGR={bh3x_m['CAGR']:.3%}  Total={bh3x_m['Total Return']:.1f}x  "
          f"MDD={bh3x_m['MDD']:.2%}")

    # ══════════════════════════════════════════════
    # Part 11: eulb 5편 재현 — ^NDX (2006-2024)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 11: eulb 5편 재현 — ^NDX (2006-06-21 ~ 2024-06-09)")
    print("  조건: expense=0, tbill=0, lag=1, commission=0.2%, 3x only")
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

    # --- eulb 5편 (3, 161) 직접 확인 ---
    print(f"\n  --- eulb 5편 (3, 161) 직접 확인 (2006-2024, lag=1, comm=0.2%) ---")
    sig_161 = signal_dual_ma(ndx_eulb5, slow=161, fast=3)
    cum_161 = run_lrs(ndx_eulb5, sig_161, leverage=3.0, expense_ratio=0.0,
                      tbill_rate=0.0, signal_lag=1, commission=eulb_commission)
    m_161 = calc_metrics(cum_161, tbill_rate=0.0)
    tpy_161 = signal_trades_per_year(sig_161)
    print(f"  (3, 161): CAGR={m_161['CAGR']:.3%}  Total={m_161['Total Return']:.1f}x  "
          f"Sharpe={m_161['Sharpe']:.3f}  MDD={m_161['MDD']:.2%}  Trades/Yr={tpy_161:.1f}")

    bh3x_5 = run_buy_and_hold(ndx_eulb5, leverage=3.0, expense_ratio=0.0)
    bh3x_5m = calc_metrics(bh3x_5, tbill_rate=0.0)
    print(f"  B&H 3x:  CAGR={bh3x_5m['CAGR']:.3%}  Total={bh3x_5m['Total Return']:.1f}x  "
          f"MDD={bh3x_5m['MDD']:.2%}")

    # ══════════════════════════════════════════════
    # Part 12: TQQQ-Calibrated NDX Grid Search (1985-2025)
    # ══════════════════════════════════════════════
    CALIBRATED_ER = 0.035

    print("\n" + "=" * 70)
    print("  PART 12: TQQQ-Calibrated NDX Grid Search (1985-2025)")
    print(f"  조건: expense={CALIBRATED_ER:.2%} (calibrated), tbill=Ken French RF, lag=1, comm=0.2%, 3x only")
    print("=" * 70)

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

    # --- Part 12 vs Part 10 비교 ---
    print("\n" + "=" * 70)
    print("  Part 12 vs Part 10: eulb 주요 조합 비교 (calibrated vs eulb 조건)")
    print(f"  Part 12: ER={CALIBRATED_ER:.2%}, tbill=Ken French RF, lag=1, comm=0.2%")
    print(f"  Part 10: ER=0%, tbill=0, lag=1, comm=0.2%")
    print("=" * 70)

    for fast, slow, desc in eulb_combos:
        sig = signal_dual_ma(ndx_price, slow=slow, fast=fast)
        cum_cal = run_lrs(ndx_price, sig, leverage=3.0,
                          expense_ratio=CALIBRATED_ER,
                          tbill_rate=rf_series_grid,
                          signal_lag=1, commission=eulb_commission)
        m_cal = calc_metrics(cum_cal, rf_series=rf_series_grid)

        cum_eulb = run_lrs(ndx_price, sig, leverage=3.0,
                           expense_ratio=0.0, tbill_rate=0.0,
                           signal_lag=1, commission=eulb_commission)
        m_eulb = calc_metrics(cum_eulb, tbill_rate=0.0)

        print(f"  ({fast:2d}, {slow:3d}) [{desc}]")
        print(f"    Calibrated: CAGR={m_cal['CAGR']:.3%}  Total={m_cal['Total Return']:>10.1f}x  Sharpe={m_cal['Sharpe']:.3f}")
        print(f"    eulb cond:  CAGR={m_eulb['CAGR']:.3%}  Total={m_eulb['Total Return']:>10.1f}x  Sharpe={m_eulb['Sharpe']:.3f}")
        print(f"    CAGR diff:  {(m_cal['CAGR'] - m_eulb['CAGR'])*100:+.2f}pp")

    bh3x_cal = run_buy_and_hold(ndx_price, leverage=3.0, expense_ratio=CALIBRATED_ER)
    bh3x_cal_m = calc_metrics(bh3x_cal, rf_series=rf_series_grid)
    print(f"\n  B&H 3x (calibrated): CAGR={bh3x_cal_m['CAGR']:.3%}  "
          f"Total={bh3x_cal_m['Total Return']:.1f}x  MDD={bh3x_cal_m['MDD']:.2%}")
    print(f"  B&H 3x (eulb cond):  CAGR={bh3x_m['CAGR']:.3%}  "
          f"Total={bh3x_m['Total Return']:.1f}x  MDD={bh3x_m['MDD']:.2%}")

    # --- 2010-present validation ---
    print(f"\n  --- 2010-present: Calibrated B&H 3x vs TQQQ (sanity check) ---")
    try:
        qqq_val = download("QQQ", start="2010-02-11", end="2025-12-31")
        tqqq_val = download("TQQQ", start="2010-02-11", end="2025-12-31")
        common_val = qqq_val.index.intersection(tqqq_val.index)
        qqq_val = qqq_val.loc[common_val]
        tqqq_val = tqqq_val.loc[common_val]

        sim_cal = run_buy_and_hold(qqq_val, leverage=3.0, expense_ratio=CALIBRATED_ER)
        actual_tqqq = run_buy_and_hold(tqqq_val, leverage=1.0, expense_ratio=0.0)
        m_sim = calc_metrics(sim_cal)
        m_actual = calc_metrics(actual_tqqq)
        print(f"  Sim 3x (ER={CALIBRATED_ER:.2%}): CAGR={m_sim['CAGR']:.3%}  Total={m_sim['Total Return']:.1f}x")
        print(f"  TQQQ (actual):        CAGR={m_actual['CAGR']:.3%}  Total={m_actual['Total Return']:.1f}x")
        print(f"  CAGR diff:            {(m_sim['CAGR'] - m_actual['CAGR'])*100:+.3f}pp")
    except Exception as e:
        print(f"  [Warning] Validation skipped: {e}")

    print("\n" + "=" * 70)
    print("  Done! Parts 7-12 complete.")
    print("=" * 70)
