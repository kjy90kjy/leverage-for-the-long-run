"""Run Parts 7-12 only — grid searches with Sortino + composite heatmaps."""

import sys
import io
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from leverage_rotation import (
    download, download_ken_french_rf, run_dual_ma_analysis,
    run_eulb1_comparison, run_eulb5_spotcheck, run_part12_comparison,
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

    eulb_combos, bh3x_m = run_eulb1_comparison(ndx_price, commission=eulb_commission)

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

    run_eulb5_spotcheck(ndx_eulb5, commission=eulb_commission)

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

    run_part12_comparison(ndx_price, rf_series_grid, eulb_combos, bh3x_m,
                          calibrated_er=CALIBRATED_ER, commission=eulb_commission)

    print("\n" + "=" * 70)
    print("  Done! Parts 7-12 complete.")
    print("=" * 70)
