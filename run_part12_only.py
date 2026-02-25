"""Run Part 12 only — for quick iteration on heatmap changes."""

import sys
import io
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from leverage_rotation import (
    download, download_ken_french_rf, run_dual_ma_analysis,
)

if __name__ == "__main__":
    CALIBRATED_ER = 0.035

    # Download data
    print("Downloading NDX...")
    ndx_price = download("^NDX", start="1985-10-01", end="2025-12-31")
    print(f"  NDX: {len(ndx_price)} days ({ndx_price.index[0].date()} -> {ndx_price.index[-1].date()})")

    print("Downloading Ken French RF...")
    rf_series_grid = download_ken_french_rf()

    print(f"\n{'=' * 70}")
    print(f"  PART 12: TQQQ-Calibrated NDX Grid Search (1985-2025)")
    print(f"  expense={CALIBRATED_ER:.2%}, tbill=Ken French RF, lag=1, comm=0.2%, 3x only")
    print(f"{'=' * 70}")

    run_dual_ma_analysis(
        price=ndx_price,
        label="NDX Calibrated (1985-2025, lag=1, TQQQ-calibrated costs)",
        safe_ticker="NDX_calibrated",
        expense_ratio=CALIBRATED_ER,
        tbill_rate=rf_series_grid,
        signal_lag=1,
        rf_series=rf_series_grid,
        fast_range=range(2, 51),
        slow_range=range(50, 351, 1),
        leverage_list=[3],
        commission=0.002,
    )

    print("\nDone!")
