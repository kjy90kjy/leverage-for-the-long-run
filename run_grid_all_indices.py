"""Run 3x grid search for all three indices: S&P 500 TR, NASDAQ Composite, NDX."""

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
    GRID_KWARGS = dict(
        expense_ratio=CALIBRATED_ER,
        signal_lag=1,
        fast_range=range(2, 51),
        slow_range=range(50, 351, 1),
        leverage_list=[3],
        commission=0.002,
    )

    print("Downloading Ken French RF...")
    rf = download_ken_french_rf()

    # ═══════════════════════════════════════════
    # 1. S&P 500 Total Return (1928-2025)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  S&P 500 Total Return (1928-2025, 3x, calibrated costs)")
    print("=" * 70)

    sp = download("^GSPC", "1928-01-01", "2025-12-31", total_return=True)
    print(f"  Downloaded {len(sp)} days ({sp.index[0].date()} -> {sp.index[-1].date()})")

    run_dual_ma_analysis(
        price=sp,
        label="S&P 500 TR (1928-2025, 3x calibrated)",
        safe_ticker="GSPC_TR_3x",
        tbill_rate=rf,
        rf_series=rf,
        **GRID_KWARGS,
    )

    # ═══════════════════════════════════════════
    # 2. NASDAQ Composite (1971-2025)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  NASDAQ Composite ^IXIC (1971-2025, 3x, calibrated costs)")
    print("=" * 70)

    ixic = download("^IXIC", "1971-01-01", "2025-12-31", total_return=False)
    print(f"  Downloaded {len(ixic)} days ({ixic.index[0].date()} -> {ixic.index[-1].date()})")

    run_dual_ma_analysis(
        price=ixic,
        label="NASDAQ Composite (1971-2025, 3x calibrated)",
        safe_ticker="IXIC_3x",
        tbill_rate=rf,
        rf_series=rf,
        **GRID_KWARGS,
    )

    # ═══════════════════════════════════════════
    # 3. Nasdaq-100 (1985-2025) — same as Part 12
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Nasdaq-100 ^NDX (1985-2025, 3x, calibrated costs)")
    print("=" * 70)

    ndx = download("^NDX", "1985-10-01", "2025-12-31", total_return=False)
    print(f"  Downloaded {len(ndx)} days ({ndx.index[0].date()} -> {ndx.index[-1].date()})")

    run_dual_ma_analysis(
        price=ndx,
        label="NDX (1985-2025, 3x calibrated)",
        safe_ticker="NDX_calibrated",
        tbill_rate=rf,
        rf_series=rf,
        **GRID_KWARGS,
    )

    print("\n" + "=" * 70)
    print("  Done! All three indices complete.")
    print("=" * 70)
