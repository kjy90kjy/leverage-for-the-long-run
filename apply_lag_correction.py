"""
Helper functions to apply lag correction factors to Part 7-9 results.

This module provides utilities to correct for look-ahead bias in Part 7-9
grid searches (which used lag=0) vs Part 12 (which uses lag=1).

Usage:
    from apply_lag_correction import correct_part7_cagr, correct_part8_cagr, warn_fast2

    # Correct a single combo
    realistic = correct_part7_cagr(reported_cagr=34.35, fast=3, slow=118)
    # Output: 22.54 (realistic CAGR after correction)

    # Load corrections from CSV
    from pathlib import Path
    import pandas as pd
    corrections = pd.read_csv(Path("output") / "Part7_lag_correction_table.csv")
    for _, row in corrections.iterrows():
        print(f"MA({row['Fast']},{row['Slow']}): "
              f"{row['lag0_CAGR']:.1%} → {row['lag0_CAGR']/row['CAGR_Correction_Factor']:.1%}")
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

OUT_DIR = Path(__file__).parent / "output"

# Precomputed correction factors from fix_part79_lag_mismatch.py
PART7_CORRECTIONS = {
    (3, 118): 1.5238605203187638,
    (3, 117): 1.5573923868160395,
    (3, 116): 1.5924977697845943,
    (3, 115): 1.6151025784911952,
    (8, 211): 1.097521236836848,
}

PART8_CORRECTIONS = {
    (2, 51): 2.344404615874078,  # ⚠️ EXTREME
    (2, 52): 2.3712683748273453,  # ⚠️ EXTREME
    (2, 53): 2.3562717147136882,  # ⚠️ EXTREME
    (2, 50): 2.4340095586529404,  # ⚠️ EXTREME
    (7, 57): 1.2035809603322938,
}


def load_correction_table(part: int) -> pd.DataFrame:
    """Load correction table from CSV.

    Args:
        part: 7 or 8

    Returns:
        DataFrame with columns: Fast, Slow, CAGR_Correction_Factor, ...
    """
    if part == 7:
        csv_path = OUT_DIR / "Part7_lag_correction_table.csv"
    elif part == 8:
        csv_path = OUT_DIR / "Part8_lag_correction_table.csv"
    else:
        raise ValueError(f"Invalid part: {part} (must be 7 or 8)")

    if not csv_path.exists():
        raise FileNotFoundError(f"Correction table not found: {csv_path}")

    return pd.read_csv(csv_path)


def correct_part7_cagr(reported_cagr: float, fast: int, slow: int) -> float:
    """Correct Part 7 (GSPC) CAGR for look-ahead bias.

    Args:
        reported_cagr: CAGR from Part 7 (lag=0)
        fast: Fast MA period
        slow: Slow MA period

    Returns:
        Corrected CAGR (realistic lag=1 execution)
    """
    key = (fast, slow)

    # Use precomputed factors if available
    if key in PART7_CORRECTIONS:
        correction = PART7_CORRECTIONS[key]
        return reported_cagr / correction

    # Otherwise load from CSV
    df = load_correction_table(7)
    row = df[(df["Fast"] == fast) & (df["Slow"] == slow)]

    if len(row) == 0:
        raise ValueError(f"Combo MA({fast},{slow}) not found in Part 7 correction table")

    correction = row.iloc[0]["CAGR_Correction_Factor"]
    return reported_cagr / correction


def correct_part8_cagr(reported_cagr: float, fast: int, slow: int) -> float:
    """Correct Part 8 (IXIC) CAGR for look-ahead bias.

    ⚠️ WARNING: Check if fast=2. If so, consider using slower alternatives instead.

    Args:
        reported_cagr: CAGR from Part 8 (lag=0)
        fast: Fast MA period
        slow: Slow MA period

    Returns:
        Corrected CAGR (realistic lag=1 execution)
    """
    key = (fast, slow)

    # Warn if extreme correction factor
    if fast == 2:
        warnings.warn(
            f"⚠️  EXTREME CORRECTION NEEDED: MA({fast},{slow}) fast=2 combos have "
            f"correction factors > 2.3x (loss >50% CAGR)\n"
            f"Recommendation: Use slower alternatives (fast ≥ 7) or Part 12 (NDX) instead",
            UserWarning,
            stacklevel=2
        )

    # Use precomputed factors if available
    if key in PART8_CORRECTIONS:
        correction = PART8_CORRECTIONS[key]
        return reported_cagr / correction

    # Otherwise load from CSV
    df = load_correction_table(8)
    row = df[(df["Fast"] == fast) & (df["Slow"] == slow)]

    if len(row) == 0:
        raise ValueError(f"Combo MA({fast},{slow}) not found in Part 8 correction table")

    correction = row.iloc[0]["CAGR_Correction_Factor"]
    return reported_cagr / correction


def warn_fast2_combos():
    """Print warning about Part 8 fast=2 combos."""
    print("="*80)
    print("CRITICAL WARNING: Part 8 (IXIC) fast=2 Combos")
    print("="*80)
    print("""
DO NOT USE these combos for trading:
  MA(2,51):  76.32% -> 32.56% (correction: 2.34x)
  MA(2,52):  75.50% -> 31.84% (correction: 2.37x)
  MA(2,53):  74.94% -> 31.80% (correction: 2.36x)
  MA(2,50):  76.71% -> 31.51% (correction: 2.43x)

Why?
  - Correction factors > 2.3x = EXTREME look-ahead bias
  - Reported 76% becomes realistic 32% (44 percentage points lost!)
  - Fast=2 = only 2 days of data -> reacts to intraday noise, not signal
  - Unreliable for live trading

Recommended alternatives:
  * Use Part 8 combos with fast >= 7 (correction ~1.2x, much safer)
  * Or use Part 12 (NDX 3x, lag=1, Ken French RF) -- validated and production-ready

Reference: See output/Part8_lag_correction_table.csv for full correction table
""")


def apply_corrections_to_csv(part: int, input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """Apply lag corrections to entire CSV file.

    Args:
        part: 7 or 8
        input_csv: Path to original grid results CSV
        output_csv: Path to save corrected results (default: input_csv with "_corrected" suffix)

    Returns:
        DataFrame with corrected CAGR, Sharpe, Sortino columns
    """
    df = pd.read_csv(input_csv)

    # Determine which correction function to use
    correct_func = correct_part7_cagr if part == 7 else correct_part8_cagr

    # Apply corrections
    corrected_rows = []
    for _, row in df.iterrows():
        fast, slow = int(row["fast"]), int(row["slow"])

        corrected_row = row.copy()
        corrected_row["CAGR_original"] = row["CAGR"]
        corrected_row["CAGR"] = correct_func(row["CAGR"], fast, slow)

        # Also correct Sharpe and Sortino (scale by correction factor)
        correction = row["CAGR"] / corrected_row["CAGR"]
        corrected_row["Sharpe_original"] = row.get("Sharpe", np.nan)
        corrected_row["Sharpe"] = row.get("Sharpe", np.nan) / correction if pd.notna(row.get("Sharpe")) else np.nan
        corrected_row["Sortino_original"] = row.get("Sortino", np.nan)
        corrected_row["Sortino"] = row.get("Sortino", np.nan) / correction if pd.notna(row.get("Sortino")) else np.nan

        corrected_rows.append(corrected_row)

    df_corrected = pd.DataFrame(corrected_rows)

    # Save if output path specified
    if output_csv is None:
        output_csv = str(input_csv).replace(".csv", "_corrected.csv")

    df_corrected.to_csv(output_csv, index=False)
    print(f"✓ Saved corrected CSV: {output_csv}")

    return df_corrected


if __name__ == "__main__":
    # Example usage
    print("Lag Correction Helper Functions\n")

    # Part 7 example
    print("Part 7 (GSPC) Example:")
    reported = 0.3435
    realistic = correct_part7_cagr(reported, fast=3, slow=118)
    print(f"  MA(3,118): {reported*100:.2f}% -> {realistic*100:.2f}% (correction: {reported/realistic:.2f}x)\n")

    # Part 8 example with warning
    print("Part 8 (IXIC) Example:")
    reported = 0.7632
    realistic = correct_part8_cagr(reported, fast=2, slow=51)  # This will warn
    print(f"  MA(2,51): {reported*100:.2f}% -> {realistic*100:.2f}% (correction: {reported/realistic:.2f}x)\n")

    # Print full warning
    print()
    warn_fast2_combos()
