"""
Update NDX Signal JSON for iOS Scriptable Widget

Generates current NDX regime-switching signal and saves to JSON.
JSON is served by signal_server.py to iOS app via HTTP.

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

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from leverage_rotation import download, signal_regime_switching_dual_ma

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

OPTIMAL_PARAMS = {
    'fast_low': 12,
    'slow_low': 237,
    'fast_high': 6,
    'slow_high': 229,
    'vol_lookback': 49,
    'vol_threshold_pct': 57.3,
}

LOOKBACK_DAYS = 500  # Sufficient for 237-day MA
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "ndx_signal.json"


def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])


def calculate_virtual_ma(prices, window, test_price):
    """
    Calculate virtual MA with test_price as today's close.
    Last N-1 days unchanged, only last price changes.
    """
    if len(prices) < window:
        return np.nan
    virtual_prices = np.concatenate([prices[:-1], [test_price]])
    return np.mean(virtual_prices[-window:])


def find_crossover_price(prices, fast_window, slow_window, current_signal, direction='up'):
    """
    Find exact crossover price using binary search.

    Args:
        prices: Price array
        fast_window: Fast MA window
        slow_window: Slow MA window
        current_signal: Current signal (0=HOLD, 1=BUY)
        direction: 'up' or 'down'

    Returns:
        (crossover_price, fast_ma, slow_ma)
    """
    current_price = prices[-1]

    if direction == 'up':
        # Rising: fast > slow crossover
        low_price = current_price
        high_price = current_price * 1.10
    else:
        # Falling: fast <= slow crossover
        low_price = current_price * 0.90
        high_price = current_price

    tolerance = 0.01
    max_iterations = 100

    for _ in range(max_iterations):
        mid_price = (low_price + high_price) / 2

        fast_ma = calculate_virtual_ma(prices, fast_window, mid_price)
        slow_ma = calculate_virtual_ma(prices, slow_window, mid_price)

        if np.isnan(fast_ma) or np.isnan(slow_ma):
            return mid_price, fast_ma, slow_ma

        diff = fast_ma - slow_ma

        if direction == 'up':
            if diff > 0:
                high_price = mid_price
            else:
                low_price = mid_price
        else:
            if diff <= 0:
                low_price = mid_price
            else:
                high_price = mid_price

        if high_price - low_price < tolerance:
            break

    crossover_price = (low_price + high_price) / 2
    fast_ma = calculate_virtual_ma(prices, fast_window, crossover_price)
    slow_ma = calculate_virtual_ma(prices, slow_window, crossover_price)

    return crossover_price, fast_ma, slow_ma


def calculate_prediction(ndx_price, result):
    """
    Calculate next signal crossover price and prediction details.

    Returns dict with all prediction information for iOS widget.
    """
    p = OPTIMAL_PARAMS
    prices = ndx_price.values.astype(np.float64)
    current_price = prices[-1]

    # Current MA values
    fast_low_ma = calculate_sma(prices, p['fast_low'])
    slow_low_ma = calculate_sma(prices, p['slow_low'])
    fast_high_ma = calculate_sma(prices, p['fast_high'])
    slow_high_ma = calculate_sma(prices, p['slow_high'])

    current_signal = result['signal']
    regime = result['regime']

    prediction = {
        'current_fast_ma': None,
        'current_slow_ma': None,
        'next_signal_price': None,
        'price_change_needed': None,
        'price_pct_change': None,
        'crossover_direction': '',
    }

    if regime == 'LOW':
        prediction['current_fast_ma'] = float(fast_low_ma)
        prediction['current_slow_ma'] = float(slow_low_ma)
        fast_param = p['fast_low']
        slow_param = p['slow_low']

        if current_signal == 0:
            # HOLD → BUY
            prediction['crossover_direction'] = f"fast_low ({fast_param}d) > slow_low ({slow_param}d)"
            target_price, virtual_fast, virtual_slow = find_crossover_price(
                prices, fast_param, slow_param, current_signal, direction='up'
            )
        else:
            # BUY → HOLD
            prediction['crossover_direction'] = f"fast_low ({fast_param}d) ≤ slow_low ({slow_param}d)"
            target_price, virtual_fast, virtual_slow = find_crossover_price(
                prices, fast_param, slow_param, current_signal, direction='down'
            )

    else:  # HIGH
        prediction['current_fast_ma'] = float(fast_high_ma)
        prediction['current_slow_ma'] = float(slow_high_ma)
        fast_param = p['fast_high']
        slow_param = p['slow_high']

        if current_signal == 0:
            # HOLD → BUY
            prediction['crossover_direction'] = f"fast_high ({fast_param}d) > slow_high ({slow_param}d)"
            target_price, virtual_fast, virtual_slow = find_crossover_price(
                prices, fast_param, slow_param, current_signal, direction='up'
            )
        else:
            # BUY → HOLD
            prediction['crossover_direction'] = f"fast_high ({fast_param}d) ≤ slow_high ({slow_param}d)"
            target_price, virtual_fast, virtual_slow = find_crossover_price(
                prices, fast_param, slow_param, current_signal, direction='down'
            )

    prediction['next_signal_price'] = float(target_price)
    prediction['price_change_needed'] = float(target_price - current_price)
    prediction['price_pct_change'] = float((target_price - current_price) / current_price * 100)

    return prediction


def generate_signal():
    """Generate NDX regime-switching signal"""
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': None,
        'price': None,
        'signal': None,
        'signal_type': None,
        'regime': None,
        'volatility_pct': None,
        'status': 'ERROR',
        'message': '',
    }

    try:
        # Download NDX data
        ndx = download("^NDX",
                      start=(datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
                      end=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))

        if len(ndx) == 0:
            result['message'] = 'No data downloaded'
            return result, None

        result['date'] = ndx.index[-1].strftime('%Y-%m-%d')
        result['price'] = float(ndx.iloc[-1])

        # Generate signal
        p = OPTIMAL_PARAMS
        sig = signal_regime_switching_dual_ma(
            ndx,
            fast_low=p['fast_low'],
            slow_low=p['slow_low'],
            fast_high=p['fast_high'],
            slow_high=p['slow_high'],
            vol_lookback=p['vol_lookback'],
            vol_threshold_pct=p['vol_threshold_pct']
        )

        result['signal'] = int(sig.iloc[-1])

        # Calculate volatility percentile
        prices = ndx.values.astype(np.float64)
        ret = np.diff(np.log(prices))
        rolling_vols = []

        for i in range(p['vol_lookback'], len(prices)):
            rv = np.std(ret[i-p['vol_lookback']:i]) * np.sqrt(252)
            rolling_vols.append(rv)

        if rolling_vols:
            vol_pct = (rolling_vols[-1] <= np.percentile(rolling_vols, 50)) * 100
            result['regime'] = 'LOW' if vol_pct < p['vol_threshold_pct'] else 'HIGH'
            result['volatility_pct'] = float(rolling_vols[-1] * 100)

        # Signal type
        if result['signal'] == 1:
            if result['regime'] == 'LOW':
                result['signal_type'] = '저변동 진입'
            else:
                result['signal_type'] = '고변동 진입'
        else:
            result['signal_type'] = '관망'

        result['status'] = 'SUCCESS'
        return result, ndx

    except Exception as e:
        result['message'] = f"Error: {str(e)}"
        import traceback
        print(f"❌ Signal generation failed: {traceback.format_exc()}")
        return result, None


def save_signal_json(result, prediction):
    """Save signal and prediction to JSON file"""

    # Build JSON structure for iOS widget
    data = {
        'signal': result['signal'],
        'signal_type': result['signal_type'],
        'price': result['price'],
        'date': result['date'],
        'regime': result['regime'],
        'current_fast_ma': prediction['current_fast_ma'],
        'current_slow_ma': prediction['current_slow_ma'],
        'next_signal_price': prediction['next_signal_price'],
        'price_change_needed': prediction['price_change_needed'],
        'price_pct_change': prediction['price_pct_change'],
        'crossover_direction': prediction['crossover_direction'],
        'timestamp': result['timestamp'],
        'volatility_pct': result['volatility_pct'],
    }

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return OUTPUT_FILE


def main():
    print("=" * 80)
    print("  NDX Signal JSON Generator for iOS Scriptable Widget")
    print("=" * 80)

    # Generate signal
    result, ndx = generate_signal()

    if result['status'] != 'SUCCESS':
        print(f"❌ Failed: {result['message']}")
        return False

    print(f"✓ Signal generated: {result['date']} @ ${result['price']:.2f}")
    print(f"  Signal: {result['signal']} ({result['signal_type']})")
    print(f"  Regime: {result['regime']}")

    # Calculate prediction
    prediction = calculate_prediction(ndx, result)

    print(f"\n📈 Next Signal Trigger:")
    print(f"  Target Price: ${prediction['next_signal_price']:.0f}")
    print(f"  Change Needed: ${prediction['price_change_needed']:+.0f} ({prediction['price_pct_change']:+.2f}%)")
    print(f"  Condition: {prediction['crossover_direction']}")

    # Save JSON
    output_file = save_signal_json(result, prediction)

    print(f"\n✓ JSON saved: {output_file}")
    print("\n" + "=" * 80)
    print("  ✓ Done!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
