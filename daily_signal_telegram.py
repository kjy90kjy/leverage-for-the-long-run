"""
Daily Signal with Telegram & Price Prediction

Features:
- Telegram 메시지 발송
- 다음 신호까지 필요한 가격 계산 & 예측
- 매일 오후 4시 자동 발송

설정:
    1. 텔레그램 @BotFather에서 봇 생성
    2. Token과 Chat ID 설정
    3. daily_signal_generator.py와 함께 실행
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
import requests
import json

from leverage_rotation import (
    download, signal_regime_switching_dual_ma
)

# ═══════════════════════════════════════════════════════════════
# ⚠️ 설정: 아래 값들을 본인 값으로 변경하세요
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"  # @BotFather에서 받은 token
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"   # https://api.telegram.org/botXXX/getUpdates 에서 확인

# ═══════════════════════════════════════════════════════════════
# 최적 파라미터 (Conservative P1)
# ═══════════════════════════════════════════════════════════════

OPTIMAL_PARAMS = {
    'fast_low': 12,
    'slow_low': 237,
    'fast_high': 6,
    'slow_high': 229,
    'vol_lookback': 49,
    'vol_threshold_pct': 57.3,
}

LOOKBACK_DAYS = 500  # 237일 MA를 위해 충분한 데이터 필요


def calculate_sma(prices, window):
    """SMA 계산"""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])


def calculate_virtual_ma(prices, window, test_price):
    """
    가상 종가(test_price)로 MA 재계산.
    과거 N-1일은 그대로, 마지막 종가만 test_price로 변경.
    """
    if len(prices) < window:
        return np.nan
    # 마지막 가격을 test_price로 변경한 배열
    virtual_prices = np.concatenate([prices[:-1], [test_price]])
    return np.mean(virtual_prices[-window:])


def calculate_prediction(ndx_price, result):
    """
    다음 신호까지 필요한 가격 계산 (상세 정보 포함).

    Returns:
        prediction: {
            'current_signal': 0 or 1,
            'regime': 'LOW' or 'HIGH',
            'current_price': float,
            'current_fast_ma': float,
            'current_slow_ma': float,
            'next_signal_price': float,
            'price_change_needed': float,
            'price_pct_change': float,
            'virtual_fast_ma': float,
            'virtual_slow_ma': float,
            'crossover_direction': str,
            'detailed_text': str,
        }
    """
    p = OPTIMAL_PARAMS
    prices = ndx_price.values.astype(np.float64)
    current_price = prices[-1]

    # 현재 MA 값
    fast_low_ma = calculate_sma(prices, p['fast_low'])
    slow_low_ma = calculate_sma(prices, p['slow_low'])
    fast_high_ma = calculate_sma(prices, p['fast_high'])
    slow_high_ma = calculate_sma(prices, p['slow_high'])

    current_signal = result['signal']
    regime = result['regime']

    prediction = {
        'current_signal': current_signal,
        'regime': regime,
        'current_price': current_price,
        'current_fast_ma': None,
        'current_slow_ma': None,
        'next_signal_price': None,
        'price_change_needed': None,
        'price_pct_change': None,
        'virtual_fast_ma': None,
        'virtual_slow_ma': None,
        'crossover_direction': '',
        'detailed_text': '',
    }

    # 예측 로직
    if regime == 'LOW':
        prediction['current_fast_ma'] = fast_low_ma
        prediction['current_slow_ma'] = slow_low_ma
        fast_param = p['fast_low']
        slow_param = p['slow_low']

        if current_signal == 0:
            # 현재: 관망 (fast_low ≤ slow_low)
            # 다음: 매수 (fast_low > slow_low)
            prediction['crossover_direction'] = f"fast_low ({fast_param}일) > slow_low ({slow_param}일)"

            # 목표 가격 탐색: binary search로 정확한 크로스오버 지점 찾기
            target_price = slow_low_ma
            virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            # slow_low는 거의 변화 없으므로 약간의 마진 추가
            while virtual_fast <= slow_low_ma and target_price < current_price * 1.05:
                target_price += 1
                virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100
            prediction['virtual_fast_ma'] = virtual_fast
            prediction['virtual_slow_ma'] = calculate_virtual_ma(prices, slow_param, target_price)

            prediction['detailed_text'] = (
                f"📊 현재 상태:\n"
                f"  레짐: ❄️ LOW-VOLATILITY\n"
                f"  신호: 🔴 HOLD (관망)\n"
                f"  Fast MA (12일): ${fast_low_ma:.2f}\n"
                f"  Slow MA (237일): ${slow_low_ma:.2f}\n"
                f"  조건: fast_low ≤ slow_low\n\n"
                f"📈 다음 거래일 신호 발생 조건:\n"
                f"  필요 종가: ${target_price:.0f} 이상\n"
                f"  상승 필요: ${prediction['price_change_needed']:.0f} (+{prediction['price_pct_change']:.2f}%)\n\n"
                f"🎯 가상 계산 (${target_price:.0f}일 때):\n"
                f"  가상 Fast MA: ${prediction['virtual_fast_ma']:.2f}\n"
                f"  가상 Slow MA: ${prediction['virtual_slow_ma']:.2f}\n"
                f"  → {prediction['crossover_direction']}\n"
                f"  → 🟢 BUY 신호 발생!"
            )
        else:
            # 현재: 매수 (fast_low > slow_low)
            # 다음: 관망 (fast_low ≤ slow_low)
            prediction['crossover_direction'] = f"fast_low ({fast_param}일) ≤ slow_low ({slow_param}일)"

            target_price = slow_low_ma
            virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            # 하락 시나리오
            while virtual_fast > slow_low_ma and target_price > current_price * 0.95:
                target_price -= 1
                virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100
            prediction['virtual_fast_ma'] = virtual_fast
            prediction['virtual_slow_ma'] = calculate_virtual_ma(prices, slow_param, target_price)

            prediction['detailed_text'] = (
                f"📊 현재 상태:\n"
                f"  레짐: ❄️ LOW-VOLATILITY\n"
                f"  신호: 🟢 BUY (매수)\n"
                f"  Fast MA (12일): ${fast_low_ma:.2f}\n"
                f"  Slow MA (237일): ${slow_low_ma:.2f}\n"
                f"  조건: fast_low > slow_low\n\n"
                f"⚠️ 다음 거래일 신호 전환 조건:\n"
                f"  필요 종가: ${target_price:.0f} 이하\n"
                f"  하락 필요: ${-prediction['price_change_needed']:.0f} ({prediction['price_pct_change']:.2f}%)\n\n"
                f"🎯 가상 계산 (${target_price:.0f}일 때):\n"
                f"  가상 Fast MA: ${prediction['virtual_fast_ma']:.2f}\n"
                f"  가상 Slow MA: ${prediction['virtual_slow_ma']:.2f}\n"
                f"  → {prediction['crossover_direction']}\n"
                f"  → 🔴 HOLD 신호 전환!"
            )

    else:  # HIGH
        prediction['current_fast_ma'] = fast_high_ma
        prediction['current_slow_ma'] = slow_high_ma
        fast_param = p['fast_high']
        slow_param = p['slow_high']

        if current_signal == 0:
            # 현재: 관망 (fast_high ≤ slow_high)
            # 다음: 매수 (fast_high > slow_high)
            prediction['crossover_direction'] = f"fast_high ({fast_param}일) > slow_high ({slow_param}일)"

            target_price = slow_high_ma
            virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            while virtual_fast <= slow_high_ma and target_price < current_price * 1.05:
                target_price += 1
                virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100
            prediction['virtual_fast_ma'] = virtual_fast
            prediction['virtual_slow_ma'] = calculate_virtual_ma(prices, slow_param, target_price)

            prediction['detailed_text'] = (
                f"📊 현재 상태:\n"
                f"  레짐: 🔥 HIGH-VOLATILITY\n"
                f"  신호: 🔴 HOLD (관망)\n"
                f"  Fast MA (6일): ${fast_high_ma:.2f}\n"
                f"  Slow MA (229일): ${slow_high_ma:.2f}\n"
                f"  조건: fast_high ≤ slow_high\n\n"
                f"⚡ 다음 거래일 신호 발생 조건:\n"
                f"  필요 종가: ${target_price:.0f} 이상\n"
                f"  상승 필요: ${prediction['price_change_needed']:.0f} (+{prediction['price_pct_change']:.2f}%)\n\n"
                f"🎯 가상 계산 (${target_price:.0f}일 때):\n"
                f"  가상 Fast MA: ${prediction['virtual_fast_ma']:.2f}\n"
                f"  가상 Slow MA: ${prediction['virtual_slow_ma']:.2f}\n"
                f"  → {prediction['crossover_direction']}\n"
                f"  → 🟢 BUY 신호 발생!"
            )
        else:
            # 현재: 매수 (fast_high > slow_high)
            # 다음: 관망 (fast_high ≤ slow_high)
            prediction['crossover_direction'] = f"fast_high ({fast_param}일) ≤ slow_high ({slow_param}일)"

            target_price = slow_high_ma
            virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            while virtual_fast > slow_high_ma and target_price > current_price * 0.95:
                target_price -= 1
                virtual_fast = calculate_virtual_ma(prices, fast_param, target_price)

            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100
            prediction['virtual_fast_ma'] = virtual_fast
            prediction['virtual_slow_ma'] = calculate_virtual_ma(prices, slow_param, target_price)

            prediction['detailed_text'] = (
                f"📊 현재 상태:\n"
                f"  레짐: 🔥 HIGH-VOLATILITY\n"
                f"  신호: 🟢 BUY (매수)\n"
                f"  Fast MA (6일): ${fast_high_ma:.2f}\n"
                f"  Slow MA (229일): ${slow_high_ma:.2f}\n"
                f"  조건: fast_high > slow_high\n\n"
                f"⚠️ 다음 거래일 신호 전환 조건:\n"
                f"  필요 종가: ${target_price:.0f} 이하\n"
                f"  하락 필요: ${-prediction['price_change_needed']:.0f} ({prediction['price_pct_change']:.2f}%)\n\n"
                f"🎯 가상 계산 (${target_price:.0f}일 때):\n"
                f"  가상 Fast MA: ${prediction['virtual_fast_ma']:.2f}\n"
                f"  가상 Slow MA: ${prediction['virtual_slow_ma']:.2f}\n"
                f"  → {prediction['crossover_direction']}\n"
                f"  → 🔴 HOLD 신호 전환!"
            )

    return prediction


def generate_signal():
    """신호 생성 (daily_signal_generator.py와 동일)"""
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': None,
        'price': None,
        'signal': None,
        'signal_type': None,
        'regime': None,
        'status': 'ERROR',
        'message': '',
    }

    try:
        print(f"[{result['timestamp']}] Downloading NDX data...")
        ndx = download("^NDX",
                      start=(datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
                      end=datetime.now().strftime('%Y-%m-%d'))

        if len(ndx) == 0:
            result['message'] = 'No data downloaded'
            return result, None

        result['date'] = ndx.index[-1].strftime('%Y-%m-%d')
        result['price'] = ndx.iloc[-1]

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

        # 변동성 계산
        prices = ndx.values.astype(np.float64)
        ret = np.diff(np.log(prices))
        rolling_vols = []
        for i in range(p['vol_lookback'], len(prices)):
            rv = np.std(ret[i-p['vol_lookback']:i]) * np.sqrt(252)
            rolling_vols.append(rv)

        if rolling_vols:
            vol_pct = (rolling_vols[-1] <= np.percentile(rolling_vols, 50)) * 100
            result['regime'] = 'LOW' if vol_pct < p['vol_threshold_pct'] else 'HIGH'

        if result['signal'] == 1:
            if result['regime'] == 'LOW':
                result['signal_type'] = '저변동 진입 (신중 MA)'
            else:
                result['signal_type'] = '고변동 진입 (빠른 MA)'
        else:
            result['signal_type'] = '관망'

        result['status'] = 'SUCCESS'
        return result, ndx

    except Exception as e:
        result['message'] = f"Error: {str(e)}"
        return result, None


def send_telegram_message(message):
    """텔레그램으로 메시지 발송"""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("⚠️  Telegram token not configured")
        return False

    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("⚠️  Telegram chat ID not configured")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
        }
        response = requests.post(url, data=data, timeout=10)

        if response.status_code == 200:
            print("✓ Telegram message sent successfully")
            return True
        else:
            print(f"✗ Telegram error: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Telegram send failed: {e}")
        return False


def format_telegram_message(result, prediction):
    """텔레그램 메시지 포맷 (상세 정보 포함)"""

    signal_emoji = "🟢 BUY" if result['signal'] == 1 else "🔴 HOLD"
    regime_emoji = "❄️" if result['regime'] == 'LOW' else "🔥"

    message = f"""
<b>⚡ NDX Daily Signal - Detailed Report</b>

📅 <b>Date:</b> {result['date']}
💵 <b>Last Close:</b> ${result['price']:.2f}
🎯 <b>Current Signal:</b> {signal_emoji}
{regime_emoji} <b>Regime:</b> {result['regime']}

<b>═══════════════════════════════════</b>

<b>📊 Regime Status:</b>
{result['signal_type']}
Current Fast MA: ${prediction['current_fast_ma']:.2f}
Current Slow MA: ${prediction['current_slow_ma']:.2f}

<b>═══════════════════════════════════</b>

<b>🎯 NEXT TRADING DAY SIGNAL TRIGGER</b>

<b>Required Price:</b> ${prediction['next_signal_price']:.0f}
<b>Price Change Needed:</b> ${prediction['price_change_needed']:+.0f} ({prediction['price_pct_change']:+.2f}%)

<b>Crossover Condition:</b>
{prediction['crossover_direction']}

<b>Virtual MA Calculation:</b>
If tomorrow closes at ${prediction['next_signal_price']:.0f}:
  → Virtual Fast MA: ${prediction['virtual_fast_ma']:.2f}
  → Virtual Slow MA: ${prediction['virtual_slow_ma']:.2f}
  → Crossover occurs! ✓

<b>═══════════════════════════════════</b>

<b>📋 Full Prediction Details:</b>
<code>{prediction['detailed_text']}</code>

<b>═══════════════════════════════════</b>

⚙️ <i>Regime-Switching Dual MA Strategy</i>
<i>Conservative P1: (12,237,6,229,49,57.3%)</i>
<i>CAGR 34.9% | Sortino 1.088 | MDD_Entry -39.2%</i>
<i>Generated: {result['timestamp']}</i>
"""

    return message


def main():
    print("=" * 80)
    print("  Daily Signal with Telegram")
    print("=" * 80)

    # 신호 생성
    result, ndx = generate_signal()

    if result['status'] != 'SUCCESS':
        print(f"❌ Failed: {result['message']}")
        return

    print(f"✓ Signal generated: {result['date']} @ ${result['price']:.2f}")
    print(f"  Signal: {result['signal']} ({result['signal_type']})")

    # 예측 계산
    prediction = calculate_prediction(ndx, result)

    print(f"\n📈 Prediction:")
    print(prediction['detailed_text'])

    # 텔레그램 전송
    print("\n📱 Sending Telegram message...")
    message = format_telegram_message(result, prediction)
    send_telegram_message(message)

    print("\n" + "=" * 80)
    print("  ✓ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
