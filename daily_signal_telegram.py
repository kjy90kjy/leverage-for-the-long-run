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

LOOKBACK_DAYS = 300


def calculate_sma(prices, window):
    """SMA 계산"""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])


def calculate_prediction(ndx_price, result):
    """
    다음 신호까지 필요한 가격 계산.

    Returns:
        prediction: {
            'current_signal': 0 or 1,
            'regime': 'LOW' or 'HIGH',
            'current_price': float,
            'next_signal_price': float,
            'price_change_needed': float,
            'price_pct_change': float,
            'prediction_text': str,
        }
    """
    p = OPTIMAL_PARAMS
    prices = ndx_price.values.astype(np.float64)

    # 현재 MA 값
    fast_low_ma = calculate_sma(prices, p['fast_low'])
    slow_low_ma = calculate_sma(prices, p['slow_low'])
    fast_high_ma = calculate_sma(prices, p['fast_high'])
    slow_high_ma = calculate_sma(prices, p['slow_high'])

    current_price = prices[-1]
    current_signal = result['signal']
    regime = result['regime']

    prediction = {
        'current_signal': current_signal,
        'regime': regime,
        'current_price': current_price,
        'next_signal_price': None,
        'price_change_needed': None,
        'price_pct_change': None,
        'prediction_text': '',
    }

    # 예측 로직
    if regime == 'LOW':
        if current_signal == 0:
            # 현재: 관망 (fast_low ≤ slow_low)
            # 다음: 매수 (fast_low > slow_low)
            target_price = slow_low_ma
            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100

            if prediction['price_pct_change'] > 0:
                prediction['prediction_text'] = (
                    f"📈 내일 ${target_price:.0f} 이상이면 LOW-VOL 매수 신호 발생!\n"
                    f"   필요 상승: ${prediction['price_change_needed']:.0f} (+{prediction['price_pct_change']:.2f}%)"
                )
            else:
                prediction['prediction_text'] = (
                    f"📉 내일 ${target_price:.0f} 이상 하강하면 매수 신호 (현재 이미 조건 근처)\n"
                    f"   필요 상승: ${prediction['price_change_needed']:.0f} ({prediction['price_pct_change']:.2f}%)"
                )
        else:
            # 현재: 매수 (fast_low > slow_low)
            # 다음: 관망 (fast_low ≤ slow_low)
            target_price = slow_low_ma
            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100

            prediction['prediction_text'] = (
                f"⚠️ LOW-VOL 매도 신호: 내일 ${target_price:.0f} 이하로 내려가면 HOLD로 전환\n"
                f"   필요 하락: ${-prediction['price_change_needed']:.0f} ({prediction['price_pct_change']:.2f}%)"
            )

    else:  # HIGH
        if current_signal == 0:
            target_price = slow_high_ma
            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100

            if prediction['price_pct_change'] > 0:
                prediction['prediction_text'] = (
                    f"⚡ 내일 ${target_price:.0f} 이상이면 HIGH-VOL 빠른 매수 신호!\n"
                    f"   필요 상승: ${prediction['price_change_needed']:.0f} (+{prediction['price_pct_change']:.2f}%)"
                )
            else:
                prediction['prediction_text'] = (
                    f"⚡ 내일 ${target_price:.0f} 이상 상승하면 매수 신호\n"
                    f"   필요 상승: ${prediction['price_change_needed']:.0f} ({prediction['price_pct_change']:.2f}%)"
                )
        else:
            target_price = slow_high_ma
            prediction['next_signal_price'] = target_price
            prediction['price_change_needed'] = target_price - current_price
            prediction['price_pct_change'] = (target_price - current_price) / current_price * 100

            prediction['prediction_text'] = (
                f"⚠️ HIGH-VOL 매도 신호: 내일 ${target_price:.0f} 이하로 내려가면 HOLD로 전환\n"
                f"   필요 하락: ${-prediction['price_change_needed']:.0f} ({prediction['price_pct_change']:.2f}%)"
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
    """텔레그램 메시지 포맷"""

    signal_emoji = "🟢 BUY" if result['signal'] == 1 else "🔴 HOLD"
    regime_emoji = "❄️" if result['regime'] == 'LOW' else "🔥"

    message = f"""
<b>⚡ NDX Daily Signal Report</b>

📅 <b>Date:</b> {result['date']}
💵 <b>Price:</b> ${result['price']:.2f}
🎯 <b>Signal:</b> {signal_emoji}
{regime_emoji} <b>Regime:</b> {result['regime']}

<b>═══════════════════════</b>

<b>📊 Current Status:</b>
Signal Type: {result['signal_type']}

<b>📈 Tomorrow's Prediction:</b>
{prediction['prediction_text']}

<b>═══════════════════════</b>

⚙️ <i>Regime-Switching Strategy</i>
<i>Conservative P1 (12,237,6,229,49,57.3%)</i>
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
    print(prediction['prediction_text'])

    # 텔레그램 전송
    print("\n📱 Sending Telegram message...")
    message = format_telegram_message(result, prediction)
    send_telegram_message(message)

    print("\n" + "=" * 80)
    print("  ✓ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
