"""
Daily Signal Generator: 매일 종가 수신 → 신호 생성 → 결과 저장

실행:
    python daily_signal_generator.py

윈도우 작업 스케줄러 설정:
    - 트리거: 매일 오후 4시 (NYSE 종장)
    - 프로그램: python.exe
    - 인수: C:\path\to\daily_signal_generator.py
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

from leverage_rotation import (
    download, signal_regime_switching_dual_ma
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

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

LOOKBACK_DAYS = 300  # 과거 300일 데이터 (MA 237일 고려)


def generate_signal():
    """
    최신 데이터를 다운로드하고 신호를 생성.

    Returns:
        dict: {
            'timestamp': 생성 시간,
            'date': 종가 날짜,
            'price': NDX 종가,
            'signal': 0 (관망) 또는 1 (매수),
            'signal_type': '저변동 진입' / '고변동 진입' / '관망',
            'fast_low_ma': 저변동 빠른 MA,
            'slow_low_ma': 저변동 느린 MA,
            'fast_high_ma': 고변동 빠른 MA,
            'slow_high_ma': 고변동 느린 MA,
            'volatility_pct': 현재 변동성 백분위,
            'regime': 'LOW' 또는 'HIGH',
            'status': 'SUCCESS' 또는 'ERROR',
            'message': 상세 메시지,
        }
    """
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': None,
        'price': None,
        'signal': None,
        'signal_type': None,
        'fast_low_ma': None,
        'slow_low_ma': None,
        'fast_high_ma': None,
        'slow_high_ma': None,
        'volatility_pct': None,
        'regime': None,
        'status': 'ERROR',
        'message': '',
    }

    try:
        # 데이터 다운로드
        print(f"[{result['timestamp']}] Downloading NDX data (last {LOOKBACK_DAYS} days)...")
        ndx = download("^NDX",
                      start=(datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
                      end=datetime.now().strftime('%Y-%m-%d'))

        if len(ndx) == 0:
            result['message'] = 'No data downloaded'
            return result

        # 최신 가격
        result['date'] = ndx.index[-1].strftime('%Y-%m-%d')
        result['price'] = ndx.iloc[-1]
        print(f"  Latest: {result['date']} @ ${result['price']:.2f}")

        # 신호 생성
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

        # 최신 신호
        result['signal'] = int(sig.iloc[-1])
        print(f"  Signal: {result['signal']} (0=관망, 1=매수)")

        # MA 값 추출 (직접 계산)
        prices = ndx.values.astype(np.float64)

        # SMA 계산 (간단한 방식)
        def calc_sma(prices, window):
            if len(prices) < window:
                return np.nan
            return np.mean(prices[-window:])

        result['fast_low_ma'] = calc_sma(prices, p['fast_low'])
        result['slow_low_ma'] = calc_sma(prices, p['slow_low'])
        result['fast_high_ma'] = calc_sma(prices, p['fast_high'])
        result['slow_high_ma'] = calc_sma(prices, p['slow_high'])

        # 변동성 계산
        ret = np.diff(np.log(prices))
        vol_daily = np.std(ret[-p['vol_lookback']:]) if len(ret) >= p['vol_lookback'] else np.nan
        vol_annual = vol_daily * np.sqrt(252)

        # 변동성 백분위 (전체 데이터 기준)
        rolling_vols = []
        for i in range(p['vol_lookback'], len(prices)):
            rv = np.std(ret[i-p['vol_lookback']:i]) * np.sqrt(252)
            rolling_vols.append(rv)

        if rolling_vols:
            vol_pct = (rolling_vols[-1] <= np.percentile(rolling_vols, 50)) * 100
            result['volatility_pct'] = vol_pct
            result['regime'] = 'LOW' if vol_pct < p['vol_threshold_pct'] else 'HIGH'

        # 신호 타입
        if result['signal'] == 1:
            if result['regime'] == 'LOW':
                result['signal_type'] = '저변동 진입 (신중 MA)'
            else:
                result['signal_type'] = '고변동 진입 (빠른 MA)'
        else:
            result['signal_type'] = '관망'

        result['status'] = 'SUCCESS'
        result['message'] = f"Signal generated successfully at {result['date']}"

    except Exception as e:
        result['status'] = 'ERROR'
        result['message'] = f"Error: {str(e)}"
        print(f"  ERROR: {result['message']}")

    return result


def append_to_log(result):
    """
    신호를 로그 파일에 추가.
    """
    log_file = OUT_DIR / "daily_signals.csv"

    # CSV 컬럼 정의
    columns = [
        'timestamp', 'date', 'price', 'signal', 'signal_type',
        'fast_low_ma', 'slow_low_ma', 'fast_high_ma', 'slow_high_ma',
        'volatility_pct', 'regime', 'status', 'message'
    ]

    # 기존 로그 읽기
    if log_file.exists():
        df_existing = pd.read_csv(log_file)
    else:
        df_existing = pd.DataFrame(columns=columns)

    # 새 행 추가
    df_new = pd.DataFrame([result])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # 최신 100개만 유지 (파일 크기 관리)
    df_combined = df_combined.tail(100)

    df_combined.to_csv(log_file, index=False)
    print(f"  → {log_file} updated ({len(df_combined)} records)")

    return df_combined


def create_html_report(df):
    """
    HTML 리포트 생성 (브라우저로 볼 수 있음).
    """
    html_file = OUT_DIR / "daily_signals.html"

    # 색상 지정
    def color_signal(val):
        if val == 1:
            return 'background-color: #90EE90'  # Green
        elif val == 0:
            return 'background-color: #FFB6C6'  # Light red
        return ''

    def color_regime(val):
        if val == 'HIGH':
            return 'background-color: #FFE4B5'  # Orange
        elif val == 'LOW':
            return 'background-color: #E0F0FF'  # Light blue
        return ''

    # HTML 생성
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Daily Regime-Switching Signal</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px; }}
            .latest {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .signal-box {{ font-size: 24px; font-weight: bold; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .signal-buy {{ background-color: #90EE90; color: #155724; }}
            .signal-hold {{ background-color: #FFB6C6; color: #721c24; }}
            table {{ width: 100%; border-collapse: collapse; background-color: white; }}
            th {{ background-color: #34495e; color: white; padding: 10px; text-align: left; }}
            td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f9f9f9; }}
            .status-success {{ color: green; }}
            .status-error {{ color: red; }}
            .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ Daily Regime-Switching Signal</h1>
            <p>NDX 3x Leverage Rotation Strategy</p>
        </div>

        <div class="latest">
            <h2>Latest Signal</h2>
            <p>Generated: {df.iloc[-1]['timestamp']}</p>
            <p>Date: {df.iloc[-1]['date']} | Price: ${df.iloc[-1]['price']:.2f}</p>
    """

    latest = df.iloc[-1]
    if latest['status'] == 'SUCCESS':
        signal_text = "🟢 BUY" if latest['signal'] == 1 else "🔴 HOLD"
        signal_class = "signal-buy" if latest['signal'] == 1 else "signal-hold"
        html_content += f"""
            <div class="signal-box {signal_class}">
                {signal_text}
            </div>
            <p><strong>Signal Type:</strong> {latest['signal_type']}</p>
            <p><strong>Regime:</strong> {latest['regime']} (Vol: {latest['volatility_pct']:.0f}%)</p>
            <table style="width: 50%;">
                <tr>
                    <td>Low-Vol MA (fast/slow):</td>
                    <td>${latest['fast_low_ma']:.2f} / ${latest['slow_low_ma']:.2f}</td>
                </tr>
                <tr>
                    <td>High-Vol MA (fast/slow):</td>
                    <td>${latest['fast_high_ma']:.2f} / ${latest['slow_high_ma']:.2f}</td>
                </tr>
            </table>
        """
    else:
        html_content += f'<p class="status-error">❌ ERROR: {latest["message"]}</p>'

    html_content += """
        </div>

        <h2>Signal History (Last 20)</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Date</th>
                <th>Price</th>
                <th>Signal</th>
                <th>Type</th>
                <th>Regime</th>
                <th>Status</th>
            </tr>
    """

    for _, row in df.tail(20).iterrows():
        signal_display = "🟢 BUY" if row['signal'] == 1 else "🔴 HOLD"
        status_class = "status-success" if row['status'] == 'SUCCESS' else "status-error"
        html_content += f"""
            <tr>
                <td>{row['timestamp']}</td>
                <td>{row['date']}</td>
                <td>${row['price']:.2f}</td>
                <td><strong>{signal_display}</strong></td>
                <td>{row['signal_type']}</td>
                <td>{row['regime']}</td>
                <td class="{status_class}">{row['status']}</td>
            </tr>
        """

    html_content += """
        </table>

        <div class="footer">
            <p>🤖 Automated Regime-Switching Signal Generator</p>
            <p>Data: Yahoo Finance (^NDX)</p>
            <p>Strategy: Conservative P1 (12,237,6,229,49,57.3%)</p>
            <p>Next update: Daily at 4 PM EST (NYSE close)</p>
        </div>
    </body>
    </html>
    """

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  → {html_file} created")


def main():
    print("=" * 80)
    print("  Daily Regime-Switching Signal Generator")
    print(f"  Parameters: {OPTIMAL_PARAMS}")
    print("=" * 80)

    # 신호 생성
    result = generate_signal()

    print(f"\n  Status: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"  Date: {result['date']}")
        print(f"  Price: ${result['price']:.2f}")
        print(f"  Signal: {result['signal']} ({result['signal_type']})")
        print(f"  Regime: {result['regime']}")
    else:
        print(f"  Message: {result['message']}")

    # 로그 저장
    print("\nUpdating logs...")
    df = append_to_log(result)

    # HTML 리포트 생성
    print("Generating HTML report...")
    create_html_report(df)

    print("\n" + "=" * 80)
    print("  ✓ Done!")
    print(f"  📊 View report: {OUT_DIR / 'daily_signals.html'}")
    print(f"  📝 View logs: {OUT_DIR / 'daily_signals.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
