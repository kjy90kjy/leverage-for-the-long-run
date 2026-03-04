"""
HTTP Server for NDX Signal JSON

Serves output/ndx_signal.json via HTTP on Tailscale IP.
Access from iOS Scriptable: http://100.x.x.x:8765/ndx_signal.json

Run manually:
  python signal_server.py

Or as scheduled service:
  - start_server.bat (manual start)
  - Windows Task Scheduler (auto-start at logon via start_server.bat)
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys
from pathlib import Path
import logging

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

PORT = 8765
OUTPUT_DIR = Path(__file__).parent.parent / "output"
HOST = "0.0.0.0"  # Listen on all network interfaces (Tailscale will bind to 100.x.x.x)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class SignalRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for signal requests"""

    def do_GET(self):
        """Handle GET request"""
        logger.info(f"Request: {self.path} from {self.client_address[0]}")

        # Handle /ndx_signal.json
        if self.path == '/ndx_signal.json':
            signal_file = OUTPUT_DIR / "ndx_signal.json"

            if not signal_file.exists():
                self.send_response(404)
                self.send_header('Content-type', 'text/json')
                self.end_headers()
                self.wfile.write(b'{"error": "Signal file not found. Run update_signal_json.py first."}')
                logger.warning("Signal file not found!")
                return

            try:
                with open(signal_file, 'rb') as f:
                    data = f.read()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Content-Length', len(data))
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data)
                logger.info("✓ Signal JSON sent")
                return

            except Exception as e:
                logger.error(f"Error reading signal file: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'text/json')
                self.end_headers()
                self.wfile.write(b'{"error": "Server error"}')
                return

        # Handle /health (status check)
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
            return

        # Handle root path
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""
<html>
<head><title>NDX Signal Server</title></head>
<body style="font-family: monospace; margin: 20px;">
    <h1>NDX Signal Server</h1>
    <p>Server is running on port {PORT}</p>
    <p><strong>Endpoints:</strong></p>
    <ul>
        <li><a href="/ndx_signal.json">/ndx_signal.json</a> - Current signal JSON</li>
        <li><a href="/health">/health</a> - Health check</li>
    </ul>
    <p><strong>iOS Access:</strong></p>
    <p>http://100.x.x.x:8765/ndx_signal.json</p>
    <p style="color: #666; font-size: 12px;">
        Replace 100.x.x.x with your Tailscale IP address.
        Find it with: <code>tailscale ip -4</code>
    </p>
</body>
</html>
            """.encode('utf-8')
            self.wfile.write(html)
            return

        # 404 for other paths
        self.send_response(404)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Not Found')

    def log_message(self, format, *args):
        """Suppress default HTTP logging (we use logging module)"""
        pass


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create and start server
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, SignalRequestHandler)

    logger.info("=" * 80)
    logger.info("NDX Signal Server")
    logger.info("=" * 80)
    logger.info(f"Listening on {HOST}:{PORT}")
    logger.info(f"Serving from: {OUTPUT_DIR}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  - Local: http://localhost:{PORT}/ndx_signal.json")
    logger.info(f"  - Tailscale: http://100.x.x.x:{PORT}/ndx_signal.json")
    logger.info(f"  - Health: http://localhost:{PORT}/health")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 80)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        httpd.shutdown()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
