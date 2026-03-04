# iOS Scriptable Widget: NDX Real-Time Signal

Display NDX regime-switching trading signal on iPhone home screen. Updates hourly via Windows PC → Tailscale VPN → iOS app.

## Architecture

```
Windows PC (Tailscale IP: 100.x.x.x)
  ├── [Hourly] Task Scheduler
  │     └── python scriptable/update_signal_json.py
  │           → Calculates signal, saves to output/ndx_signal.json
  │
  └── [Always Running] signal_server.py (port 8765)
        └── Serves: http://100.x.x.x:8765/ndx_signal.json

iOS (Tailscale connected)
  └── Scriptable Widget
        └── Fetches JSON → Displays signal, price, next trigger
```

## Setup (Complete Instructions)

### 1. Install Tailscale (Both PC and iOS)

**Windows PC:**
```bash
# Download from: https://tailscale.com/download/windows
# Install and log in with your account
tailscale ip -4   # Note this IP address
```

**iOS:**
```
1. App Store → Search "Tailscale"
2. Install and log in with same account
3. After connecting, you'll see your iOS device IP
```

### 2. Setup Windows Tasks (Administrator Required)

```bash
# In Command Prompt (Run as Administrator):
cd C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
scriptable\setup_scheduler.bat
```

This creates two scheduled tasks:
- **NDX Signal Updater**: Hourly (07:00~18:00 ET / 20:00~07:00 KST)
- **NDX Signal Server**: Starts at Windows logon

### 3. Start HTTP Server

#### Option A: Manual (for testing)
```bash
python scriptable/signal_server.py
```

#### Option B: Automatic (via Task Scheduler)
Already set up by `setup_scheduler.bat`. Server starts automatically when you log in.

### 4. Test Locally (Windows)

```bash
# Generate signal manually
python scriptable/update_signal_json.py

# Check JSON was created
type output\ndx_signal.json

# Test HTTP server
python scriptable/signal_server.py

# In another terminal, test the endpoint:
# Open browser: http://localhost:8765/ndx_signal.json
```

Expected output:
```json
{
  "signal": 1,
  "signal_type": "저변동 진입",
  "price": 25034.37,
  "date": "2026-03-03",
  "regime": "LOW",
  "current_fast_ma": 24567.89,
  "current_slow_ma": 23456.78,
  "next_signal_price": 25500.00,
  "price_change_needed": 465.63,
  "price_pct_change": 1.86,
  "crossover_direction": "fast_low (12d) ≤ slow_low (237d)",
  "timestamp": "2026-03-03 21:00:01",
  "volatility_pct": 18.5
}
```

### 5. Find Your Tailscale IP

**Windows PC:**
```bash
tailscale ip -4
# Output example: 100.67.89.123
```

**iOS:**
- Open Tailscale app
- Tap Settings → See "This Device"
- IP shown in blue text (e.g., 100.45.67.890)

### 6. Configure iOS Scriptable Widget

1. **Open scriptable_widget.js and replace Tailscale IP:**
   ```javascript
   // Change this:
   const DATA_URL = "http://100.x.x.x:8765/ndx_signal.json";

   // To your actual PC IP (from step 5):
   const DATA_URL = "http://100.67.89.123:8765/ndx_signal.json";
   ```

2. **On iOS:**
   - Open Scriptable app (download from App Store if not installed)
   - Tap "+" → Create script
   - Paste entire contents of `scriptable_widget.js`
   - Tap "Done" (save with any name, e.g., "NDX Signal")

3. **Add to Home Screen:**
   - Long-press empty area on iPhone home screen
   - Tap "+" → Scriptable
   - Select your script name (e.g., "NDX Signal")
   - Choose widget size: Small or Medium (recommended: Small)
   - Tap "Add Widget"

### 7. Test iOS Widget

1. Make sure:
   - ✅ Tailscale is running on both PC and iOS
   - ✅ HTTP server is running: `python scriptable/signal_server.py`
   - ✅ JSON file exists: `output/ndx_signal.json`
   - ✅ Scriptable DATA_URL has correct IP address

2. On iOS:
   - Open Tailscale app → verify connected to network
   - Go to home screen → widget should update
   - If blank: tap widget to refresh

3. If widget shows "⚠️ No Data":
   - Check Windows Task Scheduler: `schtasks /query /tn "NDX Signal*"`
   - Manually run: `python scriptable/update_signal_json.py`
   - Check JSON exists: `type output\ndx_signal.json`

## Running Schedule

The signal updates **hourly** during US market hours:
- **07:00 ~ 18:00 ET** (US hours)
- **20:00 ~ 07:00 KST** (Korea time, next day)

Outside these hours, widget shows last available signal.

## Files

| File | Purpose |
|------|---------|
| `update_signal_json.py` | Generate signal, save to JSON (hourly via Task Scheduler) |
| `signal_server.py` | HTTP server serving JSON (always running) |
| `start_server.bat` | Manual server start (or auto-start via Task Scheduler) |
| `setup_scheduler.bat` | Setup Windows Tasks (run once as Admin) |
| `README.md` | This file |

## Troubleshooting

### ❌ Widget shows "⚠️ No Data"

```bash
# 1. Check JSON file exists
type C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run\output\ndx_signal.json

# 2. Manually generate signal
python C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run\scriptable\update_signal_json.py

# 3. Check server is running
# Windows Command Prompt:
netstat -ano | find "8765"
# Should show python.exe listening on 8765

# 4. Test locally
# Browser: http://localhost:8765/ndx_signal.json
# Should show JSON

# 5. On iOS, verify Tailscale connection
# Tailscale app → check device is online
```

### ❌ iOS can't reach Windows PC

```bash
# 1. Check Tailscale is connected (both devices)
tailscale status  # Windows
# Should show "Connected to 100.x.x.x"

# 2. Check firewall isn't blocking port 8765
# Windows Defender Firewall → allow Python through firewall
netsh advfirewall firewall add rule name="Python 8765" dir=in action=allow protocol=tcp localport=8765

# 3. Verify actual IP (not 100.x.x.x)
ipconfig | find "IPv4"
# and verify it's different from Tailscale IP
```

### ❌ Task Scheduler not running hourly

```bash
# Check if tasks were created
schtasks /query /tn "NDX Signal*" /fo table /v

# If not found, re-run setup (as Administrator):
# Command Prompt (Run as Admin):
cd C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
scriptable\setup_scheduler.bat

# Test manually
python scriptable\update_signal_json.py
```

### ❌ "ModuleNotFoundError: No module named 'leverage_rotation'"

```bash
# update_signal_json.py can't find main module
# Make sure you run from project root or check sys.path is correct

# Try running from project root:
cd C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run
python scriptable\update_signal_json.py
```

## Advanced: Custom Tailscale Configuration

By default, Tailscale allows direct device-to-device access. To restrict access:

1. **Tailscale Admin Console:** https://login.tailscale.com/admin
2. Create ACL rule to allow iOS → PC port 8765 only
3. Example:
   ```
   {
     "acls": [
       {
         "action": "accept",
         "src": ["tag:iphone"],
         "dst": ["tag:windows:8765"],
       },
     ],
   }
   ```

## One-Line Install (PowerShell, if automated)

```powershell
# Run all setup at once (Windows only, assumes paths)
python "C:\path\to\scriptable\update_signal_json.py"
python "C:\path\to\scriptable\signal_server.py"
```

## Support

For issues:
1. Check `output/ndx_signal.json` exists and has current timestamp
2. Verify `python scriptable/signal_server.py` shows "✓ Signal JSON sent" in logs
3. Check Tailscale is connected on both devices
4. Test locally first: `http://localhost:8765/ndx_signal.json`

## Next Steps

After successful setup:

1. **Keep server running:** Minimize server window or run as background service
2. **Monitor widget:** Check if signal updates hourly on iOS
3. **Trade with signal:** Use widget to guide trading decisions
4. **Optional: Add alerts** via Telegram (see `signals/daily_signal_telegram.py`)

---

**Last Updated:** 2026-03-03
**Version:** 1.0
