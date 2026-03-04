@echo off
REM Setup Windows Task Scheduler for NDX Signal Updates
REM Run this with Administrator privileges

echo ============================================================
echo  NDX Signal Task Scheduler Setup
echo ============================================================

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: This script must be run as Administrator!
    echo.
    echo How to fix:
    echo   1. Press Win+R
    echo   2. Type: cmd
    echo   3. Press Ctrl+Shift+Enter (Run as Administrator)
    echo   4. Run this script again
    echo.
    pause
    exit /b 1
)

setlocal enabledelayedexpansion

REM Project root
set PROJECT_ROOT=C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run

REM ================================================================
REM Task 1: Update Signal JSON hourly (07:00~18:00 ET = 20:00~07:00 KST)
REM ================================================================

echo.
echo Creating Task 1: NDX Signal Hourly Update...
echo.

REM Delete existing task if present
schtasks /delete /tn "NDX Signal Updater" /f >nul 2>&1

REM Create new task
REM Windows local time (KST): 20:00~07:00 (8PM to 7AM next day)
REM Trigger: Hourly from 20:00 to 07:00

for /L %%H in (20,1,23) do (
    echo   Creating trigger for hour %%H:00...
    schtasks /create /tn "NDX Signal Updater %%H" ^
             /tr "python %PROJECT_ROOT%\scriptable\update_signal_json.py" ^
             /sc once /st %%H:00 /f >nul 2>&1
)

for /L %%H in (0,1,6) do (
    echo   Creating trigger for hour %%H:00 (next day)...
    schtasks /create /tn "NDX Signal Updater %%H" ^
             /tr "python %PROJECT_ROOT%\scriptable\update_signal_json.py" ^
             /sc once /st %%H:00 /f >nul 2>&1
)

echo ✓ Created hourly update tasks (20:00~07:00 KST)

REM ================================================================
REM Task 2: Start HTTP Server at logon (Optional)
REM ================================================================

echo.
echo Creating Task 2: Start Signal Server at Logon...
echo.

REM Delete existing task if present
schtasks /delete /tn "NDX Signal Server" /f >nul 2>&1

REM Create task: Run at logon in background
schtasks /create /tn "NDX Signal Server" ^
         /tr "python %PROJECT_ROOT%\scriptable\signal_server.py" ^
         /sc onlogon /rl highest /f /np >nul 2>&1

echo ✓ Created logon startup task

REM ================================================================
REM Verify installation
REM ================================================================

echo.
echo ============================================================
echo  Verification
echo ============================================================
echo.
echo Active NDX Signal tasks:
schtasks /query /tn "NDX Signal*" /fo table /v 2>nul | find "NDX"

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Start server manually: %PROJECT_ROOT%\scriptable\start_server.bat
echo   2. Test signal generation: python %PROJECT_ROOT%\scriptable\update_signal_json.py
echo   3. Check JSON file: %PROJECT_ROOT%\output\ndx_signal.json
echo   4. Setup iOS Scriptable:
echo      - Find your Tailscale IP: tailscale ip -4
echo      - Update scriptable_widget.js DATA_URL to http://YOUR_IP:8765/ndx_signal.json
echo.
echo To remove tasks later:
echo   schtasks /delete /tn "NDX Signal*" /f
echo.
pause
