@echo off
REM Start NDX Signal HTTP Server
REM Run this manually or via Task Scheduler at logon

cd /d "C:\Users\anti_\Documents\260224_백테스트\leverage-for-the-long-run"
python scriptable\signal_server.py

REM Keep window open if script exits with error
pause
