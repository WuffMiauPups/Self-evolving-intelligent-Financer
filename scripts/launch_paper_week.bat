@echo off
title SETAI PAPER WEEK - KEEP THIS WINDOW OPEN (closing it kills the trader)
cd /d "%~dp0.."
echo ============================================================
echo  SETAI unattended paper week (MASTER_PLAN 9.2)
echo.
echo  - This window must STAY OPEN all week. Closed consoles
echo    killed three runs during the week of 07-14 (10.10).
echo  - The machine stays ON (check Windows sleep settings).
echo  - Live view:  .venv\Scripts\python.exe scripts\trading_dashboard.py
echo                then open http://127.0.0.1:8787
echo  - Stop DELIBERATELY with Ctrl+C (graceful shutdown: cancels
echo    orders, persists memory, final snapshot).
echo ============================================================
echo.
".venv\Scripts\python.exe" scripts\preflight_monday.py
if errorlevel 1 (
  echo.
  echo PREFLIGHT FAILED - trader NOT started. Fix the checks above.
  pause
  exit /b 1
)
echo.
".venv\Scripts\python.exe" scripts\run_paper.py
echo.
echo Trader exited. Tail of logs\seif_runtime.log has the reason.
pause
