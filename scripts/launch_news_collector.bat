@echo off
rem ===================================================================
rem NEWSBOX N0 collector (NEWSBOX_PLAN, ungated): one-time historical
rem backfill 2022-01 -> now, then the incremental poller forever.
rem Archive: runs\news\news_YYYYMM.jsonl (append-only, id-dedup).
rem
rem Re-runnable: runs\news\.backfill_done skips the backfill; the poller
rem resumes from the newest archived timestamp, so machine-off gaps
rem self-heal. Launch via launch_news_collector_hidden.vbs (10.10 - no
rem console window to close). Survives session restarts; dies only with
rem the machine.
rem ===================================================================
cd /d "%~dp0.."
set STATUS=logs\news_collector.status
echo N0_START %date% %time% >> "%STATUS%"

if exist runs\news\.backfill_done goto poll
echo N0_BACKFILL_BEGIN %date% %time% >> "%STATUS%"
".venv\Scripts\python.exe" scripts\collect_news.py --backfill --start 2022-01-01 > logs\news_backfill.log 2>&1
if %errorlevel% neq 0 (
    echo N0_BACKFILL_FAILED exit=%errorlevel% %date% %time% >> "%STATUS%"
    exit /b %errorlevel%
)
echo done > runs\news\.backfill_done
echo N0_BACKFILL_DONE %date% %time% >> "%STATUS%"

:poll
echo N0_POLL_BEGIN %date% %time% >> "%STATUS%"
".venv\Scripts\python.exe" scripts\collect_news.py --poll --interval 60 >> logs\news_poller.log 2>&1
echo N0_POLL_EXITED exit=%errorlevel% %date% %time% >> "%STATUS%"
