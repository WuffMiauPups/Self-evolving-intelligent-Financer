@echo off
rem Full-data noise-floor sweep (MASTER_PLAN 7.0.1): seeds 7/13/29, GPU.
rem Launch OUTSIDE any assistant session (double-click, plain terminal, or
rem   schtasks /run /tn SETAI_noise_floor_sweep
rem after enabling the task) so it survives session restarts - 10.7.
rem Output: runs\sweep_<id>_runNNN.json summaries + this log.
cd /d "%~dp0.."
".venv\Scripts\python.exe" scripts\sweep_train.py --real-data --sweep-config configs\sweep_seed_variance.yaml --rank-by val_report.balanced_accuracy >> logs\sweep_gpu_%date:~-4%%date:~-7,2%%date:~-10,2%.log 2>&1
