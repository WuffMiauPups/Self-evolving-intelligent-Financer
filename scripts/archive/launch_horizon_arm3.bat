@echo off
rem ===================================================================
rem MASTER_PLAN 9.4 arm-3 (horizon-extension). 3-seed retrains at:
rem   H=15 baseline  = configs/default.yaml + configs/training.yaml (primary=15)
rem   H=30 candidate = + configs/experiments/horizon_30.yaml        (primary=30)
rem Seeds 7/13/29 (configs/sweep_seed_variance.yaml). session_aware_labels
rem is ON for both (default) -> the ONLY per-arm change is the direction
rem horizon. Sweeps ISOLATE artifacts; canonical brain.pt is never touched.
rem
rem Launch OUTSIDE any assistant session via launch_horizon_arm3_hidden.vbs
rem so no console window exists to close (MASTER_PLAN 10 note 10). The
rem machine must stay ON for the full ~3h.
rem ===================================================================
cd /d "%~dp0.."
set STATUS=logs\horizon_arm3.status
echo ARM3_START %date% %time% > "%STATUS%"

rem --- H=15 baseline (primary_direction_horizon=15) ---
echo ARM3_H15_BEGIN %date% %time% >> "%STATUS%"
".venv\Scripts\python.exe" scripts\sweep_train.py --real-data --config configs\default.yaml --config configs\training.yaml --sweep-config configs\sweep_seed_variance.yaml --rank-by val_report.balanced_accuracy > logs\horizon_arm3_h15.log 2>&1
echo ARM3_H15_DONE exit=%errorlevel% %date% %time% >> "%STATUS%"

rem --- H=30 candidate (primary_direction_horizon=30) ---
echo ARM3_H30_BEGIN %date% %time% >> "%STATUS%"
".venv\Scripts\python.exe" scripts\sweep_train.py --real-data --config configs\default.yaml --config configs\training.yaml --config configs\experiments\horizon_30.yaml --sweep-config configs\sweep_seed_variance.yaml --rank-by val_report.balanced_accuracy > logs\horizon_arm3_h30.log 2>&1
echo ARM3_H30_DONE exit=%errorlevel% %date% %time% >> "%STATUS%"

echo ARM3_ALL_DONE %date% %time% >> "%STATUS%"
