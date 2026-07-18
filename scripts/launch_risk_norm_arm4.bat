@echo off
rem ===================================================================
rem MASTER_PLAN 9.4 arm-4 (risk-target per-symbol norm). 3-seed retrain:
rem   knob ON = configs/default.yaml + configs/training.yaml
rem            + configs/experiments/risk_target_norm.yaml (primary=15).
rem Seeds 7/13/29 (configs/sweep_seed_variance.yaml). The ONLY change vs
rem the knob-OFF baseline is training.risk_target_per_symbol_norm=true;
rem the knob-OFF baseline is arm-3's H=15 sweep (checkpoints/sweep_20260718_065658).
rem The knob is a post-cache transform -> reuses the sample cache (fast).
rem Sweeps ISOLATE artifacts; canonical brain.pt is never touched.
rem
rem Launch OUTSIDE any assistant session via launch_risk_norm_arm4_hidden.vbs
rem so no console window exists to close (MASTER_PLAN 10 note 10).
rem ===================================================================
cd /d "%~dp0.."
set STATUS=logs\risk_norm_arm4.status
echo ARM4_START %date% %time% > "%STATUS%"
".venv\Scripts\python.exe" scripts\sweep_train.py --real-data --config configs\default.yaml --config configs\training.yaml --config configs\experiments\risk_target_norm.yaml --sweep-config configs\sweep_seed_variance.yaml --rank-by val_report.balanced_accuracy > logs\risk_norm_arm4.log 2>&1
echo ARM4_DONE exit=%errorlevel% %date% %time% >> "%STATUS%"
