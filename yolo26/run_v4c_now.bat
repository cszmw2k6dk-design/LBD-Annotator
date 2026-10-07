@echo off
rem Launcher for the self-healing scheduled task (every 5 minutes).
rem ASCII only + CRLF line endings: cmd misparses this file otherwise.
rem Skips when the user paused training via the pause script (runs\STOP).
cd /d "%~dp0"
if exist "runs\STOP" exit /b 0
".venv\Scripts\python.exe" run_v2.py --name yolo26n_2560_v4c --data "dataset_v4\data.yaml" --start "runs\detect\yolo26n_2560_v4\weights\best.pt" --epochs 30 --imgsz 2560 --lr0 0.0005
