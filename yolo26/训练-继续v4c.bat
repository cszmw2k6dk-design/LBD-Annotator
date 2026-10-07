@echo off
rem ASCII only + CRLF: cmd misparses this file otherwise.
cd /d "%~dp0"
echo ============================================================
echo  LBD training v4c - resume from last.pt (runs\detect\yolo26n_2560_v4c)
echo  data : dataset_v4\data.yaml   imgsz 2560   batch 4   30 epochs
echo ------------------------------------------------------------
echo  THIS WINDOW IS THE TRAINING PROCESS - do not close it.
echo  DO NOT press Ctrl+C in this window (it kills training).
echo ============================================================
echo.
".venv\Scripts\python.exe" run_v2.py --name yolo26n_2560_v4c --data "dataset_v4\data.yaml" --epochs 30 --imgsz 2560 --batch 4 --lr0 0.0005
echo.
echo Training exited. Press any key to close.
pause >nul
