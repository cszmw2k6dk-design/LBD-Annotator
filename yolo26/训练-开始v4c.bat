@echo off
rem ASCII only + CRLF line endings: cmd misparses this file otherwise.
cd /d "%~dp0"
echo ============================================================
echo  LBD training  v4c   dataset_v4  (1545 pages / 226211 boxes)
echo  start  : runs\detect\yolo26n_2560_v4\weights\best.pt
echo  params : imgsz 2560, batch 4, AdamW lr0=0.0005, 30 epochs
echo ------------------------------------------------------------
echo  THIS WINDOW IS THE TRAINING PROCESS - do not close it.
echo  DO NOT press Ctrl+C in this window (it kills training).
echo  To stop:  double-click the "pause" script in this folder
echo  Progress: double-click the "status" script in this folder
echo ============================================================
echo.
".venv\Scripts\python.exe" run_v2.py --name yolo26n_2560_v4c --data "dataset_v4\data.yaml" --start "runs\detect\yolo26n_2560_v4\weights\best.pt" --epochs 30 --imgsz 2560 --lr0 0.0005
echo.
echo Training exited. Press any key to close.
pause >nul
