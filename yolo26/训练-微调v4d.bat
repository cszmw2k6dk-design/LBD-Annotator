@echo off
rem ASCII only + CRLF: cmd misparses this file otherwise.
cd /d "%~dp0"
echo ============================================================
echo  LBD fine-tune v4d  (3 epochs, lr 0.0002, mosaic off)
echo  start  : runs\detect\yolo26n_2560_v4c\weights\best.pt
echo  data   : dataset_v4\data_v4_cleanval.yaml  (val = 215 clean pages)
echo ------------------------------------------------------------
echo  THIS WINDOW IS THE TRAINING PROCESS - do not close it.
echo  DO NOT press Ctrl+C in this window (it kills training).
echo ============================================================
echo.
".venv\Scripts\python.exe" run_v2.py --name yolo26n_2560_v4d --data "dataset_v4\data_v4_cleanval.yaml" --start "runs\detect\yolo26n_2560_v4c\weights\best.pt" --epochs 3 --imgsz 2560 --batch 4 --lr0 0.0002 --patience 3 --extra "mosaic=0.0"
echo.
echo Training exited. Press any key to close.
pause >nul
