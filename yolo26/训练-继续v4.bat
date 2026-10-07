@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem dataset_v4（1545 页）从 models\best 微调，30 轮，imgsz 2560。
rem 有 last.pt 就自动续训，否则从头开始。
".venv\Scripts\python.exe" run_v2.py --name yolo26n_2560_v4 --data "dataset_v4\data.yaml" --start "..\models\best\LBD-yolo26n-2560.pt" --epochs 30 --imgsz 2560
echo.
pause
