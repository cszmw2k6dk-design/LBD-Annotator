@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" run_v2.py --epochs 30 --imgsz 2560
echo.
pause
