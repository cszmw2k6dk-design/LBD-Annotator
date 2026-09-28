@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" run_v2.py --status
echo.
echo ---------- health.log (last 12) ----------
powershell -NoProfile -Command "if (Test-Path 'runs\health.log') { Get-Content 'runs\health.log' -Encoding UTF8 -Tail 12 } else { 'no health log yet' }"
echo.
pause
