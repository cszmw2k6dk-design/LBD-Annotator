@echo off
cd /d "%~dp0"
echo stop > "runs\STOP"
echo.
echo [PAUSE] Training will stop within 10 seconds. Progress (last.pt) is kept.
echo [PAUSE] Double-click "训练-继续.bat" later to continue from where it stopped.
echo.
timeout /t 6 >nul
