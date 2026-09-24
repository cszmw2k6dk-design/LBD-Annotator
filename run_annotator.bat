@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=python"
where py >nul 2>&1 && set "PY=py"
if exist "%~dp0..\.build-venv\Scripts\python.exe" set "PY=%~dp0..\.build-venv\Scripts\python.exe"
"%PY%" "%~dp0lbd_annotator.py" %*
if errorlevel 1 pause
