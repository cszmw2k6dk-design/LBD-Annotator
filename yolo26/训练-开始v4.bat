@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  LBD 训练  dataset_v4  (1545 页 / 226211 框)
echo  起点 : ..\models\best\LBD-yolo26n-2560.pt
echo  参数 : imgsz 2560, batch 4, AdamW lr0=5e-4, 30 轮
echo  预计 : 约 90 分钟/轮, 整轮 45-50 小时
echo ------------------------------------------------------------
echo  这个窗口是训练主进程, 请不要关它。
echo   暂停训练 : 双击 训练-暂停.bat
echo   继续训练 : 双击 训练-继续v4.bat
echo   查看进度 : 双击 训练-状态.bat
echo ============================================================
echo.
".venv\Scripts\python.exe" run_v2.py --name yolo26n_2560_v4 --data "dataset_v4\data.yaml" --start "..\models\best\LBD-yolo26n-2560.pt" --epochs 30 --imgsz 2560
echo.
echo 训练进程已退出。按任意键关闭窗口。
pause >nul
