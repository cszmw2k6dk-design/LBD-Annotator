@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  LBD 训练 v4b  dataset_v4  (1545 页 / 226211 框)
echo  起点 : ..\models\best\LBD-yolo26n-2560.pt
echo  参数 : imgsz 2560, batch 4, AdamW **lr0=0.0001** (比上次小 5 倍), 30 轮
echo  --------------------------------------------------------
echo  为什么重跑：上一轮 lr0=0.0005 太大，把旧图纸的能力覆盖了
echo    (新数据 0.743-^>0.936 变好，旧数据 0.925-^>0.618 变差)
echo  这一轮目标：旧数据守住 ^>0.90，新数据往上追
echo  --------------------------------------------------------
echo  这个窗口是训练主进程, 请不要关它。
echo   暂停 : 训练-暂停.bat   继续 : 训练-继续v4.bat
echo   进度 : 训练-状态.bat
echo ============================================================
echo.
".venv\Scripts\python.exe" run_v2.py --name yolo26n_2560_v4b --data "dataset_v4\data.yaml" --start "..\models\best\LBD-yolo26n-2560.pt" --epochs 30 --imgsz 2560 --lr0 0.0001
echo.
echo 训练进程已退出。按任意键关闭窗口。
pause >nul
