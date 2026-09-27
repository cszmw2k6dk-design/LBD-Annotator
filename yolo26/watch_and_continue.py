"""等当前训练结束后，自动从 best.pt 接着做一轮微调续训（+40 轮）。

为什么不能直接改 epochs：ultralytics 的 resume 会把 checkpoint 里存的
train_args 整个恢复，新传的 epochs/imgsz 等都会被忽略（标注工具面板里
"勾上后 epochs 会被忽略"就是这个原因）。所以续训要用「载入 best.pt 再训」
的方式，配一个更低的学习率。

用法: python watch_and_continue.py            （后台跑着就行）
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"

# 等这轮跑完
WAIT_FOR = "yolo26n_1920_tracker"
WAIT_RUN_DIR = HERE / "runs" / "detect" / WAIT_FOR

# 接着跑的微调（用上面那轮的最好权重，学习率降到 1/3 左右）
FT_NAME = "yolo26n_1920_ft40"
FT_FROM = WAIT_RUN_DIR / "weights" / "best.pt"
FT_ARGS = [
    "train_yolo26.py", "--model", str(FT_FROM),
    "--imgsz", "1920", "--epochs", "40", "--batch", "8", "--cache", "disk",
    "--patience", "15", "--name", FT_NAME,
    "rect=True", "amp=False", "save_period=5", "cos_lr=True", "seed=0",
    "lr0=0.0005", "erasing=0.1", "scale=0.3",
]

LOG = HERE / "runs" / "watch_continue.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def training_alive() -> int | None:
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        cmd = " ".join(str(x) for x in (p.info.get("cmdline") or []))
        if "train_yolo26.py" in cmd and WAIT_FOR in cmd:
            return p.info["pid"]
    return None


def main() -> int:
    log(f"看门狗启动，等待 {WAIT_FOR} 结束…")
    while True:
        pid = training_alive()
        if pid is None:
            break
        time.sleep(30)
    log(f"{WAIT_FOR} 已结束，准备续训")

    if not FT_FROM.exists():
        log(f"[中止] 找不到 {FT_FROM}")
        return 1
    if (HERE / "runs" / "detect" / FT_NAME / "results.csv").exists():
        log(f"[跳过] {FT_NAME} 已经有结果了")
        return 0

    log(f"启动续训：{FT_NAME}，从 {FT_FROM.name} 开始，40 轮，lr0=0.0005")
    out = (HERE / "runs" / f"{FT_NAME}_train.log").open("w", encoding="utf-8")
    err = (HERE / "runs" / f"{FT_NAME}_train.err.log").open("w", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    p = subprocess.Popen([str(PY), *FT_ARGS], cwd=HERE, stdout=out, stderr=err,
                         creationflags=flags, close_fds=True)
    log(f"续训已启动，PID={p.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
