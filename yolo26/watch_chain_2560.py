"""第 3 步：等阶段 2（1920 续训）跑完，自动接着做一轮 2560 分辨率微调。

为什么要提到 2560：Tracker 在 imgsz 1920 下只有约 5 像素宽，细线漏检的根因是像素不够。
数据集本来就是按长边 2560 预处理的，训 2560 时 ultralytics 不用再缩放（r=1.0），
目标变成约 6.8 像素。阶段 1 的后 5 轮 mAP50-95 已经平在 0.767~0.772，说明 1920 到头了。

启动前会自动比较前两轮的结果，从更好的那份 best.pt 出发。
用法: python watch_chain_2560.py
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
RUNS = HERE / "runs" / "detect"

WAIT_FOR = "yolo26n_1920_ft40"                  # 等这轮结束
CANDIDATES = ("yolo26n_1920_tracker", "yolo26n_1920_ft40")   # 从这两份里挑起点

NEXT_NAME = "yolo26n_2560_ft25"
NEXT_ARGS = [
    "train_yolo26.py", "--imgsz", "2560", "--epochs", "25", "--batch", "4",
    "--cache", "disk", "--patience", "10", "--name", NEXT_NAME,
    "rect=True", "amp=False", "save_period=5", "cos_lr=True", "seed=0",
    "lr0=0.0005", "erasing=0.1", "scale=0.3",
]

LOG = HERE / "runs" / "watch_chain2560.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def training_alive(tag: str) -> bool:
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        cmd = " ".join(str(x) for x in (p.info.get("cmdline") or []))
        if "train_yolo26.py" in cmd and tag in cmd:
            return True
    return False


def fitness(run: str) -> float:
    """ultralytics 选 best 用的就是 0.1*mAP50 + 0.9*mAP50-95。"""
    p = RUNS / run / "results.csv"
    if not p.exists():
        return -1.0
    rows = list(csv.DictReader(p.open(encoding="utf-8", errors="replace")))
    if not rows:
        return -1.0
    return max(0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
               for r in rows)


def best_run() -> str:
    scored = [(fitness(r), r) for r in CANDIDATES
              if (RUNS / r / "weights" / "best.pt").exists()]
    if not scored:
        return ""
    scored.sort(reverse=True)
    for f, r in scored:
        log(f"  候选 {r}: fitness {f:.4f}")
    return scored[0][1]


def main() -> int:
    log(f"看门狗启动，等待 {WAIT_FOR} 结束…")
    while training_alive(WAIT_FOR):
        time.sleep(30)
    log(f"{WAIT_FOR} 已结束，比较前两轮，挑续训起点")

    src = best_run()
    if not src:
        log("[中止] 找不到可用的 best.pt")
        return 1
    weights = RUNS / src / "weights" / "best.pt"
    log(f"起点选定：{src}  ->  {weights}")

    if (RUNS / NEXT_NAME / "results.csv").exists():
        log(f"[跳过] {NEXT_NAME} 已经有结果了")
        return 0

    # 把 --model <起点权重> 插在脚本名后面
    args = [NEXT_ARGS[0], "--model", str(weights), *NEXT_ARGS[1:]]
    log(f"启动 2560 微调：{NEXT_NAME}，25 轮，batch 4，lr0=0.0005")
    out = (HERE / "runs" / f"{NEXT_NAME}_train.log").open("w", encoding="utf-8")
    err = (HERE / "runs" / f"{NEXT_NAME}_train.err.log").open("w", encoding="utf-8")
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    p = subprocess.Popen([str(PY), *args], cwd=HERE, stdout=out, stderr=err,
                         creationflags=flags, close_fds=True)
    log(f"已启动，PID={p.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
