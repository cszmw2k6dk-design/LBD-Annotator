"""看训练进度：当前第几轮、还要多久、各类指标、best 在哪。

用法:
    python progress.py                      # 自动挑最新的那轮训练
    python progress.py --name yolo26n_1920_tracker
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "detect"
LOGS = HERE / "runs"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BAR = re.compile(
    r"(\d+)/(\d+)\s+.*?(\d+)%\s+\S*\s*(\d+)/(\d+)\s+"
    r"(?:([\d.]+)(s/it|it/s)\s+)?([\d:]+)<([\d:]+)"
)
CLSROW = re.compile(
    r"^\s+(\S+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
)


def hms(sec: float) -> str:
    sec = int(max(0, sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}小时{m}分" if h else f"{m}分{s}秒"


def to_sec(t: str) -> float:
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def pick_run(name: str) -> Path | None:
    if not RUNS.is_dir():
        return None
    if name:
        return RUNS / name
    cands = [d for d in RUNS.iterdir() if d.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda d: (d / "results.csv").stat().st_mtime
               if (d / "results.csv").exists() else d.stat().st_mtime)


def pick_log(name: str) -> Path | None:
    logs = sorted(LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    logs = [p for p in logs if p.stat().st_size > 0]
    if not logs:
        return None
    if name:
        for p in logs:
            if name in p.name:
                return p
    return logs[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    run = pick_run(args.name)
    log = Path(args.log) if args.log else pick_log(args.name)

    running = False
    if log and log.exists():
        age = time.time() - log.stat().st_mtime
        running = age < 180
        state = "训练进行中" if running else f"日志已 {hms(age)} 没更新（可能已停止/结束）"
        print(f"日志: {log.name}   [{state}]")
    else:
        print("没找到训练日志")

    stopped = False
    if run is None or not run.exists():
        print(f"还没有结果目录（第一轮跑完后才会生成 {RUNS}）")
        if not log:
            return 1
    else:
        cfg = {}
        ay = run / "args.yaml"
        if ay.exists():
            for line in ay.read_text(encoding="utf-8", errors="replace").splitlines():
                k, _, v = line.partition(": ")
                cfg[k.strip()] = v.strip()
        print(f"结果目录: {run.name}")
        print(f"  模型 {cfg.get('model', '?')}  imgsz {cfg.get('imgsz', '?')}  "
              f"计划 {cfg.get('epochs', '?')} 轮  batch {cfg.get('batch', '?')}  "
              f"patience {cfg.get('patience', '?')}")

        rows = []
        rp = run / "results.csv"
        if rp.exists():
            with rp.open(encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
        if rows:
            done = len(rows)
            total = int(float(cfg.get("epochs", done) or done))
            times = [float(r["time"]) for r in rows]
            per_epoch = (times[-1] - times[0]) / max(1, done - 1) if done > 1 else times[-1]
            best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"] or 0))
            last = rows[-1]
            print(f"  已完成 {done}/{total} 轮   每轮约 {hms(per_epoch)}   已跑 {hms(times[-1])}")
            if done < total:
                print(f"  预计还要 {hms((total - done) * per_epoch)}"
                      f"（大约 {time.strftime('%m-%d %H:%M', time.localtime(time.time() + (total - done) * per_epoch))} 结束）")
            print(f"  最近一轮: mAP50 {last['metrics/mAP50(B)']}  mAP50-95 {last['metrics/mAP50-95(B)']}"
                  f"  P {last['metrics/precision(B)']}  R {last['metrics/recall(B)']}")
            print(f"  最好一轮: 第 {best['epoch']} 轮  mAP50 {best['metrics/mAP50(B)']}"
                  f"  mAP50-95 {best['metrics/mAP50-95(B)']}")
            tail = rows[-5:]
            print("  近几轮 mAP50: " + "  ".join(f"{r['epoch']}={float(r['metrics/mAP50(B)']):.3f}"
                                                 for r in tail))
        else:
            print("  results.csv 还没生成（第 1 轮跑完才有）")

        best_pt = run / "weights" / "best.pt"
        if best_pt.exists():
            print(f"  best.pt: {best_pt}  ({time.strftime('%H:%M:%S', time.localtime(best_pt.stat().st_mtime))} 更新)")

    if log and log.exists():
        lines = [ANSI.sub("", ln) for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()]
        cur = None
        for ln in lines:
            m = BAR.search(ln)
            if m:
                cur = m
        if cur and running:
            ep, eps, pct, it, its, speed, unit, ela, rem = cur.groups()
            shown = f"{speed}{unit}" if speed else "?"
            eta = f"，本轮还要 {hms(to_sec(rem))}" if rem else ""
            print(f"  当前: 第 {ep}/{eps} 轮  {pct}%  ({it}/{its} batch, {shown}){eta}")

        # 每次验证都是 "all" 打头，后面可能跟 Node/Tracker 行（有些 ultralytics 版本不打印分类别）。
        # 所以按 "all" 切块，只显示最后一块，别把历次的 all 混在一起。
        blocks: list[list] = []
        cur: list = []
        for ln in lines:
            m = CLSROW.match(ln)
            if m and m.group(1) in ("all", "Node", "Tracker", "Typical", "Box"):
                if cur and m.group(1) == "all":
                    blocks.append(cur)
                    cur = []
                cur.append(m.groups())
            elif not ln.strip() and cur:
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        if blocks:
            last = blocks[-1]
            has_cls = any(b[0] in ("Node", "Tracker", "Typical", "Box") for b in last)
            title = "最近一次验证（按类别）" if has_cls else \
                "最近一次验证（整体；这一版 ultralytics 的日志没打印分类别行）"
            print(f"  {title}:")
            print(f"    {'类别':<10}{'框数':>7}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>11}")
            for name, _imgs, n, p, r, m50, m5095 in last:
                print(f"    {name:<10}{n:>7}{float(p):>9.3f}{float(r):>9.3f}"
                      f"{float(m50):>9.3f}{float(m5095):>11.3f}")
        elif not stopped:
            print("  还没跑过验证（第 1 轮结束时会有一张分类别的小表）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
