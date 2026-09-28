"""新数据集（dataset_v2）长跑训练器：崩了自动续训，全程写日志。

为什么要单独一个入口：本机 CPU 跑 2560 一轮要 ~80 分钟、整轮 40 小时，
中间任何一次重启/断电/误关都会中断。ultralytics 每轮会存 last.pt，
所以只要进程挂了就用 --resume 从 last.pt 接上，最多丢当前这一轮。

用法:
    # 首次开训（从上一版 best 微调）
    python run_v2.py --epochs 40 --imgsz 2560

    # 中途挂了以后重新拉起来（会自动判断该 resume 还是从头开始）
    python run_v2.py --epochs 40 --imgsz 2560

    # 只是想看一眼现在什么状态，不启动训练
    python run_v2.py --status
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:  # 没有 psutil 就退化成只跑训练、不体检
    psutil = None

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
RUNS = HERE / "runs" / "detect"
LOG = HERE / "runs" / "run_v2.log"
HEALTH = HERE / "runs" / "health.log"
# 想让训练停下来，就在 runs\ 下放一个叫 STOP 的文件（暂停训练.bat 会帮你建）
STOP_FILE = HERE / "runs" / "STOP"

# 体检阈值：只对"会让训练真的崩掉"的情况下手
#
# 为什么不拿内存当停止条件：这个数据集第一次跑时 ultralytics 会写 14.7GB 的 .npy 磁盘缓存，
# 写的那十几分钟里 psutil 报"可用内存 18MB"，看着吓死人，但训练其实一直是 20 秒/批全速在跑，
# 缓存写完就自己恢复到 8GB。拿它当停止条件只会白白误杀长跑。
# 内存真不够时系统会 OOM 杀掉训练，外层本来就会自动续训，不需要我们提前停。
MIN_FREE_DISK = 8 * 1024 ** 3         # C 盘剩余低于 8GB —— 缓存和 checkpoint 写不下
STALL_MINUTES = 45                    # 指标和日志 45 分钟没动 —— 疑似卡死
CHECK_INTERVAL = 10                   # 每 10 秒看一眼（暂停标志 + 体检）
POWER_BAD_NEEDED = 18                 # 没插电连续 18 次（约 3 分钟）才停，搬电脑不算


def log(msg: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def health(msg: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with HEALTH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def health_check() -> tuple[str | None, str | None]:
    """返回 (立刻停的理由, 需要持续一段时间才停的理由)。"""
    if psutil is None:
        return None, None
    free = shutil.disk_usage("C:\\").free
    if free < MIN_FREE_DISK:
        return f"C 盘只剩 {free / 1e9:.1f}GB（低于 {MIN_FREE_DISK / 1e9:.0f}GB），训练随时写崩", None
    bat = psutil.sensors_battery()
    if bat is not None and not bat.power_plugged:
        return None, f"笔记本没插电源（电量 {bat.percent:.0f}%）—— 40 小时长训必须插着电"
    return None, None


def health_snapshot() -> str:
    if psutil is None:
        return "psutil 不可用，跳过体检"
    vm = psutil.virtual_memory()
    free = shutil.disk_usage("C:\\").free
    try:
        freq = psutil.cpu_freq()
        f = f"{freq.current / 1000:.2f}GHz" if freq else "?"
    except Exception:  # noqa: BLE001
        f = "?"
    bat = psutil.sensors_battery()
    p = ("插电" if bat.power_plugged else f"电池{bat.percent:.0f}%") if bat else "?"
    return (f"可用内存 {vm.available / 1e9:.1f}GB / C盘剩 {free / 1e9:.0f}GB / "
            f"CPU {f} / 负载 {psutil.cpu_percent():.0f}% / {p}")


def supervise(proc: subprocess.Popen, reason: list[str], watch: list[Path]) -> None:
    """一边等训练、一边体检 + 看暂停标志。要停就把训练进程收掉。"""
    last_snap = 0.0
    hard_streak = 0
    power_streak = 0
    while proc.poll() is None:
        if STOP_FILE.exists():
            reason.append("用户手动暂停（runs\\STOP）")
            break
        if psutil is not None:
            hard, soft = health_check()
            if hard:
                hard_streak += 1
                health(f"[严重 {hard_streak}/2] {hard}")
                if hard_streak >= 2:     # 连续两次确认再动手，避免瞬时抖动误停
                    reason.append(hard)
                    break
            else:
                hard_streak = 0
            # 拔电源可能是搬个电脑，给 3 分钟缓冲；一直没插回来才停
            if soft:
                power_streak += 1
                if power_streak in (6, 18):
                    health(f"[提示 {power_streak}/{POWER_BAD_NEEDED}] {soft}")
                if power_streak >= POWER_BAD_NEEDED:
                    reason.append(soft)
                    break
            else:
                if power_streak >= 6:
                    health("[提示] 已重新插电，继续跑")
                power_streak = 0
            now = time.time()
            if now - last_snap >= 600:   # 每 10 分钟记一条体检，方便事后看有没有异常
                last_snap = now
                health(health_snapshot())
        # 卡死判定：指标文件（每轮更新）和训练日志都超过 45 分钟没动
        mt = [p.stat().st_mtime for p in watch if p.exists()]
        if mt and (time.time() - max(mt)) > STALL_MINUTES * 60:
            reason.append(f"训练日志和指标已经 {STALL_MINUTES} 分钟没有任何更新，疑似卡死")
            break
        time.sleep(CHECK_INTERVAL)

    if reason:
        log(f"[停止] {reason[0]} —— 正在收掉训练进程")
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        health(f"[已停止] {reason[0]}")


def run_state(name: str) -> dict:
    """看一眼这轮训练跑到哪了。"""
    d = RUNS / name
    out = {"name": name, "dir": d, "last": d / "weights" / "last.pt",
           "best": d / "weights" / "best.pt", "csv": d / "results.csv",
           "epochs_done": 0, "epochs_total": 0, "best_fitness": float("-inf"),
           "last_mtime": 0.0}
    if out["csv"].exists():
        lines = [l for l in out["csv"].read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) > 1:
            hdr = [h.strip() for h in lines[0].split(",")]
            out["epochs_done"] = len(lines) - 1
            # ultralytics 选 best 用的是 fitness = 0.1*mAP50 + 0.9*mAP50-95
            fi = {k: hdr.index(k) for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)") if k in hdr}
            if len(fi) == 2:
                a, b = fi["metrics/mAP50(B)"], fi["metrics/mAP50-95(B)"]
                fits = [0.1 * float(r.split(",")[a]) + 0.9 * float(r.split(",")[b])
                        for r in lines[1:]]
                out["best_fitness"] = max(fits)
        out["last_mtime"] = out["csv"].stat().st_mtime
    return out


def print_status(name: str) -> None:
    st = run_state(name)
    if not st["csv"].exists():
        print(f"[状态] {name}: 还没开始（没有 results.csv）")
        return
    done, best = st["epochs_done"], st["best_fitness"]
    when = datetime.fromtimestamp(st["last_mtime"])
    print(f"[状态] {name}: 已完成 {done} 轮，当前最好 fitness={best:.4f}，"
          f"最后更新 {when:%m-%d %H:%M}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo26n_2560_v2")
    ap.add_argument("--data", default=str(HERE / "dataset_v2" / "data.yaml"))
    ap.add_argument("--start", default=str(HERE.parent / "models" / "best" / "LBD-yolo26n-2560.pt"),
                    help="首次开训的起点权重（默认上一版 best）")
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-restarts", type=int, default=20)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        print_status(args.name)
        return 0

    data = Path(args.data)
    if not data.exists():
        print(f"[错误] 找不到 {data}，先用 prepare_yolo.py 生成 dataset_v2", file=sys.stderr)
        return 2

    log("=" * 60)
    log(f"启动长跑：name={args.name} imgsz={args.imgsz} epochs={args.epochs} "
        f"batch={args.batch} patience={args.patience}")
    log(f"数据 {data}")
    if STOP_FILE.exists():          # 上次是手动暂停的，这次重新拉起来就把标志清掉
        STOP_FILE.unlink()
        log("已清除上次的暂停标志 runs\\STOP")
    log(f"体检 {health_snapshot()}")

    fails = 0
    while True:
        st = run_state(args.name)
        resume = st["last"].exists()
        if resume:
            log(f"从 last.pt 续训（已完成 {st['epochs_done']} 轮）")
            cmd = [str(PY), "train_yolo26.py", "--resume", "--name", args.name]
        else:
            if not Path(args.start).exists():
                log(f"[错误] 起点权重不存在 {args.start}")
                return 2
            log(f"从头开训，起点 {args.start}")
            cmd = [
                str(PY), "train_yolo26.py",
                "--model", args.start,
                "--data", str(data),
                "--imgsz", str(args.imgsz),
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--workers", str(args.workers),
                "--cache", "disk",
                "--patience", str(args.patience),
                "--name", args.name,
                "rect=True", "amp=False", "cos_lr=True", "seed=0",
                "lr0=0.0005", "save_period=5",
            ]

        t0 = time.time()
        reason: list[str] = []
        with (HERE / "runs" / f"{args.name}_train.log").open("a", encoding="utf-8") as out, \
             (HERE / "runs" / f"{args.name}_train.err.log").open("a", encoding="utf-8") as err:
            proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=out, stderr=err)
            log(f"训练进程 PID={proc.pid}")
            # 体检 + 响应暂停；要停就把训练收掉。watch 是判断"卡死"用的两个文件。
            supervise(proc, reason, [
                HERE / "runs" / f"{args.name}_train.log",
                HERE / "runs" / f"{args.name}_train.err.log",
                RUNS / args.name / "results.csv",
            ])
            rc = proc.poll()
            if rc is None:
                rc = proc.wait()
        dt = (time.time() - t0) / 3600

        st = run_state(args.name)
        if reason:
            log(f"[暂停] {reason[0]}；这轮已完成 {st['epochs_done']} 轮，"
                f"目前最好 fitness={st['best_fitness']:.4f}")
            log(f"[暂停] 进度没丢，重跑本脚本会从 last.pt 接着往下训 -> {st['last']}")
            return 0

        if rc == 0:
            log(f"训练正常结束，用时 {dt:.1f} 小时，共 {st['epochs_done']} 轮，"
                f"最好 fitness={st['best_fitness']:.4f}")
            log(f"best.pt -> {st['best']}")
            return 0

        fails += 1
        log(f"[异常] 退出码 {rc}，跑了 {dt:.1f} 小时 / {st['epochs_done']} 轮，"
            f"第 {fails} 次失败；看 runs/{args.name}_train.err.log")
        if fails > args.max_restarts:
            log(f"[放弃] 连续失败 {fails} 次，超过上限 {args.max_restarts}，停手等人工处理")
            return 3
        log("60 秒后自动续训")
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
