"""YOLO26 训练入口（CPU / 16 线程）。

把所有缓存目录都指到本目录下，避免往 AppData 写。

例子:
    python train_yolo26.py --model yolo26n.pt --imgsz 1280 --epochs 100
    python train_yolo26.py --resume                 # 从 runs/<name>/weights/last.pt 续训
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402


def keep_awake() -> None:
    """训练期间别让系统睡眠（进程结束自动失效，不改系统设置）。"""
    try:
        import ctypes

        es_continuous = 0x80000000
        es_system_required = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"])
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--subset", type=float, default=1.0, help="用多少比例的数据（调试用）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("extra", nargs="*", help="额外的 key=value，直接透传给 ultralytics")
    args = ap.parse_args()

    threads = args.threads or (os.cpu_count() or 8)
    torch.set_num_threads(threads)
    keep_awake()
    os.chdir(HERE)

    from ultralytics import YOLO

    overrides = {}
    for item in args.extra:
        if "=" not in item:
            print(f"[错误] 额外参数必须写成 key=value，收到: {item!r}", file=sys.stderr)
            return 2
        k, _, v = item.partition("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        overrides[k] = v

    if args.resume:
        name = args.name or "train"
        last = HERE / "runs" / "detect" / name / "weights" / "last.pt"
        if not last.exists():
            print(f"[错误] 找不到 {last}，无法续训", file=sys.stderr)
            return 2
        print(f"[续训] {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        return 0

    # ultralytics 的 optimizer=auto 会自己算 lr0 并把传进来的 lr0/lrf/momentum 丢掉
    # （日志里会写 "ignoring 'lr0=...'"）。要显式控制学习率就得把优化器也定死。
    if any(k in overrides for k in ("lr0", "lrf", "momentum")) and "optimizer" not in overrides:
        overrides["optimizer"] = "AdamW"
        print("[提示] 检测到 lr0/lrf/momentum，已把 optimizer 固定为 AdamW（否则会被 auto 忽略）")

    name = args.name or f"{Path(args.model).stem}_imgsz{args.imgsz}"
    print(f"[开始] model={args.model} imgsz={args.imgsz} epochs={args.epochs} "
          f"batch={args.batch} workers={args.workers} threads={threads} device={args.device}")
    print(f"[数据] {args.data}")
    print(f"[输出] {HERE / 'runs' / 'detect' / name}")
    t0 = time.time()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(HERE / "runs" / "detect"),
        name=name,
        exist_ok=True,
        patience=args.patience,
        cache=(False if args.cache == "none" else args.cache),
        fraction=args.subset,
        plots=True,
        val=True,
        save=True,
        verbose=True,
        **overrides,
    )
    dt = time.time() - t0
    print(f"[结束] 总用时 {dt / 60:.1f} 分钟")
    best = HERE / "runs" / "detect" / name / "weights" / "best.pt"
    print(f"[模型] {best}  存在={best.exists()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
