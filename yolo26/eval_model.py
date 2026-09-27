"""对指定权重跑一次验证，打印分类别精度；可选再导出几张预测图。

用法:
    python eval_model.py                                  # 用最新的 best.pt
    python eval_model.py --weights runs/detect/xxx/weights/best.pt --imgsz 1920
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402


def newest_best() -> Path | None:
    runs = HERE / "runs" / "detect"
    if not runs.is_dir():
        return None
    cands = [d / "weights" / "best.pt" for d in runs.iterdir()]
    cands = [p for p in cands if p.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6, help="别把训练抢干，默认只用 6 线程")
    ap.add_argument("--split", default="val")
    ap.add_argument("--name", default="eval")
    ap.add_argument("--predict-n", type=int, default=2, help="额外导出几张带框的预测图")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from ultralytics import YOLO

    w = Path(args.weights) if args.weights else newest_best()
    if not w or not Path(w).exists():
        print("[错误] 找不到权重", w, file=sys.stderr)
        return 2
    print(f"[权重] {w}")
    print(f"[设置] imgsz={args.imgsz} batch={args.batch} threads={args.threads} split={args.split}")

    model = YOLO(str(w))
    res = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, split=args.split,
                    project=str(HERE / "runs" / "detect"), name=args.name, exist_ok=True,
                    plots=True, verbose=True)

    box = res.box
    names = res.names
    print()
    print("=" * 62)
    print(f"{'类别':<12}{'实例数':>8}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
    print("-" * 62)
    n_gt = sum(int(x) for x in getattr(box, "nt_per_class", [])) or ""
    print(f"{'全部':<12}{n_gt:>8}{box.mp:>10.3f}{box.mr:>10.3f}{box.map50:>10.3f}{box.map:>12.3f}")
    for i, cname in names.items():
        p, r, ap50, ap = box.class_result(i)
        cnt = int(box.nt_per_class[i]) if getattr(box, "nt_per_class", None) is not None else ""
        print(f"{cname:<12}{cnt:>8}{p:>10.3f}{r:>10.3f}{ap50:>10.3f}{ap:>12.3f}")
    print("=" * 62)

    if args.predict_n > 0:
        imgs = sorted((HERE / "dataset" / "images" / args.split).glob("*.png"))[:args.predict_n]
        if imgs:
            model.predict(source=[str(p) for p in imgs], imgsz=args.imgsz, conf=args.conf,
                          project=str(HERE / "runs" / "detect"), name=f"{args.name}_pred",
                          exist_ok=True, save=True, verbose=False)
            print(f"[预测图] {HERE / 'runs' / 'detect' / (args.name + '_pred')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
