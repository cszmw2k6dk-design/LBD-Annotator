"""按"实际用法"评估：在给定置信度阈值下预测，对比标注算 P/R/F1。

跟 val 的 mAP 不一样——mAP 看的是排序质量，这里看的是你在标注工具里
把 conf 拉到某个值时，画出来的框有多少是对的、漏了多少。

用法: python practical_eval.py --imgsz 1920 --threads 6
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def newest_best() -> Path | None:
    runs = HERE / "runs" / "detect"
    cands = [d / "weights" / "best.pt" for d in runs.iterdir()] if runs.is_dir() else []
    cands = [p for p in cands if p.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--data", default=str(HERE / "dataset"))
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true", help="推理时开测试时增强（更慢）")
    ap.add_argument("--max", type=int, default=0, help="只评测前 N 张（0=全部）")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from PIL import Image
    from ultralytics import YOLO

    w = Path(args.weights) if args.weights else newest_best()
    print(f"[权重] {w}")
    names = (Path(args.data) / "classes.txt").read_text(encoding="utf-8").split()
    img_dir = Path(args.data) / "images" / args.split
    lab_dir = Path(args.data) / "labels" / args.split

    model = YOLO(str(w))
    stats: dict[float, dict[str, list[int]]] = {
        t: defaultdict(lambda: [0, 0, 0]) for t in THRESHOLDS}  # cls -> [TP, FP, FN]

    imgs = sorted(img_dir.glob("*.png"))
    if args.max:
        imgs = imgs[:args.max]
    for k, ip in enumerate(imgs, 1):
        with Image.open(ip) as im:
            W, H = im.size
        gts = []
        for line in (lab_dir / (ip.stem + ".txt")).read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            gts.append((c, ((cx - bw / 2) * W, (cy - bh / 2) * H,
                            (cx + bw / 2) * W, (cy + bh / 2) * H)))
        r = model.predict(source=str(ip), imgsz=args.imgsz, conf=0.05, iou=0.7,
                          max_det=300, verbose=False, augment=args.tta)[0]
        preds = [(int(c), list(map(float, b)), float(cf))
                 for c, b, cf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                     r.boxes.conf.tolist())] if r.boxes is not None else []

        for t in THRESHOLDS:
            used = [False] * len(gts)
            for c, b, cf in sorted(preds, key=lambda x: -x[2]):
                if cf < t:
                    continue
                best_i, best_iou = -1, args.iou_match
                for gi, (gc, gb) in enumerate(gts):
                    if used[gi] or gc != c:
                        continue
                    v = iou(b, gb)
                    if v >= best_iou:
                        best_i, best_iou = gi, v
                key = names[c] if c < len(names) else str(c)
                if best_i >= 0:
                    used[best_i] = True
                    stats[t][key][0] += 1
                    stats[t]["__all__"][0] += 1
                else:
                    stats[t][key][1] += 1
                    stats[t]["__all__"][1] += 1
            for gi, (gc, _gb) in enumerate(gts):
                if not used[gi]:
                    key = names[gc] if gc < len(names) else str(gc)
                    stats[t][key][2] += 1
                    stats[t]["__all__"][2] += 1
        if k % 40 == 0:
            print(f"  预测 {k}/{len(imgs)}")

    for t in THRESHOLDS:
        print()
        print(f"=== conf >= {t} ===")
        print(f"  {'类别':<10}{'TP':>7}{'FP':>7}{'FN':>7}{'P':>9}{'R':>9}{'F1':>9}")
        for key in ["__all__", *names]:
            tp, fp, fn = stats[t].get(key, [0, 0, 0])
            p = tp / (tp + fp) if tp + fp else 0.0
            rr = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * rr / (p + rr) if p + rr else 0.0
            label = "全部" if key == "__all__" else key
            print(f"  {label:<10}{tp:>7}{fp:>7}{fn:>7}{p:>9.3f}{rr:>9.3f}{f1:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
