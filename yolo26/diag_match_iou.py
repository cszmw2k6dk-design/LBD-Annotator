"""量一下：模型框和标注框指向同一个物体时，重叠度到底是多少。

这决定"重叠 ≥90% 才算重复"这个门槛是否可行：
如果模型框和标注框通常只有 0.7~0.9 的重叠，那 90% 的门槛就抓不到它们。

用法: python diag_match_iou.py --weights <best.pt> [--imgsz 1920] [--conf 0.25]
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
from PIL import Image  # noqa: E402


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from ultralytics import YOLO

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    img_dir = HERE / "dataset" / "images" / "val"
    lab_dir = HERE / "dataset" / "labels" / "val"
    imgs = sorted(img_dir.glob("*.png"))
    if args.max:
        imgs = imgs[:args.max]
    model = YOLO(args.weights)

    buckets = defaultdict(int)
    per_cls = defaultdict(lambda: defaultdict(int))
    dup_pred = 0          # 模型自己互相重叠 >=0.9 的对
    dup_same_cls = 0

    for ip in imgs:
        with Image.open(ip) as im:
            W, H = im.size
        gts = []
        for line in (lab_dir / (ip.stem + ".txt")).read_text(encoding="utf-8").splitlines():
            c = line.split()
            if len(c) != 5:
                continue
            k, cx, cy, w, h = int(c[0]), *map(float, c[1:])
            gts.append((names[k], ((cx - w / 2) * W, (cy - h / 2) * H,
                                   (cx + w / 2) * W, (cy + h / 2) * H)))
        r = model.predict(source=str(ip), imgsz=args.imgsz, conf=args.conf, iou=0.7,
                          max_det=300, verbose=False)[0]
        preds = [(names[int(c)], list(map(float, b)))
                 for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]

        # 每个预测框：和同类标注的最大重叠
        for lab, pb in preds:
            best = 0.0
            for gl, gb in gts:
                if gl == lab:
                    best = max(best, iou(pb, gb))
            b = ("0.0-0.3" if best < 0.3 else "0.3-0.5" if best < 0.5 else
                 "0.5-0.7" if best < 0.7 else "0.7-0.8" if best < 0.8 else
                 "0.8-0.9" if best < 0.9 else "0.9-1.0")
            buckets[b] += 1
            per_cls[lab][b] += 1

        # 模型自己内部有没有互相重叠 >=0.9 的同类框
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if preds[i][0] == preds[j][0] and iou(preds[i][1], preds[j][1]) >= 0.9:
                    dup_pred += 1
        for i in range(len(gts)):
            for j in range(i + 1, len(gts)):
                if gts[i][0] == gts[j][0] and iou(gts[i][1], gts[j][1]) >= 0.9:
                    dup_same_cls += 1

    order = ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    total = sum(buckets.values())
    print(f"[权重] {Path(args.weights).name}   imgsz={args.imgsz}   conf>={args.conf}   图片 {len(imgs)} 张")
    print(f"\n每个预测框与同类标注的最大重叠（共 {total} 个预测框）：")
    for b in order:
        n = buckets.get(b, 0)
        bar = "#" * int(60 * n / max(1, total))
        print(f"  {b:>9}  {n:>7}  {n/total*100:5.1f}%  {bar}")
    print("\n分类别（只看 >=0.7 的部分）：")
    for lab in names:
        tot = sum(per_cls[lab].values())
        hi = sum(per_cls[lab][b] for b in ("0.7-0.8", "0.8-0.9", "0.9-1.0"))
        v90 = per_cls[lab].get("0.9-1.0", 0)
        print(f"  {lab:<9} 预测 {tot:>6} 个，其中与标注重叠 >=0.7 的有 {hi}（{hi/max(1,tot)*100:.1f}%），"
              f">=0.9 的只有 {v90}（{v90/max(1,tot)*100:.1f}%）")
    print(f"\n模型一次输出里，同类互相重叠 >=0.9 的框对：{dup_pred}")
    print(f"标注里同类互相重叠 >=0.9 的框对：{dup_same_cls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
