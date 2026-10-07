"""诊断：新模型在旧图纸上多了哪些"误检"，看看到底是框错位还是标了没标签的东西。

输出：每张图一张对比图（绿=标注框，红=模型框），并统计 匹配/漏检/多检 数量。

用法:
    python diag_fp.py --weights <best.pt> --n 2 --out yolo26/preview
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

NEW_SRC = ("steelriver", "bigway", "highland")


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
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(HERE / "preview"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    D = HERE / "dataset_v4"
    imgs = [p for p in sorted((D / "images" / "val").glob("*.png"))
            if p.stem.split("_")[0] not in NEW_SRC]
    picked = imgs[:: max(1, len(imgs) // args.n)][: args.n]
    model = YOLO(args.weights)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in picked:
        with Image.open(p) as im:
            W, H = im.size
        lab = D / "labels" / "val" / (p.stem + ".txt")
        gt = []
        for ln in lab.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = ln.split()
            cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
            gt.append((int(c), (cx - bw / 2) * W, (cy - bh / 2) * H,
                       (cx + bw / 2) * W, (cy + bh / 2) * H))
        res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        preds = [(int(c), *map(float, b)) for c, b in
                 zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist())]

        # 统计：Tracker(1) 的匹配/漏检/多检
        gt1 = [g for g in gt if g[0] == 1]
        pr1 = [q for q in preds if q[0] == 1]
        matched_gt = sum(1 for g in gt1 if any(iou(g[1:], q[1:]) >= 0.5 for q in pr1))
        matched_pr = sum(1 for q in pr1 if any(iou(g[1:], q[1:]) >= 0.5 for g in gt1))
        fp = len(pr1) - matched_pr
        fn = len(gt1) - matched_gt
        print(f"[{p.stem[:44]}] 标注Tracker {len(gt1)}，模型Tracker {len(pr1)}，"
              f"匹配 {matched_pr}，漏检 {fn}，多检(误报) {fp}")

        k = 1600 / W
        small = Image.open(p).convert("RGB").resize((1600, round(H * k)), Image.LANCZOS)
        dr = ImageDraw.Draw(small)
        for c, x1, y1, x2, y2 in gt:            # 标注：绿
            dr.rectangle([x1 * k, y1 * k, x2 * k, y2 * k], outline=(0, 170, 0), width=2)
        for c, x1, y1, x2, y2 in preds:         # 预测：红
            dr.rectangle([x1 * k, y1 * k, x2 * k, y2 * k], outline=(255, 0, 0), width=1)
        out = outdir / f"fpcheck_{p.stem[:36]}.png"
        small.save(out)
        print(f"   -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
