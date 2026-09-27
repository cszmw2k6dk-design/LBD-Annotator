"""看看误检（FP）和正确框（TP）在形状上有多少区别，评估"按形状筛掉多余框"是否可行。

用法: python diag_shape.py --weights <best.pt> [--imgsz 1920] [--conf 0.25]
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


def pct(vals, q):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[min(len(v) - 1, int(q * len(v)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from ultralytics import YOLO

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    img_dir = HERE / "dataset" / "images" / "val"
    lab_dir = HERE / "dataset" / "labels" / "val"
    model = YOLO(args.weights)

    # 形状特征（全部按图片尺寸归一化）：宽、高、宽高比
    feat = {"gt": defaultdict(lambda: defaultdict(list)),
            "tp": defaultdict(lambda: defaultdict(list)),
            "fp": defaultdict(lambda: defaultdict(list))}

    for ip in sorted(img_dir.glob("*.png")):
        with Image.open(ip) as im:
            W, H = im.size
        gts = []
        for line in (lab_dir / (ip.stem + ".txt")).read_text(encoding="utf-8").splitlines():
            c = line.split()
            if len(c) != 5:
                continue
            k, cx, cy, w, h = int(c[0]), *map(float, c[1:])
            box = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            gts.append((names[k], box))
            feat["gt"][names[k]]["w"].append(w)
            feat["gt"][names[k]]["h"].append(h)
            feat["gt"][names[k]]["ar"].append((w * W) / max(1e-6, h * H))

        r = model.predict(source=str(ip), imgsz=args.imgsz, conf=0.05, iou=0.7,
                          max_det=300, verbose=False)[0]
        preds = [(names[int(c)], list(map(float, b)), float(cf))
                 for c, b, cf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                     r.boxes.conf.tolist())]
        used = [False] * len(gts)
        for lab, box, cf in sorted(preds, key=lambda x: -x[2]):
            if cf < args.conf:
                continue
            bi, bv = -1, 0.5
            for gi, (gl, gb) in enumerate(gts):
                if used[gi] or gl != lab:
                    continue
                v = iou(box, gb)
                if v >= bv:
                    bi, bv = gi, v
            key = "tp" if bi >= 0 else "fp"
            if bi >= 0:
                used[bi] = True
            w, h = (box[2] - box[0]) / W, (box[3] - box[1]) / H
            feat[key][lab]["w"].append(w)
            feat[key][lab]["h"].append(h)
            feat[key][lab]["ar"].append((box[2] - box[0]) / max(1e-6, box[3] - box[1]))

    print(f"[权重] {args.weights}   imgsz={args.imgsz}   conf>={args.conf}")
    for lab in names:
        print()
        print(f"=== {lab} ===")
        print(f"  {'':<6}{'数量':>7}{'宽(归一) p5/p50/p95':>28}{'高(归一) p5/p50/p95':>26}{'宽高比 p5/p50/p95':>26}")
        for key, title in (("gt", "标注"), ("tp", "正确框"), ("fp", "误检")):
            f = feat[key][lab]
            if not f["w"]:
                print(f"  {title:<6}{0:>7}")
                continue
            fw = f"{pct(f['w'],0.05):.3f}/{pct(f['w'],0.5):.3f}/{pct(f['w'],0.95):.3f}"
            fh = f"{pct(f['h'],0.05):.3f}/{pct(f['h'],0.5):.3f}/{pct(f['h'],0.95):.3f}"
            fa = f"{pct(f['ar'],0.05):.3f}/{pct(f['ar'],0.5):.3f}/{pct(f['ar'],0.95):.3f}"
            print(f"  {title:<6}{len(f['w']):>7}{fw:>28}{fh:>26}{fa:>26}")

    # 试几个简单的形状过滤规则，看能滤掉多少误检 / 误伤多少真框
    print()
    print("=== 试算：按标注的 p1~p99 区间做形状过滤（严格）===")
    for lab in names:
        g = feat["gt"][lab]
        if not g["w"]:
            continue
        lo_w, hi_w = pct(g["w"], 0.01), pct(g["w"], 0.99)
        lo_h, hi_h = pct(g["h"], 0.01), pct(g["h"], 0.99)
        def keep(f):
            ok = sum(1 for i in range(len(f["w"]))
                     if lo_w <= f["w"][i] <= hi_w and lo_h <= f["h"][i] <= hi_h)
            return ok, len(f["w"]) - ok
        tp_keep, tp_drop = keep(feat["tp"][lab])
        fp_keep, fp_drop = keep(feat["fp"][lab])
        print(f"  {lab}: 误检滤掉 {fp_drop}/{fp_drop+fp_keep}，"
              f"代价是正确框也丢 {tp_drop}/{tp_drop+tp_keep}"
              f"（区间 宽 {lo_w:.4f}~{hi_w:.4f}，高 {lo_h:.4f}~{hi_h:.4f}）")

    print()
    print("=== 试算：放宽规则（宽 ≤ 2×p99、高 ≥ 0.5×p5）===")
    for lab in names:
        g = feat["gt"][lab]
        if not g["w"]:
            continue
        hi_w = pct(g["w"], 0.99) * 2
        lo_h = pct(g["h"], 0.05) * 0.5
        hi_h = pct(g["h"], 0.99) * 2
        def keep(f):
            ok = sum(1 for i in range(len(f["w"]))
                     if f["w"][i] <= hi_w and lo_h <= f["h"][i] <= hi_h)
            return ok, len(f["w"]) - ok
        tp_keep, tp_drop = keep(feat["tp"][lab])
        fp_keep, fp_drop = keep(feat["fp"][lab])
        print(f"  {lab}: 误检滤掉 {fp_drop}/{fp_drop+fp_keep}"
              f"，正确框丢 {tp_drop}/{tp_drop+tp_keep}"
              f"（宽 ≤ {hi_w:.4f}，高 {lo_h:.4f}~{hi_h:.4f}）")

    print()
    print("=== 试算：只看宽高比（Tracker 应当又细又高）===")
    for lab in names:
        for lim in (0.05, 0.1, 0.2, 0.5):
            def keep(f, lim=lim):
                ok = sum(1 for x in f["ar"] if x <= lim)
                return ok, len(f["ar"]) - ok
            tp_keep, tp_drop = keep(feat["tp"][lab])
            fp_keep, fp_drop = keep(feat["fp"][lab])
            print(f"  {lab} 宽高比≤{lim}: 误检滤掉 {fp_drop}/{fp_drop+fp_keep}"
                  f"，正确框丢 {tp_drop}/{tp_drop+tp_keep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
