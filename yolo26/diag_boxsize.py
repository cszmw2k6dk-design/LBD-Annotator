"""量一下"框有时大有时小"到底是偏置还是抖动，顺便看标注本身一不一致。

用法: python diag_boxsize.py --weights <best.pt> [--imgsz 1920] [--conf 0.25]
"""
from __future__ import annotations

import argparse
import os
import statistics
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


def stats(vals):
    if not vals:
        return (0.0,) * 4
    v = sorted(vals)
    return (statistics.mean(v), statistics.median(v),
            v[int(0.05 * len(v))], v[min(len(v) - 1, int(0.95 * len(v)))])


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

    # 相对误差：(预测 - 标注) / 标注
    err = defaultdict(lambda: defaultdict(list))
    gt_widths_by_img = defaultdict(lambda: defaultdict(list))
    n_match = defaultdict(int)

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
            gt_widths_by_img[ip.stem][names[k]].append(box[2] - box[0])

        r = model.predict(source=str(ip), imgsz=args.imgsz, conf=args.conf, iou=0.7,
                          max_det=300, verbose=False)[0]
        preds = [(names[int(c)], list(map(float, b)))
                 for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        used = [False] * len(gts)
        for lab, pb in preds:
            bi, bv = -1, 0.5
            for gi, (gl, gb) in enumerate(gts):
                if used[gi] or gl != lab:
                    continue
                v = iou(pb, gb)
                if v >= bv:
                    bi, bv = gi, v
            if bi < 0:
                continue
            used[bi] = True
            n_match[lab] += 1
            _gl, gb = gts[bi]
            gw, gh = gb[2] - gb[0], gb[3] - gb[1]
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            if gw > 0 and gh > 0:
                err[lab]["dw"].append(pw / gw - 1)
                err[lab]["dh"].append(ph / gh - 1)
                err[lab]["darea"].append((pw * ph) / (gw * gh) - 1)
                err[lab]["iou"].append(bv)

    print(f"[权重] {Path(args.weights).name}  imgsz={args.imgsz}  conf>={args.conf}")
    for lab in names:
        d = err[lab]
        if not d["dw"]:
            continue
        print(f"\n=== {lab}（匹配上 {n_match[lab]} 个）===")
        print(f"  {'指标':<10}{'均值':>9}{'中位':>9}{'5%':>9}{'95%':>9}")
        for key, title in (("dw", "宽相对误差"), ("dh", "高相对误差"),
                           ("darea", "面积相对误差"), ("iou", "匹配时IoU")):
            m, md, lo, hi = stats(d[key])
            print(f"  {title:<10}{m:>9.3f}{md:>9.3f}{lo:>9.3f}{hi:>9.3f}")
        big = sum(1 for x in d["darea"] if abs(x) > 0.2)
        print(f"  面积偏差超过 ±20% 的占比：{big/len(d['darea'])*100:.1f}%")

    print("\n=== 标注自身一致性（同一张图里同类别框宽的离散度）===")
    for lab in names:
        cvs = []
        for stem, byc in gt_widths_by_img.items():
            ws = byc.get(lab, [])
            if len(ws) >= 5:
                m = statistics.mean(ws)
                if m > 0:
                    cvs.append(statistics.pstdev(ws) / m)
        if cvs:
            print(f"  {lab}: {len(cvs)} 张图可统计，图内宽度变异系数 p50={statistics.median(cvs):.3f}  "
                  f"p90={sorted(cvs)[int(0.9*len(cvs))]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
