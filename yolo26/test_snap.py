"""验证边缘吸附：同一个模型、同一批图，吸附前 vs 吸附后的框贴合度。

用工具真正会用的 Python 3.12 跑（它同时有 PySide6 + ultralytics）：
    python test_snap.py --weights <best.pt> --imgsz 1920 --conf 0.25 --max 40
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(preds, gts, thr=0.5):
    """preds/gts: [(类, (x1,y1,x2,y2)), ...]，返回 [(预测下标, 标注下标, 当时的IoU)]。"""
    used = [False] * len(gts)
    out = []
    for pi, (pl, pb) in enumerate(preds):
        bi, bv = -1, thr
        for gi, (gl, gb) in enumerate(gts):
            if used[gi] or gl != pl:
                continue
            v = iou(pb, gb)
            if v >= bv:
                bi, bv = gi, v
        if bi >= 0:
            used[bi] = True
            out.append((pi, bi, bv))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--modes", default="peak,step,grad")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    import lbd_annotator as ann          # 用工具里那份真实的吸附实现
    from ultralytics import YOLO

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    img_dir = HERE / "dataset" / "images" / "val"
    lab_dir = HERE / "dataset" / "labels" / "val"
    model = YOLO(args.weights)

    imgs = []
    for ip in sorted(img_dir.glob("*.png")):
        with Image.open(ip) as im:
            if im.size[0] > 2000:        # 只看高清那批（等于工具里 9000 渲染的缩版）
                imgs.append(ip)
    imgs = imgs[:args.max]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rec = {m: defaultdict(lambda: {"before": [], "after": [], "moved": 0, "worst": 0})
           for m in modes}
    for ip in imgs:
        gray = np.asarray(Image.open(ip).convert("L"))
        H, W = gray.shape
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
        preds = [(names[int(c)], (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                 for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        pairs = greedy_match(preds, gts)
        flat = [(lab, b[0], b[1], b[2], b[3]) for lab, b in preds]
        snaps = {m: ann.snap_boxes_to_ink(gray, flat, mode=m) for m in modes}
        for pi, gi, v0 in pairs:
            lab = preds[pi][0]
            for m in modes:
                v1 = iou(snaps[m][pi][1:5], gts[gi][1])
                rec[m][lab]["before"].append(v0)
                rec[m][lab]["after"].append(v1)
                moved = max(abs(snaps[m][pi][1 + k] - flat[pi][1 + k]) for k in range(4))
                if moved > 0.05:
                    rec[m][lab]["moved"] += 1
                if v1 < v0 - 0.02:
                    rec[m][lab]["worst"] += 1

    print(f"[权重] {Path(args.weights).name}  imgsz={args.imgsz}  conf>={args.conf}  图片 {len(imgs)} 张")
    for m in modes:
        print(f"\n=== 方式 {m} ===")
        print(f"{'类别':<9}{'匹配数':>7}{'平均IoU前':>11}{'平均IoU后':>11}{'中位IoU前':>11}"
              f"{'中位IoU后':>11}{'≥0.9前':>9}{'≥0.9后':>9}{'被动过':>8}{'变差':>7}")
        for lab in names:
            d = rec[m][lab]
            if not d["before"]:
                continue
            n = len(d["before"])
            mb = statistics.mean(d["before"]); ma = statistics.mean(d["after"])
            cb = statistics.median(d["before"]); ca = statistics.median(d["after"])
            p9b = sum(1 for v in d["before"] if v >= 0.9) / n * 100
            p9a = sum(1 for v in d["after"] if v >= 0.9) / n * 100
            print(f"{lab:<9}{n:>7}{mb:>11.4f}{ma:>11.4f}{cb:>11.4f}{ca:>11.4f}"
                  f"{p9b:>8.1f}%{p9a:>8.1f}%{d['moved']:>8}{d['worst']:>7}")
        allb = [v for lab in names for v in rec[m][lab]["before"]]
        alla = [v for lab in names for v in rec[m][lab]["after"]]
        if allb:
            print(f"合计：平均 IoU {statistics.mean(allb):.4f} -> {statistics.mean(alla):.4f}"
                  f"（{statistics.mean(alla) - statistics.mean(allb):+.4f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
