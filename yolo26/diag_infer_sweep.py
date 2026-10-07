"""推理参数扫描：同一版权重，换 NMS IoU / agnostic_nms / TTA，量出精确率-召回率曲线。

用途：找"不重训就能提高精确率"的最佳推理设置。
  1) 一次推理（低 conf 全留下），之后只做阈值过滤，所以一档 conf 不用重跑；
  2) 逐类（Node / Tracker）给 P / R / F1 + 重复框数（预测数 - 匹配数）。

用法:
  python diag_infer_sweep.py --weights runs/detect/yolo26n_2560_v4c/weights/best.pt \
         --tag base --iou 0.7 --max-pages 120
  python diag_infer_sweep.py ... --tag iou50 --iou 0.5
  python diag_infer_sweep.py ... --tag iou50_agn --iou 0.5 --agnostic-nms
  python diag_infer_sweep.py ... --tag tta --iou 0.5 --augment
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(gts, preds, thr=0.5):
    """一对一贪心匹配（同类）。返回 tp, fp, fn。"""
    used = [False] * len(preds)
    tp = 0
    for gc, g in gts:
        bj, bv = -1, thr
        for j, (pc, p) in enumerate(preds):
            if used[j] or pc != gc:
                continue
            v = iou(g, p)
            if v >= bv:
                bv, bj = v, j
        if bj >= 0:
            used[bj] = True
            tp += 1
    return tp, sum(1 for u in used if not u), len(gts) - tp


def read_gt(label_path: Path, W: int, H: int):
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        c = ln.split()
        if len(c) != 5:
            continue
        k, cx, cy, bw, bh = int(c[0]), *map(float, c[1:])
        out.append((k, ((cx - bw / 2) * W, (cy - bh / 2) * H,
                        (cx + bw / 2) * W, (cy + bh / 2) * H)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=str(HERE / "dataset_v4"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--agnostic-nms", action="store_true")
    ap.add_argument("--augment", action="store_true", help="TTA")
    ap.add_argument("--max-det", type=int, default=300,
                    help="每页最多保留多少个框（ultralytics 默认 300；框多的图纸会被截断）")
    ap.add_argument("--sweep", default="0.25,0.35,0.45,0.55")
    ap.add_argument("--exclude", default="Driver|Sabal Palm",
                    help="按图纸名排除（正则，默认排掉口径不同的那两套）")
    ap.add_argument("--include", default="", help="只留匹配的图纸名（正则，可选）")
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--out", default=str(HERE / "runs" / "diag"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from PIL import Image
    from ultralytics import YOLO

    D = Path(args.data)
    imgdir = D / "images" / args.split
    labdir = D / "labels" / args.split
    skip = re.compile(args.exclude, re.I)
    keep = re.compile(args.include, re.I) if args.include else None
    pages = [p for p in sorted(imgdir.glob("*.png"))
             if not skip.search(re.sub(r"_p\d+$", "", p.stem))
             and (keep is None or keep.search(re.sub(r"_p\d+$", "", p.stem)))]
    if args.max_pages and len(pages) > args.max_pages:
        step = len(pages) / float(args.max_pages)
        pages = [pages[int(i * step)] for i in range(args.max_pages)]
    print(f"[{args.tag}] 页数 {len(pages)}  iou={args.iou} agnostic={args.agnostic_nms} "
          f"TTA={args.augment} conf>={args.conf_min}")

    model = YOLO(args.weights)
    cache = []
    for i in range(0, len(pages), args.batch):
        chunk = pages[i:i + args.batch]
        res = model.predict([str(p) for p in chunk], imgsz=args.imgsz,
                            conf=args.conf_min, iou=args.iou,
                            agnostic_nms=args.agnostic_nms, augment=args.augment,
                            max_det=args.max_det,
                            verbose=False, device="cpu")
        for p, r in zip(chunk, res):
            with Image.open(p) as im:
                W, H = im.size
            gt = read_gt(labdir / (p.stem + ".txt"), W, H)
            pred = [(int(c), tuple(map(float, b)), float(s))
                    for c, b, s in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                       r.boxes.conf.tolist())]
            cache.append((p.stem.split("_p")[0], gt, pred))
        print(f"   推理 {min(i + args.batch, len(pages))}/{len(pages)}", flush=True)

    confs = [float(v) for v in args.sweep.split(",")]
    names = {0: "Node", 1: "Tracker"}
    rows = []
    print(f"\n=== {args.tag}：逐类 P/R/F1（一对一匹配，IoU>=0.5）===")
    print(f"{'类别':<9}{'conf':>6}{'标注':>7}{'预测':>7}{'TP':>7}{'FP':>7}{'FN':>7}"
          f"{'重复框':>8}{'P':>8}{'R':>8}{'F1':>8}")
    for cls in (1, 0):                       # 先 Tracker 再 Node
        for thr in confs:
            tp = fp = fn = ngt = npr = 0
            for _draw, gt, pred in cache:
                g = [x for x in gt if x[0] == cls]
                pr = [(c, b) for c, b, s in pred if c == cls and s >= thr]
                a, b, c_ = match(g, pr)
                tp += a
                fp += b
                fn += c_
                ngt += len(g)
                npr += len(pr)
            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn) if tp + fn else 0.0
            F1 = 2 * P * R / (P + R) if P + R else 0.0
            dup = npr - tp
            print(f"{names[cls]:<9}{thr:>6.2f}{ngt:>7}{npr:>7}{tp:>7}{fp:>7}{fn:>7}"
                  f"{dup:>8}{P:>8.3f}{R:>8.3f}{F1:>8.3f}")
            rows.append({"tag": args.tag, "iou": args.iou, "agnostic": args.agnostic_nms,
                         "tta": args.augment, "class": names[cls], "conf": thr,
                         "gt": ngt, "pred": npr, "tp": tp, "fp": fp, "fn": fn,
                         "dup": dup, "P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4)})
    out = Path(args.out) / "infer_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = not out.exists()
    with out.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if header:
            w.writeheader()
        w.writerows(rows)
    print(f"\n[写入] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
