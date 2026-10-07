"""按图纸统计精确率/召回率，用来定位"精确率到底掉在哪些图纸上"。

做法：
  1. 用很低的 conf（默认 0.05）跑一次推理，把所有候选框都留下；
  2. 之后只做阈值过滤，就同时得到每一档 conf 下的 P/R（不用重复推理）；
  3. 按图纸分组统计 Tracker / Node 的 TP / FP / FN（一对一贪心匹配，IoU>=0.5）。

用法:
    python diag_per_drawing.py --weights <best.pt> --tag v4c_ep23 --per-drawing 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
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


def match_one_to_one(gts, preds, thr: float = 0.5):
    """返回 (tp, fp, fn)，一对一贪心匹配（同类、IoU>=thr）。"""
    used = [False] * len(preds)
    tp = 0
    for gc, g in gts:
        best_j, best_v = -1, thr
        for j, (pc, p) in enumerate(preds):
            if used[j] or pc != gc:
                continue
            v = iou(g, p)
            if v >= best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fn = len(gts) - tp
    fp = sum(1 for u in used if not u)
    return tp, fp, fn


def read_gt(label_path: Path, W: int, H: int):
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        c, cx, cy, bw, bh = ln.split()
        cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
        out.append((int(c), ((cx - bw / 2) * W, (cy - bh / 2) * H,
                             (cx + bw / 2) * W, (cy + bh / 2) * H)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--per-drawing", type=int, default=8, help="每套图纸抽多少页")
    ap.add_argument("--conf-min", type=float, default=0.05, help="推理时保留的低阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    ap.add_argument("--sweep", default="0.15,0.25,0.35,0.45,0.55")
    ap.add_argument("--picture-conf", type=float, default=0.25, help="逐图纸表用的阈值")
    ap.add_argument("--include-new", action="store_true", help="连新数据图纸一起评")
    ap.add_argument("--out", default=str(HERE / "runs" / "diag"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from PIL import Image
    from ultralytics import YOLO

    D = HERE / "dataset_v4"
    all_imgs = sorted((D / "images" / "val").glob("*.png"))
    by_drawing = defaultdict(list)
    for p in all_imgs:
        drawing = p.stem.split("_p")[0]
        is_new = p.stem.split("_")[0].replace(" ", "").lower() in NEW_SRC
        if is_new and not args.include_new:
            continue
        by_drawing[drawing].append(p)

    picked = []
    for drawing, pages in sorted(by_drawing.items()):
        step = max(1, len(pages) // args.per_drawing)
        picked += pages[::step][: args.per_drawing]
    print(f"[计划] {len(by_drawing)} 套图纸，抽 {len(picked)} 页做推理（conf>={args.conf_min}）")

    model = YOLO(args.weights)
    # 每页：{drawing: [(cls, box, score), ...]}
    per_page = defaultdict(list)
    for i in range(0, len(picked), args.batch):
        chunk = picked[i:i + args.batch]
        results = model.predict([str(p) for p in chunk], imgsz=args.imgsz,
                                conf=args.conf_min, iou=args.iou, verbose=False,
                                device="cpu")
        for p, res in zip(chunk, results):
            drawing = p.stem.split("_p")[0]
            per_page[drawing].append(
                (p, [(int(c), tuple(map(float, b)), float(s)) for c, b, s in
                     zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist(),
                         res.boxes.conf.tolist())]))
        print(f"   推理 {min(i + args.batch, len(picked))}/{len(picked)}", flush=True)

    thresholds = [float(v) for v in args.sweep.split(",")]
    csv_rows = []
    print("\n=== 逐图纸（Tracker，conf=%.2f）===" % args.picture_conf)
    print(f"{'图纸':<38}{'页':>4}{'标注':>7}{'预测':>7}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>8}{'R':>8}")
    for drawing, pages in sorted(per_page.items()):
        gtN = prN = 0
        tp = fp = fn = 0
        for p, preds_all in pages:
            with Image.open(p) as im:
                W, H = im.size
            gt = read_gt(D / "labels" / "val" / (p.stem + ".txt"), W, H)
            gt1 = [g for g in gt if g[0] == 1]
            pr1 = [(c, b) for c, b, s in preds_all
                   if c == 1 and s >= args.picture_conf]
            a, b, c_ = match_one_to_one(gt1, pr1)
            tp += a
            fp += b
            fn += c_
            gtN += len(gt1)
            prN += len(pr1)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        print(f"{drawing[:36]:<38}{len(pages):>4}{gtN:>7}{prN:>7}{tp:>6}{fp:>6}{fn:>6}"
              f"{P:>8.3f}{R:>8.3f}")
        csv_rows.append((args.tag, drawing, len(pages), gtN, prN, tp, fp, fn,
                         round(P, 4), round(R, 4), args.picture_conf))

    print("\n=== 阈值扫描（全部抽样页合计，Node / Tracker 分开）===")
    print(f"{'类别':<9}{'conf':>6}{'标注':>8}{'预测':>8}{'TP':>7}{'FP':>7}{'FN':>7}"
          f"{'P':>8}{'R':>8}")
    # 只读一次图，避免重复 IO
    cache = []
    for drawing, pages in per_page.items():
        for p, preds_all in pages:
            with Image.open(p) as im:
                W, H = im.size
            cache.append((p, tuple(preds_all), read_gt(
                D / "labels" / "val" / (p.stem + ".txt"), W, H)))
    for cls_id, cls_name in ((0, "Node"), (1, "Tracker")):
        for thr in thresholds:
            tp = fp = fn = gtN = prN = 0
            for p, preds_all, gt in cache:
                gtc = [g for g in gt if g[0] == cls_id]
                prc = [(c, b) for c, b, s in preds_all if c == cls_id and s >= thr]
                a, b, c_ = match_one_to_one(gtc, prc)
                tp += a
                fp += b
                fn += c_
                gtN += len(gtc)
                prN += len(prc)
            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn) if tp + fn else 0.0
            print(f"{cls_name:<9}{thr:>6.2f}{gtN:>8}{prN:>8}{tp:>7}{fp:>7}{fn:>7}"
                  f"{P:>8.3f}{R:>8.3f}")
            csv_rows.append((f"{args.tag}_conf{thr}", f"ALL_{cls_name}", len(picked),
                             gtN, prN, tp, fp, fn, round(P, 4), round(R, 4), thr))

    out = Path(args.out) / f"per_drawing_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tag", "drawing", "pages", "gt", "pred", "tp", "fp", "fn",
                    "P", "R", "conf"])
        w.writerows(csv_rows)
    print(f"\n[写入] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
