"""难例清单：拿当前模型跑一遍训练集，找出"模型和标注分歧最大"的图。

分歧大基本只有两种解释：标注漏了/多标了，或者模型确实学不会这个样本。
人工过一遍 top 清单，数据的质量就能提一档。

用法: python hard_examples.py --weights <best.pt> --imgsz 2560 --conf 0.25
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
PAGE_RE = re.compile(r"_p0*\d+$", re.IGNORECASE)


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
    ap.add_argument("--split", default="train")
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from ultralytics import YOLO

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    img_dir = HERE / "dataset" / "images" / args.split
    lab_dir = HERE / "dataset" / "labels" / args.split
    out_dir = HERE / "runs" / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    per_img = []
    boxes_out = []
    imgs = sorted(img_dir.glob("*.png"))
    for k, ip in enumerate(imgs, 1):
        with Image.open(ip) as im:
            W, H = im.size
        gts = []
        for line in (lab_dir / (ip.stem + ".txt")).read_text(encoding="utf-8").splitlines():
            c = line.split()
            if len(c) != 5:
                continue
            kk, cx, cy, bw, bh = int(c[0]), *map(float, c[1:])
            gts.append((names[kk], ((cx - bw / 2) * W, (cy - bh / 2) * H,
                                    (cx + bw / 2) * W, (cy + bh / 2) * H)))
        r = model.predict(source=str(ip), imgsz=args.imgsz, conf=args.conf, iou=0.7,
                          max_det=300, verbose=False)[0]
        preds = [(names[int(c)], list(map(float, b)), float(cf))
                 for c, b, cf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                     r.boxes.conf.tolist())]
        used_gt = [False] * len(gts)
        used_pr = [False] * len(preds)
        # 贪心匹配（预测按置信度从高到低）
        for pi in sorted(range(len(preds)), key=lambda i: -preds[i][2]):
            pl, pb = preds[pi][0], preds[pi][1]
            if used_pr[pi]:
                continue
            bi, bv = -1, 0.5
            for gi, (gl, gb) in enumerate(gts):
                if used_gt[gi] or gl != pl:
                    continue
                v = iou(pb, gb)
                if v >= bv:
                    bi, bv = gi, v
            if bi >= 0:
                used_gt[bi] = True
                used_pr[pi] = True
        fp = [preds[i] for i in range(len(preds)) if not used_pr[i]]
        fn = [gts[i] for i in range(len(gts)) if not used_gt[i]]
        per_img.append({
            "image": ip.stem, "drawing": PAGE_RE.sub("", ip.stem),
            "gt": len(gts), "pred": len(preds),
            "漏标候选(模型找到但没标注)": len(fp), "多标候选(标注了但模型没找到)": len(fn),
            "分歧数": len(fp) + len(fn),
        })
        for lab, b, cf in fp:
            boxes_out.append({"image": ip.stem, "类型": "模型找到、标注没有",
                              "class": lab, "conf": round(cf, 3),
                              "x1": round(b[0], 1), "y1": round(b[1], 1),
                              "x2": round(b[2], 1), "y2": round(b[3], 1),
                              "宽": round(b[2] - b[0], 1), "高": round(b[3] - b[1], 1)})
        for lab, b in fn:
            boxes_out.append({"image": ip.stem, "类型": "标注了、模型没找到",
                              "class": lab, "conf": "",
                              "x1": round(b[0], 1), "y1": round(b[1], 1),
                              "x2": round(b[2], 1), "y2": round(b[3], 1),
                              "宽": round(b[2] - b[0], 1), "高": round(b[3] - b[1], 1)})
        if k % 50 == 0:
            print(f"    {k}/{len(imgs)}")

    per_img.sort(key=lambda r: -r["分歧数"])
    p1 = out_dir / "hard_examples.csv"
    with p1.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_img[0].keys()))
        w.writeheader(); w.writerows(per_img)
    boxes_out.sort(key=lambda r: (r["image"], r["类型"]))
    p2 = out_dir / "hard_boxes.csv"
    if boxes_out:
        with p2.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(boxes_out[0].keys()))
            w.writeheader(); w.writerows(boxes_out)

    tot_fp = sum(r["漏标候选(模型找到但没标注)"] for r in per_img)
    tot_fn = sum(r["多标候选(标注了但模型没找到)"] for r in per_img)
    tot_gt = sum(r["gt"] for r in per_img)
    print(f"\n[样本] {args.split} 集 {len(imgs)} 张图 / 标注 {tot_gt} 个框")
    print(f"分歧总计 {tot_fp + tot_fn} 个：模型找到但标注没有 {tot_fp}，标注了但模型没找到 {tot_fn}")
    print(f"有分歧的图 {sum(1 for r in per_img if r['分歧数'])} 张，占 {sum(1 for r in per_img if r['分歧数'])/len(imgs)*100:.1f}%")
    print(f"\n分歧最多的 {args.top} 张图：")
    print(f"  {'图名':<46}{'标注':>6}{'预测':>6}{'模型多找':>9}{'模型漏':>8}{'分歧':>7}")
    for r in per_img[:args.top]:
        print(f"  {r['image'][:45]:<46}{r['gt']:>6}{r['pred']:>6}"
              f"{r['漏标候选(模型找到但没标注)']:>9}{r['多标候选(标注了但模型没找到)']:>8}{r['分歧数']:>7}")
    print(f"\n按图纸看分歧总数（前 10）：")
    agg = defaultdict(int)
    for r in per_img:
        agg[r["drawing"]] += r["分歧数"]
    for name, n in sorted(agg.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>6}  {name[:60]}")
    print(f"\n清单已写出：\n  {p1}（每张图一行）\n  {p2}（每个具体框一行，可直接拿去核对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
