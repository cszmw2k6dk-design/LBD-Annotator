"""标注偏移清单：按"框边离图纸真实边界的距离"排序，挑出最该返工的标注框。

不用推理，只读图和标注，所以很快。输出 runs/diag/annot_offset.csv。

用法: python annot_offset.py [--split train] [--radius 4]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PAGE_RE = re.compile(r"_p0*\d+$", re.IGNORECASE)
# 台阶强度门槛：低于这个值就认为"附近没有真正的边界"，那条边不参与判断。
# 参考实测：Tracker 的边界台阶强度中位 33，Node 是 78。
STRONG_STEP = 20.0


def step_at(prof, pos, radius, w=2):
    """pos 附近最强的亮度台阶：返回 (带符号距离, 强度, 峰值是否顶在窗口边上)。

    峰值顶在窗口边上说明"附近没有真正的边界"，那个距离不可信 —— 要单独标出来。
    """
    n = len(prof)
    lo = max(1, int(pos - radius))
    hi = min(n - 2, int(pos + radius))
    if hi - lo < 2:
        return 0.0, 0.0, True
    seg = prof[lo - 1:hi + 2].astype(np.float32)
    csum = np.concatenate([[0.0], np.cumsum(seg)])
    idx = np.arange(1, len(seg) - 1)
    a0 = np.maximum(0, idx - w)
    b1 = np.minimum(len(seg), idx + w)
    left = (csum[idx] - csum[a0]) / np.maximum(1, idx - a0)
    right = (csum[b1] - csum[idx]) / np.maximum(1, b1 - idx)
    resp = np.abs(right - left)
    j = int(np.argmax(resp))
    at_edge = j in (0, len(resp) - 1)
    return float(lo + j - pos), float(resp[j]), at_edge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--strong", type=float, default=STRONG_STEP)
    args = ap.parse_args()
    strong = args.strong

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    img_dir = HERE / "dataset" / "images" / args.split
    lab_dir = HERE / "dataset" / "labels" / args.split
    out_dir = HERE / "runs" / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    by_drawing = defaultdict(lambda: [0, 0])   # 图纸 -> [框数, 偏差>2px 的框数]
    imgs = sorted(img_dir.glob("*.png"))
    for k, ip in enumerate(imgs, 1):
        with Image.open(ip) as im:
            W, H = im.size
            gray = np.asarray(im.convert("L"), dtype=np.float32)
        dark = 255.0 - gray
        col_cum = np.cumsum(dark, axis=0)       # 每列前缀和 → 任意行区间的列剖面 O(W)
        row_cum = np.cumsum(dark, axis=1)
        drawing = PAGE_RE.sub("", ip.stem)
        for line in (lab_dir / (ip.stem + ".txt")).read_text(encoding="utf-8").splitlines():
            c = line.split()
            if len(c) != 5:
                continue
            kk, cx, cy, bw, bh = int(c[0]), *map(float, c[1:])
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
            ya = int(max(0, np.floor(y1)))
            yb = int(min(H, np.ceil(y2)))
            xa = int(max(0, np.floor(x1)))
            xb = int(min(W, np.ceil(x2)))
            if yb - ya < 4 or xb - xa < 2:
                continue
            colprof = (col_cum[yb - 1] - (col_cum[ya - 1] if ya > 0 else 0)) / (yb - ya)
            rowprof = (row_cum[:, xb - 1] - (row_cum[:, xa - 1] if xa > 0 else 0)) / (xb - xa)
            d, strengths, at_edges = {}, [], []
            for key, prof, pos in (("left", colprof, x1), ("right", colprof, x2),
                                   ("top", rowprof, y1), ("bottom", rowprof, y2)):
                off, st, at_e = step_at(prof, pos, args.radius)
                d[key] = off
                strengths.append(st)
                at_edges.append(at_e)
            # 只统计"附近确实有一条明显的线"的边：强度够、且峰值没顶在窗口边上
            judged = [abs(d[k]) for k, st, ae in
                      zip(("left", "right", "top", "bottom"), strengths, at_edges)
                      if st >= strong and not ae]
            off_line = sum(1 for v in judged if v > 1.5)
            on_line = sum(1 for v in judged if v <= 1.0)
            worst = max((abs(v) for v in d.values()), default=0.0)
            strength = min(strengths)
            by_drawing[drawing][0] += 1
            if off_line >= 2:
                by_drawing[drawing][1] += 1
            rows.append({
                "image": ip.stem, "class": names[kk],
                "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                "left": round(d["left"], 2), "right": round(d["right"], 2),
                "top": round(d["top"], 2), "bottom": round(d["bottom"], 2),
                "有线的边": len(judged), "贴上的边": on_line, "没贴上的边": off_line,
                "没贴上的平均偏移": round(statistics.mean([v for v in judged if v > 1.5]), 2)
                                    if off_line else 0.0,
                "worst_px": round(worst, 2), "edge_strength": round(strength, 1),
                "w_px": round(x2 - x1, 1), "h_px": round(y2 - y1, 1),
            })
        if k % 100 == 0:
            print(f"    {k}/{len(imgs)}")

    # 排序：先看"有几条边明明有线却没贴上去"，再看偏得多不多
    rows.sort(key=lambda r: (-r["没贴上的边"], -r["没贴上的平均偏移"], -r["worst_px"]))
    csv_path = out_dir / "annot_offset.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n[样本] {args.split} 集 {len(imgs)} 张图 / {len(rows)} 个框")
    h = Counter(("有线0条" if r["有线的边"] == 0 else
                 f"有线{r['有线的边']}条→贴上{r['贴上的边']}条" ) for r in rows)
    print("每条边先问两件事：旁边有没有一条明显的线？框边有没有贴上去？")
    for key, n in sorted(h.items()):
        print(f"  {key:<22}{n:>7}  {n/len(rows)*100:5.1f}%")
    bad = sum(1 for r in rows if r["没贴上的边"] >= 2)
    print(f"\n返工候选（≥2 条边旁边有线却没贴上去）：{bad} 个（{bad/len(rows)*100:.1f}%）")
    print("  分类别：", {n: sum(1 for r in rows if r["class"] == n and r["没贴上的边"] >= 2)
                        for n in names})

    print(f"\n按批次的偏差比例（偏 >2px 的框 / 该批次框数）：")
    ranked = sorted(by_drawing.items(), key=lambda kv: -(kv[1][1] / max(1, kv[1][0])))
    for name, (tot, nbad) in ranked[:12]:
        print(f"  {nbad/tot*100:5.1f}%  ({nbad:>4}/{tot:<4})  {name[:60]}")

    print(f"\n最需要先看的 {args.top} 个框（坐标是 2560 底图里的像素）：")
    print(f"  {'图名':<36}{'类':<9}{'宽':>6}{'高':>6}{'有线':>5}{'贴上':>5}{'没贴':>5}"
          f"{'平均偏':>7}{'左':>6}{'右':>6}{'上':>6}{'下':>6}")
    for r in rows[:args.top]:
        print(f"  {r['image'][:35]:<36}{r['class']:<9}{r['w_px']:>6.1f}{r['h_px']:>6.1f}"
              f"{r['有线的边']:>5}{r['贴上的边']:>5}{r['没贴上的边']:>5}"
              f"{r['没贴上的平均偏移']:>7.1f}"
              f"{r['left']:>6.1f}{r['right']:>6.1f}{r['top']:>6.1f}{r['bottom']:>6.1f}")
    print(f"\n完整清单已写出：{csv_path}（用 Excel 打开可排序筛选）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
