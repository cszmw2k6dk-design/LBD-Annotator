"""量一下：人工标注的框边，离图纸上真实的亮度台阶有多远。

如果标注本身有 1~2 像素的随手抖动，那"把框吸到几何边界"就会偏离标注 ——
指标会变差，但视觉上反而更贴。这个测试决定吸附到底该不该做。

用法: python diag_gt_edge.py --max 30
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def step_response(prof, w=2):
    """每个位置的"台阶响应"：右边 w 个的均值 - 左边 w 个的均值（取绝对值）。"""
    p = prof.astype(np.float32)
    n = p.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        a = p[max(0, i - w):i]
        b = p[i:i + w]
        if a.size and b.size:
            out[i] = abs(float(b.mean()) - float(a.mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--radius", type=float, default=3.0)
    args = ap.parse_args()

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    img_dir = HERE / "dataset" / "images" / "val"
    lab_dir = HERE / "dataset" / "labels" / "val"

    imgs = []
    for ip in sorted(img_dir.glob("*.png")):
        with Image.open(ip) as im:
            if im.size[0] > 2000:
                imgs.append(ip)
    imgs = imgs[:args.max]

    r = args.radius
    dist = {n: {"left": [], "right": [], "top": [], "bottom": [], "strength": []}
            for n in names}
    for ip in imgs:
        gray = np.asarray(Image.open(ip).convert("L"))
        H, W = gray.shape
        dark = 255.0 - gray.astype(np.float32)
        rows = [ln.split() for ln in (lab_dir / (ip.stem + ".txt")).read_text(
            encoding="utf-8").splitlines()]
        for c in rows:
            if len(c) != 5:
                continue
            k, cx, cy, bw, bh = int(c[0]), *map(float, c[1:])
            lab = names[k]
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
            if x2 - x1 < 3 or y2 - y1 < 6:
                continue
            ya, yb = int(max(0, y1)), int(min(H, y2))
            xa, xb = int(max(0, x1)), int(min(W, x2))
            colprof = dark[ya:yb, :].mean(axis=0)
            rowprof = dark[:, xa:xb].mean(axis=1)
            sr_c = step_response(colprof)
            sr_r = step_response(rowprof)
            for edge, prof, sr, pos in (("left", colprof, sr_c, x1),
                                        ("right", colprof, sr_c, x2)):
                lo, hi = int(max(0, pos - r)), int(min(len(sr) - 1, pos + r))
                if hi - lo < 2:
                    continue
                seg = sr[lo:hi + 1]
                j = int(np.argmax(seg))
                dist[lab][edge].append(lo + j - pos)
                dist[lab]["strength"].append(float(seg[j]))
            for edge, prof, sr, pos in (("top", rowprof, sr_r, y1),
                                        ("bottom", rowprof, sr_r, y2)):
                lo, hi = int(max(0, pos - r)), int(min(len(sr) - 1, pos + r))
                if hi - lo < 2:
                    continue
                seg = sr[lo:hi + 1]
                j = int(np.argmax(seg))
                dist[lab][edge].append(lo + j - pos)
                dist[lab]["strength"].append(float(seg[j]))

    print(f"[样本] {len(imgs)} 张高清验证图，统计标注框边到最强台阶的有符号距离（像素）")
    for lab in names:
        d = dist[lab]
        allv = d["left"] + d["right"] + d["top"] + d["bottom"]
        if not allv:
            continue
        tied = sum(1 for v in allv if abs(v) <= 1)
        print(f"\n=== {lab}（{len(allv)} 条边）===")
        print(f"  平均 {statistics.mean(allv):+.3f}   中位 {statistics.median(allv):+.1f}   "
              f"距边界 ≤1px 的占比 {tied/len(allv)*100:.1f}%   "
              f"台阶强度中位 {statistics.median(d['strength']):.1f}")
        hist = {}
        for v in allv:
            v = max(-3, min(3, int(round(v))))
            hist[v] = hist.get(v, 0) + 1
        for k in sorted(hist):
            bar = "#" * int(60 * hist[k] / len(allv))
            print(f"    {k:+d} px : {hist[k]:>6} {hist[k]/len(allv)*100:5.1f}% {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
