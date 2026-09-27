"""抽查 YOLO 数据集：把 labels/*.txt 画回图上，确认坐标对齐。

用法: python verify_dataset.py [--n 4] [--split train]
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DS = HERE / "dataset"
COLORS = {0: (255, 0, 0), 1: (0, 110, 255), 2: (0, 170, 0), 3: (200, 0, 200)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--outdir", default=str(HERE / "preview"))
    args = ap.parse_args()

    names = (DS / "classes.txt").read_text(encoding="utf-8").split()
    imgs = sorted((DS / "images" / args.split).glob("*.png"))
    stats = Counter()
    sizes = Counter()
    for ip in imgs:
        lp = DS / "labels" / args.split / (ip.stem + ".txt")
        for line in lp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                stats[int(line.split()[0])] += 1
        with Image.open(ip) as im:
            sizes[im.size] += 1
    print(f"{args.split}: {len(imgs)} 张")
    print("  类别框数:", {names[k]: v for k, v in sorted(stats.items())})
    print("  图片尺寸:", dict(sizes))
    print("  总框数:", sum(stats.values()), " 每图平均:", round(sum(stats.values()) / max(1, len(imgs)), 1))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    random.Random(1).shuffle(imgs)
    for ip in imgs[:args.n]:
        lp = DS / "labels" / args.split / (ip.stem + ".txt")
        with Image.open(ip) as im:
            im = im.convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                d.rectangle([x1, y1, x2, y2], outline=COLORS.get(c, (0, 0, 0)),
                            width=max(2, round(W / 900)))
            im.thumbnail((1500, 1500))
            dst = outdir / f"check_{args.split}_{ip.stem[:50]}.png"
            im.save(dst)
        print("  ", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
