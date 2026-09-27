"""把某个 json 的框画到图上，用于肉眼确认标注对象和尺度。

用法: python preview_boxes.py [关键字] [--n 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_yolo import collect_pairs, read_shapes  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402

COLORS = {"Node": (255, 0, 0), "Typical": (0, 128, 255)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keyword", nargs="?", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--zoom", type=float, default=0.0,
                    help=">0 时额外输出按该倍率放大的局部裁剪")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "preview"))
    args = ap.parse_args()

    pairs = collect_pairs()
    picked = [p for p in pairs if args.keyword.lower() in p["stem"].lower()][:args.n]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in picked:
        shapes = read_shapes(p["json"])
        with Image.open(p["image"]) as im:
            im = im.convert("RGB")
            W, H = im.size
            if args.zoom > 0:
                for cls in ("Node", "Typical"):
                    cand = [s for s in shapes if s["label"] == cls]
                    if not cand:
                        continue
                    s = cand[0]
                    cx, cy = (s["x1"] + s["x2"]) / 2, (s["y1"] + s["y2"]) / 2
                    half = 0.06 * max(W, H)
                    box = (max(0, int(cx - half)), max(0, int(cy - half)),
                           min(W, int(cx + half)), min(H, int(cy + half)))
                    crop = im.crop(box)
                    crop = crop.resize((int(crop.width * args.zoom), int(crop.height * args.zoom)),
                                       Image.LANCZOS)
                    cd = ImageDraw.Draw(crop)
                    for t in cand[:3]:
                        col = COLORS.get(t["label"], (0, 200, 0))
                        cd.rectangle([(t["x1"] - box[0]) * args.zoom, (t["y1"] - box[1]) * args.zoom,
                                      (t["x2"] - box[0]) * args.zoom, (t["y2"] - box[1]) * args.zoom],
                                     outline=col, width=2)
                    zdst = outdir / f"zoom_{cls}_{p['stem'][:40]}.png"
                    crop.save(zdst)
                    print(f"  {zdst}  ({crop.width}x{crop.height})")
            d = ImageDraw.Draw(im)
            for s in shapes:
                col = COLORS.get(s["label"], (0, 200, 0))
                w = max(2, round(W / 900))
                d.rectangle([s["x1"], s["y1"], s["x2"], s["y2"]], outline=col, width=w)
            dst = outdir / f"{p['stem']}_{W}x{H}.png"
            im.thumbnail((1600, 1600))
            im.save(dst)
        print(f"{dst}  框数={len(shapes)}  原图={W}x{H}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
