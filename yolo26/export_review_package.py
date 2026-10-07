"""把转换好的 agent3 数据导出成「标注软件能直接打开」的包：每页一张 PNG + 一个 JSON。

用途：人工逐页核对/修正自动生成的框。X-AnyLabeling 的 JSON 用绝对像素坐标，
所以这里把 YOLO 的归一化坐标换算回像素。

文件命名带两个页号：<name>_json017_pdf020.png / .json
  json017 = agent3 JSON 里的页码（原始标注的页码）
  pdf020  = 对应 PDF 的实际页码（= JSON 页号 + offset）

用法:
    python export_review_package.py --src yolo26/agent3_src/bigway --name bigway \
        --out yolo26/agent3_check --from-pdf 20
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = ["Node", "Tracker"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="prepare_agent3.py 的输出目录")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-pdf", type=int, default=1, help="只导出 PDF 页号 >= 这个值的页")
    ap.add_argument("--offset", type=int, default=0, help="PDF 页号 = JSON 页号 + 偏移")
    args = ap.parse_args()

    from PIL import Image

    src, out = Path(args.src), Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((src / "images").glob("*.png"))
    n_ok = n_skip = 0
    for im_path in imgs:
        m = re.search(r"_p(\d+)$", im_path.stem)
        pdf_page = int(m.group(1)) if m else 0
        json_page = pdf_page - args.offset
        if pdf_page < args.from_pdf:
            n_skip += 1
            continue
        with Image.open(im_path) as im:
            W, H = im.size
        lab = src / "labels" / (im_path.stem + ".txt")
        shapes = []
        for ln in lab.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = ln.split()
            cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
            shapes.append({
                "label": CLASSES[int(c)],
                "score": None,
                "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y1, 2)],
                           [round(x2, 2), round(y2, 2)], [round(x1, 2), round(y2, 2)]],
                "group_id": None,
                "description": "",
                "difficulty": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
            })
        base = f"{args.name}_json{json_page:03d}_pdf{pdf_page:03d}"
        shutil.copy2(im_path, out / (base + ".png"))
        (out / (base + ".json")).write_text(json.dumps({
            "version": "4.0.0-beta.11",
            "flags": {},
            "shapes": shapes,
            "imagePath": base + ".png",
            "imageData": None,
            "imageHeight": H,
            "imageWidth": W,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ok += 1
    print(f"[{args.name}] 导出 {n_ok} 页到 {out}（跳过 {n_skip} 页：PDF 页号 < {args.from_pdf}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
