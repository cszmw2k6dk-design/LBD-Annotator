"""3200 分辨率可行性探针：渲染少量页面到 3200，供 train_yolo26.py 实跑一个小 epoch。

目的：现在所有实测都是 2560 的。上 3200 之前要知道三件事——
1. 16GB 内存 + batch 4 会不会 OOM；
2. 真实每批耗时（推算值是不含内存压力的理论值）；
3. 磁盘/缓存代价。

只渲染 32 页（24 训练 + 8 验证），跑 1 个 epoch 就够判断，几分钟出结果。

用法:
    python probe_3200.py --build            # 生成 yolo26/probe3200/ 数据集
    python train_yolo26.py --model <权重> --data yolo26/probe3200/data.yaml \
        --imgsz 3200 --epochs 1 --batch 4 --workers 4 --cache none --name probe3200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "probe3200"

# 原始标注源（9000x6000 的 png + 同名 json）。本机路径，不用 scan_dataset 的默认值，
# 因为那份默认值被改成了公司电脑的目录。
SRC_ROOTS = [
    Path(r"C:\Users\szk\Desktop\batch_003"),
    Path(r"C:\Users\szk\Desktop\batch_004"),
    Path(r"C:\Users\szk\Desktop\batch_005"),
    Path(r"C:\Users\szk\Desktop\LBD-biaozhu\build_tmp\newdata"),
]

RENAME = {"Typical": "Tracker", "Node?": "Node", "Tracker?": "Tracker",
          "Typical?": "Tracker", "Node_?": "Node", "Tracker_?": "Tracker"}
CLASSES = ["Node", "Tracker"]


def pairs(limit_root: int = 40):
    """每个源目录抓一部分成对的 png+json。"""
    out = []
    for root in SRC_ROOTS:
        if not root.is_dir():
            print(f"[跳过] {root}")
            continue
        by_stem: dict[tuple[Path, str], dict[str, Path]] = {}
        for f in root.rglob("*"):
            ext = f.suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg", ".json"):
                continue
            slot = by_stem.setdefault((f.parent, f.stem), {})
            if ext == ".json":
                slot["json"] = f
            else:
                slot["img"] = f
        got = 0
        for _key, slot in by_stem.items():
            if "img" in slot and "json" in slot:
                out.append((slot["img"], slot["json"]))
                got += 1
            if got >= limit_root:
                break
        print(f"[源] {root.name}: 取 {got} 对")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--max-side", type=int, default=3200)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-val", type=int, default=8)
    args = ap.parse_args()

    if not args.build:
        print(__doc__)
        return 0

    from PIL import Image

    ps = pairs()
    random.Random(0).shuffle(ps)
    need = args.n_train + args.n_val
    if len(ps) < need:
        print(f"[错误] 只找到 {len(ps)} 对，不够 {need}", file=sys.stderr)
        return 2

    for split, chunk in (("train", ps[:args.n_train]), ("val", ps[args.n_train:need])):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, js in chunk:
            data = json.loads(js.read_text(encoding="utf-8"))
            with Image.open(img) as im:
                im = im.convert("RGB")
                W, H = im.size
                k = args.max_side / max(W, H)
                nw, nh = round(W * k), round(H * k)
                im.resize((nw, nh), Image.LANCZOS).save(
                    OUT / "images" / split / (img.stem + ".png"), format="PNG")
            lines = []
            for sh in data.get("shapes") or []:
                lab = RENAME.get((sh.get("label") or "").strip(),
                                 (sh.get("label") or "").strip())
                if lab not in CLASSES:
                    continue
                pts = sh.get("points") or []
                if not pts:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                x1, x2 = min(xs) * k, max(xs) * k
                y1, y2 = min(ys) * k, max(ys) * k
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0.5 or bh <= 0.5:
                    continue
                lines.append("%d %.6f %.6f %.6f %.6f" % (
                    CLASSES.index(lab), (x1 + x2) / 2 / nw, (y1 + y2) / 2 / nh,
                    bw / nw, bh / nh))
            (OUT / "labels" / split / (img.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[写] {split}: {len(chunk)} 页")

    (OUT / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (OUT / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n%s\n" % (
            OUT.resolve().as_posix(),
            "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))),
        encoding="utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"[完成] {OUT}  共 {need} 页，占用 {size:.1f} MB（这里只放了 32 页，"
          f"全量 1300 页大约 {size / need * 1300 / 1000:.1f} GB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
