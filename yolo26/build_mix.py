"""把已有数据集和 agent3 转出来的新数据合并成训练集，并生成核对材料。

切分规则（避免同图泄漏）：
- 老数据（dataset_v2）保持它原有的 train/val 划分不动（那本来就是按图纸分的）
- 新数据（agent3_src/*）按"张"分组：每张图纸是一组，按固定步长轮流进 val，
  这样 val 里也有新数据的样本，又不会让同一张图同时出现在两边

输出：
- 数据集目录（images/labels + data.yaml）
- summary.csv：各来源的页数/框数/类别分布
- preview/：抽查页面（把标签画到图上），供人工核对

用法:
    python build_mix.py --out yolo26/dataset_v3 --val-every 6 --include steelriver,bigway
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = ["Node", "Tracker"]


def count_boxes(label_file: Path) -> Counter:
    c: Counter = Counter()
    if label_file.exists():
        for ln in label_file.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                c[ln.split()[0]] += 1
    return c


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.hardlink_to(src)          # 同盘硬链接，几乎不占额外空间
        except OSError:
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "dataset_v2"), help="已有数据集（保持其切分）")
    ap.add_argument("--src-root", default=str(HERE / "agent3_src"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--include", default="steelriver,bigway", help="逗号分隔；空=不加新数据")
    ap.add_argument("--val-every", type=int, default=6, help="新数据每 N 张抽 1 张进 val")
    ap.add_argument("--copy", action="store_true", help="复制而不是硬链接（跨盘时才需要）")
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    summary = []
    preview_jobs = []

    def add_page(img: Path, label: Path, split: str, source: str) -> None:
        link_or_copy(img, out / "images" / split / img.name, args.copy)
        if label.exists():
            link_or_copy(label, out / "labels" / split / label.name, args.copy)
        else:
            (out / "labels" / split / (img.stem + ".txt")).write_text("", encoding="utf-8")

    base = Path(args.base)
    for split in ("train", "val"):
        imgs = sorted((base / "images" / split).glob("*.png"))
        box = Counter()
        for im in imgs:
            lab = base / "labels" / split / (im.stem + ".txt")
            box.update(count_boxes(lab))
            add_page(im, lab, split, "dataset_v2")
        summary.append(("dataset_v2", split, len(imgs), box.get("0", 0), box.get("1", 0)))
        preview_jobs += [(p, "dataset_v2") for p in imgs[:2]]

    for name in [s.strip() for s in args.include.split(",") if s.strip()]:
        src = Path(args.src_root) / name
        imgs = sorted((src / "images").glob("*.png"))
        box = Counter()
        n_val = 0
        for i, im in enumerate(imgs):
            split = "val" if (i % args.val_every == args.val_every - 1) else "train"
            n_val += split == "val"
            lab = src / "labels" / (im.stem + ".txt")
            box.update(count_boxes(lab))
            add_page(im, lab, split, name)
        summary.append((f"{name}+(train)", "train", len(imgs) - n_val,
                        box.get("0", 0), box.get("1", 0)))
        summary.append((f"{name}+(val)", "val", n_val, 0, 0))
        preview_jobs += [(p, name) for p in imgs[:2]]
        # 单独统计 val 那几页的框
        vb = Counter()
        for i, im in enumerate(imgs):
            if i % args.val_every == args.val_every - 1:
                vb.update(count_boxes(src / "labels" / (im.stem + ".txt")))
        summary[-1] = (f"{name}+(val)", "val", n_val, vb.get("0", 0), vb.get("1", 0))
        summary[-2] = (f"{name}+(train)", "train", len(imgs) - n_val,
                       box.get("0", 0) - vb.get("0", 0), box.get("1", 0) - vb.get("1", 0))

    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n%s\n" % (
            out.resolve().as_posix(),
            "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))),
        encoding="utf-8")

    with (out.parent / (out.name + "_summary.csv")).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "pages", "Node_boxes", "Tracker_boxes"])
        w.writerows(summary)

    # 核对图：把标签画回图上
    pdir = out.parent / (out.name + "_preview")
    pdir.mkdir(parents=True, exist_ok=True)
    for p, src_name in preview_jobs:
        target = next((out / "images" / s / p.name for s in ("train", "val")
                       if (out / "images" / s / p.name).exists()), None)
        if not target:
            continue
        split = target.parent.name
        lab = out / "labels" / split / (p.stem + ".txt")
        im = Image.open(target).convert("RGB")
        W, H = im.size
        k = 1500 / W
        small = im.resize((1500, round(H * k)), Image.LANCZOS)
        dr = ImageDraw.Draw(small)
        for ln in lab.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            c, cx, cy, bw, bh = ln.split()
            cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
            x1, y1 = (cx - bw / 2) * W * k, (cy - bh / 2) * H * k
            x2, y2 = (cx + bw / 2) * W * k, (cy + bh / 2) * H * k
            dr.rectangle([x1, y1, x2, y2],
                         outline=(255, 0, 0) if c == "0" else (0, 110, 255), width=2)
        small.save(pdir / f"{src_name}_{split}_{p.stem[:40]}.png")

    print(f"[完成] {out}")
    print(f"{'来源':<24}{'划分':<8}{'页数':>6}{'Node框':>10}{'Tracker框':>12}")
    for s in summary:
        print(f"{s[0]:<24}{s[1]:<8}{s[2]:>6}{s[3]:>10}{s[4]:>12}")
    print(f"\n核对图 {len(list(pdir.glob('*.png')))} 张 -> {pdir}")
    print(f"汇总表 -> {out.parent / (out.name + '_summary.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
