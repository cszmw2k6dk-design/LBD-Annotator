"""把 X-AnyLabeling 的 json+png 标注整理成 YOLO 数据集。

要点
- 只收「同一目录下同名 png + json」的成对样本（x-anylabeling 的坐标是相对它自己那张图的）
- 同一张图纸存在高/低分辨率两份时，只保留一份（默认保留分辨率高的）
- 大图统一缩到 --max-side，标注同步缩放；小图原样放过去
- 按图纸（去掉 _pNN 后缀）分组切 train/val，避免同一张图的不同页同时进训练和验证

用法:
    python prepare_yolo.py --dry-run          # 只出报告，不写文件
    python prepare_yolo.py                    # 生成 dataset/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_dataset import IMAGE_EXTS, PAGE_RE, ROOTS  # noqa: E402

OUT_DEFAULT = Path(__file__).resolve().parent / "dataset"
# 类别顺序跟标注工具 lbd_annotator.py 的 CLASSES 对齐（0=Node, 1=Tracker, 2=Box）
CLASS_ORDER = ["Node", "Tracker", "Box"]
# 原始 json 里的叫法 -> 导出用的叫法（A-F/G-Mhalf 只标了 Node，batch_003-007 标了 Node + Typical）
# 带问号的是"补标建议框"的标记（见 export_missing.py）：在一份标注里保留这个标记
# 便于复查，但训练时一律按正式类别算，不会多出第三个类。
RENAME = {"Typical": "Tracker",
          "Node?": "Node", "Tracker?": "Tracker", "Typical?": "Tracker",
          "Node_?": "Node", "Tracker_?": "Tracker"}


def iter_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def collect_pairs() -> list[dict]:
    """返回 [(image, json)] 的成对样本，同目录同名才算一对。"""
    per_dir: dict[Path, dict[str, Path]] = defaultdict(dict)
    for r in ROOTS:
        root = Path(r)
        if not root.is_dir():
            continue
        for f in iter_files(root):
            if f.suffix.lower() in IMAGE_EXTS or f.suffix.lower() == ".json":
                per_dir[f.parent].setdefault(f.suffix.lower(), {})[f.stem] = f

    pairs = []
    for d, by_ext in per_dir.items():
        imgs = {}
        for ext, mapping in by_ext.items():
            if ext != ".json":
                imgs.update(mapping)
        jsons = by_ext.get(".json", {})
        for stem, jp in jsons.items():
            if stem in imgs:
                pairs.append({"stem": stem, "image": imgs[stem], "json": jp})
    return pairs


def read_shapes(jp: Path) -> list[dict]:
    data = json.loads(jp.read_text(encoding="utf-8"))
    out = []
    for sh in data.get("shapes") or []:
        pts = sh.get("points") or []
        if not pts:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        out.append({
            "label": (sh.get("label") or "").strip(),
            "x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys),
        })
    return out


def render_one(job) -> tuple:
    """子进程任务：把原图写成数据集里的图，返回 (split, stem, 缩放比例)。"""
    from PIL import Image

    src, stem, split, max_side, out = job
    dst = Path(out) / "images" / split / (stem + ".png")
    scale = 1.0
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        long_side = max(w, h)
        if long_side > max_side:
            factor = long_side // max_side
            if factor >= 2:
                im = im.reduce(int(factor))
            cur = max(im.size)
            w2 = max(1, round(im.size[0] * max_side / cur))
            h2 = max(1, round(im.size[1] * max_side / cur))
            if im.size != (w2, h2):
                im = im.resize((w2, h2), Image.LANCZOS)
            scale = max(im.size) / long_side
        im.save(dst, format="PNG")
    return split, stem, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--max-side", type=int, default=2560, help="长边上限，超过就缩小")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dedup", choices=["res", "boxes"], default="res",
                    help="同一图纸有高低两份时保留哪份：res=分辨率高的，boxes=框更多的")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    pairs = collect_pairs()
    print(f"[1/6] 同目录成对样本: {len(pairs)}")

    # --- 读尺寸 + 读标注 ---
    for p in pairs:
        with Image.open(p["image"]) as im:
            p["w"], p["h"] = im.size
        p["shapes"] = read_shapes(p["json"])
        for s in p["shapes"]:
            s["label"] = RENAME.get(s["label"], s["label"])
        p["group"] = PAGE_RE.sub("", p["stem"])

    # --- 同名去重：同一张图纸只留一份 ---
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_stem[p["stem"]].append(p)
    dup_stems = {s: v for s, v in by_stem.items() if len(v) > 1}

    n_dup_box_diff = 0
    dup_rows = []
    kept = []
    for stem, group in by_stem.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group = sorted(group, key=lambda p: p["w"] * p["h"], reverse=True)
        if len({(p["w"], p["h"]) for p in group}) > 1:
            n_dup_box_diff += 1
        counts = [len(p["shapes"]) for p in group]
        dup_rows.append((stem, [f"{p['w']}x{p['h']}/{len(p['shapes'])}框" for p in group]))
        if args.dedup == "res":
            pick = group[0]
        else:
            pick = max(group, key=lambda p: len(p["shapes"]))
        for p in group:
            p["dropped"] = True
        pick.pop("dropped", None)
        kept.append(pick)
    print(f"[2/6] 同名图纸重复: {len(dup_stems)} 组（分辨率不同 {n_dup_box_diff} 组），去重后样本: {len(kept)}")

    # --- 坐标越界检查 ---
    oob = 0
    bad_label = Counter()
    for p in kept:
        for s in p["shapes"]:
            if s["x2"] < 0 or s["y2"] < 0 or s["x1"] > p["w"] or s["y1"] > p["h"]:
                oob += 1
            if s["label"] not in CLASS_ORDER:
                bad_label[s["label"]] += 1
    print(f"[3/6] 完全落在图外的框: {oob}；未知类别: {dict(bad_label)}")

    # --- 类别 ---
    label_cnt = Counter(s["label"] for p in kept for s in p["shapes"])
    names = [c for c in CLASS_ORDER if c in label_cnt]
    names += [c for c in label_cnt if c not in CLASS_ORDER]
    print(f"      类别: {', '.join(f'{i}={n}({label_cnt[n]})' for i, n in enumerate(names))}")

    # --- 每类框尺寸分布（占整图百分比），用来判断两类到底是什么 ---
    geo = defaultdict(list)
    for p in kept:
        for s in p["shapes"]:
            geo[s["label"]].append(((s["x2"] - s["x1"]) / p["w"] * 100, (s["y2"] - s["y1"]) / p["h"] * 100))
    print("      各类框大小(占图纸 %，中位数/90分位):")
    for lab, vals in geo.items():
        ws = sorted(v[0] for v in vals)
        hs = sorted(v[1] for v in vals)

        def pct(a, q):
            return a[min(len(a) - 1, int(q * len(a)))]
        print(f"        {lab:<10} 宽 {statistics.median(ws):6.3f}% / {pct(ws, 0.9):6.3f}%"
              f"   高 {statistics.median(hs):6.3f}% / {pct(hs, 0.9):6.3f}%   n={len(vals)}")

    # --- 切分 ---
    groups = defaultdict(list)
    for p in kept:
        groups[p["group"]].append(p)
    gnames = sorted(groups)
    random.Random(args.seed).shuffle(gnames)
    target_val = len(kept) * args.val_ratio
    val_groups, n_val = [], 0
    for g in gnames:
        if n_val >= target_val:
            break
        val_groups.append(g)
        n_val += len(groups[g])
    val_set = set(val_groups)
    splits = {"val": [p for p in kept if p["group"] in val_set],
              "train": [p for p in kept if p["group"] not in val_set]}
    print(f"[4/6] 切分: train {len(splits['train'])} 张 / val {len(splits['val'])} 张"
          f"（{len(val_groups)}/{len(gnames)} 张图纸进 val）")
    print(f"      val 图纸: {', '.join(val_groups)}")

    # --- 分辨率处理计划 ---
    need_resize = [p for p in kept if max(p["w"], p["h"]) > args.max_side]
    print(f"[5/6] 需要缩小的图: {len(need_resize)} 张（长边 > {args.max_side}）")

    if args.dry_run:
        print("[6/6] --dry-run，不写文件")
        return 0

    # --- 写数据集 ---
    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    jobs = [(p["image"], p["stem"], split, args.max_side, str(out))
            for split, items in splits.items() for p in items]
    by_stem_split = {p["stem"]: (split, p) for split, items in splits.items() for p in items}
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(render_one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            results.append(fu.result())
            if k % 50 == 0 or k == len(futs):
                print(f"      渲染 {k}/{len(futs)}")

    n_box = 0
    class_ids = {n: i for i, n in enumerate(names)}
    for split, stem, scale in results:
        p = by_stem_split[stem][1]
        W = p["w"] * scale
        H = p["h"] * scale
        lines = []
        for s in p["shapes"]:
            cid = class_ids.get(s["label"])
            if cid is None:
                continue
            x1, y1 = s["x1"] * scale, s["y1"] * scale
            x2, y2 = s["x2"] * scale, s["y2"] * scale
            x1 = min(max(x1, 0.0), W)
            x2 = min(max(x2, 0.0), W)
            y1 = min(max(y1, 0.0), H)
            y2 = min(max(y2, 0.0), H)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0.5 or bh <= 0.5:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (
                cid, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, bw / W, bh / H))
        (out / "labels" / split / (p["stem"] + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_box += len(lines)

    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n%s\n" % (
            out.as_posix(), "\n".join(f"  {i}: {n}" for i, n in enumerate(names))),
        encoding="utf-8")
    print(f"[6/6] 完成: {out}  图片 {len(results)} 张，框 {n_box} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
