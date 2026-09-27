"""A/B 诊断：同一批图，按「训练尺寸」和「工具里的原图尺寸」分别推理，比指标。

用来回答一个问题：标注工具的识别精度和训练时验证出来的精度对不上，
到底是模型泛化问题，还是喂给模型的图尺寸/尺度和训练不一致。

用法: python diag_scale.py --weights <best.pt> --conf 0.4 --max 40
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(HERE))
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from prepare_yolo import PAGE_RE, RENAME, collect_pairs, read_shapes  # noqa: E402


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(model, img_path: Path, gts, names, conf: float, imgsz: int):
    r = model.predict(source=str(img_path), imgsz=imgsz, conf=0.05, iou=0.7,
                      max_det=300, verbose=False)[0]
    preds = [(names[int(c)], list(map(float, b)), float(cf))
             for c, b, cf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                 r.boxes.conf.tolist())]
    stat = defaultdict(lambda: [0, 0, 0])
    used = [False] * len(gts)
    for lab, box, cf in sorted(preds, key=lambda x: -x[2]):
        if cf < conf:
            continue
        best_i, best_v = -1, 0.5
        for gi, (glab, gbox) in enumerate(gts):
            if used[gi] or glab != lab:
                continue
            v = iou(box, gbox)
            if v >= best_v:
                best_i, best_v = gi, v
        if best_i >= 0:
            used[best_i] = True
            stat[lab][0] += 1
            stat["__all__"][0] += 1
        else:
            stat[lab][1] += 1
            stat["__all__"][1] += 1
    for gi, (glab, _b) in enumerate(gts):
        if not used[gi]:
            stat[glab][2] += 1
            stat["__all__"][2] += 1
    return stat


def merge(a, b):
    for k, v in b.items():
        a[k][0] += v[0]
        a[k][1] += v[1]
        a[k][2] += v[2]
    return a


def show(title, stat, names):
    print(f"  {title}")
    print(f"    {'类别':<10}{'TP':>7}{'FP':>7}{'FN':>7}{'P':>9}{'R':>9}{'F1':>9}")
    for key in ["__all__", *names]:
        tp, fp, fn = stat.get(key, [0, 0, 0])
        p = tp / (tp + fp) if tp + fp else 0.0
        rr = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rr / (p + rr) if p + rr else 0.0
        print(f"    {'全部' if key == '__all__' else key:<10}{tp:>7}{fp:>7}{fn:>7}"
              f"{p:>9.3f}{rr:>9.3f}{f1:>9.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max", type=int, default=40)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from ultralytics import YOLO

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    val_stems = {p.stem for p in (HERE / "dataset" / "labels" / "val").glob("*.txt")}

    by_stem: dict[str, list] = defaultdict(list)
    for p in collect_pairs():
        if p["stem"] in val_stems:
            by_stem[p["stem"]].append(p)

    picked = []
    for stem, group in by_stem.items():
        for p in group:
            with Image.open(p["image"]) as im:
                p["w0"], p["h0"] = im.size
        group.sort(key=lambda p: p["w0"] * p["h0"], reverse=True)
        picked.append(group[0])
    # 从分辨率最高的开始取，这样 A 组（原图）和 B 组（预处理图）才有区别
    picked.sort(key=lambda p: (p["w0"] * p["h0"], p["stem"]), reverse=True)
    picked = picked[:args.max]
    hi = sum(1 for p in picked if p["w0"] > 2560)
    print(f"[样本] 验证集里 {len(val_stems)} 张，取分辨率最高的 {len(picked)} 张"
          f"（其中 {hi} 张是 9000×6000 级高清图，{len(picked) - hi} 张是 1080×720 小图）")

    model = YOLO(args.weights)
    a = defaultdict(lambda: [0, 0, 0])   # 原始（工具里那种大图）
    b = defaultdict(lambda: [0, 0, 0])   # 训练用预处理图
    for i, p in enumerate(picked, 1):
        with Image.open(p["image"]) as im:
            W, H = im.size
        gts = [(RENAME.get(s["label"], s["label"]),
                (s["x1"], s["y1"], s["x2"], s["y2"])) for s in read_shapes(p["json"])]
        merge(a, score(model, p["image"], gts, names, args.conf, args.imgsz))

        dst = HERE / "dataset" / "images" / "val" / (p["stem"] + ".png")
        lab = HERE / "dataset" / "labels" / "val" / (p["stem"] + ".txt")
        if dst.exists():
            with Image.open(dst) as im:
                W2, H2 = im.size
            g2 = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                c = line.split()
                if len(c) != 5:
                    continue
                k, cx, cy, bw, bh = int(c[0]), *map(float, c[1:])
                g2.append((names[k], ((cx - bw / 2) * W2, (cy - bh / 2) * H2,
                                      (cx + bw / 2) * W2, (cy + bh / 2) * H2)))
            merge(b, score(model, dst, g2, names, args.conf, args.imgsz))
        if i % 10 == 0:
            print(f"    {i}/{len(picked)}")
        # 释放
        gts.clear()

    print()
    print(f"[结果] conf>={args.conf}  imgsz={args.imgsz}")
    show("A. 喂原图（= 标注工具里的做法，大图）", a, names)
    show("B. 喂训练用的预处理图（= 训练/验证时的做法）", b, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
