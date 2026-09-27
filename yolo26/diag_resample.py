"""对照实验：同一张高清图，三种喂法，看指标差多少。

  orig      直接喂 9000x6000（= 标注工具现在的做法，降采样由 ultralytics/cv2 内部做）
  l2560     先用 PIL LANCZOS 缩到 2560 再喂（= 训练集预处理的做法）
  l1920     先用 PIL LANCZOS 缩到 1920 再喂（一步到位的高质量缩放）

用法: python diag_resample.py --weights <best.pt> --n 8 --conf 0.4
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
from prepare_yolo import RENAME, collect_pairs, read_shapes  # noqa: E402


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(model, path: Path, gts, names, conf: float, imgsz: int):
    r = model.predict(source=str(path), imgsz=imgsz, conf=0.05, iou=0.7,
                      max_det=300, verbose=False)[0]
    preds = [(names[int(c)], list(map(float, b)), float(cf))
             for c, b, cf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                 r.boxes.conf.tolist())]
    stat = defaultdict(lambda: [0, 0, 0])
    used = [False] * len(gts)
    for lab, box, cf in sorted(preds, key=lambda x: -x[2]):
        if cf < conf:
            continue
        bi, bv = -1, 0.5
        for gi, (gl, gb) in enumerate(gts):
            if used[gi] or gl != lab:
                continue
            v = iou(box, gb)
            if v >= bv:
                bi, bv = gi, v
        if bi >= 0:
            used[bi] = True
            stat[lab][0] += 1
            stat["__all__"][0] += 1
        else:
            stat[lab][1] += 1
            stat["__all__"][1] += 1
    for gi, (gl, _b) in enumerate(gts):
        if not used[gi]:
            stat[gl][2] += 1
            stat["__all__"][2] += 1
    return stat


def merge(a, b):
    for k, v in b.items():
        a[k][0] += v[0]; a[k][1] += v[1]; a[k][2] += v[2]
    return a


def show(title, stat, names):
    print(f"  {title}")
    print(f"    {'类别':<9}{'TP':>7}{'FP':>7}{'FN':>7}{'P':>8}{'R':>8}{'F1':>8}")
    for key in ["__all__", *names]:
        tp, fp, fn = stat.get(key, [0, 0, 0])
        p = tp / (tp + fp) if tp + fp else 0.0
        rr = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rr / (p + rr) if p + rr else 0.0
        print(f"    {'全部' if key == '__all__' else key:<9}{tp:>7}{fp:>7}{fn:>7}"
              f"{p:>8.3f}{rr:>8.3f}{f1:>8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from ultralytics import YOLO

    names = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8").split()
    val_stems = {p.stem for p in (HERE / "dataset" / "labels" / "val").glob("*.txt")}
    tmp = HERE / "runs" / "diag"
    tmp.mkdir(parents=True, exist_ok=True)

    # 取验证集里分辨率最高的 n 张（用它们各自的原始 json 当标注）
    cand = []
    for p in collect_pairs():
        if p["stem"] not in val_stems:
            continue
        with Image.open(p["image"]) as im:
            p["w0"], p["h0"] = im.size
        cand.append(p)
    cand.sort(key=lambda p: (p["w0"] * p["h0"], p["stem"]), reverse=True)
    seen, picked = set(), []
    for p in cand:
        if p["stem"] in seen or p["w0"] < 8000:
            continue
        seen.add(p["stem"])
        picked.append(p)
        if len(picked) >= args.n:
            break
    print(f"[样本] {len(picked)} 张 9000x6000 级验证图")

    model = YOLO(args.weights)
    agg = {k: defaultdict(lambda: [0, 0, 0]) for k in ("orig", "l2560", "l1920")}
    for p in picked:
        shapes = read_shapes(p["json"])
        base = [(RENAME.get(s["label"], s["label"]),
                 (s["x1"], s["y1"], s["x2"], s["y2"])) for s in shapes]
        with Image.open(p["image"]) as im:
            im = im.convert("RGB")
            W0, H0 = im.size
            for tag, target in (("l2560", 2560), ("l1920", 1920)):
                sc = target / max(W0, H0)
                im2 = im.resize((round(W0 * sc), round(H0 * sc)), Image.LANCZOS)
                f = tmp / f"{p['stem'][:40]}_{tag}.png"
                im2.save(f)
                gts = [(lab, (b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc)) for lab, b in base]
                merge(agg[tag], match(model, f, gts, names, args.conf, args.imgsz))
        merge(agg["orig"], match(model, p["image"], base, names, args.conf, args.imgsz))
        print(f"    完成 {p['stem'][:50]}")

    print()
    print(f"[结果] conf>={args.conf}  imgsz={args.imgsz}")
    show("orig  直接喂 9000x6000（工具现在的做法：内部 cv2 一步降采样）", agg["orig"], names)
    show("l2560 先 LANCZOS 缩到 2560 再喂（训练时的预处理方式）", agg["l2560"], names)
    show("l1920 先 LANCZOS 缩到 1920 再喂", agg["l1920"], names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
