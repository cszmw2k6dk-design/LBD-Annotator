"""把验证集拆成「旧数据页 / 新数据页」分别评估，用来定位掉点原因。

用途：v4 训练时 val 指标不升反降，需要区分两种解释——
  A. 学习率太大把模型带偏（旧数据上也会掉）→ 两边都变差
  B. 新数据标注有问题/风格冲突（旧数据还好，新数据很差）
所以拿同一个权重分别在两个子集上跑，和起点权重对比。

用法:
    python diag_split_eval.py --weights <best.pt> --tag v4 --out runs/diag/split_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / "ultralytics_cfg"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcfg"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "runs" / "diag" / "split_eval.csv"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.chdir(HERE)
    from ultralytics import YOLO

    rows = []
    for sub in ("old", "new"):
        yaml = HERE / "runs" / "diag" / f"data_{sub}.yaml"
        if not yaml.exists():
            print(f"[跳过] 缺 {yaml}")
            continue
        model = YOLO(args.weights)
        res = model.val(data=str(yaml), imgsz=args.imgsz, batch=args.batch,
                        split="val", verbose=False, plots=False, conf=0.001)
        names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names))
        print(f"=== {args.tag} | {sub} 子集（{len(res.box.ap50)} 类）")
        for i, cls in names.items():
            p = float(res.box.p[i])
            r = float(res.box.r[i])
            ap50 = float(res.box.ap50[i])
            ap = float(res.box.maps[i])
            print(f"   {cls:<8} P {p:.4f}  R {r:.4f}  mAP50 {ap50:.4f}  mAP50-95 {ap:.4f}")
            rows.append((args.tag, sub, cls, round(p, 4), round(r, 4),
                         round(ap50, 4), round(ap, 4)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["tag", "subset", "class", "P", "R", "mAP50", "mAP50_95"])
        w.writerows(rows)
    print(f"[写入] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
