"""把训练好的权重整理到 models/<name>/，并放好 classes.txt（标注工具有「用模型识别」会读它）。

用法:
    python export_model.py                       # 默认取最新的训练结果
    python export_model.py --name yolo26n_1280 --weight best
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "detect"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--weight", default="best", choices=["best", "last"])
    ap.add_argument("--out", default=str(HERE / "models"))
    args = ap.parse_args()

    if args.name:
        run_dir = RUNS / args.name
    else:
        cands = [d for d in RUNS.iterdir() if (d / "weights" / "best.pt").exists()]
        if not cands:
            print(f"[错误] {RUNS} 下没有训练结果")
            return 2
        run_dir = max(cands, key=lambda d: (d / "weights" / "best.pt").stat().st_mtime)
    src = run_dir / "weights" / f"{args.weight}.pt"
    if not src.exists():
        print(f"[错误] 找不到 {src}")
        return 2

    dst_dir = Path(args.out) / run_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / f"{run_dir.name}.pt")
    classes = (HERE / "dataset" / "classes.txt").read_text(encoding="utf-8")
    (dst_dir / "classes.txt").write_text(classes, encoding="utf-8")

    # meta.txt：把"这版模型该怎么用"记在权重旁边，标注工具会读它自动填 imgsz，
    # 免得又出现"模型是 2560 训的、工具里 imgsz 还写着 1920"这种错。
    imgsz = ""
    ay = run_dir / "args.yaml"
    if ay.exists():
        for line in ay.read_text(encoding="utf-8", errors="replace").splitlines():
            k, _, v = line.partition(": ")
            if k.strip() == "imgsz":
                imgsz = v.strip()
                break
    import datetime
    (dst_dir / "meta.txt").write_text(
        "imgsz=%s\nclasses=%s\nrun=%s\nexported=%s\n"
        % (imgsz, ",".join(classes.split()), run_dir.name,
           datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
        encoding="utf-8")

    print(f"[模型] {dst_dir / (run_dir.name + '.pt')}")
    print(f"[类别] {(dst_dir / 'classes.txt')}  ->  {classes.split()}")
    print(f"[参数] {dst_dir / 'meta.txt'}  imgsz={imgsz or '(没找到)'}")
    print("        在标注工具里：「用模型识别」→ 选这个 .pt（同目录的 classes.txt 会被自动读取）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
