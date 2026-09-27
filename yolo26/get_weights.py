"""下载 YOLO26 预训练权重（以及 ultralytics 画图用的字体）到本目录。

用法: python get_weights.py [yolo26n.pt yolo26s.pt ...]
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(HERE, "ultralytics_cfg"))

from ultralytics import YOLO  # noqa: E402


def main() -> int:
    want = sys.argv[1:] or ["yolo26n.pt", "yolo26s.pt"]
    for name in want:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            print(f"[已有] {path}")
            continue
        m = YOLO(name)
        got = os.path.abspath(next(iter(m.ckpt_path for _ in [0])))
        print(f"[下载] {name} -> {got}  ({os.path.getsize(got) / 1e6:.1f} MB)")

    try:
        from ultralytics.utils.checks import check_font
        for font in ("Arial.ttf", "Arial.Unicode.ttf"):
            p = check_font(font)
            print(f"[字体] {font} -> {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"[字体] 跳过: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
