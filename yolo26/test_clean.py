"""给 clean_shapes 做的自检：造几种"多余框"的场景，看它删得对不对。

用打包用的 Python 跑（只要 PySide6）：python test_clean.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lbd_annotator as ann  # noqa: E402


def mk(label, bbox, source="model", name="", conf=0.5, auto=False):
    d = {"label": label, "bbox": list(bbox), "source": source, "name": name,
         "confidence": conf, "raw": {}}
    if auto:
        d["_auto"] = True
    return d


def run(title, shapes, expect_kept):
    kept, st = ann.clean_shapes(shapes)
    ok = len(kept) == expect_kept
    print(f"[{'OK ' if ok else 'FAIL'}] {title}: 删了 {st['removed']} 个，剩 {len(kept)} "
          f"(期望 {expect_kept})  明细 {st}")
    return ok


def main() -> int:
    ok = True
    # 1) 手工框 + 一个叠在它上面的模型框 -> 模型框该删
    ok &= run("人工优先", [
        mk("Node", (100, 100, 200, 600), source="manual"),
        mk("Node", (104, 104, 204, 604)),
    ], 1)

    # 2) 两个模型框完全重合 -> 只留一个
    ok &= run("90% 重叠去重", [
        mk("Tracker", (10, 100, 18, 400), conf=0.3),
        mk("Tracker", (10, 100, 18, 400), conf=0.9),
    ], 1)

    # 2b) 两次识别的框只重叠 0.8（差几个像素）-> 也要去重，留置信度高的
    ok &= run("80% 重叠去重", [
        mk("Tracker", (10, 100, 18, 400), conf=0.9),
        mk("Tracker", (11, 100, 19, 400), conf=0.4),   # 右移 1px，IoU≈0.89
    ], 1)

    # 2c) 重叠只有 0.3 的正常相邻目标 -> 一个都不能删
    ok &= run("轻微重叠不误删", [
        mk("Tracker", (10, 100, 18, 400), conf=0.9),
        mk("Tracker", (15, 100, 23, 400), conf=0.8),   # 右移 5px，IoU≈0.375
    ], 2)

    # 3) 同一个编号出现两次（都读到过文字）-> 留置信度高的
    ok &= run("同编号去重", [
        mk("Node", (0, 0, 50, 300), name="7", conf=0.4),
        mk("Node", (400, 0, 460, 300), name="7", conf=0.8),
    ], 1)

    # 4) 编号是"按位置补的"（_auto）-> 不能拿来判重，两个都留
    ok &= run("补的号不判重", [
        mk("Node", (0, 0, 50, 300), name="7", conf=0.4, auto=True),
        mk("Node", (400, 0, 460, 300), name="7", conf=0.8, auto=True),
    ], 2)

    # 5) 形状离谱的 Tracker（又宽又扁）-> 删
    ok &= run("形状离谱", [
        mk("Tracker", (0, 0, 400, 100)),      # 宽/高 = 4
    ], 0)

    # 6) 正常的细长 Tracker -> 不能删
    ok &= run("正常 Tracker", [
        mk("Tracker", (0, 0, 7, 300)),
    ], 1)

    # 7) 手工框互相重叠 -> 一个都不动
    ok &= run("手工框不动", [
        mk("Node", (0, 0, 100, 300), source="manual"),
        mk("Node", (2, 2, 102, 302), source="manual"),
    ], 2)

    # 7b) 明确要求清手工框时 -> 同类重叠的只留一个，且优先留有编号的
    kept, st = ann.clean_shapes([
        mk("Node", (0, 0, 100, 300), source="manual", name=""),
        mk("Node", (2, 2, 102, 302), source="manual", name="INV1-LBD-07"),
    ], clean_manual=True)
    good = len(kept) == 1 and kept[0].get("name") == "INV1-LBD-07"
    print(f"[{'OK ' if good else 'FAIL'}] 清手工框重复(留编号那个): 剩 {len(kept)} "
          f"保留={kept[0].get('name')!r}  手工重复计数={st['manual_dup']}")
    ok &= good

    # 8) 模型框和一个"没名字"的手工框重叠 -> 模型框删
    ok &= run("对齐无编号人工框", [
        mk("Tracker", (0, 0, 8, 300), source="manual"),
        mk("Tracker", (1, 1, 9, 301), conf=0.99),
    ], 1)

    print("\n全部通过" if ok else "\n有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
