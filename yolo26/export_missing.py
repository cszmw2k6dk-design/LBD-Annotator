"""把"模型找到、标注里没有"的框导出成 X-AnyLabeling 标注文件（含原有标注）。

生成的每个 json 和你的原始标注同格式、同坐标空间（9000x6000 原图），
所以可以直接用 X-AnyLabeling 或 LBD 标注工具打开：原有框都在，
多出来的那些是模型建议补的（score 有值、description 写了 model_suggest）。

用法: python export_missing.py [--min 1] [--out 补标建议]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_yolo import collect_pairs  # noqa: E402

MARK = "model_suggest"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=1, help="少于这么多个漏标候选的页就不导出")
    ap.add_argument("--out", default="补标建议")
    ap.add_argument("--with-orig", action="store_true",
                    help="默认只导出建议框（打开就只看问号）；加这个把原有标注也带上")
    ap.add_argument("--ratio", type=float, default=0.0,
                    help="只导出置信度 >= 这个值的建议（0 = 全导）")
    args = ap.parse_args()

    src = HERE / "runs" / "diag" / "hard_boxes.csv"
    if not src.exists():
        print(f"[错误] 找不到 {src}，先跑 hard_examples.py")
        return 2
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    miss = defaultdict(list)
    for r in rows:
        if r["类型"] == "模型找到、标注没有":
            if args.ratio and float(r["conf"]) < args.ratio:
                continue
            miss[r["image"]].append(r)
    miss = {k: v for k, v in miss.items() if len(v) >= args.min}
    print(f"[1/3] 有漏标候选的页 {len(miss)} 张，候选框 {sum(len(v) for v in miss.values())} 个")

    # 数据集里的图 -> 原图（取同名里最大的那张，和训练用的一致）
    pairs = {}
    for p in collect_pairs():
        old = pairs.get(p["stem"])
        if old is None or p["image"].stat().st_size > old["image"].stat().st_size:
            pairs[p["stem"]] = p
    print(f"     可比对的原始标注 {len(pairs)} 份")

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = HERE / out_root

    done = skipped = 0
    index_rows = []
    for stem, items in sorted(miss.items(), key=lambda kv: -len(kv[1])):
        p = pairs.get(stem)
        if p is None:
            skipped += 1
            continue
        orig_img, orig_json = p["image"], p["json"]
        with Image.open(orig_img) as im:
            OW, OH = im.size
        ds = HERE / "dataset" / "images" / "train" / (stem + ".png")
        if not ds.exists():
            ds = HERE / "dataset" / "images" / "val" / (stem + ".png")
        if not ds.exists():
            skipped += 1
            continue
        with Image.open(ds) as im:
            DW, DH = im.size
        sc = OW / float(DW)

        base = json.loads(orig_json.read_text(encoding="utf-8"))
        orig_shapes = list(base.get("shapes") or [])
        shapes = list(orig_shapes) if args.with_orig else []
        # 建议框的类别名要跟着原文件走：你原始数据里 Tracker 那一类叫 "Typical"，
        # 那就用 "Typical?"，免得审完之后同一个文件里混着两套叫法。
        src_labels = {(s.get("label") or "").strip() for s in orig_shapes}
        tracker_name = "Typical" if "Typical" in src_labels else "Tracker"
        for r in items:
            x1, y1 = float(r["x1"]) * sc, float(r["y1"]) * sc
            x2, y2 = float(r["x2"]) * sc, float(r["y2"]) * sc
            lab = r["class"]
            if lab == "Tracker":
                lab = tracker_name

            shapes.append({
                # 加个问号：X-AnyLabeling 里是另一个颜色，一眼能看出是"待确认的建议框"。
                # prepare_yolo.py 里有映射，忘了去掉问号也不会影响训练。
                "label": lab + "?",
                "score": float(r["conf"]),
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": MARK,
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            })
        base["shapes"] = shapes
        base["imagePath"] = orig_img.name
        base["imageData"] = None
        base["imageWidth"] = OW
        base["imageHeight"] = OH
        base["checked"] = False

        sub = out_root / orig_img.parent.parent.name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / (stem + ".json")).write_text(
            json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        # 标注必须和同名图片放一起才打得开。硬链接：同一块盘不占额外空间。
        dst_img = sub / orig_img.name
        if not dst_img.exists():
            try:
                os.link(orig_img, dst_img)
            except Exception:
                shutil.copy2(orig_img, dst_img)
        done += 1
        index_rows.append({
            "图名": stem, "原始目录": orig_img.parent.parent.name,
            "原有框": len(base["shapes"]) - len(items), "补标候选": len(items),
            "类别": "/".join(sorted({r["class"] for r in items})),
            "文件": str((sub / (stem + ".json"))),
        })

    out_root.mkdir(parents=True, exist_ok=True)
    idx = out_root / "补标清单.csv"
    with idx.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["图名", "原始目录", "原有框", "补标候选", "类别", "文件"])
        w.writeheader()
        w.writerows(index_rows)
    print(f"[2/3] 写出 {done} 份 json（跳过 {skipped} 份）→ {out_root}")
    print(f"      清单：{idx}")

    (out_root / "怎么用.txt").write_text(
        "这些 json 里**只有模型建议补的框**（类别名带问号），每个 json 旁边配了同名 png。\n"
        "打开就是「只看见待确认的框」，不用在一堆原有框里找。\n"
        "\n"
        "用哪个工具打开：\n"
        "  ✔ X-AnyLabeling（就是你平时标注用的那个）\n"
        "  ✘ LBD 标注工具打不开 —— 它只认 agent3-debug 那种带 input_data.pages 的格式，\n"
        "    这类文件它会提示「这份 JSON 里没有页面」，属正常。\n"
        "\n"
        "怎么打开：X-AnyLabeling 里「打开目录」选对应的子文件夹（batch_006 这种），\n"
        "  它会自动配对同名的 png + json；也可以直接拖单张 png 进去。\n"
        "\n"
        "每一步做什么：\n"
        "  1) 打开某张图，只看带问号的框（类别名 Node? / Typical?，颜色和平时不同）\n"
        "  2) 对的留着、不对的删掉；**不用管问号**（合并时会自动去掉）\n"
        "  3) 存盘（只覆盖这个文件夹里的 json，碰不到你桌面上的原始标注）\n"
        "  4) 全部审完告诉我，我跑合并脚本：把留下的框并回你的原始标注，\n"
        "     输出到 补标结果\\ 目录，你再整体覆盖原始文件（覆盖前备份）\n"
        "\n"
        "注意：这里**不含**你原有的标注，所以不能直接拿它替换原文件（会丢框），\n"
        "     必须走上面第 4 步的合并。\n"
        "\n"
        "注意：目录名对应你桌面上的原始批次文件夹（batch_00x / A-F / G-Mhalf），\n"
        "文件名就是原来的图纸页名，方便你对号入座。\n"
        "\n"
        "优先级建议：先看补标候选最多的那几套图（见 补标清单.csv 排序）。\n",
        encoding="utf-8")
    print("[3/3] 同时写了一份 怎么用.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
