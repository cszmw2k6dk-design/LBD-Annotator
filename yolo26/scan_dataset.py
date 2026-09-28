"""扫描 X-AnyLabeling json + png 数据集，输出统计信息。

用法:  python scan_dataset.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 标注原件（X-AnyLabeling 的 png + json）在哪个目录，每台机器不一样：
# 默认是本机这套，换机器不用改文件，设环境变量 LBD_ROOTS 就行（多个目录用 ; 分隔）。
_DEFAULT_ROOTS = [
    r"C:\Users\ZhaokeShi\Downloads\标注工作\标注支架\batch_001",
    r"C:\Users\ZhaokeShi\Downloads\标注工作\标注支架\batch_002",
    r"C:\Users\ZhaokeShi\Downloads\标注工作\标注支架\batch_003",
    r"C:\Users\ZhaokeShi\Downloads\标注工作\标注支架\batch_004",
    r"C:\Users\ZhaokeShi\Downloads\标注工作\标注支架\batch_005",
    r"C:\Users\ZhaokeShi\Downloads\标注工作\标注支架\batch_006",
    r"C:\Users\ZhaokeShi\Downloads\标注工作\标注支架\batch_007",
]
ROOTS = ([p for p in os.environ["LBD_ROOTS"].split(os.pathsep) if p]
         if os.environ.get("LBD_ROOTS") else _DEFAULT_ROOTS)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PAGE_RE = re.compile(r"_p0*\d+$", re.IGNORECASE)


def iter_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def main() -> int:
    stems: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    per_root = Counter()

    for r in ROOTS:
        root = Path(r)
        if not root.is_dir():
            print(f"[跳过] 目录不存在: {root}")
            continue
        n_img = n_json = 0
        for f in iter_files(root):
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                n_img += 1
                stems[f.stem]["image"].append(f)
            elif ext == ".json":
                n_json += 1
                stems[f.stem]["json"].append(f)
        per_root[root.name] = (n_img, n_json)
        print(f"[扫描] {root.name:<10} png={n_img:<6} json={n_json:<6}")

    paired = [s for s, d in stems.items() if d["image"] and d["json"]]
    img_only = [s for s, d in stems.items() if d["image"] and not d["json"]]
    json_only = [s for s, d in stems.items() if d["json"] and not d["image"]]

    print()
    print(f"唯一图片名: {sum(1 for d in stems.values() if 'image' in d)}")
    print(f"唯一 json : {sum(1 for d in stems.values() if 'json' in d)}")
    print(f"成对      : {len(paired)}")
    print(f"只有图无标注: {len(img_only)}")
    print(f"只有标注无图: {len(json_only)}")
    if json_only:
        for s in json_only[:20]:
            print(f"    ! {s}")

    # --- 同名文件（跨目录重复）分析 ---
    dup_img = {s: d["image"] for s, d in stems.items() if len(d["image"]) > 1}
    dup_json = {s: d["json"] for s, d in stems.items() if len(d["json"]) > 1}
    print()
    print(f"跨目录同名图片: {len(dup_img)} 组 (多出 {sum(len(v) - 1 for v in dup_img.values())} 个副本)")
    print(f"跨目录同名标注: {len(dup_json)} 组 (多出 {sum(len(v) - 1 for v in dup_json.values())} 个副本)")

    same_size = diff_size = 0
    size_examples: list[str] = []
    for s, paths in dup_img.items():
        sizes = {p.stat().st_size for p in paths}
        if len(sizes) == 1:
            same_size += 1
        else:
            diff_size += 1
            if len(size_examples) < 8:
                size_examples.append(
                    f"{s}: " + ", ".join(f"{p.stat().st_size // 1024}KB[{p.parent.parent.name}]" for p in paths))
    print(f"  其中内容大小相同: {same_size} 组，大小不同(=不同分辨率): {diff_size} 组")
    for e in size_examples:
        print(f"    - {e}")

    # 同名标注是否一致
    json_same = json_diff = 0
    for s, paths in dup_json.items():
        texts = {p.read_text(encoding="utf-8") for p in paths}
        if len(texts) == 1:
            json_same += 1
        else:
            json_diff += 1
    print(f"  同名标注内容相同: {json_same} 组，内容不同: {json_diff} 组")

    # --- 图片尺寸抽样 ---
    try:
        from PIL import Image  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        Image = None
    if Image is not None:
        print()
        print("图片尺寸抽样 (每个目录 2 张):")
        for r in ROOTS:
            root = Path(r)
            if not root.is_dir():
                continue
            found = 0
            for f in iter_files(root):
                if f.suffix.lower() not in IMAGE_EXTS:
                    continue
                try:
                    with Image.open(f) as im:
                        print(f"  {root.name:<10} {im.size[0]}x{im.size[1]}  {f.name[:60]}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  {root.name:<10} 读取失败 {f.name[:40]}: {exc}")
                found += 1
                if found >= 2:
                    break

    label_counter: Counter = Counter()
    shape_types: Counter = Counter()
    empty_json: list[str] = []
    boxes_per_image: Counter = Counter()
    bad: list[str] = []
    groups: dict[str, int] = defaultdict(int)

    for s in paired:
        jp = stems[s]["json"][0]
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{jp}: {exc}")
            continue
        shapes = data.get("shapes") or []
        if not shapes:
            empty_json.append(s)
        boxes_per_image[len(shapes)] += 1
        for sh in shapes:
            label_counter[sh.get("label")] += 1
            shape_types[sh.get("shape_type")] += 1
        groups[PAGE_RE.sub("", s)] += 1

    print()
    print("类别分布 (框数):")
    for name, cnt in label_counter.most_common():
        print(f"  {name!r:<20} {cnt}")
    print()
    print("shape_type 分布:")
    for name, cnt in shape_types.most_common():
        print(f"  {name!r:<20} {cnt}")
    print()
    print(f"空标注 json: {len(empty_json)}")
    for s in empty_json[:10]:
        print(f"    - {s}")
    if bad:
        print()
        print(f"读取失败: {len(bad)}")
        for b in bad[:10]:
            print(f"    - {b}")
    print()
    print(f"总框数: {label_counter.total()}")
    print(f"每图框数 (抽样): " + ", ".join(
        f"{k}框x{v}" for k, v in sorted(boxes_per_image.items())[:8]))
    print(f"图名分组数(按 _pNN 归类): {len(groups)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
