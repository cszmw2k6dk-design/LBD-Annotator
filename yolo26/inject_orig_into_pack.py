"""把原始标注并进补标包（让标注工具里能同时看到"原有框"和"建议框"）。

补标包的 json 本来只装模型建议补的框（带问号）。这个脚本按页名找到桌面上的原始标注
（同名 json，多份时取图片最大的那份，和 export_missing.py 的规则一致），把原始 shapes
放到建议框前面，写回补标包里的同名 json。

颜色区分由标注工具负责：原有框（标签没有问号）画实线、按类别配色；
建议框（标签带问号 / description=model_suggest）画虚线、按置信度配色。

用法:
    python inject_orig_into_pack.py --pack 补标建议_v4c --roots "C:\\...\\batch_003,..."
    python inject_orig_into_pack.py --pack 补标建议_v4c --roots "..." --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_yolo import collect_pairs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="补标包目录")
    ap.add_argument("--roots", default="",
                    help="原始标注目录，逗号分隔（不传就用内置默认值）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pack = Path(args.pack)
    if not pack.is_absolute():
        pack = HERE / pack
    roots = [r.strip() for r in args.roots.split(",") if r.strip()] or None

    pairs = {}
    for p in collect_pairs(roots):
        old = pairs.get(p["stem"])
        if old is None or p["image"].stat().st_size > old["image"].stat().st_size:
            pairs[p["stem"]] = p
    print(f"[1/2] 可配对的原始标注 {len(pairs)} 份")

    n_page = n_orig = n_sug = n_miss = 0
    for js in sorted(pack.rglob("*.json")):
        src = pairs.get(js.stem)
        if src is None:
            n_miss += 1
            continue
        doc = json.loads(js.read_text(encoding="utf-8"))
        shapes = doc.get("shapes") or []
        sug = [s for s in shapes if (s.get("label") or "").strip().endswith("?")
               or s.get("description") == "model_suggest"]
        base = json.loads(Path(src["json"]).read_text(encoding="utf-8"))
        orig = [s for s in (base.get("shapes") or [])
                if not (s.get("label") or "").strip().endswith("?")
                and s.get("description") != "model_suggest"]
        doc["shapes"] = orig + sug
        doc["checked"] = False
        if not args.dry_run:
            tmp = str(js) + ".part"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            Path(tmp).replace(js)
        n_page += 1
        n_orig += len(orig)
        n_sug += len(sug)
    print(f"[2/2] {'(演练)' if args.dry_run else '已写回'} {n_page} 页："
          f"原有框 {n_orig} + 建议框 {n_sug}；找不到原始标注 {n_miss} 页")
    return 0


if __name__ == "__main__":
    sys.exit(main())
