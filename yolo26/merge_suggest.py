r"""把审完的补标建议并回原始标注。

输入：补标建议\<批次>\<页名>.json（里面只剩你留下的建议框，类别名带问号）
输出：补标结果\<批次>\<页名>.json（原始标注 + 你留下的建议框，问号已去掉）

原始标注只读不写；输出的文件由你自己决定要不要覆盖回去。

用法: python merge_suggest.py [--src 补标建议] [--out 补标结果]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_yolo import collect_pairs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="补标建议")
    ap.add_argument("--out", default="补标结果")
    ap.add_argument("--roots", default=None,
                    help="原始标注目录（逗号分隔）。不传就用内置默认值；"
                         "本机默认值指向公司电脑的路径，所以要在这台机器上合并时必须显式传。")
    args = ap.parse_args()

    src_root = Path(args.src)
    if not src_root.is_absolute():
        src_root = HERE / src_root
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = HERE / out_root

    roots = [r.strip() for r in args.roots.split(",") if r.strip()] if args.roots else None
    pairs = {}
    for p in collect_pairs(roots):
        old = pairs.get(p["stem"])
        if old is None or p["image"].stat().st_size > old["image"].stat().st_size:
            pairs[p["stem"]] = p

    rows = []
    kept_total = dropped_total = 0
    stray_total = 0
    files = sorted(f for f in src_root.rglob("*.json") if f.name != "补标清单.csv")
    for f in files:
        stem = f.stem
        p = pairs.get(stem)
        if p is None:
            print(f"[跳过] 找不到原始标注：{stem}")
            continue
        reviewed = json.loads(f.read_text(encoding="utf-8"))
        sug, stray = [], 0
        for s in (reviewed.get("shapes") or []):
            lab = (s.get("label") or "").strip()
            # 只合并"建议框"：带问号，或者还留着 model_suggest 标记的。
            # 其它形状一律不并（防止有人把原始标注文件误拷进这个文件夹，
            # 那样会把同一批框重复加两遍）。
            if not (lab.endswith("?") or s.get("description") == "model_suggest"):
                stray += 1
                continue
            sug.append(s)
        # 建议包里现在会自带一份"原始标注"（方便在标注工具里对照，实线显示），
        # 那些框本来就不该再并一次 —— 跳过即可，别每页都刷一行提醒。
        stray_total += stray
        for s in sug:                      # 去掉问号，变成正式类别
            lab = (s.get("label") or "").strip()
            if lab.endswith("?"):
                s["label"] = lab[:-1].strip()
            s["description"] = ""
        base = json.loads(Path(p["json"]).read_text(encoding="utf-8"))
        orig = list(base.get("shapes") or [])
        base["shapes"] = orig + sug
        base["checked"] = False

        sub = out_root / f.parent.name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f.name).write_text(json.dumps(base, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        kept_total += len(sug)
        rows.append({"批次": f.parent.name, "页名": stem,
                     "原有框": len(orig), "留下的建议框": len(sug),
                     "合并后": len(orig) + len(sug),
                     "输出": str(sub / f.name)})

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "合并清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["批次", "页名", "原有框", "留下的建议框",
                                           "合并后", "输出"])
        w.writeheader()
        w.writerows(rows)

    print(f"[完成] 处理 {len(rows)} 份，留下的建议框共 {kept_total} 个")
    if stray_total:
        print(f"       （另有 {stray_total} 个框是包自带的原始标注，已按规则跳过，不会重复合并）")
    print(f"       输出目录：{out_root}")
    print(f"       清单：{out_root / '合并清单.csv'}")
    print("\n下一步：确认无误后，把这些 json 拷回原始标注目录覆盖同名文件（先备份原始 json）。")
    print("       只拷 json，不要拷 png。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
