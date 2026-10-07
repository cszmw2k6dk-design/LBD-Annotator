"""Highland：直接从 agent3 JSON 里内嵌的页面图转 YOLO 数据（不需要 PDF）。

为什么特殊：这份 JSON 的页面是 6000x4000 横版，而手头的 Highland PDF 是
24"x36" 竖版（1728x2592pt @250DPI = 6000x9000），几何对不上。好在 JSON 的
input_data.pages[].png_base64 里**内嵌了页面原图**，而 bbox 就是这张图的坐标，
所以直接解出来用，零页码风险。

用法:
    python prepare_highland.py --json <json> --out yolo26/agent3_src/highland [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = ["Node", "Tracker"]          # Box 那一类暂时不要（模型只有两类）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=2560)
    ap.add_argument("--min-boxes", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("载入 JSON ...")
    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    pages = {p["page_number"]: p for p in d["input_data"]["pages"]}

    dets: dict[int, list] = {}
    for x in d.get("yolo_tracker_detection_results") or []:
        data = x.get("data")
        dd = data.get("detections") if isinstance(data, dict) else None
        if dd:
            dets[x["page_number"]] = [t for t in dd if isinstance(t, dict)]

    # 坐标空间自检
    xs = [t["bbox"][k] for v in dets.values() for t in v
          if isinstance(t.get("bbox"), dict) for k in ("x1", "x2")]
    ys = [t["bbox"][k] for v in dets.values() for t in v
          if isinstance(t.get("bbox"), dict) for k in ("y1", "y2")]
    if xs:
        print(f"bbox 范围: x {min(xs):.0f}~{max(xs):.0f}  y {min(ys):.0f}~{max(ys):.0f}"
              f"  （页面 {pages[next(iter(dets))]['width']}x{pages[next(iter(dets))]['height']}）")

    labs = Counter(t.get("label") for v in dets.values() for t in v)
    print(f"可用页 {len(dets)}，框 {sum(labs.values())}，标签 {dict(labs)}")
    if args.dry_run:
        keep = {k: len(v) for k, v in dets.items() if len(v) >= args.min_boxes}
        print(f"[dry-run] 会转换 {len(keep)} 页（丢掉 {len(dets) - len(keep)} 页：框 < {args.min_boxes}）")
        return 0

    from PIL import Image

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rows, kept = [], 0
    for q in sorted(dets):
        det = dets[q]
        if len(det) < args.min_boxes:
            rows.append((q, len(det), 0, "跳过(框太少)"))
            continue
        pg = pages.get(q)
        if not pg or not pg.get("png_base64"):
            rows.append((q, len(det), 0, "无内嵌图"))
            continue
        im = Image.open(io.BytesIO(base64.b64decode(pg["png_base64"]))).convert("RGB")
        W0, H0 = im.size
        k = args.max_side / max(W0, H0)
        nw, nh = round(W0 * k), round(H0 * k)
        stem = f"highland_p{q:03d}"
        im.resize((nw, nh), Image.LANCZOS).save(out / "images" / f"{stem}.png", format="PNG")
        lines = []
        for t in det:
            lab = t.get("label")
            if lab not in CLASSES:
                continue
            bb = t.get("bbox") or {}
            try:
                x1, y1, x2, y2 = (float(bb[c]) for c in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                continue
            bw, bh = (x2 - x1) / W0, (y2 - y1) / H0
            cx, cy = (x1 + x2) / 2 / W0, (y1 + y2) / 2 / H0
            if bw <= 0 or bh <= 0:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (CLASSES.index(lab), cx, cy, bw, bh))
        (out / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        kept += 1
        rows.append((q, len(det), len(lines), "OK"))

    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["json_page", "json_boxes", "written_boxes", "status"])
        w.writerows(rows)
    print(f"[完成] highland: {kept} 页 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
