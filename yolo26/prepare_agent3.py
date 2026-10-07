"""把 agent3-debug-v1 的 JSON（+对应 PDF）转成 YOLO 训练数据。

和 prepare_yolo.py 的区别：那边的源是现成的 png+json 对，这边只有 PDF + JSON，
JSON 里的 bbox 是"某页 250 DPI 渲染后的像素坐标"，所以要自己渲染并对齐页码。

关键点（本项目的血泪）：
- **页码要加偏移**：Bigway 的 JSON 页号比 PDF 少 3（用户确认 + 定量验证）。
- **渲染后要平滑缩放**，不要直接把原图丢给 ultralytics——它内部用 cv2 INTER_LINEAR
  硬缩会把 Tracker 的细线吃掉（历史 bug：召回 0.68→0.44）。
- 页面尺寸按 JSON 里的 width/height 换算，别硬编码 8500×5500。

用法:
    python prepare_agent3.py --pdf <pdf> --json <json> --name steelriver \
        --offset 0 --out yolo26/agent3_src/steelriver --min-boxes 20
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDFTOPPM = Path(r"C:\Users\szk\Desktop\LBD-biaozhu\dist\poppler\Library\bin\pdftoppm.exe")
CLASSES = ["Node", "Tracker"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=0, help="PDF 页号 = JSON 页号 + 偏移")
    ap.add_argument("--max-side", type=int, default=2560)
    ap.add_argument("--dpi", type=int, default=250, help="先按这个 DPI 渲染（JSON 就是 250）")
    ap.add_argument("--min-boxes", type=int, default=20, help="框少于这个数的页丢掉（防误检页）")
    ap.add_argument("--page-size", type=int, default=8500)
    args = ap.parse_args()

    from PIL import Image

    pdf, js, out = Path(args.pdf), Path(args.json), Path(args.out)
    d = json.loads(js.read_text(encoding="utf-8"))

    # 每页的框
    by: dict[int, list] = {}
    for x in d.get("yolo_tracker_detection_results") or []:
        if not isinstance(x, dict):
            continue
        data = x.get("data")
        dets = data.get("detections") if isinstance(data, dict) else None
        if dets:
            by[x["page_number"]] = [t for t in dets if isinstance(t, dict)]

    # 页面尺寸（JSON 自己记的），按页号取
    sizes: dict[int, tuple[int, int]] = {}
    for p in (d.get("input_data") or {}).get("pages") or []:
        if isinstance(p, dict) and "page_number" in p and p.get("width"):
            sizes[p["page_number"]] = (int(p["width"]), int(p["height"]))

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    rows = []
    kept = 0
    with tempfile.TemporaryDirectory() as td:
        for q in sorted(by):
            dets = by[q]
            if len(dets) < args.min_boxes:
                rows.append((q, q + args.offset, len(dets), 0, "跳过(框太少)"))
                continue
            p = q + args.offset
            subprocess.run([str(PDFTOPPM), "-f", str(p), "-l", str(p), "-r", str(args.dpi),
                            "-png", str(pdf), str(Path(td) / f"r{q}")],
                           check=True, capture_output=True)
            got = sorted(Path(td).glob(f"r{q}*.png"))
            if not got:
                rows.append((q, p, len(dets), 0, "PDF无此页"))
                continue
            with Image.open(got[0]) as im:
                im = im.convert("RGB")
                W0, H0 = im.size
                Wj, Hj = sizes.get(q, (args.page_size, int(args.page_size * H0 / W0)))
                k = args.max_side / max(W0, H0)
                nw, nh = round(W0 * k), round(H0 * k)
                stem = f"{args.name}_p{p:03d}"
                im.resize((nw, nh), Image.LANCZOS).save(
                    out / "images" / f"{stem}.png", format="PNG")
            # 像素 -> 归一化；JSON 坐标基于 Wj x Hj，先缩放到渲染尺寸再归一化
            sx, sy = W0 / Wj, H0 / Hj
            lines = []
            for dt in dets:
                lab = dt.get("label")
                if lab not in CLASSES:
                    continue
                bb = dt.get("bbox") or {}
                try:
                    x1, y1, x2, y2 = (float(bb[t]) for t in ("x1", "y1", "x2", "y2"))
                except (KeyError, TypeError, ValueError):
                    continue
                x1, x2 = x1 * sx, x2 * sx
                y1, y2 = y1 * sy, y2 * sy
                bw, bh = (x2 - x1) / W0, (y2 - y1) / H0
                cx, cy = (x1 + x2) / 2 / W0, (y1 + y2) / 2 / H0
                if bw <= 0 or bh <= 0:
                    continue
                lines.append("%d %.6f %.6f %.6f %.6f" % (CLASSES.index(lab), cx, cy, bw, bh))
            (out / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            kept += 1
            rows.append((q, p, len(dets), len(lines), "OK"))
            for f in got:
                f.unlink()

    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["json_page", "pdf_page", "json_boxes", "written_boxes", "status"])
        w.writerows(rows)
    print(f"[完成] {args.name}: 转换 {kept} 页 -> {out}")
    print(f"       跳过 {sum(1 for r in rows if r[4] != 'OK')} 页（框太少/无此页）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
