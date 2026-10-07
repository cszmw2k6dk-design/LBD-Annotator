"""按给定偏移逐页验证 agent3 JSON 与 PDF 的对应关系（全页，不抽样）。

判据（已在 Steel River 上标定：正确配对 0~1% 的框内没有竖线，错配 4~35%）：
    对每个 Tracker 框，数它内部"贯穿框高 50% 以上"的竖线列数。
    板条是竖条，框对了里面就有 1~3 条；框错了就落在空白处 -> 0 条。
    统计"0 条"的占比，超过阈值就说明这一页对不上。

用法:
    python verify_page_offset.py --pdf <pdf> --json <json> --offset 3 --out report.csv
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--offset", type=int, default=0, help="PDF 页号 = JSON 页号 + 这个偏移")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--page-size", type=int, default=8500)
    ap.add_argument("--bad", type=float, default=0.05, help="0 条占比超过它就算可疑")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from PIL import Image

    pdf, js = Path(args.pdf), Path(args.json)
    d = json.loads(js.read_text(encoding="utf-8"))
    by: dict[int, list] = {}
    for x in d.get("yolo_tracker_detection_results") or []:
        if not isinstance(x, dict):
            continue
        data = x.get("data")
        dets = data.get("detections") if isinstance(data, dict) else None
        if dets:
            by[x["page_number"]] = [dt for dt in dets if isinstance(dt, dict)]

    rows = []
    bad_pages = []
    with tempfile.TemporaryDirectory() as td:
        for q in sorted(by):
            p = q + args.offset
            subprocess.run([str(PDFTOPPM), "-f", str(p), "-l", str(p), "-r", str(args.dpi),
                            "-gray", "-png", str(pdf), str(Path(td) / f"v{p}")],
                           check=True, capture_output=True)
            got = sorted(Path(td).glob(f"v{p}*.png"))
            if not got:
                rows.append((q, p, 0, 0.0, 0.0, 0.0, "PDF无此页"))
                bad_pages.append((q, p, "PDF无此页"))
                continue
            im = Image.open(got[0]).convert("L")
            W, H = im.size
            px = im.load()
            k = W / args.page_size
            stats = []
            for dt in by[q]:
                if dt.get("label") != "Tracker":
                    continue
                bb = dt.get("bbox") or {}
                try:
                    x1, y1, x2, y2 = (float(bb[t]) for t in ("x1", "y1", "x2", "y2"))
                except (KeyError, TypeError, ValueError):
                    continue
                X1, Y1, X2, Y2 = int(x1 * k), int(y1 * k), int(x2 * k), int(y2 * k)
                if X2 - X1 < 3 or Y2 - Y1 < 8:
                    continue
                h = Y2 - Y1
                n = 0
                for X in range(X1, X2 + 1):
                    c = sum(1 for Y in range(Y1, Y2 + 1, 2) if px[X, Y] < 200) * 2
                    if c > h * 0.5:
                        n += 1
                stats.append(n)
            if not stats:
                rows.append((q, p, 0, 0.0, 0.0, 0.0, "无可用框"))
                continue
            zero = sum(1 for v in stats if v == 0) / len(stats)
            good = sum(1 for v in stats if 1 <= v <= 3) / len(stats)
            flag = "OK" if zero <= args.bad else "!! 可疑"
            rows.append((q, p, len(stats), 0.0, round(zero, 4), round(good, 4), flag))
            if flag != "OK":
                bad_pages.append((q, p, f"0条占比 {zero * 100:.1f}%"))
            for f in got:
                f.unlink()

    print(f"共核对 {len(rows)} 页（偏移 {args.offset:+d}）")
    if bad_pages:
        print("可疑页：")
        for q, p, why in bad_pages:
            print(f"   JSON {q} -> PDF {p}: {why}")
    else:
        print("全部通过：每页的 Tracker 框内都有竖线，没有落在空白的")
    if args.out:
        with Path(args.out).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["json_page", "pdf_page", "boxes", "x", "zero_pct", "hit_pct", "verdict"])
            w.writerows(rows)
        print(f"报告: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
