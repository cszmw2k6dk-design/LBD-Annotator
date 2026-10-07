"""核对 agent3 JSON 的页码/坐标能不能对上前端 PDF：把框直接画到 PDF 页上。

为什么需要它：这类 JSON 里的 bbox 是「某一版 PDF 的某个页码 + 250 DPI 像素坐标」，
换一份 PDF（不同交付批次、不同修订）页码就会错位，训练数据会被污染。
肉眼看一眼叠加图，是唯一可靠的确认方式。

用法:
    python check_pdf_json_pages.py --pdf <pdf> --json <json> --pages 1,16,20
    python check_pdf_json_pages.py --pdf <pdf> --json <json> --scan 12   # 抽样自动打分
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDFTOPPM = Path(r"C:\Users\szk\Desktop\LBD-biaozhu\dist\poppler\Library\bin\pdftoppm.exe")
OUTDIR = HERE / "preview"


def detections(json_path: Path) -> dict[int, list[dict]]:
    """page_number -> [ {label, bbox}, ... ]，取 tracker 那一路的结果。"""
    d = json.loads(json_path.read_text(encoding="utf-8"))
    by_page: dict[int, list[dict]] = {}
    for x in d.get("yolo_tracker_detection_results") or []:
        if not isinstance(x, dict):
            continue
        pn = x.get("page_number")
        data = x.get("data")
        dets = data.get("detections") if isinstance(data, dict) else None
        if not isinstance(pn, int) or not dets:
            continue
        by_page[pn] = dets
    return by_page


def render(pdf: Path, page: int, dpi: int, tmp: Path) -> Path | None:
    prefix = tmp / f"p{page}"
    cmd = [str(PDFTOPPM), "-f", str(page), "-l", str(page), "-r", str(dpi),
           "-png", str(pdf), str(prefix)]
    subprocess.run(cmd, check=True, capture_output=True)
    got = sorted(tmp.glob(f"p{page}*.png"))
    return got[0] if got else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--pages", default="")
    ap.add_argument("--scan", type=int, default=0, help="抽样这么多页自动打分（看有没有错位页）")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--width", type=int, default=1800)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    pdf, js = Path(args.pdf), Path(args.json)
    if not pdf.exists() or not js.exists():
        print(f"[错误] 文件不存在: {pdf if not pdf.exists() else js}", file=sys.stderr)
        return 2

    by_page = detections(js)
    have = sorted(by_page)
    print(f"JSON 有框的页: {len(have)} 页，页码 {min(have)}..{max(have)}")
    print(f"   框数合计: {sum(len(v) for v in by_page.values())}")

    pages = ([int(p) for p in re.split(r"[,\s]+", args.pages) if p]
             if args.pages else have[:2])
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if args.scan:
            # 抽 --scan 页（尽量均匀铺满整个范围），算"框内墨迹占比"：
            # 对得上 -> 框里是阵列的斜线/灰底，占比高；错位 -> 框落在空白处，接近 0。
            import statistics
            step = max(1, len(have) // args.scan)
            sample = have[::step][:args.scan]
            print(f"\n抽样 {len(sample)} 页打分（框内墨迹占比，越大越可能是对的页）:")
            print(f"   {'页码':>6} {'框数':>6} {'框内墨迹中位数':>14} {'整页墨迹占比':>12}")
            scores = []
            for page in sample:
                png = render(pdf, page, args.dpi, tmp)
                if png is None:
                    print(f"   {page:>6}  渲染失败")
                    continue
                im = Image.open(png).convert("L")
                W, H = im.size
                px = im.load()
                total_dark = 0
                for y in range(0, H, 7):
                    for x in range(0, W, 7):
                        if px[x, y] < 200:
                            total_dark += 1
                page_dark = total_dark * 49 / (W * H)
                fr = []
                for dt in by_page.get(page, []):
                    bb = dt.get("bbox") or {}
                    try:
                        x1, y1 = max(0, int(bb["x1"])), max(0, int(bb["y1"]))
                        x2, y2 = min(W, int(bb["x2"])), min(H, int(bb["y2"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    d = n = 0
                    for y in range(y1, y2, 7):
                        for x in range(x1, x2, 7):
                            n += 1
                            if px[x, y] < 200:
                                d += 1
                    if n:
                        fr.append(d / n)
                med = statistics.median(fr) if fr else float("nan")
                scores.append((page, len(fr), med))
                print(f"   {page:>6} {len(fr):>6} {med:>14.3f} {page_dark:>12.3f}")
            good = [s for s in scores if s[2] == s[2] and s[2] > 0.15]
            print(f"\n   -> {len(good)}/{len(scores)} 页看起来对得上"
                  f"（阈值 0.15；明显偏低的页就是页码错位）")
            return 0
        for page in pages:
            png = render(pdf, page, args.dpi, tmp)
            if png is None:
                print(f"  [页 {page}] 渲染失败（页码超出 PDF 范围？）")
                continue
            im = Image.open(png).convert("RGB")
            W, H = im.size
            k = args.width / W
            small = im.resize((args.width, round(H * k)), Image.LANCZOS)
            dr = ImageDraw.Draw(small)
            dets = by_page.get(page, [])
            for dt in dets:
                bb = dt.get("bbox") or {}
                try:
                    x1, y1, x2, y2 = (float(bb["x1"]), float(bb["y1"]),
                                      float(bb["x2"]), float(bb["y2"]))
                except (KeyError, TypeError, ValueError):
                    continue
                col = (255, 0, 0) if (dt.get("label") == "Node") else (0, 110, 255)
                dr.rectangle([x1 * k, y1 * k, x2 * k, y2 * k], outline=col, width=2)
            out = OUTDIR / f"pagecheck_{pdf.stem[:24]}_p{page}.png".replace(" ", "_")
            small.save(out)
            print(f"  [页 {page}] PDF 渲染 {W}x{H}，JSON 该页 {len(dets)} 个框 -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
