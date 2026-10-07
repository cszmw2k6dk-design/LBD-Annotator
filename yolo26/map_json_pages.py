"""给 agent3 JSON 的页码和目标 PDF 做"指纹匹配"，解决页码错位问题。

背景：JSON 里的 bbox 是「某一版 PDF 的页码 + 250 DPI 像素坐标」。换一份 PDF
（不同交付批次、页数不同）后页码会漂，直接按页号套就会把框画到别的图上。

思路：阵列布置图有大量"通高竖线"（板条），JSON 的框正好框住这些竖线。
所以拿"竖线位置落在框内 x 范围的比例"当指纹 —— 同一张图得分会明显最高。
先渲染低 DPI（默认 50）做粗筛，快且够用。

用法:
    python map_json_pages.py --pdf <pdf> --json <json> --cache <临时渲染目录>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDFTOPPM = Path(r"C:\Users\szk\Desktop\LBD-biaozhu\dist\poppler\Library\bin\pdftoppm.exe")


def render_all(pdf: Path, cache: Path, dpi: int) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(cache.glob("pg-*.png"))
    if not files:
        print(f"[渲染] {pdf.name} @{dpi}DPI -> {cache}")
        subprocess.run([str(PDFTOPPM), "-r", str(dpi), "-gray", "-png",
                        str(pdf), str(cache / "pg")], check=True, capture_output=True)
        files = sorted(cache.glob("pg-*.png"))
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--cache", required=True, help="低 DPI 渲染缓存目录")
    ap.add_argument("--dpi", type=int, default=50)
    ap.add_argument("--page-size", type=int, default=8500, help="JSON 坐标的页宽像素")
    ap.add_argument("--probe", default="20,35,100,157", help="已知答案的页，用来验证匹配是否可信")
    args = ap.parse_args()

    from PIL import Image

    pdf, js, cache = Path(args.pdf), Path(args.json), Path(args.cache)
    d = json.loads(js.read_text(encoding="utf-8"))
    by_page: dict[int, list[tuple[float, float]]] = {}
    for x in d.get("yolo_tracker_detection_results") or []:
        if not isinstance(x, dict):
            continue
        data = x.get("data")
        dets = data.get("detections") if isinstance(data, dict) else None
        if not dets:
            continue
        spans = []
        for dt in dets:
            bb = dt.get("bbox") or {}
            try:
                spans.append((float(bb["x1"]), float(bb["x2"])))
            except (KeyError, TypeError, ValueError):
                pass
        if spans:
            by_page[x["page_number"]] = spans

    files = render_all(pdf, cache, args.dpi)
    print(f"[渲染] 共 {len(files)} 页")

    print("[指纹] 统计每页的通高竖线位置 ...")
    lines: list[set[int]] = []
    for f in files:
        im = Image.open(f).convert("L")
        W, H = im.size
        px = im.load()
        thr = H * 0.25
        cols = set()
        for x in range(W):
            c = 0
            for y in range(0, H, 2):
                if px[x, y] < 210:
                    c += 1
            if c * 2 > thr:
                cols.add(x)
        lines.append(cols)
    W0 = lines and max(max(c) for c in lines if c) + 1 or 0
    print(f"[指纹] 页宽 {W0}px（@{args.dpi}DPI）")

    def score(es_page: int, json_page: int) -> float:
        """竖线落在 JSON 框 x 范围内的比例 - 覆盖过多的惩罚。"""
        cols = lines[es_page - 1]
        if not cols:
            return -1.0
        spans = by_page.get(json_page) or []
        k = W0 / args.page_size
        inside = 0
        for c in cols:
            for x1, x2 in spans:
                if x1 * k <= c <= x2 * k:
                    inside += 1
                    break
        cover = sum(1 for x in range(W0)
                    if any(x1 * k <= x <= x2 * k for x1, x2 in spans)) / W0
        return inside / len(cols) - 0.5 * cover

    probes = [int(p) for p in args.probe.split(",") if p.strip()]
    print("\n[验证] 已知页的最优匹配（JSON页 -> PDF页 得分 / 原页得分）:")
    for q in probes:
        if q not in by_page:
            continue
        scores = [(score(p, q), p) for p in range(1, len(files) + 1)]
        scores.sort(reverse=True)
        best, bp = scores[0]
        same = dict((p, s) for s, p in scores).get(q, float("nan"))
        print(f"   JSON 第 {q:>3} 页 -> PDF 第 {bp:>3} 页（得分 {best:.3f}）；"
              f"若按同页号则是 PDF 第 {q} 页（得分 {same:.3f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
