"""逐页核对 agent3 JSON 的页码是否和 PDF 对得上（全页覆盖，不抽样）。

做法：在低 DPI 下把每页切成格点
- 墨迹格点：格子里暗像素占比超过阈值（阵列/表格/标题栏都会有墨迹）
- 覆盖格点：JSON 的框盖到的格子
然后算双向指标：
    召回 recall = 墨迹格点里被框盖住的比例   （该框的地方有没有框）
    精确 prec   = 框盖住的格点里真有墨迹的比例（框有没有盖到空白）
    F1 = 两者调和平均
同一张图两次都高；页码错位时通常召回塌掉（阵列没被框住）或精确塌掉（框盖空白）。

对已知答案的数据集（页数一致、人工验证过）跑一遍，就能标定这个指标可不可信。

用法:
    python verify_page_mapping.py --pdf <pdf> --json <json> --cache <缓存目录> --out <报告csv>
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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


def ink_grid(png: Path, cell: int, dark: int, min_frac: float) -> np.ndarray:
    a = np.asarray(Image.open(png).convert("L"))
    H, W = a.shape
    gh, gw = H // cell, W // cell
    a = (a[:gh * cell, :gw * cell] < dark).astype(np.float32)
    frac = a.reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    return frac > min_frac


def cover_grid(spans: list[tuple[float, float, float, float]],
               W: int, H: int, cell: int, page_px: int) -> np.ndarray:
    gh, gw = H // cell, W // cell
    g = np.zeros((gh, gw), dtype=bool)
    k = W / page_px
    for x1, y1, x2, y2 in spans:
        c1 = max(0, int(x1 * k) // cell)
        c2 = min(gw - 1, int(x2 * k) // cell)
        r1 = max(0, int(y1 * k) // cell)
        r2 = min(gh - 1, int(y2 * k) // cell)
        if c2 >= c1 and r2 >= r1:
            g[r1:r2 + 1, c1:c2 + 1] = True
    return g


def score(ink: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    ni, nc = ink.sum(), cov.sum()
    if not ni or not nc:
        return 0.0, 0.0, 0.0
    inter = np.logical_and(ink, cov).sum()
    rec = inter / ni
    pre = inter / nc
    f1 = 2 * rec * pre / (rec + pre) if (rec + pre) else 0.0
    return rec, pre, f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--dpi", type=int, default=50)
    ap.add_argument("--cell", type=int, default=20)
    ap.add_argument("--dark", type=int, default=200)
    ap.add_argument("--ink", type=float, default=0.03)
    ap.add_argument("--page-size", type=int, default=8500)
    ap.add_argument("--search", type=int, default=8,
                    help="在同页号 ±这个范围内找最优匹配（0=不限范围，全页搜索）")
    args = ap.parse_args()

    pdf, js, cache = Path(args.pdf), Path(args.json), Path(args.cache)
    d = json.loads(js.read_text(encoding="utf-8"))
    spans_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for x in d.get("yolo_tracker_detection_results") or []:
        if not isinstance(x, dict):
            continue
        data = x.get("data")
        dets = data.get("detections") if isinstance(data, dict) else None
        if not dets:
            continue
        sp = []
        for dt in dets:
            bb = dt.get("bbox") or {}
            try:
                sp.append((float(bb["x1"]), float(bb["y1"]),
                           float(bb["x2"]), float(bb["y2"])))
            except (KeyError, TypeError, ValueError):
                pass
        if sp:
            spans_by_page[x["page_number"]] = sp

    files = render_all(pdf, cache, args.dpi)
    print(f"[渲染] {len(files)} 页")

    print("[指纹] 逐页算墨迹格点 ...")
    inks: dict[int, np.ndarray] = {}
    W0 = H0 = 0
    for i, f in enumerate(files, 1):
        inks[i] = ink_grid(f, args.cell, args.dark, args.ink)
        if W0 == 0:
            H0, W0 = inks[i].shape
            W0 *= args.cell
            H0 *= args.cell

    rows = []
    print(f"\n{'JSON页':>7} {'PDF同页F1':>10} {'最优PDF页':>10} {'该页F1':>8}  判定")
    ok = bad = 0
    for q in sorted(spans_by_page):
        cov = cover_grid(spans_by_page[q], W0, H0, args.cell, args.page_size)
        if q not in inks:
            print(f"{q:>7}   （PDF 没有这一页）")
            bad += 1
            continue
        rec, pre, f1_self = score(inks[q], cov)
        lo = max(1, q - args.search) if args.search else 1
        hi = min(len(files), q + args.search) if args.search else len(files)
        best_p, best_f1 = q, f1_self
        for p in range(lo, hi + 1):
            if p == q:
                continue
            r, pr, f = score(inks[p], cov)
            if f > best_f1:
                best_p, best_f1 = p, f
        # 判据：同页号是不是"最优匹配"。绝对 F1 天然偏低（框只盖阵列，盖不到
        # 标题栏/表格/说明，召回被这些压住），所以用相对比较才靠谱。
        flag = "OK" if best_p == q else f"!! 最优其实是第 {best_p} 页"
        if flag == "OK":
            ok += 1
        else:
            bad += 1
            print(f"{q:>7} {f1_self:>10.3f} {best_p:>10} {best_f1:>8.3f}  {flag}")
        rows.append((q, round(f1_self, 4), best_p, round(best_f1, 4), flag))

    print(f"\n汇总：{ok} 页对得上，{bad} 页存疑（上面的列表；没打印的都正常）")
    if args.out:
        with Path(args.out).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["json_page", "f1_self", "best_pdf_page", "best_f1", "verdict"])
            w.writerows(rows)
        print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
