#!/usr/bin/env python
"""Generate a tiny synthetic dataset in the exact RDD2022 folder/VOC format
(cracks = random poly-lines, alligator = crack meshes, potholes = dark ellipses)
so CI and the smoke test can exercise the full pipeline without the 12 GB download.

    python scripts/make_synthetic_rdd.py --out data/synthetic/RDD2022 --n 60
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

XML = """<annotation><filename>{fn}</filename><size><width>{w}</width><height>{h}</height><depth>3</depth></size>{objs}</annotation>"""
OBJ = "<object><name>{n}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"


def road(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(90, 140)
    img = np.full((h, w, 3), base, np.uint8)
    img = cv2.add(img, rng.normal(0, 12, (h, w, 3)).clip(-40, 40).astype(np.int8), dtype=cv2.CV_8U)
    cv2.line(img, (w // 2, 0), (w // 2, h), (220, 220, 220), 4)
    return cv2.GaussianBlur(img, (3, 3), 0)


def draw(img, kind: str, rnd: random.Random):
    h, w = img.shape[:2]
    x, y = rnd.randint(20, w - 120), rnd.randint(20, h - 120)
    dark = (35, 35, 35)
    if kind == "D00":       # longitudinal: mostly vertical
        pts = [(x + rnd.randint(-6, 6), y + i * 12) for i in range(8)]
    elif kind == "D10":     # transverse: mostly horizontal
        pts = [(x + i * 14, y + rnd.randint(-5, 5)) for i in range(8)]
    elif kind == "D20":     # alligator: mesh
        for i in range(5):
            cv2.line(img, (x, y + i * 18), (x + 90, y + i * 18 + rnd.randint(-6, 6)), dark, 2)
            cv2.line(img, (x + i * 18, y), (x + i * 18 + rnd.randint(-6, 6), y + 90), dark, 2)
        return x, y, x + 96, y + 96
    else:                   # pothole
        ax, ay = rnd.randint(18, 40), rnd.randint(12, 28)
        cv2.ellipse(img, (x + ax, y + ay), (ax, ay), 0, 0, 360, (25, 25, 30), -1)
        return x, y, x + 2 * ax, y + 2 * ay
    cv2.polylines(img, [np.array(pts)], False, dark, 2)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3


def main(out: str, n: int, countries=("Japan", "India")) -> None:
    rnd, rng = random.Random(0), np.random.default_rng(0)
    for c in countries:
        idir = Path(out) / c / "train" / "images"; xdir = Path(out) / c / "train" / "annotations" / "xmls"
        idir.mkdir(parents=True, exist_ok=True); xdir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            w, h = 480, 360
            img = road(w, h, rng)
            objs = []
            for _ in range(rnd.randint(0, 3)):
                k = rnd.choice(["D00", "D10", "D20", "D40"])
                x1, y1, x2, y2 = draw(img, k, rnd)
                objs.append(OBJ.format(n=k, x1=max(0, x1), y1=max(0, y1), x2=min(w, x2), y2=min(h, y2)))
            if rnd.random() < 0.1:   # out-of-scope label to exercise cleaning
                objs.append(OBJ.format(n="D43", x1=5, y1=5, x2=40, y2=40))
            fn = f"{c}_{i:06d}"
            cv2.imwrite(str(idir / f"{fn}.jpg"), img)
            (xdir / f"{fn}.xml").write_text(XML.format(fn=fn + ".jpg", w=w, h=h, objs="".join(objs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/RDD2022")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    main(a.out, a.n)
