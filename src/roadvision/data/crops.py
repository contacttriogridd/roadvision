"""Build the crop-classification dataset for the ViT refiner.

For every ground-truth box we cut a context-padded crop (class D00/D10/D20/D40).
A fifth class, BG, teaches the ViT to reject false alarms:
  * random background patches that do not overlap any damage (IoU < 0.05), sized
    like real boxes, and
  * hard negatives — YOLO detections on the training split that match no ground
    truth (mined with `--detector`), i.e. exactly the errors the ViT must fix.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..utils.common import get_logger, seed_everything
from ..utils.config import load_config

log = get_logger("crops")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def padded_crop(img: np.ndarray, box, pad: float, min_side: int = 24) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, min_side), max(y2 - y1, min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    xa, ya = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    xb, yb = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    return img[ya:yb, xa:xb]


def build_crops(cfg, detector: str | None = None) -> Path:
    seed_everything(cfg.project.seed)
    rng = random.Random(cfg.project.seed)
    proc = Path(cfg.data.processed_dir)
    out = Path(cfg.data.crops_dir)
    names = list(cfg.data.classes) + ["BG"]
    imgs = pd.read_csv(proc / "images.csv")
    boxes = pd.read_csv(proc / "boxes.csv")
    groups = {k: g for k, g in boxes.groupby("stem")}
    sizes = boxes[["w", "h"]].to_numpy() if len(boxes) else np.array([[64, 64]])
    yolo = None
    if detector and Path(detector).exists():
        from ultralytics import YOLO

        yolo = YOLO(detector)
        log.info("Mining hard negatives with %s", detector)
    rows = []
    for r in imgs.itertuples():
        img = cv2.imread(str(proc / "images" / r.split / f"{r.stem}.jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        g = groups.get(r.stem)
        gt = g[["x1", "y1", "x2", "y2"]].to_numpy() if g is not None else np.zeros((0, 4))
        d = out / r.split
        if g is not None:
            for i, b in enumerate(g.itertuples()):
                c = padded_crop(img, (b.x1, b.y1, b.x2, b.y2), cfg.vit.context_pad)
                if c.size:
                    p = d / b.label / f"{r.stem}_{i}.jpg"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(p), c)
                    rows.append((str(p), r.split, b.label, "gt"))
        negs = []
        if yolo is not None and r.split == "train":
            res = yolo.predict(img, conf=0.1, verbose=False, imgsz=cfg.yolo.imgsz)[0]
            pb = res.boxes.xyxy.cpu().numpy()
            if len(pb):
                ok = iou_xyxy(pb, gt).max(1) < 0.3 if len(gt) else np.ones(len(pb), bool)
                negs += [("hard", tuple(b)) for b in pb[ok][: cfg.vit.neg_per_image]]
        tries = 0
        while len(negs) < cfg.vit.neg_per_image * (2 if yolo is None else 1) and tries < 30:
            tries += 1
            w, h = sizes[rng.randrange(len(sizes))]
            w, h = min(w, W - 1), min(h, H - 1)
            x1, y1 = rng.uniform(0, W - w), rng.uniform(0, H - h)
            cand = np.array([[x1, y1, x1 + w, y1 + h]])
            if len(gt) == 0 or iou_xyxy(cand, gt).max() < 0.05:
                negs.append(("random", tuple(cand[0])))
        for j, (kind, b) in enumerate(negs):
            c = padded_crop(img, b, cfg.vit.context_pad)
            if c.size:
                p = d / "BG" / f"{r.stem}_{kind}{j}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), c)
                rows.append((str(p), r.split, "BG", kind))
    df = pd.DataFrame(rows, columns=["path", "split", "label", "source"])
    df["y"] = df.label.map({n: i for i, n in enumerate(names)})
    df.to_csv(out / "crops.csv", index=False)
    log.info("Crops: %s", df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict())
    return out / "crops.csv"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--detector", default=None, help="YOLO weights for hard-negative mining")
    a = ap.parse_args()
    build_crops(load_config(a.config, a.override), a.detector)
