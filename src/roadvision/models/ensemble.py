"""Weighted Boxes Fusion (Solovyev et al., 2021) for a YOLO + RT-DETR ensemble.

Unlike NMS, WBF averages the coordinates of overlapping boxes from different
models (weighted by confidence x model weight) instead of discarding them, and
down-weights boxes that only some of the models agree on.
"""
from __future__ import annotations

import numpy as np


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def weighted_boxes_fusion(preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]], weights: list[float],
                          iou_thr: float = 0.55, skip_thr: float = 0.0):
    """preds: one (boxes[N,4] xyxy, scores[N], labels[N]) tuple per model.
    Returns fused (boxes[K,4], scores[K], labels[K])."""
    total_w = float(sum(weights))
    items = [(int(l), float(s), m, np.asarray(b, float))
             for m, (bs, ss, ls) in enumerate(preds) for b, s, l in zip(bs, ss, ls) if s >= skip_thr]
    out_b, out_s, out_l = [], [], []
    for label in sorted({it[0] for it in items}):
        cls_items = sorted([it for it in items if it[0] == label], key=lambda it: -it[1] * weights[it[2]])
        clusters: list[list] = []
        fused: list[np.ndarray] = []
        for it in cls_items:
            if fused:
                ious = _iou(it[3], np.stack(fused))
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    clusters[j].append(it)
                    w = np.array([c[1] * weights[c[2]] for c in clusters[j]])
                    fused[j] = (np.stack([c[3] for c in clusters[j]]) * w[:, None]).sum(0) / w.sum()
                    continue
            clusters.append([it])
            fused.append(it[3].copy())
        for cl, fb in zip(clusters, fused):
            mw = np.array([weights[c[2]] for c in cl])
            conf = float((np.array([c[1] for c in cl]) * mw).sum() / mw.sum())
            agree = sum(weights[m] for m in {c[2] for c in cl}) / total_w
            out_b.append(fb); out_s.append(conf * agree); out_l.append(label)
    if not out_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.stack(out_b), np.array(out_s), np.array(out_l, int)
