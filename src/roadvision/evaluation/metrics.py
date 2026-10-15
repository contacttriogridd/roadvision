"""Detection metrics implemented from scratch so that every method (YOLO, RT-DETR,
hybrid, WBF ensemble) is scored with the *same* protocol as CRDDC'2022:
a prediction is a true positive iff IoU > 0.5 with an unmatched ground-truth box
of the same class. Also provides VOC all-point AP50, confusion matrices, error
taxonomy, size/country breakdowns and paired bootstrap confidence intervals.

Data format
  preds[img] = (boxes[N,4] xyxy, scores[N], labels[N])
  gts[img]   = (boxes[M,4] xyxy, labels[M])
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..data.crops import iou_xyxy


def match_image(pb, ps, pl, gb, gl, thr: float, iou_thr: float = 0.5):
    """Greedy per-class matching by descending score. Returns tp flags for kept preds,
    kept indices, matched-gt flags."""
    keep = np.where(ps >= thr)[0]
    keep = keep[np.argsort(-ps[keep])]
    tp = np.zeros(len(keep), bool)
    gt_used = np.zeros(len(gb), bool)
    if len(keep) and len(gb):
        ious = iou_xyxy(pb[keep], gb)
        for k, i in enumerate(keep):
            cand = np.where((gl == pl[i]) & ~gt_used & (ious[k] > iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[k, cand])]
                gt_used[j] = True
                tp[k] = True
    return tp, keep, gt_used


def prf_at(preds: dict, gts: dict, thr: float, iou_thr: float = 0.5, n_cls: int = 4, images=None) -> dict:
    tp = np.zeros(n_cls); fp = np.zeros(n_cls); fn = np.zeros(n_cls)
    for img in images if images is not None else gts:
        pb, ps, pl = preds.get(img, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        gb, gl = gts[img]
        t, keep, used = match_image(pb, ps, pl, gb, gl, thr, iou_thr)
        for c in range(n_cls):
            tp[c] += t[pl[keep] == c].sum() if len(keep) else 0
            fp[c] += (~t)[pl[keep] == c].sum() if len(keep) else 0
            fn[c] += (~used)[gl == c].sum() if len(gl) else 0
    P, R = tp.sum() / max(1, tp.sum() + fp.sum()), tp.sum() / max(1, tp.sum() + fn.sum())
    per_p = tp / np.maximum(1, tp + fp); per_r = tp / np.maximum(1, tp + fn)
    per_f = 2 * per_p * per_r / np.maximum(1e-9, per_p + per_r)
    return {"precision": float(P), "recall": float(R), "f1": float(2 * P * R / max(1e-9, P + R)),
            "per_class_f1": per_f.tolist(), "per_class_precision": per_p.tolist(), "per_class_recall": per_r.tolist(),
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}


def best_threshold(preds: dict, gts: dict, n_cls: int = 4, grid=None) -> tuple[float, dict, list]:
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.025), 3)
    curve = [(float(t), prf_at(preds, gts, t, n_cls=n_cls)) for t in grid]
    t, m = max(curve, key=lambda x: x[1]["f1"])
    return t, m, curve


def ap50_per_class(preds: dict, gts: dict, n_cls: int = 4, iou_thr: float = 0.5) -> list[float]:
    """Pascal-VOC all-point interpolated AP at IoU 0.5."""
    aps = []
    for c in range(n_cls):
        scores, tps, n_gt = [], [], 0
        for img, (gb, gl) in gts.items():
            pb, ps, pl = preds.get(img, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
            m = pl == c
            gbc = gb[gl == c]
            n_gt += len(gbc)
            if not m.any():
                continue
            b, s = pb[m], ps[m]
            order = np.argsort(-s)
            used = np.zeros(len(gbc), bool)
            ious = iou_xyxy(b[order], gbc) if len(gbc) else np.zeros((len(order), 0))
            for k in range(len(order)):
                j = int(np.argmax(ious[k])) if ious.shape[1] else -1
                ok = j >= 0 and ious[k, j] > iou_thr and not used[j]
                if ok:
                    used[j] = True
                scores.append(s[order[k]]); tps.append(ok)
        if n_gt == 0:
            aps.append(float("nan")); continue
        o = np.argsort(-np.array(scores))
        tp = np.array(tps, float)[o] if tps else np.zeros(0)
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(1e-9, ctp + cfp)
        mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        i = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])))
    return aps


def pr_curve(preds: dict, gts: dict, c: int, iou_thr: float = 0.5):
    """Precision/recall arrays for one class (for plotting)."""
    scores, tps, n_gt = [], [], 0
    for img, (gb, gl) in gts.items():
        pb, ps, pl = preds.get(img, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        gbc = gb[gl == c]; n_gt += len(gbc)
        m = pl == c
        if not m.any():
            continue
        order = np.argsort(-ps[m]); b, s = pb[m][order], ps[m][order]
        used = np.zeros(len(gbc), bool)
        ious = iou_xyxy(b, gbc) if len(gbc) else np.zeros((len(b), 0))
        for k in range(len(b)):
            j = int(np.argmax(ious[k])) if ious.shape[1] else -1
            ok = j >= 0 and ious[k, j] > iou_thr and not used[j]
            if ok: used[j] = True
            scores.append(s[k]); tps.append(ok)
    if not scores or n_gt == 0:
        return np.array([0.0]), np.array([1.0])
    o = np.argsort(-np.array(scores)); tp = np.array(tps, float)[o]
    ctp = np.cumsum(tp)
    return ctp / n_gt, ctp / np.arange(1, len(tp) + 1)


def confusion(preds: dict, gts: dict, thr: float, n_cls: int = 4, iou_thr: float = 0.5) -> np.ndarray:
    """(n_cls+1)x(n_cls+1) matrix, rows = ground truth, cols = prediction, last index = background."""
    M = np.zeros((n_cls + 1, n_cls + 1), int)
    for img, (gb, gl) in gts.items():
        pb, ps, pl = preds.get(img, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        keep = ps >= thr
        pb, pl = pb[keep], pl[keep]
        used_p = np.zeros(len(pb), bool)
        ious = iou_xyxy(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb)))
        for g in np.argsort(-ious.max(1) if ious.size else np.zeros(len(gb))):
            cand = np.where(~used_p & (ious[g] > iou_thr))[0] if ious.size else []
            if len(cand):
                j = cand[np.argmax(ious[g, cand])]; used_p[j] = True; M[gl[g], pl[j]] += 1
            else:
                M[gl[g], n_cls] += 1
        for j in np.where(~used_p)[0]:
            M[n_cls, pl[j]] += 1
    return M


def error_taxonomy(preds: dict, gts: dict, thr: float, img_area: dict, buckets: dict, iou_thr: float = 0.5) -> dict:
    """Categorise every FP (classification / localisation / duplicate / background) and
    every FN by object size — the basis of the error-analysis section."""
    fp = defaultdict(int); fn_size = defaultdict(int); gt_size = defaultdict(int)
    for img, (gb, gl) in gts.items():
        pb, ps, pl = preds.get(img, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        t, keep, used = match_image(pb, ps, pl, gb, gl, thr, iou_thr)
        ious = iou_xyxy(pb[keep], gb) if len(keep) and len(gb) else np.zeros((len(keep), len(gb)))
        for k, i in enumerate(keep):
            if t[k]:
                continue
            if ious.shape[1] == 0 or ious[k].max() < 0.1:
                fp["background"] += 1
            elif (ious[k][gl == pl[i]] > iou_thr).any() if (gl == pl[i]).any() else False:
                fp["duplicate"] += 1
            elif (ious[k] > iou_thr).any():
                fp["classification"] += 1
            else:
                fp["localisation"] += 1
        area = img_area[img]
        for j in range(len(gb)):
            ra = (gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1]) / area
            s = "small" if ra < buckets["small"] else "medium" if ra < buckets["medium"] else "large"
            gt_size[s] += 1
            if not used[j]:
                fn_size[s] += 1
    recall_by_size = {s: 1 - fn_size[s] / gt_size[s] for s in gt_size if gt_size[s]}
    return {"fp_types": dict(fp), "fn_by_size": dict(fn_size), "gt_by_size": dict(gt_size),
            "recall_by_size": recall_by_size}


def bootstrap_f1(preds_a: dict, preds_b: dict | None, gts: dict, thr_a: float, thr_b: float | None = None,
                 n: int = 1000, seed: int = 0, n_cls: int = 4) -> dict:
    """Image-level bootstrap 95% CI of F1 for model A and (optionally) paired difference A-B."""
    rng = np.random.default_rng(seed)
    imgs = list(gts)
    # per-image TP/FP/FN cache -> fast resampling
    def counts(preds, thr):
        out = []
        for img in imgs:
            m = prf_at(preds, gts, thr, n_cls=n_cls, images=[img])
            out.append((m["tp"], m["fp"], m["fn"]))
        return np.array(out, float)
    ca = counts(preds_a, thr_a)
    cb = counts(preds_b, thr_b) if preds_b is not None else None
    f = lambda c: 2 * c[0] / max(1e-9, 2 * c[0] + c[1] + c[2])
    fa, diffs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(imgs), len(imgs))
        sa = ca[idx].sum(0); fa.append(f(sa))
        if cb is not None:
            diffs.append(f(sa) - f(cb[idx].sum(0)))
    res = {"f1": f(ca.sum(0)), "ci95": [float(np.percentile(fa, 2.5)), float(np.percentile(fa, 97.5))]}
    if cb is not None:
        d = np.array(diffs)
        res.update({"delta_f1": float(f(ca.sum(0)) - f(cb.sum(0))),
                    "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "p_value_one_sided": float((d <= 0).mean())})
    return res
