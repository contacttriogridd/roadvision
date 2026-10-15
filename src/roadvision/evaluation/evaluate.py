"""End-to-end evaluation + calibration of every model under one protocol.

  1. Cache raw predictions of every available mode on val and test.
  2. Grid-search the hybrid fusion weight alpha and every mode's score threshold on
     **val only** (no test leakage) -> models/hybrid_calibration.json.
  3. Score test: P/R/F1 (CRDDC protocol), AP50 per class, mAP50, confusion matrix,
     error taxonomy, per-country F1, latency/FPS, bootstrap CIs, paired test vs YOLO.
  4. Ablation of the hybrid design (alpha=1, alpha=0, no BG rejection).

    python -m roadvision.evaluation.evaluate
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..inference.predictor import RoadVisionPredictor, from_config
from ..utils.common import get_logger, write_json
from ..utils.config import load_config, workdir
from ..utils.tracking import Tracker
from .metrics import ap50_per_class, best_threshold, bootstrap_f1, confusion, error_taxonomy, prf_at

log = get_logger("evaluate")


def load_gts(cfg, split: str):
    proc = Path(cfg.data.processed_dir)
    imgs = pd.read_csv(proc / "images.csv")
    imgs = imgs[imgs.split == split]
    boxes = pd.read_csv(proc / "boxes.csv")
    g = {k: v for k, v in boxes.groupby("stem")}
    gts, meta = {}, {}
    for r in imgs.itertuples():
        b = g.get(r.stem)
        gts[r.stem] = (b[["x1", "y1", "x2", "y2"]].to_numpy(float), b.cls.to_numpy(int)) if b is not None \
            else (np.zeros((0, 4)), np.zeros(0, int))
        meta[r.stem] = {"path": str(proc / "images" / split / f"{r.stem}.jpg"), "country": r.country,
                        "area": r.width * r.height}
    return gts, meta


def cache_predictions(pred: RoadVisionPredictor, meta: dict, modes: list[str]) -> dict:
    out = {m: {} for m in modes}
    hyb_raw = {}
    lat = {m: [] for m in modes}
    for i, (stem, m) in enumerate(meta.items()):
        img = cv2.imread(m["path"])
        for mode in modes:
            t0 = time.perf_counter()
            if mode == "hybrid":
                yb, ys, yl = pred._ultra(pred.yolo, img, pred.proposal_conf)
                probs = pred.vit_probs(img, yb)
                hyb_raw[stem] = (yb, ys, yl, probs)
            else:
                b, s, l, _ = pred.raw(img, mode, low_conf=True)
                out[mode][stem] = (b, s, l)
            lat[mode].append(1000 * (time.perf_counter() - t0))
        if i % 200 == 0:
            log.info("  predicted %d/%d", i, len(meta))
    return {"preds": out, "hybrid_raw": hyb_raw, "latency_ms": {m: float(np.median(v[1:] or v)) for m, v in lat.items()}}


def fuse_all(hyb_raw: dict, alpha: float, use_bg: bool = True) -> dict:
    res = {}
    for stem, (yb, ys, yl, probs) in hyb_raw.items():
        p = probs.copy()
        if not use_bg and len(p):
            p[:, -1] = 0.0
        b, s, l, _ = RoadVisionPredictor.fuse(yb, ys, yl, p, alpha)
        res[stem] = (b, s, l)
    return res


def run(cfg) -> dict:
    n_cls = len(cfg.data.classes)
    pred = from_config(cfg)
    modes = pred.available_modes
    if not modes:
        raise SystemExit("No trained weights found in models/. Train the detectors first.")
    log.info("Evaluating modes: %s", modes)
    gts_val, meta_val = load_gts(cfg, "val")
    gts_test, meta_test = load_gts(cfg, "test")
    cv = cache_predictions(pred, meta_val, modes)
    ct = cache_predictions(pred, meta_test, modes)

    calibration = {"thresholds": {}, "alpha": 1.0}
    if "hybrid" in modes:
        grid = []
        for a in cfg.hybrid.alpha_grid:
            t, m, _ = best_threshold(fuse_all(cv["hybrid_raw"], a), gts_val, n_cls)
            grid.append({"alpha": a, "thr": t, "val_f1": m["f1"]})
            log.info("  alpha=%.2f -> val F1 %.4f @ %.3f", a, m["f1"], t)
        best = max(grid, key=lambda g: g["val_f1"])
        calibration.update(alpha=best["alpha"], alpha_grid=grid)
        cv["preds"]["hybrid"] = fuse_all(cv["hybrid_raw"], best["alpha"])
        ct["preds"]["hybrid"] = fuse_all(ct["hybrid_raw"], best["alpha"])
    curves = {}
    for mode in modes:
        t, m, curve = best_threshold(cv["preds"][mode], gts_val, n_cls)
        calibration["thresholds"][mode] = t
        curves[mode] = [(c[0], c[1]["f1"], c[1]["precision"], c[1]["recall"]) for c in curve]
    write_json(calibration, cfg.app.calibration)

    results = {"calibration": calibration, "modes": {}, "f1_curves_val": curves}
    area = {k: v["area"] for k, v in meta_test.items()}
    for mode in modes:
        P = ct["preds"][mode]
        thr = calibration["thresholds"][mode]
        m = prf_at(P, gts_test, thr, n_cls=n_cls)
        aps = ap50_per_class(P, gts_test, n_cls)
        countries = {}
        for c in sorted({v["country"] for v in meta_test.values()}):
            ims = [k for k, v in meta_test.items() if v["country"] == c]
            countries[c] = prf_at(P, gts_test, thr, n_cls=n_cls, images=ims)["f1"]
        results["modes"][mode] = {
            **m, "threshold": thr, "AP50_per_class": aps, "mAP50": float(np.nanmean(aps)),
            "confusion": confusion(P, gts_test, thr, n_cls).tolist(),
            "errors": error_taxonomy(P, gts_test, thr, area, cfg.evaluation.size_buckets),
            "f1_by_country": countries, "latency_ms": ct["latency_ms"][mode],
            "fps": 1000.0 / max(1e-6, ct["latency_ms"][mode]),
            "bootstrap": bootstrap_f1(P, None, gts_test, thr, n=cfg.evaluation.bootstrap, n_cls=n_cls),
        }
        log.info("%-8s test F1 %.4f  P %.4f  R %.4f  mAP50 %.4f  %.1f FPS", mode, m["f1"], m["precision"],
                 m["recall"], results["modes"][mode]["mAP50"], results["modes"][mode]["fps"])
    if "hybrid" in modes and "yolo" in modes:
        results["paired_hybrid_vs_yolo"] = bootstrap_f1(
            ct["preds"]["hybrid"], ct["preds"]["yolo"], gts_test, calibration["thresholds"]["hybrid"],
            calibration["thresholds"]["yolo"], n=cfg.evaluation.bootstrap, n_cls=n_cls)
        abl = {}
        for name, a, bg in [("YOLO proposals only (alpha=1)", 1.0, True), ("ViT only (alpha=0)", 0.0, True),
                            ("Hybrid w/o BG rejection", calibration["alpha"], False),
                            ("Hybrid (full)", calibration["alpha"], True)]:
            t, _, _ = best_threshold(fuse_all(cv["hybrid_raw"], a, bg), gts_val, n_cls)
            mt = prf_at(fuse_all(ct["hybrid_raw"], a, bg), gts_test, t, n_cls=n_cls)
            abl[name] = {"f1": mt["f1"], "precision": mt["precision"], "recall": mt["recall"], "thr": t}
        results["ablation"] = abl
    out = workdir(cfg, "metrics")
    write_json(results, out / "eval_results.json")
    with open(out / "pred_cache.pkl", "wb") as f:
        pickle.dump({"test": ct["preds"], "gts_test": gts_test, "meta_test": meta_test}, f)
    tracker = Tracker(cfg.tracking.mlflow_uri, cfg.tracking.experiment, workdir(cfg, "tracking"))
    with tracker.run("evaluation", tags={"stage": "eval"}):
        tracker.params({"alpha": calibration["alpha"], **{f"thr_{k}": v for k, v in calibration["thresholds"].items()}})
        for mode, r in results["modes"].items():
            tracker.metrics({f"{mode}_test_f1": r["f1"], f"{mode}_test_mAP50": r["mAP50"], f"{mode}_fps": r["fps"]})
        tracker.artifact(out / "eval_results.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    run(load_config(a.config, a.override))
