"""Training/validation of the one-stage CNN detector (YOLOv8/YOLO11) and the
transformer detector (RT-DETR) through Ultralytics, with MLflow logging.

    python -m roadvision.models.detectors --kind yolo   [--params runs/optuna/yolo_best.json]
    python -m roadvision.models.detectors --kind rtdetr
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from ..utils.common import get_logger, read_json, seed_everything, write_json
from ..utils.config import load_config, workdir
from ..utils.tracking import Tracker

log = get_logger("detectors")

TRAIN_KEYS = {"yolo": ["imgsz", "epochs", "batch", "patience", "optimizer", "lr0", "weight_decay", "momentum",
                       "mosaic", "mixup", "close_mosaic", "cos_lr", "workers"],
              "rtdetr": ["imgsz", "epochs", "batch", "patience", "lr0", "workers"]}


def load_model(kind: str, weights: str):
    from ultralytics import RTDETR, YOLO

    return RTDETR(weights) if kind == "rtdetr" else YOLO(weights)


def train_detector(cfg, kind: str = "yolo", params: dict | None = None, name: str | None = None,
                   tracker: Tracker | None = None, fraction: float = 1.0, callbacks: dict | None = None) -> dict:
    seed_everything(cfg.project.seed)
    mcfg = cfg[kind]
    args = {k: mcfg[k] for k in TRAIN_KEYS[kind] if k in mcfg}
    args.update(params or {})
    data_yaml = str(Path(cfg.data.processed_dir) / "dataset.yaml")
    model = load_model(kind, mcfg["weights"])
    for event, fn in (callbacks or {}).items():
        model.add_callback(event, fn)
    name = name or kind
    project = str(workdir(cfg, "detect").resolve())
    results = model.train(data=data_yaml, project=project, name=name, exist_ok=True, device=cfg.project.device,
                          seed=cfg.project.seed, deterministic=True, fraction=fraction, plots=True, verbose=False,
                          **args)
    save_dir = Path(model.trainer.save_dir)
    summary = {"kind": kind, "save_dir": str(save_dir), "params": args}
    csv = save_dir / "results.csv"
    if tracker is not None:
        tracker.params({"kind": kind, "weights": mcfg["weights"], **args})
        if csv.exists():
            hist = pd.read_csv(csv)
            hist.columns = [c.strip() for c in hist.columns]
            for i, row in hist.iterrows():
                tracker.metrics({k: row[k] for k in hist.columns if k != "epoch"}, step=int(i))
    rd = getattr(results, "results_dict", {}) or {}
    summary["val"] = {k: float(v) for k, v in rd.items()}
    return summary


def evaluate_split(cfg, kind: str, weights: str, split: str = "test") -> dict:
    model = load_model(kind, weights)
    m = model.val(data=str(Path(cfg.data.processed_dir) / "dataset.yaml"), split=split, imgsz=cfg[kind]["imgsz"],
                  device=cfg.project.device, plots=True, verbose=False,
                  project=str(workdir(cfg, "detect").resolve()), name=f"{kind}_{split}", exist_ok=True)
    box = m.box
    per_class = {cfg.data.classes[int(c)]: float(ap) for c, ap in zip(box.ap_class_index, box.ap50)}
    sp = m.speed  # ms per image: preprocess / inference / postprocess
    return {"mAP50": float(box.map50), "mAP50_95": float(box.map), "precision": float(box.mp),
            "recall": float(box.mr), "AP50_per_class": per_class, "speed_ms": {k: float(v) for k, v in sp.items()},
            "fps": 1000.0 / max(1e-6, sum(sp.values()))}


def main(cfg, kind: str, params_path: str | None) -> None:
    params = read_json(params_path) if params_path and Path(params_path).exists() else None
    tracker = Tracker(cfg.tracking.mlflow_uri, cfg.tracking.experiment, workdir(cfg, "tracking"))
    with tracker.run(f"{kind}_final", tags={"stage": "train", "model": kind}):
        summary = train_detector(cfg, kind, params, name=kind, tracker=tracker)
        best = Path(summary["save_dir"]) / "weights" / "best.pt"
        dst = Path(cfg.app.weights_dir) / f"{kind}_best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        summary["test"] = evaluate_split(cfg, kind, str(dst), "test")
        tracker.metrics({f"test_{k}": v for k, v in summary["test"].items() if isinstance(v, float)})
        tracker.artifact(Path(summary["save_dir"]))
    write_json(summary, workdir(cfg, "metrics") / f"{kind}_summary.json")
    log.info("%s test: mAP50=%.4f mAP50-95=%.4f FPS=%.1f", kind, summary["test"]["mAP50"],
             summary["test"]["mAP50_95"], summary["test"]["fps"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--kind", choices=["yolo", "rtdetr"], default="yolo")
    ap.add_argument("--params", default=None, help="JSON of tuned hyper-parameters (from Optuna)")
    a = ap.parse_args()
    main(load_config(a.config, a.override), a.kind, a.params)
