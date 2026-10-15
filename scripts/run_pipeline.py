#!/usr/bin/env python
"""One command, whole project. Each stage is idempotent and can be run alone.

    python scripts/run_pipeline.py --stages all
    python scripts/run_pipeline.py --stages prepare eda
    python scripts/run_pipeline.py --stages all --override configs/smoke.yaml      # CPU smoke test
    python scripts/run_pipeline.py --stages all --override configs/china_811.yaml  # benchmark protocol

Stage order:
  download -> prepare -> tune_yolo -> train_yolo -> crops -> tune_vit -> train_vit
  -> train_rtdetr -> evaluate -> benchmark -> explain -> plots -> report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadvision.utils.common import get_logger  # noqa: E402
from roadvision.utils.config import load_config, save_config, workdir  # noqa: E402

log = get_logger("pipeline")
ORDER = ["download", "prepare", "tune_yolo", "train_yolo", "crops", "tune_vit", "train_vit", "train_rtdetr",
         "evaluate", "benchmark", "explain", "plots", "report"]


def stage(name: str, cfg) -> None:
    opt = Path(cfg.project.workdir) / "optuna"
    if name == "download":
        from roadvision.data.download import download
        download(list(cfg.data.countries), cfg.data.raw_dir)
    elif name == "prepare":
        from roadvision.data.prepare import main
        main(cfg)
    elif name == "tune_yolo":
        from roadvision.tuning.optuna_search import tune
        tune(cfg, "yolo")
    elif name == "train_yolo":
        from roadvision.models.detectors import main
        main(cfg, "yolo", str(opt / "yolo_best.json"))
    elif name == "crops":
        from roadvision.data.crops import build_crops
        build_crops(cfg, cfg.app.detector)
    elif name == "tune_vit":
        from roadvision.tuning.optuna_search import tune
        tune(cfg, "vit")
    elif name == "train_vit":
        from roadvision.models.vit_refiner import main
        main(cfg, str(opt / "vit_best.json"))
    elif name == "train_rtdetr":
        from roadvision.models.detectors import main
        main(cfg, "rtdetr", None)
    elif name == "evaluate":
        from roadvision.evaluation.evaluate import run
        run(cfg)
    elif name == "benchmark":
        from roadvision.evaluation.benchmark import build
        build(cfg)
    elif name == "explain":
        from roadvision.explain.explainability import main
        main(cfg)
    elif name == "plots":
        from roadvision.visualization.plots import main
        main(cfg)
    elif name == "report":
        from roadvision.evaluation.report_assets import main
        main(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--stages", nargs="+", default=["all"])
    ap.add_argument("--skip", nargs="*", default=[], help="e.g. --skip download tune_yolo tune_vit")
    a = ap.parse_args()
    cfg = load_config(a.config, a.override)
    save_config(cfg, workdir(cfg) / "resolved_config.yaml")
    stages = ORDER if a.stages == ["all"] else a.stages
    for s in [s for s in stages if s not in a.skip]:
        t0 = time.time()
        log.info("===== stage: %s =====", s)
        stage(s, cfg)
        log.info("===== %s done in %.1fs =====", s, time.time() - t0)


if __name__ == "__main__":
    main()
