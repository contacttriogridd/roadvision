"""Comparison against published state of the art on RDD2020/RDD2022.

Literature numbers are transcribed from the cited papers (see report references).
`comparable` marks rows evaluated under exactly the protocol we can reproduce
(China subset, random 8:1:1 split — run with configs/china_811.yaml); other rows
use hidden challenge test sets and are reported as indicative context only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..utils.common import get_logger, read_json, write_json
from ..utils.config import load_config, workdir

log = get_logger("benchmark")

LITERATURE = [
    # method, source, protocol, F1, mAP50, comparable
    ("YOLO ensemble (GRDDC'20 winner)", "Arya et al. 2020 [GRDDC]", "RDD2020 test1 (JP/IN/CZ)", 0.67, None, False),
    ("YOLOv5/v7 + Cascade R-CNN-Swin ens. (CRDDC'22 #1)", "Arya et al. 2022 [CRDDC]", "RDD2022 hidden test, 6 countries", 0.770, None, False),
    ("YOLOv7 single model (CRDDC'22 #3)", "Arya et al. 2022 [CRDDC]", "RDD2022 hidden test, 6 countries", 0.741, None, False),
    ("Faster R-CNN-R50", "Lin et al. 2025 [StripRFNet]", "RDD2022-China 8:1:1", 0.776, 0.830, True),
    ("YOLOv8s", "Lin et al. 2025 [StripRFNet]", "RDD2022-China 8:1:1", 0.807, 0.839, True),
    ("YOLO11s", "Lin et al. 2025 [StripRFNet]", "RDD2022-China 8:1:1", 0.772, 0.815, True),
    ("RT-DETRv2-R18", "Lin et al. 2025 [StripRFNet]", "RDD2022-China 8:1:1", 0.767, 0.800, True),
    ("StripRFNet", "Lin et al. 2025 [StripRFNet]", "RDD2022-China 8:1:1", 0.816, 0.844, True),
]

PRETTY = {"yolo": "RoadVision YOLO", "rtdetr": "RoadVision RT-DETR", "ensemble": "RoadVision WBF (YOLO+RT-DETR)",
          "hybrid": "RoadVision Hybrid YOLO+ViT (ours)"}


def build(cfg) -> pd.DataFrame:
    path = workdir(cfg, "metrics") / "eval_results.json"
    if not path.exists():
        log.warning("No eval_results.json yet — run the evaluate stage first")
        return pd.DataFrame()
    res = read_json(path)
    proto = "RDD2022-China 8:1:1" if cfg.data.protocol == "china_811" else "RDD2022 (our 8:1:1 strat. split)"
    rows = [dict(method=m, source=s, protocol=p, f1=f, mAP50=a, comparable=c, ours=False) for m, s, p, f, a, c in LITERATURE]
    for mode, r in res["modes"].items():
        rows.append(dict(method=PRETTY.get(mode, mode), source="this work", protocol=proto, f1=r["f1"],
                         mAP50=r["mAP50"], comparable=cfg.data.protocol == "china_811", ours=True,
                         f1_ci_low=r["bootstrap"]["ci95"][0], f1_ci_high=r["bootstrap"]["ci95"][1], fps=r["fps"]))
    df = pd.DataFrame(rows)
    out = workdir(cfg, "metrics")
    df.to_csv(out / "benchmark.csv", index=False)
    write_json(df.to_dict(orient="records"), out / "benchmark.json")
    log.info("\n%s", df[["method", "protocol", "f1", "mAP50"]].to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    build(load_config(a.config, a.override))
