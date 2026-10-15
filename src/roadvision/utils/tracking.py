"""Thin MLflow wrapper. Falls back to JSON-lines logging if MLflow is unavailable,
so the pipeline never breaks because of the tracker."""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

from .common import get_logger

log = get_logger("tracking")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, uri: str, experiment: str, fallback_dir: str | Path = "runs/tracking"):
        self.fallback = Path(fallback_dir)
        self.fallback.mkdir(parents=True, exist_ok=True)
        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow
        except Exception as exc:  # pragma: no cover - only without mlflow
            log.warning("MLflow unavailable (%s); using JSON-lines fallback", exc)
            self.mlflow = None
        self._jsonl: Path | None = None

    @contextlib.contextmanager
    def run(self, name: str, tags: dict | None = None, nested: bool = False) -> Iterator["Tracker"]:
        self._jsonl = self.fallback / f"{int(time.time())}_{name}.jsonl"
        if self.mlflow:
            with self.mlflow.start_run(run_name=name, nested=nested, tags=tags or {}):
                yield self
        else:
            yield self

    def _write(self, kind: str, payload: dict) -> None:
        if self._jsonl:
            with open(self._jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": kind, **payload}, default=str) + "\n")

    def params(self, params: dict[str, Any]) -> None:
        flat = {k: str(v)[:500] for k, v in _flatten(params).items()}
        self._write("params", flat)
        if self.mlflow:
            self.mlflow.log_params(flat)

    def metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k.replace("(", "_").replace(")", "").replace("/", "_"): float(v)
                 for k, v in metrics.items() if v is not None}
        self._write("metrics", {"step": step, **clean})
        if self.mlflow:
            self.mlflow.log_metrics(clean, step=step)

    def artifact(self, path: str | Path) -> None:
        if self.mlflow and Path(path).exists():
            if Path(path).is_dir():
                self.mlflow.log_artifacts(str(path))
            else:
                self.mlflow.log_artifact(str(path))
