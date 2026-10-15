"""YAML configuration loading with deep-merge overrides and dot-access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.yolo.epochs) that stays JSON/YAML serialisable."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _wrap(d: Any) -> Any:
    if isinstance(d, dict):
        return Cfg({k: _wrap(v) for k, v in d.items()})
    return d


def load_config(path: str | Path = "configs/default.yaml", overrides: list[str | Path] | None = None) -> Cfg:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for o in overrides or []:
        with open(o, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return _wrap(cfg)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)


def workdir(cfg: Cfg, *parts: str) -> Path:
    p = Path(cfg.project.workdir, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
