"""Fetch deployment weights at start-up when they are not baked into the image.

Sources (first match wins):
  RV_WEIGHTS_URL   base URL holding the files, e.g. a GitHub release
                   https://github.com/<user>/roadvision/releases/download/v1.0/
  RV_WEIGHTS_REPO  Hugging Face model repo (or cfg.app.hf_weights_repo)
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from ..utils.common import get_logger

log = get_logger("weights")
KEYS = ("detector", "rtdetr", "vit", "calibration")


def ensure_weights(cfg) -> None:
    targets = [Path(cfg.app[k]) for k in KEYS] + [Path(cfg.app.weights_dir) / "eval_summary.json"]
    missing = [p for p in targets if not p.exists()]
    if not missing:
        return
    url = os.environ.get("RV_WEIGHTS_URL", "").rstrip("/")
    repo = os.environ.get("RV_WEIGHTS_REPO") or cfg.app.get("hf_weights_repo", "")
    for dst in missing:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:  # optional files (e.g. rtdetr) may be absent from the source
            if url:
                tmp = dst.with_suffix(dst.suffix + ".part")
                with urllib.request.urlopen(f"{url}/{dst.name}", timeout=300) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                tmp.replace(dst)
            elif repo:
                from huggingface_hub import hf_hub_download

                hf_hub_download(repo, dst.name, local_dir=str(dst.parent))
            else:
                continue
            log.info("downloaded %s", dst.name)
        except Exception as exc:
            log.info("%s not available: %s", dst.name, exc)
