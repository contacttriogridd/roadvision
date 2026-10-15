import subprocess
import sys
from pathlib import Path

import pandas as pd

from roadvision.data.prepare import main as prepare
from roadvision.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_on_synthetic(tmp_path):
    raw = tmp_path / "RDD2022"
    subprocess.check_call([sys.executable, str(ROOT / "scripts/make_synthetic_rdd.py"), "--out", str(raw), "--n", "30"])
    cfg = load_config(ROOT / "configs/default.yaml", [ROOT / "configs/smoke.yaml"])
    cfg["data"]["raw_dir"] = str(raw)
    cfg["data"]["processed_dir"] = str(tmp_path / "proc")
    prepare(cfg)
    proc = tmp_path / "proc"
    imgs = pd.read_csv(proc / "images.csv")
    boxes = pd.read_csv(proc / "boxes.csv")
    assert set(imgs.split) == {"train", "val", "test"}
    assert not set(boxes.label) - {"D00", "D10", "D20", "D40"}       # D43 dropped
    assert (boxes.x2 > boxes.x1).all() and (boxes.y2 > boxes.y1).all()
    for r in imgs.itertuples():
        lab = proc / "labels" / r.split / f"{r.stem}.txt"
        assert lab.exists()
        for line in filter(None, lab.read_text().splitlines()):
            c, *xywh = line.split()
            assert 0 <= int(c) < 4 and all(0 <= float(v) <= 1 for v in xywh)
    assert (proc / "dataset.yaml").exists()
