"""Download RDD2022 (Arya et al., 2022) from the official CRDDC'2022 bucket.

Usage:  python -m roadvision.data.download --countries Japan India Czech United_States China_MotorBike China_Drone
Kaggle/Colab alternative: any RDD2022 mirror works as long as it unzips to
data/raw/RDD2022/<Country>/train/{images,annotations/xmls}.
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from ..utils.common import get_logger

log = get_logger("download")
BASE = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/"
FILES = {
    "Japan": "RDD2022_Japan.zip",
    "India": "RDD2022_India.zip",
    "Czech": "RDD2022_Czech.zip",
    "Norway": "RDD2022_Norway.zip",
    "United_States": "RDD2022_United_States.zip",
    "China_MotorBike": "RDD2022_China_MotorBike.zip",
    "China_Drone": "RDD2022_China_Drone.zip",
}


def _progress(block: int, bsize: int, total: int) -> None:
    if total > 0 and block % 500 == 0:
        log.info("  %.1f%% of %.0f MB", 100 * block * bsize / total, total / 1e6)


def download(countries: list[str], out_dir: str = "data/raw/RDD2022") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for c in countries:
        target = out / c
        if (target / "train").exists() or any(target.glob("**/train")):
            log.info("%s already present, skipping", c)
            continue
        zpath = out / FILES[c]
        if not zpath.exists():
            log.info("Downloading %s", FILES[c])
            urllib.request.urlretrieve(BASE + FILES[c], zpath, _progress)
        log.info("Extracting %s", zpath.name)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(target)
        zpath.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=[k for k in FILES if k != "Norway"])
    ap.add_argument("--out", default="data/raw/RDD2022")
    a = ap.parse_args()
    download(a.countries, a.out)
