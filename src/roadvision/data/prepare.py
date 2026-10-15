"""RDD2022 (Pascal-VOC XML) -> cleaned, stratified YOLO dataset + tabular manifests.

Data-engineering steps (all counted in cleaning_report.json):
  * missing / corrupt images and XMLs are skipped
  * out-of-scope labels (D43, D44, D50, Repair, ...) are dropped
  * boxes are clipped to the image; degenerate boxes (< min_box_px) removed
  * exact duplicate boxes removed
  * annotation-free images are kept only as a controlled fraction of hard negatives
  * stratified split on (country, dominant class) so every split sees every domain
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from ..utils.common import get_logger, seed_everything, write_json
from ..utils.config import load_config

log = get_logger("prepare")


def find_country_pairs(raw_dir: Path, country: str) -> list[tuple[Path, Path | None]]:
    """Return (image, xml|None) pairs for one country, tolerant to zip nesting."""
    root = raw_dir / country
    pairs = []
    for img_dir in root.glob("**/train/images"):
        xml_dir = img_dir.parent / "annotations" / "xmls"
        for img in sorted(img_dir.glob("*.jpg")):
            xml = xml_dir / f"{img.stem}.xml"
            pairs.append((img, xml if xml.exists() else None))
    return pairs


def parse_voc(xml_path: Path) -> tuple[int | None, int | None, list[tuple[str, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    r = tree.getroot()
    size = r.find("size")
    w = int(float(size.findtext("width"))) if size is not None and size.findtext("width") else None
    h = int(float(size.findtext("height"))) if size is not None and size.findtext("height") else None
    objs = []
    for o in r.iter("object"):
        name = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if bb is None:
            continue
        try:
            objs.append((name, *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        except (TypeError, ValueError):
            continue
    return w, h, objs


def build_records(cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = Path(cfg.data.raw_dir)
    classes = list(cfg.data.classes)
    drop = set(cfg.data.drop_labels)
    stats: Counter = Counter()
    images, boxes = [], []
    for country in cfg.data.countries:
        pairs = find_country_pairs(raw, country)
        if not pairs:
            log.warning("No data found for %s under %s", country, raw)
        for img_path, xml_path in pairs:
            stats["images_seen"] += 1
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception:
                stats["corrupt_images"] += 1
                continue
            objs = []
            if xml_path is None:
                stats["missing_xml"] += 1
            else:
                try:
                    _, _, objs = parse_voc(xml_path)
                except ET.ParseError:
                    stats["corrupt_xml"] += 1
            seen = set()
            kept = []
            for name, x1, y1, x2, y2 in objs:
                stats["boxes_seen"] += 1
                if name in drop or name not in classes:
                    stats[f"dropped_label_{name}"] += 1
                    continue
                cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                    stats["clipped_boxes"] += 1
                if cx2 - cx1 < cfg.data.min_box_px or cy2 - cy1 < cfg.data.min_box_px:
                    stats["degenerate_boxes"] += 1
                    continue
                key = (name, round(cx1), round(cy1), round(cx2), round(cy2))
                if key in seen:
                    stats["duplicate_boxes"] += 1
                    continue
                seen.add(key)
                kept.append((classes.index(name), cx1, cy1, cx2, cy2))
            cls_counts = Counter(c for c, *_ in kept)
            dominant = classes[cls_counts.most_common(1)[0][0]] if kept else "NEG"
            images.append(dict(image=str(img_path), stem=f"{country}__{img_path.stem}", country=country,
                               width=W, height=H, n_boxes=len(kept), dominant=dominant))
            for c, x1, y1, x2, y2 in kept:
                boxes.append(dict(stem=f"{country}__{img_path.stem}", country=country, cls=c, label=classes[c],
                                  x1=x1, y1=y1, x2=x2, y2=y2, w=x2 - x1, h=y2 - y1,
                                  rel_area=(x2 - x1) * (y2 - y1) / (W * H),
                                  cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                                  aspect=(x2 - x1) / max(1e-6, (y2 - y1))))
    img_df, box_df = pd.DataFrame(images), pd.DataFrame(boxes)
    # controlled sub-sampling of negatives (missingness handling: unlabeled images)
    if len(img_df):
        neg = img_df[img_df.n_boxes == 0]
        n_keep = int(cfg.data.keep_negatives * (len(img_df) - len(neg)))
        drop_idx = neg.sample(frac=1.0, random_state=cfg.project.seed).index[n_keep:]
        stats["negatives_total"] = len(neg)
        stats["negatives_kept"] = len(neg) - len(drop_idx)
        img_df = img_df.drop(index=drop_idx).reset_index(drop=True)
    stats["images_kept"] = len(img_df)
    stats["boxes_kept"] = len(box_df)
    return img_df, box_df, dict(stats)


def assign_splits(img_df: pd.DataFrame, cfg) -> pd.DataFrame:
    s = cfg.data.split
    seed = cfg.project.seed
    df = img_df.copy()
    if cfg.data.protocol == "china_811":
        idx = list(df.index)
        random.Random(seed).shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(0.8 * n), int(0.1 * n)
        df.loc[idx[:n_tr], "split"] = "train"
        df.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        df.loc[idx[n_tr + n_va:], "split"] = "test"
        return df
    key = df.country + "|" + df.dominant
    counts = key.value_counts()
    key = key.where(key.map(counts) >= 10, df.country)          # merge rare strata
    key = key.where(key.map(key.value_counts()) >= 3, "rare")
    tr, tmp = train_test_split(df.index, test_size=s["val"] + s["test"], random_state=seed, stratify=key)
    rel = s["test"] / (s["val"] + s["test"])
    k2 = key.loc[tmp]
    strat = k2 if k2.value_counts().min() >= 2 else None
    va, te = train_test_split(tmp, test_size=rel, random_state=seed, stratify=strat)
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    return df


def _place(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_yolo(img_df: pd.DataFrame, box_df: pd.DataFrame, cfg) -> Path:
    out = Path(cfg.data.processed_dir)
    for sp in ("train", "val", "test"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    groups = {k: g for k, g in box_df.groupby("stem")} if len(box_df) else {}
    for r in img_df.itertuples():
        _place(Path(r.image), out / "images" / r.split / f"{r.stem}.jpg")
        lines = []
        for b in groups.get(r.stem, pd.DataFrame()).itertuples():
            cx, cy = (b.x1 + b.x2) / 2 / r.width, (b.y1 + b.y2) / 2 / r.height
            lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {b.w / r.width:.6f} {b.h / r.height:.6f}")
        (out / "labels" / r.split / f"{r.stem}.txt").write_text("\n".join(lines))
    yaml_path = out / "dataset.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(cfg.data.classes))
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    return yaml_path


def main(cfg) -> None:
    seed_everything(cfg.project.seed)
    img_df, box_df, stats = build_records(cfg)
    if img_df.empty:
        raise SystemExit("No images found — check data.raw_dir / data.countries")
    img_df = assign_splits(img_df, cfg)
    box_df = box_df.merge(img_df[["stem", "split"]], on="stem", how="inner")
    yaml_path = export_yolo(img_df, box_df, cfg)
    out = Path(cfg.data.processed_dir)
    img_df.to_csv(out / "images.csv", index=False)
    box_df.to_csv(out / "boxes.csv", index=False)
    stats["split_sizes"] = img_df.split.value_counts().to_dict()
    stats["class_counts"] = box_df.label.value_counts().to_dict()
    write_json(stats, out / "cleaning_report.json")
    log.info("Dataset ready: %s | %s", yaml_path, stats["split_sizes"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    main(load_config(a.config, a.override))
