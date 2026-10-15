"""Albumentations pipelines.

* `road_augment` — photometric/weather augmentation that mimics dash-cam conditions
  (glare, rain, fog, shadows, motion blur). Used for ViT crop training and for the
  augmentation gallery figure. YOLO itself uses Ultralytics' on-the-fly mosaic/mixup/HSV.
* `vit_train_tf` / `vit_eval_tf` — tensor transforms for the ViT refiner.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def road_augment(a):
    import albumentations as A  # training-only dependency

    return A.Compose([
        A.HorizontalFlip(p=a.hflip),
        A.RandomBrightnessContrast(0.3, 0.3, p=a.brightness_contrast),
        A.HueSaturationValue(10, 25, 20, p=a.hue_sat),
        A.OneOf([A.MotionBlur(blur_limit=7), A.GaussianBlur(blur_limit=(3, 5))], p=a.motion_blur),
        A.OneOf([A.RandomRain(), A.RandomFog(), A.RandomShadow()], p=a.rain_fog_shadow),
        A.CLAHE(p=a.clahe),
    ])


def vit_train_tf(a, size: int):
    import albumentations as A

    return A.Compose([
        road_augment(a),
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.6, 1.66)),
        A.Affine(rotate=(-10, 10), p=0.3),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def vit_eval_tf(size: int):
    """Resize + ImageNet normalisation; same output as A.Compose([A.Resize, A.Normalize]) without
    importing albumentations, so the inference runtime stays slim."""
    mean = np.array(IMAGENET_MEAN, np.float32) * 255.0
    std = np.array(IMAGENET_STD, np.float32) * 255.0

    def tf(image: np.ndarray) -> dict:
        img = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return {"image": (img - mean) / std}

    return tf


def to_chw(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)
