"""Lightweight colour-histogram appearance for occlusion and camera handover."""
from __future__ import annotations

from typing import Any

import numpy as np

from .utils import BBox


def crop_histogram(image: Any, bbox: BBox) -> np.ndarray | None:
    import cv2

    x1, y1, x2, y2 = [max(0, int(value)) for value in bbox]
    h, w = image.shape[:2]
    x1, x2 = min(x1, w - 1), min(max(x2, x1 + 1), w)
    y1, y2 = min(y1, h - 1), min(max(y2, y1 + 1), h)
    crop = image[y1:y2, x1:x2]
    if crop.size < 16:
        return None
    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.astype("float32")


def appearance_similarity(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.0
    import cv2

    return float(cv2.compareHist(first, second, cv2.HISTCMP_CORREL))
