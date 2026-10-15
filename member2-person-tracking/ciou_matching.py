"""CIoU cost matrices and Hungarian assignment for tracking."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils import BBox, as_bbox


def ciou(first: BBox, second: BBox) -> float:
    """Return Complete IoU in [-1, 1] for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_w, inter_h = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    diagonal = (max(ax2, bx2) - min(ax1, bx1)) ** 2 + (max(ay2, by2) - min(ay1, by1)) ** 2
    centre = ((acx - bcx) ** 2 + (acy - bcy) ** 2) / diagonal if diagonal else 0.0
    import math
    wa, ha = max(ax2 - ax1, 1e-6), max(ay2 - ay1, 1e-6)
    wb, hb = max(bx2 - bx1, 1e-6), max(by2 - by1, 1e-6)
    v = (4 / math.pi**2) * (math.atan(wb / hb) - math.atan(wa / ha)) ** 2
    denominator = 1 - iou + v
    alpha = v / denominator if denominator else 0.0
    return iou - centre - alpha * v


def ciou_cost_matrix(tracks: Sequence[BBox], detections: Sequence[BBox]) -> np.ndarray:
    return np.array([[1.0 - ciou(as_bbox(track), as_bbox(det)) for det in detections]
                     for track in tracks], dtype=np.float32)


def hungarian_matches(
    tracks: Sequence[BBox], detections: Sequence[BBox], max_distance: float = 0.3
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(range(len(tracks))), set(range(len(detections)))
    costs = ciou_cost_matrix(tracks, detections)
    rows, columns = linear_sum_assignment(costs)
    matches = [(int(row), int(column)) for row, column in zip(rows, columns)
               if costs[row, column] <= max_distance]
    used_tracks, used_detections = {row for row, _ in matches}, {column for _, column in matches}
    return matches, set(range(len(tracks))) - used_tracks, set(range(len(detections))) - used_detections
