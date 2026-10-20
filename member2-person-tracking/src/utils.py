"""Geometry and presentation helpers used by the tracking component."""

from __future__ import annotations

from typing import Sequence

BBox = tuple[float, float, float, float]


def as_bbox(values: Sequence[float]) -> BBox:
    if len(values) != 4:
        raise ValueError("A bounding box must contain four values")
    x1, y1, x2, y2 = map(float, values)
    if x2 < x1 or y2 < y1:
        raise ValueError("Bounding box maximum values must not precede minimum values")
    return x1, y1, x2, y2


def box_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def iou(first: BBox, second: BBox) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union else 0.0


def ciou(first: BBox, second: BBox) -> float:
    """Complete IoU: overlap plus centre distance and aspect-ratio agreement."""
    import math

    overlap = iou(first, second)
    acx, acy = (first[0] + first[2]) / 2, (first[1] + first[3]) / 2
    bcx, bcy = (second[0] + second[2]) / 2, (second[1] + second[3]) / 2
    enclose_w = max(first[2], second[2]) - min(first[0], second[0])
    enclose_h = max(first[3], second[3]) - min(first[1], second[1])
    diagonal = enclose_w * enclose_w + enclose_h * enclose_h
    centre = ((acx - bcx) ** 2 + (acy - bcy) ** 2) / diagonal if diagonal else 0.0
    wa, ha = max(first[2] - first[0], 1e-6), max(first[3] - first[1], 1e-6)
    wb, hb = max(second[2] - second[0], 1e-6), max(second[3] - second[1], 1e-6)
    v = (4 / math.pi ** 2) * (math.atan(wb / hb) - math.atan(wa / ha)) ** 2
    denominator = 1 - overlap + v
    alpha = v / denominator if denominator else 0.0
    return overlap - centre - alpha * v


def centre_distance(first: BBox, second: BBox) -> float:
    acx, acy = (first[0] + first[2]) / 2, (first[1] + first[3]) / 2
    bcx, bcy = (second[0] + second[2]) / 2, (second[1] + second[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def center(bbox: BBox) -> list[int]:
    return [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]


def bottom_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def serialise_bbox(bbox: BBox) -> list[int]:
    return [round(value) for value in bbox]
