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


def iou(first: BBox, second: BBox) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    union = first_area + max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1]) - intersection
    return intersection / union if union else 0.0


def center(bbox: BBox) -> list[int]:
    return [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]


def bottom_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def serialise_bbox(bbox: BBox) -> list[int]:
    return [round(value) for value in bbox]
