"""Geometry used by zone assignment and behaviour measurements."""

from __future__ import annotations

import math
from typing import Sequence


def distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(second[0] - first[0], second[1] - first[1])


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y, inside = point[0], point[1], False
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        x1, y1, x2, y2 = previous[0], previous[1], current[0], current[1]
        if (y1 > y) != (y2 > y) and x <= (x2 - x1) * (y - y1) / (y2 - y1 + 0.0) + x1:
            inside = not inside
    return inside


def zone_for_point(
    point: Sequence[float], zones: dict[str, list[list[float]]]
) -> str | None:
    """Return the first zone polygon that contains the point."""
    for name, polygon in zones.items():
        if point_in_polygon(point, polygon):
            return name
    return None


def bottom_center(bbox: Sequence[float]) -> tuple[float, float]:
    """Foot point used for zone membership. Matches Member 2's zone anchor."""
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union of two boxes."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    first_area = bbox_area(first)
    second_area = bbox_area(second)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def is_shelf(zone: str | None) -> bool:
    """A shelf zone is any zone whose name contains 'shelf'."""
    return bool(zone) and "shelf" in zone.lower()


def shelf_key(zone: str | None) -> str | None:
    """Normalise a shelf name so 'Shelf_A' and 'Shelf A' stay distinct."""
    if not is_shelf(zone):
        return None
    return zone.strip().lower()
