"""Retail-zone membership and per-person dwell/transition analytics."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import BBox, bottom_center


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y, inside = point[0], point[1], False
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        x1, y1, x2, y2 = previous[0], previous[1], current[0], current[1]
        if (y1 > y) != (y2 > y) and x <= (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@dataclass
class PersonZoneState:
    current_zone: str | None = None
    entered_at: float | None = None
    zone_history: list[str] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)


class ZoneAnalytics:
    def __init__(self, zones_path: str | Path):
        raw = json.loads(Path(zones_path).read_text(encoding="utf-8"))
        self.zones: dict[str, list[list[float]]] = raw["zones"] if isinstance(raw, dict) else raw
        self._base_zones = {name: [list(point) for point in polygon] for name, polygon in self.zones.items()}
        self.start_zone = raw.get("start_zone") if isinstance(raw, dict) else None
        self.end_zone = raw.get("end_zone") if isinstance(raw, dict) else None
        self.reference_size = tuple(raw["reference_size"]) if isinstance(raw, dict) and raw.get("reference_size") else None
        self.people: dict[int, PersonZoneState] = {}

    def configure_frame_size(self, width: int, height: int) -> None:
        if not self.reference_size:
            return
        reference_width, reference_height = self.reference_size
        if reference_width <= 0 or reference_height <= 0:
            raise ValueError("Zone reference_size must contain positive width and height")
        scale_x, scale_y = width / reference_width, height / reference_height
        self.zones = {
            name: [[point[0] * scale_x, point[1] * scale_y] for point in polygon]
            for name, polygon in self._base_zones.items()
        }

    def zone_for_bbox(self, bbox: BBox) -> str | None:
        point = bottom_center(bbox)
        return next((name for name, polygon in self.zones.items() if point_in_polygon(point, polygon)), None)

    def update(self, person_id: int, bbox: BBox, timestamp_seconds: float) -> PersonZoneState:
        state = self.people.setdefault(person_id, PersonZoneState())
        new_zone = self.zone_for_bbox(bbox)
        if new_zone != state.current_zone:
            if state.current_zone is not None:
                state.transitions.append({"from": state.current_zone, "to": new_zone, "exited_at": timestamp_seconds})
            state.current_zone = new_zone
            state.entered_at = timestamp_seconds if new_zone is not None else None
            if new_zone is not None:
                state.zone_history.append(new_zone)
        return state

    def dwell_time(self, person_id: int, timestamp_seconds: float) -> float:
        state = self.people[person_id]
        return 0.0 if state.entered_at is None else max(0.0, timestamp_seconds - state.entered_at)

    def reached_end_zone(self, person_id: int) -> bool:
        state = self.people[person_id]
        return self.end_zone is not None and self.end_zone in state.zone_history
