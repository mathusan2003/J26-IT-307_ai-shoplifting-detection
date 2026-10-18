"""Privacy-preserving zone, occupancy, dwell, and transition telemetry."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import cv2


class RetailAnalytics:
    def __init__(self, zones: Mapping[str, Sequence[Sequence[float]]], fps: float = 30.0) -> None:
        self.zones = {name: tuple(tuple(float(v) for v in point) for point in polygon)
                      for name, polygon in zones.items()}
        self.fps = fps
        self.current: dict[int, str | None] = {}
        self.entered_at: dict[tuple[int, str], float] = {}
        self.dwell: Counter[str] = Counter()
        self.transitions: list[dict[str, object]] = []
        self.entry_count = 0
        self.exit_count = 0

    def zone_for(self, center: Sequence[float]) -> str | None:
        point = (float(center[0]), float(center[1]))
        for name, polygon in self.zones.items():
            contour = __import__("numpy").array(polygon, dtype="float32")
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return name
        return None

    def update(self, person_id: int, center: Sequence[float], timestamp: float) -> str | None:
        zone = self.zone_for(center)
        previous = self.current.get(person_id)
        if zone != previous:
            if previous is not None:
                entered = self.entered_at.pop((person_id, previous), timestamp)
                self.dwell[previous] += max(0.0, timestamp - entered)
                self.transitions.append({"person_id": person_id, "from": previous,
                                         "to": zone, "timestamp": timestamp})
                if previous.lower() in {"entrance", "entry/exit zone"}:
                    self.exit_count += 1
            if zone is not None:
                self.entered_at[(person_id, zone)] = timestamp
                if zone.lower() in {"entrance", "entry/exit zone"} and previous is None:
                    self.entry_count += 1
        self.current[person_id] = zone
        return zone

    def snapshot(self, active_ids: Sequence[int], timestamp: float) -> dict[str, object]:
        dwell = dict(self.dwell)
        for (person_id, zone), started in self.entered_at.items():
            del person_id
            dwell[zone] = dwell.get(zone, 0.0) + max(0.0, timestamp - started)
        return {"active_occupancy": len(active_ids), "entry_count": self.entry_count,
                "exit_count": self.exit_count, "dwell_seconds": dwell,
                "transitions": self.transitions[-100:]}


class TelemetryWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.frames: list[dict[str, object]] = []

    def append(self, frame_id: int, timestamp: float, tracks: Sequence[Mapping[str, object]],
               analytics: dict[str, object]) -> None:
        mins, secs = divmod(timestamp, 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{int(hours):02d}:{int(mins):02d}:{secs:06.3f}"
        self.frames.append({"frame_id": frame_id, "timestamp": time_str,
                            "tracks": list(tracks), "analytics": analytics})

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.frames, indent=2), encoding="utf-8")
