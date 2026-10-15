"""Configuration for the retail multi-person tracking application."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackingConfig:
    model_weights: Path = Path("models/yolov8n-person-track.pt")
    camera_source: str = "Dataset/videos/not_shoplifting1.mp4"
    output_json: Path = Path("outputs/not_shoplifting1_tracking.json")
    output_video: Path = Path("outputs/not_shoplifting1_tracked.mp4")
    high_threshold: float = 0.6
    low_threshold: float = 0.1
    new_track_threshold: float = 0.7
    ciou_distance_threshold: float = 0.3
    track_buffer: int = 90
    fps: float = 30.0
    display: bool = True
    zones: dict[str, list[list[float]]] = field(default_factory=dict)

    @classmethod
    def from_zone_file(cls, path: str | Path, **overrides: object) -> "TrackingConfig":
        import json

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        zones = raw.get("zones", raw) if isinstance(raw, dict) else raw
        return cls(zones=zones, **overrides)
