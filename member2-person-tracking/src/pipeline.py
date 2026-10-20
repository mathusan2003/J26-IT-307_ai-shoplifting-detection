"""Load detection JSON and run the existing ByteTracker over a frame sequence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tracker import ByteTracker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "tracker_config.yaml"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_tracker_settings(path: Path = DEFAULT_CONFIG) -> dict[str, float]:
    settings: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        settings[key.strip()] = float(value.strip())
    return settings


def make_tracker(settings: dict[str, float] | None = None) -> ByteTracker:
    values = settings or {}
    return ByteTracker(
        high_threshold=values.get("high_confidence_threshold", values.get("track_thresh", 0.2)),
        low_threshold=values.get("low_confidence_threshold", 0.1),
        match_iou_threshold=values.get("match_iou_threshold", values.get("match_thresh", 0.2)),
        max_missed_frames=int(values.get("max_missed_frames", values.get("track_buffer", 60))),
        reentry_frames=int(values.get("reentry_buffer", values.get("reentry_frames", 90))),
        new_track_threshold=values.get("new_track_threshold", values.get("track_thresh", 0.2)),
    )


def sorted_detection_frames(source: Any) -> list[dict[str, Any]]:
    frames = source if isinstance(source, list) else [source]
    return sorted(frames, key=lambda item: item["frame_id"])


def track_detection_sequence(
    frames: list[dict[str, Any]],
    tracker: ByteTracker | None = None,
) -> list[dict[str, Any]]:
    tracker = tracker or ByteTracker()
    results: list[dict[str, Any]] = []
    for frame in sorted_detection_frames(frames):
        people = [item for item in frame.get("detections", []) if item.get("class") == "person"]
        results.append({
            "frame_id": frame["frame_id"],
            "tracks": [track.to_contract() for track in tracker.update(people)],
        })
    return results
