"""Load and normalise Member 2 tracking frames for movement history.

A tracking sequence is an ordered list of per-frame objects. This module
accepts the shared tracking contract (frame_id, tracks, person_id, bbox,
center) and the extra fields this module needs when they are present
(timestamp, zone or shelf, position). Member 2 video telemetry may send
``timestamp`` as elapsed seconds (a JSON number). Clock strings such as
``HH:MM:SS`` are still accepted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tracking_sequence(path: str | Path) -> list[dict[str, Any]]:
    """Load one tracking frame or a sequence of frames from a JSON file.

    Accepted file shapes:

    - a single frame object
    - a list of frame objects, in the order they should be read
    - an object with a ``frames`` list
    """
    file_path = Path(path)
    try:
        # utf-8-sig accepts the BOM that Windows editors add. Member 2 reads
        # JSON the same way. Plain UTF-8 files still load.
        with file_path.open(encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"tracking file is not valid JSON: {file_path}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        frames = payload["frames"]
    elif isinstance(payload, dict):
        frames = [payload]
    elif isinstance(payload, list):
        frames = payload
    else:
        raise ValueError(
            "tracking file must contain a frame object, a list of frames, "
            "or an object with a frames list"
        )

    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame at index {index} must be a JSON object")
    return frames


def observations_from_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one tracking frame into movement observations.

    Each observation is a plain dict with person_id, frame_id, timestamp,
    position, zone, and bbox. ``bbox`` is kept when the track has one, so
    later behaviour checks can use box size without changing the movement
    history JSON. One track entry produces one observation. An empty
    ``tracks`` list produces no observations.
    """
    if not isinstance(frame, dict):
        raise ValueError("tracking frame must be a JSON object")

    frame_id = _require_frame_id(frame)
    frame_timestamp = _read_timestamp(frame, f"frame {frame_id}")
    tracks = frame.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError(f"frame {frame_id} is missing a tracks list")

    observations: list[dict[str, Any]] = []
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise ValueError(f"frame {frame_id} track {index} must be a JSON object")
        person_id = _require_person_id(track, frame_id, index)
        track_timestamp = _read_timestamp(track, f"frame {frame_id} person {person_id}")
        observations.append(
            {
                "person_id": person_id,
                "frame_id": frame_id,
                "timestamp": track_timestamp if track_timestamp is not None else frame_timestamp,
                "position": _read_position(track, frame_id, person_id),
                "zone": _read_zone(track, frame_id, person_id),
                "bbox": _read_bbox(track, frame_id, person_id),
            }
        )
    return observations


def _require_frame_id(frame: dict[str, Any]) -> int:
    if "frame_id" not in frame:
        raise ValueError("tracking frame is missing frame_id")
    frame_id = frame["frame_id"]
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return frame_id


def _require_person_id(track: dict[str, Any], frame_id: int, index: int) -> int:
    if "person_id" not in track:
        raise ValueError(f"frame {frame_id} track {index} is missing person_id")
    person_id = track["person_id"]
    if isinstance(person_id, bool) or not isinstance(person_id, int):
        raise ValueError(f"frame {frame_id} track {index} person_id must be an integer")
    return person_id


def _read_timestamp(container: dict[str, Any], label: str) -> str | None:
    if "timestamp" not in container or container["timestamp"] is None:
        return None
    timestamp = container["timestamp"]
    if isinstance(timestamp, bool):
        raise ValueError(f"{label} timestamp must be a non-empty string or a number of seconds")
    if isinstance(timestamp, (int, float)):
        if timestamp < 0:
            raise ValueError(f"{label} timestamp must not be negative")
        # Member 2 video telemetry writes elapsed seconds as a JSON number.
        return f"{float(timestamp):.3f}"
    if not isinstance(timestamp, str) or timestamp.strip() == "":
        raise ValueError(f"{label} timestamp must be a non-empty string or a number of seconds")
    return timestamp.strip()


def _read_position(track: dict[str, Any], frame_id: int, person_id: int) -> list[int | float]:
    label = f"frame {frame_id} person {person_id}"
    if _present(track, "center"):
        return _coordinate_pair(track["center"], f"{label} center")
    if _present(track, "position"):
        return _coordinate_pair(track["position"], f"{label} position")
    if _present(track, "bbox"):
        return _center_from_bbox(track["bbox"], label)
    raise ValueError(f"{label} is missing center, position, and bbox")


def _read_zone(track: dict[str, Any], frame_id: int, person_id: int) -> str | None:
    label = f"frame {frame_id} person {person_id}"
    if _present(track, "zone"):
        return _zone_name(track["zone"], f"{label} zone")
    if _present(track, "shelf"):
        return _zone_name(track["shelf"], f"{label} shelf")
    return None


def _read_bbox(
    track: dict[str, Any], frame_id: int, person_id: int
) -> list[float] | None:
    """Return a validated box, or None when the track does not include one."""
    if not _present(track, "bbox"):
        return None
    label = f"frame {frame_id} person {person_id}"
    value = track["bbox"]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} bbox must be [x_min, y_min, x_max, y_max]")
    x_min = float(_coordinate(value[0], f"{label} bbox"))
    y_min = float(_coordinate(value[1], f"{label} bbox"))
    x_max = float(_coordinate(value[2], f"{label} bbox"))
    y_max = float(_coordinate(value[3], f"{label} bbox"))
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"{label} bbox maximums must be greater than or equal to minimums")
    return [x_min, y_min, x_max, y_max]


def _present(track: dict[str, Any], field: str) -> bool:
    return field in track and track[field] is not None


def _zone_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _coordinate_pair(value: Any, label: str) -> list[int | float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [x, y]")
    return [_coordinate(value[0], label), _coordinate(value[1], label)]


def _center_from_bbox(value: Any, label: str) -> list[int | float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} bbox must be [x_min, y_min, x_max, y_max]")
    x_min = _coordinate(value[0], f"{label} bbox")
    y_min = _coordinate(value[1], f"{label} bbox")
    x_max = _coordinate(value[2], f"{label} bbox")
    y_max = _coordinate(value[3], f"{label} bbox")
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"{label} bbox maximums must be greater than or equal to minimums")
    return [
        _whole_number((x_min + x_max) / 2),
        _whole_number((y_min + y_max) / 2),
    ]


def _coordinate(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} coordinates must be numbers")
    return value


def _whole_number(value: float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
