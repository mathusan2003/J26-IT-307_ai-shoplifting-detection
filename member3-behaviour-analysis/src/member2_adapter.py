"""Normalise Member 2 tracking output for Member 3 analysis.

Member 2 writes several JSON shapes. The shared contract is ``frame_id`` plus
``tracks`` with ``person_id``, ``bbox``, and ``center``. Video telemetry also
adds a numeric ``timestamp``, ``zone``, ``trajectory``, and ``analytics``.
The dataset runner may add ``confidence`` and zone flags. This module maps
those files onto the contract Member 3 already analyses.

It does not run person detection or tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from behaviour import analyse_tracking_frames
from tracking_input import load_tracking_sequence


def format_elapsed_s(value: float | int) -> str:
    """Format wall-clock or video elapsed seconds the way live frames do."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("elapsed seconds must be a number")
    if value < 0:
        raise ValueError("elapsed seconds must not be negative")
    return f"{float(value):.3f}"


def frame_from_member2_tracks(
    frame_id: int,
    tracks: Sequence[dict[str, Any]] | None,
    elapsed_s: float | int | None = None,
    timestamp: Any = None,
) -> dict[str, Any]:
    """Build one analysis frame from Member 2 tracker rows.

    ``elapsed_s`` is real elapsed time. Use it on the webcam path so
    loitering is not computed as ``frame_id / 25``.
    """
    frame: dict[str, Any] = {
        "frame_id": _as_index(frame_id, "frame_id"),
        "tracks": normalize_tracks(tracks, skip_invalid=True),
    }
    stamped = _coerce_timestamp(elapsed_s if elapsed_s is not None else timestamp)
    if stamped is not None:
        frame["timestamp"] = stamped
    return frame


def normalize_track(track: dict[str, Any], *, skip_invalid: bool = False) -> dict[str, Any] | None:
    """Keep contract fields, plus zone when Member 2 provided one."""
    try:
        if not isinstance(track, dict):
            raise ValueError("track must be a JSON object")
        person_id = _as_index(track.get("person_id"), "person_id")
        cleaned: dict[str, Any] = {"person_id": person_id}
        if _present(track, "bbox"):
            cleaned["bbox"] = _as_bbox(track["bbox"])
        if _present(track, "center"):
            cleaned["center"] = _as_point(track["center"], "center")
        elif _present(track, "position"):
            cleaned["center"] = _as_point(track["position"], "position")
        elif "bbox" in cleaned:
            cleaned["center"] = _center_from_bbox(cleaned["bbox"])
        else:
            raise ValueError(f"person {person_id} is missing center, position, and bbox")
        if _present(track, "zone"):
            cleaned["zone"] = _as_zone(track["zone"])
        elif _present(track, "shelf"):
            cleaned["zone"] = _as_zone(track["shelf"])
        track_time = _coerce_timestamp(track.get("timestamp") if "timestamp" in track else None)
        if track_time is not None:
            cleaned["timestamp"] = track_time
        return cleaned
    except (TypeError, ValueError):
        if skip_invalid:
            return None
        raise


def normalize_tracks(
    tracks: Sequence[Any] | None, *, skip_invalid: bool = True
) -> list[dict[str, Any]]:
    """Return valid tracks. Missing or non-list ``tracks`` become empty."""
    if tracks is None:
        return []
    if not isinstance(tracks, list):
        if skip_invalid:
            return []
        raise ValueError("tracks must be a list")
    cleaned: list[dict[str, Any]] = []
    for track in tracks:
        item = normalize_track(track, skip_invalid=skip_invalid)
        if item is not None:
            cleaned.append(item)
    return cleaned


def normalize_frame(frame: Any, *, skip_invalid: bool = True) -> dict[str, Any] | None:
    """Drop telemetry extras and accept Member 2 numeric timestamps."""
    try:
        if not isinstance(frame, dict):
            raise ValueError("tracking frame must be a JSON object")
        frame_id = _as_index(frame.get("frame_id"), "frame_id")
        cleaned = frame_from_member2_tracks(
            frame_id,
            frame.get("tracks"),
            timestamp=frame.get("timestamp") if "timestamp" in frame else None,
        )
        return cleaned
    except (TypeError, ValueError):
        if skip_invalid:
            return None
        raise


def normalize_sequence(
    source: Any,
    *,
    skip_invalid: bool = True,
    source_fps: float | None = None,
) -> list[dict[str, Any]]:
    """Normalise a Member 2 payload into an ordered list of analysis frames."""
    frames = _as_frame_list(source)
    cleaned: list[dict[str, Any]] = []
    for frame in frames:
        item = normalize_frame(frame, skip_invalid=skip_invalid)
        if item is not None:
            cleaned.append(item)
    if source_fps is not None:
        cleaned = apply_source_fps(cleaned, source_fps)
    return cleaned


def apply_source_fps(frames: Sequence[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    """Stamp frames that have no timestamp using Member 2's video frame rate.

    Live webcam frames should pass ``elapsed_s`` instead. This helper is for
    recorded tracking JSON that only has ``frame_id``.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("source_fps must be a positive number")
    stamped: list[dict[str, Any]] = []
    for frame in frames:
        item = dict(frame)
        if "timestamp" not in item or item["timestamp"] is None:
            item["timestamp"] = format_elapsed_s(int(item["frame_id"]) / float(fps))
        stamped.append(item)
    return stamped


def load_member2_tracking_sequence(
    path: str | Path,
    *,
    skip_invalid: bool = True,
    source_fps: float | None = None,
) -> list[dict[str, Any]]:
    """Load Member 2 tracking JSON and return contract frames for analysis."""
    return normalize_sequence(
        load_tracking_sequence(path),
        skip_invalid=skip_invalid,
        source_fps=source_fps,
    )


def analyse_member2_frames(
    source: Any,
    detections: Sequence[dict[str, Any]] | None = None,
    settings: Any = None,
    zones: dict[str, list[list[float]]] | None = None,
    *,
    skip_invalid: bool = True,
    source_fps: float | None = None,
) -> list[dict[str, Any]]:
    """Analyse Member 2 frames without running detection or tracking again."""
    frames = normalize_sequence(source, skip_invalid=skip_invalid, source_fps=source_fps)
    return analyse_tracking_frames(frames, detections, settings, zones)


def _as_frame_list(source: Any) -> list[Any]:
    if isinstance(source, dict) and isinstance(source.get("frames"), list):
        return source["frames"]
    if isinstance(source, dict):
        return [source]
    if isinstance(source, list):
        return source
    raise ValueError(
        "tracking payload must be a frame object, a list of frames, or an object with a frames list"
    )


def _as_index(value: Any, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be an integer")
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        value = value.item()
    if isinstance(value, int):
        if label == "frame_id" and value < 0:
            raise ValueError("frame_id must be a non-negative integer")
        return value
    if isinstance(value, float) and value.is_integer():
        return _as_index(int(value), label)
    raise ValueError(f"{label} must be an integer")


def _as_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be [x_min, y_min, x_max, y_max]")
    box = [float(_as_number(item, "bbox")) for item in value]
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("bbox maximums must be greater than or equal to minimums")
    return box


def _as_point(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [x, y]")
    return [float(_as_number(value[0], label)), float(_as_number(value[1], label))]


def _center_from_bbox(bbox: Sequence[float]) -> list[float]:
    return [(float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2]


def _as_zone(value: Any) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("zone must be a non-empty string")
    return value.strip()


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if hasattr(value, "item"):
            return _as_number(value.item(), label)
        raise ValueError(f"{label} coordinates must be numbers")
    return float(value)


def _coerce_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("timestamp must be a non-empty string or a number of seconds")
    if isinstance(value, (int, float)):
        return format_elapsed_s(value)
    if isinstance(value, str):
        if value.strip() == "":
            raise ValueError("timestamp must be a non-empty string or a number of seconds")
        return value.strip()
    raise ValueError("timestamp must be a non-empty string or a number of seconds")


def _present(track: dict[str, Any], field: str) -> bool:
    return field in track and track[field] is not None
