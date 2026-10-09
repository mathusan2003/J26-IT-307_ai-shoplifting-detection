"""Draw track IDs, boxes, and short trajectories onto video frames."""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from .utils import as_bbox
from .zone_analytics import ZoneAnalytics

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_TRAIL_LENGTH = 30


def require_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("Install opencv-python-headless to draw tracking overlays") from _IMPORT_ERROR


def colour_for_id(person_id: int) -> tuple[int, int, int]:
    """Stable BGR colour so the same ID keeps the same overlay colour."""
    return ((37 * person_id) % 256, (17 * person_id + 80) % 256, (73 * person_id + 40) % 256)


def zone_label(analytics: ZoneAnalytics | None, person_id: int, bbox: list[int], timestamp: float) -> str | None:
    if analytics is None:
        return None
    state = analytics.update(person_id, as_bbox(bbox), timestamp)
    return state.current_zone


def draw_tracks(
    frame: Any,
    tracks: Iterable[Mapping[str, Any]],
    trails: dict[int, deque[tuple[int, int]]] | None = None,
    trail_length: int = DEFAULT_TRAIL_LENGTH,
    labels: Mapping[int, str] | None = None,
) -> Any:
    """Paint boxes, IDs, and polylines onto a BGR frame. Mutates and returns ``frame``."""
    require_opencv()
    if trails is None:
        trails = defaultdict(lambda: deque(maxlen=trail_length))
    labels = labels or {}
    for track in tracks:
        person_id = int(track["person_id"])
        x1, y1, x2, y2 = [int(value) for value in track["bbox"]]
        cx, cy = [int(value) for value in track["center"]]
        colour = colour_for_id(person_id)
        trails[person_id].append((cx, cy))
        points = list(trails[person_id])
        if len(points) >= 2:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, colour, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        caption = f"ID {person_id}"
        zone = labels.get(person_id)
        if zone:
            caption = f"{caption} | {zone}"
        cv2.putText(frame, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return frame


def annotate_frames(
    frames: Iterable[tuple[int, Any]],
    tracks_by_frame: Mapping[int, list[Mapping[str, Any]]],
    output_path: Path,
    fps: float = 30.0,
    analytics: ZoneAnalytics | None = None,
    trail_length: int = DEFAULT_TRAIL_LENGTH,
) -> int:
    """Write an annotated video. Returns the number of frames written."""
    require_opencv()
    writer = None
    written = 0
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_length))
    try:
        for frame_id, image in frames:
            if writer is None:
                height, width = image.shape[:2]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1.0, float(fps)),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}")
            tracks = tracks_by_frame.get(frame_id, [])
            labels = {}
            timestamp = frame_id / max(1.0, float(fps))
            for track in tracks:
                zone = zone_label(analytics, int(track["person_id"]), list(track["bbox"]), timestamp)
                if zone:
                    labels[int(track["person_id"])] = zone
            writer.write(draw_tracks(image, tracks, trails, trail_length, labels))
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return written


def iter_video_frames(video_path: Path, frame_id_start: int = 0) -> Iterable[tuple[int, Any]]:
    require_opencv()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield frame_id_start + index, image
            index += 1
    finally:
        capture.release()
