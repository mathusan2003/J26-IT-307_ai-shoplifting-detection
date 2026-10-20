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

DEFAULT_TRAIL_LENGTH = 60


def _dashed_rectangle(frame: Any, top_left: tuple[int, int], bottom_right: tuple[int, int], colour: tuple[int, int, int]) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    for x in range(x1, x2, 10):
        cv2.line(frame, (x, y1), (min(x + 5, x2), y1), colour, 2)
        cv2.line(frame, (x, y2), (min(x + 5, x2), y2), colour, 2)
    for y in range(y1, y2, 10):
        cv2.line(frame, (x1, y), (x1, min(y + 5, y2)), colour, 2)
        cv2.line(frame, (x2, y), (x2, min(y + 5, y2)), colour, 2)


def draw_alerts(frame: Any, alerts: Iterable[str]) -> Any:
    """Paint Re-ID / camera-switch badges in the top-right of the frame."""
    require_opencv()
    y = 42
    for text in alerts:
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        x = max(12, frame.shape[1] - size[0] - 24)
        cv2.rectangle(frame, (x - 8, y - 24), (x + size[0] + 8, y + 8), (0, 140, 255), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        y += 36
    return frame


def draw_occupancy_hud(frame: Any, occupancy: Mapping[str, Any] | None, timestamp: str | None = None) -> Any:
    require_opencv()
    if occupancy is None:
        return frame
    active = occupancy.get("active_people", occupancy.get("active_occupancy", 0))
    lost = occupancy.get("temporarily_lost", 0)
    text = f"People on camera now {active}  |  Lost {lost}"
    if timestamp:
        text = f"{timestamp}  |  {text}"
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 520), 42), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def require_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("Install opencv-python-headless to draw tracking overlays") from _IMPORT_ERROR


def colour_for_id(person_id: int) -> tuple[int, int, int]:
    """Stable BGR colour so the same ID keeps the same overlay colour."""
    return ((37 * person_id) % 256, (17 * person_id + 80) % 256, (73 * person_id + 40) % 256)


def is_overlay_shelf(zone: str | None) -> bool:
    return bool(zone) and "shelf" in zone.lower()


def zone_caption(analytics: ZoneAnalytics | None, zone: str | None) -> str | None:
    if not zone or is_overlay_shelf(zone):
        return None
    if analytics and zone == analytics.start_zone:
        return f"ENTRY | {zone}"
    if analytics and zone == analytics.end_zone:
        return f"EXIT | {zone}"
    return zone


def zone_label(analytics: ZoneAnalytics | None, person_id: int, bbox: list[int], timestamp: float) -> str | None:
    if analytics is None:
        return None
    return zone_caption(analytics, analytics.update(person_id, as_bbox(bbox), timestamp).current_zone)


def draw_zones(frame: Any, analytics: ZoneAnalytics | None) -> Any:
    """Draw retail polygons, with ENTRY and EXIT highlighted."""
    require_opencv()
    if analytics is None:
        return frame
    overlay = frame.copy()
    for name, polygon in analytics.zones.items():
        if is_overlay_shelf(name):
            continue
        points = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
        if name == analytics.start_zone:
            colour, title = (40, 200, 40), f"ENTRY: {name}"
        elif name == analytics.end_zone:
            colour, title = (40, 40, 220), f"EXIT: {name}"
        else:
            colour, title = (170, 170, 170), name
        cv2.fillPoly(overlay, [points], colour)
        cv2.polylines(frame, [points], True, colour, 2)
        x, y = int(points[0][0]), int(points[0][1])
        cv2.putText(frame, title, (x + 8, min(frame.shape[0] - 10, y + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, dst=frame)
    return frame


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
        status = str(track.get("status") or "active")
        if status == "active":
            trails[person_id].append((cx, cy))
        points = list(trails[person_id])
        if len(points) >= 2:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, colour, 2)
        if status != "active":
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        caption = f"ID: #{person_id}"
        zone = labels.get(person_id)
        if zone:
            caption = f"{caption} | {zone}"
        text_y = max(22, y1 - 8)
        cv2.putText(frame, caption, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, caption, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
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
                if analytics is not None:
                    analytics.configure_frame_size(width, height)
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
                person_id = int(track["person_id"])
                if "zone" in track:
                    caption = zone_caption(analytics, track.get("zone"))
                else:
                    caption = zone_label(analytics, person_id, list(track["bbox"]), timestamp)
                if caption:
                    labels[person_id] = caption
            canvas = draw_zones(image, analytics)
            canvas = draw_tracks(canvas, tracks, trails, trail_length, labels)
            occupancy = {"active_people": sum(1 for track in tracks if str(track.get("status") or "active") == "active"),
                         "temporarily_lost": sum(1 for track in tracks if track.get("status") == "temporarily_lost")}
            writer.write(draw_occupancy_hud(canvas, occupancy, f"t={timestamp:.2f}s"))
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
