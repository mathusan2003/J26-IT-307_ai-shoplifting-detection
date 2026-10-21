"""Load Member 1's object detector without changing Member 2 person tracking.

Object boxes are not person IDs, behaviour scores, concealment, or shoplifting.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

MEMBER1_SRC = Path(__file__).resolve().parents[2] / "member1-object-detection" / "src"
PROTOTYPE_NOTE = (
    "Member 1 pretrained YOLO prototype. COCO classes only — not retail SKUs, "
    "concealment, or shoplifting."
)

_DETECTOR_CLS: Any | None = None
_DRAW = None
_EMPTY = None


def empty_object_state(
    *,
    frame_id: int = 0,
    timestamp: float = 0.0,
    error: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    return {
        "available": error is None,
        "error": error,
        "frame_id": int(frame_id),
        "timestamp": round(float(timestamp), 3),
        "model": model,
        "note": PROTOTYPE_NOTE,
        "detections": [],
        "infer_fps": None,
        "device": None,
        "class_count": 0,
    }


def _load_member1() -> tuple[Any, Any]:
    global _DETECTOR_CLS, _DRAW
    if _DETECTOR_CLS is not None and _DRAW is not None:
        return _DETECTOR_CLS, _DRAW
    if not MEMBER1_SRC.is_dir():
        raise FileNotFoundError(f"Member 1 source folder not found: {MEMBER1_SRC}")
    src = str(MEMBER1_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    from object_detector import ObjectDetector  # type: ignore
    from visualize import draw_detections  # type: ignore

    _DETECTOR_CLS = ObjectDetector
    _DRAW = draw_detections
    return ObjectDetector, draw_detections


class DashboardObjects:
    """Run Member 1 once per shared live-session frame."""

    def __init__(self) -> None:
        self.detector: Any | None = None
        self.error: str | None = None
        self.infer_fps: float | None = None
        self._loaded = False

    def _ensure(self) -> Any | None:
        if self._loaded:
            return self.detector
        self._loaded = True
        try:
            cls, _ = _load_member1()
            self.detector = cls()
            self.error = None
        except Exception as exc:  # noqa: BLE001 - keep tracking alive
            self.detector = None
            self.error = f"Member 1 object detection is unavailable: {exc}"
        return self.detector

    def update(
        self,
        frame: Any,
        frame_id: int,
        timestamp: float,
        source: str | None,
    ) -> tuple[dict[str, Any], Any | None]:
        import time

        detector = self._ensure()
        if detector is None:
            return empty_object_state(frame_id=frame_id, timestamp=timestamp, error=self.error), None
        try:
            started = time.perf_counter()
            payload = detector.detect_objects(frame, frame_id=frame_id, timestamp=timestamp)
            elapsed = time.perf_counter() - started
            instant = 1.0 / max(elapsed, 1e-6)
            self.infer_fps = instant if self.infer_fps is None else (0.8 * self.infer_fps + 0.2 * instant)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return empty_object_state(frame_id=frame_id, timestamp=timestamp, error=self.error), None
        _, draw = _load_member1()
        canvas = draw(frame, payload, source=source, infer_fps=self.infer_fps)
        state = {
            "available": True,
            "error": None,
            "frame_id": payload.get("frame_id", frame_id),
            "timestamp": payload.get("timestamp", timestamp),
            "model": payload.get("model"),
            "note": payload.get("note") or PROTOTYPE_NOTE,
            "detections": payload.get("detections") or [],
            "infer_fps": None if self.infer_fps is None else round(float(self.infer_fps), 1),
            "device": payload.get("device"),
            "class_count": payload.get("class_count") or 0,
            "width": payload.get("width"),
            "height": payload.get("height"),
        }
        return state, canvas


def member3_object_rows(object_state: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Pass Member 1 boxes into Member 3. None means the object detector did not run."""
    if not object_state:
        return None
    if object_state.get("error"):
        return None
    return list(object_state.get("detections") or [])


def log_m3_objects(stage: str, **fields: Any) -> None:
    parts = " ".join(f"{key}={value!r}" for key, value in fields.items())
    print(f"[M3-OBJECTS] {stage}: {parts}", flush=True)
