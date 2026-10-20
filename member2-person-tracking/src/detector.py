"""Optional YOLOv8 person detector adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

MODULE_ROOT = Path(__file__).resolve().parents[1]
PERSON_CLASS_ID = 0
ACTIVE_WEIGHTS = "models/yolov8n-person-track.pt"
MODEL_PRIORITY = [
    ACTIVE_WEIGHTS,
    "models/yolov8n-cam-multiperson.pt",
    "models/yolov8n.pt",
]


def resolve_weights(weights: str | Path | None = None, roots: Sequence[Path] | None = None) -> Path:
    """Pick the first weight file that exists, preferring an explicit path.

    Missing files fall through ``MODEL_PRIORITY``. Search roots default to the
    working directory and this component folder.
    """
    requested = Path(MODEL_PRIORITY[0] if weights is None else weights)
    search_roots = [Path.cwd(), MODULE_ROOT] if roots is None else list(roots)
    candidates: list[Path] = [requested] if requested.is_absolute() else [root / requested for root in search_roots]
    for relative in MODEL_PRIORITY:
        candidates.extend(root / relative for root in search_roots)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return requested


def is_person_box(bbox: Sequence[float], frame_width: int, frame_height: int) -> bool:
    """Keep standing, seated, and short pedestrians; drop banners, produce piles, and specks.

    Aspect ratio is width/height. Seated shoppers are often near-square (~0.7–1.4).
    Produce shelves and hanging signs are much wider than they are tall.
    """
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
    min_height = max(18.0, 0.02 * frame_height)
    if height < min_height or width < 10:
        return False
    ratio = width / height
    if ratio < 0.14 or ratio > 1.55:
        return False
    if width * height > 0.45 * frame_width * frame_height:
        return False
    return True


class PersonDetector:
    """Filter YOLOv8 predictions down to COCO class 0 (person)."""

    def __init__(
        self,
        weights: str | Path = ACTIVE_WEIGHTS,
        confidence: float = 0.20,
        imgsz: int = 1080,
        max_det: int = 300,
        nms_iou: float = 0.60,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install ultralytics to run direct YOLO detection") from exc
        path = resolve_weights(weights)
        requested = Path(weights)
        if not requested.is_file() and path != requested:
            print(f"weights not found: {weights}; using {path}")
        self.weights = path
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.imgsz = imgsz
        self.max_det = max_det
        self.nms_iou = nms_iou

    def detect(self, frame: Any) -> list[dict[str, object]]:
        result = self.model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )[0]
        frame_h, frame_w = (frame.shape[0], frame.shape[1]) if hasattr(frame, "shape") else (1080, 1920)
        people = []
        for box in result.boxes:
            bbox = [float(value) for value in box.xyxy[0].tolist()]
            if not is_person_box(bbox, frame_w, frame_h):
                continue
            people.append({"class": "person", "confidence": float(box.conf[0]), "bbox": bbox})
        return people
