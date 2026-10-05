"""Optional YOLOv8 person detector adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class PersonDetector:
    """Filter YOLOv8 predictions down to COCO class 0 (person)."""
    def __init__(self, weights: str | Path = "models/yolov8n.pt", confidence: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install ultralytics to run direct YOLO detection") from exc
        self.model = YOLO(str(weights))
        self.confidence = confidence

    def detect(self, frame: Any) -> list[dict[str, object]]:
        result = self.model(frame, classes=[0], conf=self.confidence, verbose=False)[0]
        return [{"class": "person", "confidence": float(box.conf[0]),
                 "bbox": [float(value) for value in box.xyxy[0].tolist()]}
                for box in result.boxes]
