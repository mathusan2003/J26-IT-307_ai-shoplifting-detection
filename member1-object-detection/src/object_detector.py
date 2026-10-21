"""Pretrained YOLO object detector for Member 1.

This module reports the classes the loaded model actually supports. It does
not track people, score behaviour, or decide shoplifting. Person-only
Member 2 weights are rejected so they are not silently reused as a general
object detector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
MEMBER2_COCO_WEIGHTS = REPO_ROOT / "member2-person-tracking" / "models" / "yolov8n.pt"
LOCAL_COCO_WEIGHTS = MODULE_ROOT / "models" / "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IMGSZ = 640
DEFAULT_NMS_IOU = 0.50
DEFAULT_MAX_DET = 50
PROTOTYPE_NOTE = (
    "Member 1 pretrained YOLO prototype. Detections are COCO classes from the "
    "loaded weights. This is not retail SKU recognition, concealment, "
    "object-to-person ownership, or shoplifting."
)


class WeightsNotFoundError(FileNotFoundError):
    """Raised when the requested YOLO weights file is not on disk."""


class IncompatibleWeightsError(ValueError):
    """Raised when the weights are person-only or otherwise not general objects."""


class InferenceError(RuntimeError):
    """Raised when the model is loaded but a frame cannot be processed."""


def load_detector_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Read optional YAML settings. Missing files keep the code defaults."""
    settings = {
        "confidence": DEFAULT_CONFIDENCE,
        "imgsz": DEFAULT_IMGSZ,
        "nms_iou": DEFAULT_NMS_IOU,
        "max_det": DEFAULT_MAX_DET,
        "device": "auto",
        "classes": [],
    }
    config_path = Path(path) if path is not None else MODULE_ROOT / "configs" / "detector.yaml"
    if not config_path.is_file():
        return settings
    try:
        import yaml
    except ImportError:
        return settings
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return settings
    for key in settings:
        if key in payload and payload[key] is not None:
            settings[key] = payload[key]
    return settings


def resolve_object_weights(weights: str | Path | None = None) -> Path:
    """Resolve a general-object weight file. Never fall back to person-only models."""
    if weights is not None:
        path = Path(weights)
        if not path.is_absolute():
            for root in (Path.cwd(), MODULE_ROOT, REPO_ROOT):
                candidate = root / path
                if candidate.is_file():
                    return candidate
            path = (MODULE_ROOT / path).resolve()
        if not path.is_file():
            raise WeightsNotFoundError(
                f"Object-detection weights not found: {path}. "
                "Place COCO yolov8n.pt under member1-object-detection/models/ "
                "or pass --weights. Person-only Member 2 files are not substituted."
            )
        return path

    for candidate in (LOCAL_COCO_WEIGHTS, MEMBER2_COCO_WEIGHTS):
        if candidate.is_file():
            return candidate
    searched = f"{LOCAL_COCO_WEIGHTS} or {MEMBER2_COCO_WEIGHTS}"
    raise WeightsNotFoundError(
        "No compatible pretrained object-detection weights found. Expected "
        f"{searched}. Do not use yolov8n-person-track.pt; that model is person-only. "
        "Supply COCO yolov8n.pt or pass an explicit --weights path. "
        "This prototype will not download weights."
    )


def is_general_object_model(names: dict[Any, Any] | None) -> bool:
    """True when the model has more than the person/item class."""
    if not names:
        return False
    labels = [str(value).strip().lower() for value in names.values()]
    non_person = [label for label in labels if label not in {"person", "people", "item", "pedestrian"}]
    return len(labels) >= 2 and len(non_person) >= 1


def resolve_device(requested: str | None = "auto") -> str:
    """Use CUDA only when requested and actually available."""
    choice = (requested or "auto").strip().lower()
    if choice in {"cpu", "none"}:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    wants_cuda = choice in {"auto", "cuda", "0", "gpu"}
    if wants_cuda and torch.cuda.is_available():
        return "0"
    return "cpu"


def empty_frame_result(
    *,
    frame_id: int = 0,
    timestamp: float = 0.0,
    model: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Stable JSON when a frame has no detections or the detector is unavailable."""
    return {
        "frame_id": int(frame_id),
        "timestamp": round(float(timestamp), 3),
        "model": model,
        "note": PROTOTYPE_NOTE,
        "error": error,
        "detections": [],
    }


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> dict[str, float] | None:
    """Clamp a box to the frame. Degenerate boxes are dropped, not invented."""
    if width <= 0 or height <= 0 or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    left = max(0.0, min(float(width), left))
    right = max(0.0, min(float(width), right))
    top = max(0.0, min(float(height), top))
    bottom = max(0.0, min(float(height), bottom))
    if right - left < 1.0 or bottom - top < 1.0:
        return None
    return {
        "x1": round(left, 1),
        "y1": round(top, 1),
        "x2": round(right, 1),
        "y2": round(bottom, 1),
    }


def pipeline_detections(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Map prototype detections onto the team contract: class, confidence, bbox list."""
    rows = []
    for item in (result or {}).get("detections") or []:
        box = item.get("bbox") or {}
        rows.append(
            {
                "class": item.get("class_name"),
                "confidence": item.get("confidence"),
                "bbox": [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")],
            }
        )
    return rows


class ObjectDetector:
    """Load one YOLO model and run it on successive frames."""

    def __init__(
        self,
        weights: str | Path | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        imgsz: int = DEFAULT_IMGSZ,
        max_det: int = DEFAULT_MAX_DET,
        nms_iou: float = DEFAULT_NMS_IOU,
        device: str = "auto",
        classes: Sequence[Any] | None = None,
        model: Any | None = None,
    ):
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.nms_iou = float(nms_iou)
        self.device = resolve_device(device)
        if model is not None:
            self.weights = Path(weights) if weights is not None else Path("mock")
            self.model = model
            self.names = dict(getattr(model, "names", {}) or {})
        else:
            path = resolve_object_weights(weights)
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Install ultralytics to run Member 1 object detection "
                    "(see member1-object-detection/requirements.txt)."
                ) from exc
            self.weights = path
            self.model = YOLO(str(path))
            self.names = dict(getattr(self.model, "names", {}) or {})
        if not is_general_object_model(self.names):
            labels = ", ".join(str(value) for value in self.names.values()) or "unknown"
            raise IncompatibleWeightsError(
                f"{self.weights} is not a general object-detection model "
                f"(classes: {labels}). Use COCO yolov8n.pt. Person-only "
                "Member 2 weights are not accepted here."
            )
        self.class_ids = self._resolve_class_filter(classes)

    @property
    def model_name(self) -> str:
        return self.weights.name

    def supported_classes(self) -> dict[int, str]:
        return {int(key): str(value) for key, value in self.names.items()}

    def _resolve_class_filter(self, selected: Sequence[Any] | None) -> list[int] | None:
        if not selected:
            return None
        lookup = {str(value).strip().lower(): int(key) for key, value in self.names.items()}
        ids: list[int] = []
        unknown: list[str] = []
        for item in selected:
            if item is None or item == "":
                continue
            if isinstance(item, int) or str(item).isdigit():
                class_id = int(item)
                if class_id not in self.names:
                    unknown.append(str(item))
                    continue
                ids.append(class_id)
                continue
            key = str(item).strip().lower()
            if key not in lookup:
                unknown.append(str(item))
                continue
            ids.append(lookup[key])
        if unknown:
            supported = ", ".join(f"{idx}:{name}" for idx, name in sorted(self.supported_classes().items())[:12])
            raise ValueError(
                "Unknown class filter {0}. The loaded model supports: {1}…".format(
                    ", ".join(unknown), supported
                )
            )
        return ids or None

    def detect_objects(
        self,
        frame: Any,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> dict[str, Any]:
        """Run the loaded model once and return structured detections for this frame."""
        if frame is None or not hasattr(frame, "shape") or len(getattr(frame, "shape", ())) < 2:
            return empty_frame_result(
                frame_id=frame_id,
                timestamp=timestamp,
                model=self.model_name,
                error="Empty or invalid frame. No detections were produced.",
            )
        height, width = int(frame.shape[0]), int(frame.shape[1])
        try:
            result = self.model(
                frame,
                conf=self.confidence,
                iou=self.nms_iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                classes=self.class_ids,
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001 - surface a clear inference error
            raise InferenceError(f"Member 1 YOLO inference failed on frame {frame_id}: {exc}") from exc

        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for index, box in enumerate(boxes, start=1):
                class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
                class_name = str(self.names.get(class_id, f"class_{class_id}"))
                confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                if confidence < self.confidence:
                    continue
                if self.class_ids is not None and class_id not in self.class_ids:
                    continue
                raw_bbox = [float(value) for value in box.xyxy[0].tolist()]
                bbox = clip_bbox(raw_bbox, width, height)
                if bbox is None:
                    continue
                detections.append(
                    {
                        "detection_id": index,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    }
                )
        payload = empty_frame_result(
            frame_id=frame_id,
            timestamp=timestamp,
            model=self.model_name,
        )
        payload["detections"] = detections
        payload["width"] = width
        payload["height"] = height
        payload["device"] = self.device
        payload["class_count"] = len(self.names)
        return payload
