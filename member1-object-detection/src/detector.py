
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parents[2] / "yolov8n.pt"


class ObjectDetector:
    """Detect objects and produce the team's agreed JSON structure."""

    def __init__(self, model_path=MODEL_PATH, confidence=0.25):
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect_frame(self, frame, frame_id):
        if frame is None:
            raise ValueError("Input frame cannot be empty.")

        if not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer.")

        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "class": self.model.names[class_id],
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox": [
                    int(value)
                    for value in box.xyxy[0].tolist()
                ]
            })

        return {
            "frame_id": frame_id,
            "detections": detections
        }

    def detect_image(self, image_path, frame_id=0):
        frame = cv2.imread(str(image_path))

        if frame is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        return self.detect_frame(frame, frame_id)


def save_json(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
