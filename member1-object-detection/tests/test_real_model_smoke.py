"""Optional real-weight smoke test. Skips when COCO yolov8n.pt is absent."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from object_detector import (
    ObjectDetector,
    WeightsNotFoundError,
    is_general_object_model,
    resolve_object_weights,
)


def _coco_path() -> Path | None:
    try:
        path = resolve_object_weights()
    except WeightsNotFoundError:
        return None
    return path if path.is_file() else None


class RealModelSmokeTests(unittest.TestCase):
    def test_local_coco_weights_load_and_return_structured_output(self) -> None:
        path = _coco_path()
        if path is None:
            self.skipTest("COCO yolov8n.pt is not on disk")
        import numpy as np

        detector = ObjectDetector(path, confidence=0.35, imgsz=320, max_det=10)
        self.assertTrue(is_general_object_model(detector.supported_classes()))
        self.assertIn("bottle", detector.supported_classes().values())
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detector.detect_objects(frame, frame_id=0, timestamp=0.0)
        json.dumps(result)
        self.assertEqual(result["model"], path.name)
        self.assertIsInstance(result["detections"], list)
        self.assertEqual(result["error"], None)


if __name__ == "__main__":
    unittest.main()
