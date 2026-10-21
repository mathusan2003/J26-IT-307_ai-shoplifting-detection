"""Mocked Member 1 detector tests. These do not open a webcam or download weights."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from capture import CaptureError, is_webcam_source, open_source
from object_detector import (
    IncompatibleWeightsError,
    InferenceError,
    ObjectDetector,
    WeightsNotFoundError,
    clip_bbox,
    empty_frame_result,
    is_general_object_model,
    pipeline_detections,
    resolve_device,
    resolve_object_weights,
)
from visualize import draw_detections


COCO_NAMES = {0: "person", 39: "bottle", 24: "backpack", 26: "handbag", 41: "cup"}


class _Tensor:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, _idx):
        return self._value

    def tolist(self):
        return list(self._value)


class _Box:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [_Tensor(xyxy)]


class DummyModel:
    def __init__(self, names=None, boxes=None, error=None):
        self.names = names if names is not None else COCO_NAMES
        self.boxes = boxes or []
        self.error = error
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append({"frame_shape": getattr(frame, "shape", None), "kwargs": kwargs})
        if self.error:
            raise RuntimeError(self.error)
        return [type("Result", (), {"boxes": self.boxes})()]


def _detector(boxes=None, **kwargs):
    model = DummyModel(boxes=boxes)
    return ObjectDetector(weights="mock-coco.pt", model=model, **kwargs), model


class WeightsAndModelTests(unittest.TestCase):
    def test_missing_weights_path_returns_clear_error(self) -> None:
        missing = MODULE_ROOT / "models" / "does-not-exist.pt"
        with self.assertRaises(WeightsNotFoundError) as caught:
            resolve_object_weights(missing)
        self.assertIn("not found", str(caught.exception).lower())
        self.assertIn("not substituted", str(caught.exception).lower())

    def test_person_only_model_is_rejected(self) -> None:
        with self.assertRaises(IncompatibleWeightsError) as caught:
            ObjectDetector(model=DummyModel(names={0: "person"}))
        self.assertIn("not a general object-detection model", str(caught.exception))

    def test_item_only_member2_weights_are_rejected(self) -> None:
        with self.assertRaises(IncompatibleWeightsError):
            ObjectDetector(model=DummyModel(names={0: "item"}))

    def test_coco_names_are_accepted(self) -> None:
        self.assertTrue(is_general_object_model(COCO_NAMES))
        detector, _ = _detector()
        self.assertIn(39, detector.supported_classes())
        self.assertEqual(detector.supported_classes()[39], "bottle")


class DetectionContractTests(unittest.TestCase):
    def test_empty_results_keep_required_structure(self) -> None:
        detector, _ = _detector(boxes=[])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(frame, frame_id=4, timestamp=1.25)
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["frame_id"], 4)
        self.assertEqual(result["timestamp"], 1.25)
        self.assertIn("note", result)
        self.assertIsNone(result["error"])
        json.dumps(result)

    def test_invalid_frame_returns_empty_structure(self) -> None:
        detector, _ = _detector()
        result = detector.detect_objects(None, frame_id=0, timestamp=0.0)
        self.assertEqual(result["detections"], [])
        self.assertIn("invalid frame", result["error"].lower())

    def test_detection_fields_and_json(self) -> None:
        detector, _ = _detector(boxes=[_Box(39, 0.87, [120.2, 80.4, 190.9, 260.1])])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(frame, frame_id=12, timestamp=1.2)
        self.assertEqual(len(result["detections"]), 1)
        item = result["detections"][0]
        for key in ("detection_id", "class_id", "class_name", "confidence", "bbox"):
            self.assertIn(key, item)
        self.assertEqual(item["class_id"], 39)
        self.assertEqual(item["class_name"], "bottle")
        self.assertEqual(item["confidence"], 0.87)
        self.assertEqual(item["bbox"], {"x1": 120.2, "y1": 80.4, "x2": 190.9, "y2": 260.1})
        json.dumps(result)
        contract = pipeline_detections(result)
        self.assertEqual(contract[0]["class"], "bottle")
        self.assertEqual(contract[0]["bbox"], [120.2, 80.4, 190.9, 260.1])

    def test_bbox_is_clipped_to_the_frame(self) -> None:
        box = clip_bbox([-20, -5, 800, 900], 640, 480)
        self.assertEqual(box, {"x1": 0.0, "y1": 0.0, "x2": 640.0, "y2": 480.0})
        self.assertIsNone(clip_bbox([10, 10, 10.4, 10.4], 640, 480))

    def test_class_filter_keeps_only_requested_names(self) -> None:
        boxes = [_Box(39, 0.9, [10, 10, 40, 80]), _Box(0, 0.9, [50, 10, 90, 120])]
        detector, model = _detector(boxes=boxes, classes=["bottle"])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(frame)
        self.assertEqual([item["class_name"] for item in result["detections"]], ["bottle"])
        self.assertEqual(model.calls[0]["kwargs"]["classes"], [39])

    def test_unknown_class_filter_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _detector(classes=["shampoo"])
        self.assertIn("Unknown class filter", str(caught.exception))

    def test_confidence_threshold_drops_weak_boxes(self) -> None:
        detector, model = _detector(boxes=[_Box(39, 0.20, [10, 10, 40, 80])], confidence=0.50)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_objects(frame)
        self.assertEqual(result["detections"], [])
        self.assertEqual(model.calls[0]["kwargs"]["conf"], 0.50)

    def test_inference_failure_is_wrapped(self) -> None:
        detector = ObjectDetector(model=DummyModel(error="cuda oom"))
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(InferenceError) as caught:
            detector.detect_objects(frame, frame_id=3)
        self.assertIn("frame 3", str(caught.exception))

    def test_empty_frame_helper_is_serialisable(self) -> None:
        payload = empty_frame_result(frame_id=0, timestamp=0.0, model="yolov8n.pt")
        encoded = json.dumps(payload)
        self.assertIn("detections", encoded)
        self.assertIn("[]", encoded)


class CaptureAndOverlayTests(unittest.TestCase):
    def test_invalid_video_path_is_handled(self) -> None:
        with self.assertRaises(CaptureError) as caught:
            open_source(MODULE_ROOT / "outputs" / "missing.mp4")
        self.assertIn("not found", str(caught.exception).lower())
        self.assertTrue(is_webcam_source("webcam"))
        self.assertFalse(is_webcam_source("clip.mp4"))

    def test_unreadable_file_is_not_treated_as_video(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.mp4"
            path.write_bytes(b"not a video")
            try:
                capture, _name = open_source(path)
            except CaptureError as exc:
                self.assertIn("could not read", str(exc).lower())
                return
            ok, _frame = capture.read()
            capture.release()
            self.assertFalse(ok)

    def test_overlay_draws_real_boxes_only(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        result = {
            "frame_id": 1,
            "model": "yolov8n.pt",
            "detections": [
                {"class_id": 39, "class_name": "bottle", "confidence": 0.91, "bbox": {"x1": 10, "y1": 10, "x2": 40, "y2": 80}}
            ],
        }
        shown = draw_detections(frame, result, source="webcam", infer_fps=8.2)
        self.assertEqual(shown.shape, frame.shape)
        empty = draw_detections(frame, {"frame_id": 2, "detections": []}, source="webcam")
        self.assertEqual(empty.shape, frame.shape)

    def test_cpu_is_selected_when_cuda_is_unavailable(self) -> None:
        self.assertEqual(resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
