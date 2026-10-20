import tempfile
import unittest
from pathlib import Path

from src.detector import ACTIVE_WEIGHTS, MODEL_PRIORITY, PERSON_CLASS_ID, PersonDetector, is_person_box, resolve_weights


class WeightPriorityTests(unittest.TestCase):
    def test_person_box_keeps_seated_and_short_people(self):
        self.assertTrue(is_person_box([100, 80, 180, 320], 1280, 720))
        self.assertTrue(is_person_box([200, 400, 340, 530], 1280, 720))
        self.assertTrue(is_person_box([500, 620, 545, 690], 1920, 1080))
        self.assertFalse(is_person_box([10, 10, 400, 80], 1280, 720))
        self.assertFalse(is_person_box([10, 10, 20, 22], 1280, 720))

    def test_priority_order_prefers_person_track_weights(self):
        self.assertEqual(ACTIVE_WEIGHTS, "models/yolov8n-person-track.pt")
        self.assertEqual(
            MODEL_PRIORITY,
            [
                "models/yolov8n-person-track.pt",
                "models/yolov8n-cam-multiperson.pt",
                "models/yolov8n.pt",
            ],
        )
        self.assertEqual(PERSON_CLASS_ID, 0)

    def test_missing_file_falls_through_priority_and_explicit_file_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            (models / "yolov8n.pt").write_bytes(b"base")
            (models / "yolov8n-cam-multiperson.pt").write_bytes(b"cam")
            resolved = resolve_weights("models/yolov8n-person-track.pt", roots=[root])
            self.assertEqual(resolved.name, "yolov8n-cam-multiperson.pt")
            (models / "yolov8n-person-track.pt").write_bytes(b"tuned")
            self.assertEqual(resolve_weights(None, roots=[root]).name, "yolov8n-person-track.pt")
            explicit = models / "yolov8n.pt"
            self.assertEqual(resolve_weights(explicit, roots=[root]), explicit)

    def test_detect_requests_only_the_person_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "tiny.pt"
            weights.write_bytes(b"weights")
            calls = []

            class _Value:
                def __init__(self, value):
                    self._value = value

                def tolist(self):
                    return self._value

            class _Box:
                conf = [0.88]
                xyxy = [_Value([120.0, 80.0, 220.0, 380.0])]

            class _Result:
                boxes = [_Box()]

            class _Model:
                def __call__(self, frame, **kwargs):
                    calls.append({"frame": frame, **kwargs})
                    return [_Result()]

            class _YOLO:
                def __init__(self, path):
                    self.path = path

                def __call__(self, *args, **kwargs):
                    return _Model()(*args, **kwargs)

            import ultralytics

            original = ultralytics.YOLO
            ultralytics.YOLO = _YOLO
            try:
                detector = PersonDetector(weights, confidence=0.1)
                detections = detector.detect("frame")
            finally:
                ultralytics.YOLO = original
            self.assertEqual(calls[0]["classes"], [0])
            self.assertEqual(calls[0]["conf"], 0.1)
            self.assertEqual(calls[0]["imgsz"], 1080)
            self.assertEqual(calls[0]["iou"], 0.60)
            self.assertEqual(detections[0]["class"], "person")
            self.assertEqual(detector.weights, weights)

    def test_detector_defaults_match_crowded_inference_settings(self):
        self.assertEqual(PersonDetector.__init__.__defaults__[1], 0.20)
        self.assertEqual(PersonDetector.__init__.__defaults__[2], 1080)
        self.assertEqual(PersonDetector.__init__.__defaults__[4], 0.60)


if __name__ == "__main__":
    unittest.main()
