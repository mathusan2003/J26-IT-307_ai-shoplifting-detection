import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline import load_tracker_settings, make_tracker, read_json, track_detection_sequence

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_detections.json"


class PipelineTests(unittest.TestCase):
    def test_sample_fixture_tracks_people_and_ignores_other_classes(self):
        results = track_detection_sequence(read_json(FIXTURE))
        self.assertEqual([frame["frame_id"] for frame in results], [0, 1, 2, 3])
        self.assertEqual({track["person_id"] for track in results[0]["tracks"]}, {1, 2})
        self.assertEqual(results[2]["tracks"][0]["person_id"], 1)
        for frame in results:
            self.assertEqual(len(frame["tracks"]), 2)
            for track in frame["tracks"]:
                self.assertEqual(set(track), {"person_id", "bbox", "center"})

    def test_read_json_accepts_utf8_bom(self):
        payload = [{"frame_id": 1, "detections": [{"class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]}]
        with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as handle:
            handle.write(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
            path = Path(handle.name)
        try:
            self.assertEqual(read_json(path)[0]["frame_id"], 1)
        finally:
            path.unlink()

    def test_config_loads_numeric_thresholds(self):
        settings = load_tracker_settings(ROOT / "configs" / "tracker_config.yaml")
        self.assertEqual(settings["track_thresh"], 0.2)
        self.assertEqual(settings["high_confidence_threshold"], 0.2)
        self.assertEqual(settings["match_thresh"], 0.2)
        self.assertEqual(settings["track_buffer"], 60.0)
        self.assertEqual(settings["reentry_buffer"], 90.0)
        self.assertEqual(settings["fps"], 30.0)
        tracker = make_tracker(settings)
        self.assertEqual(tracker.high_threshold, 0.2)
        self.assertEqual(tracker.match_iou_threshold, 0.2)
        self.assertEqual(tracker.max_missed_frames, 60)
        self.assertEqual(tracker.reentry_frames, 90)


if __name__ == "__main__":
    unittest.main()
