import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline import read_json, track_detection_sequence
from src.visualize import annotate_frames, colour_for_id, draw_tracks, draw_zones, require_opencv
from src.zone_analytics import ZoneAnalytics

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_detections.json"


class VisualizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_opencv()
        import cv2
        import numpy as np
        cls.cv2 = cv2
        cls.np = np

    def test_colour_is_stable_for_the_same_id(self):
        self.assertEqual(colour_for_id(3), colour_for_id(3))
        self.assertNotEqual(colour_for_id(1), colour_for_id(2))

    def test_draw_zones_marks_entry_and_exit(self):
        frame = self.np.zeros((720, 1280, 3), dtype=self.np.uint8)
        painted = draw_zones(frame, ZoneAnalytics(ROOT / "configs" / "zones.json"))
        self.assertGreater(int(painted.sum()), 0)

    def test_draw_zones_skips_shelf_labels(self):
        from src.visualize import zone_caption

        self.assertIsNone(zone_caption(None, "Shelf B"))
        self.assertIsNone(zone_caption(None, "Shelf A"))

    def test_draw_tracks_paints_box_and_id(self):
        frame = self.np.zeros((80, 80, 3), dtype=self.np.uint8)
        draw_tracks(frame, [{"person_id": 4, "bbox": [10, 10, 40, 50], "center": [25, 30]}])
        self.assertGreater(int(frame.sum()), 0)

    def test_draw_tracks_skips_lost_kalman_boxes(self):
        frame = self.np.zeros((80, 80, 3), dtype=self.np.uint8)
        draw_tracks(frame, [{
            "person_id": 9,
            "bbox": [10, 10, 40, 50],
            "center": [25, 30],
            "status": "temporarily_lost",
        }])
        self.assertEqual(int(frame.sum()), 0)

    def test_annotate_video_from_fixture(self):
        sequence = track_detection_sequence(read_json(FIXTURE))
        tracks_by_frame = {int(frame["frame_id"]): frame["tracks"] for frame in sequence}
        frames = []
        for frame_id in range(4):
            image = self.np.zeros((360, 640, 3), dtype=self.np.uint8)
            image[:] = (30, 30, 30)
            frames.append((frame_id, image))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "annotated.mp4"
            written = annotate_frames(
                frames,
                tracks_by_frame,
                output,
                fps=10,
                analytics=ZoneAnalytics(ROOT / "configs" / "zones.json"),
            )
            self.assertEqual(written, 4)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            capture = self.cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                ok, image = capture.read()
                self.assertTrue(ok)
                self.assertEqual(image.shape[1], 640)
            finally:
                capture.release()

    def test_demo_cli_writes_video_and_contract_json(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "input.mp4"
            writer = self.cv2.VideoWriter(str(video), self.cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(self.np.full((360, 640, 3), 40, dtype=self.np.uint8))
            writer.release()
            output_video = Path(folder) / "annotated.mp4"
            output_json = Path(folder) / "tracking_log.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "demo_track.py"),
                    "--detections",
                    str(FIXTURE),
                    "--video",
                    str(video),
                    "--output-video",
                    str(output_video),
                    "--output-json",
                    str(output_json),
                    "--no-zones",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["tracks"][0].keys(), {"person_id", "bbox", "center"})
            self.assertTrue(output_video.exists())
            self.assertGreater(output_video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
