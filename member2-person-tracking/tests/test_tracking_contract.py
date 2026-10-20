import unittest

from src.pipeline import make_tracker, track_detection_sequence
from src.utils import serialise_bbox


def _assert_contract_frame(testcase: unittest.TestCase, frame: dict) -> None:
    testcase.assertIsInstance(frame["frame_id"], int)
    testcase.assertGreaterEqual(frame["frame_id"], 0)
    testcase.assertIsInstance(frame["tracks"], list)
    for track in frame["tracks"]:
        testcase.assertEqual(set(track), {"person_id", "bbox", "center"})
        testcase.assertIsInstance(track["person_id"], int)
        testcase.assertEqual(track["bbox"], serialise_bbox(track["bbox"]))
        testcase.assertEqual(len(track["bbox"]), 4)
        testcase.assertEqual(len(track["center"]), 2)
        testcase.assertTrue(all(isinstance(value, int) for value in track["bbox"]))
        testcase.assertTrue(all(isinstance(value, int) for value in track["center"]))
        x1, y1, x2, y2 = track["bbox"]
        testcase.assertLessEqual(x1, x2)
        testcase.assertLessEqual(y1, y2)


class TrackingContractTests(unittest.TestCase):
    def test_sample_sequence_matches_member3_schema(self):
        frames = [
            {"frame_id": 0, "detections": [{"class": "person", "confidence": 0.93, "bbox": [20.4, 120.2, 110.6, 300.4]}]},
            {"frame_id": 1, "detections": [{"class": "person", "confidence": 0.22, "bbox": [24.2, 121.1, 114.8, 301.7]}]},
            {"frame_id": 2, "detections": [{"class": "cart", "confidence": 0.8, "bbox": [500, 200, 580, 280]}]},
        ]
        results = track_detection_sequence(frames, make_tracker())
        self.assertEqual([frame["frame_id"] for frame in results], [0, 1, 2])
        self.assertEqual(len(results[0]["tracks"]), 1)
        self.assertEqual(results[0]["tracks"][0]["person_id"], results[1]["tracks"][0]["person_id"])
        self.assertEqual(results[2]["tracks"], [])
        for frame in results:
            _assert_contract_frame(self, frame)


if __name__ == "__main__":
    unittest.main()
