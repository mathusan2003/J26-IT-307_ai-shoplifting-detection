"""Member 1 dashboard bridge keeps Tracking/Behaviour contracts additive."""
from __future__ import annotations

import unittest

from src.object_bridge import empty_object_state, member3_object_rows


class ObjectBridgeTests(unittest.TestCase):
    def test_empty_state_has_no_invented_boxes(self) -> None:
        payload = empty_object_state(frame_id=3, timestamp=1.5, error=None, model=None)
        self.assertEqual(payload["detections"], [])
        self.assertTrue(payload["available"])
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["frame_id"], 3)
        self.assertIn("coco", payload["note"].lower())

    def test_error_state_is_unavailable(self) -> None:
        payload = empty_object_state(error="weights missing")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["error"], "weights missing")
        self.assertEqual(payload["detections"], [])


class Member3ObjectRowTests(unittest.TestCase):
    def test_error_state_is_none_so_misses_are_not_invented(self) -> None:
        self.assertIsNone(member3_object_rows(None))
        self.assertIsNone(member3_object_rows(empty_object_state(error="no weights")))
        empty = member3_object_rows({"error": None, "detections": []})
        self.assertEqual(empty, [])
        rows = member3_object_rows(
            {
                "error": None,
                "detections": [
                    {"class_name": "bottle", "confidence": 0.8, "bbox": {"x1": 1, "y1": 2, "x2": 10, "y2": 20}}
                ],
            }
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["class_name"], "bottle")
