import unittest

from src.tracker import ByteTracker


class ByteTrackerTests(unittest.TestCase):
    def test_preserves_identity_across_motion(self):
        tracker = ByteTracker(match_iou_threshold=0.1)
        first = tracker.update([{"bbox": [10, 10, 30, 50], "confidence": 0.9}])
        second = tracker.update([{"bbox": [14, 10, 34, 50], "confidence": 0.9}])
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_low_confidence_second_pass_prevents_id_loss(self):
        tracker = ByteTracker(match_iou_threshold=0.1)
        track_id = tracker.update([{"bbox": [10, 10, 30, 50], "confidence": 0.9}])[0].track_id
        updated = tracker.update([{"bbox": [11, 10, 31, 50], "confidence": 0.2}])
        self.assertEqual(updated[0].track_id, track_id)

    def test_missing_detection_is_predicted_then_removed(self):
        tracker = ByteTracker(max_missed_frames=1)
        tracker.update([{"bbox": [10, 10, 30, 50], "confidence": 0.9}])
        self.assertEqual(tracker.update([]), [])
        tracker.update([])
        self.assertEqual(tracker.tracks, [])


if __name__ == "__main__":
    unittest.main()
