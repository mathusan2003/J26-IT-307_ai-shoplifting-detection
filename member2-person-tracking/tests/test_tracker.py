import unittest

from src.tracker import ByteTracker
from src.utils import ciou


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
        tracker = ByteTracker(max_missed_frames=1, reentry_frames=0)
        tracker.update([{"bbox": [10, 10, 30, 50], "confidence": 0.9}])
        self.assertEqual(tracker.update([]), [])
        tracker.update([])
        self.assertEqual(tracker.tracks, [])

    def test_crossing_people_keep_original_ids(self):
        tracker = ByteTracker(match_iou_threshold=0.15)
        tracker.set_frame_size(240, 80)
        left = [10, 40, 70, 100, 140, 170]
        right = [170, 140, 100, 70, 40, 10]
        start = tracker.update([
            {"bbox": [left[0], 10, left[0] + 30, 60], "confidence": 0.9},
            {"bbox": [right[0], 10, right[0] + 30, 60], "confidence": 0.9},
        ])
        left_id = sorted(start, key=lambda track: track.bbox[0])[0].track_id
        right_id = sorted(start, key=lambda track: track.bbox[0])[1].track_id
        for lx, rx in zip(left[1:], right[1:]):
            swapped = tracker.update([
                {"bbox": [lx, 10, lx + 30, 60], "confidence": 0.9},
                {"bbox": [rx, 10, rx + 30, 60], "confidence": 0.9},
            ])
        by_x = sorted(swapped, key=lambda track: track.bbox[0])
        self.assertEqual(by_x[0].track_id, right_id)
        self.assertEqual(by_x[1].track_id, left_id)

    def test_occlusion_gap_reuses_id_via_kalman(self):
        tracker = ByteTracker(max_missed_frames=8, match_iou_threshold=0.15)
        track_id = tracker.update([{"bbox": [20, 20, 50, 80], "confidence": 0.92}])[0].track_id
        tracker.update([])
        tracker.update([])
        recovered = tracker.update([{"bbox": [26, 20, 56, 80], "confidence": 0.4}])
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].track_id, track_id)

    def test_exit_and_reenter_reuses_parked_id(self):
        tracker = ByteTracker(max_missed_frames=2, reentry_frames=20)
        tracker.set_frame_size(200, 80)
        track_id = tracker.update([{"bbox": [5, 10, 35, 70], "confidence": 0.9}])[0].track_id
        tracker.update([{"bbox": [5, 10, 35, 70], "confidence": 0.9}])
        tracker.update([{"bbox": [5, 10, 35, 70], "confidence": 0.9}])
        for _ in range(4):
            tracker.update([])
        self.assertEqual(tracker.tracks, [])
        returned = tracker.update([{"bbox": [8, 12, 38, 72], "confidence": 0.88}])
        self.assertEqual(returned[0].track_id, track_id)

    def test_crowded_frame_assigns_unique_ids(self):
        tracker = ByteTracker()
        boxes = [{"bbox": [x, 10, x + 20, 70], "confidence": 0.8} for x in range(0, 180, 30)]
        tracks = tracker.update(boxes)
        self.assertEqual(len(tracks), len(boxes))
        self.assertEqual(len({track.track_id for track in tracks}), len(boxes))

    def test_identical_boxes_have_unit_ciou(self):
        box = (10.0, 10.0, 40.0, 80.0)
        self.assertAlmostEqual(ciou(box, box), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
