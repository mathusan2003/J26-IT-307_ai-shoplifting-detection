import unittest

from src.evaluation import evaluate_tracking, measure_fps


def frame(frame_id, tracks):
    return {"frame_id": frame_id, "tracks": tracks}


def track(person_id, bbox):
    return {"person_id": person_id, "bbox": bbox, "center": [0, 0]}


class EvaluationTests(unittest.TestCase):
    def test_perfect_tracking_scores_one(self):
        truth = [frame(1, [track(1, [0, 0, 20, 20])]), frame(2, [track(1, [2, 0, 22, 20])])]
        metrics = evaluate_tracking(truth, truth)
        self.assertEqual(metrics.id_switches, 0)
        self.assertEqual(metrics.idf1, 1.0)
        self.assertEqual(metrics.mota, 1.0)

    def test_id_switch_and_false_positive_are_counted(self):
        truth = [frame(1, [track(1, [0, 0, 20, 20])]), frame(2, [track(1, [2, 0, 22, 20])])]
        prediction = [frame(1, [track(9, [0, 0, 20, 20])]), frame(2, [track(10, [2, 0, 22, 20]), track(11, [50, 0, 70, 20])])]
        metrics = evaluate_tracking(prediction, truth)
        self.assertEqual(metrics.id_switches, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.mota, 0.0)

    def test_occlusion_recovery_uses_original_tracker_id(self):
        truth = [frame(index, [track(1, [0, 0, 20, 20])]) for index in range(1, 5)]
        prediction = [frame(1, [track(7, [0, 0, 20, 20])]), frame(2, []), frame(3, []), frame(4, [track(7, [0, 0, 20, 20])])]
        metrics = evaluate_tracking(prediction, truth, occlusion_min_frames=2)
        self.assertEqual((metrics.occlusion_events, metrics.recovered_occlusions), (1, 1))
        self.assertEqual(metrics.occlusion_recovery_rate, 1.0)

    def test_fps_returns_positive_value(self):
        self.assertGreater(measure_fps(lambda value: value, range(30), warmup_frames=2), 0)


if __name__ == "__main__":
    unittest.main()
