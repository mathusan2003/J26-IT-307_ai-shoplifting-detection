import unittest
from pathlib import Path

from src.tracker import ByteTracker
from src.tracking_outcomes import TrackingOutcomes, format_timestamp
from src.zone_analytics import ZoneAnalytics

ROOT = Path(__file__).resolve().parents[1]


class TrackingOutcomesTests(unittest.TestCase):
    def test_timestamp_format(self):
        self.assertEqual(format_timestamp(75.5), "00:01:15.500")

    def test_multiple_people_keep_persistent_ids(self):
        outcomes = TrackingOutcomes(fps=10, analytics=ZoneAnalytics(ROOT / "configs" / "zones.json"))
        outcomes.observe(0, [
            {"person_id": 1, "bbox": [20, 100, 80, 200], "center": [50, 150]},
            {"person_id": 2, "bbox": [960, 80, 1040, 240], "center": [1000, 160]},
        ])
        outcomes.observe(1, [
            {"person_id": 1, "bbox": [28, 100, 88, 200], "center": [58, 150]},
            {"person_id": 2, "bbox": [970, 80, 1050, 240], "center": [1010, 160]},
        ])
        summary = outcomes.summary()
        self.assertEqual(summary["unique_person_ids"], [1, 2])
        self.assertEqual(summary["occupancy"]["max_active"], 2)
        self.assertEqual(len(summary["people"][0]["trajectory"]), 2)

    def test_lost_then_terminated_status(self):
        outcomes = TrackingOutcomes(fps=1, max_missed_frames=2)
        outcomes.observe(0, [{"person_id": 7, "bbox": [0, 0, 10, 10], "center": [5, 5]}])
        outcomes.observe(1, [])
        outcomes.observe(2, [])
        outcomes.observe(3, [])
        summary = outcomes.summary()
        person = summary["people"][0]
        statuses = [point["status"] for point in person["trajectory"]]
        self.assertIn("temporarily_lost", statuses)
        self.assertEqual(person["status"], "terminated")

    def test_shelf_visit_and_entry_exit(self):
        analytics = ZoneAnalytics(ROOT / "configs" / "zones.json")
        outcomes = TrackingOutcomes(fps=1, analytics=analytics)
        outcomes.observe(0, [{"person_id": 1, "bbox": [20, 100, 80, 200], "center": [50, 150]}])
        outcomes.observe(5, [{"person_id": 1, "bbox": [660, 400, 720, 560], "center": [690, 480]}])
        outcomes.observe(8, [{"person_id": 1, "bbox": [980, 80, 1100, 300], "center": [1040, 190]}])
        summary = outcomes.summary()
        person = summary["people"][0]
        self.assertIsNotNone(person["ingress"])
        self.assertIsNotNone(person["egress"])
        self.assertTrue(any(visit["is_shelf"] for visit in person["shelf_visits"]))
        self.assertGreaterEqual(len(person["zone_transitions"]), 2)

    def test_tracker_reports_temporarily_lost(self):
        tracker = ByteTracker(max_missed_frames=3, match_iou_threshold=0.1)
        tracker.update([{"bbox": [10, 10, 30, 50], "confidence": 0.9}])
        tracker.update([])
        records = tracker.status_records()
        self.assertEqual(records[0]["status"], "temporarily_lost")
        self.assertEqual(records[0]["person_id"], 1)


if __name__ == "__main__":
    unittest.main()
