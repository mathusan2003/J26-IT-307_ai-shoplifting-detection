import unittest
from pathlib import Path

from src.zone_analytics import ZoneAnalytics

ROOT = Path(__file__).resolve().parents[1]


class ZoneAnalyticsTests(unittest.TestCase):
    def test_bottom_center_selects_zone_and_records_transition(self):
        analytics = ZoneAnalytics(ROOT / "configs/zones.json")
        first = analytics.update(1, (20, 100, 100, 200), 0.0)
        self.assertEqual(first.current_zone, "Entrance")
        second = analytics.update(1, (260, 100, 340, 200), 4.0)
        self.assertEqual(second.current_zone, "Aisle 1")
        self.assertEqual(second.zone_history, ["Entrance", "Aisle 1"])
        self.assertEqual(second.transitions[0]["from"], "Entrance")

    def test_dwell_time(self):
        analytics = ZoneAnalytics(ROOT / "configs/zones.json")
        analytics.update(9, (20, 100, 100, 200), 10.0)
        self.assertEqual(analytics.dwell_time(9, 18.5), 8.5)


if __name__ == "__main__":
    unittest.main()
