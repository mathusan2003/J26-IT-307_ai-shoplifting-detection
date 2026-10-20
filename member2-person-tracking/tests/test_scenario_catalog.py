import unittest
from pathlib import Path

from src.scenario_catalog import catalogue_videos, classify_video


class ScenarioCatalogTests(unittest.TestCase):
    def test_groups_panel_scenarios(self):
        rows = catalogue_videos([
            Path("longer_store_input.mp4"),
            Path("10901926-hd_1920_1080_30fps.mp4"),
            Path("12589709_3840_2160_60fps.mp4"),
        ])
        groups = {row["file"]: row["group"] for row in rows}
        self.assertEqual(groups["longer_store_input.mp4"], "occlusion")
        self.assertIn(groups["12589709_3840_2160_60fps.mp4"], {"crowded", "cross_cam"})
        self.assertTrue(any("occlusion" in row["description"].lower() or "Occlusion" in row["description"] for row in rows))

    def test_cross_camera_gets_camera_numbers(self):
        group, camera, description = classify_video(Path("cam2_aisle.mp4"))
        self.assertEqual(group, "cross_cam")
        self.assertIn("handover", description.lower())


if __name__ == "__main__":
    unittest.main()
