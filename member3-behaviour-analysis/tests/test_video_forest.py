"""Random Forest comparison on the frozen Normal versus Shoplifting split."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from train_video_baseline import random_forest_from_features
from video_dataset import FEATURES_PATH, SPLIT_PATH

METRICS_PATH = MODULE_ROOT / "evaluation" / "video_baseline_metrics.json"


class RandomForestSplitTests(unittest.TestCase):
    def test_saved_duplicate_hashes_stay_in_one_split(self) -> None:
        split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        locations: dict[str, str] = {}
        for split_name, rows in split["splits"].items():
            for row in rows:
                previous = locations.get(row["sha256"])
                if previous is None:
                    locations[row["sha256"]] = split_name
                else:
                    self.assertEqual(previous, split_name)

    def test_selection_ignores_test_labels(self) -> None:
        features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        original = random_forest_from_features(features)
        corrupted = json.loads(json.dumps(features))
        for row in corrupted["rows"]:
            if row["split"] == "test":
                row["label"] = "normal" if row["label"] == "shoplifting" else "shoplifting"
        flipped = random_forest_from_features(corrupted)
        self.assertEqual(original["model"], flipped["model"])
        self.assertEqual(original["validation"]["f1"], flipped["validation"]["f1"])
        self.assertNotEqual(original["test"]["confusion_matrix"], flipped["test"]["confusion_matrix"])
        self.assertFalse(original["hyperparameter_selection"]["test_rows_used"])

    def test_report_does_not_use_behaviour_labels(self) -> None:
        features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        report = random_forest_from_features(features)
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.assertFalse(report["behaviour_labels_used"])
        self.assertFalse(report["detects_behaviour_categories"])
        self.assertFalse(committed["behaviour_labels_used"])
        self.assertFalse(committed["random_forest"]["behaviour_labels_used"])
        self.assertEqual(committed["random_forest"]["test"]["confusion_matrix"], report["test"]["confusion_matrix"])
        self.assertIn("false_alarm_rate", committed["test"])
        self.assertIn("false_alarm_rate", committed["random_forest"]["test"])


if __name__ == "__main__":
    unittest.main()
