"""Checks for the SYNTHETIC multi-label trainer. Not a real-world accuracy claim."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from show_video_baseline import load_saved_video_metrics
from synthetic_behaviour_model import (
    METRICS_PATH,
    MODEL_PATH,
    SCHEMA_PATH,
    load_classifier,
)
from synthetic_tracking import (
    TRAINING_ROOT,
    assign_sequence_splits,
    generate_training_set,
)
from features import FEATURE_NAMES
from predict_synthetic_behaviour import predict_sequence
from video_baseline_smoke import SMOKE_METRICS_PATH
from video_dataset import FEATURES_PATH

OLD_MODEL = MODULE_ROOT / "models" / "behaviour_model.json"
VIDEO_METRICS = MODULE_ROOT / "evaluation" / "video_baseline_metrics.json"


class SyntheticBehaviourModelTests(unittest.TestCase):
    def test_existing_binary_model_file_was_not_replaced(self) -> None:
        payload = json.loads(OLD_MODEL.read_text(encoding="utf-8"))
        self.assertEqual(payload["model_type"], "l2_logistic_regression")
        self.assertNotIn("one_vs_rest", payload["model_type"])
        self.assertFalse(json.loads(MODEL_PATH.read_text(encoding="utf-8"))["replaces_existing_behaviour_model"])

    def test_training_split_is_by_sequence_not_frame(self) -> None:
        sequences = generate_training_set(10, 13, 1.5)
        splits = assign_sequence_splits(sequences, 10)
        self.assertEqual(len(sequences), 50)
        self.assertEqual(len(splits), 50)
        seen: dict[str, str] = {}
        for item in sequences:
            self.assertIn(item.sequence_id, splits)
            self.assertNotIn(item.sequence_id, seen)
            seen[item.sequence_id] = splits[item.sequence_id]
        locations: dict[str, set[str]] = {}
        for item in sequences:
            locations.setdefault(item.sequence_id, set()).add(splits[item.sequence_id])
        self.assertTrue(all(len(places) == 1 for places in locations.values()))
        counts = {"train": 0, "validation": 0, "test": 0}
        for name in splits.values():
            counts[name] += 1
        self.assertEqual(counts["train"], 30)
        self.assertEqual(counts["validation"], 5)
        self.assertEqual(counts["test"], 15)

    def test_same_seed_rebuilds_the_same_training_set(self) -> None:
        first = generate_training_set(10, 13, 1.5)
        second = generate_training_set(10, 13, 1.5)
        self.assertEqual([item.sequence_id for item in first], [item.sequence_id for item in second])
        self.assertEqual(first[0].tracking, second[0].tracking)
        self.assertNotEqual(generate_training_set(10, 99, 1.5)[0].tracking, first[0].tracking)

    def test_scaler_mean_matches_train_rows_only(self) -> None:
        payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        manifest = json.loads((TRAINING_ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(manifest["split_unit"], "sequence_id")
        train_ids = {row["sequence_id"] for row in manifest["sequences"] if row["split"] == "train"}
        test_ids = {row["sequence_id"] for row in manifest["sequences"] if row["split"] == "test"}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(payload["mean"], json.loads(METRICS_PATH.read_text(encoding="utf-8"))["train_feature_mean"])

    def test_saved_report_is_labelled_synthetic(self) -> None:
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertTrue(report["synthetic"])
        self.assertFalse(report["real_world_accuracy"])
        self.assertEqual(report["label_source"], "generator_motion_plan")
        self.assertFalse(report["mixed_with_model1_video_evaluation"])
        self.assertFalse(report["hyperparameter_search_on_test"])
        self.assertEqual(schema["annotation_format"], "multi_label")
        self.assertEqual(schema["feature_names"], list(FEATURE_NAMES))
        model = load_classifier()
        self.assertEqual(len(model.predict_proba([0.0] * 15)), 5)

    def test_concealment_is_withheld_without_object_detections(self) -> None:
        tracking = MODULE_ROOT / "samples" / "synthetic_scenarios" / "tracking" / "possible_concealment.json"
        detections = MODULE_ROOT / "samples" / "synthetic_scenarios" / "detections" / "possible_concealment.json"
        blocked = predict_sequence(tracking, None)
        self.assertIn("possible_concealment", blocked["synthetic_ml"]["people"][0]["withheld"])
        self.assertNotIn("possible_concealment", blocked["synthetic_ml"]["people"][0]["predicted"])
        allowed = predict_sequence(tracking, detections)
        self.assertTrue(allowed["synthetic_ml"]["object_evidence"])
        self.assertEqual(set(allowed["member4_contract"][0]), {"person_id", "behaviours", "behaviour_score", "assessment"})

    def test_video_smoke_file_is_separate_from_real_video_metrics(self) -> None:
        if SMOKE_METRICS_PATH.is_file():
            smoke = json.loads(SMOKE_METRICS_PATH.read_text(encoding="utf-8"))
            self.assertTrue(smoke["synthetic"])
            self.assertFalse(smoke["writes_video_baseline_metrics"])
            self.assertFalse(smoke["mixed_with_real_video_evaluation"])

    def test_model1_saved_metrics_were_not_rewritten_as_synthetic(self) -> None:
        metrics = load_saved_video_metrics(VIDEO_METRICS)
        self.assertEqual(metrics["task"], "video_level_normal_vs_shoplifting")
        self.assertFalse(metrics["behaviour_labels_used"])
        self.assertAlmostEqual(metrics["test"]["f1"], 0.5714285714285714, places=6)
        self.assertEqual(metrics["model"]["C"], 10.0)
        self.assertNotIn("synthetic_tracking_behaviour_multilabel", json.dumps(metrics))
        features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(features["feature_names"]), 39)


if __name__ == "__main__":
    unittest.main()
