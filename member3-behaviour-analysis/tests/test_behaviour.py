"""Tests for behaviour indicators, the Member 4 contract, and the trained model."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT / "src"))

from behaviour import analyse_tracking_frames, validate_behaviour_result
from config import load_settings, load_zones
from dataset import (
    build_labelled_sequences,
    load_dataset,
    member2_contract_loiter_frames,
)
from detectors import INDICATOR_TYPES, Indicator
from evaluation import rule_predictions, train_and_evaluate
from geometry import zone_for_point
from model import load_model
from movement_history import build_movement_histories
from tracking_input import load_tracking_sequence

SAMPLE_TRACKING = MODULE_ROOT / "samples" / "sample_tracking_frames.json"
CONTRACT_LOITER = MODULE_ROOT / "samples" / "member2_contract_loiter.json"
METRICS_PATH = MODULE_ROOT / "evaluation" / "metrics.json"


class ZoneTests(unittest.TestCase):
    def test_zone_polygons_match_member2_layout(self) -> None:
        ours = json.loads((MODULE_ROOT / "configs" / "zones.json").read_text(encoding="utf-8"))
        theirs = json.loads(
            (REPO_ROOT / "member2-person-tracking" / "configs" / "zones.json").read_text(encoding="utf-8")
        )
        self.assertEqual(ours["zones"], theirs["zones"])

    def test_foot_point_lands_in_the_expected_shelf(self) -> None:
        zones = load_zones()
        self.assertEqual(zone_for_point((100, 100), zones), "Entrance")
        self.assertEqual(zone_for_point((400, 100), zones), "Aisle 1")
        self.assertEqual(zone_for_point((750, 100), zones), "Shelf A")
        self.assertEqual(zone_for_point((750, 500), zones), "Shelf B")
        self.assertEqual(zone_for_point((1000, 100), zones), "Checkout")
        self.assertEqual(zone_for_point((750, 300), zones), "Shelf A")


class DetectorTests(unittest.TestCase):
    def test_every_labelled_sequence_matches_its_rule_labels(self) -> None:
        settings = load_settings()
        zones = load_zones()
        for sequence in build_labelled_sequences():
            predicted = rule_predictions(sequence, settings, zones)
            self.assertEqual(predicted, sequence["labels"], sequence["id"])

    def test_member2_contract_track_without_zone_or_timestamp_is_loitering(self) -> None:
        results = analyse_tracking_frames(member2_contract_loiter_frames())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["person_id"], 12)
        self.assertEqual(results[0]["behaviours"], [{"type": "loitering", "score": 20}])
        self.assertEqual(results[0]["behaviour_score"], 20)
        validate_behaviour_result(results[0])

    def test_phase1_sample_has_no_suspicious_indicator(self) -> None:
        frames = load_tracking_sequence(SAMPLE_TRACKING)
        histories = build_movement_histories(frames)
        results = analyse_tracking_frames(frames)
        self.assertEqual([item["person_id"] for item in histories], [7, 12])
        self.assertEqual([item["person_id"] for item in results], [7, 12])
        for result in results:
            self.assertEqual(result["behaviours"], [])
            self.assertEqual(result["behaviour_score"], 0)
            self.assertEqual(result["assessment"]["suspicious_level"], "none")
            self.assertEqual(result["assessment"]["combined_score"], 0)

    def test_people_are_scored_separately(self) -> None:
        loiter = member2_contract_loiter_frames(person_id=12)
        for index, frame in enumerate(loiter):
            foot_x = 250 + index * 20
            frame["tracks"].append(
                {
                    "person_id": 7,
                    "bbox": [foot_x - 30, 20, foot_x + 30, 200],
                    "center": [foot_x, 110],
                }
            )
        results = {item["person_id"]: item for item in analyse_tracking_frames(loiter)}
        self.assertEqual(results[12]["behaviours"][0]["type"], "loitering")
        self.assertEqual(results[7]["behaviours"], [])
        self.assertEqual(results[7]["behaviour_score"], 0)

    def test_behaviour_score_is_capped_at_100(self) -> None:
        import behaviour

        original = behaviour.detect_indicators

        def forced(_profile, _settings):
            return [Indicator(name, 40) for name in INDICATOR_TYPES]

        behaviour.detect_indicators = forced
        try:
            result = analyse_tracking_frames(
                [
                    {
                        "frame_id": 0,
                        "timestamp": "00:00:00",
                        "tracks": [{"person_id": 1, "center": [10, 10], "zone": "Aisle 1"}],
                    }
                ]
            )[0]
        finally:
            behaviour.detect_indicators = original
        self.assertEqual(result["behaviour_score"], 100)
        self.assertEqual(len(result["behaviours"]), len(INDICATOR_TYPES))
        validate_behaviour_result(result)

    def test_repeated_shelf_visit_scores_25_for_one_return(self) -> None:
        frames = []
        plan = (
            (0, "Shelf_A", [450, 320]),
            (1, "Aisle_1", [300, 320]),
            (2, "Shelf_A", [450, 320]),
        )
        for second, zone, position in plan:
            frames.append(
                {
                    "frame_id": second,
                    "timestamp": f"00:00:0{second}",
                    "tracks": [{"person_id": 4, "position": position, "zone": zone}],
                }
            )
        result = analyse_tracking_frames(frames)[0]
        self.assertEqual(result["behaviours"], [{"type": "repeated_shelf_visit", "score": 25}])
        self.assertEqual(result["behaviour_score"], 25)

    def test_empty_tracks_produce_no_people(self) -> None:
        self.assertEqual(analyse_tracking_frames([{"frame_id": 0, "tracks": []}]), [])


class ContractFileTests(unittest.TestCase):
    def test_committed_member2_sample_matches_the_generator(self) -> None:
        self.assertEqual(
            json.loads(CONTRACT_LOITER.read_text(encoding="utf-8")),
            member2_contract_loiter_frames(),
        )

    def test_analyse_cli_prints_contract_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_ROOT / "src" / "analyse_behaviour.py"), str(CONTRACT_LOITER)],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload[0]["behaviours"][0]["type"], "loitering")
        self.assertEqual(set(payload[0]["behaviours"][0]), {"type", "score"})
        self.assertEqual(payload[0]["behaviour_score"], 20)
        self.assertEqual(payload[0]["assessment"]["suspicious_level"], "none")
        validate_behaviour_result(payload[0])
        self.assertEqual(
            set(payload[0]),
            {"person_id", "behaviours", "behaviour_score", "assessment"},
        )


class ModelEvaluationTests(unittest.TestCase):
    def test_held_out_split_meets_the_metric_floor(self) -> None:
        _model, report = train_and_evaluate(build_labelled_sequences())
        self.assertGreaterEqual(report["test"]["model"]["f1"], 0.8)
        self.assertGreaterEqual(report["test"]["model"]["precision"], 0.8)
        self.assertGreaterEqual(report["test"]["model"]["recall"], 0.8)
        for name, metrics in report["test"]["rules"].items():
            self.assertGreaterEqual(metrics["f1"], 0.8, name)
        self.assertEqual(set(report["leave_one_family_out"]), set(INDICATOR_TYPES))

    def test_saved_model_and_metrics_match_a_fresh_training_run(self) -> None:
        fresh_model, fresh = train_and_evaluate(load_dataset())
        saved = load_model()
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        rows = [
            [0.0] * len(saved.feature_names),
            [20.0, 100.0, 10.0, 5.0, 12.0, 1.0, 0.0, 1.0, 20.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ]
        for row in rows:
            self.assertAlmostEqual(saved.predict_proba(row), fresh_model.predict_proba(row), places=6)
        self.assertAlmostEqual(committed["test"]["model"]["f1"], fresh["test"]["model"]["f1"], places=6)
        self.assertEqual(
            committed["test"]["model"]["confusion_matrix"],
            fresh["test"]["model"]["confusion_matrix"],
        )
        self.assertIn("synthetic", committed["dataset"]["source"])
        self.assertIn("No retail CCTV", committed["dataset"]["limitation"])


if __name__ == "__main__":
    unittest.main()
