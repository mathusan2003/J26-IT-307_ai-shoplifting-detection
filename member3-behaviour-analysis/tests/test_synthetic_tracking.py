"""Pipeline checks for the synthetic tracking fixtures.

These tests do not measure real-world accuracy. Labels are the motion plan,
stored apart from the tracking JSON.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from annotation_guard import AnnotationRejected, validate_annotation
from behaviour import analyse_tracking_frames, validate_behaviour_result
from config import load_settings, load_zones
from features import FEATURE_NAMES, build_profile, index_detections
from geometry import bottom_center, distance, zone_for_point
from model import load_model
from movement_history import MovementHistoryStore
from synthetic_tracking import (
    OUTPUT_ROOT,
    SyntheticInputError,
    SyntheticSettings,
    generate_dataset,
    require_concealment_detections,
)

SETTINGS = load_settings()
ZONES = load_zones()


class SyntheticTrackingTests(unittest.TestCase):
    def test_saved_files_match_the_default_seed_and_keep_labels_separate(self) -> None:
        first = generate_dataset(SyntheticSettings(seed=13))
        second = generate_dataset(SyntheticSettings(seed=13))
        self.assertEqual([_public(item) for item in first], [_public(item) for item in second])
        self.assertNotEqual(
            generate_dataset(SyntheticSettings(seed=13, jitter_px=1.0))[0].tracking,
            generate_dataset(SyntheticSettings(seed=99, jitter_px=1.0))[0].tracking,
        )
        manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["synthetic"])
        self.assertTrue(manifest["not_real_cctv"])
        self.assertEqual(manifest["fps"], 25)
        self.assertEqual(len(manifest["sequences"]), 5)
        for sequence in first:
            tracking = json.loads((OUTPUT_ROOT / "tracking" / f"{sequence.sequence_id}.json").read_text(encoding="utf-8"))
            labels = json.loads((OUTPUT_ROOT / "labels" / f"{sequence.sequence_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(tracking, sequence.tracking)
            self.assertEqual(labels, sequence.labels)
            self.assertNotIn("intended", json.dumps(tracking))
            self.assertTrue(labels["not_a_reviewed_annotation"])
            self.assertFalse(labels["real_world_accuracy"])
            with self.assertRaises(AnnotationRejected):
                validate_annotation(labels)

    def test_tracking_schema_ids_and_coordinates(self) -> None:
        for sequence in generate_dataset():
            person_ids = set()
            previous = -1
            for frame in sequence.tracking:
                self.assertIsInstance(frame["frame_id"], int)
                self.assertGreater(frame["frame_id"], previous)
                previous = frame["frame_id"]
                self.assertEqual(set(frame), {"frame_id", "tracks"})
                self.assertEqual(len(frame["tracks"]), 1)
                track = frame["tracks"][0]
                self.assertEqual(set(track), {"person_id", "bbox", "center"})
                person_ids.add(track["person_id"])
                self.assertEqual(track["person_id"], sequence.person_id)
                x_min, y_min, x_max, y_max = track["bbox"]
                self.assertLessEqual(x_min, x_max)
                self.assertLessEqual(y_min, y_max)
                self.assertEqual(len(track["center"]), 2)
                for value in (*track["bbox"], *track["center"]):
                    self.assertIsInstance(value, int)
            self.assertEqual(person_ids, {sequence.person_id})

    def test_each_plan_has_the_movement_it_names(self) -> None:
        by_id = {item.sequence_id: item for item in generate_dataset()}
        normal = _profile(by_id["normal_shopping"])
        self.assertLess(normal.longest_still_s, 15)
        self.assertLess(normal.max_speed, 250)
        self.assertEqual(normal.shelf_counts, {})
        loitering = _profile(by_id["loitering"])
        self.assertGreaterEqual(loitering.longest_still_s, 15)
        self.assertEqual(max(loitering.shelf_counts.values()), 1)
        repeated = _profile(by_id["repeated_shelf_visits"])
        self.assertGreaterEqual(repeated.shelf_counts.get("shelf a", 0), 2)
        self.assertLess(repeated.max_speed, 250)
        abnormal = _profile(by_id["abnormal_movement"])
        self.assertGreaterEqual(abnormal.max_speed, 250)
        self.assertGreaterEqual(abnormal.reversal_count, 4)
        concealment = _profile(by_id["possible_concealment"])
        self.assertLessEqual(concealment.bbox_shrink_ratio, 0.72)
        self.assertGreaterEqual(concealment.object_overlap_frames, 3)
        self.assertTrue(concealment.object_lost)
        feet = [bottom_center(frame["tracks"][0]["bbox"]) for frame in by_id["loitering"].tracking]
        self.assertTrue(all(distance(feet[0], foot) <= 40 for foot in feet))
        self.assertEqual(zone_for_point(feet[0], ZONES), "Shelf A")

    def test_existing_pipeline_reads_the_fixtures(self) -> None:
        model = load_model()
        for sequence in generate_dataset():
            results = analyse_tracking_frames(sequence.tracking, sequence.detections)
            self.assertEqual(len(results), 1)
            validate_behaviour_result(results[0])
            self.assertEqual(results[0]["person_id"], sequence.person_id)
            fired = {item["type"] for item in results[0]["behaviours"]}
            intended = {name for name, present in sequence.labels["intended"].items() if present}
            self.assertEqual(fired, intended)
            profile = _profile(sequence)
            probability = model.predict_proba(profile.as_vector())
            self.assertEqual(len(profile.as_vector()), len(FEATURE_NAMES))
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_concealment_without_a_non_person_object_is_rejected(self) -> None:
        with self.assertRaises(SyntheticInputError):
            require_concealment_detections(None)
        with self.assertRaises(SyntheticInputError):
            require_concealment_detections(
                [
                    {
                        "frame_id": 0,
                        "detections": [{"class": "person", "confidence": 0.99, "bbox": [710, 100, 790, 300]}],
                    }
                ]
            )
        sequence = {item.sequence_id: item for item in generate_dataset()}["possible_concealment"]
        silent = _profile(sequence, detections=None)
        self.assertEqual(silent.object_overlap_frames, 0)
        self.assertFalse(silent.object_lost)


def _profile(sequence, detections=...):
    store = MovementHistoryStore()
    store.ingest_frames(sequence.tracking)
    records = store.records_for(sequence.person_id)
    if detections is ...:
        indexed = index_detections(sequence.detections)
    else:
        indexed = index_detections(detections)
    return build_profile(records, indexed, SETTINGS, ZONES)


def _public(sequence) -> tuple:
    return sequence.sequence_id, sequence.tracking, sequence.labels, sequence.detections


if __name__ == "__main__":
    unittest.main()
