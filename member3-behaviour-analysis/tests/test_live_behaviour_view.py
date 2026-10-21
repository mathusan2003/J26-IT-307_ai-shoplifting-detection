"""Live overlay filters. The repository detectors are not changed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from behaviour import analyse_tracking_frames
from live_behaviour_view import (
    LIVE_EXCLUDED,
    contract_frame,
    explain_live_detectors,
    has_non_person_objects,
    history_lengths,
    live_view_from_result,
    probe_member2_live_pipeline,
)
from synthetic_tracking import generate_dataset


class LiveBehaviourViewTests(unittest.TestCase):
    def test_repeated_shelf_is_omitted_from_live_score(self) -> None:
        sequence = {item.sequence_id: item for item in generate_dataset()}["repeated_shelf_visits"]
        result = analyse_tracking_frames(sequence.tracking)[0]
        self.assertIn("repeated_shelf_visit", {item["type"] for item in result["behaviours"]})
        view = live_view_from_result(result)
        self.assertEqual([item["type"] for item in view["behaviours"]], [])
        self.assertEqual(view["live_score"], 0)
        self.assertEqual(view["excluded_from_live"], list(LIVE_EXCLUDED))
        self.assertIn("repeated_shelf_visit", {item["type"] for item in result["behaviours"]})

    def test_loitering_and_abnormal_stay_visible(self) -> None:
        by_id = {item.sequence_id: item for item in generate_dataset()}
        loiter = live_view_from_result(analyse_tracking_frames(by_id["loitering"].tracking)[0])
        abnormal = live_view_from_result(analyse_tracking_frames(by_id["abnormal_movement"].tracking)[0])
        self.assertEqual([item["type"] for item in loiter["behaviours"]], ["loitering"])
        self.assertGreater(loiter["live_score"], 0)
        self.assertEqual([item["type"] for item in abnormal["behaviours"]], ["abnormal_movement"])

    def test_object_types_need_non_person_evidence(self) -> None:
        sequence = {item.sequence_id: item for item in generate_dataset()}["possible_concealment"]
        result = analyse_tracking_frames(sequence.tracking, sequence.detections)[0]
        hidden = live_view_from_result(result, has_object_detections=False)
        shown = live_view_from_result(result, has_object_detections=True)
        self.assertNotIn("possible_concealment", {item["type"] for item in hidden["behaviours"]})
        self.assertNotIn("object_interaction", {item["type"] for item in hidden["behaviours"]})
        self.assertIn("possible_concealment", {item["type"] for item in shown["behaviours"]})
        self.assertFalse(has_non_person_objects([{"class": "person", "bbox": [0, 0, 1, 1]}]))
        self.assertTrue(has_non_person_objects([{"class": "bottle", "bbox": [0, 0, 1, 1]}]))

    def test_contract_frame_matches_member2_shape(self) -> None:
        frame = contract_frame(3, [{"person_id": 7, "bbox": [1, 2, 5, 8], "center": [3, 5]}])
        self.assertEqual(set(frame), {"frame_id", "tracks"})
        self.assertEqual(set(frame["tracks"][0]), {"person_id", "bbox", "center"})

    def test_contract_frame_stores_elapsed_seconds_as_string(self) -> None:
        frame = contract_frame(
            4,
            [{"person_id": 2, "bbox": [10, 20, 40, 80], "center": [25, 50]}],
            timestamp_s=16.25,
        )
        self.assertEqual(frame["frame_id"], 4)
        self.assertEqual(frame["timestamp"], "16.250")
        self.assertIsInstance(frame["timestamp"], str)
        self.assertEqual(frame["tracks"][0]["person_id"], 2)
        self.assertEqual(frame["tracks"][0]["center"], [25.0, 50.0])
        self.assertEqual(frame["tracks"][0]["bbox"], [10.0, 20.0, 40.0, 80.0])

    def test_movement_history_grows_on_every_appended_frame(self) -> None:
        history: list[dict] = []
        for frame_id in range(5):
            history.append(_still_frame(frame_id, timestamp_s=float(frame_id)))
            lengths = history_lengths(history)
            self.assertEqual(lengths[1], frame_id + 1)

    def test_frame_id_clock_does_not_meet_loitering_on_a_short_stream(self) -> None:
        frames = [_still_frame(frame_id) for frame_id in range(20)]
        result = analyse_tracking_frames(frames)[0]
        report = explain_live_detectors(frames)[0]
        self.assertLess(report["duration_s"], 15)
        self.assertLess(report["longest_still_s"], 15)
        self.assertEqual([item["type"] for item in result["behaviours"]], [])
        self.assertTrue(any(item.startswith("loitering_unavailable") for item in report["reasons"]))

    def test_elapsed_timestamps_invoke_loitering_in_the_live_view(self) -> None:
        frames = [_still_frame(frame_id, timestamp_s=float(frame_id)) for frame_id in range(17)]
        result = analyse_tracking_frames(frames)[0]
        view = live_view_from_result(result)
        report = explain_live_detectors(frames)[0]
        self.assertEqual(report["history_length"], 17)
        self.assertGreaterEqual(report["longest_still_s"], 15)
        self.assertTrue(report["loitering_ready"])
        self.assertIn("loitering", {item["type"] for item in result["behaviours"]})
        self.assertEqual([item["type"] for item in view["behaviours"]], ["loitering"])
        self.assertGreater(view["live_score"], 0)

    def test_two_distant_points_with_short_dt_invoke_abnormal_movement(self) -> None:
        frames = [
            contract_frame(
                0,
                [{"person_id": 3, "bbox": [80, 160, 120, 240], "center": [100, 200]}],
                timestamp_s=0.0,
            ),
            contract_frame(
                1,
                [{"person_id": 3, "bbox": [120, 160, 160, 240], "center": [140, 200]}],
                timestamp_s=0.1,
            ),
        ]
        result = analyse_tracking_frames(frames)[0]
        view = live_view_from_result(result)
        report = explain_live_detectors(frames)[0]
        self.assertGreaterEqual(report["max_speed"], 250)
        self.assertTrue(report["abnormal_ready"])
        self.assertEqual([item["type"] for item in result["behaviours"]], ["abnormal_movement"])
        self.assertEqual([item["type"] for item in view["behaviours"]], ["abnormal_movement"])
        self.assertGreater(view["live_score"], 0)

    def test_repeated_shelf_stays_excluded_from_timestamped_live_score(self) -> None:
        sequence = {item.sequence_id: item for item in generate_dataset()}["repeated_shelf_visits"]
        result = analyse_tracking_frames(sequence.tracking)[0]
        view = live_view_from_result(result)
        self.assertIn("repeated_shelf_visit", {item["type"] for item in result["behaviours"]})
        self.assertNotIn("repeated_shelf_visit", {item["type"] for item in view["behaviours"]})
        self.assertEqual(view["live_score"], 0)

    def test_probe_reports_missing_weights(self) -> None:
        report = probe_member2_live_pipeline(MODULE_ROOT / "this-weight-does-not-exist.pt")
        self.assertFalse(report["ready"])
        self.assertTrue(any("weights not found" in item for item in report["missing"]))


def _still_frame(frame_id: int, timestamp_s: float | None = None) -> dict:
    return contract_frame(
        frame_id,
        [{"person_id": 1, "bbox": [100, 100, 180, 300], "center": [140, 200]}],
        timestamp_s=timestamp_s,
    )


if __name__ == "__main__":
    unittest.main()
