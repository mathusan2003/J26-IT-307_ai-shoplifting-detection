"""Live overlay filters. The repository detectors are not changed."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from behaviour import analyse_tracking_frames
from live_behaviour_view import (
    LIVE_EXCLUDED,
    MEMBER2_ROOT,
    contract_frame,
    explain_live_detectors,
    format_overlay_lines,
    has_non_person_objects,
    history_lengths,
    keep_whole_people,
    live_view_from_result,
    probe_member2_live_pipeline,
    resolve_member2_live_weights,
)
from synthetic_tracking import generate_dataset


class LiveBehaviourViewTests(unittest.TestCase):
    def test_repeated_shelf_is_shown_when_detector_fires(self) -> None:
        sequence = {item.sequence_id: item for item in generate_dataset()}["repeated_shelf_visits"]
        result = analyse_tracking_frames(sequence.tracking)[0]
        self.assertIn("repeated_shelf_visit", {item["type"] for item in result["behaviours"]})
        view = live_view_from_result(result)
        self.assertEqual([item["type"] for item in view["behaviours"]], ["repeated_shelf_visit"])
        self.assertGreater(view["live_score"], 0)
        self.assertEqual(view["excluded_from_live"], list(LIVE_EXCLUDED))
        self.assertEqual(view["suspicious_level"], result["assessment"]["suspicious_level"])
        self.assertEqual(view["combined_score"], result["assessment"]["combined_score"])

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
        self.assertTrue(any(item.startswith("loitering waiting") for item in report["reasons"]))

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

    def test_repeated_shelf_stays_on_timestamped_live_score(self) -> None:
        sequence = {item.sequence_id: item for item in generate_dataset()}["repeated_shelf_visits"]
        result = analyse_tracking_frames(sequence.tracking)[0]
        view = live_view_from_result(result)
        self.assertIn("repeated_shelf_visit", {item["type"] for item in result["behaviours"]})
        self.assertIn("repeated_shelf_visit", {item["type"] for item in view["behaviours"]})
        self.assertGreater(view["live_score"], 0)
        self.assertTrue(any(line.startswith("repeated_shelf_visit:") for line in format_overlay_lines(view)))

    def test_probe_reports_missing_weights_when_no_local_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory)
            report = probe_member2_live_pipeline(empty / "missing.pt", roots=[empty])
        self.assertFalse(report["weights_exist"])
        self.assertTrue(any("weights not found" in item for item in report["missing"]))

    def test_resolve_weights_uses_an_existing_member2_file(self) -> None:
        resolved = resolve_member2_live_weights()
        coco = MEMBER2_ROOT / "models" / "yolov8n.pt"
        self.assertTrue(resolved.is_file(), f"expected an on-disk weight file, got {resolved}")
        self.assertEqual(resolved.suffix, ".pt")
        self.assertNotEqual(resolved.name, ".gitkeep")
        self.assertEqual(resolved.resolve().parent.name, "models")
        if coco.is_file():
            self.assertEqual(resolved.resolve(), coco.resolve())

    def test_keep_whole_people_keeps_two_side_by_side_webcam_boxes(self) -> None:
        detections = [
            {"class": "person", "confidence": 0.82, "bbox": [209.3, 149.2, 646.3, 503.8]},
            {"class": "person", "confidence": 0.53, "bbox": [7.9, 76.0, 222.7, 502.6]},
        ]
        kept = keep_whole_people(detections, 649, 508)
        self.assertEqual(len(kept), 2)
        areas = sorted(_area(item["bbox"]) for item in kept)
        self.assertGreater(areas[0], 0.20 * 649 * 508)
        self.assertGreater(areas[1], 0.20 * 649 * 508)

    def test_keep_whole_people_drops_nested_face_and_tiny_specs(self) -> None:
        body = {"class": "person", "confidence": 0.9, "bbox": [40.0, 40.0, 220.0, 460.0]}
        face = {"class": "person", "confidence": 0.8, "bbox": [80.0, 50.0, 160.0, 150.0]}
        spec = {"class": "person", "confidence": 0.7, "bbox": [10.0, 10.0, 24.0, 32.0]}
        kept = keep_whole_people([body, face, spec], 640, 480)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["bbox"], [40.0, 40.0, 220.0, 460.0])

    def test_keep_whole_people_does_not_apply_cctv_area_cap(self) -> None:
        close = {"class": "person", "confidence": 0.88, "bbox": [20.0, 30.0, 500.0, 470.0]}
        self.assertGreater(_area(close["bbox"]) / (640 * 480), 0.45)
        kept = keep_whole_people([close], 640, 480)
        self.assertEqual(len(kept), 1)

    def test_keep_whole_people_keeps_bottom_clipped_webcam_closeup(self) -> None:
        close = {"class": "person", "confidence": 0.754, "bbox": [76.3, 163.3, 631.5, 479.7]}
        width, height = 631.5 - 76.3, 479.7 - 163.3
        self.assertGreater(width / height, 1.50)
        kept = keep_whole_people([close], 640, 480)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["bbox"], [76.3, 163.3, 631.5, 479.7])


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _still_frame(frame_id: int, timestamp_s: float | None = None) -> dict:
    return contract_frame(
        frame_id,
        [{"person_id": 1, "bbox": [100, 100, 180, 300], "center": [140, 200]}],
        timestamp_s=timestamp_s,
    )


if __name__ == "__main__":
    unittest.main()
