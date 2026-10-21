"""Member 2 tracking output into Member 3 analysis.

These tests feed contract, telemetry, and messy tracker rows into the
existing detectors. They do not run YOLO and they do not change scores.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT / "src"))

from behaviour import analyse_tracking_frames, validate_behaviour_result
from live_behaviour_view import LiveBehaviourSession, format_overlay_lines
from member2_adapter import (
    analyse_member2_frames,
    apply_source_fps,
    frame_from_member2_tracks,
    load_member2_tracking_sequence,
    normalize_sequence,
)
from movement_history import MovementHistoryStore
from tracking_input import observations_from_frame

MEMBER2_SCRIPT = """
import json
import sys
from pathlib import Path
from src.pipeline import (
    DEFAULT_CONFIG,
    load_tracker_settings,
    make_tracker,
    read_json,
    track_detection_sequence,
)
frames = read_json(Path("tests/fixtures/sample_detections.json"))
result = track_detection_sequence(frames, make_tracker(load_tracker_settings(DEFAULT_CONFIG)))
json.dump(result, sys.stdout)
"""

SAMPLE_TRACKING = MODULE_ROOT / "samples" / "sample_tracking_frames.json"
CONTRACT_LOITER = MODULE_ROOT / "samples" / "member2_contract_loiter.json"


class Member2ContractHandoffTests(unittest.TestCase):
    def test_member2_pipeline_output_is_analysed_without_redetecting(self) -> None:
        frames = _member2_pipeline_frames()
        self.assertEqual(set(frames[0]), {"frame_id", "tracks"})
        self.assertEqual(set(frames[0]["tracks"][0]), {"person_id", "bbox", "center"})
        self.assertEqual({track["person_id"] for track in frames[0]["tracks"]}, {1, 2})
        for frame in frames:
            self.assertEqual(len(frame["tracks"]), 2)

        results = analyse_member2_frames(frames)
        for result in results:
            validate_behaviour_result(result)
            self.assertEqual(
                set(result),
                {"person_id", "behaviours", "behaviour_score", "assessment"},
            )
        self.assertEqual({item["person_id"] for item in results}, {1, 2})

    def test_sample_tracking_frames_keep_member4_fields(self) -> None:
        frames = load_member2_tracking_sequence(SAMPLE_TRACKING)
        results = analyse_tracking_frames(frames)
        self.assertEqual([item["person_id"] for item in results], [7, 12])
        for result in results:
            validate_behaviour_result(result)
            self.assertEqual(result["behaviours"], [])
            self.assertEqual(result["behaviour_score"], 0)
            self.assertEqual(result["assessment"]["suspicious_level"], "none")

    def test_member2_contract_loiter_file_still_scores_loitering_only(self) -> None:
        results = analyse_member2_frames(load_member2_tracking_sequence(CONTRACT_LOITER))
        self.assertEqual(results[0]["person_id"], 12)
        self.assertEqual(results[0]["behaviours"], [{"type": "loitering", "score": 20}])
        self.assertEqual(results[0]["assessment"]["suspicious_level"], "none")
        validate_behaviour_result(results[0])


class Member2TelemetryFormatTests(unittest.TestCase):
    def test_numeric_telemetry_timestamp_is_used_as_elapsed_time(self) -> None:
        observations = observations_from_frame(_telemetry_frame(4, 2.5, [_track(3, 100, 200)]))
        self.assertEqual(observations[0]["timestamp"], "2.500")
        self.assertEqual(observations[0]["person_id"], 3)
        self.assertEqual(observations[0]["bbox"], [65.0, 20.0, 135.0, 200.0])
        self.assertEqual(observations[0]["position"], [100.0, 110.0])

    def test_telemetry_extras_are_stripped_and_zone_is_kept(self) -> None:
        payload = [
            _telemetry_frame(
                0,
                0.0,
                [
                    {
                        **_track(8, 750, 300),
                        "trajectory": [[750, 200], [751, 201]],
                        "zone": "Shelf A",
                        "confidence": 0.91,
                        "start_zone": False,
                        "end_zone": False,
                        "reached_end_zone": False,
                    }
                ],
            )
        ]
        frames = normalize_sequence(payload)
        self.assertEqual(set(frames[0]["tracks"][0]), {"person_id", "bbox", "center", "zone"})
        self.assertEqual(frames[0]["tracks"][0]["zone"], "Shelf A")
        self.assertNotIn("analytics", frames[0])
        self.assertEqual(frames[0]["timestamp"], "0.000")

    def test_dataset_runner_one_based_frames_are_accepted(self) -> None:
        frames = normalize_sequence(
            [
                {
                    "frame_id": 1,
                    "tracks": [
                        {
                            **_track(4, 400, 200),
                            "confidence": 0.8,
                            "zone": "Aisle 1",
                            "start_zone": False,
                            "end_zone": False,
                            "reached_end_zone": False,
                        }
                    ],
                }
            ]
        )
        self.assertEqual(frames[0]["frame_id"], 1)
        results = analyse_member2_frames(frames)
        self.assertEqual(results[0]["person_id"], 4)
        validate_behaviour_result(results[0])


class MultiPersonAndGapTests(unittest.TestCase):
    def test_empty_frames_keep_earlier_history_and_add_no_people(self) -> None:
        frames = [
            frame_from_member2_tracks(0, [_track(1, 400, 200)], elapsed_s=0.0),
            frame_from_member2_tracks(1, [], elapsed_s=1.0),
            {"frame_id": 2, "tracks": [], "timestamp": 2.0},
            frame_from_member2_tracks(3, [_track(1, 400, 200)], elapsed_s=3.0),
        ]
        store = MovementHistoryStore()
        store.ingest_frames(normalize_sequence(frames))
        self.assertEqual(store.person_ids(), [1])
        self.assertEqual([record.frame_id for record in store.records_for(1)], [0, 3])

    def test_missing_tracks_key_is_treated_as_empty(self) -> None:
        frames = normalize_sequence([{"frame_id": 5, "timestamp": 1.0}])
        self.assertEqual(frames[0]["tracks"], [])
        self.assertEqual(analyse_member2_frames(frames), [])

    def test_multiple_people_are_scored_separately(self) -> None:
        frames = []
        for second in range(16):
            frames.append(
                frame_from_member2_tracks(
                    second,
                    [
                        _track(1, 400, 200),
                        _track(2, 240 + second * 80, 200),
                    ],
                    elapsed_s=float(second),
                )
            )
        results = {item["person_id"]: item for item in analyse_member2_frames(frames)}
        self.assertEqual(set(results), {1, 2})
        self.assertEqual(results[1]["behaviours"], [{"type": "loitering", "score": 20}])
        self.assertNotIn("loitering", {item["type"] for item in results[2]["behaviours"]})
        validate_behaviour_result(results[1])
        validate_behaviour_result(results[2])

    def test_persistent_ids_survive_a_missed_frame(self) -> None:
        frames = [
            frame_from_member2_tracks(0, [_track(12, 400, 200), _track(7, 100, 100)], elapsed_s=0.0),
            frame_from_member2_tracks(1, [_track(7, 110, 100)], elapsed_s=1.0),
            frame_from_member2_tracks(2, [_track(12, 400, 200), _track(7, 120, 100)], elapsed_s=2.0),
        ]
        store = MovementHistoryStore()
        store.ingest_frames(frames)
        self.assertEqual([record.frame_id for record in store.records_for(12)], [0, 2])
        self.assertEqual([record.frame_id for record in store.records_for(7)], [0, 1, 2])

    def test_temporary_id_change_starts_a_new_history(self) -> None:
        frames = [
            frame_from_member2_tracks(0, [_track(1, 400, 200)], elapsed_s=0.0),
            frame_from_member2_tracks(1, [_track(1, 401, 200)], elapsed_s=1.0),
            frame_from_member2_tracks(2, [_track(9, 402, 200)], elapsed_s=2.0),
            frame_from_member2_tracks(3, [_track(9, 403, 200)], elapsed_s=3.0),
        ]
        store = MovementHistoryStore()
        store.ingest_frames(frames)
        self.assertEqual(store.person_ids(), [1, 9])
        self.assertEqual(len(store.records_for(1)), 2)
        self.assertEqual(len(store.records_for(9)), 2)
        results = analyse_member2_frames(frames)
        self.assertEqual({item["person_id"] for item in results}, {1, 9})
        for result in results:
            self.assertEqual(result["behaviours"], [])
            validate_behaviour_result(result)


class ElapsedTimeAndLoiteringTests(unittest.TestCase):
    def test_slow_webcam_frames_need_elapsed_time_for_loitering(self) -> None:
        unstamped = [_still(frame_id) for frame_id in range(20)]
        elapsed = [_still(frame_id, timestamp_s=float(frame_id)) for frame_id in range(16)]
        self.assertEqual(analyse_member2_frames(unstamped)[0]["behaviours"], [])
        self.assertEqual(
            analyse_member2_frames(elapsed)[0]["behaviours"],
            [{"type": "loitering", "score": 20}],
        )

    def test_walking_outside_the_radius_is_not_loitering(self) -> None:
        frames = []
        for second in range(20):
            frames.append(_still(second, x=200 + second * 50, timestamp_s=float(second)))
        result = analyse_member2_frames(frames)[0]
        self.assertNotIn("loitering", {item["type"] for item in result["behaviours"]})
        validate_behaviour_result(result)

    def test_return_to_the_same_spot_after_leaving_is_not_one_loiter(self) -> None:
        frames = []
        for second in range(16):
            x = 200 if second in {0, 15} else 200 + 80
            frames.append(_still(second, x=x, timestamp_s=float(second)))
        result = analyse_member2_frames(frames)[0]
        self.assertNotIn("loitering", {item["type"] for item in result["behaviours"]})

    def test_source_fps_stamps_recorded_video_frame_ids(self) -> None:
        frames = apply_source_fps(
            [frame_from_member2_tracks(0, [_track(1, 400, 200)]), frame_from_member2_tracks(30, [_track(1, 400, 200)])],
            30.0,
        )
        self.assertEqual(frames[0]["timestamp"], "0.000")
        self.assertEqual(frames[1]["timestamp"], "1.000")


class MalformedTrackingTests(unittest.TestCase):
    def test_malformed_tracks_are_skipped_without_dropping_valid_people(self) -> None:
        frames = normalize_sequence(
            [
                {
                    "frame_id": 0,
                    "timestamp": 0.0,
                    "tracks": [
                        _track(1, 400, 200),
                        "not-a-track",
                        {"person_id": "abc", "bbox": [1, 2, 3, 4]},
                        {"bbox": [10, 10, 20, 20]},
                        {"person_id": 2, "bbox": [10, 10, 5, 20]},
                        _track(3, 750, 300),
                    ],
                }
            ]
        )
        self.assertEqual([track["person_id"] for track in frames[0]["tracks"]], [1, 3])
        results = analyse_member2_frames(frames)
        self.assertEqual({item["person_id"] for item in results}, {1, 3})

    def test_invalid_frame_objects_are_skipped(self) -> None:
        frames = normalize_sequence(
            [
                {"tracks": [_track(1, 400, 200)]},
                "bad-frame",
                {"frame_id": 2, "tracks": [_track(1, 400, 200)], "timestamp": 2.0},
            ]
        )
        self.assertEqual([frame["frame_id"] for frame in frames], [2])

    def test_person_only_detections_do_not_create_object_interaction(self) -> None:
        frames = [
            frame_from_member2_tracks(second, [_track(5, 400 + second * 8, 200)], elapsed_s=float(second))
            for second in range(6)
        ]
        detections = [
            {
                "frame_id": second,
                "detections": [
                    {"class": "person", "confidence": 0.99, "bbox": [270, 20, 340, 200]},
                ],
            }
            for second in range(6)
        ]
        result = analyse_member2_frames(frames, detections)[0]
        types = {item["type"] for item in result["behaviours"]}
        self.assertNotIn("object_interaction", types)
        self.assertNotIn("possible_concealment", types)

    def test_mixed_clocks_still_raise_in_the_analyser(self) -> None:
        frames = [
            frame_from_member2_tracks(0, [_track(1, 400, 200)], elapsed_s=0.0),
            frame_from_member2_tracks(1, [_track(1, 400, 200)]),
        ]
        with self.assertRaises(ValueError):
            analyse_tracking_frames(frames)


class LiveWebcamPathTests(unittest.TestCase):
    def test_live_loop_passes_member2_rows_and_elapsed_time(self) -> None:
        history = []
        views = {}
        started = 0.0
        for frame_id, person_id in enumerate((1, 1, 1, 2, 2)):
            elapsed = started + frame_id * 0.4
            tracks = [_track(person_id, 400, 200)]
            history.append(frame_from_member2_tracks(frame_id, tracks, elapsed_s=elapsed))
            results = analyse_member2_frames(history)
            views = {item["person_id"]: item for item in results}
        self.assertEqual(set(views), {1, 2})
        self.assertAlmostEqual(
            float(history[-1]["timestamp"]) - float(history[0]["timestamp"]),
            1.6,
        )
        self.assertLess(float(history[-1]["timestamp"]), 15)
        for result in views.values():
            self.assertEqual(result["behaviours"], [])
            validate_behaviour_result(result)

    def test_adapter_round_trips_a_member2_json_file(self) -> None:
        payload = [_telemetry_frame(index, index / 30.0, [_track(1, 400, 200)]) for index in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking_log.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            frames = load_member2_tracking_sequence(path)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[2]["timestamp"], "0.067")
        self.assertEqual(frames[2]["tracks"][0]["person_id"], 1)


class LiveMember2ToMember3SessionTests(unittest.TestCase):
    def test_single_frame_does_not_invent_loitering_or_normal_movement(self) -> None:
        session = LiveBehaviourSession()
        views = session.update(0, [_track(4, 400, 200)], elapsed_s=0.0)
        overlay = " ".join(format_overlay_lines(views[4]))
        self.assertEqual(views[4]["behaviours"], [])
        self.assertEqual(views[4]["live_score"], 0)
        self.assertEqual(views[4]["combined_score"], 0)
        self.assertEqual(views[4]["suspicious_level"], "none")
        self.assertEqual(views[4]["contributing"], [])
        self.assertEqual(session.history_length(4), 1)
        self.assertNotIn("loitering:20", overlay)
        self.assertNotIn("normal_movement", overlay)
        self.assertTrue(any("need more frames" in item for item in views[4]["pending"]))
        validate_behaviour_result(session.last_results[4])

    def test_loitering_score_updates_only_after_persistent_history(self) -> None:
        session = LiveBehaviourSession()
        views = {}
        for second in range(17):
            views = session.update(second, [_track(1, 400, 200)], elapsed_s=float(second))
            if second < 15:
                self.assertEqual(views[1]["behaviours"], [])
                self.assertEqual(views[1]["live_score"], 0)
                self.assertTrue(any(item.startswith("loitering waiting") for item in views[1]["pending"]))
        self.assertEqual(session.history_length(1), 17)
        self.assertEqual(views[1]["behaviours"], [{"type": "loitering", "score": 21}])
        self.assertEqual(views[1]["live_score"], 21)
        self.assertEqual(views[1]["suspicious_level"], "none")
        self.assertEqual(views[1]["contributing"], [])
        overlay = format_overlay_lines(views[1])
        self.assertEqual(overlay[0], "loitering:21")
        self.assertIn("level=none", overlay[1])
        self.assertEqual(overlay[2], "contrib=none")
        validate_behaviour_result(session.last_results[1])

    def test_multiple_people_keep_separate_histories_and_scores(self) -> None:
        session = LiveBehaviourSession()
        views = {}
        for second in range(16):
            views = session.update(
                second,
                [_track(1, 400, 200), _track(2, 240 + second * 80, 200)],
                elapsed_s=float(second),
            )
        self.assertEqual(set(views), {1, 2})
        self.assertEqual(session.history_length(1), 16)
        self.assertEqual(session.history_length(2), 16)
        self.assertEqual(views[1]["behaviours"], [{"type": "loitering", "score": 20}])
        self.assertNotIn("loitering", {item["type"] for item in views[2]["behaviours"]})
        self.assertEqual(views[1]["suspicious_level"], "none")
        self.assertEqual(views[2]["suspicious_level"], "none")
        validate_behaviour_result(session.last_results[1])
        validate_behaviour_result(session.last_results[2])

    def test_empty_frames_do_not_reset_history(self) -> None:
        session = LiveBehaviourSession()
        session.update(0, [_track(8, 400, 200)], elapsed_s=0.0)
        session.update(1, [], elapsed_s=1.0)
        session.update(2, None, elapsed_s=2.0)
        views = session.update(3, [_track(8, 400, 200)], elapsed_s=3.0)
        self.assertEqual(session.store.person_ids(), [8])
        self.assertEqual([record.frame_id for record in session.store.records_for(8)], [0, 3])
        self.assertEqual(views[8]["live_score"], 0)
        self.assertEqual(session.history_length(8), 2)

    def test_missing_optional_zone_and_center_still_score(self) -> None:
        session = LiveBehaviourSession()
        box = [365, 20, 435, 200]
        views = session.update(
            0,
            [{"person_id": 6, "bbox": box}],
            elapsed_s=0.0,
        )
        self.assertEqual(session.last_frame["tracks"][0]["center"], [400.0, 110.0])
        self.assertNotIn("zone", session.last_frame["tracks"][0])
        self.assertEqual(views[6]["behaviours"], [])
        self.assertTrue(any("repeated_shelf_visit waiting" in item for item in views[6]["pending"]))
        validate_behaviour_result(session.last_results[6])

    def test_person_only_detections_do_not_claim_concealment_or_interaction(self) -> None:
        session = LiveBehaviourSession()
        views = {}
        people = [{"class": "person", "confidence": 0.99, "bbox": [365, 20, 435, 200]}]
        for second in range(6):
            views = session.update(
                second,
                [_track(5, 400 + second * 8, 200)],
                elapsed_s=float(second),
                detections=people,
            )
        types = {item["type"] for item in views[5]["behaviours"]}
        overlay = " ".join(format_overlay_lines(views[5]))
        self.assertNotIn("object_interaction", types)
        self.assertNotIn("possible_concealment", types)
        self.assertNotIn("object_interaction:", overlay)
        self.assertNotIn("possible_concealment:", overlay)
        self.assertTrue(any("person boxes only" in item for item in views[5]["pending"]))

    def test_abnormal_movement_score_updates_when_history_shows_high_speed(self) -> None:
        session = LiveBehaviourSession()
        first = session.update(
            0,
            [_track(3, 100, 200, height=80, width=40)],
            elapsed_s=0.0,
        )
        self.assertEqual(first[3]["behaviours"], [])
        second = session.update(
            1,
            [_track(3, 140, 200, height=80, width=40)],
            elapsed_s=0.1,
        )
        self.assertEqual(second[3]["behaviours"], [{"type": "abnormal_movement", "score": 10}])
        self.assertEqual(second[3]["live_score"], 10)
        self.assertEqual(second[3]["suspicious_level"], "none")
        self.assertIn("abnormal_movement:10", format_overlay_lines(second[3])[0])
        validate_behaviour_result(session.last_results[3])

    def test_member2_tracker_keeps_two_stable_ids(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "member2-person-tracking"))
        from src.pipeline import load_tracker_settings, make_tracker

        tracker = make_tracker(load_tracker_settings())
        seen = None
        for step in range(8):
            detections = [
                {"class": "person", "confidence": 0.82, "bbox": [10 + step, 40, 140 + step, 420]},
                {"class": "person", "confidence": 0.71, "bbox": [320 + step, 60, 450 + step, 430]},
            ]
            tracks = tracker.update(detections)
            ids = sorted(int(track.track_id) for track in tracks)
            self.assertEqual(len(ids), 2)
            if seen is None:
                seen = ids
            else:
                self.assertEqual(ids, seen)


def _member2_pipeline_frames() -> list[dict]:
    completed = subprocess.run(
        [sys.executable, "-c", MEMBER2_SCRIPT],
        cwd=REPO_ROOT / "member2-person-tracking",
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _track(person_id: int, foot_x: int, foot_y: int, height: int = 180, width: int = 70) -> dict:
    return {
        "person_id": person_id,
        "bbox": [foot_x - width // 2, foot_y - height, foot_x + width // 2, foot_y],
        "center": [float(foot_x), float(foot_y - height // 2)],
    }


def _still(frame_id: int, x: int = 400, timestamp_s: float | None = None) -> dict:
    return frame_from_member2_tracks(frame_id, [_track(1, x, 200)], elapsed_s=timestamp_s)


def _telemetry_frame(frame_id: int, timestamp: float, tracks: list[dict]) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "tracks": tracks,
        "analytics": {
            "active_occupancy": len(tracks),
            "entry_count": 0,
            "exit_count": 0,
            "dwell_seconds": {},
            "transitions": [],
        },
    }


if __name__ == "__main__":
    unittest.main()
