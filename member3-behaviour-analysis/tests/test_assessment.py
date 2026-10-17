"""Combined assessment: normal shopping, isolated signals, windows, and missing time."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from assessment import TimedIndicator, assess_indicators
from behaviour import analyse_tracking_frames, validate_behaviour_result
from config import load_settings

SETTINGS = load_settings()


class NormalShoppingTests(unittest.TestCase):
    def test_examining_a_product_keeps_its_score_and_is_not_suspicious(self) -> None:
        result = analyse_tracking_frames(_handling_frames(9))[0]
        self.assertEqual(result["behaviours"], [{"type": "object_interaction", "score": 15}])
        self.assertEqual(result["behaviour_score"], 15)
        self.assertEqual(result["assessment"]["suspicious_level"], "none")
        self.assertEqual(result["assessment"]["combined_score"], 0)
        self.assertEqual(
            result["assessment"]["not_contributing"],
            [{"type": "object_interaction", "reason": "context_only"}],
        )
        self.assertEqual(set(result["behaviours"][0]), {"type", "score"})
        validate_behaviour_result(result)

    def test_loitering_with_product_handling_stays_non_suspicious(self) -> None:
        assessment = assess_indicators(
            [
                _event("loitering", 20, 0, 20),
                _event("object_interaction", 15, 2, 8),
            ],
            SETTINGS.assessment,
        )
        self.assertEqual(assessment["suspicious_level"], "none")
        self.assertEqual(assessment["combined_score"], 0)
        self.assertEqual(
            [item["reason"] for item in assessment["not_contributing"]],
            ["context_only", "context_only"],
        )


class IsolatedIndicatorTests(unittest.TestCase):
    def test_each_indicator_alone_stays_below_a_suspicious_level(self) -> None:
        for name in (
            "loitering",
            "repeated_shelf_visit",
            "abnormal_movement",
            "object_interaction",
            "possible_concealment",
        ):
            assessment = assess_indicators(
                [_event(name, 20, 0, 5)],
                SETTINGS.assessment,
            )
            self.assertEqual(assessment["suspicious_level"], "none", name)
            self.assertEqual(assessment["combined_score"], 0, name)
            self.assertEqual(assessment["contributing"], [], name)

    def test_loitering_and_a_repeated_shelf_visit_do_not_alert(self) -> None:
        result = analyse_tracking_frames(_loiter_then_return())[0]
        found = {item["type"] for item in result["behaviours"]}
        self.assertIn("loitering", found)
        self.assertIn("repeated_shelf_visit", found)
        self.assertGreater(result["behaviour_score"], 0)
        self.assertEqual(result["assessment"]["suspicious_level"], "none")
        self.assertEqual(result["assessment"]["combined_score"], 0)
        validate_behaviour_result(result)


class TimeWindowTests(unittest.TestCase):
    def test_signals_inside_the_window_set_a_level_without_replacing_detector_scores(self) -> None:
        result = analyse_tracking_frames(*_concealment_with_bottle())[0]
        self.assertEqual(
            result["behaviours"],
            [
                {"type": "object_interaction", "score": 10},
                {"type": "possible_concealment", "score": 30},
            ],
        )
        self.assertEqual(result["behaviour_score"], 40)
        self.assertEqual(result["assessment"]["suspicious_level"], "medium")
        self.assertEqual(result["assessment"]["combined_score"], 40)
        self.assertEqual(
            result["assessment"]["contributing"],
            ["object_interaction", "possible_concealment"],
        )
        validate_behaviour_result(result)

    def test_three_signals_in_one_window_are_high_and_keep_each_score(self) -> None:
        assessment = assess_indicators(
            [
                _event("loitering", 20, 0, 16),
                _event("abnormal_movement", 10, 1, 4),
                _event("possible_concealment", 18, 2, 6),
            ],
            SETTINGS.assessment,
        )
        self.assertEqual(assessment["suspicious_level"], "high")
        self.assertEqual(assessment["combined_score"], 48)
        self.assertEqual(
            assessment["contributing"],
            ["loitering", "abnormal_movement", "possible_concealment"],
        )

    def test_indicators_outside_the_window_do_not_combine(self) -> None:
        events = [
            _event("abnormal_movement", 10, 0, 2),
            _event("possible_concealment", 18, 80, 84),
        ]
        expired = assess_indicators(events, SETTINGS.assessment)
        self.assertEqual(expired["suspicious_level"], "none")
        self.assertEqual(expired["combined_score"], 0)
        self.assertEqual(
            expired["not_contributing"],
            [
                {"type": "abnormal_movement", "reason": "outside_window"},
                {"type": "possible_concealment", "reason": "outside_window"},
            ],
        )

        wider = replace(SETTINGS.assessment, window_seconds=120)
        combined = assess_indicators(events, wider)
        self.assertEqual(combined["suspicious_level"], "medium")
        self.assertEqual(combined["combined_score"], 28)
        self.assertEqual(combined["window_seconds"], 120)

    def test_a_later_fast_walk_does_not_inherit_earlier_loitering(self) -> None:
        result = analyse_tracking_frames(_loiter_then_much_later_zigzag())[0]
        self.assertEqual(result["behaviour_score"], 50)
        self.assertEqual(
            result["behaviours"],
            [
                {"type": "loitering", "score": 20},
                {"type": "abnormal_movement", "score": 30},
            ],
        )
        self.assertEqual(result["assessment"]["suspicious_level"], "none")
        self.assertEqual(result["assessment"]["combined_score"], 0)
        self.assertEqual(result["assessment"]["contributing"], [])
        validate_behaviour_result(result)


class MissingDataTests(unittest.TestCase):
    def test_indicators_without_time_are_not_combined(self) -> None:
        assessment = assess_indicators(
            [
                TimedIndicator("abnormal_movement", 10, None, None),
                TimedIndicator("possible_concealment", 18, None, None),
            ],
            SETTINGS.assessment,
        )
        self.assertEqual(assessment["suspicious_level"], "none")
        self.assertEqual(assessment["combined_score"], 0)
        self.assertEqual(
            [item["reason"] for item in assessment["not_contributing"]],
            ["missing_time", "missing_time"],
        )

    def test_one_missing_time_does_not_borrow_the_other_indicator(self) -> None:
        assessment = assess_indicators(
            [
                _event("possible_concealment", 18, 0, 3),
                TimedIndicator("abnormal_movement", 10, None, None),
            ],
            SETTINGS.assessment,
        )
        self.assertEqual(assessment["suspicious_level"], "none")
        self.assertEqual(assessment["contributing"], [])
        reasons = {item["type"]: item["reason"] for item in assessment["not_contributing"]}
        self.assertEqual(reasons["abnormal_movement"], "missing_time")
        self.assertEqual(reasons["possible_concealment"], "isolated")

    def test_missing_box_zone_and_detections_do_not_create_a_suspicious_level(self) -> None:
        frames = [
            {
                "frame_id": index,
                "tracks": [{"person_id": 3, "center": [20 + index, 20]}],
            }
            for index in range(3)
        ]
        result = analyse_tracking_frames(frames, detections=None)[0]
        self.assertEqual(result["behaviours"], [])
        self.assertEqual(result["behaviour_score"], 0)
        self.assertEqual(result["assessment"]["suspicious_level"], "none")
        self.assertEqual(result["assessment"]["not_contributing"], [])
        validate_behaviour_result(result)

    def test_reversed_interval_is_missing_time(self) -> None:
        assessment = assess_indicators(
            [TimedIndicator("possible_concealment", 18, 5, 1)],
            SETTINGS.assessment,
        )
        self.assertEqual(assessment["not_contributing"][0]["reason"], "missing_time")
        self.assertEqual(assessment["suspicious_level"], "none")


def _event(name: str, score: int, start_s: float, end_s: float) -> TimedIndicator:
    return TimedIndicator(name, score, start_s, end_s)


def _handling_frames(count: int) -> list[dict]:
    frames = []
    for second in range(count):
        foot_x = 750 if second % 2 == 0 else 762
        frames.append(_frame(second, 7, foot_x, 300, 200, f"00:00:0{second}"))
    return frames


def _loiter_then_return() -> list[dict]:
    frames = []
    for second in range(16):
        frames.append(_frame(second, 4, 750, 300, 200, f"00:00:{second:02d}"))
    frames.append(_frame(20, 4, 400, 200, 180, "00:00:20"))
    frames.append(_frame(24, 4, 750, 300, 200, "00:00:24"))
    return frames


def _concealment_with_bottle() -> tuple[list[dict], list[dict]]:
    frames = []
    detections = []
    for second in range(12):
        height = 200 if second < 6 else 80
        frames.append(_frame(second, 13, 750, 300, height, f"00:00:{second:02d}"))
        objects = []
        if second < 6:
            objects.append({"class": "bottle", "confidence": 0.91, "bbox": [735, 175, 765, 215]})
        detections.append({"frame_id": second, "detections": objects})
    return frames, detections


def _loiter_then_much_later_zigzag() -> list[dict]:
    frames = [_frame(second * 25, 8, 400, 200, 180) for second in range(16)]
    for second in range(80, 87):
        foot_x = 250 if second % 2 == 0 else 620
        frames.append(_frame(second * 25, 8, foot_x, 200, 180))
    return frames


def _frame(
    frame_id: int,
    person_id: int,
    foot_x: int,
    foot_y: int,
    height: int,
    timestamp: str | None = None,
) -> dict:
    bbox = [foot_x - 35, foot_y - height, foot_x + 35, foot_y]
    frame = {
        "frame_id": frame_id,
        "tracks": [
            {
                "person_id": person_id,
                "bbox": bbox,
                "center": [foot_x, foot_y - height // 2],
            }
        ],
    }
    if timestamp is not None:
        frame["timestamp"] = timestamp
    return frame


if __name__ == "__main__":
    unittest.main()
