"""Direct checks for each behaviour detector and for Member 2 tracking JSON."""

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
from tracking_input import load_tracking_sequence

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


class DetectorCaseTests(unittest.TestCase):
    def test_loitering_starts_at_fifteen_seconds_and_does_not_imply_handling(self) -> None:
        under = analyse_tracking_frames(_still_frames(14, foot=(400, 200)))
        at_threshold = analyse_tracking_frames(_still_frames(15, foot=(400, 200)))
        over = analyse_tracking_frames(_still_frames(16, foot=(400, 200)))

        self.assertEqual(under[0]["behaviours"], [])
        self.assertEqual(at_threshold[0]["behaviours"], [{"type": "loitering", "score": 20}])
        self.assertEqual(over[0]["behaviours"], [{"type": "loitering", "score": 21}])
        self.assertEqual(over[0]["behaviour_score"], 21)

    def test_loitering_survives_frames_arriving_out_of_order(self) -> None:
        frames = list(reversed(_still_frames(15, foot=(400, 200))))
        result = analyse_tracking_frames(frames)[0]
        self.assertEqual(result["behaviours"], [{"type": "loitering", "score": 20}])

    def test_repeated_shelf_visit_needs_the_same_shelf_and_caps_at_40(self) -> None:
        different_shelves = analyse_tracking_frames(
            _path(
                [
                    (400, 400, 180),
                    (550, 350, 180),
                    (700, 300, 200),
                    (710, 300, 200),
                    (560, 350, 180),
                    (750, 500, 200),
                    (760, 500, 200),
                    (880, 450, 180),
                    (1050, 400, 180),
                ]
            )
        )
        one_return = analyse_tracking_frames(
            _path(
                [
                    (750, 300, 200),
                    (751, 300, 200),
                    (600, 300, 180),
                    (450, 300, 180),
                    (600, 300, 180),
                    (750, 300, 200),
                    (751, 300, 200),
                ]
            )
        )
        two_returns = analyse_tracking_frames(
            _path(
                [
                    (750, 300, 200),
                    (751, 300, 200),
                    (600, 300, 180),
                    (450, 300, 180),
                    (600, 300, 180),
                    (750, 300, 200),
                    (751, 300, 200),
                    (600, 300, 180),
                    (450, 300, 180),
                    (600, 300, 180),
                    (750, 300, 200),
                    (751, 300, 200),
                ]
            )
        )

        self.assertEqual(different_shelves[0]["behaviours"], [])
        self.assertEqual(one_return[0]["behaviours"], [{"type": "repeated_shelf_visit", "score": 25}])
        self.assertEqual(two_returns[0]["behaviours"], [{"type": "repeated_shelf_visit", "score": 40}])

    def test_abnormal_movement_scores_a_fast_zigzag_and_ignores_a_brisk_walk(self) -> None:
        zigzag = []
        for second in range(7):
            foot_x = 250 if second % 2 == 0 else 620
            zigzag.append(_frame(second, 6, foot_x, 200, 180, timestamp=f"00:00:0{second}"))
        walk = []
        for second, foot_x in enumerate(range(240, 1100, 140)):
            walk.append(_frame(second, 5, foot_x, 200, 180, timestamp=f"00:00:0{second}"))

        self.assertEqual(
            analyse_tracking_frames(zigzag)[0]["behaviours"],
            [{"type": "abnormal_movement", "score": 30}],
        )
        self.assertEqual(analyse_tracking_frames(walk)[0]["behaviours"], [])

    def test_object_interaction_from_handling_or_a_nearby_object(self) -> None:
        handling = analyse_tracking_frames(_handling_frames(9))
        too_short = analyse_tracking_frames(_handling_frames(4))
        near_object = analyse_tracking_frames(*_bottle_sequence(confidence=0.9, present_frames=4))
        no_detection_file = analyse_tracking_frames(_bottle_sequence(confidence=0.9, present_frames=4)[0])
        ignored_person = analyse_tracking_frames(*_bottle_sequence(confidence=0.99, present_frames=4, class_name="person"))
        low_confidence = analyse_tracking_frames(*_bottle_sequence(confidence=0.2, present_frames=4))

        self.assertEqual(handling[0]["behaviours"], [{"type": "object_interaction", "score": 15}])
        self.assertEqual(too_short[0]["behaviours"], [])
        self.assertEqual(near_object[0]["behaviours"], [{"type": "object_interaction", "score": 10}])
        self.assertEqual(no_detection_file[0]["behaviours"], [])
        self.assertEqual(ignored_person[0]["behaviours"], [])
        self.assertEqual(low_confidence[0]["behaviours"], [])

    def test_concealment_requires_a_later_shrink_after_shelf_or_object_contact(self) -> None:
        shrink_on_shelf = analyse_tracking_frames(_shrink_frames(foot=(750, 300), shrink=True))
        shrink_in_aisle = analyse_tracking_frames(_shrink_frames(foot=(400, 400), shrink=True))
        stands_up = analyse_tracking_frames(_shrink_frames(foot=(750, 300), shrink=False))
        with_loss = analyse_tracking_frames(*_concealment_with_bottle())

        self.assertEqual(shrink_on_shelf[0]["behaviours"], [{"type": "possible_concealment", "score": 18}])
        self.assertEqual(shrink_in_aisle[0]["behaviours"], [])
        self.assertEqual(stands_up[0]["behaviours"], [])
        self.assertEqual(
            with_loss[0]["behaviours"],
            [
                {"type": "object_interaction", "score": 10},
                {"type": "possible_concealment", "score": 30},
            ],
        )
        self.assertEqual(with_loss[0]["behaviour_score"], 40)
        validate_behaviour_result(with_loss[0])


class TrackingJsonTests(unittest.TestCase):
    def test_bom_tracking_file_loads(self) -> None:
        payload = [{"frame_id": 0, "tracks": [{"person_id": 1, "center": [10, 10]}]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.json"
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
            self.assertEqual(load_tracking_sequence(path), payload)

    def test_single_frame_and_frames_wrapper_are_accepted(self) -> None:
        frame = _frame(0, 1, 400, 200, 180)
        with tempfile.TemporaryDirectory() as directory:
            single = Path(directory) / "one.json"
            wrapped = Path(directory) / "many.json"
            single.write_text(json.dumps(frame), encoding="utf-8")
            wrapped.write_text(json.dumps({"frames": [frame]}), encoding="utf-8")
            self.assertEqual(load_tracking_sequence(single), [frame])
            self.assertEqual(load_tracking_sequence(wrapped), [frame])

    def test_member2_tracker_json_keeps_ids_and_contract_fields(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", MEMBER2_SCRIPT],
            cwd=REPO_ROOT / "member2-person-tracking",
            check=True,
            capture_output=True,
            text=True,
        )
        frames = json.loads(completed.stdout)
        self.assertEqual(set(frames[0]), {"frame_id", "tracks"})
        self.assertEqual(set(frames[0]["tracks"][0]), {"person_id", "bbox", "center"})

        results = analyse_tracking_frames(frames)
        for result in results:
            validate_behaviour_result(result)
        by_person = {item["person_id"]: item for item in results}
        self.assertEqual(set(by_person), {1, 2})
        # Four consecutive frames move about 28 px each. With no timestamp,
        # time is frame_id / 25, so that step is about 700 px/s.
        self.assertEqual(by_person[1]["behaviours"], [{"type": "abnormal_movement", "score": 10}])
        self.assertEqual(by_person[2]["behaviours"], [{"type": "abnormal_movement", "score": 10}])
        self.assertEqual(by_person[1]["behaviour_score"], 10)


def _still_frames(seconds: int, foot: tuple[int, int]) -> list[dict]:
    foot_x, foot_y = foot
    return [_frame(index * 25, 1, foot_x, foot_y, 180) for index in range(seconds + 1)]


def _handling_frames(count: int) -> list[dict]:
    frames = []
    for second in range(count):
        foot_x = 750 if second % 2 == 0 else 762
        frames.append(_frame(second, 7, foot_x, 300, 200, timestamp=f"00:00:0{second}"))
    return frames


def _path(points: list[tuple[int, int, int]]) -> list[dict]:
    frames = []
    for second, (foot_x, foot_y, height) in enumerate(points):
        frames.append(_frame(second, 1, foot_x, foot_y, height, timestamp=f"00:00:{second:02d}"))
    return frames


def _shrink_frames(foot: tuple[int, int], shrink: bool) -> list[dict]:
    frames = []
    foot_x, foot_y = foot
    for second in range(8):
        if shrink:
            height = 200 if second < 4 else 80
        else:
            height = 80 if second < 4 else 200
        frames.append(_frame(second, 10, foot_x, foot_y, height, timestamp=f"00:00:0{second}"))
    return frames


def _bottle_sequence(
    confidence: float, present_frames: int, class_name: str = "bottle"
) -> tuple[list[dict], list[dict]]:
    frames = []
    detections = []
    for second in range(6):
        foot_x = 300 + second * 20
        frames.append(_frame(second, 9, foot_x, 200, 180, timestamp=f"00:00:0{second}"))
        objects = []
        if second < present_frames:
            objects.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [foot_x - 10, 90, foot_x + 10, 130],
                }
            )
        detections.append({"frame_id": second, "detections": objects})
    return frames, detections


def _concealment_with_bottle() -> tuple[list[dict], list[dict]]:
    frames = []
    detections = []
    for second in range(12):
        height = 200 if second < 6 else 80
        frames.append(_frame(second, 13, 750, 300, height, timestamp=f"00:00:{second:02d}"))
        objects = []
        if second < 6:
            objects.append({"class": "bottle", "confidence": 0.91, "bbox": [735, 175, 765, 215]})
        detections.append({"frame_id": second, "detections": objects})
    return frames, detections


def _frame(
    frame_id: int,
    person_id: int,
    foot_x: int,
    foot_y: int,
    height: int,
    width: int = 70,
    timestamp: str | None = None,
) -> dict:
    bbox = [foot_x - width // 2, foot_y - height, foot_x + width // 2, foot_y]
    center = [foot_x, foot_y - height // 2]
    frame = {
        "frame_id": frame_id,
        "tracks": [{"person_id": person_id, "bbox": bbox, "center": center}],
    }
    if timestamp is not None:
        frame["timestamp"] = timestamp
    return frame


if __name__ == "__main__":
    unittest.main()
