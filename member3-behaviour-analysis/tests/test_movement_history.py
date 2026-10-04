"""Tests for Phase 1 movement history."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from movement_history import MovementHistoryStore, build_movement_histories
from tracking_input import load_tracking_sequence

SAMPLE_PATH = MODULE_ROOT / "samples" / "sample_tracking_frames.json"


class MovementHistoryTests(unittest.TestCase):
    def test_one_person_history_is_stored_correctly(self) -> None:
        frames = [
            {
                "frame_id": 100,
                "timestamp": "10:15:20",
                "tracks": [
                    {"person_id": 12, "center": [450, 320], "zone": "Shelf_A"},
                ],
            },
            {
                "frame_id": 110,
                "timestamp": "10:15:30",
                "tracks": [
                    {"person_id": 12, "center": [455, 325], "zone": "Shelf_A"},
                ],
            },
            {
                "frame_id": 150,
                "timestamp": "10:16:10",
                "tracks": [
                    {"person_id": 12, "center": [520, 330], "zone": "Shelf_B"},
                ],
            },
        ]

        history = build_movement_histories(frames)

        self.assertEqual(
            history,
            [
                {
                    "person_id": 12,
                    "movement_history": [
                        {
                            "frame_id": 100,
                            "timestamp": "10:15:20",
                            "position": [450, 320],
                            "zone": "Shelf_A",
                        },
                        {
                            "frame_id": 110,
                            "timestamp": "10:15:30",
                            "position": [455, 325],
                            "zone": "Shelf_A",
                        },
                        {
                            "frame_id": 150,
                            "timestamp": "10:16:10",
                            "position": [520, 330],
                            "zone": "Shelf_B",
                        },
                    ],
                }
            ],
        )

    def test_multiple_people_histories_are_separated(self) -> None:
        frame = {
            "frame_id": 100,
            "timestamp": "10:15:20",
            "tracks": [
                {"person_id": 12, "center": [450, 320], "zone": "Shelf_A"},
                {"person_id": 7, "position": [100, 80], "zone": "Entrance"},
            ],
        }
        later = {
            "frame_id": 110,
            "timestamp": "10:15:30",
            "tracks": [
                {"person_id": 7, "center": [140, 90], "zone": "Aisle_1"},
            ],
        }
        store = MovementHistoryStore()
        store.ingest_frame(frame)
        store.ingest_frame(later)

        person_12 = store.export_person(12)
        person_7 = store.export_person(7)

        self.assertEqual([record["position"] for record in person_12["movement_history"]], [[450, 320]])
        self.assertEqual(
            [record["position"] for record in person_7["movement_history"]],
            [[100, 80], [140, 90]],
        )
        self.assertEqual(store.person_ids(), [7, 12])
        self.assertEqual(
            [item["person_id"] for item in store.export()],
            [7, 12],
        )

    def test_records_remain_in_chronological_order(self) -> None:
        store = MovementHistoryStore()
        store.ingest_frame(
            {
                "frame_id": 150,
                "timestamp": "10:16:10",
                "tracks": [{"person_id": 12, "center": [520, 330], "zone": "Shelf_B"}],
            }
        )
        store.ingest_frame(
            {
                "frame_id": 100,
                "timestamp": "10:15:20",
                "tracks": [{"person_id": 12, "center": [450, 320], "zone": "Shelf_A"}],
            }
        )
        store.ingest_frame(
            {
                "frame_id": 110,
                "timestamp": "10:15:30",
                "tracks": [{"person_id": 12, "center": [455, 325], "zone": "Shelf_A"}],
            }
        )

        exported = store.export_person(12)
        self.assertEqual(
            [record["frame_id"] for record in exported["movement_history"]],
            [100, 110, 150],
        )
        self.assertEqual(
            [record["timestamp"] for record in exported["movement_history"]],
            ["10:15:20", "10:15:30", "10:16:10"],
        )

    def test_zone_and_shelf_changes_are_preserved(self) -> None:
        frames = [
            {
                "frame_id": 1,
                "timestamp": "10:00:01",
                "tracks": [{"person_id": 3, "center": [10, 10], "zone": "Shelf_A"}],
            },
            {
                "frame_id": 2,
                "timestamp": "10:00:02",
                "tracks": [{"person_id": 3, "center": [12, 11], "zone": "Shelf_A"}],
            },
            {
                "frame_id": 3,
                "timestamp": "10:00:03",
                "tracks": [{"person_id": 3, "center": [40, 12], "shelf": "Shelf_B"}],
            },
            {
                "frame_id": 4,
                "timestamp": "10:00:04",
                "tracks": [{"person_id": 3, "center": [11, 10], "zone": "Shelf_A"}],
            },
        ]

        zones = [
            record["zone"]
            for record in build_movement_histories(frames)[0]["movement_history"]
        ]

        self.assertEqual(zones, ["Shelf_A", "Shelf_A", "Shelf_B", "Shelf_A"])

    def test_sample_file_builds_separate_histories(self) -> None:
        histories = build_movement_histories(load_tracking_sequence(SAMPLE_PATH))
        by_person = {item["person_id"]: item["movement_history"] for item in histories}

        self.assertEqual(set(by_person), {7, 12})
        self.assertEqual(
            [record["zone"] for record in by_person[12]],
            ["Shelf_A", "Shelf_A", "Shelf_B"],
        )
        self.assertEqual(
            [record["zone"] for record in by_person[7]],
            ["Entrance", "Aisle_1"],
        )
        self.assertEqual(by_person[12][0]["position"], [450, 320])
        self.assertEqual(by_person[12][-1]["timestamp"], "10:16:10")

    def test_frames_without_timestamp_or_zone_still_record_position(self) -> None:
        history = build_movement_histories(
            [
                {
                    "frame_id": 100,
                    "tracks": [
                        {
                            "person_id": 12,
                            "bbox": [100, 120, 250, 400],
                            "center": [175, 260],
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            history[0]["movement_history"],
            [
                {
                    "frame_id": 100,
                    "timestamp": None,
                    "position": [175, 260],
                    "zone": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
