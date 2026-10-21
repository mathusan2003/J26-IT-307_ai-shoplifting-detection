"""Pickup and concealment need temporal evidence. Proximity and lost boxes are not enough."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from item_association import ItemAssociationSession, adapt_member1_detections, member1_to_member3_detections
from live_behaviour_view import LiveBehaviourSession


SHELF_ZONES = {"Shelf A": [[0.0, 0.0], [260.0, 0.0], [260.0, 400.0], [0.0, 400.0]]}
FRAME_SIZE = (640, 480)


def _person(person_id: int, x: float, y: float = 200.0, zone: str | None = None) -> dict:
    width, height = 70.0, 180.0
    row = {
        "person_id": person_id,
        "bbox": [x - width / 2, y - height, x + width / 2, y],
        "center": [x, y - height / 2],
        "status": "active",
    }
    if zone is not None:
        row["zone"] = zone
    return row


def _member1(class_name: str, x: float, y: float, width: float = 40.0, height: float = 50.0, detection_id: int = 1) -> dict:
    return {
        "detection_id": detection_id,
        "class_id": 39,
        "class_name": class_name,
        "confidence": 0.86,
        "bbox": {"x1": x, "y1": y, "x2": x + width, "y2": y + height},
    }


def _types(alerts: dict[int, list[dict]]) -> set[str]:
    return {item["type"] for rows in alerts.values() for item in rows}


class FormatAdapterTests(unittest.TestCase):
    def test_member1_dict_bbox_maps_to_member3_rows_and_drops_people(self) -> None:
        rows = member1_to_member3_detections(
            [
                _member1("person", 10, 10),
                _member1("bottle", 100, 80),
                {"class": "handbag", "confidence": 0.7, "bbox": [1, 2, 20, 30]},
            ]
        )
        self.assertEqual([item["class"] for item in rows], ["bottle", "handbag"])
        self.assertEqual(rows[0]["bbox"], [100.0, 80.0, 140.0, 130.0])

    def test_unavailable_none_is_not_an_empty_success(self) -> None:
        rows, debug = adapt_member1_detections(None)
        self.assertIsNone(rows)
        self.assertEqual(debug["input"], "unavailable")

    def test_empty_success_stays_empty_list(self) -> None:
        rows, debug = adapt_member1_detections([])
        self.assertEqual(rows, [])
        self.assertEqual(debug["input"], "empty")
        self.assertEqual(debug["raw"], 0)

    def test_numpy_like_and_xmin_boxes_are_not_dropped(self) -> None:
        class _Array:
            def tolist(self):
                return [10.0, 20.0, 40.0, 60.0]

        rows, debug = adapt_member1_detections(
            [
                {"class_name": "cup", "confidence": 0.4, "bbox": _Array()},
                {"class": "book", "confidence": 0.5, "bbox": {"xmin": 1, "ymin": 2, "xmax": 21, "ymax": 32}},
            ]
        )
        self.assertEqual(debug["input"], "populated")
        self.assertEqual([item["class"] for item in rows], ["cup", "book"])
        self.assertEqual(rows[0]["bbox"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(rows[1]["bbox"], [1.0, 2.0, 21.0, 32.0])


class PickupTests(unittest.TestCase):
    def test_associated_co_motion_is_pickup_once(self) -> None:
        session = ItemAssociationSession()
        last = {}
        for frame in range(8):
            x = 400.0 + frame * 40.0
            last = session.update(
                frame,
                float(frame),
                [_person(7, x)],
                [_member1("bottle", x - 20, 90)],
                frame_size=FRAME_SIZE,
            )
            if frame < 2:
                self.assertNotIn("item_pickup_suspected", _types(last))
        self.assertIn("item_pickup_suspected", {item["type"] for item in session.alerts.get(7, [])})
        pickup = [item for item in session.alerts[7] if item["type"] == "item_pickup_suspected"]
        self.assertEqual(len(pickup), 1)
        self.assertEqual(pickup[0]["person_id"], 7)
        self.assertEqual(pickup[0]["object_class"], "bottle")
        self.assertEqual(pickup[0]["level"], "low")
        self.assertEqual(pickup[0]["score"], 18)
        self.assertIn("Proximity alone is not a pickup", pickup[0]["evidence"])
        later = session.update(
            9,
            9.0,
            [_person(7, 720)],
            [_member1("bottle", 700, 90)],
            frame_size=FRAME_SIZE,
        )
        self.assertEqual(_types(later), set())

    def test_proximity_without_object_motion_is_not_pickup(self) -> None:
        session = ItemAssociationSession()
        for frame in range(6):
            session.update(
                frame,
                float(frame),
                [_person(3, 400)],
                [_member1("bottle", 380, 90)],
                frame_size=FRAME_SIZE,
            )
        self.assertEqual(session.alerts.get(3, []), [])

    def test_person_walks_past_static_object_is_not_pickup(self) -> None:
        session = ItemAssociationSession()
        for frame in range(6):
            session.update(
                frame,
                float(frame),
                [_person(4, 360.0 + frame * 50.0)],
                [_member1("bottle", 380, 90)],
                frame_size=FRAME_SIZE,
            )
        self.assertNotIn("item_pickup_suspected", {item["type"] for item in session.alerts.get(4, [])})


class ConcealmentTests(unittest.TestCase):
    def test_inward_then_in_frame_miss_is_possible_concealment_once(self) -> None:
        session = ItemAssociationSession()
        starts = [460.0, 420.0, 380.0]
        for frame, x in enumerate(starts):
            session.update(frame, float(frame), [_person(2, 400)], [_member1("bottle", x, 90)], frame_size=FRAME_SIZE)
        for frame in range(3, 7):
            last = session.update(frame, float(frame), [_person(2, 400)], [], frame_size=FRAME_SIZE)
        hidden = [item for item in session.alerts.get(2, []) if item["type"] == "possible_item_concealment"]
        self.assertEqual(len(hidden), 1)
        self.assertEqual(hidden[0]["level"], "medium")
        self.assertEqual(hidden[0]["score"], 24)
        self.assertIn("not proof", hidden[0]["evidence"])
        self.assertEqual(_types(last), set())

    def test_lost_boxes_without_inward_motion_are_not_concealment(self) -> None:
        session = ItemAssociationSession()
        for frame in range(3):
            session.update(frame, float(frame), [_person(8, 400)], [_member1("bottle", 380, 90)], frame_size=FRAME_SIZE)
        for frame in range(3, 8):
            session.update(frame, float(frame), [_person(8, 400)], [], frame_size=FRAME_SIZE)
        self.assertEqual(session.alerts.get(8, []), [])

    def test_object_leaving_frame_edge_is_not_concealment(self) -> None:
        session = ItemAssociationSession()
        xs = [500.0, 560.0, 590.0]
        for frame, x in enumerate(xs):
            session.update(frame, float(frame), [_person(5, 480)], [_member1("bottle", x, 90)], frame_size=FRAME_SIZE)
        for frame in range(3, 7):
            session.update(frame, float(frame), [_person(5, 480)], [], frame_size=FRAME_SIZE)
        self.assertNotIn(
            "possible_item_concealment",
            {item["type"] for item in session.alerts.get(5, [])},
        )

    def test_missing_person_track_does_not_claim_concealment(self) -> None:
        session = ItemAssociationSession()
        starts = [460.0, 420.0, 380.0]
        for frame, x in enumerate(starts):
            session.update(frame, float(frame), [_person(9, 400)], [_member1("bottle", x, 90)], frame_size=FRAME_SIZE)
        for frame in range(3, 7):
            session.update(frame, float(frame), [], [], frame_size=FRAME_SIZE)
        self.assertEqual(session.alerts.get(9, []), [])

    def test_unavailable_detector_does_not_count_as_vanishing(self) -> None:
        session = ItemAssociationSession()
        starts = [460.0, 420.0, 380.0]
        for frame, x in enumerate(starts):
            session.update(frame, float(frame), [_person(1, 400)], [_member1("bottle", x, 90)], frame_size=FRAME_SIZE)
        for frame in range(3, 8):
            session.update(frame, float(frame), [_person(1, 400)], None, frame_size=FRAME_SIZE)
        self.assertEqual(session.alerts.get(1, []), [])


class ShelfAndNormalTests(unittest.TestCase):
    def test_shelf_removal_needs_pickup_and_leaving_the_shelf(self) -> None:
        session = ItemAssociationSession()
        for frame in range(8):
            x = 120.0 + frame * 50.0
            zone = "Shelf A" if x < 240 else "Aisle 1"
            session.update(
                frame,
                float(frame),
                [_person(6, x, zone=zone)],
                [_member1("bottle", x - 20, 90)],
                zones=SHELF_ZONES,
                frame_size=FRAME_SIZE,
            )
        types = {item["type"] for item in session.alerts.get(6, [])}
        self.assertIn("item_pickup_suspected", types)
        self.assertIn("item_removed_from_shelf", types)
        removed = [item for item in session.alerts[6] if item["type"] == "item_removed_from_shelf"]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["level"], "low")

    def test_repeated_shelf_interaction_needs_two_object_visits(self) -> None:
        session = ItemAssociationSession()
        path = [120.0, 120.0, 400.0, 400.0, 120.0, 120.0]
        for frame, x in enumerate(path):
            zone = "Shelf A" if x < 240 else "Aisle 1"
            session.update(
                frame,
                float(frame),
                [_person(11, x, zone=zone)],
                [_member1("handbag", x - 15, 90)],
                zones=SHELF_ZONES,
                frame_size=FRAME_SIZE,
            )
        types = {item["type"] for item in session.alerts.get(11, [])}
        self.assertIn("repeated_shelf_interaction", types)
        repeats = [item for item in session.alerts[11] if item["type"] == "repeated_shelf_interaction"]
        self.assertEqual(len(repeats), 1)

    def test_independent_object_motion_away_from_people_is_normal(self) -> None:
        session = ItemAssociationSession()
        for frame in range(6):
            session.update(
                frame,
                float(frame),
                [_person(12, 80)],
                [_member1("bottle", 400.0 + frame * 40.0, 90)],
                frame_size=FRAME_SIZE,
            )
        self.assertEqual(session.alerts, {})


class LiveSessionIntegrationTests(unittest.TestCase):
    def test_live_session_keeps_member4_contract_free_of_item_types(self) -> None:
        session = LiveBehaviourSession(zones={"Aisle 1": [[0, 0], [800, 0], [800, 400], [0, 400]]})
        views = {}
        for frame in range(6):
            x = 400.0 + frame * 40.0
            views = session.update(
                frame,
                [_person(7, x)],
                elapsed_s=float(frame),
                detections=[_member1("bottle", x - 20, 90)],
                frame_size=FRAME_SIZE,
            )
        self.assertIn("item_pickup_suspected", {item["type"] for item in views[7]["behaviours"]})
        self.assertTrue(views[7]["item_alerts"])
        contract = session.last_results[7]
        self.assertNotIn("item_pickup_suspected", {item["type"] for item in contract["behaviours"]})
        self.assertIn("assessment", contract)

    def test_person_only_live_session_does_not_invent_item_alerts(self) -> None:
        session = LiveBehaviourSession()
        views = session.update(
            0,
            [_person(4, 400)],
            elapsed_s=0.0,
            detections=[{"class": "person", "confidence": 0.9, "bbox": [365, 20, 435, 200]}],
        )
        self.assertEqual(views[4]["item_alerts"], [])
        self.assertTrue(any("item_pickup_suspected unavailable" in item for item in views[4]["pending"]))


if __name__ == "__main__":
    unittest.main()
