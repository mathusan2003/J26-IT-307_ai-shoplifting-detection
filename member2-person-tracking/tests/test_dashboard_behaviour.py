"""Dashboard mapping of Member 2 tracks onto Member 3 live scores.

These tests do not run YOLO. They reuse Member 3's LiveBehaviourSession.
"""
from __future__ import annotations

import unittest

from src.behaviour_bridge import (
    DashboardBehaviour,
    build_behaviour_state,
    build_pipeline_status,
    dashboard_person_payload,
    empty_behaviour_state,
    format_member3_overlay_lines,
    indicator_rows,
    tracks_for_member3,
    webcam_person_detector_kwargs,
)


TRACKING_STATE_KEYS = {
    "camera",
    "source",
    "weights",
    "paused",
    "algorithm",
    "fps",
    "latency_ms",
    "frame_id",
    "timestamp",
    "summary",
    "zones",
    "occupancy_by_zone",
    "tracks",
    "events",
    "selected",
    "status",
    "alerts",
    "scenario",
    "eval",
    "value",
}


def _track(person_id: int, x: float, y: float, status: str = "active", zone: str | None = None) -> dict:
    width, height = 70.0, 180.0
    box = [x - width / 2, y - height, x + width / 2, y]
    row = {
        "person_id": person_id,
        "bbox": box,
        "center": [x, y - height / 2],
        "status": status,
    }
    if zone is not None:
        row["zone"] = zone
    return row


class ContractTrackTests(unittest.TestCase):
    def test_active_tracks_keep_id_bbox_center_and_zone(self) -> None:
        records = [_track(4, 400, 200, zone="Aisle 1"), _track(9, 100, 100, status="temporarily_lost")]
        people = [{"person_id": 4, "zone": "Shelf A"}]
        tracks = tracks_for_member3(records, people)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["person_id"], 4)
        self.assertEqual(tracks[0]["bbox"], records[0]["bbox"])
        self.assertEqual(tracks[0]["center"], records[0]["center"])
        self.assertEqual(tracks[0]["zone"], "Shelf A")

    def test_missing_optional_zone_and_center_still_pass_bbox(self) -> None:
        records = [{"person_id": 6, "bbox": [365, 20, 435, 200], "status": "active"}]
        tracks = tracks_for_member3(records, None)
        self.assertEqual(tracks[0]["person_id"], 6)
        self.assertEqual(tracks[0]["bbox"], [365, 20, 435, 200])
        self.assertNotIn("zone", tracks[0])
        self.assertNotIn("center", tracks[0])

    def test_invalid_rows_are_skipped(self) -> None:
        tracks = tracks_for_member3(
            [None, {"status": "active"}, {"person_id": "x", "bbox": [0, 0, 1, 1], "status": "active"}],
            [{"person_id": "bad"}],
        )
        self.assertEqual(tracks, [])


class ScoreMappingTests(unittest.TestCase):
    def test_payload_maps_member3_fields_and_missing_optionals(self) -> None:
        view = {
            "person_id": 12,
            "behaviours": [{"type": "loitering", "score": 20}],
            "live_score": 20,
            "behaviour_score": 20,
            "combined_score": 0,
            "suspicious_level": "none",
            "contributing": [],
            "not_contributing": [{"type": "loitering", "reason": "isolated"}],
            "pending": ["object_interaction unavailable: person boxes only, not merchandise proof"],
            "history_length": 16,
            "hidden_without_objects": ["possible_concealment"],
            "object_evidence": False,
        }
        payload = dashboard_person_payload(view, zone=None, status="active", duration_s=15.0, window_seconds=8.0)
        self.assertEqual(payload["person_id"], 12)
        self.assertIsNone(payload["zone"])
        self.assertEqual(payload["behaviours"], [{"type": "loitering", "score": 20}])
        self.assertEqual(payload["behaviour_score"], 20)
        self.assertEqual(payload["displayed_score"], 20)
        self.assertEqual(payload["combined_score"], 0)
        self.assertEqual(payload["suspicious_level"], "none")
        self.assertEqual(payload["contributing"], [])
        self.assertEqual(payload["not_contributing"][0]["type"], "loitering")
        self.assertIn("person boxes only", payload["pending"][0])
        self.assertEqual(payload["duration_s"], 15.0)
        self.assertEqual(payload["window_seconds"], 8.0)
        self.assertTrue(payload["not_shoplifting_proof"])
        self.assertTrue(payload["scored"])
        self.assertEqual(payload["score_status"], "scored")

    def test_empty_state_has_no_invented_people_or_scores(self) -> None:
        empty = empty_behaviour_state()
        self.assertTrue(empty["available"])
        self.assertIsNone(empty["error"])
        self.assertEqual(empty["people"], [])
        self.assertFalse(empty["object_evidence"])


class HistoryAndMultiPersonTests(unittest.TestCase):
    def test_history_persists_across_missing_frames(self) -> None:
        engine = DashboardBehaviour()
        self.assertIsNone(engine.error, engine.error)
        first = engine.update(0, [_track(8, 400, 200)], None, 0.0, detections=_people_only())
        self.assertEqual(first["people"][0]["history_length"], 1)
        gap = engine.update(1, [], None, 1.0, detections=_people_only())
        self.assertEqual(engine.session.history_length(8), 1)
        self.assertEqual(gap["people"][0]["person_id"], 8)
        self.assertFalse(gap["people"][0]["visible"])
        resumed = engine.update(3, [_track(8, 400, 200)], None, 3.0, detections=_people_only())
        self.assertEqual(engine.session.history_length(8), 2)
        self.assertEqual(resumed["people"][0]["history_length"], 2)
        self.assertEqual(resumed["people"][0]["behaviours"], [])
        self.assertEqual(resumed["people"][0]["behaviour_score"], 0)

    def test_multiple_people_keep_separate_scores(self) -> None:
        engine = DashboardBehaviour()
        payload = {}
        for second in range(16):
            payload = engine.update(
                second,
                [_track(1, 400, 200, zone="Aisle 1"), _track(2, 240 + second * 80, 200, zone="Aisle 2")],
                [
                    {"person_id": 1, "zone": "Aisle 1", "status": "active"},
                    {"person_id": 2, "zone": "Aisle 2", "status": "active"},
                ],
                float(second),
                detections=_people_only(),
            )
        by_id = {row["person_id"]: row for row in payload["people"]}
        self.assertEqual(set(by_id), {1, 2})
        self.assertEqual(by_id[1]["behaviours"], [{"type": "loitering", "score": 20}])
        self.assertEqual(by_id[1]["behaviour_score"], 20)
        self.assertEqual(by_id[1]["combined_score"], 0)
        self.assertEqual(by_id[1]["suspicious_level"], "none")
        self.assertEqual(by_id[1]["zone"], "Aisle 1")
        self.assertNotIn("loitering", {item["type"] for item in by_id[2]["behaviours"]})
        self.assertEqual(by_id[2]["behaviour_score"], 0)
        self.assertTrue(any("person boxes only" in item for item in by_id[1]["pending"]))
        self.assertNotIn("object_interaction", {item["type"] for item in by_id[1]["behaviours"]})
        self.assertNotIn("possible_concealment", {item["type"] for item in by_id[1]["behaviours"]})
        self.assertEqual(
            next(item["status"] for item in by_id[1]["indicators"] if item["type"] == "loitering"),
            "detected",
        )
        self.assertEqual(
            next(item["score"] for item in by_id[1]["indicators"] if item["type"] == "loitering"),
            20,
        )
        self.assertTrue(payload["events"])
        self.assertEqual(payload["events"][0]["type"], "loitering")
        self.assertEqual(payload["events"][0]["person_id"], 1)
        self.assertIsNotNone(by_id[1]["history"])
        self.assertGreaterEqual(by_id[1]["history"]["length"], 16)
        self.assertIn("process_ms", payload)

    def test_single_frame_does_not_invent_loitering_or_concealment(self) -> None:
        engine = DashboardBehaviour()
        payload = engine.update(0, [_track(4, 400, 200)], None, 0.0, detections=_people_only())
        person = payload["people"][0]
        self.assertEqual(person["behaviours"], [])
        self.assertEqual(person["behaviour_score"], 0)
        self.assertEqual(person["combined_score"], 0)
        self.assertEqual(person["suspicious_level"], "none")
        self.assertTrue(any("need more frames" in item for item in person["pending"]))
        self.assertTrue(any("person boxes only" in item for item in person["pending"]))
        self.assertTrue(person["scored"])
        self.assertEqual(person["score_status"], "pending")
        self.assertEqual(person["behaviour_score"], 0)
        by_type = {item["type"]: item for item in person["indicators"]}
        self.assertEqual(by_type["loitering"]["status"], "waiting")
        self.assertIsNone(by_type["loitering"]["score"])
        self.assertEqual(by_type["object_interaction"]["status"], "unavailable")
        self.assertEqual(by_type["possible_concealment"]["status"], "unavailable")


class ApiPayloadTests(unittest.TestCase):
    def test_state_keeps_tracking_keys_and_adds_behaviour(self) -> None:
        engine = DashboardBehaviour()
        behaviour = engine.update(0, [_track(1, 400, 200)], None, 0.0)
        tracking = {key: ("ok" if key != "behaviour" else None) for key in TRACKING_STATE_KEYS}
        tracking["behaviour"] = behaviour
        self.assertTrue(TRACKING_STATE_KEYS.issubset(tracking))
        self.assertIn("people", tracking["behaviour"])
        self.assertEqual(tracking["behaviour"]["people"][0]["person_id"], 1)

    def test_behaviour_endpoint_shape_from_create_app(self) -> None:
        from serve_dashboard import create_app

        snapshot = {
            "camera": "Camera 01 - Aisle 3",
            "source": None,
            "weights": "models/yolov8n.pt",
            "paused": True,
            "algorithm": "BYTETRACK_ACTIVE",
            "fps": 0.0,
            "latency_ms": 0.0,
            "frame_id": 0,
            "timestamp": "00:00:00.000",
            "summary": {"people_now": 0, "active": 0, "lost": 0},
            "zones": {"Entrance": 0, "Inside Store": 0, "Checkout": 0},
            "occupancy_by_zone": {},
            "tracks": [],
            "events": [],
            "selected": None,
            "status": "paused",
            "alerts": [],
            "scenario": None,
            "eval": {"mota": 0.0},
            "value": {"people_now": 0, "fps": 0.0},
            "behaviour": empty_behaviour_state(),
        }

        class FakeSession:
            weights = "models/yolov8n.pt"

            def snapshot(self):
                return snapshot

            def list_sources(self):
                return []

            def control(self, action, source=None, person_id=None):
                return snapshot

            def mjpeg_bytes(self):
                return b""

            def mjpeg_behaviour_bytes(self):
                return b""

            def mjpeg_objects_bytes(self):
                return b""

        app = create_app(FakeSession())
        routes = {getattr(route, "path", None): route for route in app.routes}
        self.assertIn("/api/state", routes)
        self.assertIn("/api/behaviour", routes)
        self.assertIn("/api/objects", routes)
        self.assertIn("/api/sources", routes)
        self.assertIn("/stream.mjpg", routes)
        self.assertIn("/stream_behaviour.mjpg", routes)
        self.assertIn("/stream_objects.mjpg", routes)
        state = routes["/api/state"].endpoint()
        self.assertTrue(TRACKING_STATE_KEYS.issubset(state))
        self.assertIn("behaviour", state)
        self.assertEqual(state["tracks"], [])
        behaviour = routes["/api/behaviour"].endpoint()
        self.assertEqual(behaviour["people"], [])
        self.assertTrue(behaviour["available"])
        self.assertIn("person box", behaviour["note"].lower())
        objects = routes["/api/objects"].endpoint()
        self.assertEqual(objects["detections"], [])
        self.assertIn("coco", (objects.get("note") or "").lower())
        self.assertIn("not", (objects.get("note") or "").lower())

    def test_build_state_survives_missing_optional_view_fields(self) -> None:
        class StubSession:
            store = None
            last_results = {}

        payload = build_behaviour_state(
            StubSession(),
            {3: {"person_id": 3}},
            {},
            [_track(3, 10, 10)],
            None,
            frame_id=4,
            elapsed_s=0.2,
        )
        person = payload["people"][0]
        self.assertEqual(person["person_id"], 3)
        self.assertEqual(person["behaviours"], [])
        self.assertEqual(person["behaviour_score"], 0)
        self.assertEqual(person["suspicious_level"], "none")
        self.assertIsNone(person["zone"])
        self.assertIsNone(person["duration_s"])
        self.assertIsNone(person["window_seconds"])


class IndicatorStateTests(unittest.TestCase):
    def test_indicator_rows_keep_null_score_when_waiting_or_unavailable(self) -> None:
        rows = {item["type"]: item for item in indicator_rows(
            [],
            [
                "need more frames: one observation is not loitering or a repeated visit",
                "loitering waiting: still 0.0s/15s inside 40px",
                "object_interaction unavailable: person boxes only, not merchandise proof",
                "possible_concealment unavailable: person boxes only, not concealment proof",
                "abnormal_movement waiting: speed 0/250 px/s, reversals 0/4",
            ],
            ["object_interaction", "possible_concealment"],
        )}
        self.assertEqual(rows["loitering"]["status"], "waiting")
        self.assertIsNone(rows["loitering"]["score"])
        self.assertEqual(rows["repeated_shelf_visit"]["status"], "waiting")
        self.assertEqual(rows["abnormal_movement"]["status"], "waiting")
        self.assertEqual(rows["object_interaction"]["status"], "unavailable")
        self.assertIsNone(rows["object_interaction"]["score"])
        self.assertEqual(rows["possible_concealment"]["status"], "unavailable")

    def test_detected_indicator_keeps_real_zero_distinct_from_unavailable(self) -> None:
        rows = {item["type"]: item for item in indicator_rows(
            [{"type": "loitering", "score": 20}],
            ["object_interaction unavailable: person boxes only, not merchandise proof"],
            ["object_interaction", "possible_concealment"],
        )}
        self.assertEqual(rows["loitering"]["status"], "detected")
        self.assertEqual(rows["loitering"]["score"], 20)
        self.assertEqual(rows["abnormal_movement"]["status"], "not_detected")
        self.assertIsNone(rows["abnormal_movement"]["score"])


class PipelineStatusTests(unittest.TestCase):
    def test_weights_file_alone_is_configured_not_running(self) -> None:
        status = build_pipeline_status(
            weights="models/yolov8n.pt",
            weights_exist=True,
            detector_loaded=False,
            paused=True,
            has_source=False,
            playing=False,
            algorithm="BYTETRACK_ACTIVE",
            active_tracks=0,
            frame_id=0,
            timestamp="00:00:00.000",
            pipeline_fps=None,
            detect_fps=None,
            behaviour_available=True,
            behaviour_loaded=True,
            behaviour_error=None,
            behaviour_people=0,
            source=None,
        )
        self.assertEqual(status["detector"]["state"], "configured")
        self.assertFalse(status["detector"]["loaded"])
        self.assertEqual(status["tracker"]["state"], "idle")
        self.assertEqual(status["behaviour"]["state"], "ready")
        self.assertIsNone(status["detect_fps"])
        self.assertIsNone(status["pipeline_fps"])

    def test_loaded_paused_session_is_not_claimed_running(self) -> None:
        status = build_pipeline_status(
            weights="models/yolov8n.pt",
            weights_exist=True,
            detector_loaded=True,
            paused=True,
            has_source=True,
            playing=False,
            algorithm="BYTETRACK_ACTIVE",
            active_tracks=2,
            frame_id=12,
            timestamp="00:00:01.200",
            pipeline_fps=4.8,
            detect_fps=8.1,
            behaviour_available=True,
            behaviour_loaded=True,
            behaviour_error=None,
            behaviour_people=2,
            source="clip.mp4",
        )
        self.assertEqual(status["detector"]["state"], "loaded")
        self.assertEqual(status["tracker"]["state"], "paused")
        self.assertEqual(status["behaviour"]["state"], "ready")
        self.assertEqual(status["tracker"]["active_tracks"], 2)
        self.assertEqual(status["detect_fps"], 8.1)
        self.assertEqual(status["pipeline_fps"], 4.8)

    def test_playing_session_reports_running_components(self) -> None:
        status = build_pipeline_status(
            weights="models/yolov8n.pt",
            weights_exist=True,
            detector_loaded=True,
            paused=False,
            has_source=True,
            playing=True,
            algorithm="BYTETRACK_ACTIVE",
            active_tracks=3,
            frame_id=40,
            timestamp="00:00:04.000",
            pipeline_fps=5.0,
            detect_fps=9.2,
            behaviour_available=True,
            behaviour_loaded=True,
            behaviour_error=None,
            behaviour_people=3,
            source="clip.mp4",
        )
        self.assertEqual(status["detector"]["state"], "running")
        self.assertEqual(status["tracker"]["state"], "running")
        self.assertEqual(status["behaviour"]["state"], "running")

    def test_missing_optional_behaviour_fields_stay_unscored_in_join(self) -> None:
        payload = build_behaviour_state(
            type("S", (), {"store": None, "last_results": {}})(),
            {1: {"person_id": 1, "pending": ["need more frames: one observation is not loitering or a repeated visit"]}},
            {},
            [_track(1, 10, 10), _track(2, 80, 10)],
            None,
            frame_id=1,
            elapsed_s=0.0,
        )
        by_id = {row["person_id"]: row for row in payload["people"]}
        self.assertTrue(by_id[1]["scored"])
        self.assertEqual(by_id[1]["score_status"], "pending")
        self.assertNotIn(2, by_id)


class Member2ConfigIsolationTests(unittest.TestCase):
    def test_member3_import_does_not_replace_member2_config(self) -> None:
        from src.behaviour_bridge import load_live_behaviour_session

        load_live_behaviour_session()
        import config

        self.assertTrue(hasattr(config, "TrackingConfig"))


class Member3LiveOverlayTests(unittest.TestCase):
    def test_overlay_lines_reuse_member3_demo_format(self) -> None:
        lines = format_member3_overlay_lines(
            {
                "behaviours": [{"type": "loitering", "score": 20}],
                "displayed_score": 20,
                "combined_score": 20,
                "suspicious_level": "none",
                "contributing": ["loitering"],
                "pending": [],
            }
        )
        self.assertTrue(any("loitering:20" in line for line in lines))
        self.assertTrue(any("score=20" in line and "combined=20" in line for line in lines))
        self.assertTrue(any("contrib=loitering" in line for line in lines))

    def test_webcam_detector_kwargs_match_live_demo(self) -> None:
        kwargs = webcam_person_detector_kwargs()
        self.assertEqual(kwargs["confidence"], 0.45)
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertEqual(kwargs["max_det"], 20)
        self.assertEqual(kwargs["nms_iou"], 0.50)

    def test_webcam_detect_keeps_bottom_clipped_closeup(self) -> None:
        import numpy as np
        from src.behaviour_bridge import detect_webcam_people

        class _Tensor:
            def __init__(self, value):
                self._value = value

            def __getitem__(self, _idx):
                return self._value

            def tolist(self):
                return list(self._value)

        class _Box:
            cls = [0]
            conf = [0.75]
            xyxy = [_Tensor([76.0, 163.0, 632.0, 480.0])]

        class DummyDetector:
            confidence = 0.45
            nms_iou = 0.50
            imgsz = 640
            max_det = 20

            def model(self, _frame, **_kwargs):
                return [type("Result", (), {"boxes": [_Box()]})()]

            def detect(self, _frame):
                return []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kept = detect_webcam_people(DummyDetector(), frame)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["class"], "person")


class AlertCropTests(unittest.TestCase):
    def test_alert_crop_uses_the_tracked_person_box(self) -> None:
        import numpy as np
        from src.live_session import LiveSession

        session = LiveSession.__new__(LiveSession)
        session.cv2 = __import__("cv2")
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[20:180, 40:110] = (0, 0, 255)
        behaviour = {
            "events": [
                {"person_id": 7, "type": "item_pickup_suspected", "score": 18, "elapsed_s": 1.0, "level": "low"},
                {"person_id": 7, "type": "abnormal_movement", "score": 28, "elapsed_s": 1.0, "level": "none"},
            ]
        }
        session._stamp_alert_crops(
            frame,
            [{"person_id": 7, "bbox": [40, 20, 110, 180], "status": "active"}],
            behaviour,
        )
        self.assertEqual(len(behaviour["suspected_alerts"]), 1)
        alert = behaviour["suspected_alerts"][0]
        self.assertEqual(alert["person_id"], 7)
        self.assertEqual(alert["type"], "item_pickup_suspected")
        self.assertTrue(str(alert["crop_jpeg"]).startswith("data:image/jpeg;base64,"))


class WebcamSourceTests(unittest.TestCase):
    def test_dashboard_sources_always_include_webcam(self) -> None:
        from pathlib import Path

        from src.live_session import WEBCAM_SOURCE_ID, dashboard_sources, is_webcam_source, source_label

        rows = dashboard_sources([])
        self.assertEqual(rows[0]["id"], WEBCAM_SOURCE_ID)
        self.assertEqual(rows[0]["group"], "live")
        self.assertEqual(len(rows), 1)
        self.assertTrue(is_webcam_source("webcam"))
        self.assertTrue(is_webcam_source("Live webcam"))
        self.assertFalse(is_webcam_source("clip.mp4"))
        self.assertEqual(source_label("webcam"), WEBCAM_SOURCE_ID)
        self.assertEqual(source_label(Path("aisle.mp4")), "aisle.mp4")
        self.assertTrue(is_webcam_source(WEBCAM_SOURCE_ID))
        self.assertFalse(is_webcam_source(None))


def _people_only() -> list[dict]:
    return [{"class": "person", "confidence": 0.9, "bbox": [365, 20, 435, 200]}]


def _member1_bottle(x: float, y: float = 90.0) -> dict:
    return {
        "detection_id": 1,
        "class_id": 39,
        "class_name": "bottle",
        "confidence": 0.88,
        "bbox": {"x1": x, "y1": y, "x2": x + 40.0, "y2": y + 50.0},
    }


class ItemAlertDashboardTests(unittest.TestCase):
    def test_member1_objects_and_member2_tracks_create_one_pickup_alert(self) -> None:
        engine = DashboardBehaviour()
        payload = {}
        for frame in range(8):
            x = 400.0 + frame * 40.0
            payload = engine.update(
                frame,
                [_track(7, x, 200)],
                [{"person_id": 7, "zone": "Aisle 1", "status": "active"}],
                float(frame),
                detections=[_member1_bottle(x - 20)],
                frame_size=(640, 480),
            )
        person = payload["people"][0]
        types = {item["type"] for item in person["item_alerts"]}
        self.assertIn("item_pickup_suspected", types)
        pickup = next(item for item in person["item_alerts"] if item["type"] == "item_pickup_suspected")
        self.assertEqual(pickup["person_id"], 7)
        self.assertEqual(pickup["object_class"], "bottle")
        self.assertEqual(pickup["level"], "low")
        self.assertTrue(pickup["evidence"])
        event_pickups = [item for item in payload["events"] if item["type"] == "item_pickup_suspected"]
        self.assertEqual(len(event_pickups), 1)
        self.assertEqual(event_pickups[0]["object_class"], "bottle")
        self.assertIn("item_pickup_suspected", {item["type"] for item in payload["item_alerts"]})
        later = engine.update(
            9,
            [_track(7, 720, 200)],
            [{"person_id": 7, "zone": "Aisle 1", "status": "active"}],
            9.0,
            detections=[_member1_bottle(700)],
            frame_size=(640, 480),
        )
        self.assertEqual(len([item for item in later["events"] if item["type"] == "item_pickup_suspected"]), 1)
        suspected = later["suspected_alerts"]
        self.assertTrue(any(item["type"] == "item_pickup_suspected" and item["person_id"] == 7 for item in suspected))

    def test_missing_object_detector_keeps_item_indicators_unavailable(self) -> None:
        engine = DashboardBehaviour()
        payload = engine.update(0, [_track(4, 400, 200)], None, 0.0, detections=None)
        person = payload["people"][0]
        by_type = {item["type"]: item for item in person["indicators"]}
        self.assertEqual(by_type["item_pickup_suspected"]["status"], "unavailable")
        self.assertEqual(by_type["possible_item_concealment"]["status"], "unavailable")
        self.assertEqual(person["item_alerts"], [])
        self.assertFalse(payload["object_evidence"])
        self.assertIsNone(payload["object_debug"].get("non_person"))
        self.assertEqual(payload["object_input"], "unavailable")
        self.assertEqual(payload["object_detections"], [])

    def test_populated_objects_reach_api_without_requiring_an_alert(self) -> None:
        engine = DashboardBehaviour()
        payload = engine.update(
            0,
            [_track(4, 400, 200)],
            None,
            0.0,
            detections=[_member1_bottle(380)],
            frame_size=(640, 480),
        )
        self.assertEqual(payload["object_input"], "populated")
        self.assertTrue(payload["object_evidence"])
        self.assertEqual(payload["object_detections"][0]["class"], "bottle")
        self.assertEqual(payload["object_detections"][0]["bbox"], [380.0, 90.0, 420.0, 140.0])
        self.assertEqual(payload["item_alerts"], [])
        self.assertEqual(payload["object_debug"]["raw"], 1)
        self.assertEqual(payload["object_debug"]["non_person"], 1)

    def test_empty_detections_are_empty_not_unavailable(self) -> None:
        engine = DashboardBehaviour()
        payload = engine.update(0, [_track(4, 400, 200)], None, 0.0, detections=[])
        self.assertEqual(payload["object_input"], "empty")
        self.assertEqual(payload["object_detections"], [])
        self.assertFalse(payload["object_evidence"])
        self.assertEqual(payload["object_debug"]["raw"], 0)

    def test_objects_without_people_still_appear_on_the_payload(self) -> None:
        engine = DashboardBehaviour()
        payload = engine.update(0, [], None, 0.0, detections=[_member1_bottle(200)])
        self.assertEqual(payload["people"], [])
        self.assertEqual(payload["object_input"], "populated")
        self.assertEqual([item["class"] for item in payload["object_detections"]], ["bottle"])
        self.assertTrue(payload["object_evidence"])


if __name__ == "__main__":
    unittest.main()
