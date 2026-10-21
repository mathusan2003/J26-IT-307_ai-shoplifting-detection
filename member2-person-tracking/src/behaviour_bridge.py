"""Feed Member 2 live tracks into Member 3's existing behaviour session.

This module does not detect people, does not track IDs, and does not copy
Member 3 scoring rules. It loads ``LiveBehaviourSession`` from Member 3 and
maps the live views onto the dashboard JSON contract.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

MEMBER3_SRC = Path(__file__).resolve().parents[2] / "member3-behaviour-analysis" / "src"
_MEMBER3_TOP_LEVEL = (
    "assessment",
    "behaviour",
    "config",
    "detectors",
    "features",
    "geometry",
    "item_association",
    "live_behaviour_view",
    "member2_adapter",
    "movement_history",
    "tracking_input",
)

BEHAVIOUR_NOTE = (
    "Member 3 scores from Member 2 tracks and Member 1 COCO boxes. "
    "A person box is not shoplifting. Pickup and concealment stay suspected "
    "indicators, not proof of theft."
)
INDICATOR_TYPES = (
    "loitering",
    "repeated_shelf_visit",
    "abnormal_movement",
    "object_interaction",
    "possible_concealment",
    "item_pickup_suspected",
    "possible_item_concealment",
    "item_removed_from_shelf",
    "repeated_shelf_interaction",
)
OBJECT_EVIDENCE_TYPES = (
    "object_interaction",
    "possible_concealment",
    "item_pickup_suspected",
    "possible_item_concealment",
    "item_removed_from_shelf",
    "repeated_shelf_interaction",
)
ITEM_ALERT_TYPES = (
    "item_pickup_suspected",
    "possible_item_concealment",
    "item_removed_from_shelf",
    "repeated_shelf_interaction",
)
SHOPLIFTING_ALERT_TYPES = (
    "item_pickup_suspected",
    "possible_item_concealment",
    "item_removed_from_shelf",
    "possible_concealment",
)

_LIVE_SESSION_CLS: Any | None = None


def detect_webcam_people(detector: Any, frame: Any) -> list[dict[str, Any]]:
    """Reuse Member 3's webcam person filter on the already-loaded Member 2 model."""
    live = _member3_live_view()
    if live is None:
        return list(detector.detect(frame) or [])
    return list(live.detect_live_people(detector, frame) or [])


def webcam_person_detector_kwargs() -> dict[str, Any]:
    """Same YOLO settings as ``member3_live_demo.py`` (COCO-scale webcam)."""
    live = _member3_live_view()
    if live is None:
        return {"confidence": 0.45, "imgsz": 640, "max_det": 20, "nms_iou": 0.50}
    return {
        "confidence": float(live.LIVE_DETECT_CONFIDENCE),
        "imgsz": int(live.LIVE_IMGSZ),
        "max_det": int(live.LIVE_MAX_DET),
        "nms_iou": float(live.LIVE_NMS_IOU),
    }


def format_member3_overlay_lines(person: dict[str, Any] | None, zone: str | None = None) -> list[str]:
    """Reuse the OpenCV demo overlay text for one dashboard person."""
    view = {
        "behaviours": (person or {}).get("behaviours") or [],
        "live_score": (person or {}).get("displayed_score", (person or {}).get("live_score", 0)),
        "combined_score": (person or {}).get("combined_score", 0),
        "suspicious_level": (person or {}).get("suspicious_level") or "none",
        "contributing": (person or {}).get("contributing") or [],
        "pending": (person or {}).get("pending") or [],
    }
    live = _member3_live_view()
    if live is None:
        behaviours = view["behaviours"]
        first = (
            " ".join(f"{item['type']}:{item['score']}" for item in behaviours)
            if behaviours
            else "no indicators yet"
        )
        return [
            first,
            f"score={view['live_score']} combined={view['combined_score']} level={view['suspicious_level']}",
        ]
    return list(live.format_overlay_lines(view, zone=zone))


def _member3_live_view() -> Any | None:
    try:
        load_live_behaviour_session()
    except Exception:
        return None
    return sys.modules.get("_member3.live_behaviour_view")


def load_live_behaviour_session() -> Any:
    """Import Member 3's LiveBehaviourSession without shadowing Member 2's config."""
    global _LIVE_SESSION_CLS
    if _LIVE_SESSION_CLS is not None:
        return _LIVE_SESSION_CLS
    if not MEMBER3_SRC.is_dir():
        raise FileNotFoundError(f"Member 3 source folder not found: {MEMBER3_SRC}")

    src = str(MEMBER3_SRC)
    saved = {name: sys.modules.pop(name) for name in _MEMBER3_TOP_LEVEL if name in sys.modules}
    inserted = src not in sys.path
    if inserted:
        sys.path.insert(0, src)
    try:
        from live_behaviour_view import LiveBehaviourSession  # type: ignore

        for name in _MEMBER3_TOP_LEVEL:
            module = sys.modules.get(name)
            if module is not None:
                sys.modules[f"_member3.{name}"] = module
        _LIVE_SESSION_CLS = LiveBehaviourSession
        return LiveBehaviourSession
    finally:
        for name in _MEMBER3_TOP_LEVEL:
            previous = saved.get(name)
            if previous is not None:
                sys.modules[name] = previous
            elif name in sys.modules:
                del sys.modules[name]
        if inserted and src in sys.path:
            try:
                sys.path.remove(src)
            except ValueError:
                pass


def empty_behaviour_state(
    *,
    frame_id: int = 0,
    elapsed_s: float = 0.0,
    error: str | None = None,
) -> dict[str, Any]:
    """Dashboard behaviour payload when nobody is scored yet, or analysis failed."""
    return {
        "available": error is None,
        "error": error,
        "frame_id": int(frame_id),
        "elapsed_s": round(float(elapsed_s), 3),
        "object_evidence": False,
        "note": BEHAVIOUR_NOTE,
        "people": [],
        "events": [],
        "has_tracking_input": False,
        "process_ms": None,
        "people_analysed": 0,
        "assessments_complete": 0,
        "item_alerts": [],
        "object_detections": [],
        "object_input": "unavailable" if error else "empty",
        "object_debug": {},
        "suspected_alerts": [],
    }


def tracks_for_member3(
    records: Sequence[dict[str, Any]] | None,
    people_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Keep active Member 2 tracks in the contract Member 3 already analyses.

    Temporarily lost people are omitted so their last known box is not treated
    as a new observation. Their history stays in the Member 3 store.
    """
    zones: dict[int, str] = {}
    for row in people_rows or []:
        if not isinstance(row, dict):
            continue
        try:
            person_id = int(row["person_id"])
        except (KeyError, TypeError, ValueError):
            continue
        zone = row.get("zone")
        if isinstance(zone, str) and zone.strip():
            zones[person_id] = zone.strip()

    tracks: list[dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if record.get("status") not in (None, "active"):
            continue
        try:
            person_id = int(record.get("person_id"))
        except (TypeError, ValueError):
            continue
        item: dict[str, Any] = {"person_id": person_id}
        if record.get("bbox") is not None:
            item["bbox"] = record["bbox"]
        if record.get("center") is not None:
            item["center"] = record["center"]
        zone = zones.get(person_id) or record.get("zone")
        if isinstance(zone, str) and zone.strip():
            item["zone"] = zone.strip()
        if "bbox" not in item and "center" not in item:
            continue
        tracks.append(item)
    return tracks


def dashboard_person_payload(
    view: dict[str, Any],
    *,
    zone: str | None = None,
    status: str | None = None,
    duration_s: float | None = None,
    window_seconds: float | None = None,
    visible: bool = True,
) -> dict[str, Any]:
    """Map one Member 3 live view onto dashboard fields. Missing values stay empty."""
    behaviours = [
        {"type": str(item.get("type")), "score": int(item.get("score") or 0)}
        for item in (view.get("behaviours") or [])
        if isinstance(item, dict) and item.get("type")
    ]
    contributing = [str(name) for name in (view.get("contributing") or []) if name]
    not_contributing = _normalise_not_contributing(view.get("not_contributing"))
    pending = [str(item) for item in (view.get("pending") or []) if item]
    hidden = [str(item) for item in (view.get("hidden_without_objects") or []) if item]
    return {
        "person_id": int(view.get("person_id")),
        "zone": zone,
        "status": status,
        "visible": bool(visible),
        "behaviours": behaviours,
        "behaviour_score": int(view.get("behaviour_score") or 0),
        "displayed_score": int(view.get("live_score") or 0),
        "combined_score": int(view.get("combined_score") or 0),
        "suspicious_level": view.get("suspicious_level") or "none",
        "contributing": contributing,
        "not_contributing": not_contributing,
        "pending": pending,
        "history_length": int(view.get("history_length") or 0),
        "duration_s": None if duration_s is None else round(float(duration_s), 3),
        "window_seconds": None if window_seconds is None else float(window_seconds),
        "hidden_without_objects": hidden,
        "object_evidence": bool(view.get("object_evidence")),
        "not_shoplifting_proof": True,
        "scored": True,
        "score_status": "pending" if _is_pending(pending, behaviours) else "scored",
        "indicators": indicator_rows(behaviours, pending, hidden),
        "history": None,
        "item_alerts": list(view.get("item_alerts") or []),
        "item_alert_level": view.get("item_alert_level") or "none",
    }


def build_behaviour_state(
    session: Any,
    views: dict[int, dict[str, Any]] | None,
    cache: dict[int, dict[str, Any]],
    records: Sequence[dict[str, Any]] | None,
    people_rows: Sequence[dict[str, Any]] | None,
    *,
    frame_id: int,
    elapsed_s: float,
    error: str | None = None,
    events: Sequence[dict[str, Any]] | None = None,
    process_ms: float | None = None,
    has_tracking_input: bool | None = None,
) -> dict[str, Any]:
    """Build the additive ``behaviour`` object for ``/api/state``."""
    if error:
        return empty_behaviour_state(frame_id=frame_id, elapsed_s=elapsed_s, error=error)

    people_by_id = _index_people(people_rows)
    records_by_id = _index_records(records)
    payload_people: list[dict[str, Any]] = []
    seen: set[int] = set()

    ordered_ids: list[int] = []
    for record in records or []:
        if not isinstance(record, dict) or record.get("status") == "terminated":
            continue
        try:
            ordered_ids.append(int(record["person_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    for person_id in list(views or {}) + list(cache):
        if person_id not in ordered_ids:
            ordered_ids.append(int(person_id))

    for person_id in ordered_ids:
        if person_id in seen:
            continue
        seen.add(person_id)
        view = (views or {}).get(person_id) or cache.get(person_id)
        if not isinstance(view, dict):
            continue
        person = people_by_id.get(person_id) or {}
        record = records_by_id.get(person_id) or {}
        zone = person.get("zone") or record.get("zone") or view.get("zone")
        status = record.get("status") or person.get("status")
        visible = status == "active" and person_id in (views or {})
        person_payload = dashboard_person_payload(
            view,
            zone=zone if isinstance(zone, str) and zone.strip() else None,
            status=status if isinstance(status, str) else None,
            duration_s=_duration_s(session, person_id),
            window_seconds=_window_seconds(session, person_id),
            visible=visible,
        )
        person_payload["history"] = movement_history_payload(session, person_id, person_payload.get("zone"))
        payload_people.append(person_payload)

    object_rows = getattr(session, "last_object_rows", None)
    object_debug = dict(getattr(session, "last_object_debug", None) or {})
    object_input = str(getattr(session, "last_object_input", None) or object_debug.get("input") or "empty")
    object_detections = list(object_rows or [])
    object_evidence = bool(object_detections) or any(bool(item.get("object_evidence")) for item in payload_people)
    complete = sum(1 for item in payload_people if item.get("score_status") == "scored")
    return {
        "available": True,
        "error": None,
        "frame_id": int(frame_id),
        "elapsed_s": round(float(elapsed_s), 3),
        "object_evidence": object_evidence,
        "note": BEHAVIOUR_NOTE,
        "people": payload_people,
        "events": list(events or []),
        "has_tracking_input": (
            bool(has_tracking_input)
            if has_tracking_input is not None
            else any(
                isinstance(row, dict) and row.get("status") in (None, "active") for row in (records or [])
            )
        ),
        "process_ms": None if process_ms is None else round(float(process_ms), 1),
        "people_analysed": len(payload_people),
        "assessments_complete": complete,
        "item_alerts": [
            alert
            for person in payload_people
            for alert in (person.get("item_alerts") or [])
        ],
        "object_detections": object_detections,
        "object_input": object_input,
        "object_debug": object_debug,
    }


def build_pipeline_status(
    *,
    weights: str | None,
    weights_exist: bool,
    detector_loaded: bool,
    paused: bool,
    has_source: bool,
    playing: bool,
    algorithm: str,
    active_tracks: int,
    frame_id: int,
    timestamp: str,
    pipeline_fps: float | None,
    detect_fps: float | None,
    behaviour_available: bool,
    behaviour_loaded: bool,
    behaviour_error: str | None,
    behaviour_people: int,
    source: str | None,
) -> dict[str, Any]:
    """Honest runtime status. A weights file alone is not a running model."""
    if playing and detector_loaded:
        detector_state = "running"
    elif detector_loaded:
        detector_state = "loaded"
    elif weights_exist:
        detector_state = "configured"
    else:
        detector_state = "missing_weights"

    if playing:
        tracker_state = "running"
    elif has_source:
        tracker_state = "paused"
    else:
        tracker_state = "idle"

    if behaviour_error and not behaviour_loaded:
        behaviour_state = "error"
    elif playing and behaviour_loaded:
        behaviour_state = "running"
    elif behaviour_loaded:
        behaviour_state = "ready"
    else:
        behaviour_state = "unavailable"

    return {
        "detector": {
            "name": "YOLOv8 person detector",
            "weights": weights,
            "weights_exist": bool(weights_exist),
            "loaded": bool(detector_loaded),
            "state": detector_state,
        },
        "tracker": {
            "name": "ByteTrack",
            "algorithm": algorithm,
            "state": tracker_state,
            "active_tracks": int(active_tracks),
        },
        "behaviour": {
            "name": "Member 3 LiveBehaviourSession",
            "loaded": bool(behaviour_loaded),
            "available": bool(behaviour_available),
            "state": behaviour_state,
            "error": behaviour_error,
            "people_scored": int(behaviour_people),
        },
        "frame_id": int(frame_id),
        "timestamp": timestamp,
        "pipeline_fps": pipeline_fps,
        "detect_fps": detect_fps,
        "paused": bool(paused),
        "has_source": bool(has_source),
        "source": source,
        "stream": "/stream.mjpg",
    }


class DashboardBehaviour:
    """Own one Member 3 live session for the Member 2 dashboard loop."""

    def __init__(self) -> None:
        self.session: Any | None = None
        self.cache: dict[int, dict[str, Any]] = {}
        self.error: str | None = None
        self._create()

    def _create(self, zones: dict[str, list[list[float]]] | None = None) -> None:
        try:
            cls = load_live_behaviour_session()
            self.session = cls()
            if zones:
                self.session.set_zones(zones)
            self.cache = {}
            self.events: deque[dict[str, Any]] = deque(maxlen=40)
            self._seen_firings: dict[tuple, int] = {}
            self.error = None
        except Exception as exc:  # noqa: BLE001 - dashboard must keep tracking
            self.session = None
            self.cache = {}
            self.events = deque(maxlen=40)
            self._seen_firings = {}
            self.error = f"Member 3 behaviour analysis is unavailable: {exc}"

    def reset(self, zones: dict[str, list[list[float]]] | None = None) -> None:
        self._create(zones)

    def set_zones(self, zones: dict[str, list[list[float]]]) -> None:
        if self.session is not None:
            self.session.set_zones(zones)

    def runtime_status(self) -> dict[str, Any]:
        loaded = self.session is not None
        people = 0
        if loaded:
            try:
                people = len(self.session.store.person_ids())
            except Exception:  # noqa: BLE001
                people = len(self.cache)
        return {
            "loaded": loaded,
            "available": loaded and self.error is None,
            "error": self.error,
            "people_scored": people,
        }

    def update(
        self,
        frame_id: int,
        records: Sequence[dict[str, Any]] | None,
        people_rows: Sequence[dict[str, Any]] | None,
        elapsed_s: float,
        detections: Sequence[dict[str, Any]] | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        if self.session is None:
            return empty_behaviour_state(frame_id=frame_id, elapsed_s=elapsed_s, error=self.error)
        try:
            started = time.perf_counter()
            tracks = tracks_for_member3(records, people_rows)
            views = self.session.update(
                frame_id,
                tracks,
                elapsed_s=elapsed_s,
                detections=detections,
                frame_size=frame_size,
            )
            self.cache.update(views)
            terminated = {
                int(row["person_id"])
                for row in (records or [])
                if isinstance(row, dict) and row.get("status") == "terminated" and "person_id" in row
            }
            for person_id in list(self.cache):
                if person_id in terminated:
                    self.cache.pop(person_id, None)
                    self._seen_firings = {
                        key: score for key, score in self._seen_firings.items() if key[0] != person_id
                    }
            payload = build_behaviour_state(
                self.session,
                views,
                self.cache,
                records,
                people_rows,
                frame_id=frame_id,
                elapsed_s=elapsed_s,
                process_ms=(time.perf_counter() - started) * 1000,
                has_tracking_input=bool(tracks),
            )
            self._record_events(payload["people"], elapsed_s)
            payload["events"] = list(self.events)
            payload["suspected_alerts"] = [
                item
                for item in self.events
                if item.get("person_id") is not None and item.get("type") in SHOPLIFTING_ALERT_TYPES
            ]
            return payload
        except Exception as exc:  # noqa: BLE001 - keep the tracking page alive
            self.error = str(exc)
            return empty_behaviour_state(frame_id=frame_id, elapsed_s=elapsed_s, error=self.error)

    def _record_events(self, people: Sequence[dict[str, Any]], elapsed_s: float) -> None:
        for person in people:
            try:
                person_id = int(person["person_id"])
            except (KeyError, TypeError, ValueError):
                continue
            for item in person.get("behaviours") or []:
                name = item.get("type")
                if not name or name in ITEM_ALERT_TYPES:
                    continue
                score = int(item.get("score") or 0)
                key = (person_id, str(name), "")
                if self._seen_firings.get(key) == score:
                    continue
                self._seen_firings[key] = score
                self.events.appendleft(
                    {
                        "elapsed_s": round(float(elapsed_s), 3),
                        "person_id": person_id,
                        "type": str(name),
                        "score": score,
                        "zone": person.get("zone"),
                        "level": person.get("item_alert_level") or person.get("suspicious_level"),
                        "evidence": None,
                        "object_class": None,
                        "object_id": None,
                    }
                )
            for alert in person.get("item_alerts") or []:
                name = alert.get("type")
                if not name:
                    continue
                score = int(alert.get("score") or 0)
                object_id = alert.get("object_id")
                key = (person_id, str(name), object_id)
                if key in self._seen_firings:
                    continue
                self._seen_firings[key] = score
                self.events.appendleft(
                    {
                        "elapsed_s": round(float(elapsed_s), 3),
                        "person_id": person_id,
                        "type": str(name),
                        "score": score,
                        "zone": person.get("zone"),
                        "level": alert.get("level") or "low",
                        "evidence": alert.get("evidence"),
                        "object_class": alert.get("object_class"),
                        "object_id": object_id,
                    }
                )


def indicator_rows(
    behaviours: Sequence[dict[str, Any]],
    pending: Sequence[str],
    hidden: Sequence[str],
) -> list[dict[str, Any]]:
    """Map Member 3 detector output onto detected / waiting / unavailable / not_detected."""
    fired = {item["type"]: int(item["score"]) for item in behaviours if item.get("type")}
    hidden_set = set(hidden)
    rows: list[dict[str, Any]] = []
    for name in INDICATOR_TYPES:
        if name in fired:
            rows.append({"type": name, "status": "detected", "score": fired[name], "detail": None})
            continue
        detail = next((item for item in pending if item.startswith(f"{name} unavailable")), None)
        if name in hidden_set or (name in OBJECT_EVIDENCE_TYPES and detail):
            rows.append(
                {
                    "type": name,
                    "status": "unavailable",
                    "score": None,
                    "detail": detail or f"{name} unavailable: required evidence is missing",
                }
            )
            continue
        waiting = next(
            (
                item
                for item in pending
                if item.startswith(f"{name} waiting")
                or (name in {"loitering", "repeated_shelf_visit"} and "need more frames" in item)
            ),
            None,
        )
        if waiting:
            rows.append({"type": name, "status": "waiting", "score": None, "detail": waiting})
            continue
        rows.append({"type": name, "status": "not_detected", "score": None, "detail": None})
    return rows


def movement_history_payload(session: Any, person_id: int, current_zone: str | None) -> dict[str, Any]:
    """Movement-history fields from Member 3's store and motion profile."""
    payload: dict[str, Any] = {
        "length": 0,
        "duration_s": None,
        "zone": current_zone,
        "zones_visited": [],
        "zone_transitions": [],
        "shelf_counts": {},
        "shelf_repeats": 0,
        "had_shelf_contact": False,
        "longest_still_s": None,
        "samples": [],
    }
    store = getattr(session, "store", None)
    if store is None:
        return payload
    try:
        records = store.records_for(person_id)
    except KeyError:
        return payload
    payload["length"] = len(records)
    payload["duration_s"] = _duration_s(session, person_id)
    zones: list[str] = []
    transitions: list[dict[str, str]] = []
    previous: str | None = None
    samples: list[dict[str, Any]] = []
    for record in records:
        zone = getattr(record, "zone", None)
        if isinstance(zone, str) and zone.strip():
            zone = zone.strip()
            if zone not in zones:
                zones.append(zone)
            if previous is not None and zone != previous:
                transitions.append({"from": previous, "to": zone})
            previous = zone
        samples.append(
            {
                "frame_id": getattr(record, "frame_id", None),
                "timestamp": getattr(record, "timestamp", None),
                "zone": zone if isinstance(zone, str) else None,
            }
        )
    payload["zones_visited"] = zones
    payload["zone_transitions"] = transitions
    payload["samples"] = samples[-12:]
    profile = _motion_profile(session, person_id)
    if profile is not None:
        counts = {str(name): int(count) for name, count in (getattr(profile, "shelf_counts", {}) or {}).items()}
        payload["shelf_counts"] = counts
        payload["shelf_repeats"] = sum(count - 1 for count in counts.values() if count >= 2)
        payload["had_shelf_contact"] = bool(getattr(profile, "had_shelf_contact", False))
        still = getattr(profile, "longest_still_s", None)
        if isinstance(still, (int, float)) and not isinstance(still, bool):
            payload["longest_still_s"] = round(float(still), 3)
    return payload


def _motion_profile(session: Any, person_id: int) -> Any | None:
    features = sys.modules.get("_member3.features")
    if features is None or session is None:
        return None
    try:
        records = session.store.records_for(person_id)
    except (AttributeError, KeyError):
        return None
    try:
        return features.build_profile(
            records,
            getattr(session, "detection_index", {}) or {},
            session.settings,
            session.zones,
        )
    except Exception:  # noqa: BLE001
        return None


def _index_people(people_rows: Sequence[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in people_rows or []:
        if not isinstance(row, dict) or "person_id" not in row:
            continue
        try:
            indexed[int(row["person_id"])] = row
        except (TypeError, ValueError):
            continue
    return indexed


def _index_records(records: Sequence[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in records or []:
        if not isinstance(row, dict) or "person_id" not in row:
            continue
        try:
            indexed[int(row["person_id"])] = row
        except (TypeError, ValueError):
            continue
    return indexed


def _duration_s(session: Any, person_id: int) -> float | None:
    store = getattr(session, "store", None)
    if store is None:
        return None
    try:
        records = store.records_for(person_id)
    except KeyError:
        return None
    if not records:
        return None
    first = _as_seconds(getattr(records[0], "timestamp", None))
    last = _as_seconds(getattr(records[-1], "timestamp", None))
    if first is None or last is None:
        return None
    return max(0.0, last - first)


def _window_seconds(session: Any, person_id: int) -> float | None:
    results = getattr(session, "last_results", {}) or {}
    assessment = (results.get(person_id) or {}).get("assessment") or {}
    value = assessment.get("window_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _as_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        parts = text.split(":")
        if len(parts) != 3:
            return None
        try:
            hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None
        return hours * 3600 + minutes * 60 + seconds


def _is_pending(pending: Sequence[str], behaviours: Sequence[dict[str, Any]]) -> bool:
    if behaviours:
        return False
    return any("waiting" in item or "need more frames" in item for item in pending)


def _normalise_not_contributing(value: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for item in value or []:
        if isinstance(item, dict) and item.get("type"):
            items.append({"type": str(item["type"]), "reason": str(item.get("reason") or "")})
        elif isinstance(item, str) and item.strip():
            items.append({"type": item.strip(), "reason": ""})
    return items
