"""Associate Member 1 object boxes with Member 2 person tracks over time.

This does not see hands, pockets, or retail SKUs. Proximity alone is not a
pickup. A missing box is not concealment. Events are suspected indicators,
not shoplifting proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from geometry import bbox_center, distance, iou, shelf_key, zone_for_point

PERSON_CLASSES = {"", "person", "people", "pedestrian"}
BAG_CLASSES = {"backpack", "handbag", "suitcase"}
ITEM_TYPES = (
    "item_pickup_suspected",
    "possible_item_concealment",
    "item_removed_from_shelf",
    "repeated_shelf_interaction",
)
MATCH_IOU = 0.30
MATCH_CENTER_PX = 90.0
ASSOCIATE_IOU = 0.01
ASSOCIATE_DISTANCE_PX = 90.0
ASSOCIATE_FRAMES = 3
COMOVE_OBJECT_PX = 40.0
COMOVE_PERSON_PX = 30.0
INWARD_PX = 20.0
ABSENCE_FRAMES = 3
EDGE_MARGIN_PX = 40.0
MAX_MISSED = 8
LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def member1_to_member3_detections(detections: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Map Member 1 prototype boxes onto Member 3's class/confidence/bbox-list rows."""
    rows, _debug = adapt_member1_detections(detections)
    return list(rows or [])


def adapt_member1_detections(
    detections: Sequence[Any] | None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Convert Member 1 rows without turning a detector failure into an empty success."""
    if detections is None:
        return None, {
            "input": "unavailable",
            "raw": None,
            "non_person": None,
            "dropped_person": 0,
            "dropped_invalid": 0,
            "classes": [],
            "keys": None,
            "sample": [],
        }
    rows: list[dict[str, Any]] = []
    dropped_person = 0
    dropped_invalid = 0
    keys: list[str] | None = None
    for item in detections:
        mapped = _coerce_detection(item)
        if mapped is None:
            dropped_invalid += 1
            continue
        if keys is None:
            keys = mapped["raw_keys"]
        if mapped["class"] in PERSON_CLASSES:
            dropped_person += 1
            continue
        rows.append(
            {
                "class": mapped["class"],
                "confidence": mapped["confidence"],
                "bbox": mapped["bbox"],
                "detection_id": mapped["detection_id"],
            }
        )
    return rows, {
        "input": "populated" if rows else "empty",
        "raw": len(list(detections)),
        "non_person": len(rows),
        "dropped_person": dropped_person,
        "dropped_invalid": dropped_invalid,
        "classes": [item["class"] for item in rows],
        "keys": keys,
        "sample": [
            {"class": item["class"], "bbox": item["bbox"], "confidence": item["confidence"]}
            for item in rows[:4]
        ],
    }


def _coerce_detection(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        if not hasattr(item, "bbox") and not hasattr(item, "xyxy"):
            return None
        item = {
            "class_name": getattr(item, "class_name", None),
            "class": getattr(item, "class", None),
            "confidence": getattr(item, "confidence", None) or getattr(item, "conf", None),
            "bbox": getattr(item, "bbox", None) or getattr(item, "xyxy", None),
            "detection_id": getattr(item, "detection_id", None),
        }
    raw_keys = sorted(str(key) for key in item.keys())
    name = item.get("class_name") or item.get("class") or item.get("label") or item.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    box = _as_xyxy(item.get("bbox") if item.get("bbox") is not None else item.get("xyxy"))
    if box is None:
        return None
    try:
        confidence = float(item["confidence"]) if item.get("confidence") is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "class": name.strip().lower(),
        "confidence": confidence,
        "bbox": box,
        "detection_id": item.get("detection_id"),
        "raw_keys": raw_keys,
    }


def _as_xyxy(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    if hasattr(bbox, "tolist") and not isinstance(bbox, (list, tuple, dict)):
        try:
            bbox = bbox.tolist()
        except Exception:
            return None
    values: list[Any] | None = None
    if isinstance(bbox, dict):
        for keys in (("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax"), ("left", "top", "right", "bottom")):
            if set(keys) <= set(bbox):
                values = [bbox[keys[0]], bbox[keys[1]], bbox[keys[2]], bbox[keys[3]]]
                break
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = list(bbox)
    if values is None:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def _level_max(left: str, right: str) -> str:
    return left if LEVEL_RANK.get(left, 0) >= LEVEL_RANK.get(right, 0) else right


@dataclass
class ObjectTrack:
    object_id: int
    class_name: str
    frames: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)
    person_ids: list[int | None] = field(default_factory=list)
    person_centers: list[tuple[float, float] | None] = field(default_factory=list)
    zones: list[str | None] = field(default_factory=list)
    missed: int = 0
    fired: set[str] = field(default_factory=set)


class ItemAssociationSession:
    """Keep object IDs across frames and emit live item indicators."""

    def __init__(self) -> None:
        self._next_id = 1
        self.tracks: dict[int, ObjectTrack] = {}
        self.person_shelf_visits: dict[tuple[int, str], int] = {}
        self.person_in_shelf: dict[tuple[int, str], bool] = {}
        self.alerts: dict[int, list[dict[str, Any]]] = {}
        self._active_keys: set[tuple[int, str, int | None]] = set()

    def reset(self) -> None:
        self.__init__()

    def update(
        self,
        frame_id: int,
        elapsed_s: float,
        tracks: Sequence[dict[str, Any]] | None,
        detections: Sequence[dict[str, Any]] | None,
        zones: dict[str, list[list[float]]] | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> dict[int, list[dict[str, Any]]]:
        people = [row for row in (tracks or []) if isinstance(row, dict) and "person_id" in row]
        if detections is None:
            return {int(row["person_id"]): [] for row in people}
        objects = member1_to_member3_detections(detections)
        width, height = frame_size if frame_size else (0, 0)
        self._match_objects(frame_id, float(elapsed_s), objects, zones or {})
        self._associate_people(people)
        self._update_shelf_visits(people, zones or {})
        frame_alerts: dict[int, list[dict[str, Any]]] = {}
        for track in list(self.tracks.values()):
            for event in self._events_for(track, people, width, height, elapsed_s):
                person_id = int(event["person_id"])
                object_id = None if event["type"] == "repeated_shelf_interaction" else event.get("object_id")
                key = (person_id, event["type"], object_id)
                if key in self._active_keys:
                    continue
                track.fired.add(event["type"])
                self._active_keys.add(key)
                frame_alerts.setdefault(person_id, []).append(event)
                self.alerts.setdefault(person_id, []).append(event)
        self._drop_stale()
        current = {int(row["person_id"]): list(frame_alerts.get(int(row["person_id"]), [])) for row in people}
        for person_id, events in frame_alerts.items():
            current.setdefault(person_id, events)
        return current

    def pending_reasons(self, *, has_object_detections: bool, fired_types: Sequence[str]) -> list[str]:
        fired = set(fired_types or [])
        reasons: list[str] = []
        if not has_object_detections:
            for name in ITEM_TYPES:
                if name not in fired:
                    reasons.append(
                        f"{name} unavailable: no non-person Member 1 boxes; "
                        "COCO cannot prove shelf theft or concealment"
                    )
            return reasons
        if "item_pickup_suspected" not in fired:
            reasons.append(
                "item_pickup_suspected waiting: object must stay associated and move with the person, "
                "not merely pass nearby"
            )
        if "possible_item_concealment" not in fired:
            reasons.append(
                "possible_item_concealment waiting: need inward motion then a miss while the person "
                "stays inside the frame; a lost box is not hiding"
            )
        if "item_removed_from_shelf" not in fired:
            reasons.append(
                "item_removed_from_shelf waiting: object must start in a shelf zone and then move with the person"
            )
        if "repeated_shelf_interaction" not in fired:
            reasons.append(
                "repeated_shelf_interaction waiting: leave and return to the same shelf with a nearby object"
            )
        return reasons

    def _match_objects(
        self,
        frame_id: int,
        elapsed_s: float,
        detections: list[dict[str, Any]],
        zones: dict[str, list[list[float]]],
    ) -> None:
        used: set[int] = set()
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class"]
            best_id, best_score = None, -1.0
            for track in self.tracks.values():
                if track.object_id in used or track.class_name != class_name or not track.bboxes:
                    continue
                overlap = iou(track.bboxes[-1], bbox)
                dist = distance(bbox_center(track.bboxes[-1]), bbox_center(bbox))
                if overlap < MATCH_IOU and dist > MATCH_CENTER_PX:
                    continue
                score = overlap * 100.0 + (MATCH_CENTER_PX - min(dist, MATCH_CENTER_PX))
                if score > best_score:
                    best_id, best_score = track.object_id, score
            if best_id is None:
                track = ObjectTrack(object_id=self._next_id, class_name=class_name)
                self._next_id += 1
                self.tracks[track.object_id] = track
            else:
                track = self.tracks[best_id]
            used.add(track.object_id)
            track.missed = 0
            track.frames.append(frame_id)
            track.times.append(elapsed_s)
            track.bboxes.append(bbox)
            center = bbox_center(bbox)
            track.zones.append(zone_for_point(center, zones) if zones else None)
            track.person_ids.append(None)
            track.person_centers.append(None)
        for track in self.tracks.values():
            if track.object_id not in used:
                track.missed += 1

    def _associate_people(self, people: list[dict[str, Any]]) -> None:
        for track in self.tracks.values():
            if track.missed or not track.bboxes:
                continue
            bbox = track.bboxes[-1]
            best_id, best_score = None, 0.0
            for person in people:
                person_box = _as_xyxy(person.get("bbox"))
                if person_box is None:
                    continue
                overlap = iou(bbox, person_box)
                center = person.get("center")
                if not isinstance(center, (list, tuple)) or len(center) < 2:
                    center = bbox_center(person_box)
                near = distance(bbox_center(bbox), (float(center[0]), float(center[1])))
                if overlap >= ASSOCIATE_IOU or near <= ASSOCIATE_DISTANCE_PX:
                    score = overlap + (1.0 / max(near, 1.0))
                    if score > best_score:
                        best_id, best_score = int(person["person_id"]), score
            track.person_ids[-1] = best_id
            if best_id is None:
                continue
            person = next((row for row in people if int(row["person_id"]) == best_id), None)
            if person is None:
                continue
            center = person.get("center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                track.person_centers[-1] = (float(center[0]), float(center[1]))
            else:
                person_box = _as_xyxy(person.get("bbox"))
                if person_box is not None:
                    track.person_centers[-1] = bbox_center(person_box)

    def _update_shelf_visits(self, people: list[dict[str, Any]], zones: dict[str, list[list[float]]]) -> None:
        associated = {track.person_ids[-1] for track in self.tracks.values() if track.person_ids and track.person_ids[-1]}
        for person in people:
            person_id = int(person["person_id"])
            zone = person.get("zone")
            if not isinstance(zone, str):
                box = _as_xyxy(person.get("bbox"))
                if box is not None and zones:
                    zone = zone_for_point(bbox_center(box), zones)
            key_name = shelf_key(zone if isinstance(zone, str) else None)
            if key_name is None:
                for shelf in list(self.person_in_shelf):
                    if shelf[0] == person_id:
                        self.person_in_shelf[shelf] = False
                continue
            key = (person_id, key_name)
            has_object = person_id in associated
            was_inside = self.person_in_shelf.get(key, False)
            if has_object and not was_inside:
                self.person_shelf_visits[key] = self.person_shelf_visits.get(key, 0) + 1
            self.person_in_shelf[key] = has_object

    def _events_for(
        self,
        track: ObjectTrack,
        people: list[dict[str, Any]],
        width: int,
        height: int,
        elapsed_s: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        person_id = _dominant_person(track)
        if person_id is None:
            return events
        person = next((row for row in people if int(row["person_id"]) == person_id), None)
        pickup = None
        if "item_pickup_suspected" not in track.fired:
            pickup = _pickup_event(track, person_id, elapsed_s)
            if pickup:
                events.append(pickup)
        if pickup or "item_pickup_suspected" in track.fired:
            removed = _shelf_removal_event(track, person_id, person, elapsed_s)
            if removed and "item_removed_from_shelf" not in track.fired:
                events.append(removed)
        concealed = _concealment_event(track, person_id, person, width, height, elapsed_s)
        if concealed and "possible_item_concealment" not in track.fired:
            events.append(concealed)
        repeated = _repeated_shelf_event(track, person_id, self.person_shelf_visits, elapsed_s)
        if repeated and "repeated_shelf_interaction" not in track.fired:
            events.append(repeated)
        return events

    def _drop_stale(self) -> None:
        stale = [object_id for object_id, track in self.tracks.items() if track.missed > MAX_MISSED]
        for object_id in stale:
            self.tracks.pop(object_id, None)


def _dominant_person(track: ObjectTrack) -> int | None:
    counts: dict[int, int] = {}
    for person_id in track.person_ids:
        if person_id is None:
            continue
        counts[person_id] = counts.get(person_id, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _associated_indices(track: ObjectTrack, person_id: int) -> list[int]:
    return [index for index, value in enumerate(track.person_ids) if value == person_id]


def _alert(
    *,
    person_id: int,
    type_name: str,
    score: int,
    level: str,
    track: ObjectTrack,
    elapsed_s: float,
    evidence: str,
    frame_id: int | None = None,
) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "type": type_name,
        "score": score,
        "level": level,
        "object_id": track.object_id,
        "object_class": track.class_name,
        "frame_id": frame_id if frame_id is not None else (track.frames[-1] if track.frames else None),
        "elapsed_s": round(float(elapsed_s), 3),
        "evidence": evidence,
        "behaviour_score": score,
        "combined_score": None,
        "not_shoplifting_proof": True,
    }


def _pickup_event(track: ObjectTrack, person_id: int, elapsed_s: float) -> dict[str, Any] | None:
    indices = _associated_indices(track, person_id)
    if len(indices) < ASSOCIATE_FRAMES:
        return None
    first, last = indices[0], indices[-1]
    if last <= first:
        return None
    object_move = distance(bbox_center(track.bboxes[first]), bbox_center(track.bboxes[last]))
    if object_move < COMOVE_OBJECT_PX:
        return None
    person_first = track.person_centers[first] if first < len(track.person_centers) else None
    person_last = track.person_centers[last] if last < len(track.person_centers) else None
    if person_first is None or person_last is None:
        return None
    person_move = distance(person_first, person_last)
    if person_move < COMOVE_PERSON_PX:
        return None
    return _alert(
        person_id=person_id,
        type_name="item_pickup_suspected",
        score=18,
        level="low",
        track=track,
        elapsed_s=elapsed_s,
        frame_id=track.frames[last],
        evidence=(
            f"{track.class_name} object {track.object_id} stayed associated for "
            f"{len(indices)} frames, moved {object_move:.0f}px, and person {person_id} "
            f"moved {person_move:.0f}px with it. Proximity alone is not a pickup; "
            "COCO cannot confirm a hand grasp or a store SKU."
        ),
    )


def _shelf_removal_event(
    track: ObjectTrack,
    person_id: int,
    person: dict[str, Any] | None,
    elapsed_s: float,
) -> dict[str, Any] | None:
    if "item_pickup_suspected" not in track.fired and not _pickup_event(track, person_id, elapsed_s):
        return None
    start_shelf = next((shelf_key(zone) for zone in track.zones if shelf_key(zone)), None)
    if start_shelf is None:
        return None
    current_zone = person.get("zone") if person else None
    current_shelf = shelf_key(current_zone if isinstance(current_zone, str) else None)
    if current_shelf == start_shelf:
        return None
    return _alert(
        person_id=person_id,
        type_name="item_removed_from_shelf",
        score=16,
        level="low",
        track=track,
        elapsed_s=elapsed_s,
        evidence=(
            f"{track.class_name} object {track.object_id} started in {start_shelf} and later "
            f"moved with person {person_id} away from that shelf. Not proof the item left the store."
        ),
    )


def _concealment_event(
    track: ObjectTrack,
    person_id: int,
    person: dict[str, Any] | None,
    width: int,
    height: int,
    elapsed_s: float,
) -> dict[str, Any] | None:
    indices = _associated_indices(track, person_id)
    if len(indices) < ASSOCIATE_FRAMES:
        return None
    if track.missed < ABSENCE_FRAMES:
        return None
    if person is None:
        return None
    first, last = indices[0], indices[-1]
    start_person = track.person_centers[first] if first < len(track.person_centers) else None
    end_person = track.person_centers[last] if last < len(track.person_centers) else None
    if start_person is None or end_person is None:
        return None
    start_dist = distance(start_person, bbox_center(track.bboxes[first]))
    end_dist = distance(end_person, bbox_center(track.bboxes[last]))
    if start_dist - end_dist < INWARD_PX:
        return None
    last_box = track.bboxes[last]
    if width > 0 and height > 0 and _near_frame_edge(last_box, width, height, include_top=True):
        return None
    person_box = _as_xyxy(person.get("bbox"))
    if person_box is not None and width > 0 and height > 0 and _near_frame_edge(person_box, width, height, include_top=False):
        return None
    bag_note = " A bag/backpack class was not required and was not used as proof."
    return _alert(
        person_id=person_id,
        type_name="possible_item_concealment",
        score=24,
        level="medium",
        track=track,
        elapsed_s=elapsed_s,
        evidence=(
            f"{track.class_name} object {track.object_id} moved {start_dist - end_dist:.0f}px closer "
            f"to person {person_id}, then was unseen for {track.missed} frames while the person "
            f"stayed in view. A missing COCO box is not proof the item was hidden.{bag_note}"
        ),
    )


def _repeated_shelf_event(
    track: ObjectTrack,
    person_id: int,
    visits: dict[tuple[int, str], int],
    elapsed_s: float,
) -> dict[str, Any] | None:
    start_shelf = next((shelf_key(zone) for zone in track.zones if shelf_key(zone)), None)
    if start_shelf is None:
        return None
    count = visits.get((person_id, start_shelf), 0)
    if count < 2:
        return None
    return _alert(
        person_id=person_id,
        type_name="repeated_shelf_interaction",
        score=15,
        level="low",
        track=track,
        elapsed_s=elapsed_s,
        evidence=(
            f"Person {person_id} returned to {start_shelf} with a nearby {track.class_name} "
            f"({count} object-associated visits). This is not a purchase or theft label."
        ),
    )


def _person_object_distance(person: dict[str, Any], bbox: list[float]) -> float | None:
    person_box = _as_xyxy(person.get("bbox"))
    center = person.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        person_center = (float(center[0]), float(center[1]))
    elif person_box is not None:
        person_center = bbox_center(person_box)
    else:
        return None
    return distance(person_center, bbox_center(bbox))


def _near_frame_edge(bbox: Sequence[float], width: int, height: int, *, include_top: bool = True) -> bool:
    x1, y1, x2, y2 = bbox
    if x1 <= EDGE_MARGIN_PX or x2 >= width - EDGE_MARGIN_PX or y2 >= height - EDGE_MARGIN_PX:
        return True
    return include_top and y1 <= EDGE_MARGIN_PX
