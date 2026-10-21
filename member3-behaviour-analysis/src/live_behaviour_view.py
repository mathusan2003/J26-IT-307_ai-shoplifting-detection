"""Live-display helpers for the Member 2 + Member 3 webcam demo.

The five rule detectors stay unchanged. This module feeds Member 2 track
rows into the existing analyser, keeps per-person history across frames,
and decides what the live window may show. Object interaction and possible
concealment stay hidden on the person-only camera path: Member 2's
PersonDetector returns person boxes only, and person detection is not
concealment, object interaction, or shoplifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from behaviour import analyse_person
from config import BehaviourSettings, load_settings, load_zones
from features import FEATURE_NAMES, MotionProfile, build_profile, index_detections
from member2_adapter import frame_from_member2_tracks
from item_association import ITEM_TYPES, LEVEL_RANK, ItemAssociationSession, adapt_member1_detections
from movement_history import MovementHistoryStore

LIVE_EXCLUDED: tuple[str, ...] = ()
OBJECT_EVIDENCE_TYPES = ("object_interaction", "possible_concealment") + ITEM_TYPES
MEMBER2_ROOT = Path(__file__).resolve().parents[2] / "member2-person-tracking"
# Webcam close-ups need COCO yolov8n.pt. The retail fine-tunes
# (yolov8n-person-track.pt, yolov8n-cam-multiperson.pt) were trained for
# distant CCTV and split one nearby person into head/torso/arm boxes.
COCO_PERSON_WEIGHTS = MEMBER2_ROOT / "models" / "yolov8n.pt"
PREFERRED_WEIGHTS = COCO_PERSON_WEIGHTS
DEFAULT_WEIGHTS = COCO_PERSON_WEIGHTS
LIVE_DETECT_CONFIDENCE = 0.45
LIVE_IMGSZ = 640
LIVE_NMS_IOU = 0.50
LIVE_MAX_DET = 20
LIVE_MIN_HEIGHT_FRAC = 0.12
LIVE_MIN_ASPECT = 0.18
# Close-up webcam boxes often clip at the bottom of the frame, so width/height
# exceeds the 1.50 CCTV cap used to drop banners. 2.2 still rejects wide signs.
LIVE_MAX_ASPECT = 2.2
LIVE_CONTAINMENT = 0.50
LIVE_MERGE_IOU = 0.35


def live_view_from_result(
    result: dict[str, Any],
    has_object_detections: bool = False,
    pending: Sequence[str] | None = None,
    history_length: int = 0,
) -> dict[str, Any]:
    """Copy one analyser result for the live overlay.

    The Member 4 contract object is not modified. The live score is the
    capped sum of the indicators that are allowed on screen. It is not
    Member 4's risk_score.
    """
    visible: list[dict[str, Any]] = []
    hidden: list[str] = []
    for item in result.get("behaviours", []):
        name = item.get("type")
        if name in LIVE_EXCLUDED:
            hidden.append(str(name))
            continue
        if name in OBJECT_EVIDENCE_TYPES and not has_object_detections:
            hidden.append(str(name))
            continue
        visible.append({"type": item["type"], "score": item["score"]})
    assessment = result.get("assessment") if isinstance(result.get("assessment"), dict) else {}
    contributing = [
        name
        for name in assessment.get("contributing", [])
        if name not in hidden and name not in LIVE_EXCLUDED
        and (name not in OBJECT_EVIDENCE_TYPES or has_object_detections)
    ]
    return {
        "person_id": result["person_id"],
        "behaviours": visible,
        "live_score": min(100, sum(int(item["score"]) for item in visible)),
        "behaviour_score": int(result.get("behaviour_score") or 0),
        "combined_score": int(assessment.get("combined_score") or 0),
        "suspicious_level": assessment.get("suspicious_level") or "none",
        "contributing": contributing,
        "not_contributing": list(assessment.get("not_contributing") or []),
        "pending": list(pending or []),
        "history_length": int(history_length),
        "excluded_from_live": list(LIVE_EXCLUDED),
        "hidden_without_objects": hidden,
        "object_evidence": has_object_detections,
        "not_member4_risk_score": True,
        "not_shoplifting_proof": True,
    }


def format_overlay_lines(view: dict[str, Any], zone: str | None = None) -> list[str]:
    """Compact overlay lines: indicators, scores, level, and waiting status."""
    lines: list[str] = []
    behaviours = view.get("behaviours") or []
    if behaviours:
        lines.append(" ".join(f"{item['type']}:{item['score']}" for item in behaviours))
    else:
        pending = [item for item in view.get("pending") or [] if "waiting" in item or "need more frames" in item]
        if pending:
            lines.append(pending[0])
        else:
            lines.append("no indicators yet")
    contrib = view.get("contributing") or []
    lines.append(
        f"score={view.get('live_score', 0)} combined={view.get('combined_score', 0)} "
        f"level={view.get('suspicious_level', 'none')}"
    )
    lines.append("contrib=" + (",".join(contrib) if contrib else "none"))
    if zone:
        lines.append(f"zone={zone}")
    return lines


def contract_frame(
    frame_id: int,
    tracks: list[dict[str, Any]],
    timestamp_s: float | None = None,
) -> dict[str, Any]:
    """Build one Member 2 tracking frame from tracker contract rows.

    ``timestamp_s`` is elapsed seconds as a numeric string. Member 3 uses
    that clock when it is present, instead of ``frame_id / 25``. Live YOLO
    frames are slower than 25 fps, so frame_id alone under-counts time.
    """
    return frame_from_member2_tracks(frame_id, tracks, elapsed_s=timestamp_s)


def attach_member2_zones(
    tracks: Sequence[dict[str, Any]],
    analytics: Any,
    elapsed_s: float,
    as_bbox: Any,
) -> list[dict[str, Any]]:
    """Copy contract tracks and add Member 2's current zone when it exists."""
    enriched: list[dict[str, Any]] = []
    for track in tracks:
        row = dict(track)
        if analytics is None or row.get("zone"):
            enriched.append(row)
            continue
        try:
            bbox = as_bbox(row["bbox"])
            person_id = int(row["person_id"])
        except (KeyError, TypeError, ValueError):
            enriched.append(row)
            continue
        zone = analytics.update(person_id, bbox, elapsed_s).current_zone
        if zone:
            row["zone"] = zone
        enriched.append(row)
    return enriched


def history_lengths(frames: list[dict[str, Any]]) -> dict[int, int]:
    """Number of ingested positions per person, in frame order."""
    store = MovementHistoryStore()
    store.ingest_frames(frames)
    return {person_id: len(store.records_for(person_id)) for person_id in store.person_ids()}


def pending_detector_reasons(
    profile: MotionProfile,
    settings: BehaviourSettings,
    *,
    history_length: int,
    has_object_detections: bool,
    fired_types: Sequence[str] | None = None,
) -> list[str]:
    """Explain why a live detector has not scored yet. Thresholds are unchanged."""
    fired = set(fired_types or [])
    loiter = settings.loitering
    abnormal = settings.abnormal_movement
    reasons: list[str] = []
    if history_length < 2:
        reasons.append("need more frames: one observation is not loitering or a repeated visit")
    if "loitering" not in fired and profile.longest_still_s < loiter.min_seconds:
        reasons.append(
            f"loitering waiting: still {profile.longest_still_s:.1f}s/{loiter.min_seconds:.0f}s "
            f"inside {loiter.radius_px:.0f}px"
        )
    repeats = sum(count - 1 for count in profile.shelf_counts.values() if count >= 2)
    if "repeated_shelf_visit" not in fired:
        if not profile.had_shelf_contact:
            reasons.append("repeated_shelf_visit waiting: no shelf zone yet")
        elif repeats < 1:
            reasons.append("repeated_shelf_visit waiting: leave and return to the same shelf")
    if "abnormal_movement" not in fired:
        reasons.append(
            f"abnormal_movement waiting: speed {profile.max_speed:.0f}/"
            f"{abnormal.high_speed_px_per_s:.0f} px/s, "
            f"reversals {profile.reversal_count}/{abnormal.min_reversals}"
        )
    if not has_object_detections:
        if "object_interaction" not in fired:
            reasons.append("object_interaction unavailable: person boxes only, not merchandise proof")
        if "possible_concealment" not in fired:
            reasons.append("possible_concealment unavailable: person boxes only, not concealment proof")
    return reasons


def explain_live_detectors(
    frames: list[dict[str, Any]],
    settings: BehaviourSettings | None = None,
    zones: dict[str, list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    """Return detector inputs and why live scores are zero or available.

    Thresholds come from the existing config. This does not change them.
    """
    active = settings if settings is not None else load_settings()
    active_zones = zones if zones is not None else load_zones()
    store = MovementHistoryStore()
    store.ingest_frames(frames)
    reports = []
    for person_id in store.person_ids():
        records = store.records_for(person_id)
        profile = build_profile(records, index_detections(None), active, active_zones)
        loiter = active.loitering
        abnormal = active.abnormal_movement
        still_ok = profile.longest_still_s >= loiter.min_seconds
        speed_ok = profile.max_speed >= abnormal.high_speed_px_per_s
        reversal_ok = profile.reversal_count >= abnormal.min_reversals
        reasons = pending_detector_reasons(
            profile,
            active,
            history_length=len(records),
            has_object_detections=False,
        )
        reports.append(
            {
                "person_id": person_id,
                "history_length": len(records),
                "first_center": [records[0].position[0], records[0].position[1]],
                "last_center": [records[-1].position[0], records[-1].position[1]],
                "duration_s": profile.duration_s,
                "longest_still_s": profile.longest_still_s,
                "loitering_min_s": loiter.min_seconds,
                "loitering_radius_px": loiter.radius_px,
                "max_speed": profile.max_speed,
                "abnormal_speed_min": abnormal.high_speed_px_per_s,
                "reversal_count": profile.reversal_count,
                "reversal_min": abnormal.min_reversals,
                "loitering_ready": still_ok,
                "abnormal_ready": speed_ok or reversal_ok,
                "reasons": reasons,
                "feature_vector_width": len(profile.as_vector()),
                "feature_names": list(FEATURE_NAMES),
            }
        )
    return reports


def detect_live_people(
    detector: Any,
    frame: Any,
    debug: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run Member 2's already-loaded YOLO once, then keep whole-person boxes.

    PersonDetector.detect() also applies a CCTV area cap of 45% of the frame.
    A laptop webcam person often exceeds that, so the live path uses the same
    model object and a webcam filter instead of a second detector.
    """
    result = detector.model(
        frame,
        classes=[0],
        conf=detector.confidence,
        iou=detector.nms_iou,
        imgsz=detector.imgsz,
        max_det=detector.max_det,
        verbose=False,
    )[0]
    height, width = (frame.shape[0], frame.shape[1]) if hasattr(frame, "shape") else (1080, 1920)
    people: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else 0
        if class_id != 0:
            continue
        people.append(
            {
                "class": "person",
                "confidence": float(box.conf[0]),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            }
        )
    kept = keep_whole_people(people, width, height)
    if debug is not None:
        debug["raw"] = people
        debug["kept"] = kept
        debug["width"] = width
        debug["height"] = height
    return kept


def keep_whole_people(
    detections: Sequence[dict[str, Any]] | None,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Keep one full-person box. Drop face, torso, and arm fragments.

    Member 2's crowded-CCTV filters still allow those crops. The live webcam
    path applies this extra pass so Member 3 does not score body parts.
    """
    candidates: list[dict[str, Any]] = []
    for item in detections or []:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        box = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        width, height = max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])
        if height < LIVE_MIN_HEIGHT_FRAC * frame_height or width < 10:
            continue
        ratio = width / height if height else 0.0
        if ratio < LIVE_MIN_ASPECT or ratio > LIVE_MAX_ASPECT:
            continue
        kept = dict(item)
        kept["bbox"] = box
        candidates.append(kept)
    candidates.sort(key=lambda item: (_box_area(item["bbox"]), float(item.get("confidence") or 0.0)), reverse=True)
    people: list[dict[str, Any]] = []
    for item in candidates:
        box = item["bbox"]
        duplicate = False
        for kept in people:
            other = kept["bbox"]
            if _iou(box, other) >= LIVE_MERGE_IOU or _containment(box, other) >= LIVE_CONTAINMENT:
                duplicate = True
                break
        if not duplicate:
            people.append(item)
    return people


def _box_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    return max(0.0, right - left) * max(0.0, bottom - top)


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = _intersection_area(first, second)
    if not overlap:
        return 0.0
    union = _box_area(first) + _box_area(second) - overlap
    return overlap / union if union else 0.0


def _containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    area = _box_area(inner)
    if area <= 0:
        return 0.0
    return _intersection_area(inner, outer) / area


def has_non_person_objects(detections: list[dict[str, Any]] | None) -> bool:
    for item in detections or []:
        name = item.get("class") or item.get("class_name")
        if isinstance(name, str) and name.strip().lower() not in {"", "person", "people", "pedestrian"}:
            return True
    return False


class LiveBehaviourSession:
    """Keep Member 2 identities and run Member 3 detectors on growing history.

    One MovementHistoryStore is reused. Empty frames do not reset a person.
    A new person_id starts a new history. This does not run detection or tracking.
    """

    def __init__(
        self,
        settings: BehaviourSettings | None = None,
        zones: dict[str, list[list[float]]] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_settings()
        self.zones = zones if zones is not None else load_zones()
        self.store = MovementHistoryStore()
        self.detection_index: dict[int, list[dict[str, Any]]] = {}
        self.views: dict[int, dict[str, Any]] = {}
        self.last_results: dict[int, dict[str, Any]] = {}
        self.last_frame: dict[str, Any] | None = None
        self.items = ItemAssociationSession()
        self.last_object_rows: list[dict[str, Any]] | None = None
        self.last_object_input: str = "unavailable"
        self.last_object_debug: dict[str, Any] = {}

    def set_zones(self, zones: dict[str, list[list[float]]]) -> None:
        """Use Member 2's scaled polygons when the camera size is known."""
        self.zones = zones

    def update(
        self,
        frame_id: int,
        tracks: Sequence[dict[str, Any]] | None,
        elapsed_s: float | int | None,
        detections: Sequence[dict[str, Any]] | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Ingest one live tracking frame and score currently visible people."""
        frame = frame_from_member2_tracks(frame_id, tracks, elapsed_s=elapsed_s)
        self.last_frame = frame
        self.store.ingest_frame(frame)
        detection_rows, object_debug = adapt_member1_detections(detections)
        self.last_object_rows = detection_rows
        self.last_object_debug = object_debug
        self.last_object_input = str(object_debug.get("input") or "empty")
        has_objects = bool(detection_rows)
        detection_frames = (
            None
            if detection_rows is None
            else [{"frame_id": frame["frame_id"], "detections": detection_rows}]
        )
        for frame_key, items in index_detections(detection_frames).items():
            self.detection_index.setdefault(frame_key, []).extend(items)
        detection_index = self.detection_index
        visible_ids = [int(track["person_id"]) for track in frame["tracks"]]
        item_alerts = self.items.update(
            frame_id,
            float(elapsed_s or 0.0),
            frame["tracks"],
            detection_rows,
            self.zones,
            frame_size,
        )
        object_debug["object_tracks"] = len(self.items.tracks)
        object_debug["associated"] = sum(
            1
            for track in self.items.tracks.values()
            if track.person_ids and track.person_ids[-1] is not None and track.missed == 0
        )
        object_debug["new_item_alerts"] = sum(len(rows) for rows in item_alerts.values())
        self.last_object_debug = object_debug
        self.views = {}
        for person_id in visible_ids:
            result = analyse_person(person_id, self.store, detection_index, self.settings, self.zones)
            records = self.store.records_for(person_id)
            profile = build_profile(records, detection_index, self.settings, self.zones)
            sticky = list(self.items.alerts.get(person_id) or [])
            fired = [item["type"] for item in result.get("behaviours", [])] + [item["type"] for item in sticky]
            pending = pending_detector_reasons(
                profile,
                self.settings,
                history_length=len(records),
                has_object_detections=has_objects,
                fired_types=fired,
            )
            pending.extend(self.items.pending_reasons(has_object_detections=has_objects, fired_types=fired))
            view = live_view_from_result(
                result,
                has_objects,
                pending=pending,
                history_length=len(records),
            )
            view["item_alerts"] = sticky
            view["item_alerts_new"] = list(item_alerts.get(person_id) or [])
            if sticky:
                extra = [{"type": item["type"], "score": int(item["score"])} for item in sticky]
                view["behaviours"] = list(view.get("behaviours") or []) + extra
                added = sum(int(item["score"]) for item in extra)
                view["live_score"] = min(100, int(view.get("live_score") or 0) + added)
                best = "none"
                for item in sticky:
                    level = str(item.get("level") or "none")
                    if LEVEL_RANK.get(level, 0) > LEVEL_RANK.get(best, 0):
                        best = level
                    item["combined_score"] = int(view.get("combined_score") or 0)
                    item["displayed_score"] = int(view["live_score"])
                view["item_alert_level"] = best
            else:
                view["item_alert_level"] = "none"
            self.views[person_id] = view
            self.last_results[person_id] = result
        return self.views

    def history_length(self, person_id: int) -> int:
        try:
            return len(self.store.records_for(person_id))
        except KeyError:
            return 0


def available_member2_weight_files(roots: Sequence[Path] | None = None) -> list[Path]:
    """Return existing Member 2 weight files in the detector's search order."""
    detector = _member2_detector_module()
    search_roots = [Path.cwd(), MEMBER2_ROOT] if roots is None else list(roots)
    found: list[Path] = []
    seen: set[str] = set()
    for relative in detector.MODEL_PRIORITY:
        for root in search_roots:
            candidate = root / relative if not Path(relative).is_absolute() else Path(relative)
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                found.append(candidate)
    return found


def resolve_member2_live_weights(
    weights: Path | str | None = None,
    roots: Sequence[Path] | None = None,
) -> Path:
    """Use Member 2's existing weight resolver. Do not invent a missing file."""
    detector = _member2_detector_module()
    requested = Path(DEFAULT_WEIGHTS) if weights is None else Path(weights)
    search_roots = [Path.cwd(), MEMBER2_ROOT] if roots is None else list(roots)
    return detector.resolve_weights(requested, roots=search_roots)


def probe_member2_live_pipeline(
    weights: Path | None = None,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Report whether Member 2's detector and tracker can be imported."""
    requested = Path(weights) if weights is not None else DEFAULT_WEIGHTS
    resolved = resolve_member2_live_weights(requested, roots=roots)
    available = available_member2_weight_files(roots=roots)
    report: dict[str, Any] = {
        "member2_root": str(MEMBER2_ROOT),
        "member2_root_exists": MEMBER2_ROOT.is_dir(),
        "requested_weights": str(requested),
        "weights": str(resolved),
        "weights_exist": resolved.is_file(),
        "available_weights": [str(path) for path in available],
        "ultralytics": False,
        "tracker": False,
        "ready": False,
        "missing": [],
    }
    if not report["member2_root_exists"]:
        report["missing"].append(f"Member 2 folder not found: {MEMBER2_ROOT}")
    if not report["weights_exist"]:
        report["missing"].append(f"YOLOv8 weights not found: {resolved}")
    try:
        import ultralytics  # noqa: F401

        report["ultralytics"] = True
    except ImportError:
        report["missing"].append("Python package ultralytics is not installed")
    try:
        load_member2_tracker()
        report["tracker"] = True
    except Exception as exc:
        report["missing"].append(f"Member 2 tracker could not be imported: {exc}")
    report["ready"] = not report["missing"]
    return report


def load_member2_tracker():
    """Import Member 2's ByteTrack factory without changing Member 2 files."""
    import sys

    root = str(MEMBER2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.pipeline import load_tracker_settings, make_tracker

    return load_tracker_settings, make_tracker


def load_member2_detector(weights: Path | None = None):
    """Import Member 2's person-only YOLO adapter with webcam-safe settings.

    Crowded CCTV defaults (conf 0.20, imgsz 1080) treat close body parts as
    extra people. The live camera uses COCO-scale inference instead.
    """
    import sys

    root = str(MEMBER2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.detector import PersonDetector

    resolved = resolve_member2_live_weights(weights)
    return PersonDetector(
        resolved,
        confidence=LIVE_DETECT_CONFIDENCE,
        imgsz=LIVE_IMGSZ,
        max_det=LIVE_MAX_DET,
        nms_iou=LIVE_NMS_IOU,
    )


def load_member2_overlay():
    """Reuse Member 2's box, ID, trail, and zone overlay without copying it."""
    import sys

    root = str(MEMBER2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.utils import as_bbox
    from src.visualize import draw_tracks
    from src.zone_analytics import ZoneAnalytics

    return as_bbox, draw_tracks, ZoneAnalytics


def _member2_detector_module():
    import sys

    root = str(MEMBER2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src import detector as member2_detector

    return member2_detector
