"""Live-display helpers for the Member 3 webcam demo.

The five rule detectors stay unchanged. This module only decides what the
live window may show. repeated_shelf_visit is kept in the repository and
is dropped from the live overlay and live score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import BehaviourSettings, load_settings, load_zones
from features import FEATURE_NAMES, build_profile, index_detections
from movement_history import MovementHistoryStore

LIVE_EXCLUDED = ("repeated_shelf_visit",)
OBJECT_EVIDENCE_TYPES = ("object_interaction", "possible_concealment")
MEMBER2_ROOT = Path(__file__).resolve().parents[2] / "member2-person-tracking"
DEFAULT_WEIGHTS = MEMBER2_ROOT / "models" / "yolov8n.pt"


def live_view_from_result(
    result: dict[str, Any],
    has_object_detections: bool = False,
) -> dict[str, Any]:
    """Copy one analyser result for the live overlay.

    The Member 4 contract object is not modified. The live score is the
    capped sum of the indicators that are allowed on screen. It is not
    Member 4's risk_score.
    """
    visible: list[dict[str, Any]] = []
    for item in result.get("behaviours", []):
        name = item.get("type")
        if name in LIVE_EXCLUDED:
            continue
        if name in OBJECT_EVIDENCE_TYPES and not has_object_detections:
            continue
        visible.append({"type": item["type"], "score": item["score"]})
    return {
        "person_id": result["person_id"],
        "behaviours": visible,
        "live_score": min(100, sum(int(item["score"]) for item in visible)),
        "excluded_from_live": list(LIVE_EXCLUDED),
        "object_evidence": has_object_detections,
        "not_member4_risk_score": True,
    }


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
    from member2_adapter import frame_from_member2_tracks

    return frame_from_member2_tracks(frame_id, tracks, elapsed_s=timestamp_s)


def history_lengths(frames: list[dict[str, Any]]) -> dict[int, int]:
    """Number of ingested positions per person, in frame order."""
    store = MovementHistoryStore()
    store.ingest_frames(frames)
    return {person_id: len(store.records_for(person_id)) for person_id in store.person_ids()}


def explain_live_detectors(
    frames: list[dict[str, Any]],
    settings: BehaviourSettings | None = None,
) -> list[dict[str, Any]]:
    """Return detector inputs and why live scores are zero or available.

    Thresholds come from the existing config. This does not change them.
    """
    active = settings if settings is not None else load_settings()
    store = MovementHistoryStore()
    store.ingest_frames(frames)
    reports = []
    for person_id in store.person_ids():
        records = store.records_for(person_id)
        profile = build_profile(records, index_detections(None), active, load_zones())
        loiter = active.loitering
        abnormal = active.abnormal_movement
        still_ok = profile.longest_still_s >= loiter.min_seconds
        speed_ok = profile.max_speed >= abnormal.high_speed_px_per_s
        reversal_ok = profile.reversal_count >= abnormal.min_reversals
        reasons = []
        if len(records) < 2:
            reasons.append("need_at_least_two_positions")
        if not still_ok:
            reasons.append(
                f"loitering_unavailable: longest_still {profile.longest_still_s:.2f}s < {loiter.min_seconds:.0f}s "
                f"or movement exceeded {loiter.radius_px:.0f}px"
            )
        if not speed_ok and not reversal_ok:
            reasons.append(
                f"abnormal_unavailable: max_speed {profile.max_speed:.1f} < {abnormal.high_speed_px_per_s:.0f} px/s "
                f"and reversals {profile.reversal_count} < {abnormal.min_reversals}"
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


def has_non_person_objects(detections: list[dict[str, Any]] | None) -> bool:
    for item in detections or []:
        name = item.get("class")
        if isinstance(name, str) and name.strip().lower() not in {"", "person"}:
            return True
    return False


def probe_member2_live_pipeline(weights: Path | None = None) -> dict[str, Any]:
    """Report whether Member 2's detector and tracker can be imported."""
    weight_path = weights if weights is not None else DEFAULT_WEIGHTS
    report: dict[str, Any] = {
        "member2_root": str(MEMBER2_ROOT),
        "member2_root_exists": MEMBER2_ROOT.is_dir(),
        "weights": str(weight_path),
        "weights_exist": weight_path.is_file(),
        "ultralytics": False,
        "tracker": False,
        "ready": False,
        "missing": [],
    }
    if not report["member2_root_exists"]:
        report["missing"].append(f"Member 2 folder not found: {MEMBER2_ROOT}")
    if not report["weights_exist"]:
        report["missing"].append(f"YOLOv8 weights not found: {weight_path}")
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
    """Import Member 2's person-only YOLO adapter."""
    import sys

    root = str(MEMBER2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.detector import PersonDetector

    return PersonDetector(weights if weights is not None else DEFAULT_WEIGHTS)


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
