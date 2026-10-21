"""Reject behaviour-training inputs that are not reviewed behaviour labels.

Video folder names are not behaviour targets. This module does not train a
classifier and does not score the five behaviours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from video_dataset import BEHAVIOUR_TYPES

FOLDER_LABELS = ("normal", "shoplifting")
CHECKLIST_VALUES = ("present", "absent", "cannot_tell")
COMPARISON_APPROACHES = ("rule_based", "single_indicator", "multi_indicator", "ml")


class AnnotationRejected(ValueError):
    """The file cannot be used as behaviour ground truth."""


def load_behaviour_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load reviewed annotation JSON files from a file or a directory."""
    file_path = Path(path)
    if not file_path.exists():
        raise AnnotationRejected("reviewed ground-truth behaviour labels are missing")
    if file_path.is_dir():
        files = sorted(
            item for item in file_path.iterdir() if item.is_file() and item.suffix.lower() == ".json"
        )
        if not files:
            raise AnnotationRejected("reviewed ground-truth behaviour labels are missing")
        return [load_annotation_file(item) for item in files]
    return [load_annotation_file(file_path)]


def load_annotation_file(path: str | Path) -> dict[str, Any]:
    """Read one annotation file and reject an incomplete review."""
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationRejected(f"annotation file is not valid JSON: {file_path}") from exc
    if not isinstance(payload, dict):
        raise AnnotationRejected("annotation input is empty")
    return validate_annotation(payload)


def validate_annotation(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Accept one reviewed checklist. Folder names cannot fill a behaviour."""
    if not isinstance(payload, dict) or not payload:
        raise AnnotationRejected("annotation input is empty")
    for key in ("label", "behaviour_target", "positive_class"):
        if payload.get(key) in FOLDER_LABELS:
            raise AnnotationRejected("a video folder label cannot be a behaviour target")
    if payload.get("review_status") != "reviewed":
        raise AnnotationRejected("annotation is incomplete: review_status must be reviewed")
    checklist = payload.get("checklist")
    if not isinstance(checklist, dict):
        raise AnnotationRejected("annotation is incomplete: checklist is missing")
    if set(checklist) != set(BEHAVIOUR_TYPES):
        raise AnnotationRejected("annotation is incomplete: checklist must contain the five behaviours")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise AnnotationRejected("annotation is incomplete: segments must be a list")
    for behaviour in BEHAVIOUR_TYPES:
        value = checklist[behaviour]
        if value in FOLDER_LABELS:
            raise AnnotationRejected("a video folder label cannot be a behaviour target")
        if value not in CHECKLIST_VALUES:
            raise AnnotationRejected(f"annotation is incomplete: {behaviour} has no decision")
        if value == "present" and not _present_segment(segments, behaviour):
            raise AnnotationRejected(
                f"annotation is incomplete: {behaviour} is present without a timed visible segment"
            )
    return payload


def compare_behaviour_approaches(annotations: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    """Block a comparison when reviewed behaviour labels are missing.

    A present review list is accepted and still produces no metrics. No
    behaviour classifier has been trained, and no approach predictions are
    supplied here.
    """
    if not annotations:
        raise AnnotationRejected("reviewed ground-truth behaviour labels are missing")
    reviewed = [validate_annotation(item) for item in annotations]
    return {
        "evaluated": False,
        "approaches": list(COMPARISON_APPROACHES),
        "reason": (
            "Reviewed behaviour labels were accepted. No behaviour classifier has been trained "
            "and no approach predictions were supplied, so no metrics were produced."
        ),
        "metrics": None,
        "reviewed_count": len(reviewed),
    }


def _present_segment(segments: Sequence[Any], behaviour: str) -> bool:
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("behaviour") != behaviour:
            continue
        start = segment.get("start_s")
        end = segment.get("end_s")
        evidence = segment.get("visible_evidence", segment.get("what_was_seen"))
        if isinstance(start, bool) or isinstance(end, bool):
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end <= start:
            continue
        if isinstance(evidence, str) and evidence.strip():
            return True
    return False
