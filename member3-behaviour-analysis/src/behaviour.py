"""Turn a Member 2 tracking sequence into Member 4 behaviour JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from assessment import EXCLUSION_REASONS, SUSPICIOUS_LEVELS, TimedIndicator, assess_indicators
from config import BehaviourSettings, load_settings, load_zones
from detectors import INDICATOR_TYPES, detect_indicators
from features import build_profile, index_detections
from movement_history import MovementHistoryStore
from tracking_input import load_tracking_sequence

BEHAVIOUR_SCORE_CAP = 100


def load_detection_sequence(path: str | Path) -> list[dict[str, Any]]:
    """Load optional object detections aligned by ``frame_id``.

    The file uses Member 1's detection shape. Person boxes are ignored when
    interactions are measured. This module only reads the file.
    """
    frames = load_tracking_sequence(path)
    for index, frame in enumerate(frames):
        if not isinstance(frame.get("detections"), list):
            raise ValueError(f"frame at index {index} is missing a detections list")
    return frames


def analyse_tracking_frames(
    frames: Sequence[dict[str, Any]],
    detections: Sequence[dict[str, Any]] | None = None,
    settings: BehaviourSettings | None = None,
    zones: dict[str, list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    """Analyse every tracked person and return contract behaviour results."""
    active_settings = settings if settings is not None else load_settings()
    active_zones = zones if zones is not None else load_zones()
    store = MovementHistoryStore()
    store.ingest_frames(frames)
    detection_index = index_detections(detections)
    return [
        analyse_person(person_id, store, detection_index, active_settings, active_zones)
        for person_id in store.person_ids()
    ]


def analyse_person(
    person_id: int,
    store: MovementHistoryStore,
    detections_by_frame: dict[int, list[dict[str, Any]]],
    settings: BehaviourSettings,
    zones: dict[str, list[list[float]]],
) -> dict[str, Any]:
    """Build one behaviour result for a person already stored in the history."""
    records = store.records_for(person_id)
    profile = build_profile(records, detections_by_frame, settings, zones)
    indicators = detect_indicators(profile, settings)
    behaviour_score = min(BEHAVIOUR_SCORE_CAP, sum(item.score for item in indicators))
    events = [
        TimedIndicator(item.type, item.score, *_span_bounds(profile.evidence_spans.get(item.type)))
        for item in indicators
    ]
    return {
        "person_id": person_id,
        "behaviours": [item.to_dict() for item in indicators],
        "behaviour_score": behaviour_score,
        "assessment": assess_indicators(events, settings.assessment),
    }


def _span_bounds(span: tuple[float, float] | None) -> tuple[float | None, float | None]:
    if span is None:
        return None, None
    return span[0], span[1]


def validate_behaviour_result(result: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a result does not match the Member 4 contract.

    ``behaviours`` and ``behaviour_score`` keep their original meaning.
    ``assessment`` is an extra object Member 4 may ignore.
    """
    expected = {"person_id", "behaviours", "behaviour_score", "assessment"}
    if set(result) != expected:
        raise ValueError(
            "behaviour result fields must be person_id, behaviours, behaviour_score, assessment"
        )
    person_id = result["person_id"]
    if isinstance(person_id, bool) or not isinstance(person_id, int):
        raise ValueError("person_id must be an integer")
    behaviours = result["behaviours"]
    if not isinstance(behaviours, list):
        raise ValueError("behaviours must be a list")
    seen: list[str] = []
    for item in behaviours:
        if set(item) != {"type", "score"}:
            raise ValueError("each behaviour must contain type and score")
        if item["type"] not in INDICATOR_TYPES:
            raise ValueError(f"unknown behaviour type: {item['type']}")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, int) or score <= 0:
            raise ValueError("behaviour scores must be positive integers")
        seen.append(item["type"])
    if seen != sorted(seen, key=INDICATOR_TYPES.index):
        raise ValueError("behaviours must follow the documented indicator order")
    total = result["behaviour_score"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0 or total > BEHAVIOUR_SCORE_CAP:
        raise ValueError("behaviour_score must be an integer from 0 to 100")
    if total != min(BEHAVIOUR_SCORE_CAP, sum(item["score"] for item in behaviours)):
        raise ValueError("behaviour_score must be the capped sum of indicator scores")
    _validate_assessment(result["assessment"], behaviours)


def _validate_assessment(assessment: dict[str, Any], behaviours: list[dict[str, Any]]) -> None:
    expected = {
        "suspicious_level",
        "combined_score",
        "window_seconds",
        "contributing",
        "not_contributing",
    }
    if not isinstance(assessment, dict) or set(assessment) != expected:
        raise ValueError("assessment fields do not match the combined-assessment object")
    level = assessment["suspicious_level"]
    if level not in SUSPICIOUS_LEVELS:
        raise ValueError("suspicious_level must be none, medium, or high")
    window = assessment["window_seconds"]
    if isinstance(window, bool) or not isinstance(window, (int, float)) or window <= 0:
        raise ValueError("window_seconds must be a positive number")
    combined = assessment["combined_score"]
    if isinstance(combined, bool) or not isinstance(combined, int) or combined < 0 or combined > BEHAVIOUR_SCORE_CAP:
        raise ValueError("combined_score must be an integer from 0 to 100")
    contributing = assessment["contributing"]
    not_contributing = assessment["not_contributing"]
    if not isinstance(contributing, list) or not isinstance(not_contributing, list):
        raise ValueError("assessment indicator lists must be lists")
    scores = {item["type"]: item["score"] for item in behaviours}
    for name in contributing:
        if name not in scores:
            raise ValueError("contributing indicators must be detector results")
    for item in not_contributing:
        if not isinstance(item, dict) or set(item) != {"type", "reason"}:
            raise ValueError("each excluded indicator must contain type and reason")
        if item["type"] not in scores:
            raise ValueError("excluded indicators must be detector results")
        if item["reason"] not in EXCLUSION_REASONS:
            raise ValueError(f"unknown assessment reason: {item['reason']}")
    excluded = [item["type"] for item in not_contributing]
    if set(contributing) & set(excluded) or set(contributing) | set(excluded) != set(scores):
        raise ValueError("assessment must account for every detector result once")
    if contributing != [name for name in INDICATOR_TYPES if name in contributing]:
        raise ValueError("contributing indicators must follow the documented indicator order")
    if excluded != [name for name in INDICATOR_TYPES if name in excluded]:
        raise ValueError("excluded indicators must follow the documented indicator order")
    if level == "none":
        if contributing or combined != 0:
            raise ValueError("a none level has no contributing indicators and a combined score of 0")
        return
    if not contributing:
        raise ValueError("a suspicious level needs contributing indicators")
    expected_score = min(BEHAVIOUR_SCORE_CAP, sum(scores[name] for name in contributing))
    if combined != expected_score:
        raise ValueError("combined_score must be the capped sum of contributing indicator scores")


def write_behaviour_results(results: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write behaviour JSON for Member 4."""
    for result in results:
        validate_behaviour_result(result)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(list(results), indent=2) + "\n", encoding="utf-8")
