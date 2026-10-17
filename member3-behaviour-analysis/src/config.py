"""Load behaviour thresholds and the local retail zone map."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = MODULE_ROOT / "configs" / "behaviour_config.json"
DEFAULT_ZONES_PATH = MODULE_ROOT / "configs" / "zones.json"


@dataclass(frozen=True)
class LoiteringSettings:
    radius_px: float
    min_seconds: float
    base_score: int
    per_extra_second: float
    max_score: int


@dataclass(frozen=True)
class RepeatedShelfSettings:
    score_per_repeat: int
    max_score: int


@dataclass(frozen=True)
class AbnormalMovementSettings:
    high_speed_px_per_s: float
    reversal_min_speed_px_per_s: float
    min_reversals: int
    min_tortuosity: float
    min_path_px: float
    speed_score: int
    reversal_score_per_event: int
    reversal_score_cap: int
    tortuosity_score: int
    max_score: int


@dataclass(frozen=True)
class ObjectInteractionSettings:
    min_handling_speed_px_per_s: float
    max_handling_speed_px_per_s: float
    min_handling_seconds: float
    object_distance_px: float
    min_object_iou: float
    min_object_confidence: float
    min_overlap_frames: int
    handling_score_cap: int
    object_score_per_frame: int
    object_score_cap: int
    max_score: int


@dataclass(frozen=True)
class ConcealmentSettings:
    shrink_ratio: float
    min_bbox_samples: int
    min_overlap_frames: int
    absence_frames: int
    shrink_score: int
    loss_score: int
    max_score: int


@dataclass(frozen=True)
class ModelSettings:
    learning_rate: float
    epochs: int
    l2: float
    threshold: float


@dataclass(frozen=True)
class AssessmentSettings:
    """Cutoffs for the combined suspicious level.

    These are assumptions. They are not a measured retail false-alarm rate.
    """

    window_seconds: float
    min_indicators: int
    context_only: tuple[str, ...]
    medium_min_indicators: int
    high_min_indicators: int
    assumption_note: str


@dataclass(frozen=True)
class BehaviourSettings:
    fps: float
    loitering: LoiteringSettings
    repeated_shelf_visit: RepeatedShelfSettings
    abnormal_movement: AbnormalMovementSettings
    object_interaction: ObjectInteractionSettings
    possible_concealment: ConcealmentSettings
    model: ModelSettings
    assessment: AssessmentSettings


_INDICATOR_NAMES = (
    "loitering",
    "repeated_shelf_visit",
    "abnormal_movement",
    "object_interaction",
    "possible_concealment",
)


def load_settings(path: str | Path | None = None) -> BehaviourSettings:
    """Load thresholds from JSON."""
    file_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"behaviour config must be a JSON object: {file_path}")
    return BehaviourSettings(
        fps=_number(payload, "fps"),
        loitering=LoiteringSettings(
            radius_px=_number(payload["loitering"], "radius_px"),
            min_seconds=_number(payload["loitering"], "min_seconds"),
            base_score=_whole(payload["loitering"], "base_score"),
            per_extra_second=_number(payload["loitering"], "per_extra_second"),
            max_score=_whole(payload["loitering"], "max_score"),
        ),
        repeated_shelf_visit=RepeatedShelfSettings(
            score_per_repeat=_whole(payload["repeated_shelf_visit"], "score_per_repeat"),
            max_score=_whole(payload["repeated_shelf_visit"], "max_score"),
        ),
        abnormal_movement=AbnormalMovementSettings(
            high_speed_px_per_s=_number(payload["abnormal_movement"], "high_speed_px_per_s"),
            reversal_min_speed_px_per_s=_number(
                payload["abnormal_movement"], "reversal_min_speed_px_per_s"
            ),
            min_reversals=_whole(payload["abnormal_movement"], "min_reversals"),
            min_tortuosity=_number(payload["abnormal_movement"], "min_tortuosity"),
            min_path_px=_number(payload["abnormal_movement"], "min_path_px"),
            speed_score=_whole(payload["abnormal_movement"], "speed_score"),
            reversal_score_per_event=_whole(
                payload["abnormal_movement"], "reversal_score_per_event"
            ),
            reversal_score_cap=_whole(payload["abnormal_movement"], "reversal_score_cap"),
            tortuosity_score=_whole(payload["abnormal_movement"], "tortuosity_score"),
            max_score=_whole(payload["abnormal_movement"], "max_score"),
        ),
        object_interaction=ObjectInteractionSettings(
            min_handling_speed_px_per_s=_number(
                payload["object_interaction"], "min_handling_speed_px_per_s"
            ),
            max_handling_speed_px_per_s=_number(
                payload["object_interaction"], "max_handling_speed_px_per_s"
            ),
            min_handling_seconds=_number(payload["object_interaction"], "min_handling_seconds"),
            object_distance_px=_number(payload["object_interaction"], "object_distance_px"),
            min_object_iou=_number(payload["object_interaction"], "min_object_iou"),
            min_object_confidence=_number(
                payload["object_interaction"], "min_object_confidence"
            ),
            min_overlap_frames=_whole(payload["object_interaction"], "min_overlap_frames"),
            handling_score_cap=_whole(payload["object_interaction"], "handling_score_cap"),
            object_score_per_frame=_whole(
                payload["object_interaction"], "object_score_per_frame"
            ),
            object_score_cap=_whole(payload["object_interaction"], "object_score_cap"),
            max_score=_whole(payload["object_interaction"], "max_score"),
        ),
        possible_concealment=ConcealmentSettings(
            shrink_ratio=_number(payload["possible_concealment"], "shrink_ratio"),
            min_bbox_samples=_whole(payload["possible_concealment"], "min_bbox_samples"),
            min_overlap_frames=_whole(payload["possible_concealment"], "min_overlap_frames"),
            absence_frames=_whole(payload["possible_concealment"], "absence_frames"),
            shrink_score=_whole(payload["possible_concealment"], "shrink_score"),
            loss_score=_whole(payload["possible_concealment"], "loss_score"),
            max_score=_whole(payload["possible_concealment"], "max_score"),
        ),
        model=ModelSettings(
            learning_rate=_number(payload["model"], "learning_rate"),
            epochs=_whole(payload["model"], "epochs"),
            l2=_number(payload["model"], "l2"),
            threshold=_number(payload["model"], "threshold"),
        ),
        assessment=_assessment(payload["assessment"]),
    )


def load_zones(path: str | Path | None = None) -> dict[str, list[list[float]]]:
    """Load named zone polygons."""
    file_path = Path(path) if path is not None else DEFAULT_ZONES_PATH
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    zones = payload["zones"] if isinstance(payload, dict) else payload
    if not isinstance(zones, dict) or not zones:
        raise ValueError(f"zone file must contain a zones object: {file_path}")
    return zones


def _number(container: dict[str, Any], key: str) -> float:
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _whole(container: dict[str, Any], key: str) -> int:
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _assessment(container: dict[str, Any]) -> AssessmentSettings:
    if not isinstance(container, dict):
        raise ValueError("assessment must be a JSON object")
    window_seconds = _number(container, "window_seconds")
    min_indicators = _whole(container, "min_indicators")
    medium_min_indicators = _whole(container, "medium_min_indicators")
    high_min_indicators = _whole(container, "high_min_indicators")
    context_only = _name_list(container, "context_only")
    note = container.get("assumption_note")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if min_indicators < 2:
        raise ValueError("min_indicators must be at least 2")
    if medium_min_indicators < min_indicators or high_min_indicators <= medium_min_indicators:
        raise ValueError("indicator cutoffs must rise from min to medium to high")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("assumption_note must explain that the cutoffs are unvalidated")
    return AssessmentSettings(
        window_seconds=window_seconds,
        min_indicators=min_indicators,
        context_only=context_only,
        medium_min_indicators=medium_min_indicators,
        high_min_indicators=high_min_indicators,
        assumption_note=note.strip(),
    )


def _name_list(container: dict[str, Any], key: str) -> tuple[str, ...]:
    value = container[key]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{key} must be a list of indicator names")
    names = tuple(item.strip() for item in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{key} must not repeat a name")
    unknown = [name for name in names if name not in _INDICATOR_NAMES]
    if unknown:
        raise ValueError(f"{key} has unknown indicator names: {', '.join(unknown)}")
    return names
