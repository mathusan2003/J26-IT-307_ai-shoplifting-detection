"""Rule scores for the five suspicious-behaviour indicators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import BehaviourSettings
from features import MotionProfile

INDICATOR_TYPES = (
    "loitering",
    "repeated_shelf_visit",
    "abnormal_movement",
    "object_interaction",
    "possible_concealment",
)


@dataclass(frozen=True)
class Indicator:
    """One behaviour indicator in the Member 4 contract."""

    type: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "score": self.score}


def detect_indicators(profile: MotionProfile, settings: BehaviourSettings) -> list[Indicator]:
    """Return the indicators that fired, in contract order."""
    candidates = (
        loitering_indicator(profile, settings),
        repeated_shelf_indicator(profile, settings),
        abnormal_movement_indicator(profile, settings),
        object_interaction_indicator(profile, settings),
        concealment_indicator(profile, settings),
    )
    return [item for item in candidates if item is not None and item.score > 0]


def loitering_indicator(profile: MotionProfile, settings: BehaviourSettings) -> Indicator | None:
    """Score a person who stays inside a small radius for a long time."""
    rule = settings.loitering
    if profile.longest_still_s < rule.min_seconds:
        return None
    extra = profile.longest_still_s - rule.min_seconds
    score = _round_half_up(rule.base_score + extra * rule.per_extra_second)
    return Indicator("loitering", _cap(score, rule.max_score))


def repeated_shelf_indicator(profile: MotionProfile, settings: BehaviourSettings) -> Indicator | None:
    """Score extra visits after the first visit to the same shelf."""
    rule = settings.repeated_shelf_visit
    repeats = sum(count - 1 for count in profile.shelf_counts.values() if count >= 2)
    if repeats < 1:
        return None
    return Indicator("repeated_shelf_visit", _cap(repeats * rule.score_per_repeat, rule.max_score))


def abnormal_movement_indicator(
    profile: MotionProfile, settings: BehaviourSettings
) -> Indicator | None:
    """Score unusually fast movement or a run of sharp direction changes.

    A folded path adds to the score only after speed or reversals have
    already triggered the indicator. Returning to a shelf is folded too,
    and that pattern belongs to repeated shelf visits.
    """
    rule = settings.abnormal_movement
    parts = 0
    triggered = False
    if profile.max_speed >= rule.high_speed_px_per_s:
        triggered = True
        parts += rule.speed_score
    if profile.reversal_count >= rule.min_reversals:
        triggered = True
        parts += min(rule.reversal_score_cap, profile.reversal_count * rule.reversal_score_per_event)
    if (
        triggered
        and profile.path_length_px >= rule.min_path_px
        and profile.tortuosity >= rule.min_tortuosity
    ):
        parts += rule.tortuosity_score
    if not triggered:
        return None
    return Indicator("abnormal_movement", _cap(parts, rule.max_score))


def object_interaction_indicator(
    profile: MotionProfile, settings: BehaviourSettings
) -> Indicator | None:
    """Score handling motion on a shelf, or sustained proximity to an object."""
    rule = settings.object_interaction
    dwell_part = 0
    if profile.shelf_slow_s >= rule.min_handling_seconds:
        scaled = rule.handling_score_cap * profile.shelf_slow_s / rule.min_handling_seconds
        dwell_part = min(rule.handling_score_cap, _round_half_up(scaled))
    object_part = 0
    if profile.object_overlap_frames >= rule.min_overlap_frames:
        object_part = min(
            rule.object_score_cap,
            profile.object_overlap_frames * rule.object_score_per_frame,
        )
    if dwell_part == 0 and object_part == 0:
        return None
    return Indicator("object_interaction", _cap(dwell_part + object_part, rule.max_score))


def concealment_indicator(profile: MotionProfile, settings: BehaviourSettings) -> Indicator | None:
    """Score a box-size drop after shelf or object contact.

    This is a possible-concealment signal. It does not see hands or pockets.
    A drop with no earlier shelf contact and no nearby object is ignored,
    because the person may simply be farther from the camera.
    """
    rule = settings.possible_concealment
    shrunk = profile.bbox_shrink_ratio <= rule.shrink_ratio
    context = profile.had_shelf_contact or profile.object_overlap_frames >= rule.min_overlap_frames
    if not shrunk or not context:
        return None
    score = rule.shrink_score
    if profile.object_lost:
        score += rule.loss_score
    return Indicator("possible_concealment", _cap(score, rule.max_score))


def _cap(score: int, maximum: int) -> int:
    return max(0, min(maximum, score))


def _round_half_up(value: float) -> int:
    return int(value + 0.5)
