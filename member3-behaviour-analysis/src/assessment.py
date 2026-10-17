"""Combine behavioural signals that fall inside one configurable time window.

Detector scores stay on each behaviour. This result is a separate suspicious
level. It is not Member 4's risk level, and the cutoffs are unvalidated
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import AssessmentSettings
from detectors import INDICATOR_TYPES

SUSPICIOUS_LEVELS = ("none", "medium", "high")
EXCLUSION_REASONS = ("isolated", "outside_window", "context_only", "missing_time")
COMBINED_SCORE_CAP = 100


@dataclass(frozen=True)
class TimedIndicator:
    """One detector result placed on the track clock."""

    type: str
    score: int
    start_s: float | None
    end_s: float | None


def assess_indicators(
    events: list[TimedIndicator], settings: AssessmentSettings
) -> dict[str, Any]:
    """Return the combined assessment for one person.

    Loitering, a repeated shelf visit, and examining or picking up a product
    do not raise a suspicious level, alone or together. A level is assigned
    only when a signal outside that set shares a time window with at least
    one other signal. An indicator with no start and end time is not treated
    as if it happened together with the others.
    """
    _reject_duplicates(events)
    context = set(settings.context_only)
    missing = [event for event in events if not _has_interval(event)]
    timed = [event for event in events if _has_interval(event)]
    groups = _clusters(timed, settings.window_seconds)
    chosen = _strongest_qualifying(groups, settings, context)
    used = set() if chosen is None else {event.type for event in chosen}
    not_contributing: list[dict[str, str]] = []
    for event in missing:
        not_contributing.append({"type": event.type, "reason": "missing_time"})
    for group in groups:
        if any(event.type in used for event in group):
            continue
        reason = _exclusion_reason(group, context, len(timed))
        for event in group:
            not_contributing.append({"type": event.type, "reason": reason})
    contributing = [name for name in INDICATOR_TYPES if name in used]
    not_contributing.sort(key=lambda item: INDICATOR_TYPES.index(item["type"]))
    if chosen is None:
        level = "none"
        combined = 0
    else:
        count = len(chosen)
        level = "high" if count >= settings.high_min_indicators else "medium"
        combined = min(COMBINED_SCORE_CAP, sum(event.score for event in chosen))
    return {
        "suspicious_level": level,
        "combined_score": combined,
        "window_seconds": settings.window_seconds,
        "contributing": contributing,
        "not_contributing": not_contributing,
    }


def _has_interval(event: TimedIndicator) -> bool:
    if event.start_s is None or event.end_s is None:
        return False
    return event.end_s >= event.start_s


def _reject_duplicates(events: list[TimedIndicator]) -> None:
    seen: list[str] = []
    for event in events:
        if event.type in seen:
            raise ValueError(f"duplicate indicator in assessment: {event.type}")
        seen.append(event.type)


def _clusters(events: list[TimedIndicator], window_seconds: float) -> list[list[TimedIndicator]]:
    parent = list(range(len(events)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            if _gap(events[left], events[right]) <= window_seconds:
                parent[find(left)] = find(right)
    grouped: dict[int, list[TimedIndicator]] = {}
    for index, event in enumerate(events):
        grouped.setdefault(find(index), []).append(event)
    return list(grouped.values())


def _gap(left: TimedIndicator, right: TimedIndicator) -> float:
    assert left.start_s is not None and left.end_s is not None
    assert right.start_s is not None and right.end_s is not None
    if left.end_s < right.start_s:
        return right.start_s - left.end_s
    if right.end_s < left.start_s:
        return left.start_s - right.end_s
    return 0.0


def _strongest_qualifying(
    groups: list[list[TimedIndicator]],
    settings: AssessmentSettings,
    context: set[str],
) -> list[TimedIndicator] | None:
    qualifying = [group for group in groups if _qualifies(group, settings, context)]
    if not qualifying:
        return None
    return max(
        qualifying,
        key=lambda group: (
            sum(event.score for event in group),
            len(group),
            -min(event.start_s or 0.0 for event in group),
        ),
    )


def _qualifies(group: list[TimedIndicator], settings: AssessmentSettings, context: set[str]) -> bool:
    if len(group) < settings.min_indicators:
        return False
    return any(event.type not in context for event in group)


def _exclusion_reason(
    group: list[TimedIndicator], context: set[str], timed_count: int
) -> str:
    if all(event.type in context for event in group):
        return "context_only"
    if len(group) == 1 and timed_count == 1:
        return "isolated"
    return "outside_window"
