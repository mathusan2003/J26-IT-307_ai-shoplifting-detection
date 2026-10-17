"""Measurements shared by the rule detectors and the behaviour model."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from config import BehaviourSettings
from geometry import (
    bbox_area,
    bbox_center,
    bottom_center,
    distance,
    iou,
    shelf_key,
    zone_for_point,
)
from movement_history import MovementRecord

FEATURE_NAMES = (
    "duration_s",
    "path_length_px",
    "net_displacement_px",
    "mean_speed",
    "max_speed",
    "speed_std",
    "reversal_count",
    "tortuosity",
    "longest_still_s",
    "max_same_shelf_visits",
    "shelf_visit_total",
    "shelf_slow_s",
    "bbox_shrink_ratio",
    "object_overlap_frames",
    "object_lost",
)

_CLOCK = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$")


@dataclass(frozen=True)
class MotionProfile:
    """Numeric description of one person's track."""

    duration_s: float
    path_length_px: float
    net_displacement_px: float
    mean_speed: float
    max_speed: float
    speed_std: float
    reversal_count: int
    tortuosity: float
    longest_still_s: float
    shelf_counts: dict[str, int]
    shelf_slow_s: float
    bbox_shrink_ratio: float
    object_overlap_frames: int
    object_lost: bool
    had_shelf_contact: bool
    zones: tuple[str | None, ...]
    evidence_spans: dict[str, tuple[float, float]]

    def as_vector(self) -> list[float]:
        """Feature vector in ``FEATURE_NAMES`` order."""
        counts = self.shelf_counts.values()
        return [
            self.duration_s,
            self.path_length_px,
            self.net_displacement_px,
            self.mean_speed,
            self.max_speed,
            self.speed_std,
            float(self.reversal_count),
            self.tortuosity,
            self.longest_still_s,
            float(max(counts) if counts else 0),
            float(sum(counts) if counts else 0),
            self.shelf_slow_s,
            self.bbox_shrink_ratio,
            float(self.object_overlap_frames),
            1.0 if self.object_lost else 0.0,
        ]


def index_detections(frames: Sequence[dict[str, Any]] | None) -> dict[int, list[dict[str, Any]]]:
    """Index optional Member 1 detection frames by ``frame_id``."""
    indexed: dict[int, list[dict[str, Any]]] = {}
    for frame in frames or []:
        frame_id = frame.get("frame_id")
        detections = frame.get("detections", [])
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            continue
        if not isinstance(detections, list):
            continue
        indexed.setdefault(frame_id, []).extend(
            item for item in detections if isinstance(item, dict)
        )
    return indexed


def build_profile(
    records: Sequence[MovementRecord],
    detections_by_frame: dict[int, list[dict[str, Any]]],
    settings: BehaviourSettings,
    zones: dict[str, list[list[float]]],
) -> MotionProfile:
    """Measure one person's movement, shelf contact, and nearby objects."""
    times = _times(records, settings.fps)
    resolved = tuple(_resolve_zone(record, zones) for record in records)
    positions = [record.position for record in records]
    steps = _steps(times, positions, settings.abnormal_movement.reversal_min_speed_px_per_s)
    shelf_counts = _shelf_counts(resolved)
    overlap = _object_overlap(records, detections_by_frame, settings)
    return MotionProfile(
        duration_s=max(0.0, times[-1] - times[0]) if times else 0.0,
        path_length_px=steps["path_length"],
        net_displacement_px=distance(positions[0], positions[-1]) if len(positions) >= 2 else 0.0,
        mean_speed=steps["mean_speed"],
        max_speed=steps["max_speed"],
        speed_std=steps["speed_std"],
        reversal_count=steps["reversals"],
        tortuosity=steps["path_length"] / max(distance(positions[0], positions[-1]) if len(positions) >= 2 else 0.0, 1.0),
        longest_still_s=_longest_still(times, positions, settings.loitering.radius_px),
        shelf_counts=shelf_counts,
        shelf_slow_s=_shelf_handling_seconds(times, positions, resolved, settings),
        bbox_shrink_ratio=_shrink_ratio(records, settings.possible_concealment.min_bbox_samples),
        object_overlap_frames=overlap["frames"],
        object_lost=overlap["lost"],
        had_shelf_contact=any(shelf_key(zone) for zone in resolved),
        zones=resolved,
        evidence_spans=_evidence_spans(times, positions, resolved, records, detections_by_frame, settings),
    )


def _resolve_zone(record: MovementRecord, zones: dict[str, list[list[float]]]) -> str | None:
    if record.zone:
        return record.zone
    if record.bbox is not None:
        return zone_for_point(bottom_center(record.bbox), zones)
    return zone_for_point(record.position, zones)


def _times(records: Sequence[MovementRecord], fps: float) -> list[float]:
    has_timestamp = [record.timestamp is not None for record in records]
    if any(has_timestamp) and not all(has_timestamp):
        raise ValueError("a person's track must timestamp every frame or none of them")
    if records and records[0].timestamp is not None:
        return [_clock_to_seconds(record.timestamp) for record in records]
    if fps <= 0:
        raise ValueError("fps must be positive when timestamps are absent")
    return [record.frame_id / fps for record in records]


def _clock_to_seconds(timestamp: str) -> float:
    match = _CLOCK.match(timestamp.strip())
    if match:
        hours, minutes, seconds, fraction = match.groups()
        value = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        if fraction:
            value += float(f"0.{fraction}")
        return float(value)
    try:
        return float(timestamp)
    except ValueError as exc:
        raise ValueError(f"timestamp must be HH:MM:SS or a number of seconds: {timestamp}") from exc


def _steps(
    times: Sequence[float], positions: Sequence[Sequence[float]], reversal_min_speed: float
) -> dict[str, float | int]:
    speeds: list[float] = []
    vectors: list[tuple[float, float, float]] = []
    path = 0.0
    reversals = 0
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if dt <= 0:
            continue
        dx = float(positions[index][0] - positions[index - 1][0])
        dy = float(positions[index][1] - positions[index - 1][1])
        length = math.hypot(dx, dy)
        speed = length / dt
        path += length
        speeds.append(speed)
        vectors.append((dx, dy, speed))
    for index in range(1, len(vectors)):
        previous, current = vectors[index - 1], vectors[index]
        if previous[2] < reversal_min_speed or current[2] < reversal_min_speed:
            continue
        if previous[0] * current[0] + previous[1] * current[1] < 0:
            reversals += 1
    return {
        "path_length": path,
        "mean_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "max_speed": max(speeds) if speeds else 0.0,
        "speed_std": _population_std(speeds),
        "reversals": reversals,
    }


def _longest_still(
    times: Sequence[float], positions: Sequence[Sequence[float]], radius: float
) -> float:
    best = 0.0
    for start in range(len(times)):
        for end in range(start, len(times)):
            if distance(positions[start], positions[end]) > radius:
                break
            best = max(best, times[end] - times[start])
    return best


def _shelf_counts(zones: Sequence[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    previous: str | None = None
    for zone in zones:
        shelf = shelf_key(zone)
        if shelf is None:
            previous = None
            continue
        if shelf != previous:
            counts[shelf] = counts.get(shelf, 0) + 1
            previous = shelf
    return counts


def _shelf_handling_seconds(
    times: Sequence[float],
    positions: Sequence[Sequence[float]],
    zones: Sequence[str | None],
    settings: BehaviourSettings,
) -> float:
    """Longest stretch of small in-shelf movements, in seconds."""
    interaction = settings.object_interaction
    longest = 0.0
    current = 0.0
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if dt <= 0:
            continue
        speed = distance(positions[index - 1], positions[index]) / dt
        in_shelf = shelf_key(zones[index - 1]) is not None and shelf_key(zones[index]) is not None
        handling = interaction.min_handling_speed_px_per_s <= speed <= interaction.max_handling_speed_px_per_s
        if in_shelf and handling and shelf_key(zones[index - 1]) == shelf_key(zones[index]):
            current += dt
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def _shrink_ratio(records: Sequence[MovementRecord], min_samples: int) -> float:
    areas = [bbox_area(record.bbox) for record in records if record.bbox is not None]
    if len(areas) < min_samples:
        return 1.0
    midpoint = len(areas) // 2
    early = _median(areas[:midpoint])
    late_min = min(areas[midpoint:])
    if early <= 0:
        return 1.0
    return late_min / early


def _object_overlap(
    records: Sequence[MovementRecord],
    detections_by_frame: dict[int, list[dict[str, Any]]],
    settings: BehaviourSettings,
) -> dict[str, int | bool]:
    interaction = settings.object_interaction
    concealment = settings.possible_concealment
    near_flags: list[bool] = []
    class_hits: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        nearby = _nearby_classes(record, detections_by_frame.get(record.frame_id, []), interaction)
        near_flags.append(bool(nearby))
        for name in nearby:
            class_hits.setdefault(name, []).append(index)
    lost = False
    for hits in class_hits.values():
        if len(hits) < concealment.min_overlap_frames:
            continue
        last_hit = hits[-1]
        absence = concealment.absence_frames
        if last_hit + absence >= len(records):
            continue
        later = set(hits)
        if all(frame not in later for frame in range(last_hit + 1, last_hit + 1 + absence)):
            lost = True
            break
    return {"frames": sum(near_flags), "lost": lost}


def _nearby_classes(
    record: MovementRecord,
    detections: Sequence[dict[str, Any]],
    interaction: Any,
) -> set[str]:
    names: set[str] = set()
    for detection in detections:
        class_name = detection.get("class")
        if not isinstance(class_name, str) or class_name.strip().lower() == "person":
            continue
        confidence = detection.get("confidence", 1)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            continue
        if float(confidence) < interaction.min_object_confidence:
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if _object_is_near(record, bbox, interaction):
            names.add(class_name.strip().lower())
    return names


def _object_is_near(record: MovementRecord, bbox: Sequence[float], interaction: Any) -> bool:
    if record.bbox is not None and iou(record.bbox, bbox) >= interaction.min_object_iou:
        return True
    return distance(record.position, bbox_center(bbox)) <= interaction.object_distance_px


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _evidence_spans(
    times: Sequence[float],
    positions: Sequence[Sequence[float]],
    zones: Sequence[str | None],
    records: Sequence[MovementRecord],
    detections_by_frame: dict[int, list[dict[str, Any]]],
    settings: BehaviourSettings,
) -> dict[str, tuple[float, float]]:
    """Time span of the evidence behind each indicator.

    A missing span means the assessment cannot place that indicator in a
    window. The detector scores do not read this dictionary.
    """
    spans: dict[str, tuple[float, float]] = {}
    still = _still_span(times, positions, settings.loitering.radius_px)
    if still is not None:
        spans["loitering"] = still
    repeats = _repeat_span(times, zones)
    if repeats is not None:
        spans["repeated_shelf_visit"] = repeats
    abnormal = _abnormal_span(
        times,
        positions,
        settings.abnormal_movement.high_speed_px_per_s,
        settings.abnormal_movement.reversal_min_speed_px_per_s,
    )
    if abnormal is not None:
        spans["abnormal_movement"] = abnormal
    interaction = _interaction_span(times, positions, zones, records, detections_by_frame, settings)
    if interaction is not None:
        spans["object_interaction"] = interaction
    concealment = _concealment_span(times, records, detections_by_frame, settings)
    if concealment is not None:
        spans["possible_concealment"] = concealment
    return spans


def _still_span(
    times: Sequence[float], positions: Sequence[Sequence[float]], radius: float
) -> tuple[float, float] | None:
    best = 0.0
    span: tuple[float, float] | None = None
    for start in range(len(times)):
        for end in range(start, len(times)):
            if distance(positions[start], positions[end]) > radius:
                break
            duration = times[end] - times[start]
            if duration > best:
                best = duration
                span = (times[start], times[end])
    return span


def _repeat_span(times: Sequence[float], zones: Sequence[str | None]) -> tuple[float, float] | None:
    counts: dict[str, int] = {}
    previous: str | None = None
    returns: list[float] = []
    for index, zone in enumerate(zones):
        shelf = shelf_key(zone)
        if shelf is None:
            previous = None
            continue
        if shelf != previous:
            counts[shelf] = counts.get(shelf, 0) + 1
            if counts[shelf] >= 2:
                returns.append(times[index])
            previous = shelf
    if not returns:
        return None
    return (returns[0], returns[-1])


def _abnormal_span(
    times: Sequence[float],
    positions: Sequence[Sequence[float]],
    high_speed: float,
    reversal_min_speed: float,
) -> tuple[float, float] | None:
    moments: list[float] = []
    vectors: list[tuple[float, float, float, float, float]] = []
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if dt <= 0:
            continue
        dx = float(positions[index][0] - positions[index - 1][0])
        dy = float(positions[index][1] - positions[index - 1][1])
        speed = math.hypot(dx, dy) / dt
        vectors.append((dx, dy, speed, times[index - 1], times[index]))
        if speed >= high_speed:
            moments.extend((times[index - 1], times[index]))
    for index in range(1, len(vectors)):
        previous, current = vectors[index - 1], vectors[index]
        if previous[2] < reversal_min_speed or current[2] < reversal_min_speed:
            continue
        if previous[0] * current[0] + previous[1] * current[1] < 0:
            moments.extend((previous[3], current[4]))
    if not moments:
        return None
    return (min(moments), max(moments))


def _interaction_span(
    times: Sequence[float],
    positions: Sequence[Sequence[float]],
    zones: Sequence[str | None],
    records: Sequence[MovementRecord],
    detections_by_frame: dict[int, list[dict[str, Any]]],
    settings: BehaviourSettings,
) -> tuple[float, float] | None:
    rule = settings.object_interaction
    parts: list[tuple[float, float]] = []
    handling = _handling_span(times, positions, zones, settings)
    if handling is not None and handling[1] - handling[0] >= rule.min_handling_seconds:
        parts.append(handling)
    near_times = [
        times[index]
        for index, record in enumerate(records)
        if _nearby_classes(record, detections_by_frame.get(record.frame_id, []), rule)
    ]
    if len(near_times) >= rule.min_overlap_frames:
        parts.append((near_times[0], near_times[-1]))
    return _merge_spans(parts)


def _handling_span(
    times: Sequence[float],
    positions: Sequence[Sequence[float]],
    zones: Sequence[str | None],
    settings: BehaviourSettings,
) -> tuple[float, float] | None:
    interaction = settings.object_interaction
    longest = 0.0
    current = 0.0
    current_start = 0.0
    best: tuple[float, float] | None = None
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if dt <= 0:
            continue
        speed = distance(positions[index - 1], positions[index]) / dt
        same_shelf = shelf_key(zones[index - 1]) is not None and shelf_key(zones[index - 1]) == shelf_key(
            zones[index]
        )
        handling = interaction.min_handling_speed_px_per_s <= speed <= interaction.max_handling_speed_px_per_s
        if same_shelf and handling:
            if current == 0.0:
                current_start = times[index - 1]
            current += dt
            if current > longest:
                longest = current
                best = (current_start, times[index])
        else:
            current = 0.0
    return best


def _concealment_span(
    times: Sequence[float],
    records: Sequence[MovementRecord],
    detections_by_frame: dict[int, list[dict[str, Any]]],
    settings: BehaviourSettings,
) -> tuple[float, float] | None:
    rule = settings.possible_concealment
    interaction = settings.object_interaction
    parts: list[tuple[float, float]] = []
    samples = [
        (times[index], bbox_area(record.bbox))
        for index, record in enumerate(records)
        if record.bbox is not None
    ]
    if len(samples) >= rule.min_bbox_samples:
        midpoint = len(samples) // 2
        early = _median([area for _, area in samples[:midpoint]])
        late = samples[midpoint:]
        smallest = min(late, key=lambda item: item[1])
        if early > 0 and smallest[1] / early <= rule.shrink_ratio:
            start = late[0][0]
            end = smallest[0]
            parts.append((start, end) if end >= start else (end, start))
    hits_by_class: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        for name in _nearby_classes(record, detections_by_frame.get(record.frame_id, []), interaction):
            hits_by_class.setdefault(name, []).append(index)
    for hits in hits_by_class.values():
        if len(hits) < rule.min_overlap_frames:
            continue
        last_hit = hits[-1]
        absence = rule.absence_frames
        if last_hit + absence >= len(records):
            continue
        later = set(hits)
        if all(frame not in later for frame in range(last_hit + 1, last_hit + 1 + absence)):
            parts.append((times[last_hit], times[last_hit + absence]))
    return _merge_spans(parts)


def _merge_spans(spans: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if not spans:
        return None
    return (min(span[0] for span in spans), max(span[1] for span in spans))


def _population_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
