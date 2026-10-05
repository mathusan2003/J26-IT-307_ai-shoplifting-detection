"""MOT evaluation utilities for Component 2 research experiments.

Both arguments to :func:`evaluate_tracking` are sequences of the tracking
objects described in ``docs/api-contract.md``. Ground-truth tracks use the
same shape as predictions: ``person_id``, ``bbox``, and ``center``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Iterable, Sequence

from .utils import as_bbox, iou


@dataclass
class TrackingMetrics:
    ground_truth_detections: int
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    idf1: float
    mota: float
    occlusion_events: int
    recovered_occlusions: int
    occlusion_recovery_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _tracks_by_frame(sequence: Sequence[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    frames: dict[int, list[dict[str, object]]] = {}
    for frame in sequence:
        frame_id = int(frame["frame_id"])
        if frame_id in frames:
            raise ValueError(f"Duplicate frame_id: {frame_id}")
        frames[frame_id] = list(frame.get("tracks", []))
    return frames


def _match_frame(
    ground_truth: Sequence[dict[str, object]], predicted: Sequence[dict[str, object]], threshold: float
) -> tuple[list[tuple[dict[str, object], dict[str, object]]], int, int]:
    """Greedily match boxes by IoU; one box may participate only once."""
    candidates = sorted(
        ((iou(as_bbox(gt["bbox"]), as_bbox(pred["bbox"])), gt_index, pred_index)
        for gt_index, gt in enumerate(ground_truth) for pred_index, pred in enumerate(predicted)),
        reverse=True,
    )
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[dict[str, object], dict[str, object]]] = []
    for overlap, gt_index, pred_index in candidates:
        if overlap < threshold:
            break
        if gt_index not in used_gt and pred_index not in used_predictions:
            used_gt.add(gt_index)
            used_predictions.add(pred_index)
            matches.append((ground_truth[gt_index], predicted[pred_index]))
    return matches, len(predicted) - len(used_predictions), len(ground_truth) - len(used_gt)


def _identity_true_positives(counts: dict[tuple[int, int], int]) -> int:
    """Find the one-to-one GT/predicted identity assignment with most matches."""
    ground_truth_ids = sorted({pair[0] for pair in counts})
    prediction_ids = sorted({pair[1] for pair in counts})
    size = max(len(ground_truth_ids), len(prediction_ids), 1)
    maximum = max(counts.values(), default=0)
    # Hungarian algorithm minimizing (maximum - match_count), padded with zeros.
    cost = [[maximum] * (size + 1) for _ in range(size + 1)]
    for row, gt_id in enumerate(ground_truth_ids, start=1):
        for column, prediction_id in enumerate(prediction_ids, start=1):
            cost[row][column] = maximum - counts.get((gt_id, prediction_id), 0)
    u, v, matched_column, way = [0] * (size + 1), [0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
    for row in range(1, size + 1):
        matched_column[0], column0 = row, 0
        min_value, used = [float("inf")] * (size + 1), [False] * (size + 1)
        while True:
            used[column0] = True
            row0, delta, next_column = matched_column[column0], float("inf"), 0
            for column in range(1, size + 1):
                if not used[column]:
                    current = cost[row0][column] - u[row0] - v[column]
                    if current < min_value[column]:
                        min_value[column], way[column] = current, column0
                    if min_value[column] < delta:
                        delta, next_column = min_value[column], column
            for column in range(size + 1):
                if used[column]:
                    u[matched_column[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = next_column
            if matched_column[column0] == 0:
                break
        while True:
            previous_column = way[column0]
            matched_column[column0] = matched_column[previous_column]
            column0 = previous_column
            if column0 == 0:
                break
    return sum(
        counts.get((ground_truth_ids[row - 1], prediction_ids[column - 1]), 0)
        for column, row in enumerate(matched_column[1:], start=1)
        if row <= len(ground_truth_ids) and column <= len(prediction_ids)
    )


def evaluate_tracking(
    predictions: Sequence[dict[str, object]],
    ground_truth: Sequence[dict[str, object]],
    iou_threshold: float = 0.5,
    occlusion_min_frames: int = 15,
) -> TrackingMetrics:
    """Calculate clear, report-ready MOT metrics for one ordered sequence.

    IDF1 here uses detection-level identity matches (IDTP) over all frames;
    predictions must use the persistent IDs produced by this component.
    Occlusion recovery measures GT identities that are unmatched for at least
    ``occlusion_min_frames`` then matched again with their pre-gap tracker ID.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if occlusion_min_frames < 1:
        raise ValueError("occlusion_min_frames must be at least 1")
    predicted_by_frame = _tracks_by_frame(predictions)
    truth_by_frame = _tracks_by_frame(ground_truth)
    true_positives = false_positives = false_negatives = id_switches = 0
    previous_prediction_id: dict[int, int] = {}
    identity_matches: defaultdict[tuple[int, int], int] = defaultdict(int)
    missing_streak: defaultdict[int, int] = defaultdict(int)
    pre_occlusion_id: dict[int, int] = {}
    occlusion_events = recovered = 0

    for frame_id in sorted(set(predicted_by_frame) | set(truth_by_frame)):
        truth = truth_by_frame.get(frame_id, [])
        predicted = predicted_by_frame.get(frame_id, [])
        matches, fp, fn = _match_frame(truth, predicted, iou_threshold)
        true_positives += len(matches)
        false_positives += fp
        false_negatives += fn
        matched_truth_ids = {int(gt["person_id"]) for gt, _ in matches}
        for gt in truth:
            identity = int(gt["person_id"])
            if identity not in matched_truth_ids:
                missing_streak[identity] += 1
                if missing_streak[identity] == occlusion_min_frames and identity in previous_prediction_id:
                    pre_occlusion_id[identity] = previous_prediction_id[identity]
                    occlusion_events += 1
        for gt, prediction in matches:
            identity, predicted_id = int(gt["person_id"]), int(prediction["person_id"])
            identity_matches[(identity, predicted_id)] += 1
            if identity in pre_occlusion_id:
                recovered += int(pre_occlusion_id.pop(identity) == predicted_id)
            elif identity in previous_prediction_id and previous_prediction_id[identity] != predicted_id:
                id_switches += 1
            previous_prediction_id[identity] = predicted_id
            missing_streak[identity] = 0

    identity_tp = _identity_true_positives(identity_matches)
    predicted_detections = true_positives + false_positives
    total_truth = true_positives + false_negatives
    identity_fp, identity_fn = predicted_detections - identity_tp, total_truth - identity_tp
    idf1_denominator = 2 * identity_tp + identity_fp + identity_fn
    idf1 = 2 * identity_tp / idf1_denominator if idf1_denominator else 0.0
    mota = 1 - (false_positives + false_negatives + id_switches) / total_truth if total_truth else 0.0
    return TrackingMetrics(total_truth, true_positives, false_positives, false_negatives, id_switches,
                           idf1, mota, occlusion_events, recovered,
                           recovered / occlusion_events if occlusion_events else 0.0)


def measure_fps(process_frame: Callable[[object], object], frames: Iterable[object], warmup_frames: int = 10) -> float:
    """Measure processing FPS after optional warm-up frames."""
    iterator = iter(frames)
    for _ in range(warmup_frames):
        try:
            process_frame(next(iterator))
        except StopIteration:
            return 0.0
    count = 0
    start = perf_counter()
    for frame in iterator:
        process_frame(frame)
        count += 1
    elapsed = perf_counter() - start
    return count / elapsed if elapsed and count else 0.0
