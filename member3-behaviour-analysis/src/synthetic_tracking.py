"""Deterministic synthetic tracking fixtures for the existing behaviour pipeline.

Tracking files use Member 2's contract: frame_id, person_id, bbox, and center.
They do not contain zone names or behaviour labels. Time is frame_id / fps
with fps 25, the rate in configs/behaviour_config.json. The foot point is the
bottom centre of the box and is placed inside configs/zones.json.

Labels are written in a separate file from the motion plan. They are not
copied from detector output. None of these records are CCTV observations.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = MODULE_ROOT / "samples" / "synthetic_scenarios"
TRAINING_ROOT = MODULE_ROOT / "datasets" / "synthetic_behaviour"

FPS = 25
BEHAVIOURS = (
    "loitering",
    "repeated_shelf_visit",
    "abnormal_movement",
    "object_interaction",
    "possible_concealment",
)

SCENARIOS = (
    "normal_shopping",
    "loitering",
    "repeated_shelf_visits",
    "abnormal_movement",
    "possible_concealment",
)


class SyntheticInputError(ValueError):
    """A synthetic scenario is missing the evidence the motion plan requires."""


@dataclass(frozen=True)
class SyntheticSettings:
    """Knobs for the motion plans. The default seed rebuilds the saved fixtures."""

    seed: int = 13
    fps: int = FPS
    jitter_px: float = 0.0
    loiter_samples: int = 16
    person_ids: tuple[int, int, int, int, int] = (11, 12, 13, 14, 15)


@dataclass(frozen=True)
class SyntheticSequence:
    sequence_id: str
    scenario: str
    person_id: int
    tracking: list[dict[str, Any]]
    labels: dict[str, Any]
    detections: list[dict[str, Any]] | None


def generate_dataset(settings: SyntheticSettings | None = None) -> list[SyntheticSequence]:
    """Return the five scenarios. The same settings rebuild the same JSON."""
    active = settings if settings is not None else SyntheticSettings()
    if active.fps <= 0:
        raise SyntheticInputError("fps must be positive")
    if active.loiter_samples < 16:
        raise SyntheticInputError("loitering needs at least 16 samples so the span can reach 15 seconds")
    builders = (
        _normal_shopping,
        _loitering,
        _repeated_shelf_visits,
        _abnormal_movement,
        _possible_concealment,
    )
    return [builder(active, person_id) for builder, person_id in zip(builders, active.person_ids)]


def write_dataset(settings: SyntheticSettings | None = None, root: Path | None = None) -> Path:
    """Write tracking, labels, and the concealment detection file under samples/."""
    active = settings if settings is not None else SyntheticSettings()
    folder = root if root is not None else OUTPUT_ROOT
    tracking_dir = folder / "tracking"
    label_dir = folder / "labels"
    detection_dir = folder / "detections"
    for path in (tracking_dir, label_dir, detection_dir):
        path.mkdir(parents=True, exist_ok=True)
    sequences = generate_dataset(active)
    listing = []
    for sequence in sequences:
        tracking_name = f"tracking/{sequence.sequence_id}.json"
        label_name = f"labels/{sequence.sequence_id}.json"
        (folder / tracking_name).write_text(_dump(sequence.tracking), encoding="utf-8")
        (folder / label_name).write_text(_dump(sequence.labels), encoding="utf-8")
        detection_name = None
        if sequence.detections is not None:
            detection_name = f"detections/{sequence.sequence_id}.json"
            (folder / detection_name).write_text(_dump(sequence.detections), encoding="utf-8")
        listing.append(
            {
                "sequence_id": sequence.sequence_id,
                "scenario": sequence.scenario,
                "person_id": sequence.person_id,
                "tracking": tracking_name,
                "labels": label_name,
                "detections": detection_name,
            }
        )
    manifest = {
        "synthetic": True,
        "not_real_cctv": True,
        "purpose": (
            "Pipeline fixtures for the existing behaviour feature extractor and detectors. "
            "Not a real-world evaluation set and not a train/test split."
        ),
        "seed": active.seed,
        "jitter_px": active.jitter_px,
        "fps": active.fps,
        "time_convention": (
            "Tracking JSON has no timestamp. frame_id increases by fps once per second. "
            "Member 3 uses frame_id / fps when a timestamp is absent."
        ),
        "zone_convention": (
            "Tracking JSON has no zone field. The foot point is the bottom centre of bbox "
            "and is placed inside the polygons in configs/zones.json."
        ),
        "label_basis": "Motion plan written by this generator. Not detector output and not a human review.",
        "object_interaction": (
            "Possible concealment has a separate Member 1 detection file. "
            "A missing file or a file with only person boxes is rejected."
        ),
        "sequences": listing,
    }
    (folder / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    return folder


def generate_training_set(
    copies_per_scenario: int = 10,
    seed: int = 13,
    jitter_px: float = 1.5,
) -> list[SyntheticSequence]:
    """Return several jittered copies of each scenario.

    Each copy is one sequence with its own person_id. Labels come from the
    motion plan, not from the detectors. This set is for a synthetic-only
    classifier and is not mixed with the real-video Model 1 split.
    """
    if copies_per_scenario < 3:
        raise SyntheticInputError("a training set needs at least 3 copies per scenario for train, validation, and test")
    sequences: list[SyntheticSequence] = []
    person_id = 100
    for copy in range(copies_per_scenario):
        settings = SyntheticSettings(
            seed=seed + copy * 31,
            jitter_px=jitter_px,
            person_ids=(person_id, person_id + 1, person_id + 2, person_id + 3, person_id + 4),
        )
        for item in generate_dataset(settings):
            sequence_id = f"{item.scenario}_{copy:02d}"
            labels = dict(item.labels)
            labels["sequence_id"] = sequence_id
            labels["copy"] = copy
            sequences.append(
                SyntheticSequence(
                    sequence_id,
                    item.scenario,
                    item.person_id,
                    item.tracking,
                    labels,
                    item.detections,
                )
            )
        person_id += 5
    return sequences


def assign_sequence_splits(
    sequences: list[SyntheticSequence],
    copies_per_scenario: int,
) -> dict[str, str]:
    """Assign whole sequences. Frames from one sequence stay in one split."""
    train_n, val_n, _test_n = _split_sizes(copies_per_scenario)
    assigned: dict[str, str] = {}
    by_scenario: dict[str, list[SyntheticSequence]] = {name: [] for name in SCENARIOS}
    for item in sequences:
        by_scenario[item.scenario].append(item)
    for scenario, items in by_scenario.items():
        items.sort(key=lambda item: item.sequence_id)
        if len(items) != copies_per_scenario:
            raise SyntheticInputError(f"{scenario} has {len(items)} copies, expected {copies_per_scenario}")
        for index, item in enumerate(items):
            if index < train_n:
                assigned[item.sequence_id] = "train"
            elif index < train_n + val_n:
                assigned[item.sequence_id] = "validation"
            else:
                assigned[item.sequence_id] = "test"
    return assigned


def write_training_set(
    copies_per_scenario: int = 10,
    seed: int = 13,
    jitter_px: float = 1.5,
    root: Path | None = None,
) -> Path:
    """Write the training sequences under datasets/synthetic_behaviour/."""
    folder = root if root is not None else TRAINING_ROOT
    tracking_dir = folder / "tracking"
    label_dir = folder / "labels"
    detection_dir = folder / "detections"
    for path in (tracking_dir, label_dir, detection_dir):
        path.mkdir(parents=True, exist_ok=True)
    sequences = generate_training_set(copies_per_scenario, seed, jitter_px)
    splits = assign_sequence_splits(sequences, copies_per_scenario)
    listing = []
    for sequence in sequences:
        tracking_name = f"tracking/{sequence.sequence_id}.json"
        label_name = f"labels/{sequence.sequence_id}.json"
        (folder / tracking_name).write_text(_dump(sequence.tracking), encoding="utf-8")
        (folder / label_name).write_text(_dump(sequence.labels), encoding="utf-8")
        detection_name = None
        if sequence.detections is not None:
            detection_name = f"detections/{sequence.sequence_id}.json"
            (folder / detection_name).write_text(_dump(sequence.detections), encoding="utf-8")
        listing.append(
            {
                "sequence_id": sequence.sequence_id,
                "scenario": sequence.scenario,
                "person_id": sequence.person_id,
                "split": splits[sequence.sequence_id],
                "tracking": tracking_name,
                "labels": label_name,
                "detections": detection_name,
            }
        )
    listing.sort(key=lambda item: item["sequence_id"])
    manifest = {
        "synthetic": True,
        "not_real_cctv": True,
        "real_world_accuracy": False,
        "purpose": (
            "SYNTHETIC training set for the multi-label behaviour classifier. "
            "Not a real CCTV evaluation and not mixed with Model 1 video features."
        ),
        "seed": seed,
        "jitter_px": jitter_px,
        "fps": FPS,
        "copies_per_scenario": copies_per_scenario,
        "split_unit": "sequence_id",
        "split_rule": (
            "Within each scenario, sort sequence_id. The first block is train, "
            "the next copy is validation, and the last copies are test. "
            "Frames from one sequence stay in one split."
        ),
        "label_basis": "Motion plan written by this generator. Not detector output and not a human review.",
        "sequences": listing,
    }
    (folder / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    return folder


def _split_sizes(copies: int) -> tuple[int, int, int]:
    if copies < 3:
        raise SyntheticInputError("need at least 3 copies per scenario")
    val_n = 1
    test_n = max(1, copies - copies * 6 // 10 - val_n)
    train_n = copies - val_n - test_n
    if train_n < 1:
        train_n, val_n, test_n = copies - 2, 1, 1
    return train_n, val_n, test_n


def require_concealment_detections(detections: list[dict[str, Any]] | None) -> None:
    """Reject concealment evidence that is missing or person-only."""
    if not detections:
        raise SyntheticInputError(
            "possible concealment requires a Member 1 detection JSON file with a non-person object. "
            "Person coordinates alone are not object-interaction evidence."
        )
    found = False
    for index, frame in enumerate(detections):
        boxes = frame.get("detections")
        if not isinstance(boxes, list):
            raise SyntheticInputError(f"detection frame {index} is missing a detections list")
        for detection in boxes:
            if not isinstance(detection, dict):
                raise SyntheticInputError(f"detection frame {index} has an entry that is not an object")
            name = detection.get("class")
            if isinstance(name, str) and name.strip().lower() not in {"", "person"}:
                found = True
    if not found:
        raise SyntheticInputError(
            "possible concealment detection file has no non-person object. "
            "A person box is not object-interaction evidence."
        )


def _normal_shopping(settings: SyntheticSettings, person_id: int) -> SyntheticSequence:
    frames = [
        _frame(settings, index, person_id, 300 + index * 20, 500, 180)
        for index in range(12)
    ]
    return _sequence("normal_shopping", person_id, frames, settings, detections=None)


def _loitering(settings: SyntheticSettings, person_id: int) -> SyntheticSequence:
    frames = [
        _frame(settings, index, person_id, 750, 300, 200) for index in range(settings.loiter_samples)
    ]
    return _sequence("loitering", person_id, frames, settings, detections=None)


def _repeated_shelf_visits(settings: SyntheticSettings, person_id: int) -> SyntheticSequence:
    outbound = list(range(750, 419, -30))
    inbound = list(range(450, 751, 30))
    feet = outbound + inbound
    frames = [_frame(settings, index, person_id, foot_x, 300, 180) for index, foot_x in enumerate(feet)]
    return _sequence("repeated_shelf_visits", person_id, frames, settings, detections=None)


def _abnormal_movement(settings: SyntheticSettings, person_id: int) -> SyntheticSequence:
    frames = []
    for index in range(8):
        foot_x = 300 if index % 2 == 0 else 620
        frames.append(_frame(settings, index, person_id, foot_x, 200, 180))
    return _sequence("abnormal_movement", person_id, frames, settings, detections=None)


def _possible_concealment(settings: SyntheticSettings, person_id: int) -> SyntheticSequence:
    frames = []
    detections = []
    for index in range(12):
        height = 200 if index < 6 else 80
        frame = _frame(settings, index, person_id, 750, 300, height)
        frames.append(frame)
        objects: list[dict[str, Any]] = []
        if index < 6:
            objects.append(
                {
                    "class": "bottle",
                    "confidence": 0.91,
                    "bbox": [735, 150, 770, 210],
                }
            )
        detections.append({"frame_id": frame["frame_id"], "detections": objects})
    require_concealment_detections(detections)
    return _sequence("possible_concealment", person_id, frames, settings, detections=detections)


def _sequence(
    scenario: str,
    person_id: int,
    tracking: list[dict[str, Any]],
    settings: SyntheticSettings,
    detections: list[dict[str, Any]] | None,
) -> SyntheticSequence:
    if scenario == "possible_concealment":
        require_concealment_detections(detections)
    intended = {name: False for name in BEHAVIOURS}
    planned = {
        "loitering": ("loitering",),
        "repeated_shelf_visits": ("repeated_shelf_visit",),
        "abnormal_movement": ("abnormal_movement",),
        "possible_concealment": ("object_interaction", "possible_concealment"),
    }
    for name in planned.get(scenario, ()):
        intended[name] = True
    labels = {
        "record_type": "synthetic_scenario_label",
        "synthetic": True,
        "not_real_cctv": True,
        "not_a_reviewed_annotation": True,
        "sequence_id": scenario,
        "scenario": scenario,
        "person_id": person_id,
        "seed": settings.seed,
        "fps": settings.fps,
        "intended": intended,
        "label_basis": "motion plan written by the generator; not detector output and not a human review",
        "real_world_accuracy": False,
    }
    if scenario == "possible_concealment":
        labels["object_evidence"] = "synthetic Member 1 detection file with class bottle"
        labels["concealment_note"] = (
            "Simulated box shrink plus a bottle that is present and then absent. "
            "This is not a validated real-world concealment detector."
        )
    return SyntheticSequence(scenario, scenario, person_id, tracking, labels, detections)


def _frame(
    settings: SyntheticSettings,
    index: int,
    person_id: int,
    foot_x: float,
    foot_y: float,
    height: float,
) -> dict[str, Any]:
    shift_x, shift_y = _shift(settings, index)
    foot_x = int(round(foot_x + shift_x))
    foot_y = int(round(foot_y + shift_y))
    width = 80
    bbox = [foot_x - width // 2, foot_y - int(height), foot_x + width // 2, foot_y]
    center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
    return {
        "frame_id": index * settings.fps,
        "tracks": [
            {
                "person_id": person_id,
                "bbox": bbox,
                "center": center,
            }
        ],
    }


def _shift(settings: SyntheticSettings, index: int) -> tuple[float, float]:
    if settings.jitter_px <= 0:
        return 0.0, 0.0
    rng = random.Random(settings.seed + index * 17)
    return rng.uniform(-settings.jitter_px, settings.jitter_px), rng.uniform(-settings.jitter_px, settings.jitter_px)


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Write synthetic tracking fixtures. Not real CCTV.")
    parser.add_argument("--training", action="store_true", help="Write the training set under datasets/synthetic_behaviour/")
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    if args.training:
        folder = write_training_set(copies_per_scenario=args.copies, seed=args.seed)
    else:
        folder = write_dataset(SyntheticSettings(seed=args.seed))
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
