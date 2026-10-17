"""Synthetic labelled tracks for training and evaluation.

The repository has no retail CCTV and no public shoplifting labels.
These sequences are generated from known movement patterns so the behaviour
model and the rule detectors can be trained and scored repeatably.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

MODULE_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = MODULE_ROOT / "datasets" / "labelled_sequences.json"

LABEL_NAMES = (
    "suspicious",
    "loitering",
    "repeated_shelf_visit",
    "abnormal_movement",
    "object_interaction",
    "possible_concealment",
)

_ZERO = {name: 0 for name in LABEL_NAMES}


def build_labelled_sequences(seed: int = 13) -> list[dict[str, Any]]:
    """Return 80 labelled sequences.

    Within each pattern, every fourth copy is the test split. The rest
    are the training split. That keeps every behaviour in both splits.
    """
    specs: list[tuple[str, int, dict[str, int], Callable[[random.Random], tuple[list, list]]]] = [
        ("normal_walk", 10, _labels(), _normal_walk),
        ("normal_browse", 10, _labels(), _normal_browse),
        ("normal_hurried", 10, _labels(), _normal_hurried),
        ("normal_shelf_pass", 10, _labels(), _normal_shelf_pass),
        ("loitering", 8, _labels(suspicious=1, loitering=1), _loitering),
        ("repeated_shelf_visit", 8, _labels(suspicious=1, repeated_shelf_visit=1), _repeated_shelf),
        ("abnormal_movement", 8, _labels(suspicious=1, abnormal_movement=1), _abnormal),
        ("object_interaction", 8, _labels(suspicious=1, object_interaction=1), _interaction),
        (
            "possible_concealment",
            8,
            _labels(suspicious=1, object_interaction=1, possible_concealment=1),
            _concealment,
        ),
    ]
    sequences: list[dict[str, Any]] = []
    index = 0
    for family, copies, labels, builder in specs:
        for copy in range(copies):
            frames, detections = builder(random.Random(seed + index * 17 + copy))
            sequences.append(
                {
                    "id": f"{family}_{copy:02d}",
                    "family": family,
                    "split": "test" if copy % 4 == 0 else "train",
                    "labels": labels,
                    "frames": frames,
                    "detections": detections,
                }
            )
            index += 1
    return sequences


def write_dataset(path: str | Path | None = None, seed: int = 13) -> Path:
    """Write the labelled sequences as JSON."""
    file_path = Path(path) if path is not None else DATASET_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = build_labelled_sequences(seed)
    file_path.write_text(json.dumps(sequences, indent=2) + "\n", encoding="utf-8")
    return file_path


def load_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    file_path = Path(path) if path is not None else DATASET_PATH
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("labelled dataset must be a JSON list")
    return payload


def member2_contract_loiter_frames(person_id: int = 12) -> list[dict[str, Any]]:
    """Tracking JSON in Member 2's contract shape for a 15 second shelf stand.

    There is no timestamp and no zone. Time comes from ``frame_id / fps``
    and the shelf comes from the box foot point.
    """
    frames = []
    for index in range(16):
        bbox = [710, 100, 790, 300]
        frames.append(
            {
                "frame_id": index * 25,
                "tracks": [
                    {
                        "person_id": person_id,
                        "bbox": bbox,
                        "center": [750, 200],
                    }
                ],
            }
        )
    return frames


def _labels(**overrides: int) -> dict[str, int]:
    labels = dict(_ZERO)
    labels.update(overrides)
    return labels


def _normal_walk(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = []
    for second in range(16):
        frames.append(_person(second, 50 + 80 * second, 500, None, 200, rng))
    return frames, []


def _normal_browse(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = [
        _person(second, 250 + 25 * second, 220, "Aisle 1", 180, rng) for second in range(12)
    ]
    return frames, []


def _normal_hurried(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = [
        _person(second, foot_x, 200, "Aisle 1", 180, rng, jitter=0.0)
        for second, foot_x in enumerate((240, 380, 520, 600))
    ]
    return frames, []


def _normal_shelf_pass(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan = (
        (0, 400, 400, "Aisle 1", 180),
        (1, 550, 350, "Aisle 1", 180),
        (2, 700, 300, "Shelf A", 200),
        (3, 710, 300, "Shelf A", 200),
        (4, 560, 350, "Aisle 1", 180),
        (5, 420, 400, "Aisle 1", 180),
    )
    frames = [
        _person(second, x_pos, y_pos, zone, height, rng, jitter=0.0)
        for second, x_pos, y_pos, zone, height in plan
    ]
    return frames, []


def _loitering(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = [_person(second, 750, 300, "Shelf A", 200, rng, jitter=0.2) for second in range(21)]
    return frames, []


def _repeated_shelf(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan = [
        (0, 750, 300, "Shelf A", 200),
        (1, 751, 300, "Shelf A", 200),
        (2, 600, 300, "Aisle 1", 180),
        (3, 450, 300, "Aisle 1", 180),
        (4, 600, 300, "Aisle 1", 180),
        (5, 750, 300, "Shelf A", 200),
        (6, 751, 300, "Shelf A", 200),
        (7, 600, 300, "Aisle 1", 180),
        (8, 450, 300, "Aisle 1", 180),
        (9, 600, 300, "Aisle 1", 180),
        (10, 750, 300, "Shelf A", 200),
        (11, 751, 300, "Shelf A", 200),
    ]
    frames = [
        _person(second, x_pos, y_pos, zone, height, rng, jitter=0.0)
        for second, x_pos, y_pos, zone, height in plan
    ]
    return frames, []


def _abnormal(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = []
    for second in range(7):
        foot_x = 250 if second % 2 == 0 else 620
        frames.append(_person(second, foot_x, 200, "Aisle 1", 180, rng, jitter=0.0))
    return frames, []


def _interaction(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = []
    for second in range(9):
        foot_x = 750 if second % 2 == 0 else 762
        frames.append(_person(second, foot_x, 300, "Shelf A", 200, rng, jitter=0.0))
    return frames, []


def _concealment(rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = []
    detections = []
    for second in range(12):
        height = 200 if second < 6 else 80
        frames.append(_person(second, 750, 300, "Shelf A", height, rng, jitter=0.0))
        objects = []
        if second < 6:
            objects.append(
                {"class": "bottle", "confidence": 0.91, "bbox": [735, 175, 765, 215]}
            )
        detections.append({"frame_id": second, "detections": objects})
    return frames, detections


def _person(
    second: int,
    foot_x: float,
    foot_y: float,
    zone: str | None,
    height: float,
    rng: random.Random,
    jitter: float = 0.2,
    person_id: int = 1,
) -> dict[str, Any]:
    shift = rng.uniform(-jitter, jitter) if jitter else 0.0
    width = 70.0
    x_pos = foot_x + shift
    bbox = [x_pos - width / 2, foot_y - height, x_pos + width / 2, foot_y]
    center = [round((bbox[0] + bbox[2]) / 2, 4), round((bbox[1] + bbox[3]) / 2, 4)]
    track: dict[str, Any] = {
        "person_id": person_id,
        "bbox": [round(value, 4) for value in bbox],
        "center": center,
    }
    if zone is not None:
        track["zone"] = zone
    return {
        "frame_id": second,
        "timestamp": f"00:00:{second:02d}",
        "tracks": [track],
    }
