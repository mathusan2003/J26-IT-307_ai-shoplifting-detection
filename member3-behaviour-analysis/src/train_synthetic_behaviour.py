"""Train the SYNTHETIC multi-label behaviour classifier.

Labels come from the generator motion plan, not from the rule detectors.
The report is not a real-world accuracy claim and is not written into
evaluation/video_baseline_metrics.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from config import load_settings, load_zones
from features import FEATURE_NAMES, build_profile, index_detections
from movement_history import MovementHistoryStore
from synthetic_behaviour_model import (
    METRICS_PATH,
    MODEL_PATH,
    SCHEMA_PATH,
    SyntheticBehaviourClassifier,
    per_class_metrics,
    save_classifier,
)
from synthetic_tracking import (
    BEHAVIOURS,
    TRAINING_ROOT,
    assign_sequence_splits,
    generate_training_set,
    write_training_set,
)
from video_dataset import write_json

COPIES = 10
SEED = 13
JITTER = 1.5


def feature_row(sequence, settings, zones) -> list[float]:
    store = MovementHistoryStore()
    store.ingest_frames(sequence.tracking)
    records = store.records_for(sequence.person_id)
    profile = build_profile(records, index_detections(sequence.detections), settings, zones)
    return profile.as_vector()


def label_row(sequence) -> list[int]:
    intended = sequence.labels["intended"]
    return [int(bool(intended[name])) for name in BEHAVIOURS]


def train_and_evaluate(
    copies_per_scenario: int = COPIES,
    seed: int = SEED,
    jitter_px: float = JITTER,
) -> dict[str, Any]:
    settings = load_settings()
    zones = load_zones()
    sequences = generate_training_set(copies_per_scenario, seed, jitter_px)
    splits = assign_sequence_splits(sequences, copies_per_scenario)
    grouped: dict[str, list] = {"train": [], "validation": [], "test": []}
    for sequence in sequences:
        grouped[splits[sequence.sequence_id]].append(sequence)
    train_x = [feature_row(item, settings, zones) for item in grouped["train"]]
    train_y = [label_row(item) for item in grouped["train"]]
    started = time.perf_counter()
    model = SyntheticBehaviourClassifier()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started

    def score(items: list) -> dict[str, Any]:
        rows = [feature_row(item, settings, zones) for item in items]
        truth = [label_row(item) for item in items]
        guessed = [list(model.predict(row).values()) for row in rows]
        metrics = per_class_metrics(truth, guessed)
        metrics["sequences"] = [
            {
                "sequence_id": item.sequence_id,
                "scenario": item.scenario,
                "split": splits[item.sequence_id],
            }
            for item in items
        ]
        return metrics

    train_feature_mean = [sum(row[index] for row in train_x) / len(train_x) for index in range(len(FEATURE_NAMES))]
    report = {
        "task": "synthetic_tracking_behaviour_multilabel",
        "synthetic": True,
        "not_real_cctv": True,
        "real_world_accuracy": False,
        "label_source": "generator_motion_plan",
        "label_basis": "Motion plan written by src/synthetic_tracking.py. Not detector output and not a human review.",
        "mixed_with_model1_video_evaluation": False,
        "replaces_existing_behaviour_model": False,
        "annotation_format": "multi_label",
        "why_multi_label": (
            "The concealment scenario is labelled both object_interaction and possible_concealment. "
            "Normal shopping is the all-absent checklist. A single exclusive class would hide that overlap."
        ),
        "feature_names": list(FEATURE_NAMES),
        "behaviours": list(BEHAVIOURS),
        "seed": seed,
        "copies_per_scenario": copies_per_scenario,
        "split_unit": "sequence_id",
        "split_counts": {name: len(items) for name, items in grouped.items()},
        "scaler_fit_on": "train",
        "train_feature_mean": train_feature_mean,
        "hyperparameter_search_on_test": False,
        "fit_seconds": fit_seconds,
        "train": score(grouped["train"]),
        "validation": score(grouped["validation"]),
        "test": score(grouped["test"]),
        "model_path": str(MODEL_PATH),
        "schema_path": str(SCHEMA_PATH),
        "limitation": (
            "SYNTHETIC pipeline check. These metrics are not real-world accuracy "
            "and are not independent of the motion plans used to generate the tracks."
        ),
    }
    return model, report


def run(copies_per_scenario: int = COPIES, seed: int = SEED) -> dict[str, Any]:
    write_training_set(copies_per_scenario=copies_per_scenario, seed=seed, jitter_px=JITTER)
    model, report = train_and_evaluate(copies_per_scenario, seed, JITTER)
    save_classifier(model)
    write_json(METRICS_PATH, report)
    return report


def main() -> int:
    report = run()
    json.dump(
        {
            "synthetic": True,
            "real_world_accuracy": False,
            "split_counts": report["split_counts"],
            "test_macro_f1": report["test"]["macro_f1"],
            "test_exact_match_accuracy": report["test"]["exact_match_accuracy"],
            "test_per_class_f1": {name: block["f1"] for name, block in report["test"]["behaviours"].items()},
            "wrote_model": report["model_path"],
            "wrote_metrics": str(METRICS_PATH),
            "wrote_sequences": str(TRAINING_ROOT / "manifest.json"),
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
