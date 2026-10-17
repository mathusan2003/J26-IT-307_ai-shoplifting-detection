"""Train the behaviour model and score it against the labelled sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from behaviour import analyse_tracking_frames
from config import BehaviourSettings, load_settings, load_zones
from dataset import LABEL_NAMES, load_dataset
from detectors import INDICATOR_TYPES
from features import build_profile, index_detections
from model import LogisticBehaviourModel, binary_metrics, save_model
from movement_history import MovementHistoryStore

MODULE_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = MODULE_ROOT / "evaluation" / "metrics.json"


def feature_rows(
    sequences: list[dict[str, Any]],
    settings: BehaviourSettings,
    zones: dict[str, list[list[float]]],
) -> tuple[list[list[float]], list[int]]:
    """Return model features and the suspicious label for each sequence."""
    rows: list[list[float]] = []
    labels: list[int] = []
    for sequence in sequences:
        store = MovementHistoryStore()
        store.ingest_frames(sequence["frames"])
        profile = build_profile(
            store.records_for(store.person_ids()[0]),
            index_detections(sequence.get("detections")),
            settings,
            zones,
        )
        rows.append(profile.as_vector())
        labels.append(int(sequence["labels"]["suspicious"]))
    return rows, labels


def rule_predictions(sequence: dict[str, Any], settings: BehaviourSettings, zones: dict[str, list[list[float]]]) -> dict[str, int]:
    """Return 1 for each indicator the rules emit for one sequence."""
    results = analyse_tracking_frames(
        sequence["frames"], sequence.get("detections"), settings, zones
    )
    fired = {item["type"] for item in results[0]["behaviours"]} if results else set()
    predicted = {name: 0 for name in LABEL_NAMES}
    predicted["suspicious"] = int(bool(fired))
    for name in INDICATOR_TYPES:
        predicted[name] = int(name in fired)
    return predicted


def evaluate_split(
    sequences: list[dict[str, Any]],
    settings: BehaviourSettings,
    zones: dict[str, list[list[float]]],
    model: LogisticBehaviourModel | None = None,
) -> dict[str, Any]:
    """Score rules, and the model when one is supplied."""
    rule_truth = {name: [] for name in LABEL_NAMES}
    rule_guess = {name: [] for name in LABEL_NAMES}
    for sequence in sequences:
        predicted = rule_predictions(sequence, settings, zones)
        for name in LABEL_NAMES:
            rule_truth[name].append(int(sequence["labels"][name]))
            rule_guess[name].append(predicted[name])
    report: dict[str, Any] = {
        "sequences": len(sequences),
        "rules": {name: binary_metrics(rule_truth[name], rule_guess[name]) for name in LABEL_NAMES},
    }
    if model is not None and sequences:
        rows, truth = feature_rows(sequences, settings, zones)
        guessed = [model.predict(row) for row in rows]
        report["model"] = binary_metrics(truth, guessed)
    return report


def train_and_evaluate(
    sequences: list[dict[str, Any]] | None = None,
    settings: BehaviourSettings | None = None,
    zones: dict[str, list[list[float]]] | None = None,
) -> tuple[LogisticBehaviourModel, dict[str, Any]]:
    """Train on the train split and score both rules and the model on test."""
    active_settings = settings if settings is not None else load_settings()
    active_zones = zones if zones is not None else load_zones()
    data = sequences if sequences is not None else load_dataset()
    train = [item for item in data if item["split"] == "train"]
    test = [item for item in data if item["split"] == "test"]
    rows, labels = feature_rows(train, active_settings, active_zones)
    model = LogisticBehaviourModel(active_settings.model)
    model.fit(rows, labels)
    report = {
        "dataset": {
            "source": "synthetic labelled tracks",
            "limitation": (
                "No retail CCTV or public shoplifting labels are stored in this "
                "repository. Metrics describe the synthetic set only."
            ),
            "sequences": len(data),
            "train_sequences": len(train),
            "test_sequences": len(test),
            "train_suspicious": sum(item["labels"]["suspicious"] for item in train),
            "test_suspicious": sum(item["labels"]["suspicious"] for item in test),
            "features": list(model.feature_names),
            "label_names": list(LABEL_NAMES),
        },
        "model_settings": {
            "type": "l2_logistic_regression",
            "epochs": model.epochs,
            "learning_rate": model.learning_rate,
            "l2": model.l2,
            "threshold": model.threshold,
        },
        "train": evaluate_split(train, active_settings, active_zones, model),
        "test": evaluate_split(test, active_settings, active_zones, model),
        "leave_one_family_out": leave_one_family_out(data, active_settings, active_zones),
    }
    return model, report


def leave_one_family_out(
    sequences: list[dict[str, Any]],
    settings: BehaviourSettings,
    zones: dict[str, list[list[float]]],
) -> dict[str, Any]:
    """Train without one suspicious pattern and measure whether it is still recognised.

    The held-out family is entirely positive. Recall is the fraction of those
    tracks the model still calls suspicious. Specificity is measured on the
    ordinary tracks, which stay in every training run.
    """
    families = (
        "loitering",
        "repeated_shelf_visit",
        "abnormal_movement",
        "object_interaction",
        "possible_concealment",
    )
    normals = [item for item in sequences if item["labels"]["suspicious"] == 0]
    report: dict[str, Any] = {}
    for family in families:
        held = [item for item in sequences if item["family"] == family]
        train = [item for item in sequences if item["family"] != family]
        rows, labels = feature_rows(train, settings, zones)
        model = LogisticBehaviourModel(settings.model)
        model.fit(rows, labels)
        held_rows, held_labels = feature_rows(held, settings, zones)
        normal_rows, normal_labels = feature_rows(normals, settings, zones)
        held_metrics = binary_metrics(held_labels, [model.predict(row) for row in held_rows])
        normal_metrics = binary_metrics(normal_labels, [model.predict(row) for row in normal_rows])
        report[family] = {
            "held_out_sequences": len(held),
            "held_out_recall": held_metrics["recall"],
            "normal_specificity": normal_metrics["specificity"],
        }
    return report


def write_metrics(report: dict[str, Any], path: str | Path | None = None) -> Path:
    file_path = Path(path) if path is not None else METRICS_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return file_path
