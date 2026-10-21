"""Multi-label classifier for SYNTHETIC tracking sequences.

Each sequence has a motion-plan checklist. More than one behaviour can be
true at once (concealment includes object interaction). A multi-label model
matches that format. This file does not overwrite models/behaviour_model.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES
from model import binary_metrics
from synthetic_tracking import BEHAVIOURS

MODULE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = MODULE_ROOT / "models" / "synthetic_behaviour_classifier.json"
SCHEMA_PATH = MODULE_ROOT / "models" / "synthetic_behaviour_feature_schema.json"
METRICS_PATH = MODULE_ROOT / "evaluation" / "synthetic_behaviour_metrics.json"
THRESHOLD = 0.5
RANDOM_STATE = 13


class SyntheticBehaviourClassifier:
    """One-vs-rest logistic regression on the existing 15 trajectory features."""

    def __init__(self) -> None:
        self.feature_names = list(FEATURE_NAMES)
        self.behaviours = list(BEHAVIOURS)
        self.threshold = THRESHOLD
        self.scaler = StandardScaler()
        self.mean: list[float] = []
        self.scale: list[float] = []
        self.estimators: list[dict[str, Any]] = []

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[Sequence[int]]) -> None:
        if not rows:
            raise ValueError("the synthetic behaviour model needs training rows")
        matrix = np.asarray(rows, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int32)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature width does not match FEATURE_NAMES")
        if target.shape[1] != len(self.behaviours):
            raise ValueError("label width does not match the five behaviours")
        scaled = self.scaler.fit_transform(matrix)
        classifier = OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
        )
        classifier.fit(scaled, target)
        self.mean = self.scaler.mean_.astype(float).tolist()
        scale = self.scaler.scale_.astype(float).copy()
        scale[scale == 0] = 1.0
        self.scale = scale.tolist()
        self.estimators = [
            {
                "coef": estimator.coef_[0].astype(float).tolist(),
                "intercept": float(estimator.intercept_[0]),
            }
            for estimator in classifier.estimators_
        ]

    def predict_proba(self, row: Sequence[float]) -> dict[str, float]:
        if not self.estimators:
            raise ValueError("the synthetic behaviour model has not been fit")
        values = np.asarray(row, dtype=np.float64)
        scaled = (values - np.asarray(self.mean, dtype=np.float64)) / np.asarray(self.scale, dtype=np.float64)
        probabilities = {}
        for name, estimator in zip(self.behaviours, self.estimators):
            score = estimator["intercept"] + float(np.dot(estimator["coef"], scaled))
            probabilities[name] = _sigmoid(score)
        return probabilities

    def predict(self, row: Sequence[float]) -> dict[str, int]:
        probabilities = self.predict_proba(row)
        return {name: int(value >= self.threshold) for name, value in probabilities.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "one_vs_rest_logistic_regression",
            "library": "scikit-learn",
            "synthetic": True,
            "real_world_accuracy": False,
            "feature_names": self.feature_names,
            "behaviours": self.behaviours,
            "threshold": self.threshold,
            "C": 1.0,
            "random_state": RANDOM_STATE,
            "mean": self.mean,
            "scale": self.scale,
            "estimators": self.estimators,
            "replaces_existing_behaviour_model": False,
        }


def save_classifier(model: SyntheticBehaviourClassifier, path: Path | None = None) -> Path:
    file_path = path if path is not None else MODEL_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    schema = {
        "synthetic": True,
        "real_world_accuracy": False,
        "feature_names": model.feature_names,
        "behaviours": model.behaviours,
        "annotation_format": "multi_label",
        "why_multi_label": (
            "A sequence can have more than one intended behaviour. "
            "The concealment plan is labelled object_interaction and possible_concealment. "
            "Normal shopping is the all-absent case."
        ),
        "source_features": "src/features.py FEATURE_NAMES",
        "does_not_replace": "models/behaviour_model.json",
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return file_path


def load_classifier(path: Path | None = None) -> SyntheticBehaviourClassifier:
    file_path = path if path is not None else MODEL_PATH
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "one_vs_rest_logistic_regression":
        raise ValueError("file is not the synthetic multi-label classifier")
    model = SyntheticBehaviourClassifier()
    model.feature_names = list(payload["feature_names"])
    model.behaviours = list(payload["behaviours"])
    model.threshold = float(payload["threshold"])
    model.mean = [float(value) for value in payload["mean"]]
    model.scale = [float(value) for value in payload["scale"]]
    model.estimators = list(payload["estimators"])
    return model


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = float(np.exp(-value))
        return 1.0 / (1.0 + exponential)
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)


def per_class_metrics(truth: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]]) -> dict[str, Any]:
    report: dict[str, Any] = {"behaviours": {}, "exact_match_accuracy": 0.0}
    if not truth:
        raise ValueError("metrics need labelled sequences")
    matches = 0
    for actual, guess in zip(truth, predicted):
        if list(actual) == list(guess):
            matches += 1
    report["exact_match_accuracy"] = matches / len(truth)
    matrix = np.asarray(truth, dtype=np.int32)
    guessed = np.asarray(predicted, dtype=np.int32)
    for index, name in enumerate(BEHAVIOURS):
        scores = binary_metrics(matrix[:, index].tolist(), guessed[:, index].tolist())
        negatives = scores["confusion_matrix"]["false_positive"] + scores["confusion_matrix"]["true_negative"]
        scores["false_alarm_rate"] = (
            scores["confusion_matrix"]["false_positive"] / negatives if negatives else 0.0
        )
        report["behaviours"][name] = scores
    report["macro_f1"] = sum(item["f1"] for item in report["behaviours"].values()) / len(BEHAVIOURS)
    return report
