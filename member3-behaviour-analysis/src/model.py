"""L2 logistic regression for suspicious versus ordinary tracks.

The classifier uses the same measurements as the rule detectors. Rule
indicators remain the JSON sent to Member 4. This model is the learned
check that those measurements separate suspicious tracks from ordinary ones.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from config import ModelSettings
from features import FEATURE_NAMES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "behaviour_model.json"


class LogisticBehaviourModel:
    """Binary logistic regression with standardised features."""

    def __init__(self, settings: ModelSettings) -> None:
        self.learning_rate = settings.learning_rate
        self.epochs = settings.epochs
        self.l2 = settings.l2
        self.threshold = settings.threshold
        self.feature_names = list(FEATURE_NAMES)
        self.weights = [0.0 for _ in FEATURE_NAMES]
        self.bias = 0.0
        self.mean = [0.0 for _ in FEATURE_NAMES]
        self.std = [1.0 for _ in FEATURE_NAMES]

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        """Train on rows aligned with ``FEATURE_NAMES``."""
        if not rows:
            raise ValueError("the behaviour model needs at least one training row")
        if len(rows) != len(labels):
            raise ValueError("each training row needs one label")
        width = len(self.feature_names)
        if any(len(row) != width for row in rows):
            raise ValueError(f"each training row must contain {width} features")
        count = len(rows)
        self.mean = [sum(row[index] for row in rows) / count for index in range(width)]
        self.std = []
        for index in range(width):
            variance = sum((row[index] - self.mean[index]) ** 2 for row in rows) / count
            self.std.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
        self.weights = [0.0 for _ in range(width)]
        self.bias = 0.0
        for _ in range(self.epochs):
            gradient_w = [0.0 for _ in range(width)]
            gradient_b = 0.0
            for row, label in zip(rows, labels):
                error = self._probability(row) - label
                scaled = self._scale(row)
                for index in range(width):
                    gradient_w[index] += error * scaled[index]
                gradient_b += error
            for index in range(width):
                self.weights[index] -= self.learning_rate * (
                    gradient_w[index] / count + self.l2 * self.weights[index]
                )
            self.bias -= self.learning_rate * gradient_b / count

    def predict_proba(self, row: Sequence[float]) -> float:
        """Return the probability that the track is suspicious."""
        return self._probability(row)

    def predict(self, row: Sequence[float]) -> int:
        """Return 1 when the suspicious probability reaches the threshold."""
        return int(self.predict_proba(row) >= self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "l2_logistic_regression",
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogisticBehaviourModel:
        settings = ModelSettings(
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
            l2=float(payload["l2"]),
            threshold=float(payload["threshold"]),
        )
        model = cls(settings)
        model.feature_names = list(payload["feature_names"])
        model.weights = [float(value) for value in payload["weights"]]
        model.bias = float(payload["bias"])
        model.mean = [float(value) for value in payload["mean"]]
        model.std = [float(value) for value in payload["std"]]
        model.threshold = float(payload["threshold"])
        return model

    def _probability(self, row: Sequence[float]) -> float:
        scaled = self._scale(row)
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, scaled))
        return _sigmoid(score)

    def _scale(self, row: Sequence[float]) -> list[float]:
        return [(value - mean) / std for value, mean, std in zip(row, self.mean, self.std)]


def save_model(model: LogisticBehaviourModel, path: str | Path | None = None) -> Path:
    file_path = Path(path) if path is not None else MODEL_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    return file_path


def load_model(path: str | Path | None = None) -> LogisticBehaviourModel:
    file_path = Path(path) if path is not None else MODEL_PATH
    with file_path.open(encoding="utf-8") as handle:
        return LogisticBehaviourModel.from_dict(json.load(handle))


def binary_metrics(truth: Sequence[int], predicted: Sequence[int]) -> dict[str, Any]:
    """Precision, recall, F1, specificity, accuracy, and the confusion matrix."""
    if len(truth) != len(predicted) or not truth:
        raise ValueError("metrics need paired non-empty labels and predictions")
    true_positive = true_negative = false_positive = false_negative = 0
    for actual, guess in zip(truth, predicted):
        if actual == 1 and guess == 1:
            true_positive += 1
        elif actual == 0 and guess == 0:
            true_negative += 1
        elif actual == 0 and guess == 1:
            false_positive += 1
        else:
            false_negative += 1
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    return {
        "accuracy": (true_positive + true_negative) / len(truth),
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "specificity": _rate(true_negative, true_negative + false_positive),
        "support": len(truth),
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
