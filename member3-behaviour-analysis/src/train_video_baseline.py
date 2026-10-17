"""Train a video-level Normal versus Shoplifting logistic regression.

The model uses motion measurements from the real clips. It does not emit
Member 4 behaviour JSON and it does not label the five behaviour types.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model import binary_metrics
from video_dataset import (
    CATALOG_PATH,
    FEATURES_PATH,
    POSITIVE_LABEL,
    SPLIT_PATH,
    assign_splits,
    build_catalog,
    feature_names,
    features_from_catalog,
    load_json,
    write_json,
)

MODULE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = MODULE_ROOT / "models" / "video_baseline.json"
METRICS_PATH = MODULE_ROOT / "evaluation" / "video_baseline_metrics.json"
CANDIDATE_C = (0.01, 0.1, 1.0, 10.0)


def train_from_features(features: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select regularisation on validation, then score the test split once."""
    names = features["feature_names"]
    if names != feature_names():
        raise ValueError("feature names do not match the motion baseline")
    by_split = {name: [row for row in features["rows"] if row["split"] == name] for name in ("train", "validation", "test")}
    for name, rows in by_split.items():
        if not rows:
            raise ValueError(f"the {name} split has no videos")
    train_x, train_y = _matrix(by_split["train"])
    chosen = None
    for penalty in CANDIDATE_C:
        model = _fit(train_x, train_y, penalty)
        validation_scores = score_split(model, by_split["validation"])
        candidate = (validation_scores["f1"], -abs(penalty - 1.0), model, validation_scores)
        if chosen is None or candidate[:2] > chosen[:2]:
            chosen = candidate
    assert chosen is not None
    _f1, _distance, model, validation_scores = chosen
    report = {
        "task": "video_level_normal_vs_shoplifting",
        "label_source": "parent_folder",
        "positive_class": POSITIVE_LABEL,
        "behaviour_labels_used": False,
        "features": names,
        "model": {
            "type": "l2_logistic_regression",
            "library": "scikit-learn",
            "class_weight": "balanced",
            "C": model["C"],
            "C_selection": "highest validation F1; ties prefer C closest to 1",
            "threshold": 0.5,
        },
        "split_counts": {name: _label_counts(rows) for name, rows in by_split.items()},
        "train": score_split(model, by_split["train"]),
        "validation": validation_scores,
        "test": score_split(model, by_split["test"]),
        "majority_baseline_test": _majority_baseline(by_split["train"], by_split["test"]),
        "duration_only_test": _metadata_baseline(by_split["train"], by_split["test"], ["duration_s"]),
        "resolution_only_test": _metadata_baseline(by_split["train"], by_split["test"], ["width", "height"]),
    }
    return model, report


def score_split(model: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score one split. Shoplifting is the positive class."""
    features, truth = _matrix(rows)
    probability = _predict_proba(model, features)
    predicted = [int(value >= model["threshold"]) for value in probability]
    metrics = binary_metrics(truth, predicted)
    metrics["probabilities"] = [
        {"video_id": row["video_id"], "label": row["label"], "shoplifting_probability": probability[index]}
        for index, row in enumerate(rows)
    ]
    return metrics


def save_model(model: dict[str, Any], path: Path | None = None) -> Path:
    file_path = path if path is not None else MODEL_PATH
    payload = dict(model)
    write_json(file_path, payload)
    return file_path


def load_model(path: Path | None = None) -> dict[str, Any]:
    return load_json(path if path is not None else MODEL_PATH)


def run_video_baseline() -> dict[str, Any]:
    """Inspect the clips, write the split, train, and store metrics."""
    catalog = build_catalog()
    split = assign_splits(catalog["videos"])
    features = features_from_catalog(catalog, split)
    model, metrics = train_from_features(features)
    metrics["catalog_counts"] = catalog["counts"]
    metrics["missing_expected_names"] = catalog["missing_expected_names"]
    metrics["duplicate_hashes"] = catalog["duplicate_hashes"]
    metrics["hash_label_conflicts"] = catalog["hash_label_conflicts"]
    metrics["excluded"] = split["excluded"]
    metrics["split_rule"] = split["ratio"]
    metrics["limitations"] = [
        "Folder names are the only labels. Loitering, repeated shelf visits, abnormal movement, object interaction, and concealment are not annotated in these clips.",
        "RetailS and PoseLift were not found on this machine, so they were not used.",
        "The motion model is a baseline. On the locked test split its F1 is below the majority-class F1 and below a duration-only model, so clip length is a confound and motion alone is weak.",
        "Three normal clips are byte-for-byte duplicates of other normal clips. Each duplicate pair is kept in one split and counted once.",
        "shoplifting-49.mp4 is missing. No replacement label was created.",
    ]
    write_json(CATALOG_PATH, _catalog_without_features(catalog))
    write_json(FEATURES_PATH, features)
    write_json(SPLIT_PATH, split)
    save_model(model)
    write_json(METRICS_PATH, _metrics_without_probabilities(metrics))
    return metrics


def _catalog_without_features(catalog: dict[str, Any]) -> dict[str, Any]:
    stored = json.loads(json.dumps(catalog))
    for video in stored["videos"]:
        video["has_features"] = bool(video.get("features"))
        video.pop("features", None)
    return stored


def _metrics_without_probabilities(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop per-video probabilities from the committed report; scores remain."""
    stored = json.loads(json.dumps(metrics))
    for split_name in ("train", "validation", "test"):
        stored[split_name].pop("probabilities", None)
    return stored


def _fit(features: np.ndarray, labels: np.ndarray, penalty: float) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=penalty,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=13,
    )
    classifier.fit(scaled, labels)
    scale = scaler.scale_.astype(float).copy()
    scale[scale == 0] = 1.0
    return {
        "C": penalty,
        "threshold": 0.5,
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scale.tolist(),
        "coef": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "feature_names": feature_names(),
    }


def _predict_proba(model: dict[str, Any], features: np.ndarray) -> list[float]:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    scaled = (features - mean) / scale
    scores = model["intercept"] + scaled @ coef
    return [_sigmoid(float(score)) for score in scores]


def _matrix(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[int]]:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = [1 if row["label"] == POSITIVE_LABEL else 0 for row in rows]
    return features, labels


def _label_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "videos": len(rows),
        "normal": sum(row["label"] == "normal" for row in rows),
        "shoplifting": sum(row["label"] == POSITIVE_LABEL for row in rows),
    }


def _majority_baseline(train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(row["label"] == POSITIVE_LABEL for row in train_rows)
    majority = 1 if positives * 2 >= len(train_rows) else 0
    truth = [1 if row["label"] == POSITIVE_LABEL else 0 for row in test_rows]
    return binary_metrics(truth, [majority for _ in truth])


def _metadata_baseline(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    fields: list[str],
) -> dict[str, Any]:
    """Fit a logistic model on clip metadata. This is a confound check, not the baseline."""
    train_x = np.asarray([[float(row[field] or 0.0) for field in fields] for row in train_rows], dtype=np.float64)
    test_x = np.asarray([[float(row[field] or 0.0) for field in fields] for row in test_rows], dtype=np.float64)
    train_y = np.asarray([1 if row["label"] == POSITIVE_LABEL else 0 for row in train_rows])
    scaler = StandardScaler()
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=13
    )
    classifier.fit(scaler.fit_transform(train_x), train_y)
    predicted = classifier.predict(scaler.transform(test_x)).astype(int).tolist()
    truth = [1 if row["label"] == POSITIVE_LABEL else 0 for row in test_rows]
    metrics = binary_metrics(truth, predicted)
    metrics["fields"] = fields
    return metrics


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = np.exp(-value)
        return float(1.0 / (1.0 + exponential))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def main() -> int:
    metrics = run_video_baseline()
    summary = {
        "test_accuracy": metrics["test"]["accuracy"],
        "test_precision": metrics["test"]["precision"],
        "test_recall": metrics["test"]["recall"],
        "test_f1": metrics["test"]["f1"],
        "test_specificity": metrics["test"]["specificity"],
        "test_confusion_matrix": metrics["test"]["confusion_matrix"],
        "validation_f1": metrics["validation"]["f1"],
        "majority_baseline_test_accuracy": metrics["majority_baseline_test"]["accuracy"],
        "duration_only_test_f1": metrics["duration_only_test"]["f1"],
        "resolution_only_test_f1": metrics["resolution_only_test"]["f1"],
        "C": metrics["model"]["C"],
        "missing_expected_names": metrics["missing_expected_names"],
        "duplicate_hashes": len(metrics["duplicate_hashes"]),
        "excluded": len(metrics["excluded"]),
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
