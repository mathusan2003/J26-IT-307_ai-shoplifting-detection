"""SYNTHETIC smoke test for the Model 1 training interface.

This file does not read the real videos and does not write
evaluation/video_baseline_metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model import binary_metrics
from train_video_baseline import false_alarm_rate
from video_dataset import feature_names, write_json

MODULE_ROOT = Path(__file__).resolve().parents[1]
SMOKE_METRICS_PATH = MODULE_ROOT / "evaluation" / "video_baseline_smoke_metrics.json"
SEED = 13
FEATURE_COUNT = len(feature_names())


def build_smoke_rows(seed: int = SEED) -> list[dict]:
    """Build tiny synthetic motion rows. The label is not a behaviour type."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(24):
        label = "shoplifting" if index % 2 else "normal"
        features = rng.normal(0.2 if label == "shoplifting" else -0.2, 0.4, FEATURE_COUNT)
        if label == "shoplifting":
            features[0] += 1.0
        split = "train" if index < 16 else "validation" if index < 20 else "test"
        rows.append(
            {
                "video_id": f"synthetic/{label}-{index}.mp4",
                "split": split,
                "label": label,
                "features": features.astype(float).tolist(),
            }
        )
    return rows


def run_smoke() -> dict:
    """Fit the same logistic interface on synthetic rows and score a held-out block."""
    rows = build_smoke_rows()
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    train_x = np.asarray([row["features"] for row in train], dtype=np.float64)
    train_y = np.asarray([1 if row["label"] == "shoplifting" else 0 for row in train])
    scaler = StandardScaler()
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=SEED
    )
    classifier.fit(scaler.fit_transform(train_x), train_y)

    def score(split_rows: list[dict]) -> dict:
        features = np.asarray([row["features"] for row in split_rows], dtype=np.float64)
        truth = [1 if row["label"] == "shoplifting" else 0 for row in split_rows]
        predicted = classifier.predict(scaler.transform(features)).astype(int).tolist()
        metrics = binary_metrics(truth, predicted)
        metrics["false_alarm_rate"] = false_alarm_rate(metrics["confusion_matrix"])
        return metrics

    report = {
        "task": "synthetic_smoke_normal_vs_shoplifting",
        "synthetic": True,
        "not_real_cctv": True,
        "real_world_accuracy": False,
        "behaviour_labels_used": False,
        "mixed_with_real_video_evaluation": False,
        "writes_video_baseline_metrics": False,
        "feature_names": feature_names(),
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "validation": score(validation),
        "test": score(test),
        "limitation": (
            "SYNTHETIC smoke test of the training interface. "
            "It is not the real-video Model 1 result."
        ),
    }
    write_json(SMOKE_METRICS_PATH, report)
    return report


def main() -> int:
    report = run_smoke()
    json.dump(
        {
            "synthetic": True,
            "real_world_accuracy": False,
            "test_f1": report["test"]["f1"],
            "test_confusion_matrix": report["test"]["confusion_matrix"],
            "wrote": str(SMOKE_METRICS_PATH),
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
