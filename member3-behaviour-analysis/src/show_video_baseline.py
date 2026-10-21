"""Print the saved Model 1 real-video metrics without retraining.

The numbers come from evaluation/video_baseline_metrics.json. They were
fit on datasets/shoplifting-video folder labels. They are not behaviour
labels and they are not rewritten here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = MODULE_ROOT / "evaluation" / "video_baseline_metrics.json"


def load_saved_video_metrics(path: Path | None = None) -> dict:
    file_path = path if path is not None else METRICS_PATH
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if payload.get("task") != "video_level_normal_vs_shoplifting":
        raise ValueError("saved file is not the Model 1 video-level report")
    if payload.get("behaviour_labels_used") is not False:
        raise ValueError("Model 1 report must keep behaviour_labels_used false")
    return payload


def summary(metrics: dict) -> dict:
    forest = metrics.get("random_forest") or {}
    forest_test = forest.get("test") or {}
    return {
        "task": metrics["task"],
        "label_source": metrics["label_source"],
        "behaviour_labels_used": metrics["behaviour_labels_used"],
        "detects_behaviour_categories": False,
        "logistic_regression": {
            "C": metrics["model"]["C"],
            "test": {
                "accuracy": metrics["test"]["accuracy"],
                "precision": metrics["test"]["precision"],
                "recall": metrics["test"]["recall"],
                "f1": metrics["test"]["f1"],
                "false_alarm_rate": metrics["test"].get("false_alarm_rate"),
                "confusion_matrix": metrics["test"]["confusion_matrix"],
            },
        },
        "random_forest": {
            "model": forest.get("model"),
            "test": {
                "accuracy": forest_test.get("accuracy"),
                "precision": forest_test.get("precision"),
                "recall": forest_test.get("recall"),
                "f1": forest_test.get("f1"),
                "false_alarm_rate": forest_test.get("false_alarm_rate"),
                "confusion_matrix": forest_test.get("confusion_matrix"),
            },
        },
        "duration_only_test_f1": metrics["duration_only_test"]["f1"],
        "source_file": str(METRICS_PATH),
    }


def main() -> int:
    payload = summary(load_saved_video_metrics())
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
