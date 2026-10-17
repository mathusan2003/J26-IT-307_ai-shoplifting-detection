"""Train the behaviour model and write the evaluation report.

Run from ``member3-behaviour-analysis``:

    python src/train_behaviour_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dataset import member2_contract_loiter_frames, write_dataset
from evaluation import train_and_evaluate, write_metrics
from model import save_model

SAMPLE_PATH_NAME = "samples/member2_contract_loiter.json"


def main(argv: list[str] | None = None) -> int:
    del argv
    dataset_path = write_dataset()
    sample_path = Path(__file__).resolve().parents[1] / SAMPLE_PATH_NAME
    sample_path.write_text(json.dumps(member2_contract_loiter_frames(), indent=2) + "\n", encoding="utf-8")
    model, report = train_and_evaluate()
    model_path = save_model(model)
    metrics_path = write_metrics(report)
    summary = {
        "dataset": str(dataset_path),
        "model": str(model_path),
        "metrics": str(metrics_path),
        "test_model_f1": report["test"]["model"]["f1"],
        "test_model_accuracy": report["test"]["model"]["accuracy"],
        "test_rule_f1": {name: values["f1"] for name, values in report["test"]["rules"].items()},
        "leave_one_family_out_recall": {
            name: values["held_out_recall"] for name, values in report["leave_one_family_out"].items()
        },
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
