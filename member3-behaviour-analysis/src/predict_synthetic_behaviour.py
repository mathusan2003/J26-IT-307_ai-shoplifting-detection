"""Show rule scores beside the SYNTHETIC multi-label prediction.

Member 4 still receives only the rule JSON from analyse_behaviour.py.
This command prints both views for a viva. It does not add fields to the
shared contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from behaviour import analyse_tracking_frames, load_detection_sequence, validate_behaviour_result
from config import load_settings, load_zones
from features import build_profile, index_detections
from movement_history import MovementHistoryStore
from synthetic_behaviour_model import load_classifier
from synthetic_tracking import require_concealment_detections, SyntheticInputError
from tracking_input import load_tracking_sequence


def predict_sequence(tracking_path: Path, detections_path: Path | None = None) -> dict:
    frames = load_tracking_sequence(tracking_path)
    detections = load_detection_sequence(detections_path) if detections_path else None
    settings = load_settings()
    zones = load_zones()
    rules = analyse_tracking_frames(frames, detections, settings, zones)
    for result in rules:
        validate_behaviour_result(result)

    detection_ok = False
    concealment_block = None
    if detections is not None:
        try:
            require_concealment_detections(detections)
            detection_ok = True
        except SyntheticInputError as exc:
            concealment_block = str(exc)
    else:
        concealment_block = (
            "No Member 1 detection file. possible_concealment is withheld. "
            "Person coordinates alone are not object-interaction evidence."
        )

    model = load_classifier()
    store = MovementHistoryStore()
    store.ingest_frames(frames)
    people = []
    indexed = index_detections(detections)
    for person_id in store.person_ids():
        profile = build_profile(store.records_for(person_id), indexed, settings, zones)
        probabilities = model.predict_proba(profile.as_vector())
        predicted = {name: int(value >= model.threshold) for name, value in probabilities.items()}
        withheld = []
        if not detection_ok:
            predicted["possible_concealment"] = 0
            withheld.append("possible_concealment")
        people.append(
            {
                "person_id": person_id,
                "probabilities": probabilities,
                "predicted": [name for name, flag in predicted.items() if flag],
                "threshold": model.threshold,
                "withheld": withheld,
                "note": "These probabilities are not behaviour_score and are not Member 4 risk_score.",
            }
        )
    return {
        "note": (
            "member4_contract is the only object Member 4 reads. "
            "synthetic_ml is a viva display and is not part of the risk contract."
        ),
        "synthetic_ml_source": "SYNTHETIC",
        "real_world_accuracy": False,
        "member4_contract": rules,
        "synthetic_ml": {
            "model": "models/synthetic_behaviour_classifier.json",
            "replaces_rule_detectors": False,
            "object_evidence": detection_ok,
            "concealment_block": concealment_block,
            "people": people,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print rule JSON beside SYNTHETIC ML predictions.")
    parser.add_argument("tracking", type=Path)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = predict_sequence(args.tracking, args.detections)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
