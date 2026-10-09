"""Run Component 2 over Member 1 detection JSON and emit tracking JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import DEFAULT_CONFIG, load_tracker_settings, make_tracker, read_json, track_detection_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Track people from detection-result JSON.")
    parser.add_argument("input", type=Path, help="JSON object or list of detection-frame objects")
    parser.add_argument("--output", type=Path, default=Path("outputs/tracking_log.json"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    results = track_detection_sequence(read_json(args.input), make_tracker(load_tracker_settings(args.config)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
