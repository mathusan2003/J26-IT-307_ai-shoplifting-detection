"""Read Member 2 tracking JSON and write Member 4 behaviour JSON.

Run from ``member3-behaviour-analysis``:

    python src/analyse_behaviour.py samples/member2_contract_loiter.json
    python src/analyse_behaviour.py path/to/tracking.json --detections path/to/detections.json --output outputs/behaviour.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from behaviour import analyse_tracking_frames, load_detection_sequence, validate_behaviour_result, write_behaviour_results
from config import load_settings, load_zones
from tracking_input import load_tracking_sequence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse tracked people for suspicious behaviour.")
    parser.add_argument("tracking", type=Path, help="Member 2 tracking JSON")
    parser.add_argument("--detections", type=Path, help="Optional Member 1 detection JSON with non-person classes")
    parser.add_argument("--output", type=Path, help="Where to write behaviour JSON. Also printed to stdout.")
    parser.add_argument("--config", type=Path, help="Behaviour threshold JSON")
    parser.add_argument("--zones", type=Path, help="Zone polygon JSON")
    args = parser.parse_args(argv)

    frames = load_tracking_sequence(args.tracking)
    detections = load_detection_sequence(args.detections) if args.detections else None
    results = analyse_tracking_frames(
        frames,
        detections,
        load_settings(args.config),
        load_zones(args.zones),
    )
    for result in results:
        validate_behaviour_result(result)
    if args.output is not None:
        write_behaviour_results(results, args.output)
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
