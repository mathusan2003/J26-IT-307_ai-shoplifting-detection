"""Run Component 2 over Member 1 detection JSON and emit tracking JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tracker import ByteTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Track people from detection-result JSON.")
    parser.add_argument("input", type=Path, help="JSON object or list of detection-frame objects")
    parser.add_argument("--output", type=Path, default=Path("outputs/tracking_log.json"))
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    frames = source if isinstance(source, list) else [source]
    tracker, results = ByteTracker(), []
    for frame in sorted(frames, key=lambda item: item["frame_id"]):
        people = [item for item in frame.get("detections", []) if item.get("class") == "person"]
        results.append({"frame_id": frame["frame_id"], "tracks": [track.to_contract() for track in tracker.update(people)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
