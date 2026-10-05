"""Evaluate Component 2 tracking output against locally labelled ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import evaluate_tracking


def _load_sequence(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list of frame objects")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Component 2 MOT metrics.")
    parser.add_argument("predictions", type=Path, help="tracking_log JSON file")
    parser.add_argument("ground_truth", type=Path, help="labelled tracking JSON file")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold (default: 0.5)")
    parser.add_argument("--occlusion-frames", type=int, default=15, help="minimum missed frames for recovery metric")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_metrics.json"))
    args = parser.parse_args()
    metrics = evaluate_tracking(_load_sequence(args.predictions), _load_sequence(args.ground_truth),
                                args.iou, args.occlusion_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
