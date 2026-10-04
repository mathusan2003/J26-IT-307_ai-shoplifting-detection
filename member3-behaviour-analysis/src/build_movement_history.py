"""Build movement-history JSON from a tracking sequence file.

Run from ``member3-behaviour-analysis``:

    python src/build_movement_history.py samples/sample_tracking_frames.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from movement_history import build_movement_histories
from tracking_input import load_tracking_sequence


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python src/build_movement_history.py <tracking-json>",
            file=sys.stderr,
        )
        return 1

    frames = load_tracking_sequence(args[0])
    json.dump(build_movement_histories(frames), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
