"""Fine‑tune a YOLO‑v8 model on the training dataset prepared by
``prepare_training_dataset.py``.

The script expects a YOLO‑v8 ``data.yaml`` file (created by the data‑prep
script) and a base weights file (default is the official ``yolov8n.pt``).  It
invokes the Ultralytics ``YOLO`` class directly, so you do not need the
external ``ultralytics`` CLI, but you must have the ``ultralytics`` package
installed in the active virtual environment.

Example usage::

    python src/train_yolo.py \
        --data member2-person-tracking/training_dataset/data.yaml \
        --weights member2-person-tracking/models/yolov8n.pt \
        --epochs 30 \
        --batch 8 \
        --imgsz 640

The trained checkpoint will be written to ``member2-person-tracking/models``
with the suffix ``_finetuned.pt``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the YOLO data.yaml file (produced by prepare_training_dataset.py).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("member2-person-tracking/models/yolov8n.pt"),
        help="Base weights to fine‑tune from. Use a pretrained YOLOv8 checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (square).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("member2-person-tracking/models"),
        help="Directory where the fine‑tuned checkpoint will be saved.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device spec for Ultralytics (e.g., 'cpu' or '0' for first GPU).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers (default: 2 for Windows stability).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    # Verify paths
    if not args.data.is_file():
        sys.exit(f"[train_yolo] data.yaml not found: {args.data}")
    if not args.weights.is_file():
        sys.exit(f"[train_yolo] base weights not found: {args.weights}")
    args.project.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics package is required. Install it with: "
            "python -m pip install ultralytics"
        ) from exc

    model = YOLO(str(args.weights))
    print("[train_yolo] Starting fine‑tuning...")
    # Train – the YOLO API returns the model with the best checkpoint loaded.
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(args.project),
        name="finetuned",
        exist_ok=True,
        device=args.device,
        workers=args.workers,
    )
    # Locate best.pt across Ultralytics trainer save_dir or project directory
    candidates = []
    if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") and model.trainer.save_dir:
        candidates.append(Path(model.trainer.save_dir) / "weights" / "best.pt")
    candidates.extend([
        Path("runs/detect") / args.project / "finetuned" / "weights" / "best.pt",
        args.project / "finetuned" / "weights" / "best.pt",
    ])
    target_ckpt = args.project / "yolov8n_finetuned.pt"
    found = False
    for best_ckpt in candidates:
        if best_ckpt.is_file():
            target_ckpt.write_bytes(best_ckpt.read_bytes())
            print(f"[train_yolo] Fine‑tuned checkpoint saved to {target_ckpt}")
            found = True
            break
    if not found:
        print("[train_yolo] Warning: expected checkpoint not found in candidates:", candidates)

if __name__ == "__main__":
    main()
