"""Fine-tune a pretrained YOLOv8 model on the person dataset used by ByteTrack."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "configs" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "models" / "yolov8n.pt"
EXPORT_WEIGHTS = ROOT / "models" / "yolov8n-person-track.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the YOLOv8 person detector for Component 2 tracking.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError("Run prepare_person_dataset.py before training.")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name="person_bytetrack",
        exist_ok=True,
        patience=args.patience,
        workers=args.workers,
        pretrained=True,
        single_cls=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        plots=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but {best} was not written")
    EXPORT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, EXPORT_WEIGHTS)
    print(f"copied {best} -> {EXPORT_WEIGHTS}")


if __name__ == "__main__":
    main()
