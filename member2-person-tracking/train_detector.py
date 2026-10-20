"""Fine-tune YOLOv8 person detection for crowded retail tracking.

For higher recall on small/occluded shoppers, point ``--data`` at a YOLO yaml
built from CrowdHuman, MOT20 pedestrians (class 0), or extra labelled CCTV
frames, and keep mosaic/mixup/multi-scale on. Crowd-oriented example::

    python train_detector.py --data configs/data.yaml --imgsz 1280 --epochs 50 --batch 8 --device 0
"""
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
    parser.add_argument("--imgsz", type=int, default=1280, help="Use 1280 for small/distant people in crowds")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--export", type=Path, default=EXPORT_WEIGHTS, help="Where to copy best.pt")
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
        mixup=0.15,
        copy_paste=0.1,
        multi_scale=0.5,
        close_mosaic=10,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        plots=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but {best} was not written")
    args.export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export)
    print(f"copied {best} -> {args.export}")


if __name__ == "__main__":
    main()
