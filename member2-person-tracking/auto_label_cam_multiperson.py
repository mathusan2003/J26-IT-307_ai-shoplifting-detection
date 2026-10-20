"""Write class-0 YOLO boxes beside images in Dataset/cam multi-person.

Uses the local YOLOv8n baseline at confidence 0.25. Each image gets a sibling
.txt file. Images with no person still get an empty file. Existing labels are
left in place unless --force is set, so an interrupted run can continue.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "Dataset" / "cam multi-person"
DEFAULT_WEIGHTS = ROOT / "models" / "yolov8n.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: path.name.lower(),
    )


def label_text(result: object) -> str:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ""
    lines = []
    for box in boxes:
        x_center, y_center, width, height = box.xywhn[0].tolist()
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines) + "\n"


def write_label(image: Path, text: str) -> None:
    target = image.with_suffix(".txt")
    temporary = image.with_suffix(".txt.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def label_images(
    images: list[Path],
    weights: Path,
    confidence: float = 0.25,
    batch: int = 8,
) -> tuple[int, int]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    written = 0
    person_boxes = 0
    chunk_size = max(batch, 1) * 4
    for start in range(0, len(images), chunk_size):
        chunk = images[start:start + chunk_size]
        predictions = model.predict(
            source=[str(path) for path in chunk],
            conf=confidence,
            classes=[0],
            imgsz=640,
            batch=batch,
            stream=True,
            verbose=False,
            save=False,
        )
        for result in predictions:
            text = label_text(result)
            write_label(Path(result.path), text)
            written += 1
            person_boxes += text.count("\n")
            if written == 1 or written % 200 == 0 or written == len(images):
                print(f"labeled {written}/{len(images)} person_boxes={person_boxes}", flush=True)
    return written, person_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label cam multi-person images with YOLOv8n class 0.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Label only the first N images (0 means all)")
    parser.add_argument("--force", action="store_true", help="Replace labels that already exist")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Could not find baseline weights: {args.weights}")
    images = list_images(args.source)
    if args.limit > 0:
        images = images[:args.limit]
    pending = images if args.force else [path for path in images if not path.with_suffix(".txt").is_file()]
    print(f"images={len(images)} to_label={len(pending)} weights={args.weights}", flush=True)
    if not pending:
        print("Auto-annotation complete! All .txt label files created.")
        return
    written, person_boxes = label_images(pending, args.weights, args.conf, args.batch)
    print("Auto-annotation complete! All .txt label files created.")
    print(f"wrote={written} person_boxes={person_boxes}")


if __name__ == "__main__":
    main()
