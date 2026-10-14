"""Build a person-only YOLO dataset for the ByteTrack detector.

The local Dataset folder has images and videos but no bounding boxes.
This script uses a pretrained YOLOv8 person detector as a teacher, writes
YOLO labels, and splits train/val so the student can be fine-tuned on the
same crowded overhead and clip domain the tracker will run on.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
DEFAULT_WEIGHTS = ROOT / "models" / "yolov8n.pt"


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _xyxy_to_yolo(xyxy: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = xyxy
    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def _write_pair(image: Path, lines: list[str], destination: Path, stem: str) -> None:
    image_path = destination / "images" / f"{stem}{image.suffix.lower()}"
    label_path = destination / "labels" / f"{stem}.txt"
    shutil.copy2(image, image_path)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _label_image(model, image: Path, confidence: float) -> tuple[list[str], int]:
    result = model(str(image), classes=[0], conf=confidence, verbose=False)[0]
    height, width = result.orig_shape
    lines = [_xyxy_to_yolo([float(v) for v in box.xyxy[0].tolist()], width, height) for box in result.boxes]
    return lines, len(lines)


def _extract_video_frames(video: Path, output_dir: Path, every: int, limit: int) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    written: list[Path] = []
    index = 0
    kept = 0
    try:
        while kept < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                path = output_dir / f"{video.stem}_f{index:06d}.png"
                cv2.imwrite(str(path), frame)
                written.append(path)
                kept += 1
            index += 1
    finally:
        capture.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label Dataset images/videos for YOLOv8 person training.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.25, help="Teacher threshold; ByteTrack later keeps 0.10+")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-every", type=int, default=20)
    parser.add_argument("--video-limit", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DATASET / "yolo")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    staging = args.output / "_all"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    (staging / "labels").mkdir(parents=True)

    sources = _list_images(DATASET / "images")
    video_dir = DATASET / "videos"
    if video_dir.is_dir():
        frame_dir = args.output / "_video_frames"
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        for video in sorted(video_dir.glob("*.mp4")):
            sources.extend(_extract_video_frames(video, frame_dir, args.video_every, args.video_limit))

    labelled = 0
    boxes = 0
    empty = 0
    for index, image in enumerate(sources):
        lines, count = _label_image(model, image, args.confidence)
        _write_pair(image, lines, staging, f"sample_{index:06d}")
        labelled += 1
        boxes += count
        empty += int(count == 0)
        if (index + 1) % 50 == 0:
            print(f"labelled {index + 1}/{len(sources)} images, {boxes} person boxes")

    names = [path.name for path in (staging / "images").iterdir()]
    random.Random(args.seed).shuffle(names)
    split_at = max(1, int(len(names) * (1 - args.val_ratio)))
    splits = {"train": names[:split_at], "val": names[split_at:] or names[-1:]}

    for split, files in splits.items():
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in files:
            stem = Path(name).stem
            shutil.copy2(staging / "images" / name, args.output / "images" / split / name)
            shutil.copy2(staging / "labels" / f"{stem}.txt", args.output / "labels" / split / f"{stem}.txt")

    yaml_path = ROOT / "configs" / "person.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {args.output.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: person",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"images={labelled} boxes={boxes} empty={empty} train={len(splits['train'])} val={len(splits['val'])}")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
