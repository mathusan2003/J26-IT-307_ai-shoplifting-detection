"""Build a single-class YOLO dataset from Dataset/cam multi-person.

Images are split 80/20 with a fixed seed. A sibling .txt file with the same
stem is copied when it exists. Missing annotations become empty label files.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "Dataset" / "cam multi-person"
DEFAULT_OUTPUT = ROOT / "Dataset" / "yolo_cam_multiperson"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DATASET_YAML = """path: ./Dataset/yolo_cam_multiperson
train: images/train
val: images/val
names:
  0: person
"""


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().lower(),
    )


def split_images(images: list[Path], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Return a deterministic train/val split. Val is about ``val_ratio`` of the images."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    ordered = sorted(images, key=lambda path: path.as_posix().lower())
    random.Random(seed).shuffle(ordered)
    train_count = int(len(ordered) * (1 - val_ratio))
    if ordered and train_count == 0:
        train_count = len(ordered)
    elif len(ordered) > 1 and train_count == len(ordered):
        train_count -= 1
    return ordered[:train_count], ordered[train_count:]


def label_source(image: Path) -> Path | None:
    candidate = image.with_suffix(".txt")
    return candidate if candidate.is_file() else None


def _destination_stem(image: Path, source_root: Path, used: set[str]) -> str:
    relative = image.relative_to(source_root).with_suffix("")
    stem = "__".join(relative.parts)
    if stem not in used:
        used.add(stem)
        return stem
    index = 2
    while f"{stem}_{index}" in used:
        index += 1
    unique = f"{stem}_{index}"
    used.add(unique)
    return unique


def write_split(images: list[Path], source_root: Path, split_root: Path, split_name: str) -> tuple[int, int]:
    image_dir = split_root / "images" / split_name
    label_dir = split_root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    labelled = 0
    used: set[str] = set()
    for index, image in enumerate(images, start=1):
        if index == 1 or index % 1000 == 0 or index == len(images):
            print(f"{split_name}: {index}/{len(images)}", flush=True)
        stem = _destination_stem(image, source_root, used)
        shutil.copy2(image, image_dir / f"{stem}{image.suffix.lower()}")
        source_label = label_source(image)
        destination = label_dir / f"{stem}.txt"
        if source_label is None:
            destination.write_text("", encoding="utf-8")
        else:
            shutil.copy2(source_label, destination)
            if destination.read_text(encoding="utf-8").strip():
                labelled += 1
    return len(images), labelled


def write_dataset_yaml(output: Path) -> Path:
    path = output / "dataset.yaml"
    path.write_text(DATASET_YAML, encoding="utf-8")
    return path


def prepare(source: Path, output: Path, val_ratio: float = 0.2, seed: int = 42) -> dict[str, int]:
    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"No .jpg, .jpeg, or .png images found under {source}")
    if output.exists():
        shutil.rmtree(output)
    train_images, val_images = split_images(images, val_ratio=val_ratio, seed=seed)
    train_count, train_labelled = write_split(train_images, source, output, "train")
    val_count, val_labelled = write_split(val_images, source, output, "val")
    write_dataset_yaml(output)
    return {
        "images": len(images),
        "train": train_count,
        "val": val_count,
        "labelled": train_labelled + val_labelled,
        "empty_labels": len(images) - (train_labelled + val_labelled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the cam multi-person images as a single-class YOLO dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(args.source, args.output, args.val_ratio, args.seed)
    print(
        f"images={summary['images']} train={summary['train']} val={summary['val']} "
        f"labelled={summary['labelled']} empty_labels={summary['empty_labels']}"
    )
    print(f"wrote {args.output / 'dataset.yaml'}")
    if summary["labelled"] == 0:
        print(
            "warning: every label file is empty. Fine-tuning on this split would teach the "
            "detector that these frames contain no people. Add YOLO .txt boxes beside the "
            "source images and run this script again before the Colab training job."
        )


if __name__ == "__main__":
    main()
