"""Validate and organize the local video dataset without deleting originals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_videos"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned_videos"
REJECT_DIR = PROJECT_ROOT / "data" / "rejected_videos"
REPORT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "dataset_cleaning_report.csv"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm", ".m4v"}


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path, check_all_frames: bool) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    metadata: dict[str, object] = {
        "file": str(path.relative_to(RAW_DIR)),
        "source_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
        "status": "REJECT_INVALID",
        "reason": "",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "reported_frame_count": 0,
        "decoded_frame_count": 0,
        "duration_seconds": 0.0,
    }
    try:
        if not capture.isOpened():
            metadata["reason"] = "could_not_open"
            return metadata
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata.update(width=width, height=height, fps=fps, reported_frame_count=reported_count)
        if width <= 0 or height <= 0 or fps <= 0 or reported_count <= 0:
            metadata["reason"] = "invalid_metadata"
            return metadata

        decoded_count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded_count += 1
            if not check_all_frames and decoded_count >= 1:
                break
        metadata["decoded_frame_count"] = decoded_count
        metadata["duration_seconds"] = decoded_count / fps if check_all_frames else reported_count / fps
        if decoded_count == 0:
            metadata["reason"] = "no_decodable_frames"
        elif check_all_frames and decoded_count != reported_count:
            metadata["reason"] = "frame_count_mismatch"
        else:
            metadata["status"] = "CLEAN"
    finally:
        capture.release()
    return metadata


def copy_with_relative_path(source: Path, destination_root: Path) -> None:
    destination = destination_root / source.relative_to(RAW_DIR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Decode only the first frame; by default every frame is checked.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw video folder not found: {raw_dir}")

    global RAW_DIR
    RAW_DIR = raw_dir
    videos = sorted(
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No supported videos found under {raw_dir}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    REJECT_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes: dict[str, Path] = {}
    rows: list[dict[str, object]] = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] checking {video.relative_to(raw_dir)}")
        row = inspect_video(video, check_all_frames=not args.sample_only)
        content_hash = str(row["sha256"])
        if row["status"] == "CLEAN" and content_hash in seen_hashes:
            row["status"] = "REJECT_DUPLICATE"
            row["reason"] = f"same_sha256_as:{seen_hashes[content_hash].relative_to(raw_dir)}"
        elif row["status"] == "CLEAN":
            seen_hashes[content_hash] = video

        if row["status"] == "CLEAN":
            copy_with_relative_path(video, CLEAN_DIR)
        else:
            copy_with_relative_path(video, REJECT_DIR)
        rows.append(row)

    fieldnames = list(rows[0])
    with args.report.open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"wrote {args.report}")
    print("status counts:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
