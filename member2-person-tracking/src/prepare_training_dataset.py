"""Utility to prepare a YOLO‑v8 training dataset from the existing tracking outputs.

The repository already contains inference scripts that produce, for each video, a JSON
file with per‑frame person tracks (see `run_baseline.py`).  This script extracts every
frame as an image and converts the corresponding track annotations to the YOLO
format required for training.

Usage example::

    python src/prepare_training_dataset.py \
        --video-dir member2-person-tracking/Dataset/videos \
        --json-dir member2-person-tracking/outputs/json \
        --output-dir member2-person-tracking/training_dataset

After running, the directory layout will be::

    training_dataset/
        images/      # *.jpg frames
        labels/      # *.txt YOLO label files
        data.yaml    # dataset configuration for Ultralytics

You can then train a model with ``train_yolo.py`` (or directly via the Ultralytics CLI).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from pathlib import Path

import cv2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing the source MP4 videos.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Directory holding the JSON tracking files produced by run_baseline.py. "
        "Each JSON file should have the same stem as its video (e.g. foo.mp4 → foo.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_dataset"),
        help="Root folder where images/, labels/ and data.yaml will be created.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Frame extraction stride (sample every Nth frame, default: 5).",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Maximum frames to extract per video (default: unlimited).",
    )
    return parser.parse_args()

def load_json_annotations(json_path: Path) -> dict[int, list[dict]]:
    """Load the per‑frame annotation list from a tracking JSON file.

    Returns a mapping ``frame_id -> list of track dicts`` where each track dict
    contains ``person_id``, ``bbox`` (x1, y1, x2, y2) and ``center``.
    Handles both 0-indexed and 1-indexed tracking JSON formats.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure deterministic ordering by frame_id
    frame_map: dict[int, list[dict]] = {}
    min_fid = None
    for entry in data:
        fid = int(entry.get("frame_id", 0))
        frame_map.setdefault(fid, []).extend(entry.get("tracks", []))
        if min_fid is None or fid < min_fid:
            min_fid = fid
    if min_fid == 1 and 0 not in frame_map:
        # Shift 1-indexed frame IDs to 0-indexed to align with cv2 frame_idx
        frame_map = {fid - 1: trks for fid, trks in frame_map.items()}
    return frame_map

def write_yolo_label(label_path: Path, tracks: list[dict], img_w: int, img_h: int) -> None:
    """Write a YOLO‑v8 ``.txt`` label file.

    Each line: ``class_id x_center_norm y_center_norm width_norm height_norm``.
    For our use‑case the class id is always 0 (person).
    """
    with label_path.open("w", encoding="utf-8") as f:
        for trk in tracks:
            # bbox format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = trk["bbox"]
            # Convert to YOLO normalized format
            bw = x2 - x1
            bh = y2 - y1
            xc = x1 + bw / 2.0
            yc = y1 + bh / 2.0
            # Normalized values in [0, 1]
            xc_norm = xc / img_w
            yc_norm = yc / img_h
            bw_norm = bw / img_w
            bh_norm = bh / img_h
            f.write(f"0 {xc_norm:.6f} {yc_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    json_dir = args.json_dir.resolve()
    out_root = args.output_dir.resolve()

    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Optional: clean previous contents
    if any(out_root.iterdir()):
        # Keep only folders we are about to populate, remove stray files
        for child in out_root.iterdir():
            if child.is_file():
                child.unlink()

    video_paths = sorted(p for p in video_dir.rglob("*.mp4") if p.is_file())
    if not video_paths:
        sys.exit(f"[prepare_training_dataset] No MP4 videos found in {video_dir}")

    for video_path in video_paths:
        stem = video_path.stem
        json_path = json_dir / f"{stem}.json"
        if not json_path.is_file():
            print(f"[skip] JSON tracking not found for {video_path.name}, expected {json_path.name}")
            continue
        # Load annotations once
        frame_annotations = load_json_annotations(json_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[error] Cannot open video {video_path}")
            continue
        frame_idx = 0
        saved_count = 0
        stride = max(1, args.stride)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                if args.max_frames_per_video is None or saved_count < args.max_frames_per_video:
                    img_h, img_w = frame.shape[:2]
                    # Save image as JPEG (YOLO expects .jpg but any image format works)
                    img_name = f"{stem}_frame{frame_idx:06d}.jpg"
                    img_path = images_dir / img_name
                    cv2.imwrite(str(img_path), frame)
                    # Corresponding label file (may be empty if no tracks)
                    label_path = labels_dir / f"{stem}_frame{frame_idx:06d}.txt"
                    tracks = frame_annotations.get(frame_idx, [])
                    if tracks:
                        write_yolo_label(label_path, tracks, img_w, img_h)
                    else:
                        # Write an empty file – YOLO treats it as no objects
                        label_path.touch()
                    saved_count += 1
            frame_idx += 1
        cap.release()
        print(f"[processed] {video_path.name} -> {frame_idx} total frames, {saved_count} extracted (stride={stride})")

    # Create YOLO data.yaml using clean posix paths
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {out_root.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"  # For a quick start we use the same set for val
        f"nc: 1\n"
        f"names: ['person']\n",
        encoding="utf-8",
    )
    print(f"[done] Dataset prepared in {out_root}. YOLO config written to {data_yaml}")

if __name__ == "__main__":
    main()
