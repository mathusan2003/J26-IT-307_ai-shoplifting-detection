"""Run pretrained YOLOv8 person detection and persistent ByteTrack per video."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VIDEO_DIR = PROJECT_ROOT / "data" / "cleaned_videos"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"
JSON_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "json"
CSV_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
ZONES_PATH = PROJECT_ROOT / "configs" / "zones.json"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--zones", type=Path, default=ZONES_PATH)
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu or 0")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Root directory for videos, JSON, and metrics outputs.",
    )
    return parser.parse_args()


def output_path(root: Path, relative_video: Path, suffix: str) -> Path:
    return (root / relative_video).with_suffix(suffix)


def track_video(model: object, video: Path, relative_video: Path, args: argparse.Namespace) -> dict[str, object]:
    from src.zone_analytics import ZoneAnalytics

    analytics = ZoneAnalytics(args.zones)
    if not analytics.start_zone or not analytics.end_zone:
        raise ValueError(
            f"Zone config must define start_zone and end_zone: {args.zones}"
        )
    missing_zones = {analytics.start_zone, analytics.end_zone} - set(analytics.zones)
    if missing_zones:
        raise ValueError(f"Zone config references unknown zones: {sorted(missing_zones)}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open cleaned video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analytics.configure_frame_size(width, height)
    video_output = output_path(args.output_root / "videos", relative_video, ".mp4")
    json_output = output_path(args.output_root / "json", relative_video, ".json")
    csv_output = output_path(args.output_root / "metrics", relative_video, ".csv")
    video_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {video_output}")

    frames: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    frame_id = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            predictions = model.track(  # type: ignore[attr-defined]
                source=frame,
                persist=True,
                tracker=args.tracker,
                conf=args.confidence,
                iou=args.iou,
                classes=[0],
                device=args.device,
                verbose=False,
            )
            prediction = predictions[0]
            tracks: list[dict[str, object]] = []
            if prediction.boxes is not None and prediction.boxes.id is not None:
                boxes = prediction.boxes.xyxy.cpu().tolist()
                ids = prediction.boxes.id.int().cpu().tolist()
                confidences = prediction.boxes.conf.cpu().tolist()
                for box, person_id, confidence in zip(boxes, ids, confidences):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    track = {
                        "person_id": int(person_id),
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                        "confidence": float(confidence),
                    }
                    state = analytics.update(
                        int(person_id),
                        track["bbox"],
                        frame_id / fps,
                    )
                    track["zone"] = state.current_zone
                    track["start_zone"] = state.current_zone == analytics.start_zone
                    track["end_zone"] = state.current_zone == analytics.end_zone
                    track["reached_end_zone"] = analytics.reached_end_zone(int(person_id))
                    tracks.append(track)
                    csv_rows.append({
                        "frame_id": frame_id + 1,
                        "timestamp_seconds": (frame_id / fps),
                        "person_id": track["person_id"],
                        "bbox": json.dumps(track["bbox"]),
                        "center": json.dumps(track["center"]),
                        "confidence": track["confidence"],
                        "zone": track["zone"],
                        "start_zone": track["start_zone"],
                        "end_zone": track["end_zone"],
                        "reached_end_zone": track["reached_end_zone"],
                    })
            frames.append({"frame_id": frame_id + 1, "tracks": tracks})
            annotated = prediction.plot()
            for zone_name, polygon in analytics.zones.items():
                points = [(int(point[0]), int(point[1])) for point in polygon]
                cv2.polylines(
                    annotated,
                    [np.array(points, dtype="int32")],
                    True,
                    (0, 255, 255) if zone_name in {analytics.start_zone, analytics.end_zone} else (150, 150, 150),
                    2,
                )
                label_point = points[0]
                cv2.putText(
                    annotated,
                    zone_name,
                    label_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            writer.write(annotated)
            if args.show:
                cv2.imshow("YOLOv8 + ByteTrack", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    json_output.write_text(json.dumps(frames, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "frame_id", "timestamp_seconds", "person_id", "bbox", "center",
            "confidence", "zone", "start_zone", "end_zone", "reached_end_zone",
        ]
        csv_writer = csv.DictWriter(output, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(csv_rows)
    elapsed = max(time.perf_counter() - started, 1e-9)
    processing_fps = frame_id / elapsed
    print(f"{relative_video}: {frame_id} frames, {len(csv_rows)} tracks, {processing_fps:.2f} FPS")
    return {
        "video": str(relative_video),
        "frames": frame_id,
        "track_rows": len(csv_rows),
        "processing_fps": round(processing_fps, 3),
    }


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"Cleaned video folder not found: {video_dir}")
    if not args.weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")
    videos = sorted(
        path for path in video_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No supported videos found under {video_dir}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for dataset tracking. "
            "Install the component dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    model = YOLO(str(args.weights))
    summaries = []
    for video in videos:
        summaries.append(track_video(model, video, video.relative_to(video_dir), args))
    metrics_dir = args.output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["video", "frames", "track_rows", "processing_fps"])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
