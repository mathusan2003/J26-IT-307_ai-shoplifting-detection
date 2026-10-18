"""Run Pretrained YOLOv8 + ByteTrack baseline on a CCTV video.

Produces:
1. Annotated MP4 video with bounding boxes and persistent track IDs.
2. Tracking CSV (frame_id, track_id, x1, y1, x2, y2, confidence).
3. Contract-compliant JSON (frame_id, tracks[].person_id, bbox, center) for Member 3.
4. Summary evaluation metrics (processing FPS, unique IDs tracked, track continuity).
"""
from pathlib import Path
import argparse
import csv
import json
import time
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "baseline"
DEFAULT_WEIGHTS = ROOT / "models" / "yolov8n.pt"


def main():
    parser = argparse.ArgumentParser(description="Pretrained YOLOv8 + ByteTrack baseline tracking")
    parser.add_argument("--video", required=True, help="Path to input CCTV video")
    parser.add_argument("--output", default="outputs/baseline", help="Directory for output files")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Detector weights (default: models/yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for person detector")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for NMS")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (default: bytetrack.yaml)")
    parser.add_argument("--device", default=None, help="Device: '0' for CUDA GPU, 'cpu' for CPU (auto-detected if None)")
    args = parser.parse_args()

    # Resolve video path
    video_path = Path(args.video)
    if not video_path.is_absolute():
        if video_path.exists():
            video_path = video_path.resolve()
        elif (ROOT / video_path).exists():
            video_path = ROOT / video_path

    if not video_path.exists():
        # Check in data/remaining_videos, data/representative_videos, or Dataset/videos
        fallbacks = [
            ROOT / "data" / "remaining_videos" / args.video,
            ROOT / "data" / "representative_videos" / args.video,
            ROOT / "Dataset" / "videos" / args.video,
        ]
        found = False
        for fb in fallbacks:
            if fb.exists():
                video_path = fb
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Video file not found: {args.video} (checked {video_path})")

    # Resolve output directory
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve weights
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        if weights_path.exists():
            weights_path = weights_path.resolve()
        elif (ROOT / weights_path).exists():
            weights_path = ROOT / weights_path
    if not weights_path.exists():
        fallback_weights = ROOT / "models" / "yolov8n.pt"
        if fallback_weights.exists():
            weights_path = fallback_weights
        else:
            weights_path = Path("yolov8n.pt")  # Ultralytics will auto-download if missing

    # Resolve tracker config
    tracker_cfg = args.tracker
    if not Path(tracker_cfg).is_absolute() and (ROOT / tracker_cfg).exists():
        tracker_cfg = str((ROOT / tracker_cfg).resolve())

    # Determine device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("Baseline Multi-Person Tracking: Pretrained YOLOv8 + ByteTrack")
    print("=" * 60)
    print(f"Input Video : {video_path}")
    print(f"Weights     : {weights_path}")
    print(f"Device      : {device} ({'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == 0 else 'CPU'})")
    print(f"Output Dir  : {output_dir}")
    print("=" * 60)

    # Initialize YOLOv8 model
    model = YOLO(str(weights_path))

    # Read video properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stem = video_path.stem
    output_video_path = output_dir / f"{stem}_tracked.mp4"
    output_csv_path = output_dir / f"{stem}_tracking.csv"
    output_json_path = output_dir / f"{stem}_tracking.json"

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    csv_rows = []
    json_sequence = []
    unique_ids = set()
    frames_with_detections = 0
    id_trajectory_lengths = {}

    start_time = time.perf_counter()

    # Stream track inference
    results = model.track(
        source=str(video_path),
        tracker=tracker_cfg,
        conf=args.conf,
        iou=args.iou,
        classes=[0],  # class 0 = person
        stream=True,
        device=device,
        verbose=False,
    )

    frame_id = 0
    for result in results:
        annotated_frame = result.plot()
        writer.write(annotated_frame)

        frame_tracks = []
        if result.boxes is not None and len(result.boxes) > 0:
            frames_with_detections += 1
            boxes = result.boxes
            for box in boxes:
                # Track ID may be None if not yet assigned by ByteTrack
                track_id = int(box.id[0].item()) if box.id is not None else None
                conf_score = float(box.conf[0].item())
                x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]

                if track_id is not None:
                    unique_ids.add(track_id)
                    id_trajectory_lengths[track_id] = id_trajectory_lengths.get(track_id, 0) + 1

                csv_rows.append({
                    "frame_id": frame_id,
                    "track_id": track_id if track_id is not None else -1,
                    "confidence": round(conf_score, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

                if track_id is not None:
                    cx = round((x1 + x2) / 2, 1)
                    cy = round((y1 + y2) / 2, 1)
                    frame_tracks.append({
                        "person_id": track_id,
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "center": [round(cx), round(cy)],
                    })

        json_sequence.append({
            "frame_id": frame_id,
            "tracks": frame_tracks,
        })

        frame_id += 1
        if frame_id % 50 == 0 or frame_id == total_frames:
            elapsed = time.perf_counter() - start_time
            current_fps = frame_id / max(1e-5, elapsed)
            print(f"Processed frame {frame_id}/{total_frames} ({current_fps:.1f} FPS)...")

    writer.release()
    total_elapsed = time.perf_counter() - start_time
    avg_fps = frame_id / max(1e-5, total_elapsed)

    # Save CSV
    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f, fieldnames=["frame_id", "track_id", "confidence", "x1", "y1", "x2", "y2"]
        )
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    # Save JSON contract
    output_json_path.write_text(json.dumps(json_sequence, indent=2), encoding="utf-8")

    # Evaluation Summary
    print("\n" + "=" * 60)
    print("Baseline Tracking Evaluation Summary")
    print("=" * 60)
    print(f"Total Frames Processed   : {frame_id}")
    print(f"Total Processing Time    : {total_elapsed:.2f} seconds")
    print(f"Average Processing Speed : {avg_fps:.2f} FPS")
    print(f"Frames with People       : {frames_with_detections} ({frames_with_detections / max(1, frame_id) * 100:.1f}%)")
    print(f"Unique Person IDs Tracked: {len(unique_ids)}")
    print(f"Track Continuity Breakdown:")
    for pid, count in sorted(id_trajectory_lengths.items()):
        print(f"  • Person ID {pid:2d}: active for {count:3d} frames ({count / max(1, frame_id) * 100:.1f}% duration)")

    print("-" * 60)
    print(f"Outputs written:")
    print(f"  • Annotated Video : {output_video_path}")
    print(f"  • Detection CSV   : {output_csv_path}")
    print(f"  • API Contract JSON: {output_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
