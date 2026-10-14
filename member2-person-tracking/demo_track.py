"""Overlay ByteTrack IDs and trajectories on video and save annotated output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import (
    DEFAULT_CONFIG,
    load_tracker_settings,
    make_tracker,
    read_json,
    track_detection_sequence,
)
from src.visualize import annotate_frames, iter_video_frames
from src.zone_analytics import ZoneAnalytics

ROOT = Path(__file__).resolve().parent
DEFAULT_ZONES = ROOT / "configs" / "zones.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw person IDs and trajectories, then export annotated video and tracking JSON.",
    )
    parser.add_argument("--detections", type=Path, help="Member 1 detection JSON (object or list)")
    parser.add_argument("--video", type=Path, required=True, help="Input video whose frames match detection frame_id values")
    parser.add_argument("--output-video", type=Path, default=Path("outputs/annotated.mp4"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/tracking_log.json"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES, help="Polygon zones used only on the overlay")
    parser.add_argument("--no-zones", action="store_true", help="Do not draw zone names")
    parser.add_argument("--frame-id-start", type=int, default=None, help="frame_id of the first video frame")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n-person-track.pt"), help="Fine-tuned YOLOv8 person weights when --detections is omitted")
    return parser.parse_args()


def _tracks_from_detector(video: Path, frame_id_start: int, tracker, weights: Path) -> list[dict]:
    from src.detector import PersonDetector
    from src.visualize import require_opencv

    require_opencv()
    import cv2

    detector = PersonDetector(weights)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    results: list[dict] = []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame_id = frame_id_start + index
            detections = detector.detect(image)
            results.append({
                "frame_id": frame_id,
                "tracks": [track.to_contract() for track in tracker.update(detections)],
            })
            index += 1
    finally:
        capture.release()
    return results


def main() -> None:
    args = _parse_args()
    settings = load_tracker_settings(args.config)
    tracker = make_tracker(settings)
    fps = settings.get("fps", 30.0)
    if args.detections is None and args.frame_id_start is None:
        frame_id_start = 0
    elif args.detections is not None:
        sequence = track_detection_sequence(read_json(args.detections), tracker)
        if not sequence:
            raise ValueError("detection JSON contained no frames")
        frame_id_start = args.frame_id_start if args.frame_id_start is not None else int(sequence[0]["frame_id"])
    else:
        sequence = []
        frame_id_start = args.frame_id_start

    if args.detections is None:
        sequence = _tracks_from_detector(args.video, frame_id_start, tracker, args.weights)

    tracks_by_frame = {int(frame["frame_id"]): frame["tracks"] for frame in sequence}
    analytics = None if args.no_zones or not args.zones.exists() else ZoneAnalytics(args.zones)
    written = annotate_frames(
        iter_video_frames(args.video, frame_id_start),
        tracks_by_frame,
        args.output_video,
        fps,
        analytics,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(sequence, indent=2), encoding="utf-8")
    print(f"wrote {written} annotated frames to {args.output_video}")
    print(f"wrote {len(sequence)} tracking frames to {args.output_json}")


if __name__ == "__main__":
    main()
