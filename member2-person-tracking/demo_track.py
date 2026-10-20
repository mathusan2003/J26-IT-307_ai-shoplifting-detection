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
from src.tracking_outcomes import TrackingOutcomes
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
    parser.add_argument("--outcomes", type=Path, default=Path("outputs/tracking_outcomes.json"),
                        help="Full multi-person outcomes JSON (IDs, trajectories, zones, occupancy, status)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES, help="Polygon zones used only on the overlay")
    parser.add_argument("--no-zones", action="store_true", help="Do not draw zone names")
    parser.add_argument("--frame-id-start", type=int, default=None, help="frame_id of the first video frame")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n-person-track.pt"), help="Fine-tuned YOLOv8 person weights when --detections is omitted")
    parser.add_argument(
        "--ultralytics-track",
        action="store_true",
        help="Use Ultralytics' persistent ByteTrack implementation instead of the custom tracker",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Detector confidence threshold for crowded tracking")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold for tracking")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker configuration")
    parser.add_argument("--show", action="store_true", help="Display the tracking window while processing")
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
    contract: list[dict] = []
    detailed: list[dict] = []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame_id = frame_id_start + index
            detections = detector.detect(image)
            active = tracker.update(detections)
            contract.append({"frame_id": frame_id, "tracks": [track.to_contract() for track in active]})
            detailed.append({"frame_id": frame_id, "tracks": tracker.status_records()})
            index += 1
    finally:
        capture.release()
    return contract, detailed


def _tracks_from_ultralytics(
    video: Path,
    frame_id_start: int,
    weights: Path,
    tracker_config: str,
    confidence: float,
    iou_threshold: float,
    show: bool,
) -> list[dict]:
    """Run persistent Ultralytics ByteTrack and convert results to our contract."""
    from src.visualize import require_opencv

    require_opencv()
    import cv2
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"Could not find YOLO weights: {weights}")
    model = YOLO(str(weights))
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
            predictions = model.track(
                source=image,
                persist=True,
                tracker=tracker_config,
                conf=confidence,
                iou=iou_threshold,
                classes=[0],
                verbose=False,
            )
            prediction = predictions[0]
            tracks: list[dict[str, object]] = []
            if prediction.boxes is not None and prediction.boxes.id is not None:
                boxes = prediction.boxes.xyxy.cpu().numpy()
                track_ids = prediction.boxes.id.int().cpu().numpy()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    tracks.append({
                        "person_id": int(track_id),
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    })
            results.append({"frame_id": frame_id_start + index, "tracks": tracks})
            if show:
                cv2.imshow("Retail Person Tracking", prediction.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()
    return results


def main() -> None:
    args = _parse_args()
    settings = load_tracker_settings(args.config)
    tracker = make_tracker(settings)
    import cv2
    probe = cv2.VideoCapture(str(args.video))
    fps = probe.get(cv2.CAP_PROP_FPS) or settings.get("fps", 30.0)
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    probe.release()
    detailed: list[dict] | None = None
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

    if args.detections is None and args.ultralytics_track:
        sequence = _tracks_from_ultralytics(
            args.video,
            frame_id_start,
            args.weights,
            args.tracker,
            args.conf,
            args.iou,
            args.show,
        )
    elif args.detections is None:
        sequence, detailed = _tracks_from_detector(args.video, frame_id_start, tracker, args.weights)

    analytics = None if args.no_zones or not args.zones.exists() else ZoneAnalytics(args.zones)
    if analytics is not None and width and height:
        analytics.configure_frame_size(width, height)
    outcomes = TrackingOutcomes(fps, int(settings.get("max_missed_frames", 30)), analytics)
    source = detailed if detailed is not None else sequence
    overlay_by_frame: dict[int, list[dict]] = {}
    for frame in source:
        record = outcomes.observe(int(frame["frame_id"]), frame.get("tracks", []))
        overlay_by_frame[int(frame["frame_id"])] = record["people"]
    written = annotate_frames(
        iter_video_frames(args.video, frame_id_start),
        overlay_by_frame,
        args.output_video,
        fps,
        analytics,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(sequence, indent=2), encoding="utf-8")
    payload = outcomes.write(args.outcomes)
    print(f"wrote {written} annotated frames to {args.output_video}")
    print(f"wrote {len(sequence)} tracking frames to {args.output_json}")
    print(f"wrote outcomes for {len(payload['unique_person_ids'])} people to {args.outcomes}")


if __name__ == "__main__":
    main()
