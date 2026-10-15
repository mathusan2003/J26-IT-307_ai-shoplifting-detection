"""Run Component 2 from detection JSON or directly over a YOLOv8 video stream."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import DEFAULT_CONFIG, load_tracker_settings, make_tracker, read_json, track_detection_sequence


def process_video(
    video: Path,
    weights: Path,
    output: Path,
    output_video: Path | None = None,
    zones_path: Path | None = None,
    show: bool = False,
) -> None:
    import time

    import cv2
    from ultralytics import YOLO

    from bytetrack import ByteTrack
    from config import TrackingConfig
    from retail_analytics import RetailAnalytics, TelemetryWriter

    config = TrackingConfig.from_zone_file(
        zones_path or Path("configs/zones.json"),
        model_weights=weights,
        output_json=output,
        output_video=output_video or Path("outputs/tracked.mp4"),
        display=show,
    )
    detector = YOLO(str(config.model_weights))
    tracker = ByteTrack(
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
        new_track_threshold=config.new_track_threshold,
        ciou_distance_threshold=config.ciou_distance_threshold,
        track_buffer=config.track_buffer,
    )
    analytics = RetailAnalytics(config.zones, config.fps)
    telemetry = TelemetryWriter(config.output_json)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    writer = None
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            result = detector(frame, classes=[0], conf=config.low_threshold, verbose=False)[0]
            detections = [
                {"bbox": [float(value) for value in box.xyxy[0].tolist()],
                 "confidence": float(box.conf[0])}
                for box in result.boxes
            ]
            tracks = tracker.update(detections)
            timestamp = frame_id / config.fps
            contract_tracks = []
            for track in tracks:
                zone = analytics.update(track.person_id, track.center, timestamp)
                payload = track.telemetry()
                payload["zone"] = zone
                contract_tracks.append(payload)
                x1, y1, x2, y2 = map(int, track.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"ID {track.person_id} {zone or ''}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
                points = track.trajectory[-30:]
                for first, second in zip(points, points[1:]):
                    cv2.line(frame, tuple(first), tuple(second), (255, 180, 0), 2)
            snapshot = analytics.snapshot([track.person_id for track in tracks], timestamp)
            telemetry.append(frame_id, timestamp, contract_tracks, snapshot)
            fps = 1.0 / max(1e-6, time.perf_counter() - started)
            cv2.putText(frame, f"FPS {fps:.1f} | Occupancy {snapshot['active_occupancy']}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if writer is None and output_video is not None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                         config.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {output_video}")
            if writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("Retail Multi-Person Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        telemetry.write()
    print(f"processed {frame_id} frames; wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track people from detection-result JSON.")
    parser.add_argument("input", type=Path, nargs="?", help="JSON detection frames (legacy mode)")
    parser.add_argument("--video", type=Path, help="CCTV video or camera stream (video mode)")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n-person-track.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tracking_log.json"))
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--zones", type=Path, default=Path("configs/zones.json"))
    parser.add_argument("--show", action="store_true", help="Display the live OpenCV dashboard")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.video is not None:
        process_video(args.video, args.weights, args.output, args.output_video, args.zones, args.show)
        return
    if args.input is None:
        parser.error("provide either a detection JSON input or --video")
    results = track_detection_sequence(read_json(args.input), make_tracker(load_tracker_settings(args.config)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
