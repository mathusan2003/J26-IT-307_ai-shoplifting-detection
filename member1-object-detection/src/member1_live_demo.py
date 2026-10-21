"""Standalone Member 1 webcam/video prototype.

Uses one pretrained YOLO model. It does not track IDs, score behaviour, or
label shoplifting. Close the Member 2 dashboard before opening the webcam.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from capture import CaptureError, open_source
from object_detector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    IncompatibleWeightsError,
    InferenceError,
    ObjectDetector,
    WeightsNotFoundError,
    load_detector_settings,
    pipeline_detections,
    resolve_object_weights,
)
from visualize import draw_detections

WINDOW_TITLE = "Member 1 Object Detection Prototype"


def main(argv: list[str] | None = None) -> int:
    settings = load_detector_settings()
    parser = argparse.ArgumentParser(description="Member 1 pretrained YOLO object-detection prototype.")
    parser.add_argument("--source", default="webcam", help="webcam or a video file path")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=float(settings["confidence"]))
    parser.add_argument("--imgsz", type=int, default=int(settings["imgsz"]))
    parser.add_argument("--classes", default="", help="Comma-separated class names or ids. Empty = all.")
    parser.add_argument("--device", default=str(settings["device"]))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSONL output path.")
    parser.add_argument("--no-window", action="store_true", help="Skip OpenCV windows (headless).")
    args = parser.parse_args(argv)

    selected = [item.strip() for item in str(args.classes).split(",") if item.strip()]
    try:
        weights = resolve_object_weights(args.weights)
        detector = ObjectDetector(
            weights,
            confidence=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            classes=selected or settings.get("classes") or None,
        )
        capture, source_name = open_source(args.source, camera_index=args.camera)
    except (WeightsNotFoundError, IncompatibleWeightsError, CaptureError, ValueError, RuntimeError) as exc:
        print(f"Member 1 object detection was not started: {exc}")
        return 1

    print(f"Using weights: {detector.weights}")
    print(f"Device: {detector.device}  classes={len(detector.supported_classes())} (COCO pretrained prototype)")
    print("This demo does not detect every retail product and does not score shoplifting.")
    print("Press Q to close." if not args.no_window else "Headless run. No preview window.")

    writer = None
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        writer = args.json_out.open("w", encoding="utf-8")

    frame_id = 0
    started = time.perf_counter()
    infer_fps = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if source_name == "webcam":
                    print("Error: the webcam stopped returning frames.")
                    return 1
                print(f"Stopped at end of video after {frame_id} frames.")
                return 0
            timestamp = time.perf_counter() - started
            infer_started = time.perf_counter()
            try:
                result = detector.detect_objects(frame, frame_id=frame_id, timestamp=timestamp)
            except InferenceError as exc:
                print(exc)
                return 1
            elapsed = time.perf_counter() - infer_started
            instant = 1.0 / max(elapsed, 1e-6)
            infer_fps = instant if infer_fps is None else (0.8 * infer_fps + 0.2 * instant)
            result["pipeline_detections"] = pipeline_detections(result)
            line = (
                f"[frame {frame_id} t={timestamp:.2f}s] det={infer_fps:.1f}fps "
                f"n={len(result['detections'])} "
                + (
                    " ".join(
                        f"{item['class_name']}:{item['confidence']:.2f}"
                        for item in result["detections"][:8]
                    )
                    or "no_objects"
                )
            )
            print(line)
            if writer is not None:
                writer.write(json.dumps(result) + "\n")
            if not args.no_window:
                shown = draw_detections(frame, result, source=source_name, infer_fps=infer_fps)
                try:
                    cv2.imshow(WINDOW_TITLE, shown)
                except cv2.error:
                    print(
                        "OpenCV could not create a window (opencv-python-headless). "
                        "Re-run with --no-window or install opencv-python."
                    )
                    return 1
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                print(f"Stopped after {frame_id} frames (--max-frames).")
                return 0
    finally:
        capture.release()
        if writer is not None:
            writer.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
