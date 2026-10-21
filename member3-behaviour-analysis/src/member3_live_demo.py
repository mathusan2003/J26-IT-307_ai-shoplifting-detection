"""Live webcam demo: Member 2 person tracks into Member 3 behaviour scores.

Member 2's PersonDetector and ByteTrack run once per frame. Member 3 then
scores the growing per-person history through the existing adapter and
detectors. This is not Member 4 risk JSON, and person boxes alone are not
shoplifting, concealment, or object interaction.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from live_behaviour_view import (
    DEFAULT_WEIGHTS,
    LIVE_DETECT_CONFIDENCE,
    LIVE_IMGSZ,
    LIVE_MAX_DET,
    LIVE_NMS_IOU,
    MEMBER2_ROOT,
    LiveBehaviourSession,
    attach_member2_zones,
    detect_live_people,
    format_overlay_lines,
    load_member2_detector,
    load_member2_overlay,
    load_member2_tracker,
    probe_member2_live_pipeline,
    resolve_member2_live_weights,
)

WINDOW_TITLE = "Retail Multi-Person Tracking"
CAMERA_INDEX = 0
HUD_NOTE = "Member 3 history scores, not shoplifting proof"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Member 2 tracks plus Member 3 live overlay.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means until Q.")
    args = parser.parse_args(argv)

    resolved_weights = resolve_member2_live_weights(args.weights)
    probe = probe_member2_live_pipeline(resolved_weights)
    if not probe["ready"]:
        print("Live behaviour analysis was not started. Missing:")
        for item in probe["missing"]:
            print(f"  - {item}")
        if probe.get("available_weights"):
            print("Weight files found on disk:")
            for item in probe["available_weights"]:
                print(f"  - {item}")
        return 1

    print(f"Using Member 2 weights: {probe['weights']}")
    print(
        f"Live person detect: conf={LIVE_DETECT_CONFIDENCE} imgsz={LIVE_IMGSZ} "
        f"nms={LIVE_NMS_IOU} max_det={LIVE_MAX_DET} (one YOLO, person class only)"
    )
    print("Loading Member 2 person detector and ByteTrack. Person detection is not duplicated.")
    detector = load_member2_detector(resolved_weights)
    load_settings, make_tracker = load_member2_tracker()
    tracker = make_tracker(load_settings())
    as_bbox, draw_tracks, ZoneAnalytics = load_member2_overlay()
    zones_path = MEMBER2_ROOT / "configs" / "zones.json"
    analytics = ZoneAnalytics(zones_path) if zones_path.is_file() else None
    session = LiveBehaviourSession()
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Error: could not open the webcam on camera index {args.camera}.")
        print("Live behaviour analysis was not started.")
        return 1

    print("Live demo started. Press Q to close.")
    print("Loitering needs 15s still inside 40px. Repeated shelf visits need leave+return.")
    print("Abnormal movement needs speed >= 250 px/s or 4 reversals faster than 30 px/s.")
    print(HUD_NOTE)
    frame_id = 0
    started = time.perf_counter()
    last_tick = started
    sized = False
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                print("Error: the webcam stopped returning frames.")
                return 1
            if not sized:
                height, width = image.shape[:2]
                tracker.set_frame_size(width, height)
                if analytics is not None:
                    analytics.configure_frame_size(width, height)
                    session.set_zones(analytics.zones)
                sized = True
            if frame_id < 5:
                mean = float(image.mean()) if image is not None and image.size else -1.0
                print(
                    f"[frame {frame_id} capture] ok={ok} shape={tuple(image.shape)} "
                    f"dtype={image.dtype} mean={mean:.2f}"
                )
                if mean < 8:
                    print(
                        "Webcam frame is nearly black. Close the Member 2 dashboard "
                        "at http://127.0.0.1:8765/ so this demo can use the camera."
                    )
            detect_started = time.perf_counter()
            detect_debug: dict = {}
            people = detect_live_people(detector, image, debug=detect_debug)
            detect_fps = 1.0 / max(1e-6, time.perf_counter() - detect_started)
            raw_people = detect_debug.get("raw") or []
            tracks = [track.to_contract() for track in tracker.update(people)]
            if frame_id < 5:
                print(
                    f"[frame {frame_id} detect] raw={len(raw_people)} "
                    f"keep_whole={len(people)} bytetrack={len(tracks)}"
                )
                for item in raw_people[:8]:
                    print(
                        f"  raw cls=person conf={item['confidence']:.3f} bbox={item['bbox']}"
                    )
            tick = time.perf_counter()
            elapsed = tick - started
            pipeline_fps = 1.0 / max(1e-6, tick - last_tick)
            last_tick = tick
            tracks = attach_member2_zones(tracks, analytics, elapsed, as_bbox)
            views = session.update(frame_id, tracks, elapsed_s=elapsed, detections=people)
            print(
                f"[frame {frame_id} t={elapsed:.2f}s] det={detect_fps:.1f}fps pipe={pipeline_fps:.1f}fps "
                f"people={len(tracks)} "
                + (
                    " ".join(
                        (
                            f"id={int(track['person_id'])} center={track['center']} "
                            f"hist={session.history_length(int(track['person_id']))} "
                            f"behaviours={views.get(int(track['person_id']), {}).get('behaviours', [])} "
                            f"score={views.get(int(track['person_id']), {}).get('live_score', 0)} "
                            f"combined={views.get(int(track['person_id']), {}).get('combined_score', 0)} "
                            f"level={views.get(int(track['person_id']), {}).get('suspicious_level', 'none')}"
                        )
                        for track in tracks
                    )
                    or "no_tracks"
                )
            )
            shown = _draw(
                image,
                tracks,
                views,
                detect_fps,
                pipeline_fps,
                trails,
                draw_tracks,
                frame_id=frame_id,
                raw_count=len(raw_people),
                kept_count=len(people),
            )
            try:
                cv2.imshow(WINDOW_TITLE, shown)
            except cv2.error:
                print(
                    "Error: OpenCV could not create a window. "
                    "This environment may have opencv-python-headless."
                )
                return 1
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                print(f"Stopped after {frame_id} frames (--max-frames). Webcam path opened.")
                return 0
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def _draw(
    image,
    tracks,
    views,
    detect_fps: float,
    pipeline_fps: float,
    trails,
    draw_tracks,
    frame_id: int = 0,
    raw_count: int = 0,
    kept_count: int = 0,
):
    shown = image.copy()
    labels = {}
    for track in tracks:
        person_id = int(track["person_id"])
        zone = track.get("zone")
        if zone:
            labels[person_id] = str(zone)
    draw_tracks(shown, tracks, trails, labels=labels)
    for track in tracks:
        person_id = int(track["person_id"])
        x1, y1, x2, y2 = [int(value) for value in track["bbox"]]
        view = views.get(person_id, {})
        y = min(shown.shape[0] - 8, y2 + 18)
        for line in format_overlay_lines(view):
            cv2.putText(
                shown,
                line,
                (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                shown,
                line,
                (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 18
    cv2.putText(
        shown,
        f"f={frame_id} raw={raw_count} keep={kept_count} tracks={len(tracks)} "
        f"det={detect_fps:.1f} pipe={pipeline_fps:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        shown,
        HUD_NOTE,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return shown


if __name__ == "__main__":
    raise SystemExit(main())
