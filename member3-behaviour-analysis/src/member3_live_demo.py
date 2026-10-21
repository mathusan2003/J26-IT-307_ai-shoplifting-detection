"""Live webcam demo: Member 2 person tracks into Member 3 behaviour scores.

repeated_shelf_visit stays in the repository detectors. It is omitted from
this live overlay and from the live score. Object interaction and
possible concealment are shown only when a non-person detection exists.
Member 2's PersonDetector returns person boxes only, so those two
indicators stay hidden on this camera path.

This is not Member 4 risk JSON.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from behaviour import analyse_tracking_frames
from live_behaviour_view import (
    DEFAULT_WEIGHTS,
    MEMBER2_ROOT,
    contract_frame,
    explain_live_detectors,
    has_non_person_objects,
    history_lengths,
    live_view_from_result,
    load_member2_detector,
    load_member2_overlay,
    load_member2_tracker,
    probe_member2_live_pipeline,
)

WINDOW_TITLE = "Retail Multi-Person Tracking"
CAMERA_INDEX = 0
HISTORY_FRAMES = 500
ANALYZE_EVERY = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Member 2 tracks plus Member 3 live overlay.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means until Q.")
    args = parser.parse_args(argv)

    probe = probe_member2_live_pipeline(args.weights)
    if not probe["ready"]:
        print("Live behaviour analysis was not started. Missing:")
        for item in probe["missing"]:
            print(f"  - {item}")
        return 1

    print("Loading Member 2 person detector and ByteTrack. Person detection is not duplicated.")
    detector = load_member2_detector(args.weights)
    load_settings, make_tracker = load_member2_tracker()
    tracker = make_tracker(load_settings())
    as_bbox, draw_tracks, ZoneAnalytics = load_member2_overlay()
    zones_path = MEMBER2_ROOT / "configs" / "zones.json"
    analytics = ZoneAnalytics(zones_path) if zones_path.is_file() else None
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Error: could not open the webcam on camera index {args.camera}.")
        print("Live behaviour analysis was not started.")
        return 1

    print("Live demo started. Press Q to close.")
    print("Live overlay omits repeated_shelf_visit. It is not Member 4 risk_score.")
    print("Time is elapsed seconds on each frame. Loitering needs 15s still inside 40px.")
    print("Abnormal movement needs speed >= 250 px/s or 4 reversals faster than 30 px/s.")
    history: list[dict] = []
    views: dict[int, dict] = {}
    frame_id = 0
    started = time.perf_counter()
    last_tick = started
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                print("Error: the webcam stopped returning frames.")
                return 1
            people = detector.detect(image)
            tracks = [track.to_contract() for track in tracker.update(people)]
            tick = time.perf_counter()
            elapsed = tick - started
            fps = 1.0 / max(1e-6, tick - last_tick)
            last_tick = tick
            frame = contract_frame(frame_id, tracks, timestamp_s=elapsed)
            history.append(frame)
            if len(history) > HISTORY_FRAMES:
                history = history[-HISTORY_FRAMES:]
            lengths = history_lengths(history)
            print(
                f"[frame {frame_id} t={elapsed:.2f}s] people={len(tracks)} "
                + (
                    " ".join(
                        f"id={int(track['person_id'])} center={track['center']} hist={lengths.get(int(track['person_id']), 0)}"
                        for track in tracks
                    )
                    or "no_tracks"
                )
            )
            if frame_id % ANALYZE_EVERY == 0:
                has_objects = has_non_person_objects(people)
                results = analyse_tracking_frames(history)
                views = {
                    item["person_id"]: live_view_from_result(item, has_objects)
                    for item in results
                }
                for report in explain_live_detectors(history):
                    view = views.get(report["person_id"], {})
                    print(
                        f"[detect id={report['person_id']}] hist={report['history_length']} "
                        f"still={report['longest_still_s']:.2f}/{report['loitering_min_s']:.0f}s "
                        f"speed={report['max_speed']:.1f}/{report['abnormal_speed_min']:.0f} "
                        f"rev={report['reversal_count']}/{report['reversal_min']} "
                        f"live={view.get('behaviours', [])} live_score={view.get('live_score', 0)}"
                    )
                    for reason in report["reasons"]:
                        print(f"  why_zero: {reason}")
            shown = _draw(
                image,
                tracks,
                views,
                elapsed,
                fps,
                trails,
                analytics,
                as_bbox,
                draw_tracks,
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


def _draw(image, tracks, views, elapsed: float, fps: float, trails, analytics, as_bbox, draw_tracks):
    shown = image.copy()
    labels = {}
    for track in tracks:
        person_id = int(track["person_id"])
        parts = []
        if analytics is not None:
            zone = analytics.update(person_id, as_bbox(track["bbox"]), elapsed).current_zone
            if zone:
                parts.append(zone)
        view = views.get(person_id, {})
        parts.extend(f"{item['type']}:{item['score']}" for item in view.get("behaviours", []))
        if view.get("live_score"):
            parts.append(f"live={view['live_score']}")
        if parts:
            labels[person_id] = " | ".join(parts)
    draw_tracks(shown, tracks, trails, labels=labels)
    cv2.putText(
        shown,
        f"FPS {fps:.1f} | Occupancy {len(tracks)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return shown


if __name__ == "__main__":
    raise SystemExit(main())
