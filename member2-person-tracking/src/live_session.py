"""Live tracking session for the Component 2 React dashboard."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .appearance import crop_histogram
from .behaviour_bridge import (
    DashboardBehaviour,
    SHOPLIFTING_ALERT_TYPES,
    build_pipeline_status,
    detect_webcam_people,
    empty_behaviour_state,
    format_member3_overlay_lines,
    webcam_person_detector_kwargs,
)
from .object_bridge import DashboardObjects, empty_object_state, log_m3_objects, member3_object_rows
from .detector import ACTIVE_WEIGHTS, PersonDetector, resolve_weights
from .live_eval import LiveEval
from .pipeline import DEFAULT_CONFIG, load_tracker_settings, make_tracker
from .scenario_catalog import SKIP_TOKENS, catalogue_videos, classify_video
from .tracking_outcomes import TrackingOutcomes, format_timestamp, is_shelf_zone
from .visualize import colour_for_id, draw_alerts, draw_occupancy_hud, draw_tracks, draw_zones, require_opencv, zone_caption
from .zone_analytics import ZoneAnalytics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONES = ROOT / "configs" / "zones.json"
DEFAULT_WEIGHTS = ROOT / ACTIVE_WEIGHTS
WEBCAM_SOURCE_ID = "webcam"
WEBCAM_INDEX = 0


def discover_videos(root: Path = ROOT) -> list[Path]:
    folders = [root / "outputs", root / "data"]
    skip = SKIP_TOKENS
    found: list[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        found.extend(path for path in sorted(folder.glob("*.mp4")) if path.is_file())
        if folder.name == "data":
            found.extend(path for path in sorted(folder.rglob("*.mp4")) if path.is_file())
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        key = str(path.resolve())
        if key in seen or any(token in path.name.lower() for token in skip):
            continue
        seen.add(key)
        unique.append(path)
    preferred = [path for path in unique if "input" in path.name.lower()]
    rest = [path for path in unique if path not in preferred]
    return preferred + rest


def is_webcam_source(path: Path | str | None) -> bool:
    if path is None:
        return False
    name = str(path).replace("\\", "/").rstrip("/").split("/")[-1].strip().lower()
    return name in {WEBCAM_SOURCE_ID, "camera", "live", "live webcam"}


def webcam_catalogue_row() -> dict[str, str]:
    return {
        "id": WEBCAM_SOURCE_ID,
        "file": WEBCAM_SOURCE_ID,
        "label": "Live webcam",
        "group": "live",
        "group_label": "Live camera",
        "description": "Laptop camera through the shared Member 2 live session",
        "camera": "Live webcam",
    }


def dashboard_sources(videos: list[Path] | None = None) -> list[dict[str, str]]:
    """Always offer the webcam, then any discovered MP4 clips."""
    return [webcam_catalogue_row(), *catalogue_videos(videos or [])]


def source_label(source: Path | str | None) -> str | None:
    if source is None:
        return None
    if is_webcam_source(source):
        return WEBCAM_SOURCE_ID
    if isinstance(source, Path):
        return source.name
    return str(source)


class LiveSession:
    def __init__(self) -> None:
        require_opencv()
        import cv2

        self.cv2 = cv2
        self._lock = threading.Lock()
        self.paused = True
        self.running = False
        self.source: Path | str | None = None
        self.source_error: str | None = None
        self._clock_origin = time.perf_counter()
        self.weights = resolve_weights(DEFAULT_WEIGHTS)
        settings = load_tracker_settings(DEFAULT_CONFIG)
        self.fps_hint = float(settings.get("fps", 25))
        self._imgsz = 1080
        self.detector: PersonDetector | None = None
        self._detector_live_mode: bool | None = None
        self._detector_weights = self.weights
        self.tracker = make_tracker(settings)
        self.analytics = ZoneAnalytics(DEFAULT_ZONES)
        self.outcomes: TrackingOutcomes | None = None
        self.jpeg: bytes = b""
        self.behaviour_jpeg: bytes = b""
        self.objects_jpeg: bytes = b""
        self._last_frame = None
        self.latency_ms = 0.0
        self.fps = 0.0
        self.detect_fps: float | None = None
        self.frame_index = 0
        self.selected_id: int | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=40)
        self._thread: threading.Thread | None = None
        self._run_id = 0
        self.camera_id = 1
        self.eval = LiveEval()
        self.alerts: list[str] = []
        self.switch_frames = 0
        self.behaviour = DashboardBehaviour()
        self.objects = DashboardObjects()
        self._last_objects: dict[str, Any] = empty_object_state()
        self._last_state: dict[str, Any] = self._empty_state()
        videos = discover_videos()
        if videos:
            self.open_source(videos[0])
            self.paused = True
        else:
            self.open_source(WEBCAM_SOURCE_ID)
            self.paused = False

    def _empty_state(self) -> dict[str, Any]:
        return {
            "camera": "Camera 01 - Aisle 3",
            "source": None,
            "source_error": None,
            "weights": str(self.weights),
            "paused": True,
            "algorithm": "BYTETRACK_ACTIVE",
            "fps": 0.0,
            "latency_ms": 0.0,
            "frame_id": 0,
            "timestamp": "00:00:00.000",
            "summary": {
                "people_now": 0,
                "total_tracked": 0,
                "active": 0,
                "completed": 0,
                "lost": 0,
            },
            "zones": {"Entrance": 0, "Inside Store": 0, "Checkout": 0},
            "occupancy_by_zone": {},
            "tracks": [],
            "events": [],
            "selected": None,
            "status": "paused",
            "alerts": [],
            "scenario": None,
            "eval": {
                "mota": 0.0, "motp": 0.0, "idf1": 0.0, "idsw": 0,
                "restores": 0, "handovers": 0, "frames": 0,
                "reference": "live session (detections as reference, not MOT17 GT)",
            },
            "value": {
                "people_now": 0,
                "persistent_ids": [],
                "trajectory_points": 0,
                "occlusion_held": 0,
                "zone_crossings_now": 0,
                "shelf_visits_now": 0,
                "entries_now": 0,
                "exits_now": 0,
                "fps": 0.0,
            },
            "behaviour": empty_behaviour_state(),
            "objects": empty_object_state(),
            "pipeline": self._pipeline_status(0, empty_behaviour_state()),
        }

    def list_sources(self) -> list[dict[str, str]]:
        return dashboard_sources(discover_videos())

    def resolve_video(self, path: Path | str) -> Path:
        raw = Path(path)
        if raw.is_file():
            return raw
        for candidate in discover_videos():
            if candidate.name == str(path) or str(candidate) == str(path):
                return candidate
        raise FileNotFoundError(f"Video not found: {path}")

    def resolve_source(self, path: Path | str) -> Path | str:
        if is_webcam_source(path):
            return WEBCAM_SOURCE_ID
        return self.resolve_video(path)

    def _is_webcam(self) -> bool:
        return is_webcam_source(self.source)

    def _source_name(self) -> str | None:
        return source_label(self.source)

    def _open_capture(self):
        if self._is_webcam():
            capture = self.cv2.VideoCapture(WEBCAM_INDEX, self.cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = self.cv2.VideoCapture(WEBCAM_INDEX)
            return capture
        return self.cv2.VideoCapture(str(self.source))

    def open_source(self, path: Path | str) -> None:
        source = self.resolve_source(path)
        with self._lock:
            switching = self.source is not None and self.source != source
            parked, next_id = [], 1
            if switching:
                parked = list(self.tracker._parked)
                next_id = self.tracker._next_id
                self.camera_id += 1
            self.source = source
            self.source_error = None
            if self._detector_live_mode is not None and self._detector_live_mode != self._is_webcam():
                self.detector = None
            self._reset_runtime()
            if switching:
                self.tracker._parked = parked
                self.tracker._next_id = next_id
                self.tracker.camera_id = self.camera_id
                self.switch_frames = 40
                self.alerts = ["[Camera Switch]"]
            self._grab_preview()
        self._ensure_thread()

    def _grab_preview(self) -> None:
        if self.source is None:
            return
        capture = self._open_capture()
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            if self._is_webcam():
                self.source_error = (
                    "Could not open the webcam. Close other apps using the camera, then click Resume."
                )
            return
        self.source_error = None
        self._last_frame = frame
        height, width = frame.shape[:2]
        self.tracker.set_frame_size(width, height)
        self.analytics.configure_frame_size(width, height)
        self.behaviour.set_zones(self.analytics.zones)
        canvas = draw_zones(frame, self.analytics)
        prompt = "Live webcam — press Resume Tracking" if self._is_webcam() else "Press Resume Tracking"
        self.cv2.putText(canvas, prompt, (16, 70), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 180, 255), 2)
        ok_jpg, encoded = self.cv2.imencode(".jpg", canvas, [int(self.cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok_jpg:
            jpeg = encoded.tobytes()
            self.jpeg = jpeg
            self.behaviour_jpeg = jpeg
            self.objects_jpeg = jpeg

    def _reset_runtime(self) -> None:
        settings = load_tracker_settings(DEFAULT_CONFIG)
        self.tracker = make_tracker(settings)
        self.analytics = ZoneAnalytics(DEFAULT_ZONES)
        self.outcomes = TrackingOutcomes(fps=self.fps_hint, max_missed_frames=self.tracker.max_missed_frames,
                                         analytics=self.analytics)
        self.frame_index = 0
        self.events.clear()
        self.selected_id = None
        self._run_id += 1
        self.detect_fps = None
        self._clock_origin = time.perf_counter()
        self.behaviour.reset()
        self._last_state = self._empty_state()
        self._last_state["source"] = self._source_name()
        self._last_state["source_error"] = self.source_error
        self._last_state["weights"] = str(self.weights)
        self.eval = LiveEval()
        self.tracker.camera_id = self.camera_id

    def _ensure_detector(self) -> PersonDetector:
        live_mode = self._is_webcam()
        if self.detector is not None and self._detector_live_mode == live_mode:
            return self.detector
        if live_mode:
            coco = ROOT / "models" / "yolov8n.pt"
            weights = coco if coco.is_file() else self.weights
            self.detector = PersonDetector(weights, **webcam_person_detector_kwargs())
            self._detector_weights = weights
        else:
            self.detector = PersonDetector(
                self.weights, confidence=0.20, imgsz=self._imgsz, max_det=300, nms_iou=0.60,
            )
            self._detector_weights = self.weights
        self._detector_live_mode = live_mode
        return self.detector

    def _detect_people(self, frame: Any) -> list[dict[str, Any]]:
        detector = self._ensure_detector()
        if self._is_webcam():
            return detect_webcam_people(detector, frame)
        return list(detector.detect(frame) or [])

    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def control(self, action: str, source: str | None = None, person_id: int | None = None) -> dict[str, Any]:
        if action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
            self._stamp_jpeg("Loading person detector… first start can take a minute on CPU")
            self._ensure_thread()
        elif action == "reset":
            with self._lock:
                self._reset_runtime()
            self.paused = False
            self._ensure_thread()
        elif action == "open" and source:
            self.open_source(source)
            self.paused = False
        elif action == "select":
            self.selected_id = person_id
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._last_state)
            payload["paused"] = self.paused
            payload["events"] = list(self.events)
            payload["source"] = self._source_name()
            payload["source_error"] = self.source_error
            payload["eval"] = self.eval.snapshot()
            payload["alerts"] = list(self.alerts)
            if self._is_webcam():
                payload["camera"] = "Live webcam"
                payload["scenario"] = {
                    "group": "live",
                    "description": "Laptop camera through the shared Member 2 live session",
                }
            if self.source is None:
                payload["status"] = "idle"
            elif self.paused:
                payload["status"] = "paused"
            elif self.detector is None:
                payload["status"] = "loading"
            else:
                payload["status"] = "playing"
            payload["objects"] = dict(self._last_objects)
            payload["pipeline"] = self._pipeline_status(
                int((payload.get("summary") or {}).get("active") or 0),
                payload.get("behaviour") or empty_behaviour_state(),
            )
            return payload

    def mjpeg_bytes(self) -> bytes:
        with self._lock:
            return self.jpeg

    def mjpeg_behaviour_bytes(self) -> bytes:
        with self._lock:
            return self.behaviour_jpeg or self.jpeg

    def mjpeg_objects_bytes(self) -> bytes:
        with self._lock:
            return self.objects_jpeg or self.jpeg

    def _stamp_jpeg(self, text: str) -> None:
        frame = self._last_frame
        if frame is None:
            return
        canvas = frame.copy()
        canvas = draw_zones(canvas, self.analytics)
        self.cv2.putText(canvas, text[:64], (16, 70), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 180, 255), 2)
        ok_jpg, encoded = self.cv2.imencode(".jpg", canvas, [int(self.cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok_jpg:
            jpeg = encoded.tobytes()
            with self._lock:
                self.jpeg = jpeg
                self.behaviour_jpeg = jpeg
                self.objects_jpeg = jpeg

    def _loop(self) -> None:
        capture = None
        trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=48))
        opened_for = -1
        while self.running:
            if self.source is None:
                time.sleep(0.05)
                continue
            if capture is None or opened_for != self._run_id:
                if capture is not None:
                    capture.release()
                capture = self._open_capture()
                opened_for = self._run_id
                trails = defaultdict(lambda: deque(maxlen=48))
                if not capture.isOpened():
                    if self._is_webcam():
                        self.source_error = (
                            "Could not open the webcam. Close other apps using the camera, then click Resume."
                        )
                    time.sleep(0.5)
                    capture = None
                    continue
                self.source_error = None
                native_fps = capture.get(self.cv2.CAP_PROP_FPS) or 0.0
                if native_fps < 1 or (self._is_webcam() and native_fps > 60):
                    native_fps = self.fps_hint
                self.fps_hint = max(8.0, float(native_fps))
                width = int(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH) or 1280)
                height = int(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT) or 720)
                self.tracker.set_frame_size(width, height)
                self.analytics.configure_frame_size(width, height)
                self.behaviour.set_zones(self.analytics.zones)
                if self.outcomes is None:
                    self.outcomes = TrackingOutcomes(
                        fps=self.fps_hint,
                        max_missed_frames=self.tracker.max_missed_frames,
                        analytics=self.analytics,
                    )
            if self.paused:
                if self._is_webcam() and capture is not None and capture.isOpened():
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        self._last_frame = frame
                        canvas = draw_zones(frame, self.analytics)
                        self.cv2.putText(
                            canvas,
                            "Live webcam — press Resume Tracking",
                            (16, 70),
                            self.cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (40, 180, 255),
                            2,
                        )
                        ok_jpg, encoded = self.cv2.imencode(".jpg", canvas, [int(self.cv2.IMWRITE_JPEG_QUALITY), 70])
                        if ok_jpg:
                            jpeg = encoded.tobytes()
                            with self._lock:
                                self.jpeg = jpeg
                                self.behaviour_jpeg = jpeg
                                self.objects_jpeg = jpeg
                                self.source_error = None
                time.sleep(0.04)
                continue
            try:
                if self.detector is None:
                    self._stamp_jpeg("Loading person detector… first start can take a minute on CPU")
                started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    if self._is_webcam():
                        time.sleep(0.05)
                        continue
                    capture.release()
                    capture = self._open_capture()
                    self.frame_index = 0
                    self._clock_origin = time.perf_counter()
                    self.behaviour.reset(self.analytics.zones)
                    continue
                self._last_frame = frame
                raw = frame.copy()
                detect_started = time.perf_counter()
                detections = self._detect_people(frame)
                detect_ms = (time.perf_counter() - detect_started) * 1000
                instant_detect_fps = 1000.0 / max(detect_ms, 1.0)
                for det in detections:
                    det["appearance"] = crop_histogram(frame, det["bbox"])
                self.tracker.update(detections)
                records = self.tracker.status_records()
                frame_record = self.outcomes.observe(self.frame_index, records) if self.outcomes else None
                labels = {}
                for track in records:
                    caption = zone_caption(self.analytics, track.get("zone") if frame_record is None else None)
                    person = None
                    if frame_record:
                        person = next((row for row in frame_record["people"] if row["person_id"] == track["person_id"]), None)
                        caption = zone_caption(self.analytics, person["zone"] if person else None)
                    if caption:
                        labels[int(track["person_id"])] = caption
                canvas = draw_zones(frame, self.analytics)
                canvas = draw_tracks(canvas, records, trails, 48, labels)
                canvas = self._draw_untracked_detections(canvas, detections, records)
                occupancy = frame_record["occupancy"] if frame_record else {
                    "active_people": sum(1 for track in records if track["status"] == "active"),
                    "temporarily_lost": sum(1 for track in records if track["status"] == "temporarily_lost"),
                }
                timestamp = frame_record["timestamp"] if frame_record else format_timestamp(self.frame_index / self.fps_hint)
                canvas = draw_occupancy_hud(canvas, occupancy, timestamp)
                alerts = []
                if self.tracker.restored_ids:
                    alerts.append("[Re-ID Restored]")
                if self.tracker.handover_ids:
                    alerts.append("[Cross-Cam ID Preserved]")
                elif self.switch_frames > 0:
                    alerts.append("[Camera Switch]")
                    self.switch_frames -= 1
                self.alerts = alerts
                canvas = draw_alerts(canvas, alerts)
                self.cv2.putText(canvas, "BYTETRACK_ACTIVE", (16, 70), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 180, 255), 2)
                fps_text = f"FPS: {self.fps:.1f}"
                self.cv2.putText(canvas, fps_text, (16, 100), self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 180, 255), 2)
                matched = sum(1 for track in records if track["status"] == "active")
                lost_n = sum(1 for track in records if track["status"] == "temporarily_lost")
                mean_iou = (sum(self.tracker.match_ious) / len(self.tracker.match_ious)) if self.tracker.match_ious else 0.0
                self.eval.observe(
                    detections=len(detections),
                    matched=matched,
                    lost=lost_n,
                    new_ids=len(self.tracker.new_ids),
                    restores=len(self.tracker.restored_ids),
                    handovers=len(self.tracker.handover_ids),
                    mean_iou=mean_iou,
                )
                elapsed_s = (
                    time.perf_counter() - self._clock_origin
                    if self._is_webcam()
                    else self.frame_index / max(self.fps_hint, 1e-6)
                )
                object_state, object_canvas = self.objects.update(
                    raw, self.frame_index, elapsed_s, self._source_name()
                )
                ok_jpg, encoded = self.cv2.imencode(".jpg", canvas, [int(self.cv2.IMWRITE_JPEG_QUALITY), 62])
                elapsed_ms = (time.perf_counter() - started) * 1000
                instant_fps = 1000.0 / max(elapsed_ms, 1.0)
                with self._lock:
                    if ok_jpg:
                        self.jpeg = encoded.tobytes()
                    self.latency_ms = elapsed_ms
                    self.fps = (0.85 * self.fps + 0.15 * instant_fps) if self.fps else instant_fps
                    self.detect_fps = (
                        (0.85 * self.detect_fps + 0.15 * instant_detect_fps)
                        if self.detect_fps
                        else instant_detect_fps
                    )
                    people_rows = frame_record["people"] if frame_record else []
                    height, width = raw.shape[:2]
                    object_rows = member3_object_rows(object_state)
                    behaviour_state = self.behaviour.update(
                        self.frame_index,
                        records,
                        people_rows,
                        elapsed_s,
                        detections=object_rows,
                        frame_size=(width, height),
                    )
                    self._stamp_alert_crops(raw, records, behaviour_state)
                    if self.frame_index % 15 == 0:
                        raw_dets = object_state.get("detections") or []
                        sample = raw_dets[0] if raw_dets else {}
                        log_m3_objects(
                            "live_session",
                            frame=self.frame_index,
                            detector_ok=not object_state.get("error"),
                            raw=len(raw_dets),
                            passed="None" if object_rows is None else len(object_rows),
                            sample_keys=sorted(sample.keys()) if isinstance(sample, dict) else type(sample).__name__,
                            sample_classes=[
                                item.get("class_name") or item.get("class")
                                for item in raw_dets[:8]
                                if isinstance(item, dict)
                            ],
                            error=object_state.get("error"),
                        )
                        debug = behaviour_state.get("object_debug") or {}
                        log_m3_objects(
                            "member3_adapter",
                            input=behaviour_state.get("object_input"),
                            raw=debug.get("raw"),
                            non_person=debug.get("non_person"),
                            dropped_person=debug.get("dropped_person"),
                            dropped_invalid=debug.get("dropped_invalid"),
                            classes=debug.get("classes"),
                            sample=debug.get("sample"),
                        )
                        log_m3_objects(
                            "behaviour_api",
                            object_evidence=behaviour_state.get("object_evidence"),
                            object_detections=len(behaviour_state.get("object_detections") or []),
                            people=len(behaviour_state.get("people") or []),
                            item_alerts=len(behaviour_state.get("item_alerts") or []),
                        )
                    overlay = self._draw_member3_live_overlay(
                        raw,
                        records,
                        trails,
                        behaviour_state,
                        len(detections),
                        instant_detect_fps,
                        instant_fps,
                    )
                    ok_overlay, overlay_encoded = self.cv2.imencode(
                        ".jpg", overlay, [int(self.cv2.IMWRITE_JPEG_QUALITY), 62]
                    )
                    if ok_overlay:
                        self.behaviour_jpeg = overlay_encoded.tobytes()
                    elif ok_jpg:
                        self.behaviour_jpeg = encoded.tobytes()
                    if object_canvas is not None:
                        ok_obj, obj_encoded = self.cv2.imencode(
                            ".jpg", object_canvas, [int(self.cv2.IMWRITE_JPEG_QUALITY), 62]
                        )
                        if ok_obj:
                            self.objects_jpeg = obj_encoded.tobytes()
                    elif ok_jpg:
                        self.objects_jpeg = encoded.tobytes()
                    self._last_objects = object_state
                    self.frame_index += 1
                    self._push_events(frame_record)
                    self._last_state = self._build_state(
                        frame_record, records, canvas, trails, len(detections), behaviour_state,
                    )
                    self.source_error = None
            except Exception as exc:  # noqa: BLE001 - keep the MJPEG loop alive
                self.source_error = f"Live pipeline error: {exc}"
                time.sleep(0.5)
                continue
            # Yield a little so the UI stays responsive on CPU.
            time.sleep(0.001)

    def _push_events(self, frame_record: dict[str, Any] | None) -> None:
        if not frame_record or not self.outcomes:
            return
        stamp = frame_record["timestamp"]
        for person in frame_record["people"]:
            if person["status"] != "active":
                continue
            pid = person["person_id"]
            history = self.outcomes.people.get(pid, {})
            transitions = history.get("zone_transitions") or []
            if transitions and transitions[-1]["frame_id"] == frame_record["frame_id"]:
                last = transitions[-1]
                if is_shelf_zone(last.get("to")):
                    self.events.appendleft({"time": stamp, "text": f"ID {pid} shelf visit"})
                else:
                    origin = "shelf area" if is_shelf_zone(last.get("from")) else (last.get("from") or "aisle")
                    dest = last.get("to") or "aisle"
                    self.events.appendleft({
                        "time": stamp,
                        "text": f"ID {pid} moved {origin} → {dest}",
                    })
            if history.get("ingress") and history["ingress"]["frame_id"] == frame_record["frame_id"]:
                self.events.appendleft({"time": stamp, "text": f"ID {pid} entered store"})
            if history.get("egress") and history["egress"]["frame_id"] == frame_record["frame_id"]:
                self.events.appendleft({"time": stamp, "text": f"ID {pid} reached checkout / exit"})

    def _draw_untracked_detections(self, frame: Any, detections: list[dict[str, Any]], records: list[dict[str, Any]]) -> Any:
        from .utils import iou as box_iou

        tracked = [row for row in records if row.get("status") == "active"]
        for det in detections:
            bbox = det["bbox"]
            if any(box_iou(tuple(bbox), tuple(row["bbox"])) >= 0.3 for row in tracked):
                continue
            x1, y1, x2, y2 = [int(value) for value in bbox]
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
            self.cv2.putText(frame, "person", (x1, max(18, y1 - 6)), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2)
        return frame

    def _draw_member3_live_overlay(
        self,
        frame: Any,
        records: list[dict[str, Any]],
        trails: dict[int, deque[tuple[int, int]]],
        behaviour_state: dict[str, Any],
        kept_count: int,
        detect_fps: float,
        pipeline_fps: float,
    ) -> Any:
        """Same live view as member3_live_demo.py, on the shared session frame."""
        shown = frame.copy()
        labels: dict[int, str] = {}
        active = [row for row in records if row.get("status") == "active"]
        for track in active:
            zone = track.get("zone")
            if zone:
                labels[int(track["person_id"])] = str(zone)
        overlay_trails = {pid: deque(points, maxlen=48) for pid, points in trails.items()}
        object_rows = [
            item for item in (behaviour_state.get("object_detections") or []) if isinstance(item, dict)
        ]
        for item in object_rows:
            box = item.get("bbox") or []
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(value) for value in box]
            self.cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = str(item.get("class") or "object")
            self.cv2.putText(
                shown,
                label,
                (x1, max(18, y1 - 6)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
                self.cv2.LINE_AA,
            )
        draw_tracks(shown, records, overlay_trails, 48, labels)
        people = {
            int(row["person_id"]): row
            for row in (behaviour_state.get("people") or [])
            if isinstance(row, dict) and "person_id" in row
        }
        for track in active:
            person_id = int(track["person_id"])
            x1, y1, x2, y2 = [int(value) for value in track["bbox"]]
            person = people.get(person_id, {})
            y = min(shown.shape[0] - 8, y2 + 18)
            for line in format_member3_overlay_lines(person, zone=track.get("zone") or person.get("zone")):
                self.cv2.putText(shown, line, (x1, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, self.cv2.LINE_AA)
                self.cv2.putText(shown, line, (x1, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, self.cv2.LINE_AA)
                y += 18
        self.cv2.putText(
            shown,
            (
                f"f={self.frame_index} keep={kept_count} tracks={len(active)} "
                f"obj={len(object_rows)} {behaviour_state.get('object_input') or 'empty'} "
                f"det={detect_fps:.1f} pipe={pipeline_fps:.1f}"
            ),
            (10, 28),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            shown,
            "Member 3 history scores, not shoplifting proof",
            (10, 52),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        return shown

    def _build_state(
        self,
        frame_record: dict[str, Any] | None,
        records: list[dict[str, Any]],
        canvas: Any,
        trails: dict[int, deque[tuple[int, int]]],
        detections_now: int = 0,
        behaviour_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        people = [row for row in (frame_record["people"] if frame_record else []) if row["status"] != "terminated"]
        active = [row for row in people if row["status"] == "active"]
        lost = [row for row in people if row["status"] == "temporarily_lost"]
        stamp = frame_record["timestamp"] if frame_record else format_timestamp(self.frame_index / self.fps_hint)
        moment_events = [event for event in self.events if event.get("time") == stamp]
        by_zone = frame_record["occupancy"]["by_zone"] if frame_record else {}
        tracks = []
        for row in people:
            pid = int(row["person_id"])
            stored = self.outcomes.people.get(pid, {}) if self.outcomes else {}
            colour = colour_for_id(pid)
            tracks.append({
                "id": pid,
                "status": row["status"],
                "zone": row.get("zone") or "Inside Store",
                "confidence": next((track["confidence"] for track in records if track["person_id"] == pid), 0.0),
                "bbox": row["bbox"],
                "center": row["center"],
                "time_in_store": stored.get("duration_seconds", 0.0),
                "first_seen": stored.get("first_timestamp"),
                "last_seen": stored.get("last_timestamp"),
                "trail": list(trails.get(pid, [])),
                "color": f"rgb({colour[2]}, {colour[1]}, {colour[0]})",
                "shelf_visit": is_shelf_zone(row.get("zone")),
                "persistent_id": pid,
            })
        selected = None
        target = self.selected_id or (active[0]["person_id"] if active else None)
        if target is not None:
            selected = next((track for track in tracks if track["id"] == target), None)
            if selected is not None:
                crop = self._crop(canvas, selected["bbox"])
                selected = dict(selected)
                selected["crop_jpeg"] = crop
        if self._is_webcam():
            group, camera, description = (
                "live",
                "Live webcam",
                "Laptop camera through the shared Member 2 live session",
            )
        else:
            group, camera, description = classify_video(self.source) if isinstance(self.source, Path) else ("other", "Camera", "")
        return {
            "camera": camera,
            "scenario": {"group": group, "description": description},
            "alerts": list(self.alerts),
            "eval": self.eval.snapshot(),
            "source": self._source_name(),
            "source_error": self.source_error,
            "weights": str(self._detector_weights or self.weights),
            "paused": self.paused,
            "algorithm": "BYTETRACK_ACTIVE",
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "frame_id": self.frame_index,
            "timestamp": stamp,
            "summary": {
                "people_now": len(active),
                "detections_now": detections_now,
                "active": len(active),
                "lost": len(lost),
            },
            "zones": {
                "Entrance": int(by_zone.get("Entrance", 0)),
                "Inside Store": len(active),
                "Checkout": int(by_zone.get("Checkout", 0)),
            },
            "occupancy_by_zone": by_zone,
            "tracks": tracks,
            "events": moment_events,
            "selected": selected,
            "value": {
                "people_now": len(active),
                "persistent_ids": [int(row["person_id"]) for row in active],
                "trajectory_points": sum(len(trails.get(int(row["person_id"]), [])) for row in active),
                "occlusion_held": len(lost),
                "zone_crossings_now": sum(1 for event in moment_events if "moved" in event.get("text", "")),
                "shelf_visits_now": sum(1 for row in active if is_shelf_zone(row.get("zone"))),
                "entries_now": sum(1 for event in moment_events if "entered" in event.get("text", "")),
                "exits_now": sum(1 for event in moment_events if "checkout" in event.get("text", "") or "exit" in event.get("text", "")),
                "fps": round(self.fps, 1),
            },
            "behaviour": behaviour_state if behaviour_state is not None else empty_behaviour_state(),
            "objects": dict(self._last_objects) if getattr(self, "_last_objects", None) else empty_object_state(),
        }

    def _pipeline_status(self, active_tracks: int, behaviour_state: dict[str, Any]) -> dict[str, Any]:
        shown_weights = self._detector_weights or self.weights
        weights_path = Path(shown_weights) if shown_weights else None
        runtime = self.behaviour.runtime_status() if getattr(self, "behaviour", None) else {
            "loaded": False, "available": False, "error": None, "people_scored": 0,
        }
        playing = (not self.paused) and self.source is not None and self.detector is not None
        detect_fps = None if self.detect_fps is None else round(float(self.detect_fps), 1)
        pipeline_fps = round(float(self.fps), 1) if self.fps else None
        return build_pipeline_status(
            weights=str(shown_weights) if shown_weights else None,
            weights_exist=bool(weights_path and weights_path.is_file()),
            detector_loaded=self.detector is not None,
            paused=self.paused,
            has_source=self.source is not None,
            playing=playing,
            algorithm="BYTETRACK_ACTIVE",
            active_tracks=active_tracks,
            frame_id=self.frame_index,
            timestamp=(getattr(self, "_last_state", None) or {}).get("timestamp") or "00:00:00.000",
            pipeline_fps=pipeline_fps,
            detect_fps=detect_fps,
            behaviour_available=bool(runtime.get("available")),
            behaviour_loaded=bool(runtime.get("loaded")),
            behaviour_error=runtime.get("error") or behaviour_state.get("error"),
            behaviour_people=int(
                runtime.get("people_scored")
                or len(behaviour_state.get("people") or [])
            ),
            source=self._source_name(),
        )

    def _stamp_alert_crops(
        self,
        frame: Any,
        records: list[dict[str, Any]],
        behaviour_state: dict[str, Any],
    ) -> None:
        """Snap only pickup/concealment suspects. Other Member 3 events stay off the Alerts page."""
        boxes: dict[int, Any] = {}
        for row in records or []:
            if not isinstance(row, dict) or row.get("bbox") is None:
                continue
            try:
                boxes[int(row["person_id"])] = row["bbox"]
            except (KeyError, TypeError, ValueError):
                continue
        stamped = []
        for event in list(behaviour_state.get("suspected_alerts") or []) + list(behaviour_state.get("events") or []):
            if not isinstance(event, dict) or event.get("person_id") is None:
                continue
            if event.get("type") not in SHOPLIFTING_ALERT_TYPES:
                continue
            if not event.get("crop_jpeg"):
                try:
                    bbox = boxes.get(int(event["person_id"]))
                except (TypeError, ValueError):
                    bbox = None
                if bbox is not None:
                    event["crop_jpeg"] = self._crop(frame, bbox)
            if event not in stamped:
                stamped.append(event)
        behaviour_state["suspected_alerts"] = stamped

    def _crop(self, frame: Any, bbox: list[int]) -> str | None:
        import base64

        x1, y1, x2, y2 = [max(0, int(value)) for value in bbox]
        h, w = frame.shape[:2]
        x1, x2 = min(x1, w - 1), min(x2, w)
        y1, y2 = min(y1, h - 1), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = frame[y1:y2, x1:x2]
        ok, encoded = self.cv2.imencode(".jpg", patch, [int(self.cv2.IMWRITE_JPEG_QUALITY), 55])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


SESSION: LiveSession | None = None


def get_session() -> LiveSession:
    global SESSION
    if SESSION is None:
        SESSION = LiveSession()
    return SESSION
