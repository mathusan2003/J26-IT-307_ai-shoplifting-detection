"""Live tracking session for the Component 2 React dashboard."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .appearance import crop_histogram
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


class LiveSession:
    def __init__(self) -> None:
        require_opencv()
        import cv2

        self.cv2 = cv2
        self._lock = threading.Lock()
        self.paused = True
        self.running = False
        self.source: Path | None = None
        self.weights = resolve_weights(DEFAULT_WEIGHTS)
        settings = load_tracker_settings(DEFAULT_CONFIG)
        self.fps_hint = float(settings.get("fps", 25))
        self._imgsz = 1080
        self.detector: PersonDetector | None = None
        self.tracker = make_tracker(settings)
        self.analytics = ZoneAnalytics(DEFAULT_ZONES)
        self.outcomes: TrackingOutcomes | None = None
        self.jpeg: bytes = b""
        self.latency_ms = 0.0
        self.fps = 0.0
        self.frame_index = 0
        self.selected_id: int | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=40)
        self._thread: threading.Thread | None = None
        self._last_state: dict[str, Any] = self._empty_state()
        self._run_id = 0
        self.camera_id = 1
        self.eval = LiveEval()
        self.alerts: list[str] = []
        self.switch_frames = 0
        videos = discover_videos()
        if videos:
            self.open_source(videos[0])
            self.paused = True

    def _empty_state(self) -> dict[str, Any]:
        return {
            "camera": "Camera 01 - Aisle 3",
            "source": None,
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
        }

    def list_sources(self) -> list[dict[str, str]]:
        return catalogue_videos(discover_videos())

    def resolve_video(self, path: Path | str) -> Path:
        raw = Path(path)
        if raw.is_file():
            return raw
        for candidate in discover_videos():
            if candidate.name == str(path) or str(candidate) == str(path):
                return candidate
        raise FileNotFoundError(f"Video not found: {path}")

    def open_source(self, path: Path | str) -> None:
        source = self.resolve_video(path)
        with self._lock:
            switching = self.source is not None and self.source != source
            parked, next_id = [], 1
            if switching:
                parked = list(self.tracker._parked)
                next_id = self.tracker._next_id
                self.camera_id += 1
            self.source = source
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
        capture = self.cv2.VideoCapture(str(self.source))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return
        height, width = frame.shape[:2]
        self.tracker.set_frame_size(width, height)
        self.analytics.configure_frame_size(width, height)
        canvas = draw_zones(frame, self.analytics)
        self.cv2.putText(canvas, "Press Resume Tracking", (16, 70), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 180, 255), 2)
        ok_jpg, encoded = self.cv2.imencode(".jpg", canvas, [int(self.cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok_jpg:
            self.jpeg = encoded.tobytes()

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
        self._last_state = self._empty_state()
        self._last_state["source"] = self.source.name if self.source else None
        self._last_state["weights"] = str(self.weights)
        self.eval = LiveEval()
        self.tracker.camera_id = self.camera_id

    def _ensure_detector(self) -> PersonDetector:
        if self.detector is None:
            self.detector = PersonDetector(
                self.weights, confidence=0.20, imgsz=self._imgsz, max_det=300, nms_iou=0.60,
            )
        return self.detector

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
            payload["source"] = self.source.name if self.source else None
            payload["eval"] = self.eval.snapshot()
            payload["alerts"] = list(self.alerts)
            if self.paused:
                payload["status"] = "paused"
            elif self.detector is None:
                payload["status"] = "loading"
            else:
                payload["status"] = "playing"
            return payload

    def mjpeg_bytes(self) -> bytes:
        with self._lock:
            return self.jpeg

    def _loop(self) -> None:
        capture = None
        trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=48))
        opened_for = -1
        while self.running:
            if self.paused or self.source is None:
                time.sleep(0.05)
                continue
            if capture is None or opened_for != self._run_id:
                if capture is not None:
                    capture.release()
                capture = self.cv2.VideoCapture(str(self.source))
                opened_for = self._run_id
                trails = defaultdict(lambda: deque(maxlen=48))
                if not capture.isOpened():
                    time.sleep(0.5)
                    capture = None
                    continue
                native_fps = capture.get(self.cv2.CAP_PROP_FPS) or self.fps_hint
                self.fps_hint = max(8.0, float(native_fps))
                width = int(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH) or 1280)
                height = int(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT) or 720)
                self.tracker.set_frame_size(width, height)
                self.analytics.configure_frame_size(width, height)
                if self.outcomes is None:
                    self.outcomes = TrackingOutcomes(
                        fps=self.fps_hint,
                        max_missed_frames=self.tracker.max_missed_frames,
                        analytics=self.analytics,
                    )
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                capture.release()
                capture = self.cv2.VideoCapture(str(self.source))
                self.frame_index = 0
                continue
            detections = self._ensure_detector().detect(frame)
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
            ok_jpg, encoded = self.cv2.imencode(".jpg", canvas, [int(self.cv2.IMWRITE_JPEG_QUALITY), 62])
            elapsed_ms = (time.perf_counter() - started) * 1000
            instant_fps = 1000.0 / max(elapsed_ms, 1.0)
            with self._lock:
                if ok_jpg:
                    self.jpeg = encoded.tobytes()
                self.latency_ms = elapsed_ms
                self.fps = (0.85 * self.fps + 0.15 * instant_fps) if self.fps else instant_fps
                self.frame_index += 1
                self._push_events(frame_record)
                self._last_state = self._build_state(frame_record, records, canvas, trails, len(detections))
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

    def _build_state(
        self,
        frame_record: dict[str, Any] | None,
        records: list[dict[str, Any]],
        canvas: Any,
        trails: dict[int, deque[tuple[int, int]]],
        detections_now: int = 0,
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
        group, camera, description = classify_video(self.source) if self.source else ("other", "Camera", "")
        return {
            "camera": camera,
            "scenario": {"group": group, "description": description},
            "alerts": list(self.alerts),
            "eval": self.eval.snapshot(),
            "source": self.source.name if self.source else None,
            "weights": str(self.weights),
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
        }

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
