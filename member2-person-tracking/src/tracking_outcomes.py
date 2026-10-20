"""Build Member 2 tracking outcomes for every person at the same time.

This module does not classify shoplifting. It records identities, motion,
zones, shelf visits, entry/exit, occupancy, timestamps, and track status.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .utils import as_bbox
from .zone_analytics import ZoneAnalytics


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{remainder:06.3f}"


def is_shelf_zone(name: str | None) -> bool:
    return bool(name) and "shelf" in name.lower()


class TrackingOutcomes:
    def __init__(
        self,
        fps: float = 30.0,
        max_missed_frames: int = 30,
        analytics: ZoneAnalytics | None = None,
    ) -> None:
        self.fps = max(1e-6, float(fps))
        self.max_missed_frames = max_missed_frames
        self.analytics = analytics
        self.frames: list[dict[str, Any]] = []
        self.people: dict[int, dict[str, Any]] = {}
        self.entry_events: list[dict[str, Any]] = []
        self.exit_events: list[dict[str, Any]] = []
        self._missed: dict[int, int] = defaultdict(int)
        self._open_visits: dict[int, dict[str, Any]] = {}

    def observe(self, frame_id: int, tracks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        timestamp = frame_id / self.fps
        parsed = [dict(track) for track in tracks]
        visible_ids = {int(track["person_id"]) for track in parsed if track.get("status") != "terminated"}
        explicit = any("status" in track for track in parsed)

        if not explicit:
            parsed = self._infer_status(frame_id, timestamp, parsed, visible_ids)

        occupancy_by_zone: dict[str, int] = defaultdict(int)
        frame_people: list[dict[str, Any]] = []
        active_ids: list[int] = []

        for track in parsed:
            person_id = int(track["person_id"])
            x1, y1, x2, y2 = [float(value) for value in track["bbox"]]
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            centre = list(track.get("center") or [])
            if len(centre) != 2:
                x1, y1, x2, y2 = as_bbox(bbox)
                centre = [(x1 + x2) / 2, (y1 + y2) / 2]
            status = str(track.get("status") or "active")
            zone = self._zone_for(person_id, bbox, timestamp) if status != "terminated" else self._current_zone(person_id)
            dwell = self.analytics.dwell_time(person_id, timestamp) if self.analytics and person_id in self.analytics.people else 0.0
            self._touch_person(person_id, frame_id, timestamp, bbox, centre, zone, status)
            if status == "active":
                active_ids.append(person_id)
                if zone:
                    occupancy_by_zone[zone] += 1
            row = {
                "person_id": person_id,
                "bbox": [round(value) for value in bbox],
                "center": [round(value) for value in centre],
                "zone": zone,
                "status": status,
                "dwell_seconds": round(dwell, 3),
                "missed_frames": int(track.get("missed_frames") or self._missed[person_id]),
            }
            frame_people.append(row)

        frame_record = {
            "frame_id": frame_id,
            "timestamp_seconds": round(timestamp, 3),
            "timestamp": format_timestamp(timestamp),
            "occupancy": {
                "active_people": len(active_ids),
                "temporarily_lost": sum(1 for item in frame_people if item["status"] == "temporarily_lost"),
                "terminated": sum(1 for item in frame_people if item["status"] == "terminated"),
                "by_zone": dict(occupancy_by_zone),
            },
            "people": frame_people,
        }
        self.frames.append(frame_record)
        return frame_record

    def _infer_status(
        self,
        frame_id: int,
        timestamp: float,
        visible: list[dict[str, Any]],
        visible_ids: set[int],
    ) -> list[dict[str, Any]]:
        records = []
        seen = set()
        for track in visible:
            person_id = int(track["person_id"])
            seen.add(person_id)
            self._missed[person_id] = 0
            track = dict(track)
            track["status"] = "active"
            track["missed_frames"] = 0
            records.append(track)
        for person_id, person in list(self.people.items()):
            if person_id in seen or person["final_status"] == "terminated":
                continue
            self._missed[person_id] += 1
            last = person["trajectory"][-1]
            if self._missed[person_id] > self.max_missed_frames:
                records.append({
                    "person_id": person_id, "bbox": last["bbox"], "center": last["center"],
                    "status": "terminated", "missed_frames": self._missed[person_id],
                })
            else:
                records.append({
                    "person_id": person_id, "bbox": last["bbox"], "center": last["center"],
                    "status": "temporarily_lost", "missed_frames": self._missed[person_id],
                })
        return records

    def _zone_for(self, person_id: int, bbox: list[float], timestamp: float) -> str | None:
        if self.analytics is None:
            return None
        return self.analytics.update(person_id, as_bbox(bbox), timestamp).current_zone

    def _current_zone(self, person_id: int) -> str | None:
        if self.analytics is None or person_id not in self.analytics.people:
            return None
        return self.analytics.people[person_id].current_zone

    def _touch_person(
        self,
        person_id: int,
        frame_id: int,
        timestamp: float,
        bbox: list[float],
        centre: list[float],
        zone: str | None,
        status: str,
    ) -> None:
        person = self.people.get(person_id)
        if person is None:
            person = {
                "person_id": person_id,
                "first_frame": frame_id,
                "first_timestamp": format_timestamp(timestamp),
                "first_timestamp_seconds": round(timestamp, 3),
                "trajectory": [],
                "zone_transitions": [],
                "shelf_visits": [],
                "ingress": None,
                "egress": None,
                "final_status": status,
            }
            self.people[person_id] = person
        previous_zone = person["trajectory"][-1]["zone"] if person["trajectory"] else None
        if zone != previous_zone:
            if previous_zone is not None:
                person["zone_transitions"].append({
                    "from": previous_zone,
                    "to": zone,
                    "frame_id": frame_id,
                    "timestamp": format_timestamp(timestamp),
                    "timestamp_seconds": round(timestamp, 3),
                })
                self._close_visit(person, previous_zone, timestamp)
            if zone is not None:
                self._open_visits[person_id] = {"zone": zone, "entered_at": timestamp, "entered_frame": frame_id}
                self._maybe_entry_exit(person, zone, frame_id, timestamp, arriving=True)
            if previous_zone is not None:
                self._maybe_entry_exit(person, previous_zone, frame_id, timestamp, arriving=False)
        person["trajectory"].append({
            "frame_id": frame_id,
            "timestamp": format_timestamp(timestamp),
            "timestamp_seconds": round(timestamp, 3),
            "bbox": [round(value) for value in bbox],
            "center": [round(value) for value in centre],
            "zone": zone,
            "status": status,
        })
        person["last_frame"] = frame_id
        person["last_timestamp"] = format_timestamp(timestamp)
        person["last_timestamp_seconds"] = round(timestamp, 3)
        person["duration_seconds"] = round(timestamp - person["first_timestamp_seconds"], 3)
        person["final_status"] = status

    def _close_visit(self, person: dict[str, Any], zone: str, timestamp: float) -> None:
        visit = self._open_visits.pop(person["person_id"], None)
        if visit is None:
            return
        dwell = max(0.0, timestamp - visit["entered_at"])
        record = {
            "zone": zone,
            "entered_at": format_timestamp(visit["entered_at"]),
            "exited_at": format_timestamp(timestamp),
            "dwell_seconds": round(dwell, 3),
            "is_shelf": is_shelf_zone(zone),
        }
        if is_shelf_zone(zone):
            person["shelf_visits"].append(record)
        person.setdefault("zone_visits", []).append(record)

    def _maybe_entry_exit(
        self,
        person: dict[str, Any],
        zone: str,
        frame_id: int,
        timestamp: float,
        arriving: bool,
    ) -> None:
        start = self.analytics.start_zone if self.analytics else "Entrance"
        end = self.analytics.end_zone if self.analytics else "Checkout"
        event = {
            "person_id": person["person_id"],
            "zone": zone,
            "frame_id": frame_id,
            "timestamp": format_timestamp(timestamp),
            "timestamp_seconds": round(timestamp, 3),
        }
        if arriving and zone == start and person["ingress"] is None:
            person["ingress"] = event
            self.entry_events.append(event)
        if arriving and zone == end and person["egress"] is None:
            person["egress"] = event
            self.exit_events.append(event)

    def close(self) -> None:
        if not self.frames:
            return
        last_time = self.frames[-1]["timestamp_seconds"]
        for person in self.people.values():
            if person["person_id"] in self._open_visits:
                zone = self._open_visits[person["person_id"]]["zone"]
                self._close_visit(person, zone, last_time)
            if person["final_status"] == "temporarily_lost":
                person["final_status"] = "terminated"

    def summary(self) -> dict[str, Any]:
        self.close()
        active_counts = [frame["occupancy"]["active_people"] for frame in self.frames]
        zone_totals: dict[str, int] = defaultdict(int)
        for frame in self.frames:
            for zone, count in frame["occupancy"]["by_zone"].items():
                zone_totals[zone] += count
        people = []
        for person in sorted(self.people.values(), key=lambda item: item["person_id"]):
            people.append({
                "person_id": person["person_id"],
                "persistent_id": person["person_id"],
                "first_frame": person["first_frame"],
                "last_frame": person["last_frame"],
                "first_timestamp": person["first_timestamp"],
                "last_timestamp": person["last_timestamp"],
                "duration_seconds": person["duration_seconds"],
                "status": person["final_status"],
                "trajectory_points": len(person["trajectory"]),
                "trajectory": person["trajectory"],
                "zone_transitions": person["zone_transitions"],
                "shelf_visits": person["shelf_visits"],
                "zone_visits": person.get("zone_visits", []),
                "ingress": person["ingress"],
                "egress": person["egress"],
            })
        return {
            "fps": self.fps,
            "total_frames": len(self.frames),
            "unique_person_ids": [person["person_id"] for person in people],
            "occupancy": {
                "mean_active": round(sum(active_counts) / len(active_counts), 3) if active_counts else 0.0,
                "max_active": max(active_counts, default=0),
                "zone_frame_counts": dict(zone_totals),
            },
            "ingress_count": len(self.entry_events),
            "egress_count": len(self.exit_events),
            "ingress_events": self.entry_events,
            "egress_events": self.exit_events,
            "frames": self.frames,
            "people": people,
        }

    def write(self, path: Path) -> dict[str, Any]:
        payload = self.summary()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
        return payload
