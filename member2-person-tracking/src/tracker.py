"""ByteTrack-style two-pass tracker with Kalman prediction and CIoU matching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .utils import BBox, as_bbox, box_area, centre_distance, ciou, center, serialise_bbox


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    appearance: object | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "Detection":
        return cls(as_bbox(raw["bbox"]), float(raw["confidence"]), raw.get("appearance"))


@dataclass
class _Kalman1D:
    """A two-state (position, velocity) Kalman filter for one coordinate."""
    position: float
    velocity: float = 0.0
    p00: float = 10.0
    p01: float = 0.0
    p10: float = 0.0
    p11: float = 10.0
    process_position_noise: float = 1.0
    process_velocity_noise: float = 0.1
    measurement_noise: float = 4.0

    def predict(self) -> float:
        self.position += self.velocity
        old_p00, old_p01, old_p10, old_p11 = self.p00, self.p01, self.p10, self.p11
        self.p00 = old_p00 + old_p01 + old_p10 + old_p11 + self.process_position_noise
        self.p01 = old_p01 + old_p11
        self.p10 = old_p10 + old_p11
        self.p11 = old_p11 + self.process_velocity_noise
        return self.position

    def update(self, measurement: float) -> float:
        residual = measurement - self.position
        innovation = self.p00 + self.measurement_noise
        gain_position, gain_velocity = self.p00 / innovation, self.p10 / innovation
        old_p00, old_p01 = self.p00, self.p01
        self.position += gain_position * residual
        self.velocity += gain_velocity * residual
        self.p00 = (1 - gain_position) * old_p00
        self.p01 = (1 - gain_position) * old_p01
        self.p10 -= gain_velocity * old_p00
        self.p11 -= gain_velocity * old_p01
        return self.position


@dataclass
class Track:
    track_id: int
    bbox: BBox
    hits: int = 1
    missed: int = 0
    confidence: float = 0.0
    appearance: object | None = None
    _filters: list[_Kalman1D] | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        values = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        self._filters = [_Kalman1D(value) for value in values]

    def _set_bbox_from_filter(self) -> None:
        assert self._filters is not None
        x, y, width, height = (item.position for item in self._filters)
        width, height = max(1.0, abs(width)), max(1.0, abs(height))
        self.bbox = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)

    def predict(self) -> None:
        assert self._filters is not None
        for item in self._filters:
            item.predict()
        self._set_bbox_from_filter()

    def update(self, detection: Detection) -> None:
        assert self._filters is not None
        x1, y1, x2, y2 = detection.bbox
        for item, measurement in zip(self._filters, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)):
            item.update(measurement)
        self._set_bbox_from_filter()
        self.bbox, self.confidence, self.hits, self.missed = detection.bbox, detection.confidence, self.hits + 1, 0
        if detection.appearance is not None:
            self.appearance = detection.appearance

    def to_contract(self) -> dict[str, object]:
        return {"person_id": self.track_id, "bbox": serialise_bbox(self.bbox), "center": center(self.bbox)}


@dataclass
class _ParkedTrack:
    track: Track
    ttl: int
    bbox: BBox
    camera_id: int = 0


class ByteTracker:
    """High/low association, Kalman hold, CIoU matching, and short re-entry reuse."""

    def __init__(
        self,
        high_threshold: float = 0.2,
        low_threshold: float = 0.1,
        match_iou_threshold: float = 0.2,
        max_missed_frames: int = 60,
        reentry_frames: int = 90,
        new_track_threshold: float | None = None,
    ):
        self.high_threshold, self.low_threshold = high_threshold, low_threshold
        self.match_iou_threshold, self.max_missed_frames = match_iou_threshold, max_missed_frames
        self.reentry_frames = reentry_frames
        self.new_track_threshold = high_threshold if new_track_threshold is None else new_track_threshold
        self.tracks: list[Track] = []
        self._next_id = 1
        self.terminated_this_frame: list[Track] = []
        self._parked: list[_ParkedTrack] = []
        self.frame_size: tuple[int, int] | None = None
        self.camera_id = 0
        self.restored_ids: list[int] = []
        self.handover_ids: list[int] = []
        self.new_ids: list[int] = []
        self.match_ious: list[float] = []

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_size = (int(width), int(height))

    def status_records(self) -> list[dict[str, object]]:
        """Active, temporarily lost, and just-terminated tracks for outcomes logging."""
        records = []
        for track in self.tracks:
            status = "active" if track.missed == 0 else "temporarily_lost"
            records.append(self._record(track, status))
        for track in self.terminated_this_frame:
            records.append(self._record(track, "terminated"))
        return records

    @staticmethod
    def _record(track: Track, status: str) -> dict[str, object]:
        payload = track.to_contract()
        payload.update({"status": status, "missed_frames": track.missed, "hits": track.hits,
                        "confidence": track.confidence})
        return payload

    def update(self, detections: Iterable[Detection | dict[str, object]]) -> list[Track]:
        self.restored_ids, self.handover_ids, self.new_ids, self.match_ious = [], [], [], []
        parsed = [item if isinstance(item, Detection) else Detection.from_mapping(item) for item in detections]
        high = [item for item in parsed if item.confidence >= self.high_threshold]
        low = [item for item in parsed if self.low_threshold <= item.confidence < self.high_threshold]
        for track in self.tracks:
            track.predict()
            track.missed += 1
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_high = self._associate(unmatched_tracks, high, self.match_iou_threshold)
        self._associate(unmatched_tracks, low, max(0.08, self.match_iou_threshold * 0.6))
        self._associate_by_motion(unmatched_tracks, high, unmatched_high)
        self._associate_by_motion(unmatched_tracks, low)
        for index in sorted(unmatched_high):
            detection = high[index]
            if detection.confidence < self.new_track_threshold:
                continue
            restored = self._reenter(detection)
            if restored is None:
                spawned = Track(self._next_id, detection.bbox, confidence=detection.confidence,
                                appearance=detection.appearance)
                self.tracks.append(spawned)
                self.new_ids.append(spawned.track_id)
                self._next_id += 1
        self._age_parked()
        dying = [track for track in self.tracks if track.missed > self.max_missed_frames]
        self.terminated_this_frame = dying
        for track in dying:
            ttl = self.reentry_frames
            if self._near_border(track.bbox):
                ttl = int(self.reentry_frames * 1.5)
            if track.hits >= 3:
                self._parked.append(_ParkedTrack(track=track, ttl=ttl, bbox=track.bbox, camera_id=self.camera_id))
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed_frames]
        return [track for track in self.tracks if track.missed == 0]

    def _cost(self, predicted: BBox, measured: BBox) -> float:
        overlap = ciou(predicted, measured)
        area_a, area_b = box_area(predicted), box_area(measured)
        area_penalty = abs(area_a - area_b) / max(area_a, area_b, 1.0)
        return (1.0 - overlap) + 0.2 * area_penalty

    def _associate(
        self,
        unmatched_tracks: set[int],
        detections: Sequence[Detection],
        threshold: float,
    ) -> set[int]:
        unmatched_detections = set(range(len(detections)))
        if not unmatched_tracks or not detections:
            return unmatched_detections
        track_indices = list(unmatched_tracks)
        pairs: list[tuple[float, int, int]] = []
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            costs = np.array(
                [[self._cost(self.tracks[t].bbox, detections[d].bbox) for d in range(len(detections))]
                 for t in track_indices],
                dtype=float,
            )
            rows, columns = linear_sum_assignment(costs)
            pairs = [(float(ciou(self.tracks[track_indices[r]].bbox, detections[int(c)].bbox)),
                      track_indices[r], int(c)) for r, c in zip(rows, columns)]
        except Exception:
            pairs = sorted(
                ((ciou(self.tracks[t].bbox, detection.bbox), t, d)
                 for t in unmatched_tracks for d, detection in enumerate(detections)),
                reverse=True,
            )
        for overlap, track_index, detection_index in pairs:
            floor = threshold * 0.5 if self.tracks[track_index].missed > 1 else threshold
            if overlap < floor:
                continue
            if track_index in unmatched_tracks and detection_index in unmatched_detections:
                self.tracks[track_index].update(detections[detection_index])
                self.match_ious.append(max(0.0, overlap))
                unmatched_tracks.remove(track_index)
                unmatched_detections.remove(detection_index)
        return unmatched_detections

    def _associate_by_motion(
        self,
        unmatched_tracks: set[int],
        detections: Sequence[Detection],
        unmatched_detections: set[int] | None = None,
    ) -> None:
        """Recover occluded IDs when boxes no longer overlap but Kalman still points nearby."""
        if unmatched_detections is None:
            unmatched_detections = set(range(len(detections)))
        lost = [index for index in unmatched_tracks if self.tracks[index].missed > 0]
        if not lost or not unmatched_detections:
            return
        scale = 80.0
        if self.frame_size:
            scale = max(40.0, 0.12 * (self.frame_size[0] ** 2 + self.frame_size[1] ** 2) ** 0.5)
        candidates = []
        for track_index in lost:
            for detection_index in unmatched_detections:
                distance = centre_distance(self.tracks[track_index].bbox, detections[detection_index].bbox)
                candidates.append((distance, track_index, detection_index))
        for distance, track_index, detection_index in sorted(candidates):
            if distance > scale:
                break
            if track_index in unmatched_tracks and detection_index in unmatched_detections:
                self.tracks[track_index].update(detections[detection_index])
                unmatched_tracks.remove(track_index)
                unmatched_detections.remove(detection_index)

    def _reenter(self, detection: Detection) -> Track | None:
        if not self._parked:
            return None
        from .appearance import appearance_similarity

        best_index, best_score = -1, 0.35
        for index, parked in enumerate(self._parked):
            overlap = ciou(parked.bbox, detection.bbox)
            look = appearance_similarity(getattr(parked.track, "appearance", None), detection.appearance)
            cross_camera = parked.camera_id != self.camera_id
            distance = centre_distance(parked.bbox, detection.bbox)
            nearby = distance < 140 or (self.frame_size and distance < 0.2 * max(self.frame_size))
            score = look if cross_camera else (overlap + 0.55 * max(0.0, look))
            if cross_camera and look >= 0.52:
                pass
            elif not cross_camera and (overlap > 0.2 or (nearby and look >= 0.4) or look >= 0.62):
                score = max(score, overlap, look)
            else:
                continue
            if score > best_score:
                best_score, best_index = score, index
        if best_index < 0:
            return None
        parked = self._parked.pop(best_index)
        parked.track.update(detection)
        self.tracks.append(parked.track)
        self.restored_ids.append(parked.track.track_id)
        if parked.camera_id != self.camera_id:
            self.handover_ids.append(parked.track.track_id)
        return parked.track

    def _age_parked(self) -> None:
        remaining: list[_ParkedTrack] = []
        for parked in self._parked:
            parked.ttl -= 1
            if parked.ttl > 0:
                remaining.append(parked)
        self._parked = remaining

    def _near_border(self, bbox: BBox, margin: float = 40.0) -> bool:
        if not self.frame_size:
            return True
        width, height = self.frame_size
        return bbox[0] <= margin or bbox[1] <= margin or bbox[2] >= width - margin or bbox[3] >= height - margin
