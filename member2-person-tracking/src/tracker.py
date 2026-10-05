"""ByteTrack-style two-pass tracker with constant-velocity prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .utils import BBox, as_bbox, center, iou, serialise_bbox


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "Detection":
        return cls(as_bbox(raw["bbox"]), float(raw["confidence"]))


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
    _filters: list[_Kalman1D] | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        values = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        self._filters = [_Kalman1D(value) for value in values]

    def _set_bbox_from_filter(self) -> None:
        assert self._filters is not None
        x, y, width, height = (item.position for item in self._filters)
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

    def to_contract(self) -> dict[str, object]:
        return {"person_id": self.track_id, "bbox": serialise_bbox(self.bbox), "center": center(self.bbox)}


class ByteTracker:
    """High/low confidence association with retained tracks through occlusion."""
    def __init__(self, high_threshold: float = 0.5, low_threshold: float = 0.1,
                 match_iou_threshold: float = 0.3, max_missed_frames: int = 30):
        self.high_threshold, self.low_threshold = high_threshold, low_threshold
        self.match_iou_threshold, self.max_missed_frames = match_iou_threshold, max_missed_frames
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: Iterable[Detection | dict[str, object]]) -> list[Track]:
        parsed = [item if isinstance(item, Detection) else Detection.from_mapping(item) for item in detections]
        high = [item for item in parsed if item.confidence >= self.high_threshold]
        low = [item for item in parsed if self.low_threshold <= item.confidence < self.high_threshold]
        for track in self.tracks:
            track.predict()
            track.missed += 1
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_high = self._associate(unmatched_tracks, high)
        self._associate(unmatched_tracks, low)
        for index in unmatched_high:
            detection = high[index]
            self.tracks.append(Track(self._next_id, detection.bbox, confidence=detection.confidence))
            self._next_id += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed_frames]
        return [track for track in self.tracks if track.missed == 0]

    def _associate(self, unmatched_tracks: set[int], detections: Sequence[Detection]) -> set[int]:
        pairs = sorted(((iou(self.tracks[t].bbox, detection.bbox), t, d)
                        for t in unmatched_tracks for d, detection in enumerate(detections)), reverse=True)
        unmatched_detections = set(range(len(detections)))
        for overlap, track_index, detection_index in pairs:
            if overlap < self.match_iou_threshold:
                break
            if track_index in unmatched_tracks and detection_index in unmatched_detections:
                self.tracks[track_index].update(detections[detection_index])
                unmatched_tracks.remove(track_index)
                unmatched_detections.remove(detection_index)
        return unmatched_detections
