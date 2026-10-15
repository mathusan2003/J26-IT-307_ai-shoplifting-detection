"""Two-stage ByteTrack engine using CIoU and Hungarian association."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from kalman_filter import BoundingBoxKalmanFilter
from ciou_matching import hungarian_matches


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class Track:
    person_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    filter: BoundingBoxKalmanFilter = field(init=False)
    missed: int = 0
    trajectory: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.filter = BoundingBoxKalmanFilter(self.bbox)
        self.trajectory.append(self.center)

    @property
    def center(self) -> list[int]:
        return [round((self.bbox[0] + self.bbox[2]) / 2), round((self.bbox[1] + self.bbox[3]) / 2)]

    def predict(self) -> None:
        self.bbox = self.filter.predict()
        self.missed += 1

    def update(self, detection: Detection) -> None:
        self.bbox = self.filter.update(detection.bbox)
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.missed = 0
        self.trajectory.append(self.center)

    def telemetry(self) -> dict[str, object]:
        return {"person_id": self.person_id, "bbox": [round(value) for value in self.bbox],
                "center": self.center, "trajectory": self.trajectory[-30:]}


class ByteTrack:
    def __init__(self, high_threshold: float = 0.6, low_threshold: float = 0.1,
                 new_track_threshold: float = 0.7, ciou_distance_threshold: float = 0.3,
                 track_buffer: int = 90) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.new_track_threshold = new_track_threshold
        self.ciou_distance_threshold = ciou_distance_threshold
        self.track_buffer = track_buffer
        self.tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _parse(item: Detection | Mapping[str, object]) -> Detection:
        if isinstance(item, Detection):
            return item
        return Detection(tuple(float(value) for value in item["bbox"]), float(item["confidence"]))

    def _associate(self, track_indices: set[int], detections: list[Detection]) -> set[int]:
        if not track_indices or not detections:
            return set(range(len(detections)))
        indices = sorted(track_indices)
        matches, _, unmatched = hungarian_matches(
            [self.tracks[index].bbox for index in indices],
            [item.bbox for item in detections],
            self.ciou_distance_threshold,
        )
        for row, column in matches:
            self.tracks[indices[row]].update(detections[column])
            track_indices.remove(indices[row])
        return unmatched

    def update(self, detections: Iterable[Detection | Mapping[str, object]]) -> list[Track]:
        parsed = [self._parse(item) for item in detections]
        high = [item for item in parsed if item.confidence >= self.high_threshold]
        low = [item for item in parsed if self.low_threshold <= item.confidence < self.high_threshold]
        for track in self.tracks:
            track.predict()
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_high = self._associate(unmatched_tracks, high)
        self._associate(unmatched_tracks, low)
        for index in unmatched_high:
            detection = high[index]
            if detection.confidence >= self.new_track_threshold:
                self.tracks.append(Track(self._next_id, detection.bbox, detection.confidence))
                self._next_id += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.track_buffer]
        return [track for track in self.tracks if track.missed == 0]
