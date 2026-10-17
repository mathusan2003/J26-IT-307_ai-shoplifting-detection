"""Movement history for each tracked person.

Phase 1 records where each person was, when, and which zone or shelf they
were in. Later behaviour checks, including loitering, should read this
history through ``MovementHistoryStore`` instead of parsing tracking JSON
again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tracking_input import observations_from_frame


@dataclass(frozen=True)
class MovementRecord:
    """One observed position for one person at one frame."""

    person_id: int
    frame_id: int
    timestamp: str | None
    position: tuple[int | float, int | float]
    zone: str | None
    sequence: int
    bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape stored in a movement history.

        The box is kept on the record for behaviour checks. It is not part
        of the Phase 1 history JSON.
        """
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "position": [self.position[0], self.position[1]],
            "zone": self.zone,
        }


class MovementHistoryStore:
    """Group tracking observations by person and keep them in frame order.

    Call ``ingest_frame`` once per tracking result as frames arrive, or
    ``ingest_frames`` for a whole sequence. People who are absent from a
    frame keep the history they already have. Records are ordered by
    ``frame_id`` when read, so a frame that arrives late still sits in
    chronological order.
    """

    def __init__(self) -> None:
        self._records: dict[int, list[MovementRecord]] = {}
        self._next_sequence = 0

    def ingest_frame(self, frame: dict[str, Any]) -> None:
        """Append every track in one frame to the matching person history."""
        for observation in observations_from_frame(frame):
            self._append(observation)

    def ingest_frames(self, frames: Iterable[dict[str, Any]]) -> None:
        """Append each frame in the given sequence."""
        for frame in frames:
            self.ingest_frame(frame)

    def person_ids(self) -> list[int]:
        """Return the person IDs that have at least one movement record."""
        return sorted(self._records)

    def records_for(self, person_id: int) -> tuple[MovementRecord, ...]:
        """Return one person's records ordered by frame_id.

        Later detectors should use this view. The store raises ``KeyError``
        when that person has never been seen.
        """
        if person_id not in self._records:
            raise KeyError(person_id)
        ordered = sorted(self._records[person_id], key=_chronological_key)
        return tuple(ordered)

    def export_person(self, person_id: int) -> dict[str, Any]:
        """Return one person's structured movement history."""
        return {
            "person_id": person_id,
            "movement_history": [record.to_dict() for record in self.records_for(person_id)],
        }

    def export(self) -> list[dict[str, Any]]:
        """Return every person's movement history, ordered by person_id."""
        return [self.export_person(person_id) for person_id in self.person_ids()]

    def _append(self, observation: dict[str, Any]) -> None:
        person_id = observation["person_id"]
        bbox = observation.get("bbox")
        record = MovementRecord(
            person_id=person_id,
            frame_id=observation["frame_id"],
            timestamp=observation["timestamp"],
            position=(observation["position"][0], observation["position"][1]),
            zone=observation["zone"],
            sequence=self._next_sequence,
            bbox=None if bbox is None else (bbox[0], bbox[1], bbox[2], bbox[3]),
        )
        self._next_sequence += 1
        self._records.setdefault(person_id, []).append(record)


def build_movement_histories(frames: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the structured movement-history output for a tracking sequence."""
    store = MovementHistoryStore()
    store.ingest_frames(frames)
    return store.export()


def _chronological_key(record: MovementRecord) -> tuple[int, int]:
    """Order records by frame, then by the order they were ingested.

    ``frame_id`` is the timeline shared with tracking. ``sequence`` keeps
    the original track order when two records share a frame.
    """
    return (record.frame_id, record.sequence)
