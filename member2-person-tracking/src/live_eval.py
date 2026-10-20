"""Running MOT-style scores for a live clip without official MOT GT.

Detections in the current frame are the reference. These numbers are for
panel demonstration of ID continuity, not MOT17 leaderboard scores.
"""
from __future__ import annotations


class LiveEval:
    def __init__(self) -> None:
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.id_switches = 0
        self.identity_tp = 0
        self.iou_sum = 0.0
        self.iou_count = 0
        self.restores = 0
        self.handovers = 0
        self.frames = 0

    def observe(
        self,
        detections: int,
        matched: int,
        lost: int,
        new_ids: int,
        restores: int,
        handovers: int,
        mean_iou: float,
    ) -> None:
        self.frames += 1
        self.true_positives += matched
        self.false_positives += max(0, new_ids)
        self.false_negatives += max(0, lost)
        self.id_switches += max(0, new_ids)
        self.identity_tp += matched + restores
        self.restores += restores
        self.handovers += handovers
        if mean_iou > 0:
            self.iou_sum += mean_iou
            self.iou_count += 1
        _ = detections

    def snapshot(self) -> dict[str, float | int]:
        truth = self.true_positives + self.false_negatives
        mota = 1.0 - (self.false_positives + self.false_negatives + self.id_switches) / truth if truth else 0.0
        mota = max(-1.0, min(1.0, mota))
        motp = self.iou_sum / self.iou_count if self.iou_count else 0.0
        predicted = self.true_positives + self.false_positives
        denom = 2 * self.identity_tp + (predicted - self.identity_tp) + (truth - self.identity_tp)
        idf1 = (2 * self.identity_tp / denom) if denom else 0.0
        idf1 = max(0.0, min(1.0, idf1))
        return {
            "mota": round(mota, 3),
            "motp": round(motp, 3),
            "idf1": round(idf1, 3),
            "idsw": int(self.id_switches),
            "restores": int(self.restores),
            "handovers": int(self.handovers),
            "frames": int(self.frames),
            "reference": "live session (detections as reference, not MOT17 GT)",
        }
