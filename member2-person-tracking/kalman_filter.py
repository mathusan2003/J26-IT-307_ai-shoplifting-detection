"""OpenCV 8D constant-velocity Kalman filter for bounding boxes."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


class BoundingBoxKalmanFilter:
    """Track ``cx, cy, aspect_ratio, height`` and their velocities."""

    def __init__(self, bbox: Sequence[float]) -> None:
        self.filter = cv2.KalmanFilter(8, 4)
        self.filter.transitionMatrix = np.eye(8, dtype=np.float32)
        self.filter.transitionMatrix[:4, 4:] = np.eye(4, dtype=np.float32)
        self.filter.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.filter.measurementMatrix[:, :4] = np.eye(4, dtype=np.float32)
        self.filter.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.filter.processNoiseCov[4:, 4:] *= 10
        self.filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.filter.errorCovPost = np.eye(8, dtype=np.float32)
        self.filter.statePost = np.vstack((self._measurement(bbox), np.zeros((4, 1), dtype=np.float32)))

    @staticmethod
    def _measurement(bbox: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(float, bbox)
        height = max(1.0, y2 - y1)
        width = max(1.0, x2 - x1)
        return np.array(
            [[(x1 + x2) / 2], [(y1 + y2) / 2], [width / height], [height]],
            dtype=np.float32,
        )

    @staticmethod
    def _bbox(state: np.ndarray) -> tuple[float, float, float, float]:
        cx, cy, aspect, height = state[:4, 0]
        height = max(1.0, float(height))
        width = max(1.0, float(aspect) * height)
        return (float(cx - width / 2), float(cy - height / 2),
                float(cx + width / 2), float(cy + height / 2))

    def predict(self) -> tuple[float, float, float, float]:
        return self._bbox(self.filter.predict())

    def update(self, bbox: Sequence[float]) -> tuple[float, float, float, float]:
        return self._bbox(self.filter.correct(self._measurement(bbox)))
