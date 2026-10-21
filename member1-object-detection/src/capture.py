"""Webcam and video capture helpers for the Member 1 prototype."""
from __future__ import annotations

from pathlib import Path
from typing import Any

WEBCAM_TOKENS = {"webcam", "camera", "live", "0"}


class CaptureError(RuntimeError):
    """Raised when a webcam or video file cannot be opened."""


def is_webcam_source(source: str | Path | None) -> bool:
    if source is None:
        return True
    token = str(source).strip().lower()
    return token in WEBCAM_TOKENS or token.isdigit() and Path(token).suffix == ""


def open_source(source: str | Path | None = "webcam", camera_index: int = 0) -> tuple[Any, str]:
    """Open a webcam or video file. The caller must release the capture."""
    try:
        import cv2
    except ImportError as exc:
        raise CaptureError("Install opencv-python-headless to open a webcam or video.") from exc

    if source is None or is_webcam_source(source):
        index = camera_index
        if source is not None and str(source).strip().isdigit():
            index = int(str(source).strip())
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise CaptureError(
                f"Could not open the webcam on camera index {index}. "
                "Close the Member 2 dashboard or other apps using the camera."
            )
        return capture, "webcam"

    path = Path(source)
    if not path.is_file():
        raise CaptureError(f"Video file not found: {path.resolve()}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise CaptureError(f"Could not read video: {path.resolve()}")
    return capture, str(path)
