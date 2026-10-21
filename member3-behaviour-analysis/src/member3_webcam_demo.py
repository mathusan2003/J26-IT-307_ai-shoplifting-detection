"""Standalone webcam preview for the Member 3 viva.

Member 3 behaviour analysis needs Member 2 tracking JSON
(frame_id, person_id, bbox, center). This folder has no person detector
and no tracker. The live window is a camera preview only. It does not
invent boxes, run the five detectors, or claim live shoplifting detection.
"""

from __future__ import annotations

import cv2

WINDOW_TITLE = "Member 3 Webcam Demo"
CAMERA_INDEX = 0

STATUS_LINES = (
    "Live behaviour analysis: UNAVAILABLE",
    "Member 3 needs tracking JSON (person_id, bbox, center).",
    "This folder has no person detector or tracker.",
    "Webcam display only. No live AI detection.",
    "Press Q to close.",
)


def analysis_status_lines() -> tuple[str, ...]:
    """Return the banner shown on the preview. No detections are invented."""
    return STATUS_LINES


def draw_status(frame):
    """Draw the unavailable-analysis notice on a copy of the camera frame."""
    overlay = frame.copy()
    top = 12
    line_height = 28
    box_height = 16 + line_height * len(STATUS_LINES)
    cv2.rectangle(overlay, (10, top), (overlay.shape[1] - 10, top + box_height), (0, 0, 0), -1)
    shown = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    for index, line in enumerate(STATUS_LINES):
        cv2.putText(
            shown,
            line,
            (20, top + 28 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return shown


def main() -> int:
    capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        print("Error: could not open the webcam on camera index 0.")
        print("Live behaviour analysis was not started.")
        return 1

    print("Webcam preview started. Live behaviour analysis is unavailable.")
    print("Member 3 cannot detect people from this camera without tracking JSON.")
    print("Press Q in the video window to close.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Error: the webcam stopped returning frames.")
                return 1
            try:
                cv2.imshow(WINDOW_TITLE, draw_status(frame))
            except cv2.error:
                print(
                    "Error: OpenCV could not create a window. "
                    "This environment may have opencv-python-headless, which cannot display video."
                )
                return 1
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
