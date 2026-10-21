"""Open the laptop webcam and show a live preview.

Press q to close the window. This script does not run detection or tracking.
"""

from __future__ import annotations

import cv2


def main() -> int:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: could not open the webcam on camera index 0.")
        return 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Error: the webcam stopped returning frames.")
                return 1
            cv2.imshow("Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
