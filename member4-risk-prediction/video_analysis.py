
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
OUTPUT_DIR = BASE_DIR / "outputs" / "video_analysis"


def analyze_video(video_path: str) -> dict:
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")

    output_video = OUTPUT_DIR / "detected_video.mp4"
    output_heatmap = OUTPUT_DIR / "person_heatmap.jpg"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    model = YOLO(str(MODEL_PATH))
    heatmap = np.zeros((height, width), dtype=np.float32)

    frame_count = 0
    people_detected = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame,
                classes=[0],
                conf=0.35,
                verbose=False,
            )

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].cpu().tolist()
                    )

                    x1 = max(0, min(width - 1, x1))
                    x2 = max(0, min(width - 1, x2))
                    y1 = max(0, min(height - 1, y1))
                    y2 = max(0, min(height - 1, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    confidence = float(box.conf[0].cpu())
                    people_detected += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Person {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    # Accumulate heatmap activity near the person's feet.
                    cx = (x1 + x2) // 2
                    cy = y2
                    cv2.circle(
                        heatmap,
                        (cx, cy),
                        25,
                        1.0,
                        thickness=-1,
                    )

            # Overlay the accumulated heatmap on the current frame.
            if heatmap.max() > 0:
                blurred_heatmap = cv2.GaussianBlur(
                    heatmap,
                    (0, 0),
                    sigmaX=25,
                    sigmaY=25,
                )

                normalized_heatmap = cv2.normalize(
                    blurred_heatmap,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                ).astype(np.uint8)
            # Overlay heatmap on the video frame.
            if heatmap.max() > 0:
                blurred_heatmap = cv2.GaussianBlur(
                    heatmap,
                    (0, 0),
                    sigmaX=25,
                    sigmaY=25,
                )

                normalized_heatmap = cv2.normalize(
                    blurred_heatmap,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                ).astype(np.uint8)

                colored_heatmap = cv2.applyColorMap(
                    normalized_heatmap,
                    cv2.COLORMAP_JET,
                )

                frame = cv2.addWeighted(
                    frame,
                    0.50,
                    colored_heatmap,
                    0.50,
                    0,
                )

            writer.write(frame)
            frame_count += 1

    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise ValueError("No readable frames found in the video.")

    # Save a separate heatmap file for analysis only.
    if heatmap.max() > 0:
        normalized = cv2.normalize(
            heatmap,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        colored = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET,
        )

        cv2.imwrite(str(output_heatmap), colored)
    else:
        cv2.imwrite(
            str(output_heatmap),
            np.zeros((height, width, 3), dtype=np.uint8),
        )

    return {
        "frames_processed": frame_count,
        "person_detections": people_detected,
        "video_path": str(output_video),
        "heatmap_path": str(output_heatmap),
    }
