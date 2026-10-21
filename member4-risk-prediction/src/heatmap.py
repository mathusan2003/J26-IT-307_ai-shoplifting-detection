
from pathlib import Path

import cv2
import numpy as np


def generate_risk_heatmap(
    video_path,
    risk_events,
    output_path="outputs/risk_heatmap.jpg",
):
    """
    Generate a prototype heatmap over a CCTV video frame.

    risk_events example:
    [
        {"x": 320, "y": 240, "risk_score": 80},
        {"x": 350, "y": 260, "risk_score": 60},
    ]

    x and y are pixel coordinates in the selected video frame.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError("Could not read a frame from this video.")

    height, width = frame.shape[:2]
    heat_layer = np.zeros((height, width), dtype=np.float32)

    for event in risk_events:
        x = int(event["x"])
        y = int(event["y"])
        score = float(event["risk_score"])

        if not 0 <= score <= 100:
            raise ValueError("risk_score must be between 0 and 100.")

        if 0 <= x < width and 0 <= y < height:
            radius = max(15, min(width, height) // 10)
            strength = score / 100.0

            cv2.circle(
                heat_layer,
                (x, y),
                radius,
                strength,
                thickness=-1,
            )

    heat_layer = cv2.GaussianBlur(
        heat_layer, (0, 0), sigmaX=25, sigmaY=25
    )
    heat_layer = np.clip(heat_layer, 0, 1)

    heat_colours = cv2.applyColorMap(
        (heat_layer * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    # Blend the heatmap with the original CCTV frame.
    result = cv2.addWeighted(frame, 0.65, heat_colours, 0.35, 0)

    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Could not save heatmap to {output_path}")

    return {
        "output_path": str(output_path),
        "frame_width": width,
        "frame_height": height,
        "events_visualised": len(risk_events),
    }
