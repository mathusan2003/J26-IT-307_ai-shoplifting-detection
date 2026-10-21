
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.heatmap import generate_risk_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = Path(
    r"C:\Research 26-27\archive\normal\normal-3.mp4"
)

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "person_density_heatmap.jpg"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

detector = YOLO(str(PROJECT_ROOT / "yolo11n.pt"))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

events = []
frame_count = 0
max_frames = 60

while frame_count < max_frames:
    ok, frame = cap.read()

    if not ok:
        break

    # Analyse every 5th frame
    if frame_count % 5 == 0:
        result = detector.predict(
            source=frame,
            classes=[0],
            imgsz=320,
            verbose=False,
        )[0]

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )

                # Approximate foot position of the detected person
                x = int((x1 + x2) / 2)
                y = int(y2)

                events.append({
                    "x": x,
                    "y": y,
                    "risk_score": 50,
                })

    frame_count += 1

cap.release()

if not events:
    raise RuntimeError(
        "No people detected. Try another video."
    )

result = generate_risk_heatmap(
    video_path=VIDEO_PATH,
    risk_events=events,
    output_path=OUTPUT_PATH,
)

print("Person-density heatmap created!")
print("Video:", VIDEO_PATH.name)
print("Frames analysed:", frame_count)
print("Person detections recorded:", len(events))
print("Output:", OUTPUT_PATH)
print("Note: This shows detection density, not theft risk.")
