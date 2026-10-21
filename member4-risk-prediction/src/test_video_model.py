
from pathlib import Path

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "video_risk_model.joblib"
VIDEO_PATH = Path(
    r"C:\Research 26-27\archive\normal\normal-3.mp4"
)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

saved = joblib.load(MODEL_PATH)
classifier = saved["model"]
feature_names = saved["features"]

detector = YOLO("yolo11n.pt")


def extract_features(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(12, total_frames)

    if sample_count <= 0:
        cap.release()
        raise RuntimeError("Video has no readable frames.")

    indices = np.linspace(
        0, total_frames - 1, sample_count, dtype=int
    )

    counts = []
    confidences = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()

        if not ok:
            continue

        result = detector.predict(
            source=frame,
            classes=[0],
            imgsz=320,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            counts.append(0)
            confidences.append(0.0)
        else:
            scores = result.boxes.conf.cpu().numpy()
            counts.append(len(scores))
            confidences.append(float(scores.mean()))

    cap.release()

    if not counts:
        raise RuntimeError("Could not read video frames.")

    return {
        "person_frame_ratio": float(
            np.mean(np.array(counts) > 0)
        ),
        "mean_person_count": float(np.mean(counts)),
        "max_person_count": int(max(counts)),
        "mean_person_confidence": float(np.mean(confidences)),
        "person_count_std": float(np.std(counts)),
    }


features = extract_features(VIDEO_PATH)
import pandas as pd
X = pd.DataFrame([features], columns=feature_names)

prediction = int(classifier.predict(X)[0])
probabilities = classifier.predict_proba(X)[0]
class_index = list(classifier.classes_).index(prediction)
confidence = float(probabilities[class_index])

label = "Shoplifting" if prediction == 1 else "Normal"

print("\n=== VIDEO TEST RESULT ===")
print("Video:", VIDEO_PATH.name)
print("Predicted class:", label)
print(f"Model score for predicted class: {confidence:.1%}")
print("Extracted features:", features)
print(
    "\nNote: This is an exploratory video classifier, "
    "not proof of theft or actual behaviour detection."
)
