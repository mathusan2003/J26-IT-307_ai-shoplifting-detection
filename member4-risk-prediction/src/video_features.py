
import cv2
import numpy as np


FEATURE_NAMES = [
    "person_frame_ratio",
    "mean_person_count",
    "max_person_count",
    "mean_person_confidence",
    "person_count_std",
]


def extract_features(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    sample_count = min(12, total_frames)
    indices = np.linspace(
        0, total_frames - 1, sample_count, dtype=int
    )

    counts = []
    confidences = []

    try:
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
    finally:
        cap.release()

    if not counts:
        raise RuntimeError("Could not read sampled video frames.")

    return {
        "person_frame_ratio": float(
            np.mean(np.array(counts) > 0)
        ),
        "mean_person_count": float(np.mean(counts)),
        "max_person_count": int(max(counts)),
        "mean_person_confidence": float(np.mean(confidences)),
        "person_count_std": float(np.std(counts)),
    }
