
from pathlib import Path
import json
import random

import cv2
import numpy as np
import pandas as pd
import joblib

from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils import resample


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NORMAL_DIR = Path(r"C:\Research 26-27\archive\normal")
SHOPLIFT_DIR = Path(r"C:\Research 26-27\archive\shoplifting_dataset")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# Keep this order consistent with the dashboard.
FEATURE_NAMES = [
    "person_frame_ratio",
    "mean_person_count",
    "max_person_count",
    "mean_person_confidence",
    "person_count_std",
]


# --------------------------------------------------
# Video feature extraction
# --------------------------------------------------

def extract_features(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise RuntimeError(f"No frames found: {video_path}")

        # Sample at most 12 frames from each clip.
        sample_count = min(12, total_frames)

        indices = np.linspace(
            0,
            total_frames - 1,
            sample_count,
            dtype=int,
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

        if not counts:
            raise RuntimeError(
                f"Could not read sampled frames: {video_path}"
            )

        return {
            "person_frame_ratio": float(
                np.mean(np.array(counts) > 0)
            ),
            "mean_person_count": float(np.mean(counts)),
            "max_person_count": int(max(counts)),
            "mean_person_confidence": float(
                np.mean(confidences)
            ),
            "person_count_std": float(np.std(counts)),
        }

    finally:
        cap.release()


# --------------------------------------------------
# Balance training data only
# --------------------------------------------------

def balance_training_data(train):
    normal_train = train[train["label"] == 0]
    shoplifting_train = train[train["label"] == 1]

    if normal_train.empty or shoplifting_train.empty:
        raise RuntimeError(
            "Training data must contain both classes."
        )

    target_size = max(
        len(normal_train),
        len(shoplifting_train),
    )

    normal_upsampled = resample(
        normal_train,
        replace=len(normal_train) < target_size,
        n_samples=target_size,
        random_state=SEED,
    )

    shoplifting_upsampled = resample(
        shoplifting_train,
        replace=len(shoplifting_train) < target_size,
        n_samples=target_size,
        random_state=SEED,
    )

    balanced_train = pd.concat(
        [normal_upsampled, shoplifting_upsampled],
        ignore_index=True,
    )

    # Shuffle the balanced training rows.
    return balanced_train.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)


# --------------------------------------------------
# Main training workflow
# --------------------------------------------------

def main():
    if not NORMAL_DIR.is_dir():
        raise FileNotFoundError(
            f"Normal folder not found: {NORMAL_DIR}"
        )

    if not SHOPLIFT_DIR.is_dir():
        raise FileNotFoundError(
            f"Shoplifting folder not found: {SHOPLIFT_DIR}"
        )

    normal_videos = sorted(NORMAL_DIR.glob("*.mp4"))
    shoplifting_videos = sorted(SHOPLIFT_DIR.glob("*.mp4"))

    if not normal_videos or not shoplifting_videos:
        raise RuntimeError(
            "No MP4 videos found in one or both folders."
        )

    print(f"Normal videos: {len(normal_videos)}")
    print(f"Shoplifting clips: {len(shoplifting_videos)}")
    print("Loading pretrained YOLO detector...")

    detector = YOLO("yolo11n.pt")

    records = []

    video_groups = [
        (0, normal_videos),
        (1, shoplifting_videos),
    ]

    for label, videos in video_groups:
        for index, path in enumerate(videos, start=1):
            try:
                features = extract_features(path, detector)

                # Clips from one original recording share a group.
                if label == 1 and "__" in path.stem:
                    group = path.stem.split("__")[0]
                else:
                    group = path.stem

                records.append({
                    **features,
                    "label": label,
                    "group": f"{label}_{group}",
                    "video": path.name,
                })

            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")

            if index % 25 == 0:
                print(
                    f"Processed {index}/{len(videos)} "
                    f"for label {label}"
                )

    data = pd.DataFrame(records)

    if data.empty or data["label"].nunique() < 2:
        raise RuntimeError(
            "Need readable videos from both classes."
        )

    # Save extracted features for reproducibility.
    data.to_csv(
        OUTPUT_DIR / "video_features.csv",
        index=False,
    )

    print(f"\nUsable videos: {len(data)}")
    print(
        data["label"]
        .value_counts()
        .rename({0: "normal", 1: "shoplifting"})
    )

    # --------------------------------------------------
    # Group-based train/test split
    # --------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=SEED,
    )

    train_idx, test_idx = next(
        splitter.split(
            data[FEATURE_NAMES],
            data["label"],
            data["group"],
        )
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    if (
        train["label"].nunique() < 2
        or test["label"].nunique() < 2
    ):
        raise RuntimeError(
            "Train/test split does not contain both classes. "
            "Review the source groups and dataset."
        )

    # Check for source-group overlap.
    overlapping_groups = (
        set(train["group"]) & set(test["group"])
    )

    if overlapping_groups:
        raise RuntimeError(
            "Data leakage detected: source groups overlap."
        )

    print("\nOriginal training class counts:")
    print(train["label"].value_counts().sort_index())

    print("\nTest class counts (unchanged):")
    print(test["label"].value_counts().sort_index())

    # Balance training data only.
    balanced_train = balance_training_data(train)

    print("\nBalanced training class counts:")
    print(
        balanced_train["label"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(
        balanced_train[FEATURE_NAMES],
        balanced_train["label"],
    )

    predictions = model.predict(test[FEATURE_NAMES])

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    report = classification_report(
        test["label"],
        predictions,
        labels=[0, 1],
        target_names=["normal", "shoplifting"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        test["label"],
        predictions,
        labels=[0, 1],
    )

    metrics = {
        "model": (
            "YOLO person features + RandomForest "
            "with training-only oversampling"
        ),
        "train_videos": int(len(train)),
        "balanced_train_videos": int(len(balanced_train)),
        "test_videos": int(len(test)),
        "f1_macro": float(
            f1_score(
                test["label"],
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix_labels": [
            "normal",
            "shoplifting",
        ],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "note": (
            "Experimental baseline using person-detection statistics. "
            "It is not a validated theft-behaviour detector. "
            "Oversampling duplicates training examples and does not "
            "add new information. Test data remains unchanged."
        ),
    }

    # Save model and feature order expected by the dashboard.
    model_path = MODEL_DIR / "video_risk_model.joblib"

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_NAMES,
        },
        model_path,
    )

    # Save evaluation metrics.
    metrics_path = OUTPUT_DIR / "video_model_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save test predictions.
    test_results = test[
        ["video", "label", "group"]
    ].copy()

    test_results["predicted_label"] = predictions

    test_results.to_csv(
        OUTPUT_DIR / "video_test_predictions.csv",
        index=False,
    )

    print("\n=== TEST RESULTS ===")

    print(
        classification_report(
            test["label"],
            predictions,
            labels=[0, 1],
            target_names=["normal", "shoplifting"],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "(rows=true, columns=predicted):"
    )
    print(matrix)

    print(f"\nMacro F1: {metrics['f1_macro']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Saved features: "
        f"{OUTPUT_DIR / 'video_features.csv'}"
    )


if __name__ == "__main__":
    main()
