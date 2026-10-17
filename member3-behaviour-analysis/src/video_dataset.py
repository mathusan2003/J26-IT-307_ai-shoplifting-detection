"""Catalog, split, and motion features for the Normal / Shoplifting videos.

Labels come only from the parent folder name: ``normal`` or ``shoplifting``.
These clips do not annotate loitering, repeated shelf visits, abnormal
movement, object interaction, or concealment, so this module does not
create those labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = MODULE_ROOT / "datasets" / "shoplifting-video"
CATALOG_PATH = MODULE_ROOT / "datasets" / "video_catalog.json"
FEATURES_PATH = MODULE_ROOT / "datasets" / "video_features.json"
SPLIT_PATH = MODULE_ROOT / "datasets" / "video_split.json"

FOLDER_LABELS = ("normal", "shoplifting")
POSITIVE_LABEL = "shoplifting"
TEMPORAL_BINS = 16
GRID = 4
SAMPLE_SIZE = (160, 120)

BEHAVIOUR_TYPES = (
    "loitering",
    "repeated_shelf_visit",
    "abnormal_movement",
    "object_interaction",
    "possible_concealment",
)


def feature_names() -> list[str]:
    """Names of the fixed motion vector, in model order."""
    names = [f"temporal_bin_{index:02d}" for index in range(TEMPORAL_BINS)]
    names.extend(f"spatial_{index:02d}" for index in range(GRID * GRID))
    names.extend(
        [
            "motion_mean",
            "motion_std",
            "motion_max",
            "motion_p90",
            "late_early_ratio",
            "motion_center_x",
            "motion_center_y",
        ]
    )
    return names


def discover_videos(root: Path | None = None) -> list[Path]:
    """Return video files directly inside the normal and shoplifting folders."""
    video_root = root if root is not None else VIDEO_ROOT
    found: list[Path] = []
    for label in FOLDER_LABELS:
        folder = video_root / label
        if not folder.is_dir():
            continue
        found.extend(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
    return sorted(found, key=lambda path: path.as_posix().lower())


def label_from_folder(path: Path) -> str:
    """Return the folder label. The file name is not allowed to override it."""
    label = path.parent.name
    if label not in FOLDER_LABELS:
        raise ValueError(f"video is not in a normal or shoplifting folder: {path}")
    return label


def missing_expected_names(paths: list[Path]) -> dict[str, list[str]]:
    """Names that a complete 1..90 / 1..93 collection would contain but does not."""
    present = {path.name for path in paths}
    return {
        "normal": [f"normal-{index}.mp4" for index in range(1, 91) if f"normal-{index}.mp4" not in present],
        "shoplifting": [
            f"shoplifting-{index}.mp4" for index in range(1, 94) if f"shoplifting-{index}.mp4" not in present
        ],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Hash one video, decode it once, and keep folder label plus motion features."""
    video_root = root if root is not None else VIDEO_ROOT
    label = label_from_folder(path)
    relative = path.relative_to(video_root).as_posix()
    stem_matches = path.stem.startswith(label)
    record: dict[str, Any] = {
        "video_id": relative,
        "file_name": path.name,
        "label": label,
        "label_source": "parent_folder",
        "behaviour_labels": None,
        "file_name_matches_folder": stem_matches,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "readable": False,
        "frames_decoded": 0,
        "fps": None,
        "width": None,
        "height": None,
        "duration_s": None,
        "error": None,
        "features": None,
    }
    if not stem_matches:
        record["error"] = "file name does not start with its folder label"
        return record
    try:
        decoded = _decode_motion(path)
    except Exception as exc:  # decoder failures stay in the catalog instead of aborting the run
        record["error"] = str(exc)
        return record
    record.update(decoded)
    return record


def build_catalog(root: Path | None = None) -> dict[str, Any]:
    """Inspect every clip under the video root."""
    video_root = root if root is not None else VIDEO_ROOT
    paths = discover_videos(video_root)
    videos = []
    for index, path in enumerate(paths, start=1):
        print(f"inspect {index}/{len(paths)} {path.name}", flush=True)
        videos.append(inspect_video(path, video_root))
    hashes: dict[str, list[str]] = defaultdict(list)
    for video in videos:
        hashes[video["sha256"]].append(video["video_id"])
    duplicates = [
        {"sha256": digest, "video_ids": sorted(ids)}
        for digest, ids in sorted(hashes.items())
        if len(ids) > 1
    ]
    conflicts = []
    by_hash: dict[str, set[str]] = defaultdict(set)
    for video in videos:
        by_hash[video["sha256"]].add(video["label"])
    for digest, labels in sorted(by_hash.items()):
        if len(labels) > 1:
            conflicts.append({"sha256": digest, "labels": sorted(labels)})
    return {
        "video_root": "datasets/shoplifting-video",
        "label_source": "parent_folder",
        "labels": list(FOLDER_LABELS),
        "behaviour_labels": None,
        "behaviour_types_not_annotated": list(BEHAVIOUR_TYPES),
        "missing_expected_names": missing_expected_names(paths),
        "videos": videos,
        "duplicate_hashes": duplicates,
        "hash_label_conflicts": conflicts,
        "counts": {
            "files": len(videos),
            "readable": sum(video["readable"] for video in videos),
            "normal": sum(video["label"] == "normal" for video in videos),
            "shoplifting": sum(video["label"] == "shoplifting" for video in videos),
        },
    }


def split_sizes(total: int) -> tuple[int, int, int]:
    """Return train, validation, and test counts for one class.

    Each holdout is 15% of the class, rounded half up, with at least one
    video when the class has at least three videos. The remainder is train.
    """
    if total < 3:
        raise ValueError("a class needs at least 3 videos to fill train, validation, and test")
    holdout = max(1, int(total * 0.15 + 0.5))
    if holdout * 2 >= total:
        holdout = 1
    return total - 2 * holdout, holdout, holdout


def assign_splits(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Assign each unique readable video to one split.

    The unit is the file hash, so identical copies cannot land in two splits.
    Within a folder label, hashes are sorted and then cut into train,
    validation, and test. No random generator is involved.
    """
    conflict_hashes = set()
    labels_for_hash: dict[str, set[str]] = defaultdict(set)
    for video in videos:
        labels_for_hash[video["sha256"]].add(video["label"])
    for digest, labels in labels_for_hash.items():
        if len(labels) > 1:
            conflict_hashes.add(digest)

    excluded = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video in videos:
        reason = _exclude_reason(video, conflict_hashes)
        if reason is not None:
            excluded.append({"video_id": video["video_id"], "label": video["label"], "reason": reason})
            continue
        groups[video["sha256"]].append(video)

    assigned: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label in FOLDER_LABELS:
        label_groups = [groups[digest] for digest in groups if groups[digest][0]["label"] == label]
        label_groups.sort(key=lambda members: members[0]["sha256"])
        n_train, n_val, _n_test = split_sizes(len(label_groups))
        for index, members in enumerate(label_groups):
            if index < n_train:
                split_name = "train"
            elif index < n_train + n_val:
                split_name = "validation"
            else:
                split_name = "test"
            members = sorted(members, key=lambda video: video["video_id"])
            for offset, video in enumerate(members):
                assigned[split_name].append(
                    {
                        "video_id": video["video_id"],
                        "label": video["label"],
                        "sha256": video["sha256"],
                        "used_for_training": offset == 0,
                    }
                )
    for rows in assigned.values():
        rows.sort(key=lambda item: item["video_id"])
    return {
        "unit": "unique_sha256",
        "label_source": "parent_folder",
        "ratio": "within each label, sort unique hashes; the first block is train, the next 15% is validation, and the last 15% is test",
        "behaviour_labels": None,
        "splits": assigned,
        "excluded": excluded,
        "counts": {
            split_name: {
                "videos": len(rows),
                "used": sum(row["used_for_training"] for row in rows),
                "normal": sum(row["label"] == "normal" and row["used_for_training"] for row in rows),
                "shoplifting": sum(row["label"] == "shoplifting" and row["used_for_training"] for row in rows),
            }
            for split_name, rows in assigned.items()
        },
    }


def features_from_catalog(catalog: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    """Keep one feature row per video that the split uses."""
    by_id = {video["video_id"]: video for video in catalog["videos"]}
    rows = []
    for split_name, items in split["splits"].items():
        for item in items:
            if not item["used_for_training"]:
                continue
            video = by_id[item["video_id"]]
            rows.append(
                {
                    "video_id": video["video_id"],
                    "split": split_name,
                    "label": video["label"],
                    "sha256": video["sha256"],
                    "features": video["features"],
                    "duration_s": video["duration_s"],
                    "width": video["width"],
                    "height": video["height"],
                    "bytes": video["bytes"],
                }
            )
    return {"feature_names": feature_names(), "rows": rows}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _exclude_reason(video: dict[str, Any], conflict_hashes: set[str]) -> str | None:
    if video["sha256"] in conflict_hashes:
        return "same file contents have two folder labels"
    if not video["readable"]:
        return video["error"] or "video could not be decoded"
    if video.get("features") or video.get("has_features"):
        return None
    return "fewer than two frames, so motion could not be measured"


def _decode_motion(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"readable": False, "frames_decoded": 0, "error": "OpenCV could not open the file", "features": None}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    previous = None
    series: list[float] = []
    spatial = np.zeros((GRID, GRID), dtype=np.float64)
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, SAMPLE_SIZE, interpolation=cv2.INTER_AREA)
            if previous is not None:
                difference = cv2.absdiff(small, previous).astype(np.float64) / 255.0
                series.append(float(difference.mean()))
                spatial += _grid_means(difference)
            previous = small
    finally:
        capture.release()
    duration = decoded / fps if fps and decoded else None
    features = _feature_vector(series, spatial) if len(series) >= 1 else None
    return {
        "readable": decoded > 0,
        "frames_decoded": decoded,
        "fps": fps if fps else None,
        "width": width if width else None,
        "height": height if height else None,
        "duration_s": duration,
        "error": None if decoded else "no frames were decoded",
        "features": features,
    }


def _grid_means(difference: np.ndarray) -> np.ndarray:
    cell_h = difference.shape[0] // GRID
    cell_w = difference.shape[1] // GRID
    means = np.zeros((GRID, GRID), dtype=np.float64)
    for row in range(GRID):
        for column in range(GRID):
            cell = difference[row * cell_h : (row + 1) * cell_h, column * cell_w : (column + 1) * cell_w]
            means[row, column] = float(cell.mean()) if cell.size else 0.0
    return means


def _feature_vector(series: list[float], spatial_acc: np.ndarray) -> list[float]:
    values = np.asarray(series, dtype=np.float64)
    temporal = []
    for index in range(TEMPORAL_BINS):
        start = int(index * len(values) / TEMPORAL_BINS)
        end = int((index + 1) * len(values) / TEMPORAL_BINS)
        chunk = values[start:end]
        temporal.append(float(chunk.mean()) if len(chunk) else 0.0)
    spatial = spatial_acc / len(values)
    midpoint = len(values) // 2
    early = float(values[:midpoint].mean()) if midpoint else 0.0
    late = float(values[midpoint:].mean()) if len(values[midpoint:]) else 0.0
    total = float(spatial.sum())
    if total > 0:
        columns = np.arange(GRID, dtype=np.float64)
        rows = np.arange(GRID, dtype=np.float64)
        center_x = float((spatial.sum(axis=0) * columns).sum() / total / (GRID - 1))
        center_y = float((spatial.sum(axis=1) * rows).sum() / total / (GRID - 1))
    else:
        center_x = 0.5
        center_y = 0.5
    stats = [
        float(values.mean()),
        float(values.std()),
        float(values.max()),
        _percentile(values, 0.9),
        late / (early + 1e-6),
        center_x,
        center_y,
    ]
    return temporal + spatial.reshape(-1).tolist() + stats


def _percentile(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)
