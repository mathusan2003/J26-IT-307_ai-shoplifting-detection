"""Label demo clips for the panel: handover, occlusion, crowded."""
from __future__ import annotations

from pathlib import Path

SKIP_TOKENS = (
    "annotated", "tracked", "outcomes", "sanity", "check_trained",
    "how_model", "modular_tracking",
)

GROUPS = (
    ("cross_cam", "1. Cross-camera tracking / handover"),
    ("occlusion", "2. Temporary occlusion recovery"),
    ("crowded", "3. Crowded store / dense traffic"),
    ("other", "4. Other retail clips"),
)


def classify_video(path: Path) -> tuple[str, str, str]:
    name = path.name.lower()
    if any(token in name for token in ("3840", "2160", "crowd", "dense", "60fps", "10901926", "12589709")):
        return "crowded", "Camera — crowded aisle", "Crowded store / dense traffic tracking"
    if any(token in name for token in ("occlusion", "popup", "longer_store", "shelf")):
        return "occlusion", "Camera — occluded aisle", "Temporary occlusion: person hidden then reappears"
    if name[:1].isdigit() or "cam" in name:
        return "cross_cam", "Camera view", "Cross-camera handover / Re-ID across views"
    if "input" in name or "sample" in name:
        return "occlusion", "Camera — store floor", "Temporary occlusion and ID continuity"
    return "other", "Camera", "General multi-person tracking"


def catalogue_videos(paths: list[Path]) -> list[dict[str, str]]:
    cross_index = 0
    rows: list[dict[str, str]] = []
    for path in paths:
        group, camera, description = classify_video(path)
        if group == "cross_cam":
            cross_index += 1
            camera = f"Camera {cross_index:02d} — handover view"
        rows.append({
            "id": path.name,
            "file": path.name,
            "label": f"{camera} | {path.name}",
            "group": group,
            "group_label": dict(GROUPS)[group],
            "description": description,
            "camera": camera,
        })
    order = {key: index for index, (key, _) in enumerate(GROUPS)}
    return sorted(rows, key=lambda row: (order.get(row["group"], 9), row["file"]))
