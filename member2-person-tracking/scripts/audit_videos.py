"""Audit all CCTV / surveillance videos in the member2-person-tracking module.

Checks readability, frame count, FPS, resolution (width x height),
duration in seconds, file size in MB, and outputs a summary CSV report.
"""
from pathlib import Path
import cv2
import csv
import argparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIRS = [ROOT / "data", ROOT / "Dataset" / "videos"]
REPORT_DIR = ROOT / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def audit_videos(video_dirs: list[Path] | None = None, output_csv: Path | None = None) -> list[dict]:
    if video_dirs is None:
        video_dirs = DEFAULT_VIDEO_DIRS
    if output_csv is None:
        output_csv = REPORT_DIR / "video_audit.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    seen_paths = set()
    rows = []

    for vdir in video_dirs:
        if not vdir.exists():
            continue
        for video_path in sorted(vdir.rglob("*")):
            if not video_path.is_file():
                continue
            if video_path.suffix.lower() not in EXTENSIONS:
                continue
            # Avoid duplicate paths
            resolved = video_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            cap = cv2.VideoCapture(str(video_path))
            opened = cap.isOpened()
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            readable = opened and frames > 0 and fps > 0
            duration = frames / fps if readable else 0

            # Compute relative path cleanly
            try:
                rel_path = str(video_path.relative_to(ROOT))
            except ValueError:
                rel_path = str(video_path)

            rows.append({
                "video": rel_path,
                "readable": readable,
                "frames": frames,
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2),
                "size_mb": round(video_path.stat().st_size / (1024 ** 2), 2),
            })
            cap.release()

    fieldnames = [
        "video", "readable", "frames", "fps", "width", "height", "duration_seconds", "size_mb"
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Videos found: {len(rows)}")
    print(f"Readable videos: {sum(r['readable'] for r in rows)}")
    print(f"Unreadable videos: {sum(not r['readable'] for r in rows)}")
    print(f"Report saved to: {output_csv}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit CCTV video dataset")
    parser.add_argument("--dir", type=Path, help="Custom directory containing videos to audit")
    parser.add_argument("--output", type=Path, help="Path for output audit CSV")
    args = parser.parse_args()

    custom_dirs = [args.dir] if args.dir else None
    audit_videos(custom_dirs, args.output)
