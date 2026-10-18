"""Package the Member 2 person dataset and sample test videos into a zip file for Google Colab.

This script only includes the essential train/val images and labels from Dataset/yolo/
(excluding intermediate staging directories _all and _video_frames to save space).
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "Dataset" / "yolo"
OUTPUT_ZIP = ROOT / "member2_dataset_colab.zip"
SAMPLE_VIDEOS = [
    ROOT / "Dataset" / "videos" / "not_shoplifting1.mp4",
    ROOT / "Dataset" / "videos" / "shoplifting1.mp4",
]


def create_colab_package():
    if not YOLO_DIR.exists():
        print(f"[ERROR] YOLO directory not found at: {YOLO_DIR}")
        print("Please ensure Dataset/yolo exists or run prepare_person_dataset.py first.")
        sys.exit(1)

    print(f"Creating Colab dataset zip archive: {OUTPUT_ZIP.name} ...")
    total_files = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add images and labels (train and val)
        for split in ["train", "val"]:
            img_dir = YOLO_DIR / "images" / split
            lbl_dir = YOLO_DIR / "labels" / split

            if img_dir.exists():
                images = list(img_dir.glob("*.*"))
                print(f"Adding {len(images)} {split} images...")
                for img in images:
                    zf.write(img, arcname=f"dataset/images/{split}/{img.name}")
                    total_files += 1

            if lbl_dir.exists():
                labels = list(lbl_dir.glob("*.txt"))
                print(f"Adding {len(labels)} {split} label files...")
                for lbl in labels:
                    zf.write(lbl, arcname=f"dataset/labels/{split}/{lbl.name}")
                    total_files += 1

        # 2. Add sample videos if available for live tracking demo in Colab
        for vid in SAMPLE_VIDEOS:
            if vid.exists():
                print(f"Adding sample video: {vid.name}...")
                zf.write(vid, arcname=f"videos/{vid.name}")
                total_files += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 50)
    print(f"[SUCCESS] Packaged {total_files} files into: {OUTPUT_ZIP.resolve()}")
    print(f"Archive Size: {size_mb:.2f} MB")
    print("=" * 50)
    print("Next step: Upload this zip file to your Google Drive or Colab session!")


if __name__ == "__main__":
    create_colab_package()
