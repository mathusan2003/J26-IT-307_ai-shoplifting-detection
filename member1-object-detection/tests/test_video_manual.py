
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
module_root = project_root / "member1-object-detection"
sys.path.insert(0, str(module_root))

from src.detector import ObjectDetector

video_path = project_root / "data" / "sample" / "test_video.mp4"
output_path = module_root / "outputs" / "video_detections.json"

detector = ObjectDetector()

results = detector.detect_video(
    video_path=video_path,
    output_path=output_path,
    frame_step=30
)

print("Video detection test finished!")
print("Frames processed:", len(results))
print("Total detections:", sum(len(r["detections"]) for r in results))
print("Output exists:", output_path.is_file())
