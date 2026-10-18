
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
module_root = project_root / "member1-object-detection"
sys.path.insert(0, str(module_root))

from src.detector import ObjectDetector, save_json

image_path = project_root / "data" / "sample" / "bus.jpg"
output_path = module_root / "outputs" / "sample_detection.json"

detector = ObjectDetector()
result = detector.detect_image(image_path, frame_id=0)
save_json(result, output_path)

print("Detection successful!")
print("Frame ID:", result["frame_id"])
print("Objects detected:", len(result["detections"]))
print("JSON saved to:", output_path)
print(result)
