
from pathlib import Path

from src.heatmap import generate_risk_heatmap

video_path = Path(
    r"C:\Research 26-27\archive\shoplifting_dataset"
    r"\Shoplifting001_x264__Shoplifting001_x264_0.mp4"
)

# TEST ONLY: sample coordinates and scores to verify
# the heatmap visualisation function.
sample_events = [
    {"x": 200, "y": 150, "risk_score": 80},
    {"x": 320, "y": 220, "risk_score": 60},
    {"x": 400, "y": 280, "risk_score": 90},
]

result = generate_risk_heatmap(
    video_path=video_path,
    risk_events=sample_events,
    output_path="outputs/risk_heatmap.jpg",
)

print("Heatmap created successfully!")
print(result)
