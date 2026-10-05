# Component 2 — Multi-Person Tracking

This isolated module tracks `person` detections from Member 1 and writes the
contract-compliant tracking sequence consumed by Member 3. It uses a
ByteTrack-style two-pass IoU association strategy plus a constant-velocity
Kalman filter for brief missed detections. It also includes independent retail
zone dwell and transition analytics.

## Status

- [x] YOLOv8 person-detection adapter (`src/detector.py`)
- [x] Two-pass tracker and motion prediction (`src/tracker.py`)
- [x] Polygon zone analytics (`src/zone_analytics.py`)
- [x] Contract-formatted JSON writer (`main.py`)
- [x] Unit tests for identity, zones, and evaluation metrics
- [x] MOT evaluation CLI for IDSW, IDF1, MOTA, occlusion recovery, and FPS helpers
- [ ] Run the evaluation on labelled retail test clips

## Local use

From this directory, provide one detection-frame object or a list of them:

```powershell
python main.py path/to/detections.json --output outputs/tracking_log.json
python -m unittest discover -s tests -v
```

`main.py` emits only the fields currently approved in `docs/api-contract.md`:
`frame_id`, `tracks[].person_id`, `tracks[].bbox`, and `tracks[].center`.
Zone state remains internal so it does not change the shared handoff contract.

## Evaluation

Create a locally labelled ground-truth JSON sequence using the same frame and
track structure as the tracking output, but with the real `person_id` values.
Then run:

```powershell
.\.venv\Scripts\python.exe evaluate.py outputs/tracking_log.json path/to/ground_truth.json
```

The resulting `outputs/evaluation_metrics.json` reports `id_switches`,
`idf1`, `mota`, false positives/negatives, and occlusion recovery for a
15-frame-or-longer missed-detection interval. The `measure_fps` helper in
`src/evaluation.py` measures detector/tracker FPS after warm-up frames.

For direct frame detection, install `ultralytics` in the team's agreed
environment and place `yolov8n.pt` under `models/` (model weights are ignored
by Git). The provided zones target a 1280×720 view; calibrate
`configs/zones.json` for the actual camera before evaluation.

## Evaluation reporting

Use annotated clips with ground truth to report ID switches (IDSW), MOTA, and
IDF1. Measure FPS around the complete detector-to-tracker loop after a warmup,
and report hardware, input resolution, detector weights, and thresholds. The
target is at least 25 FPS.
