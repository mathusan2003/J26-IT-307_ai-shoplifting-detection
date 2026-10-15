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
- [x] OpenCV ID/trajectory overlay and annotated video export (`demo_track.py`)
- [ ] Run the evaluation on labelled retail test clips

## Local use

From this directory, provide one detection-frame object or a list of them:

```powershell
python -m pip install -r requirements.txt
python main.py tests/fixtures/sample_detections.json --output outputs/tracking_log.json
python demo_track.py --detections tests/fixtures/sample_detections.json --video path/to/clip.mp4 --output-video outputs/annotated.mp4 --output-json outputs/tracking_log.json
python -m unittest discover -s tests -v
python prepare_person_dataset.py
python train_detector.py
python demo_track.py --video Dataset/videos/not_shoplifting1.mp4 --weights models/yolov8n-person-track.pt --ultralytics-track --no-zones --output-video outputs/not_shoplifting1_tracked.mp4 --output-json outputs/not_shoplifting1_tracking.json
python main.py --video Dataset/videos/not_shoplifting1.mp4 --weights models/yolov8n-person-track.pt --output outputs/not_shoplifting1_tracking.json --output-video outputs/not_shoplifting1_tracked.mp4
```

The modular application is split into `config.py`, `kalman_filter.py`,
`ciou_matching.py`, `bytetrack.py`, and `retail_analytics.py`. Its defaults
are the specified `T_high=0.6`, `T_low=0.1`, `T_new=0.7`, CIoU distance
threshold `0.3`, and a 90-frame occlusion buffer. `main.py --video` runs
YOLOv8 class-0 inference, two-stage ByteTrack association, OpenCV dashboard
overlays, polygon zone analytics, and writes frame telemetry including
trajectories, zone, dwell, transitions, and occupancy. Use `--show` for the
live window; omit it for headless servers.

The training command starts from `models/yolov8n.pt` by default. The defaults
are 100 epochs, batch size 16, eight workers, and patience 20, with mosaic,
mixup, and camera-tilt settings suited to crowded and partially occluded
retail scenes. On a CPU-only machine, use a smaller batch and bounded run such
as `python train_detector.py --epochs 3 --batch 4 --imgsz 640 --device cpu
--workers 0`; CUDA hardware can use the longer 100-epoch run.

Training writes runs to `runs/detect/person_bytetrack/weights/best.pt` and
copies the best checkpoint to `models/yolov8n-person-track.pt` for the
tracking demo.

`prepare_person_dataset.py` reads local images from `Dataset/images/`,
images and videos recursively from `Dataset/shoplifting/`, and sampled frames
from `Dataset/videos/`, then creates the YOLOv8 layout under
`Dataset/yolo/`:

```text
Dataset/yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── (data.yaml is written to configs/data.yaml)
```

Each generated label file uses one normalized `class_id x_center y_center
width height` row per person, with class `0` named `person`. The generated
configuration uses `images/train` and `images/val` relative to the dataset
root. Dataset contents and generated training outputs remain local and are
ignored by Git.

`main.py` and `demo_track.py` emit only the fields currently approved in `docs/api-contract.md`:
`frame_id`, `tracks[].person_id`, `tracks[].bbox`, and `tracks[].center`.
Zone names can appear on the annotated video. They are not written into the shared handoff JSON.

`demo_track.py` maps video frame 0 to the first `frame_id` in the detection file unless you pass `--frame-id-start`. If `--detections` is omitted, it runs the optional YOLOv8 adapter on the video instead.

For Ultralytics' built-in persistent ByteTrack, use `--ultralytics-track`.
This passes `conf=0.1`, `iou=0.5`, `classes=[0]`, and
`tracker=bytetrack.yaml` by default, so low-confidence person detections can
participate in occlusion recovery. It writes the same contract-compliant
`person_id`, `bbox`, and `center` fields as the custom tracker. Add `--show`
for the live OpenCV window; omit it for headless processing.

A contract-shaped detection sample lives at `tests/fixtures/sample_detections.json`.

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
