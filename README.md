# AI-Based Real-Time Shoplifting Detection System

Final-year university research project. Four team members share one repository. Each member owns one module and develops it locally. The modules will later be connected through agreed data contracts.

## Project objective

Build a real-time pipeline that watches retail CCTV or recorded video, follows people through the scene, looks for suspicious movement patterns, and produces a risk level for further review.

This repository currently holds the project structure, documentation, and interface contracts only. Detection, tracking, behaviour analysis, and risk prediction are not implemented yet.

## System workflow

```text
VIDEO
  ↓
Member 1 — Object Detection
  ↓
Detection results
  ↓
Member 2 — Person Tracking
  ↓
Tracking results
  ↓
Member 3 — Suspicious Behaviour Analysis
  ↓
Behaviour results
  ↓
Member 4 — Risk Scoring / Prediction
  ↓
Final risk result
```

Video enters the object detection module. Detections are passed to person tracking. Tracks are passed to behaviour analysis. Behaviour results are passed to risk scoring. The risk result is the output intended for a later dashboard or reporting layer.

The exact JSON shapes are defined in [docs/api-contract.md](docs/api-contract.md).

## Four team modules

| Member | Folder | Responsibility |
| --- | --- | --- |
| Member 1 | `member1-object-detection/` | Detect people and objects in CCTV or video. Expected later work may use YOLO-based detection. |
| Member 2 | `member2-person-tracking/` | Track detected people across frames. Expected later work may use ByteTrack or DeepSORT. |
| Member 3 | `member3-behaviour-analysis/` | Analyse movement for loitering, repeated shelf visits, and other suspicious indicators, then produce a behaviour score. |
| Member 4 | `member4-risk-prediction/` | Combine behaviour indicators into a final risk level and score. |

Each member commits work inside their own folder. Shared contracts live in `docs/` and are changed only after the team agrees.

## Repository structure

```text
shoplifting-detection-system/
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── docs/
│   ├── project-overview.md
│   ├── system-architecture.md
│   ├── api-contract.md
│   ├── development-guidelines.md
│   └── team-contributions.md
├── data/
│   ├── raw/
│   ├── processed/
│   ├── samples/
│   └── README.md
├── member1-object-detection/
│   ├── README.md
│   ├── src/
│   ├── tests/
│   ├── models/
│   ├── configs/
│   └── outputs/
├── member2-person-tracking/
│   ├── README.md
│   ├── src/
│   ├── tests/
│   ├── configs/
│   └── outputs/
├── member3-behaviour-analysis/
│   ├── README.md
│   ├── src/
│   ├── tests/
│   ├── configs/
│   └── outputs/
├── member4-risk-prediction/
│   ├── README.md
│   ├── src/
│   ├── tests/
│   ├── models/
│   ├── configs/
│   └── outputs/
├── integration/
│   ├── README.md
│   ├── pipeline/
│   ├── interfaces/
│   └── tests/
└── results/
    ├── screenshots/
    ├── videos/
    ├── reports/
    └── README.md
```

## Development workflow

```text
main
├── member1-object-detection
├── member2-person-tracking
├── member3-behaviour-analysis
└── member4-risk-prediction
```

- Work on your own branch.
- Commit regularly inside your own module folder.
- Push your branch.
- Merge into `main` through a pull request.
- Do not overwrite another member's module.
- Discuss shared interface changes before editing `docs/api-contract.md`.
- Do not commit datasets, large model files, videos, or secrets unless the team has specifically agreed.

Details are in [docs/development-guidelines.md](docs/development-guidelines.md). Responsibilities are in [docs/team-contributions.md](docs/team-contributions.md).

## How the modules will eventually integrate

1. Each member develops and checks their module inside their own folder.
2. Neighbouring modules exchange JSON that matches [docs/api-contract.md](docs/api-contract.md). Sample files can be saved under a module `outputs/` folder for local handoff. Those outputs are gitignored.
3. When the four contracts are stable, the `integration/pipeline/` folder will connect the modules in order.
4. `integration/interfaces/` will hold the shared handoff points. `integration/tests/` will check that module outputs still match the contract.
5. Final figures, clips, and write-ups for the report go in `results/`. Large media stays local unless the team decides otherwise.

No module should depend on another member's private code. Depend on the documented JSON contract.

Object Detection Module
Pretrained YOLOv8 prototype for Member 1. It reports object boxes and class names from the loaded model. It does not track person IDs, score behaviour, or decide shoplifting.

Status
Implemented as a local prototype. Default weights are COCO yolov8n.pt (80 classes). That is not a retail product catalogue: bottles, backpacks, handbags, cups, and other COCO labels can appear. Store SKUs, concealment, ownership of an item, and shoplifting are out of scope.

Weights on this machine
Inspected under member2-person-tracking/models/:

File	Classes	Use for Member 1
yolov8n.pt	80 COCO classes	Yes — default
yolov8n-person-track.pt	1 (item)	No — person/item tracker, rejected
yolov8n_finetuned.pt	1 (person)	No
yolov8n-cam-multiperson.pt	1 (item)	No
The prototype never silently substitutes a person-only file. It does not download weights.

You may copy yolov8n.pt into member1-object-detection/models/ if you want a Member 1 local copy. The resolver checks that folder first, then the Member 2 models folder.

Commands
Webcam (close the Member 2 dashboard first — DirectShow does not share the camera):

python member1-object-detection/src/member1_live_demo.py --source webcam
Video file:

python member1-object-detection/src/member1_live_demo.py --source path/to/clip.mp4 --no-window --json-out member1-object-detection/outputs/detections.jsonl
Headless OpenCV builds need --no-window. Optional filters: --classes bottle,backpack,handbag --conf 0.4 --weights member2-person-tracking/models/yolov8n.pt.

Tests:

python -m unittest discover -s tests -p "test_*.py" -v
Run from member1-object-detection/. Mocked tests do not need a webcam. test_real_model_smoke.py loads yolov8n.pt when that file exists.

JSON contract
Prototype result (fields come from the model, not from this example):

{
  "frame_id": 12,
  "timestamp": 1.2,
  "model": "yolov8n.pt",
  "note": "Member 1 pretrained YOLO prototype. Detections are COCO classes from the loaded weights. This is not retail SKU recognition, concealment, object-to-person ownership, or shoplifting.",
  "error": null,
  "detections": [
    {
      "detection_id": 1,
      "class_id": 39,
      "class_name": "bottle",
      "confidence": 0.87,
      "bbox": {"x1": 120.2, "y1": 80.4, "x2": 190.9, "y2": 260.1}
    }
  ]
}
pipeline_detections(result) maps that onto the team handoff in docs/api-contract.md: class, confidence, bbox as [x_min, y_min, x_max, y_max]. Extra prototype fields stay in this module until the team extends the shared contract.

Python API: ObjectDetector.detect_objects(frame, frame_id, timestamp).

Integration boundaries
Member 2 keeps person detection, ByteTrack IDs, and zones.
Member 1 only emits object boxes. Overlap with a person box is not ownership.
Member 3 scoring rules are unchanged. Object detections are not auto-labelled as interaction, concealment, or shoplifting.
Member 4 may later consume these boxes through an agreed interface.
The dashboard Object Detection page, when enabled, reuses the Member 2 live session frames. Do not start this webcam demo at the same time as the dashboard.

Folder guide
Path	Use
src/	Detector, capture, overlay, live demo
tests/	Mocked unit tests and optional real-weight smoke
models/	Optional local copy of COCO weights (gitignored)
configs/detector.yaml	Confidence, image size, device
outputs/	JSONL dumps. Not committed
Component 2 — Multi-Person Tracking
This isolated module tracks person detections from Member 1 and writes the contract-compliant tracking sequence consumed by Member 3. It uses a ByteTrack-style two-pass IoU association strategy plus a constant-velocity Kalman filter for brief missed detections. It also includes independent retail zone dwell and transition analytics.

Status
 YOLOv8 person-detection adapter (src/detector.py)
 Two-pass tracker and motion prediction (src/tracker.py)
 Polygon zone analytics (src/zone_analytics.py)
 Contract-formatted JSON writer (main.py)
 Unit tests for identity, zones, and evaluation metrics
 MOT evaluation CLI for IDSW, IDF1, MOTA, occlusion recovery, and FPS helpers
 OpenCV ID/trajectory overlay and annotated video export (demo_track.py)
 Run the evaluation on labelled retail test clips
Local use
From this directory, provide one detection-frame object or a list of them:

python -m pip install -r requirements.txt
python main.py tests/fixtures/sample_detections.json --output outputs/tracking_log.json
python demo_track.py --detections tests/fixtures/sample_detections.json --video path/to/clip.mp4 --output-video outputs/annotated.mp4 --output-json outputs/tracking_log.json
python -m unittest discover -s tests -v
python prepare_person_dataset.py
python train_detector.py
python demo_track.py --video Dataset/videos/not_shoplifting1.mp4 --weights models/yolov8n-person-track.pt --ultralytics-track --no-zones --output-video outputs/not_shoplifting1_tracked.mp4 --output-json outputs/not_shoplifting1_tracking.json
python main.py --video Dataset/videos/not_shoplifting1.mp4 --weights models/yolov8n-person-track.pt --output outputs/not_shoplifting1_tracking.json --output-video outputs/not_shoplifting1_tracked.mp4
Dataset cleaning and pretrained baseline
Keep source footage in data/raw_videos/; the cleaning pipeline never deletes or modifies those files. It recursively validates supported videos, checks SHA-256 duplicates, and copies results to data/cleaned_videos/ or data/rejected_videos/:

python src\clean_dataset.py
The default run decodes every frame and writes outputs/metrics/dataset_cleaning_report.csv. Use --sample-only only for a quick preliminary inventory; it does not provide the same validation guarantee.

Run the pretrained YOLOv8 person detector with Ultralytics ByteTrack over all cleaned videos:

python src\run_tracking_dataset.py
This writes an annotated video, contract-shaped frame JSON, and frame-level CSV for each input video under outputs/videos/, outputs/json/, and outputs/metrics/. The default weights are models/yolov8n.pt; use --weights to compare a fine-tuned checkpoint later. The generated person_id values are tracker IDs, not real-world identities.

The default zones.json marks Entrance as start_zone and Checkout as end_zone. The batch tracker draws all configured polygons and adds zone, start_zone, end_zone, and reached_end_zone fields to each track. Edit the polygon coordinates to match each camera before running; the current coordinates assume a 1280x720 view.

ByteTrack is not a trainable detector model. Its behavior is controlled by association thresholds and track-buffer settings in tracker_config.yaml. Tune those settings and evaluate on ground-truth annotations rather than attempting to train ByteTrack from scratch.

The modular application is split into config.py, kalman_filter.py, ciou_matching.py, bytetrack.py, and retail_analytics.py. Its defaults are the specified T_high=0.6, T_low=0.1, T_new=0.7, CIoU distance threshold 0.3, and a 90-frame occlusion buffer. main.py --video runs YOLOv8 class-0 inference, two-stage ByteTrack association, OpenCV dashboard overlays, polygon zone analytics, and writes frame telemetry including trajectories, zone, dwell, transitions, and occupancy. Use --show for the live window; omit it for headless servers.

The training command starts from models/yolov8n.pt by default. The defaults are 100 epochs, batch size 16, eight workers, and patience 20, with mosaic, mixup, and camera-tilt settings suited to crowded and partially occluded retail scenes. On a CPU-only machine, use a smaller batch and bounded run such as python train_detector.py --epochs 3 --batch 4 --imgsz 640 --device cpu --workers 0; CUDA hardware can use the longer 100-epoch run.

Training writes runs to runs/detect/person_bytetrack/weights/best.pt and copies the best checkpoint to models/yolov8n-person-track.pt for the tracking demo.

prepare_person_dataset.py reads local images from Dataset/images/, images and videos recursively from Dataset/shoplifting/, and sampled frames from Dataset/videos/, then creates the YOLOv8 layout under Dataset/yolo/:

Dataset/yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── (data.yaml is written to configs/data.yaml)
Each generated label file uses one normalized class_id x_center y_center width height row per person, with class 0 named person. The generated configuration uses images/train and images/val relative to the dataset root. Dataset contents and generated training outputs remain local and are ignored by Git.

main.py and demo_track.py emit only the fields currently approved in docs/api-contract.md: frame_id, tracks[].person_id, tracks[].bbox, and tracks[].center. Zone names can appear on the annotated video. They are not written into the shared handoff JSON.

demo_track.py maps video frame 0 to the first frame_id in the detection file unless you pass --frame-id-start. If --detections is omitted, it runs the optional YOLOv8 adapter on the video instead.

It also writes outputs/tracking_outcomes.json with every listed Component 2 outcome: persistent IDs, trajectories, per-frame coordinates, zone transitions, shelf visits, entry/exit, occupancy, timestamps, dwell, and track status (active, temporarily_lost, terminated). That file is for this module. The Member 3 handoff JSON stays contract-only.

For Ultralytics' built-in persistent ByteTrack, use --ultralytics-track. This passes conf=0.1, iou=0.5, classes=[0], and tracker=bytetrack.yaml by default, so low-confidence person detections can participate in occlusion recovery. It writes the same contract-compliant person_id, bbox, and center fields as the custom tracker. Add --show for the live OpenCV window; omit it for headless processing.

A contract-shaped detection sample lives at tests/fixtures/sample_detections.json.

Evaluation
Create a locally labelled ground-truth JSON sequence using the same frame and track structure as the tracking output, but with the real person_id values. Then run:

.\.venv\Scripts\python.exe evaluate.py outputs/tracking_log.json path/to/ground_truth.json
The resulting outputs/evaluation_metrics.json reports id_switches, idf1, mota, false positives/negatives, and occlusion recovery for a 15-frame-or-longer missed-detection interval. The measure_fps helper in src/evaluation.py measures detector/tracker FPS after warm-up frames.

For direct frame detection, install ultralytics in the team's agreed environment and place yolov8n.pt under models/ (model weights are ignored by Git). The provided zones target a 1280×720 view; calibrate configs/zones.json for the actual camera before evaluation.

Evaluation reporting
Use annotated clips with ground truth to report ID switches (IDSW), MOTA, and IDF1. Measure FPS around the complete detector-to-tracker loop after a warmup, and report hardware, input resolution, detector weights, and thresholds. The target is at least 25 FPS.

Cam multi-person fine-tune
prepare_cam_multiperson.py copies Dataset/cam multi-person into Dataset/yolo_cam_multiperson/ (80/20, seed=42) and writes dataset.yaml with class 0: person. The prepared split is 14,389 images: 11,511 train and 2,878 val. None of the source images had a .txt box file, so all 14,389 label files are empty. Add class-0 boxes beside the source images and rerun the script before training. Zip and train from train_cam_multiperson_colab.ipynb, then save the download as models/yolov8n-cam-multiperson.pt.

python prepare_cam_multiperson.py
tar -a -cf yolo_cam_multiperson.zip -C Dataset yolo_cam_multiperson
src/detector.py loads that checkpoint first, then models/yolov8n-person-track.pt, then models/yolov8n.pt. Inference still passes classes=[0].

Benchmark
models/yolov8n-cam-multiperson.pt is not in the folder yet, so this table does not contain MOTA, IDF1, IDSW, or occlusion recovery for that checkpoint. evaluate.py needs a labelled tracking JSON in the contract shape, and no ground-truth file is in the repository. The speed row is measure_fps on the fallback checkpoint the priority list selected, after 10 warm-up frames.

Checkpoint	Hardware	Input	FPS	MOTA	IDF1	IDSW	Occlusion recovery
models/yolov8n-cam-multiperson.pt	Not trained yet	—	—	—	—	—	—
models/yolov8n-person-track.pt	Intel Core i5-13420H, CPU	544×544, Dataset/videos/not_shoplifting1.mp4, 135 timed frames	38.38	—	—	—	—
Detector confidence 0.1, person class only. Tracker settings from configs/tracker_config.yaml: high 0.50, low 0.10, match IoU 0.30, max missed frames 30. The 25 FPS target was met on this clip with the fallback checkpoint. A contract check of the 145-frame tracking JSON from that clip found only integer frame_id, person_id, bbox, and center (person ids 1–4). Fill the first row after the Colab checkpoint and a labelled ground-truth sequence are both available:

python demo_track.py --video Dataset/videos/not_shoplifting1.mp4 --weights models/yolov8n-cam-multiperson.pt --no-zones --output-video outputs/cam_multiperson_sanity.mp4 --output-json outputs/cam_multiperson_sanity.json
python evaluate.py outputs/cam_multiperson_sanity.json path/to/ground_truth.json
