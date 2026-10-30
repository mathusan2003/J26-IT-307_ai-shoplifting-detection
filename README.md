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
Suspicious Behaviour Analysis Module
Purpose
Analyse how each tracked person moves through the scene and record suspicious indicators. This module covers:

Loitering detection
Repeated shelf visit detection
Abnormal movement
Object interaction
Possible concealment
A behaviour score for each person
A time-window assessment that combines those signals before assigning a suspicious level
A trained logistic-regression model that separates suspicious tracks from ordinary ones
It does not detect objects, assign track IDs, or write Member 4's risk JSON.

Responsible member
Member 3.

Work only in member3-behaviour-analysis/. Other members may read this folder. They should not overwrite it.

Expected input
An ordered sequence of tracking results from Member 2. The shared contract is:

{
  "frame_id": 100,
  "tracks": [
    {
      "person_id": 12,
      "bbox": [100, 120, 250, 400],
      "center": [175, 260]
    }
  ]
}
timestamp and zone are optional. When they are absent, time is frame_id / fps (default 25, the same rate Member 2 targets) and the zone is the polygon that contains the bottom-centre of bbox. That anchor matches Member 2's zone analytics. shelf is accepted as another name for zone. position is accepted when center is absent.

Member 2's tracking JSON does not include zone names. The polygons in configs/zones.json are the same 1280×720 layout as member2-person-tracking/configs/zones.json. Recalibrate both files together for a different camera.

Optional object context uses Member 1's detection shape, passed separately with --detections. Only non-person classes are used. Person boxes in that file are ignored.

python src/analyse_behaviour.py path/to/tracking.json --detections path/to/detections.json --output outputs/behaviour.json
Expected output
One behaviour result per person, for Member 4. A full run is a JSON array ordered by person_id.

{
  "person_id": 12,
  "behaviours": [
    {
      "type": "loitering",
      "score": 20
    },
    {
      "type": "repeated_shelf_visit",
      "score": 25
    }
  ],
  "behaviour_score": 45
}
Allowed type values are loitering, repeated_shelf_visit, abnormal_movement, object_interaction, and possible_concealment. An indicator is omitted when it does not fire. Each behaviour object still has only type and score. behaviour_score is the sum of those indicator scores, capped at 100. It is not a suspicious alert.

assessment is a separate object. Member 4 can ignore it. suspicious_level is none, medium, or high. combined_score sums only the indicators that fell inside the selected time window, and it is 0 when the level is none. Neither field is Member 4's risk_level or risk_score.

outputs/ is local and is not committed.

How the indicators are scored
Thresholds live in configs/behaviour_config.json.

Indicator	When it fires	Score
loitering	Longest time spent inside a 40 px radius is at least 15 s	20, plus 0.5 per extra second, capped at 40
repeated_shelf_visit	The same shelf is left and entered again. A shelf is any zone whose name contains "shelf"	25 per return, capped at 40
abnormal_movement	Speed reaches 250 px/s, or there are at least 4 sharp reversals faster than 30 px/s	10 for speed, up to 10 for reversals, plus 10 when that path is also highly folded, capped at 30
object_interaction	Small handling movements (5–35 px/s) continue on one shelf for at least 4 s, or a non-person detection stays near the person for at least 3 frames	Up to 15 for handling and 10 for object proximity, capped at 25
possible_concealment	The later bounding box shrinks to 72% or less of the earlier median, after the person has been at a shelf or next to an object	18, plus 12 when a nearby object then disappears, capped at 30
Standing still is loitering, not object interaction. Handling needs visible small movement, or a real object detection. Concealment is only a possible signal: this module cannot see hands, pockets, or items inside a bag. A box that is small for the whole track is not concealment, because the person may just be far from the camera.

Combined assessment
The five detectors still score one indicator at a time. A suspicious level is assigned only afterwards, from indicators whose time spans fall inside assessment.window_seconds (30 seconds in the shipped config).

Loitering, a repeated visit to the same shelf, and examining or picking up a product (object_interaction) do not raise a suspicious level. That remains true when those signals occur together. A level requires at least two indicators in one window, and at least one of them must be outside that normal-shopping set. One indicator on its own, including abnormal movement or possible concealment, stays at none. Indicators separated by more than the window are not combined. An indicator with no start and end time is not treated as if it happened at the same time as the others.

suspicious_level	When it is used
none	No qualifying window. combined_score is 0
medium	Two qualifying indicators in one window
high	Three or more qualifying indicators in one window
not_contributing gives the reason an indicator was left out: context_only, isolated, outside_window, or missing_time.

The window, the minimum count, the normal-shopping names, and the medium/high counts are in configs/behaviour_config.json. They are configurable assumptions. They have not been validated on labelled retail video, and they are not a measured false-alarm rate. Do not describe suspicious_level as real-world accuracy.

Movement history
The Phase 1 history is still the per-person position log. Behaviour checks read it from MovementHistoryStore. Bounding boxes are kept on the in-memory record and are not written into the history JSON.

python src/build_movement_history.py samples/sample_tracking_frames.json
Behaviour model
src/model.py trains an L2-regularised logistic regression on 15 measurements (duration, speeds, reversals, shelf visits, handling time, box-shrink ratio, and object contact). The standard library is enough. No extra packages are required.

The model does not replace the indicator scores. Member 4 reads the rule JSON. The model is the learned check that those measurements separate suspicious tracks from ordinary ones.

There is no retail CCTV and no public shoplifting label file in this repository (data/ is empty). Training uses 80 synthetic sequences in datasets/labelled_sequences.json: 40 ordinary and 40 suspicious, with a fixed seed. Within each pattern, every fourth copy is held out, which gives 58 training sequences and 22 test sequences. The generator is src/dataset.py.

python src/train_behaviour_model.py
That command rewrites the dataset, models/behaviour_model.json, evaluation/metrics.json, and samples/member2_contract_loiter.json. The metrics file is the result to cite. It includes precision, recall, F1, specificity, and the confusion matrix for the rules and the model, plus a leave-one-family-out check.

The leave-one-family-out check trains without one suspicious pattern and then scores that pattern. It shows whether the linear model recognises a behaviour it did not see. The rules do not use that check; they score every pattern from the thresholds above.

On the synthetic test split (22 sequences: 10 suspicious, 12 ordinary) the rules and the logistic regression both scored accuracy 1.0, precision 1.0, recall 1.0, and F1 1.0. The model confusion matrix was 10 true positives, 12 true negatives, and no errors. The same rules also matched all 80 labelled sequences, not only the test split. These figures are in evaluation/metrics.json. They show that the pipeline is consistent with the patterns it was given. They are not a CCTV result.

Leave-one-family-out recall, with ordinary-track specificity 1.0 in every case, was:

Pattern held out	Recall on that pattern
Loitering	1.0
Repeated shelf visit	1.0
Abnormal movement	0.0
Object interaction	0.0
Possible concealment	1.0
The linear model does not recognise abnormal movement or shelf handling when those patterns were absent from training. The rule detectors still score them from the thresholds above. That gap is why Member 4 receives the rule JSON rather than a single model probability.

Real-video baseline
datasets/shoplifting-video/ contains the only real clips. The label is the folder name and nothing else:

Folder	Files on disk	Missing name
normal	90	none
shoplifting	92	shoplifting-49.mp4
All 182 files opened and decoded at 30 fps. Most are 640×480. Eight are 1920×1080 (2 normal, 6 shoplifting). Three normal files are exact copies of other normal files: normal-44 with normal-52, normal-50 with normal-51, and normal-55 with normal-56. There is no RetailS or PoseLift data on this machine, and these clips do not annotate loitering, repeated shelf visits, abnormal movement, object interaction, or concealment. Those detectors are unchanged and still run only on Member 2 tracking JSON.

The split is by file hash, not by frame. Within each folder, unique hashes are sorted. The first block is train, the next 15% is validation, and the last 15% is test. Duplicate copies stay in the same split and are not counted twice. The used counts are 125 train (61 normal, 64 shoplifting), 27 validation (13 normal, 14 shoplifting), and 27 test (13 normal, 14 shoplifting). The assignment is in datasets/video_split.json.

The baseline is an L2 logistic regression on frame-difference motion, not a deep network. C is chosen on validation. It does not write Member 4 JSON.

python -m pip install -r requirements.txt
python src/train_video_baseline.py
On the locked test split the motion model scored accuracy 0.5556, precision 0.5714, recall 0.5714, F1 0.5714, and specificity 0.5385 (8 true positives, 6 false positives, 7 true negatives, 6 false negatives). Training F1 was 0.8992 and validation F1 was 0.7143, with C 10. A majority-class guess on the same test videos has accuracy 0.5185 and F1 0.6829. A duration-only model has test accuracy 0.6667 and F1 0.6897. Motion alone is a weak baseline, and clip length already carries label information. Full figures are in evaluation/video_baseline_metrics.json.

Run the tests
From this folder:

python -m unittest discover -s tests -v
Integration
Member 2 writes a list of tracking frames. This module reads that list and writes a list of behaviour objects. Member 4 reads those objects. integration/pipeline/ is still the shared place that will call the modules in order. This folder does not copy Member 2's tracker or Member 4's risk mapping.

python src/analyse_behaviour.py samples/member2_contract_loiter.json --output outputs/behaviour.json
That sample uses only frame_id, person_id, bbox, and center. The detector result is loitering with score 20 for person 12. The combined suspicious level stays none, because loitering on its own is not an alert.

samples/sample_tracking_frames.json is the short Phase 1 example. Neither person in it is suspicious.

Folder guide
Path	Use
src/	History, detectors, model, training, and the analysis command
configs/	Score thresholds and the 1280×720 zone map
datasets/	Synthetic sequences, plus the local Normal / Shoplifting videos and their split
models/	Logistic-regression coefficients in JSON
evaluation/	Metrics from the latest training run
samples/	Small tracking sequences
tests/	Tests for this module only
outputs/	Local behaviour JSON. Not committed
Limitations
No real labelled retail video was available when the synthetic logistic regression was trained. The separate video baseline uses datasets/shoplifting-video/ and is scored in evaluation/video_baseline_metrics.json. Those clips are labelled only normal or shoplifting.
RetailS and PoseLift were not found locally, so they were not used.
The motion baseline does not beat a duration-only model on the locked test split. Do not treat it as evidence that the five behaviour detectors work on these clips.
Member 2 does not put zone names in the contract JSON. Zone assignment depends on configs/zones.json matching the camera.
Without timestamps, time assumes a constant frame rate of 25.
Object interaction from merchandise needs a detection file that contains non-person classes. Without that file, interaction is only the handling-motion rule.
Possible concealment needs bounding boxes and a shrink after shelf or object contact. It is not a proof that an item was hidden.
The logistic regression does not set risk_level. That remains Member 4's decision. assessment.suspicious_level is also not that risk level.
The 30-second window and the medium/high counts are assumptions. This repository has no labelled retail segments for loitering, repeated shelf visits, abnormal movement, object interaction, or concealment, so those cutoffs are not a measured accuracy.
The shared pipeline in integration/ is not wired yet. The handoff is the JSON file described above.
Status
Movement history, the five tracking-json indicators, the separate time-window assessment, and the Member 4 behaviour JSON are implemented. A separate video-level Normal versus Shoplifting baseline has been trained on datasets/shoplifting-video/. That baseline does not add behaviour-type labels. The assessment cutoffs are not a retail accuracy figure.
Risk Scoring / Prediction Module
Purpose
Combine behaviour indicators, and any other information the team later agrees to include, into a final risk level and risk score for each person. This is the last of the four research modules. A dashboard or report can read the result later. This folder does not implement that dashboard.

Responsible member
Member 4.

Work only in member4-risk-prediction/. Use the branch member4-risk-prediction. Other members may read this folder. They should not overwrite it.

Expected input
Behaviour results from Member 3, one object per person:

{
  "person_id": 12,
  "behaviours": [
    {
      "type": "loitering",
      "score": 20
    },
    {
      "type": "repeated_shelf_visit",
      "score": 25
    }
  ],
  "behaviour_score": 45
}
If Member 3 is not ready, use a hand-written file in this shape. Extra inputs beyond this object require an update to docs/api-contract.md before they are used.

Optional local settings belong in configs/. Local weight files, if a later method needs them, belong in models/ and are not committed.

Expected output
One risk result per person:

{
  "person_id": 12,
  "risk_level": "medium",
  "risk_score": 45
}
risk_level is low, medium, or high. The mapping from behaviour results to these fields is part of later implementation and is not fixed here.

Write local results under outputs/. Do not commit them.

Future integration point
The later output layer reads this risk result. When the four modules are connected, integration/pipeline/ is the place that passes behaviour JSON into this module and passes the risk JSON onward. Do not build the dashboard inside this folder.

Folder guide
Path	Use
src/	Risk scoring code, when implementation starts
tests/	Tests for this module only
models/	Local weight files, if needed. Not committed
configs/	Scoring settings
outputs/	Local risk JSON. Not committed
