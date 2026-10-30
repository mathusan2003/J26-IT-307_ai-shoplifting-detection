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
