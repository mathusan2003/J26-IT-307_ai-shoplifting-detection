# Object Detection Module

Pretrained YOLOv8 prototype for Member 1. It reports object boxes and class names from the loaded model. It does not track person IDs, score behaviour, or decide shoplifting.

## Status

Implemented as a local prototype. Default weights are COCO `yolov8n.pt` (80 classes). That is **not** a retail product catalogue: bottles, backpacks, handbags, cups, and other COCO labels can appear. Store SKUs, concealment, ownership of an item, and shoplifting are out of scope.

## Weights on this machine

Inspected under `member2-person-tracking/models/`:

| File | Classes | Use for Member 1 |
| --- | --- | --- |
| `yolov8n.pt` | 80 COCO classes | **Yes — default** |
| `yolov8n-person-track.pt` | 1 (`item`) | No — person/item tracker, rejected |
| `yolov8n_finetuned.pt` | 1 (`person`) | No |
| `yolov8n-cam-multiperson.pt` | 1 (`item`) | No |

The prototype never silently substitutes a person-only file. It does not download weights.

You may copy `yolov8n.pt` into `member1-object-detection/models/` if you want a Member 1 local copy. The resolver checks that folder first, then the Member 2 models folder.

## Commands

Webcam (close the Member 2 dashboard first — DirectShow does not share the camera):

```bash
python member1-object-detection/src/member1_live_demo.py --source webcam
```

Video file:

```bash
python member1-object-detection/src/member1_live_demo.py --source path/to/clip.mp4 --no-window --json-out member1-object-detection/outputs/detections.jsonl
```

Headless OpenCV builds need `--no-window`. Optional filters: `--classes bottle,backpack,handbag --conf 0.4 --weights member2-person-tracking/models/yolov8n.pt`.

Tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run from `member1-object-detection/`. Mocked tests do not need a webcam. `test_real_model_smoke.py` loads `yolov8n.pt` when that file exists.

## JSON contract

Prototype result (fields come from the model, not from this example):

```json
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
```

`pipeline_detections(result)` maps that onto the team handoff in [docs/api-contract.md](../docs/api-contract.md): `class`, `confidence`, `bbox` as `[x_min, y_min, x_max, y_max]`. Extra prototype fields stay in this module until the team extends the shared contract.

Python API: `ObjectDetector.detect_objects(frame, frame_id, timestamp)`.

## Integration boundaries

- **Member 2** keeps person detection, ByteTrack IDs, and zones.
- **Member 1** only emits object boxes. Overlap with a person box is not ownership.
- **Member 3** scoring rules are unchanged. Object detections are not auto-labelled as interaction, concealment, or shoplifting.
- **Member 4** may later consume these boxes through an agreed interface.

The dashboard Object Detection page, when enabled, reuses the Member 2 live session frames. Do not start this webcam demo at the same time as the dashboard.

## Folder guide

| Path | Use |
| --- | --- |
| `src/` | Detector, capture, overlay, live demo |
| `tests/` | Mocked unit tests and optional real-weight smoke |
| `models/` | Optional local copy of COCO weights (gitignored) |
| `configs/detector.yaml` | Confidence, image size, device |
| `outputs/` | JSONL dumps. Not committed |
