# Object Detection Module

## Purpose

Detect people and other agreed object classes in CCTV or recorded retail video. This module is the first stage of the pipeline. It reports where objects are in each frame. It does not track identity across frames and it does not judge behaviour or risk.

Expected later technology may include YOLO-based object detection. No detector is included yet.

## Responsible member

Member 1.

Work only in `member1-object-detection/`. Use the branch `member1-object-detection`. Other members may read this folder. They should not overwrite it.

## Expected input

- A video file or CCTV frame source chosen for local development.
- Frame size and frame index from that source.
- Optional local settings in `configs/` and a local weight file in `models/` once implementation starts.

Large weights and videos stay on disk. They are listed in `.gitignore`.

## Expected output

One detection result per frame, as defined in [docs/api-contract.md](../docs/api-contract.md):

```json
{
  "frame_id": 100,
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [100, 120, 250, 400]
    }
  ]
}
```

`bbox` is `[x_min, y_min, x_max, y_max]` in pixels, origin at the top-left. The class `person` is what tracking requires.

During local development, write sample results under `outputs/`. Do not commit that folder's contents.

## Future integration point

Member 2 reads these detection results in the person tracking module. Until `integration/pipeline/` exists, hand off a JSON file that matches the contract. Do not call tracking code from this folder.

## Folder guide

| Path | Use |
| --- | --- |
| `src/` | Detection code, when implementation starts |
| `tests/` | Tests for this module only |
| `models/` | Local weight files. Not committed |
| `configs/` | Detection settings |
| `outputs/` | Local detection JSON and debug artefacts. Not committed |

## Status

Structure only. Implementation has not started.
