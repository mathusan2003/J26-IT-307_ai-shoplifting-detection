# Person Tracking Module

## Purpose

Track people across video frames and keep a stable `person_id` for each person. This module turns per-frame detections into positions over time. It does not decide whether movement is suspicious and it does not produce a risk level.

Expected later technology may include ByteTrack or DeepSORT. The team has not chosen yet. No tracker is included.

## Responsible member

Member 2.

Work only in `member2-person-tracking/`. Use the branch `member2-person-tracking`. Other members may read this folder. They should not overwrite it.

## Expected input

Detection results from Member 1, one JSON object per frame:

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

Use `person` detections. Ignore other classes unless [docs/api-contract.md](../docs/api-contract.md) is updated. If Member 1 is not ready, use a small hand-written JSON file in the same shape.

Optional local settings belong in `configs/` once implementation starts.

## Expected output

One tracking result per frame:

```json
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
```

`person_id` must stay the same for the same person across frames of one video. `bbox` uses `[x_min, y_min, x_max, y_max]` pixels. `center` is `[x, y]` pixels.

Write local results under `outputs/`. Do not commit them.

## Future integration point

Member 3 reads an ordered sequence of these tracking results in the behaviour analysis module. Until `integration/pipeline/` exists, hand off JSON files that match the contract. Do not change Member 1's detector from this folder.

## Folder guide

| Path | Use |
| --- | --- |
| `src/` | Tracking code, when implementation starts |
| `tests/` | Tests for this module only |
| `configs/` | Tracker settings |
| `outputs/` | Local tracking JSON. Not committed |

## Status

Structure only. Implementation has not started.
