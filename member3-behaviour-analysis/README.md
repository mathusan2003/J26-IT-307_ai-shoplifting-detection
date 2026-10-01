# Suspicious Behaviour Analysis Module

## Purpose

Analyse how each tracked person moves through the scene and record suspicious indicators. This module covers:

- Loitering detection
- Repeated shelf visit detection
- Abnormal or suspicious movement indicators
- A behaviour score for each person

It does not detect objects, assign track IDs, or set the final risk level.

## Responsible member

Member 3.

Work only in `member3-behaviour-analysis/`. Use the branch `member3-behaviour-analysis`. Other members may read this folder. They should not overwrite it.

## Expected input

An ordered sequence of tracking results from Member 2. Each frame looks like:

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

Group records by `person_id` and read them in `frame_id` order. If Member 2 is not ready, use a hand-written sequence in this shape.

Optional thresholds and zone settings belong in `configs/` once implementation starts. Any new indicator name must be added to [docs/api-contract.md](../docs/api-contract.md) before Member 4 relies on it.

## Expected output

One behaviour result per person, not per frame:

```json
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
```

Initial indicator names are `loitering` and `repeated_shelf_visit`. How the numbers are calculated is part of later implementation and is not fixed here.

Write local results under `outputs/`. Do not commit them.

## Future integration point

Member 4 reads these behaviour results in the risk prediction module. Until `integration/pipeline/` exists, hand off a JSON file that matches the contract.

## Folder guide

| Path | Use |
| --- | --- |
| `src/` | Behaviour analysis code, when implementation starts |
| `tests/` | Tests for this module only |
| `configs/` | Thresholds, zones, and related settings |
| `outputs/` | Local behaviour JSON. Not committed |

## Status

Structure only. Implementation has not started.
