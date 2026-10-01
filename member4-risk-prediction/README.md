# Risk Scoring / Prediction Module

## Purpose

Combine behaviour indicators, and any other information the team later agrees to include, into a final risk level and risk score for each person. This is the last of the four research modules. A dashboard or report can read the result later. This folder does not implement that dashboard.

## Responsible member

Member 4.

Work only in `member4-risk-prediction/`. Use the branch `member4-risk-prediction`. Other members may read this folder. They should not overwrite it.

## Expected input

Behaviour results from Member 3, one object per person:

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

If Member 3 is not ready, use a hand-written file in this shape. Extra inputs beyond this object require an update to [docs/api-contract.md](../docs/api-contract.md) before they are used.

Optional local settings belong in `configs/`. Local weight files, if a later method needs them, belong in `models/` and are not committed.

## Expected output

One risk result per person:

```json
{
  "person_id": 12,
  "risk_level": "medium",
  "risk_score": 45
}
```

`risk_level` is `low`, `medium`, or `high`. The mapping from behaviour results to these fields is part of later implementation and is not fixed here.

Write local results under `outputs/`. Do not commit them.

## Future integration point

The later output layer reads this risk result. When the four modules are connected, `integration/pipeline/` is the place that passes behaviour JSON into this module and passes the risk JSON onward. Do not build the dashboard inside this folder.

## Folder guide

| Path | Use |
| --- | --- |
| `src/` | Risk scoring code, when implementation starts |
| `tests/` | Tests for this module only |
| `models/` | Local weight files, if needed. Not committed |
| `configs/` | Scoring settings |
| `outputs/` | Local risk JSON. Not committed |

## Status

Structure only. Implementation has not started.
