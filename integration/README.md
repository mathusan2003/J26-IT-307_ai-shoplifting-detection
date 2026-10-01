# Integration

This folder connects the four modules after each one can produce the JSON described in [docs/api-contract.md](../docs/api-contract.md). It is empty of code on purpose.

## Purpose

Run the pipeline in order:

```text
Video → Object detection → Person tracking → Behaviour analysis → Risk prediction
```

Each member keeps implementing inside their own module. This folder should call those modules, or read their outputs, without copying their algorithms here.

## Responsible use

Any member may propose integration changes. Treat this folder as shared. Change it through a pull request once at least the neighbouring contracts are stable. Do not use it as a place to rewrite another member's module.

## Expected input

A video source, then each module's output:

1. Detection results from `member1-object-detection/`
2. Tracking results from `member2-person-tracking/`
3. Behaviour results from `member3-behaviour-analysis/`
4. Risk results from `member4-risk-prediction/`

## Expected output

The final risk result from Member 4, ready for a later dashboard or report. The result shape is:

```json
{
  "person_id": 12,
  "risk_level": "medium",
  "risk_score": 45
}
```

## Future integration point

| Path | Later use |
| --- | --- |
| `pipeline/` | Code that runs the four stages in order |
| `interfaces/` | Shared handoff points that match `docs/api-contract.md` |
| `tests/` | Checks that module outputs still match the contract |

No pipeline, interface code, or tests are written yet.

## Status

Structure only. Implementation has not started.
