# Project Guidelines

## Architecture

This is a four-stage, file-contract-driven Python pipeline:

`video -> member1-object-detection -> member2-person-tracking -> member3-behaviour-analysis -> member4-risk-prediction`

- Each member owns one isolated module. Detection creates per-frame boxes, tracking assigns stable `person_id` values, behaviour analysis consumes ordered tracking history, and risk prediction produces the final per-person risk result.
- The module boundaries are intentional: do not move another member's algorithm into `integration/`, and do not make one module depend on another module's private implementation.
- The exact JSON handoff shapes and field meanings are defined in [docs/api-contract.md](../docs/api-contract.md). Update that contract before adding or renaming fields used by another module.
- `integration/` is currently a placeholder for the future pipeline and shared interfaces. `data/` and `results/` are local research artefact areas, not additional implementation modules.
- Current implementation is concentrated in `member2-person-tracking/` and Phase 1 movement-history code in `member3-behaviour-analysis/`; the other stages are scaffolds.

## Build and Test

There is no root build, lint, or test command yet. The root `requirements.txt` is intentionally empty while the project is being staged.

Set up the shared environment from the repository root when dependencies are introduced:

```powershell
python -m venv .venv
copy .env.example .env
```

### Person tracking

Run these commands from `member2-person-tracking/`. Install this module's dependencies there; its `requirements.txt` contains the currently implemented tracking stack.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m unittest tests.test_tracker -v
python main.py tests/fixtures/sample_detections.json --output outputs/tracking_log.json
```

The video/YOLO path is optional and requires local video/model files:

```powershell
python main.py --video path/to/clip.mp4 --weights models/yolov8n-person-track.pt --output outputs/tracking_log.json
```

Run `python evaluate.py <tracking-json> <ground-truth-json>` from this module to produce local evaluation metrics. Dataset cleaning, training, and batch tracking commands are documented in that module's README.

### Behaviour analysis

Run these commands from `member3-behaviour-analysis/`:

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.test_movement_history -v
python src/build_movement_history.py samples/sample_tracking_frames.json
```

The object-detection, risk-prediction, and integration test directories currently contain placeholders only.

## Conventions

- Run module commands from the module directory. The current scripts use local imports such as `from src.pipeline import ...` and `from movement_history import ...`, so changing the working directory can change whether they resolve.
- Tracking JSON is ordered by `frame_id`; tracking filters detections to `class == "person"` and is the only stage that assigns `person_id`. Behaviour history groups observations by `person_id`, preserves `frame_id` order, and can read position from `center`, `position`, or the centre of `bbox` as described in the behaviour README.
- Keep shared handoffs as UTF-8 JSON matching [docs/api-contract.md](../docs/api-contract.md). Do not add private fields to a cross-module payload and expect downstream code to infer their meaning.
- Local generated JSON, videos, datasets, model weights, and experiment outputs belong in the documented ignored locations (`**/outputs/`, `data/`, `results/`, module `models/`, and the tracking `Dataset/` and `runs/` trees). Never commit `.env`, credentials, CCTV footage, or large weights.
- Preserve the module ownership boundaries: member-specific implementation changes stay in that member's folder; shared docs, root files, contracts, and `integration/` changes should be reviewed through a pull request.
- Keep tracker configuration in `member2-person-tracking/configs/tracker_config.yaml` and zone definitions in `configs/zones.json`; do not hard-code camera-specific zone geometry in the tracking algorithm.
- The custom tracker and the optional Ultralytics path must continue emitting the approved tracking fields (`frame_id`, `tracks[].person_id`, `tracks[].bbox`, and `tracks[].center`). Zone/telemetry fields are for local tracking output unless the shared contract is changed first.

## Reference Documentation

- [README.md](../README.md) — project scope, workflow, and module responsibilities
- [docs/system-architecture.md](../docs/system-architecture.md) — stage boundaries and integration plan
- [docs/development-guidelines.md](../docs/development-guidelines.md) — branches, ownership, dependencies, and ignored artefacts
- [docs/team-contributions.md](../docs/team-contributions.md) — module ownership
- [member2-person-tracking/README.md](../member2-person-tracking/README.md) — tracking, dataset, training, and evaluation commands
- [member3-behaviour-analysis/README.md](../member3-behaviour-analysis/README.md) — movement-history input/output and Phase 1 workflow
