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

## Current development status

CURRENT STATUS:
Project structure created.
Implementation has not started yet.
