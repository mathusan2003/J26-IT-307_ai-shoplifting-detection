# Project overview

## Title

AI-Based Real-Time Shoplifting Detection System

## Team

Four-member final-year university research team. Each member owns one module in this repository.

## Objective

Detect people in retail video, keep a stable identity for each person across frames, analyse movement for suspicious patterns, and produce a risk level that can later be shown to a reviewer.

The intended pipeline is:

```text
Video → Object detection → Person tracking → Behaviour analysis → Risk prediction
```

## Scope of the current stage

This stage creates a shared repository layout and the data contracts between modules.

Included now:

- One folder per team member
- Documentation of responsibilities, architecture, and Git workflow
- JSON examples for detection, tracking, behaviour, and risk outputs
- Places for local data, model files, outputs, and final-report artefacts

Not included yet:

- Object detection, tracking, behaviour analysis, or risk prediction code
- Trained models
- A dashboard, database, or deployment setup

## How the team should use the repository

- Member 1 works in `member1-object-detection/`.
- Member 2 works in `member2-person-tracking/`.
- Member 3 works in `member3-behaviour-analysis/`.
- Member 4 works in `member4-risk-prediction/`.
- Everyone can read the other modules.
- Shared agreements live in `docs/`.
- The `integration/` folder is used when the modules are connected. It is not a fifth implementation area at this stage.

See [team-contributions.md](team-contributions.md) and [development-guidelines.md](development-guidelines.md).
