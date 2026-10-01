# Team contributions

Four members share this repository. Each member commits their own work to their own module folder. Everyone can read the other folders. Nobody replaces another member's module directly.

Shared documents in `docs/` and the root project files are changed through pull requests.

## Member 1 — Object Detection

- Folder: `member1-object-detection/`
- Branch: `member1-object-detection`
- Owns detection of people and objects from CCTV or video.
- Later provides detection results to the person tracking module.
- Expected technology may include YOLO-based object detection. The choice is still open.
- Does not assign person IDs and does not score behaviour or risk.

Commit detection code, tests, and configs only under `member1-object-detection/`. Keep model weights local.

## Member 2 — Person Tracking

- Folder: `member2-person-tracking/`
- Branch: `member2-person-tracking`
- Owns tracking of detected people across video frames.
- Later provides person IDs and movement or position information to the behaviour analysis module.
- Expected technology may include ByteTrack or DeepSORT. The choice is still open.
- Consumes Member 1's detection JSON. Does not change the detector to compensate for tracking.

Commit tracking code, tests, and configs only under `member2-person-tracking/`.

## Member 3 — Suspicious Behaviour Analysis

- Folder: `member3-behaviour-analysis/`
- Branch: `member3-behaviour-analysis`
- Owns analysis of customer movement patterns.
- Covers loitering detection, repeated shelf visit detection, abnormal or suspicious movement indicators, and behaviour score generation.
- Later sends behaviour results to the risk scoring module.
- Consumes Member 2's tracking JSON. Does not assign the final risk level.

Commit behaviour analysis code, tests, and configs only under `member3-behaviour-analysis/`.

## Member 4 — Risk Scoring / Prediction

- Folder: `member4-risk-prediction/`
- Branch: `member4-risk-prediction`
- Owns combining behaviour indicators and other agreed information into a final risk level and score.
- Later communicates that result to the output or dashboard layer through `integration/`.
- Consumes Member 3's behaviour JSON.

Commit risk scoring code, tests, configs, and notes only under `member4-risk-prediction/`. Keep model weights local.

## Shared work

| Item | Where it lives | Rule |
| --- | --- | --- |
| Data contracts | `docs/api-contract.md` | Discuss, then change by pull request |
| Pipeline connection | `integration/` | Add only when modules are ready to connect |
| Local video and datasets | `data/` | Keep on disk. Do not commit |
| Report figures and clips | `results/` | Keep on disk unless the team agrees to commit a specific file |

## Seeing each other's work

Push your branch and open a pull request into `main`. The other three members can review the branch, read the module, and comment. Merging to `main` is how finished work becomes the shared baseline.

If your module needs a sample of the previous module's output before that module is finished, use a small JSON file that matches [api-contract.md](api-contract.md). Store generated samples in your `outputs/` folder and do not commit them.
