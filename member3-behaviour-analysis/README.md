# Suspicious Behaviour Analysis Module

## Purpose

Analyse how each tracked person moves through the scene and record suspicious indicators. This module covers:

- Loitering detection
- Repeated shelf visit detection
- Abnormal movement
- Object interaction
- Possible concealment
- A behaviour score for each person
- A time-window assessment that combines those signals before assigning a suspicious level
- A trained logistic-regression model that separates suspicious tracks from ordinary ones

It does not detect objects, assign track IDs, or write Member 4's risk JSON.

## Responsible member

Member 3.

Work only in `member3-behaviour-analysis/`. Other members may read this folder. They should not overwrite it.

## Expected input

An ordered sequence of tracking results from Member 2. The shared contract is:

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

`timestamp` and `zone` are optional. When they are absent, time is `frame_id / fps` (default 25, the same rate Member 2 targets) and the zone is the polygon that contains the bottom-centre of `bbox`. That anchor matches Member 2's zone analytics. `shelf` is accepted as another name for `zone`. `position` is accepted when `center` is absent.

Member 2's tracking JSON does not include zone names. The polygons in `configs/zones.json` are the same 1280×720 layout as `member2-person-tracking/configs/zones.json`. Recalibrate both files together for a different camera.

Optional object context uses Member 1's detection shape, passed separately with `--detections`. Only non-person classes are used. Person boxes in that file are ignored.

```bash
python src/analyse_behaviour.py path/to/tracking.json --detections path/to/detections.json --output outputs/behaviour.json
```

## Expected output

One behaviour result per person, for Member 4. A full run is a JSON array ordered by `person_id`.

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

Allowed `type` values are `loitering`, `repeated_shelf_visit`, `abnormal_movement`, `object_interaction`, and `possible_concealment`. An indicator is omitted when it does not fire. Each behaviour object still has only `type` and `score`. `behaviour_score` is the sum of those indicator scores, capped at 100. It is not a suspicious alert.

`assessment` is a separate object. Member 4 can ignore it. `suspicious_level` is `none`, `medium`, or `high`. `combined_score` sums only the indicators that fell inside the selected time window, and it is 0 when the level is `none`. Neither field is Member 4's `risk_level` or `risk_score`.

`outputs/` is local and is not committed.

## How the indicators are scored

Thresholds live in `configs/behaviour_config.json`.

| Indicator | When it fires | Score |
| --- | --- | --- |
| `loitering` | Longest time spent inside a 40 px radius is at least 15 s | 20, plus 0.5 per extra second, capped at 40 |
| `repeated_shelf_visit` | The same shelf is left and entered again. A shelf is any zone whose name contains "shelf" | 25 per return, capped at 40 |
| `abnormal_movement` | Speed reaches 250 px/s, or there are at least 4 sharp reversals faster than 30 px/s | 10 for speed, up to 10 for reversals, plus 10 when that path is also highly folded, capped at 30 |
| `object_interaction` | Small handling movements (5–35 px/s) continue on one shelf for at least 4 s, or a non-person detection stays near the person for at least 3 frames | Up to 15 for handling and 10 for object proximity, capped at 25 |
| `possible_concealment` | The later bounding box shrinks to 72% or less of the earlier median, after the person has been at a shelf or next to an object | 18, plus 12 when a nearby object then disappears, capped at 30 |

Standing still is loitering, not object interaction. Handling needs visible small movement, or a real object detection. Concealment is only a possible signal: this module cannot see hands, pockets, or items inside a bag. A box that is small for the whole track is not concealment, because the person may just be far from the camera.

## Combined assessment

The five detectors still score one indicator at a time. A suspicious level is assigned only afterwards, from indicators whose time spans fall inside `assessment.window_seconds` (30 seconds in the shipped config).

Loitering, a repeated visit to the same shelf, and examining or picking up a product (`object_interaction`) do not raise a suspicious level. That remains true when those signals occur together. A level requires at least two indicators in one window, and at least one of them must be outside that normal-shopping set. One indicator on its own, including abnormal movement or possible concealment, stays at `none`. Indicators separated by more than the window are not combined. An indicator with no start and end time is not treated as if it happened at the same time as the others.

| `suspicious_level` | When it is used |
| --- | --- |
| `none` | No qualifying window. `combined_score` is 0 |
| `medium` | Two qualifying indicators in one window |
| `high` | Three or more qualifying indicators in one window |

`not_contributing` gives the reason an indicator was left out: `context_only`, `isolated`, `outside_window`, or `missing_time`.

The window, the minimum count, the normal-shopping names, and the medium/high counts are in `configs/behaviour_config.json`. They are configurable assumptions. They have not been validated on labelled retail video, and they are not a measured false-alarm rate. Do not describe `suspicious_level` as real-world accuracy.

## Movement history

The Phase 1 history is still the per-person position log. Behaviour checks read it from `MovementHistoryStore`. Bounding boxes are kept on the in-memory record and are not written into the history JSON.

```bash
python src/build_movement_history.py samples/sample_tracking_frames.json
```

## Behaviour model

`src/model.py` trains an L2-regularised logistic regression on 15 measurements (duration, speeds, reversals, shelf visits, handling time, box-shrink ratio, and object contact). The standard library is enough. No extra packages are required.

The model does not replace the indicator scores. Member 4 reads the rule JSON. The model is the learned check that those measurements separate suspicious tracks from ordinary ones.

There is no retail CCTV and no public shoplifting label file in this repository (`data/` is empty). Training uses 80 synthetic sequences in `datasets/labelled_sequences.json`: 40 ordinary and 40 suspicious, with a fixed seed. Within each pattern, every fourth copy is held out, which gives 58 training sequences and 22 test sequences. The generator is `src/dataset.py`.

```bash
python src/train_behaviour_model.py
```

That command rewrites the dataset, `models/behaviour_model.json`, `evaluation/metrics.json`, and `samples/member2_contract_loiter.json`. The metrics file is the result to cite. It includes precision, recall, F1, specificity, and the confusion matrix for the rules and the model, plus a leave-one-family-out check.

The leave-one-family-out check trains without one suspicious pattern and then scores that pattern. It shows whether the linear model recognises a behaviour it did not see. The rules do not use that check; they score every pattern from the thresholds above.

On the synthetic test split (22 sequences: 10 suspicious, 12 ordinary) the rules and the logistic regression both scored accuracy 1.0, precision 1.0, recall 1.0, and F1 1.0. The model confusion matrix was 10 true positives, 12 true negatives, and no errors. The same rules also matched all 80 labelled sequences, not only the test split. These figures are in `evaluation/metrics.json`. They show that the pipeline is consistent with the patterns it was given. They are not a CCTV result.

Leave-one-family-out recall, with ordinary-track specificity 1.0 in every case, was:

| Pattern held out | Recall on that pattern |
| --- | --- |
| Loitering | 1.0 |
| Repeated shelf visit | 1.0 |
| Abnormal movement | 0.0 |
| Object interaction | 0.0 |
| Possible concealment | 1.0 |

The linear model does not recognise abnormal movement or shelf handling when those patterns were absent from training. The rule detectors still score them from the thresholds above. That gap is why Member 4 receives the rule JSON rather than a single model probability.

## Real-video baseline

`datasets/shoplifting-video/` contains the only real clips. The label is the folder name and nothing else:

| Folder | Files on disk | Missing name |
| --- | --- | --- |
| `normal` | 90 | none |
| `shoplifting` | 92 | `shoplifting-49.mp4` |

All 182 files opened and decoded at 30 fps. Most are 640×480. Eight are 1920×1080 (2 normal, 6 shoplifting). Three normal files are exact copies of other normal files: `normal-44` with `normal-52`, `normal-50` with `normal-51`, and `normal-55` with `normal-56`. There is no RetailS or PoseLift data on this machine, and these clips do not annotate loitering, repeated shelf visits, abnormal movement, object interaction, or concealment. Those detectors are unchanged and still run only on Member 2 tracking JSON.

The split is by file hash, not by frame. Within each folder, unique hashes are sorted. The first block is train, the next 15% is validation, and the last 15% is test. Duplicate copies stay in the same split and are not counted twice. The used counts are 125 train (61 normal, 64 shoplifting), 27 validation (13 normal, 14 shoplifting), and 27 test (13 normal, 14 shoplifting). The assignment is in `datasets/video_split.json`.

The baseline is an L2 logistic regression on frame-difference motion, not a deep network. `C` is chosen on validation. It does not write Member 4 JSON.

```bash
python -m pip install -r requirements.txt
python src/train_video_baseline.py
```

On the locked test split the motion model scored accuracy 0.5556, precision 0.5714, recall 0.5714, F1 0.5714, and specificity 0.5385 (8 true positives, 6 false positives, 7 true negatives, 6 false negatives). Training F1 was 0.8992 and validation F1 was 0.7143, with `C` 10. A majority-class guess on the same test videos has accuracy 0.5185 and F1 0.6829. A duration-only model has test accuracy 0.6667 and F1 0.6897. Motion alone is a weak baseline, and clip length already carries label information. Full figures are in `evaluation/video_baseline_metrics.json`.

## Run the tests

From this folder:

```bash
python -m unittest discover -s tests -v
```

## Integration

Member 2 writes a list of tracking frames. This module reads that list and writes a list of behaviour objects. Member 4 reads those objects. `integration/pipeline/` is still the shared place that will call the modules in order. This folder does not copy Member 2's tracker or Member 4's risk mapping.

```bash
python src/analyse_behaviour.py samples/member2_contract_loiter.json --output outputs/behaviour.json
```

That sample uses only `frame_id`, `person_id`, `bbox`, and `center`. The detector result is loitering with score 20 for person 12. The combined suspicious level stays `none`, because loitering on its own is not an alert.

`samples/sample_tracking_frames.json` is the short Phase 1 example. Neither person in it is suspicious.

## Folder guide

| Path | Use |
| --- | --- |
| `src/` | History, detectors, model, training, and the analysis command |
| `configs/` | Score thresholds and the 1280×720 zone map |
| `datasets/` | Synthetic sequences, plus the local Normal / Shoplifting videos and their split |
| `models/` | Logistic-regression coefficients in JSON |
| `evaluation/` | Metrics from the latest training run |
| `samples/` | Small tracking sequences |
| `tests/` | Tests for this module only |
| `outputs/` | Local behaviour JSON. Not committed |

## Limitations

- No real labelled retail video was available when the synthetic logistic regression was trained. The separate video baseline uses `datasets/shoplifting-video/` and is scored in `evaluation/video_baseline_metrics.json`. Those clips are labelled only normal or shoplifting.
- RetailS and PoseLift were not found locally, so they were not used.
- The motion baseline does not beat a duration-only model on the locked test split. Do not treat it as evidence that the five behaviour detectors work on these clips.
- Member 2 does not put zone names in the contract JSON. Zone assignment depends on `configs/zones.json` matching the camera.
- Without timestamps, time assumes a constant frame rate of 25.
- Object interaction from merchandise needs a detection file that contains non-person classes. Without that file, interaction is only the handling-motion rule.
- Possible concealment needs bounding boxes and a shrink after shelf or object contact. It is not a proof that an item was hidden.
- The logistic regression does not set `risk_level`. That remains Member 4's decision. `assessment.suspicious_level` is also not that risk level.
- The 30-second window and the medium/high counts are assumptions. This repository has no labelled retail segments for loitering, repeated shelf visits, abnormal movement, object interaction, or concealment, so those cutoffs are not a measured accuracy.
- The shared pipeline in `integration/` is not wired yet. The handoff is the JSON file described above.

## Status

Movement history, the five tracking-json indicators, the separate time-window assessment, and the Member 4 behaviour JSON are implemented. A separate video-level Normal versus Shoplifting baseline has been trained on `datasets/shoplifting-video/`. That baseline does not add behaviour-type labels. The assessment cutoffs are not a retail accuracy figure.
