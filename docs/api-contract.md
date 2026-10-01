# API contract

This file defines the data passed between modules. It does not define algorithms.

Members may develop locally by reading and writing JSON files that match these examples. The later pipeline in `integration/` must use the same shapes.

Do not add, remove, or rename fields until the team has discussed the change and updated this file.

## Flow

```text
VIDEO
  ↓
MEMBER 1: OBJECT DETECTION
  ↓
DETECTION RESULTS
  ↓
MEMBER 2: PERSON TRACKING
  ↓
TRACKING RESULTS
  ↓
MEMBER 3: BEHAVIOUR ANALYSIS
  ↓
BEHAVIOUR RESULTS
  ↓
MEMBER 4: RISK PREDICTION
  ↓
FINAL RISK RESULT
```

| From | To | Payload |
| --- | --- | --- |
| Video source | Member 1 | Frames from a video file or CCTV stream |
| Member 1 | Member 2 | Detection result, one object per frame |
| Member 2 | Member 3 | Tracking result, one object per frame |
| Member 3 | Member 4 | Behaviour result, one object per person |
| Member 4 | Later output layer | Risk result, one object per person |

## Shared rules

- JSON, UTF-8.
- `frame_id` is a non-negative integer. The same frame uses the same `frame_id` in detection and tracking.
- `bbox` is `[x_min, y_min, x_max, y_max]` in pixels. The origin is the top-left corner of the frame.
- `center` is `[x, y]` in pixels.
- `confidence` is a number from 0 to 1.
- `person_id` is an integer assigned by tracking. It stays the same for one person across frames of the same video. Detection does not create it.
- Scores are numbers. This contract does not define how they are calculated.

## 1. Detection results

Producer: Member 1, `member1-object-detection/`.

Consumer: Member 2.

One object describes one frame.

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

| Field | Meaning |
| --- | --- |
| `frame_id` | Frame this result belongs to |
| `detections` | Objects found in the frame. Empty array when nothing is found |
| `class` | Detected class name. `person` is required for the rest of the pipeline |
| `confidence` | Detector confidence for this box |
| `bbox` | Box in `[x_min, y_min, x_max, y_max]` pixels |

Other class names may appear only after the team adds them to this contract. Member 2 uses `person` detections for tracking.

## 2. Tracking results

Producer: Member 2, `member2-person-tracking/`.

Consumer: Member 3.

One object describes one frame. `person_id` links the same person across frames.

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

| Field | Meaning |
| --- | --- |
| `frame_id` | Frame this result belongs to. Matches the detection `frame_id` |
| `tracks` | People tracked in the frame. Empty array when nobody is tracked |
| `person_id` | Identity of this person for the current video |
| `bbox` | Current box in `[x_min, y_min, x_max, y_max]` pixels |
| `center` | Current centre of the box in `[x, y]` pixels |

Member 3 should treat a video as an ordered list of these objects, sorted by `frame_id`.

## 3. Behaviour results

Producer: Member 3, `member3-behaviour-analysis/`.

Consumer: Member 4.

One object describes one person, summarised from that person's tracking history. It is not a per-frame object.

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

| Field | Meaning |
| --- | --- |
| `person_id` | Same identity assigned by tracking |
| `behaviours` | Indicators found for this person. Empty array when none are found |
| `type` | Indicator name. Initial names are `loitering` and `repeated_shelf_visit` |
| `score` | Numeric score for that indicator |
| `behaviour_score` | Overall behaviour score for this person |

Further indicator names, including abnormal movement, are added here before Member 4 depends on them.

## 4. Risk results

Producer: Member 4, `member4-risk-prediction/`.

Consumer: the later output or dashboard layer, through `integration/` when that layer exists.

One object describes one person.

```json
{
  "person_id": 12,
  "risk_level": "medium",
  "risk_score": 45
}
```

| Field | Meaning |
| --- | --- |
| `person_id` | Same identity used in tracking and behaviour results |
| `risk_level` | One of `low`, `medium`, or `high` |
| `risk_score` | Numeric risk score for this person |

The mapping from behaviour results to `risk_level` and `risk_score` belongs to Member 4's implementation. It is not defined in this file.

## Local handoff before the pipeline exists

Until `integration/pipeline/` is built, members can pass files manually:

1. Member 1 writes detection JSON that matches section 1.
2. Member 2 reads it and writes tracking JSON that matches section 2.
3. Member 3 reads the tracking sequence and writes behaviour JSON that matches section 3.
4. Member 4 reads it and writes risk JSON that matches section 4.

Keep those files in the producing module's `outputs/` folder. Do not commit them.

## Changing this contract

1. Open a pull request that updates this file.
2. Name every module that must change.
3. Wait for the affected members to review.
4. Merge the contract change before writing code that depends on the new fields.
