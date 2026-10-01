# System architecture

## Pipeline

The system is a sequence of four modules. Each module owns its method. The next module receives only the documented output.

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

| Stage | Owner | Reads | Writes |
| --- | --- | --- | --- |
| Object detection | Member 1 | Video frames | Detection results |
| Person tracking | Member 2 | Detection results | Tracking results with person IDs |
| Behaviour analysis | Member 3 | Tracking results over time | Behaviour indicators and a behaviour score |
| Risk prediction | Member 4 | Behaviour results | Risk level and risk score |

Field-level formats are in [api-contract.md](api-contract.md).

## Local development, then integration

Members develop on their own machines inside their own folders.

- Member 1 can run detection on a local video and write detection JSON.
- Member 2 can read that JSON, or a hand-made sample that matches the contract, and write tracking JSON.
- Member 3 can read tracking JSON and write behaviour JSON.
- Member 4 can read behaviour JSON and write a risk result.

Those intermediate files belong in the module `outputs/` directory and are not committed.

When the four outputs are stable, `integration/pipeline/` becomes the place that calls the modules in order. `integration/interfaces/` is the place for the shared handoff definitions. `integration/tests/` is the place for checks that a module output still matches the contract.

## Boundaries

- Detection does not assign person IDs. Tracking does.
- Tracking does not decide whether movement is suspicious. Behaviour analysis does.
- Behaviour analysis does not assign the final risk level. Risk prediction does.
- A later dashboard or report reader consumes the risk result. That layer is outside the four modules and is not part of the current structure.

## Repository roles

| Path | Role |
| --- | --- |
| `docs/` | Shared project documents and the interface contract |
| `data/` | Local raw video, processed data, and small samples |
| `member1-object-detection/` | Detection code, configs, weights, and outputs |
| `member2-person-tracking/` | Tracking code, configs, and outputs |
| `member3-behaviour-analysis/` | Behaviour analysis code, configs, and outputs |
| `member4-risk-prediction/` | Risk scoring code, configs, weights, and outputs |
| `integration/` | Later connection of the four modules |
| `results/` | Screenshots, videos, and reports for the dissertation |

## Changing the architecture

A change that affects another member's input or output must update [api-contract.md](api-contract.md) first, in a pull request the affected members review. Module-internal code can change without a contract update when the JSON shape stays the same.
