# Data

Local video and derived data for development and experiments. These folders are shared locations, not a fifth code module.

Nothing in `raw/`, `processed/`, or `samples/` is committed. `.gitignore` keeps the placeholders and ignores the contents. Do not commit datasets, CCTV recordings, or large extracts unless the team has specifically agreed.

## Folders

| Path | What to put here |
| --- | --- |
| `raw/` | Original videos or exports, unchanged |
| `processed/` | Frames, clips, or tables derived from raw data |
| `samples/` | Very small local clips used while developing a module |

## Who uses this

Any member may read local data from here. Member 1 is the usual entry point, because detection consumes video. Later stages normally consume the previous module's JSON rather than the raw video.

Agree on file names before several people depend on the same clip. A short note in your module README is enough. Do not add the video itself to Git.

## Status

Folders only. No dataset has been added.
