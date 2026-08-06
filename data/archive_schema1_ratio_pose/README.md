# Archived — schema 1 (old flat 2D ratio) training data

Moved out of `data/raw_sessions/` and `data/` on **2026-08-05**, the day
`yaw`/`pitch` switched from a unitless 2D ratio (nose position relative to the
cheekbone midpoint) to real 3D rotation angles in degrees, via `cv2.solvePnP`
(see `visionmetrics/edge/agent/geometry.py`). This was a deliberate decision
("Option A"): the two definitions are numerically incompatible, and since no
images are ever stored (privacy by design), there is no way to recompute these
rows under the new definition — they cannot be merged with anything collected
from now on.

**Do not feed these files into `build_dataset.py` / `train.py` as-is.** They
are kept here only as a historical record and in case a future need arises to
compare old vs. new model behaviour. `classifier.py`'s `feature_schema` buffer
+ `build.py`'s load-time check will loudly warn if a model trained on data
like this (schema 1) is ever loaded by the current pipeline (schema 2).

## Contents
- `raw_sessions/` — every per-session CSV that was in `data/raw_sessions/`
  (both collector tools: the console `training/collect.py` and the browser
  `edge/agent/webserver.py`), including their `*_detections.csv` siblings.
- `engagement_dataset.csv` — the merged master dataset built from the above.
- `engagement_data.csv` — the original legacy flat dataset (predates even the
  rich session schema).

## Next step
Once real sessions have been recollected with the current pipeline, run
`build_dataset.py` again — it will build a fresh `data/engagement_dataset.csv`
from `data/raw_sessions/` (now empty until new sessions land there) and the
model can be retrained for real.
