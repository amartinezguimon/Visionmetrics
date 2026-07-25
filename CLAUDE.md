# CLAUDE.md — VisionMetrics AI (AI-AD)

> Always-loaded context for a fresh chat. Keep it short and true. **When the repo
> structure or a core decision changes notably, update this file** (and add a line to
> the Update log at the bottom). ROADMAP.md is the detailed living plan; this file is
> the quick orientation.

## What this is
Privacy-preserving computer vision that measures **shop-window engagement** — foot
traffic (who passes), attention (who stops to look), dwell time — as a multi-tenant
SaaS, targeting a real-store pilot (target: August 2026).

**Golden rule (never violate):** the **video never leaves the store**. All CV runs on
an edge box; only **anonymous aggregate numbers** (counts per time window) go to the
cloud. No faces, no identities, no images stored or transmitted. Counting is by **zone
crossings, not face recognition** (GDPR / AI-Act friendly).

## Where things are
```
visionmetrics/            # the product (edge + shared + training)
  edge/agent/             #   CV pipeline — the heart. Key files below.
  edge/tools/             #   operator tools: draw_zone, calibrate, check_cameras
  edge/config/            #   device.example.yaml (per-box config)
  edge/deploy/            #   systemd/NSSM service install
  shared/                 #   schema.py — the edge↔cloud data contract
  training/               #   collect → build_dataset → train (engagement model)
    web/                  #   review.html labeling page (baked by prep.py)
    detector/             #   YOLO fine-tuning
  tests/                  #   ~unit tests (no camera/GPU needed)
cloud/                    # backend + dashboard
  app/                    #   FastAPI: ingest, auth, dashboard, admin (multi-tenant)
  web/                    #   React + Vite + TS dashboard (client + /staff back-office)
  scripts/                #   provision, seed_demo, import_report
configs/  data/  models/  fixtures/  recordings/  results/
run.py  DEMO.command  EMPEZAR_AQUI.md   # one-stop demo launcher + Spanish guide
README.md  SYSTEM_DESIGN.md  ROADMAP.md
```

### edge/agent — the pipeline (read these first for CV work)
- `pipeline.py` — `EngagementPipeline`: orchestrates the layers, **returns data, draws
  nothing**. The 7-layer flow: YOLOv8 detect → ByteTrack → `TrackReconciler` (heals id
  switches) → MediaPipe head-pose + torso → `EngagementNet` (3→16→8→1 MLP, Sigmoid →
  P(engaged)) → zone/counting gating.
- `classifier.py` — `EngagementNet` def + `EngagementClassifier.load/.probability`.
  Architecture MUST stay in lock-step with `training/train.py`.
- `zone.py` — `CountingRegion` (feet-in-polygon gate), `EngagementZone`, `GazeReference`
  (re-centre head angles onto the calibrated window direction per store).
- `engagement.py` — per-person state machine (attention windows, thresholds).
- `tracking.py`, `geometry.py`, `camera_model.py`, `capture.py` (USB/RTSP/file + reconnect).
- `service.py` — headless run loop (systemd). `viewer.py` — optional `--debug` window.
- `emitter.py` + `uplink.py` — window deltas → SQLite buffer → cloud POST (never blocks
  the CV loop).
- `config.py` — `DeviceConfig` typed loader; all per-store settings live in `device.yaml`.
- `webserver.py` — the **live operator dashboard** (stdlib `ThreadingHTTPServer`). One
  `_SharedState` under one lock; camera loop writes, HTTP handlers read. Closes the
  training loop: live label → retrain-to-candidate → compare old-vs-new → promote.
  See conventions below before editing its embedded HTML/JS.

## Run & test
```bash
source venv/bin/activate                          # venv/Scripts/activate on Windows
pip install -r requirements.txt                   # edge + training deps
pip install -r cloud/requirements.txt             # backend deps

python -m pytest visionmetrics/tests -q           # edge/training tests (no camera/GPU)
python -m pytest cloud/tests -q                   # backend tests
```
- **Easiest run:** `python run.py` (or `DEMO.command`) → menu (live model, collect data,
  draw zone). This is what the non-technical colleague uses.
- Edge service: `python -m visionmetrics.edge.agent.service --config <device.yaml> [--debug]`
- Training: `python -m visionmetrics.training.build_dataset` → `python -m visionmetrics.training.train --out models/engagement_model.pth`
- Backend/dashboard: see `cloud/README.md`.

## Conventions & guardrails
- **Verify against a fixture before changing logic.** Refactors must preserve behavior
  first (run the recorded clip in `fixtures/`), improve second.
- **All per-store/per-camera settings in `device.yaml`** — never hardcode constants.
- **Agent/LLM invariant:** any agent layer consumes the pipeline's *structured anonymous
  state* (counts, look/away, dwell, tier) — **never pixels/video, never in the CV hot
  path** (per-frame). Analytical agents run on a slow tick or batch, decoupled from the
  frame loop, with a deterministic offline fallback. (See ROADMAP "Exploratory — live
  analytical agent".)
- **`webserver.py` embedded UI:** `DASHBOARD_HTML` is a plain (non-f) triple-quoted
  string, so JS `${}` template literals pass through literally and `\n` must be written
  `\\n`. Verify UI changes with the preview workflow (extract `DASHBOARD_HTML` → temp
  html → stub `fetch` → drive functions → check canvas pixels/screenshot), not by
  guessing. Helpers: `$ = getElementById`, `pKey(fr,id)`, `download(text,name)`.
- **Model accuracy:** the real bottleneck is *features, not the classifier* — build a
  labeled EVAL set before "improving" the model. Don't gold-plate the MLP on lab data.
  Priority order is in ROADMAP "Model accuracy strategy".
- Docs: `ROADMAP.md` = detailed living plan (I maintain it); `SYSTEM_DESIGN.md` =
  blueprint; `EMPEZAR_AQUI.md` / `training/GUIA_HECTOR.md` = Spanish guides for the
  non-technical colleague.

## Working style with Hector
- Spanish-speaking founder/dev. Often explores ideas out loud ("simplemente por
  explorar") — **don't implement until an explicit go-ahead**; capture accepted ideas in
  `ROADMAP.md` under an "Exploratory / not scheduled" note (decision + why).
- Collaborators: Álvaro / Txema (partners); Hector collects the training data.
- NOTE: the default working directory in some sessions is `~/Desktop/Final Kaggle` (a
  *separate* GoEmotions NLP project). This CV project always lives in `~/Desktop/AI-AD-main/`.

## Update log
- 2026-07-24 — Created this file. Recorded the agent-design invariant (structured state,
  never pixels, never hot-path) and the exploratory live-analytical-agent idea (both in
  ROADMAP). `webserver.py` now closes the training loop (retrain/compare/promote) and
  persists `data/metrics_history.jsonl` + `data/model_performance.jsonl`.
- 2026-07-24 — **Foot-traffic flow / direction of arrival.** `pipeline.py` now resolves
  each departing passer-by's net horizontal displacement into `pipeline.flow`
  (`from_left`/`from_right` + `_engaged` split + `ambiguous`), anonymous (a direction
  sign, never a path). Surfaced live in `/api/stats` (`flow`) and persisted per session
  in `metrics_history.jsonl`; dashboard shows a "Foot-traffic flow" panel. Sides are
  camera-relative — operator names them in `device.yaml`.
