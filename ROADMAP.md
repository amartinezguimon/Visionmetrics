# VisionMetrics AI — Roadmap to Pilot (target: August 2026)

> Living source of truth for the productization effort. Claude maintains this file.
> Status legend: ☐ todo · ◐ in progress · ☑ done

## Goal
Turn the working uni prototype into a SaaS-ready product: an **edge agent** that runs
unattended in a store (reading from USB/RTSP camera), streams **anonymous aggregate
metrics** (never video) to a backend on Txema's VM, and a **web dashboard** where a
client sees their stores. Pilot in real stores by August.

## Golden rules
1. **Video never leaves the store.** Only anonymous aggregate metrics travel.
2. **Refactor preserves behavior first, improve second.** Verify against a recorded
   clip fixture before changing logic.
3. **All per-store/per-camera settings live in `device.yaml`,** never hardcoded.
4. **Everything in the cloud runs in one `docker-compose`** on the VM (no k8s).

## Architecture (3 rings)
`Camera (USB/RTSP) → Edge box (mini-PC, agent) → Cloud VM (ingest+DB+API) → Web dashboard`

---

## Phase 1 — Edge agent (Weeks 1–3) — *60% of the work, 90% of the risk*
- ☑ Monorepo structure under `visionmetrics/`
- ☑ Config system: `device.yaml` + typed loader (kills all hardcoded constants)
- ☑ Extract pure logic into tested modules:
  - ☑ `geometry.py` — face-angle extraction (shared by agent + calibrate; was duplicated)
  - ☑ `camera_model.py` — pinhole focal/distance (was duplicated)
  - ☑ `engagement.py` — per-person state machine, windows, thresholds
  - ☑ `zone.py` — soft engagement-zone confidence
  - ☑ `classifier.py` — EngagementNet load + classify (verified vs real trained weights)
- ☑ `vision/` glue: detector (YOLO), face (MediaPipe), pose (MediaPipe)
- ☑ `capture.py` — video source (USB/RTSP/**file**) with auto-reconnect
- ☑ `pipeline.py` — `EngagementPipeline` orchestrating the 7 layers, **returns data, draws nothing**
- ☑ `viewer.py` — optional OpenCV debug window (`--debug`), separate from the service
- ☑ `service.py` — headless run loop, **no `input()`, no `imshow`** (CLI: `--config`, `--debug`)
- ☑ `build.py` — wires DeviceConfig → ready EngagementPipeline
- ☑ Run as OS service: systemd unit + install script (Linux) + NSSM guide (Windows), auto-restart — see `edge/deploy/` *(install on real box pending)*
- ☑ Test fixture: recorded clip (45s, 1280x720) → pipeline runs headless against it
- ☑ Unit tests (38 total: pure logic + capture via synthetic video + pipeline via fakes)

### Clip verification findings (real footage)
- ☑ **Fixed a real product bug**: the ghost filter permanently blacklisted any
  track that showed no face in its first ~20 frames. A customer who approaches
  with their back turned and only later looks at the display was killed early
  and never recovered (engaged=0). Replaced the permanent veto with a
  *confirm-on-first-face* model (`ghost_recheck_every`): a track is counted only
  once a face confirms it's a person, and unconfirmed tracks keep being checked
  forever. Chairs (never a face) still never count. Verified: engaged 0 → 1.
- ☑ File playback now uses **video time** (frame/fps) for attention, not wall clock.
- ☑ **Investigated the "attention 21s" item against the user's ACCURATE ground
  truth — the detector is correct; there is nothing to tune.** The earlier
  "~10s looked" figure was a bad memory; the real timeline is: 1-15s not looking,
  16-24s looking (moving then still), 27-28s looking (torso turned), 29-35s not
  looking, 35-42s looking while moving ≈ **~17s genuinely looking**. The pipeline
  detected three engaged windows — 15.8-25.4s, 27.3-29.7s, 35.5-44.0s — which map
  one-to-one onto the looking segments, and stayed silent on both not-looking
  segments (the 29.7→35.5s gap matches "29-35 not looking" to the second).
  Measured attention 20.6s vs ~17s true = **~3s of over-count, ~1s per window
  end**, from the head reading frontal for ~1s while turning away. That is natural
  and acceptable, not a defect. NOTE: my prior "walking-toward-webcam false
  positives / fix via geometry+zone" conclusion was WRONG — it was built on the
  bad ~10s figure; the user was in fact looking during those windows.
- ☑ **Fixed track-id-switch double counting (field-reported by Hector).** A still
  person dropped for an instant was re-detected under a new ByteTrack id and
  re-counted (+ engagement reset). Two-layer fix: (1) larger ByteTrack
  `track_buffer` keeps a lost track's id alive longer (detector.py, configurable);
  (2) `tracking.py` `TrackReconciler` — a pure IoU-based reconciler that adopts a
  new id appearing where a track was just lost into that track (grace window +
  min-IoU, both configurable). The tracker now also `drop()`s departed people to
  free memory while banking their attention into a monotonic session total
  (`_departed_attention_s`) — which also resolves the old "attention not
  monotonic" emitter caveat. 11 new tests (IoU, adoption, expiry, the
  re-association regression, departed-but-attention-kept). Suite: 66 total.
- ☐ Known, deferred: MediaPipe finds a face ~54% of frames when subject is very
  close / filling the frame — revisit head-crop logic for close range.

### Passerby vs engagement decoupled (foot-traffic counting)
- ☑ **Decision: `passersby` = real foot traffic, not just face-confirmed people.**
  The old design only counted someone once their FACE was seen (to reject ghosts),
  which under-counts a street (back-turned / far / sideways people never counted —
  street clip read 49). Now a track counts as a passerby once it has PERSISTED
  (`passerby_min_frames`) AND either moved (`passerby_motion_px`) or shown a face
  — rejecting flicker + static furniture (a chair never does both). Engagement
  still requires a face. Replaced the old `ghost_recheck_every` confirm-on-first-
  face gate. Verified: street clip 49 → 59 passersby, engaged still 0 (correct);
  test_store unchanged (1 passerby, 1 engaged). Remaining under-count on the
  street clip is the **yolov8n (nano) model missing small/far people** + promenade
  geometry — a bigger YOLO (yolov8s/m via models.yolo) is the lever if needed.

### Robustness audit (field-reported "two people close / hugging" + general)
- ☑ **Wrong-face in overlapping crops (the reported flicker).** A head crop is
  taken from one person's bbox; when a second person is very close their face
  intrudes and MediaPipe (num_faces=1) returned an arbitrary one → engaged
  flickered between the two. Fixed: detect up to 2 faces and keep the most-
  centred one (`most_centred_face`), i.e. the crop's rightful owner. No-op for a
  lone person (verified: clip unchanged at 20.6s).
- ☑ **Reconciler could fuse two close people (regression I introduced).** Added an
  ambiguity guard: adopt a new id into a lost track only when there is ONE clear
  IoU candidate; if two lost tracks overlap, decline (counting a returner twice
  beats merging two distinct people).
- ☑ **One bad frame killed the whole agent.** `service.py` now guards
  `process_frame`: a transient failure is logged and skipped; it bails (exit 1,
  for the OS service to restart) only after 30 consecutive errors.
- ☐ Deferred (need a 2-person clip to tune, not blind): (a) aspect-ratio filter
  may drop a merged 2-person box; (b) ByteTrack ID *swap* between two people who
  cross/hug (needs appearance re-id, post-August); (c) classifier distance
  feature is sensitive to bbox jitter (low impact — saturates frontally).
- ☑ Reviewed clean: geometry/camera_model (eps + clamps), zone, capture (RTSP
  reconnect), classifier (weights_only), uplink (SQLite lock). Minor note:
  reading cv2 capture props from the main thread while the pump thread reads is
  not strictly thread-safe (low risk).

## Phase 2 — Data contract + uplink (Weeks 3–4) — *done*
- ☑ `shared/schema.py` — the edge↔cloud contract (heartbeat + metric bucket) *(done early)*
- ☑ `emitter.py` — `MetricEmitter`: watches the tracker's cumulative counters and
  emits one `MetricBucket` per closed time window (delta-based, like the old
  `write_hourly_snapshot`). Pure + tested; window length from `device.yaml`.
- ☑ `uplink.py` — `UplinkBuffer` (SQLite, deduped on `idempotency_key`, survives
  reboots) + `Uplink` (background sender thread, **never blocks the CV loop**,
  short-timeout `urllib` POST, retry-on-next-cycle, injectable transport for tests).
- ☑ Wired into `service.py`: per-frame `emitter.sample`, periodic heartbeat,
  final-window flush + clean thread shutdown. Disabled path prints buckets locally.
- ☑ 17 new unit tests (windowing/deltas + buffer durability/retry). Suite: 55 total.
- ☐ Known, deferred: attention delta clamps to >= 0 (tracker's attention sum isn't
  strictly monotonic when a track is forgotten) — a monotonic lifetime counter is
  a later hardening item. Noted in `emitter.py`.

## Phase 3 — Minimal backend on the VM (Weeks 4–6) — *in progress*
- ☑ Multi-tenant schema day 1 (`cloud/app/models.py`): `org → store → device →
  camera`, users/roles, metric_buckets, heartbeats, platform_staff; `org_id` on
  every data row; `reseller_id` nullable for the future.
- ☑ Ingest API (`cloud/app/routers/ingest.py`): `POST /v1/metrics` (device-auth,
  idempotent on (device, window_start)) + `POST /v1/heartbeat` (fleet health).
  Tenant-safe: org/store taken from the device row, not the body. Wire format
  matches `shared/schema.py`.
- ☑ Device provisioning (`cloud/app/provisioning.py` + `scripts/provision.py`):
  generate `device_id` + `api_key` (key shown once, only hash stored) + create
  org/store/user. CLI ready.
- ☑ Dashboard auth + read API: `POST /v1/auth/login` (JWT) + `/v1/me`,
  `/v1/stores`, `/v1/devices` (fleet health), `/v1/metrics/summary` &
  `/timeseries` (store + date-range filters), `/v1/users` (owner manages users).
  Every query scoped to the user's org / store. Business-language numbers.
- ☑ Runs locally on SQLite (zero setup) + 19 tests (device auth, tenant
  isolation across orgs, bucket idempotency, heartbeat-online, login, role
  guards, store-scope enforcement, summary aggregation). FastAPI (`/docs` auto).
- ☑ Alembic migrations (`cloud/migrations/`) — single source of truth for the
  schema (local SQLite + prod Postgres); stable naming convention; app no longer
  auto-creates tables. Verified: upgrade builds all tables from scratch.
- ☑ `docker-compose.yml`: API (auto-migrates on start) + Postgres + Caddy
  (auto-HTTPS) + Dockerfile + Caddyfile + .dockerignore. Same file local + VM.
- ☑ Platform back-office API (staff, cross-tenant): staff login + `/v1/admin/orgs`
  (list w/ counts + create), `/v1/admin/stores|devices` (onboard a client via API,
  device key once), `/v1/admin/fleet` (all devices, all clients). Client-user
  tokens are rejected from `/v1/admin`. 24 cloud tests. UI is Phase 4.
- ☐ One real test store sending live data to the VM (needs Docker on the VM).

## Phase 4 — SaaS dashboard (Weeks 6–8)
- ☑ Web app (`cloud/web/`): React + Vite + TypeScript + Tailwind, Chart.js — a
  faithful build of the agreed design, wired to the real API. Login (JWT), auto
  store selector when >1 store, "En directo" pill from fleet status.
- ☑ Backend data contract: `GET /v1/dashboard?store_id&year&month` →
  `{daily, dailyPrev, hourly, store, has_data}`, aggregated in the store's
  timezone; empty-store state handled. Demo seeding (`scripts/seed_demo.py`).
- ☑ Home (hourly + monthly + calendar) and Analysis (compare periods, weekday
  comparison) views in plain business language (no yaw/pitch jargon).
- ☐ Fleet health panel (devices online, FPS, last metric) — surfaced as the
  "En directo" pill for now; full panel still to build.
- ☐ **Dwell-time histogram** so the Analysis "Mín. Xs" slider is exact, not an
  estimate: the edge should emit per-bucket counts of lookers by dwell band
  (e.g. 2–5s / 5–10s / 10s+) instead of only a total. Needs edge + schema +
  ingest changes. Until then the slider's `+2s` baseline is real; above that is
  an approximation (flagged in the UI and `cloud/web/README.md`).
- ☑ Staff back-office (monitoring) UI at `/staff`: platform summary cards
  (clients/stores/devices, online/offline/never), and a fleet table for every
  camera across all clients — status, camera health (FPS, camera_ok) and a
  data-flow signal (last metric + passersby/24h). Endpoints `/v1/admin/me`,
  `/overview`, enriched `/fleet`.
- ☑ **Foot-traffic flow (direction of arrival).** The pipeline resolves each departing
  passer-by's net horizontal displacement into `pipeline.flow` (came-from-left vs
  came-from-right, with an engaged split and an `ambiguous` bucket for barely-moving
  tracks) — anonymous (a direction sign, never a path/identity). Live in `/api/stats`
  and persisted per session in `metrics_history.jsonl`; the live dashboard shows a
  "Foot-traffic flow" panel (split bar + "what share of each side stopped to look").
  Sides are camera-relative; operator names them in `device.yaml`. Verified: unit test
  on `_tally_flow` + preview render. (Enrichment idea #5 from the 2026-07-24 dashboard
  brainstorm; the remaining ideas — capture-rate funnel, window-change A/B, hour×day
  heatmap, dwell histogram — stay on the backlog.)
- ☐ Remote per-store config editing: a server→edge config channel (server stores
  the config, the edge box pulls it and reconfigures) + a simple form UI in the
  staff panel (camera position, window width, engagement-zone, thresholds), so a
  store can be tuned without a site visit. Touches edge agent + schema + ingest.

## Phase 5 — Harden + validate 2nd store (Weeks 8–10)
- ☑ **Counting zone (field feedback from Hector's live test):** an operator-drawn
  polygon (`counting_region` in the store config) bounds where people count at all
  — feet must fall inside it, for foot traffic AND engagement. Fixes "people too
  far to notice the window get counted" and far/shadow false-engagement, at the
  source. Drawn with `python -m visionmetrics.edge.tools.draw_zone`. Paired with an
  anti-flicker knob `min_engage_window_s` (discard sub-0.4s looking windows so
  micro-glances/shadows don't accumulate). `CountingRegion` in `zone.py`, gated in
  `pipeline.py`, loaded in `build.py`; 88 edge tests.
- ☐ Install in a 2nd real store (different camera/lighting) — *model generalization truth surfaces here*
- ☐ Tune FOV/thresholds per camera via `device.yaml`
- ☐ Polish service start, camera reconnection, crash recovery

---

## Model accuracy strategy — how to actually make "looking" precise
> Honest senior-ML assessment. The current model is a fine, CPU-cheap **v1**, but
> it is **not** the maximally-accurate approach. Pursue in this order — earlier
> items beat later ones on real-world accuracy per euro/hour.

**The real bottleneck is the features, not the classifier.** A face is compressed
into 3 hand-crafted numbers (yaw, pitch, distance from cheekbones/nose) and a tiny
MLP learns from those. The biggest blind spot: it reads **head direction, not eye
gaze** — someone can face forward but look elsewhere with their eyes. Making the
net bigger on the same 3 numbers buys almost nothing.

**Hard real-world requirement (founder spec):** the window can be several metres
wide × ~2 m tall, the camera must sit **discreetly in a corner** (heavily off-axis,
elevated), and people **walk past sweeping their gaze edge-to-edge** from varying
positions — not standing centred at 2 m. This breaks the "point display + camera
beside it" assumption. Honest assessment of what it takes:
- **Achievable foundation — ☑ DONE (code) / ☐ tune in store:** (a) the zone
  already supports a REGION (yaw_min/max etc. from window edges via calibrate.py)
  → a wide angular band the sweep stays inside; (b) **re-centre IMPLEMENTED**
  (`zone.GazeReference`, wired in `pipeline._score` + `build.load_gaze_reference`):
  subtracts the calibrated window direction (`engagement_zone.yaw_center/
  pitch_center`) from yaw/pitch before the classifier, so a corner camera maps
  "looking at the window" onto the model's "straight ahead". Defaults to (0,0) =
  no shift when uncalibrated (verified backward-compatible: test clip unchanged).
  Per-store tuning = the calibration JSON; values await a real install to set.
- **Honest ceilings (can't trick away):** *parallax* — a fixed angular zone is
  calibrated for ~one standing position; with a wide window + people at any
  position the true "looking" angle shifts, so a single zone is an approximation,
  not perfect. True per-person precision needs each person's floor position + the
  window's 3D geometry (a much bigger system). And *head vs eyes* — on a wide
  window people sweep with their EYES; head-pose alone won't be crisp, so **eye
  gaze (MediaPipe Iris) becomes necessary here**, not optional.
- **Path to "crisp":** foundation above → validate+tune in the real store against
  the eval set → add eye gaze. Not achievable by a calibration trick alone; the
  aggregate metric tolerates per-instant imperfection, per-person perfection does
  not come free.

**☑ Tooling for steps 2–3 is BUILT (`visionmetrics/training/`, 2026-06-17):**
collector now tags each session with condition metadata (glasses/headwear/subject/
distance tier) and writes one CSV per session to `data/raw_sessions/`;
`build_dataset.py` merges them into `data/engagement_dataset.csv` and prints a
**coverage report** (looking/away per tier + condition, flags thin cells);
`train.py` reuses the agent's `EngagementNet`, splits-before-augment, and reports
accuracy **per distance tier + per condition** → `models/engagement_metrics.json`.
Workflow (Hector collects → commits sessions → you merge+retrain) in
`visionmetrics/training/README.md`. Baseline on the legacy 1127 rows: 98.7% on the
in-distribution test split — but the coverage report already shows the gaps the new
collection must fill (far tier ≈ 0 looking; glasses/headwear all "unknown").

Priority order (do NOT jump to the bottom):
1. **Deployment geometry + per-store calibration (the zone).** Biggest real-world
   accuracy gain, ~free. Re-centre on the window + calibrate window edges (see
   founder spec above); "looking" = looking at the window, adapted per store.
2. **Build a real labeled EVAL set + metric FIRST.** ~200–300 held-out examples
   from realistic conditions the model never trains on, scored with precision/
   recall on "looking". *Step 0 of any accuracy work — you can't improve what you
   don't measure; without it every "improvement" is guesswork.* (Tooling ready —
   keep an independent session out of `raw_sessions/` and score with `train --data`.)
3. **More + diverse + real data** (consenting people; later real-store footage).
   Diversity (glasses, hats, ages, heights, distances 1–5 m, lighting) > volume.
   Current set = 1127 self-collected lab samples — too small/narrow to generalize.
   (Collector + coverage report now make this systematic.)
4. **Highest-ROI model upgrade: add eye gaze (MediaPipe Iris).** Reuses the stack;
   the single biggest model-side accuracy jump. Do it only after 1–3 exist.
5. **Learned 6DoF head pose** (e.g. 6DRepNet) to replace the jittery 3-number
   geometry. Incremental.
6. **Theoretical max: end-to-end CNN on the head crop** (pixels → P(engaged),
   no hand-crafted features). What a big player would do — but needs thousands of
   real labeled images + beefier hardware (costs the cheap mini-PC). Premature now.

**Traps to avoid:** (a) gold-plating the model before real data + eval exists — a
fancier model on 1127 lab samples still fails in a real store; (b) over-promising
per-frame precision on an inherently fuzzy target — the business metric is the
*aggregate trend*, which the smoothed head-pose proxy likely already serves.
Labeling tool: **`visionmetrics/training/collect.py`** — built on the production
vision pipeline (same detector + head-pose code the agent runs), so it captures
the same far-range faces inference can AND the recorded features match inference
1:1 (no train/serve skew). Captures continuously (L/A single shots or directed
auto-capture by label); you are the ground truth. Replaces the legacy
`src/training/data_collector.py` (which duplicated the pipeline and single-shot
recording made it miss intermittent far-range hits). Retrain = `src/training/
train.py` → drop in `models/engagement_model.pth`. Tip: collect the bulk via
directed auto, but build the held-out EVAL set via careful manual L/A (it's the
measuring stick). *(Lands around Phase 5, once a real install exists.)*

## Exploratory — live analytical agent (idea, not scheduled)
> Captured 2026-07-24 from a design chat. Not on the pilot critical path; note so it
> isn't re-litigated. Post-August unless a pilot explicitly asks for it.

Give the metrics a **voice**: an LLM/agent that periodically narrates *what's
happening now* in anonymous, aggregate terms — a self-filling feed on the dashboard
(*"18:00–18:10 · 18 pasaron, 4 pararon (22%), atención 6s. Ritmo estable."*).

- **Firm decision — the agent never sees pixels.** It consumes the *structured
  anonymous state* the pipeline already produces (counts, look/away, dwell, tier),
  not video. Keeps the golden rule intact and dodges the latency/cost of an LLM in
  the CV hot path.
- **Cadence: sampled, not live, not post-mortem.** A daemon tick every ~10 min reads
  a rolling window of scalars + the previous tick's summary (so it can say "ritmo
  bajando vs bloque previo"). Rejected alternatives: LLM-per-frame (breaks real-time
  + all-local) and end-of-day only (loses the temporal texture).
- **Hook points (verified in `edge/agent/webserver.py`):** ring buffer +
  `insights` timeline on `_SharedState` (~L79); push cumulative scalars 1/s right
  after the `state.stats.update` in the loop (~L898); a `_insight_tick` daemon
  thread started in `run()` that diffs window start/end → deltas → agent →
  `state.insights`; `GET /api/insights` endpoint (~L360); a feed panel polling ~30s.
  Keep a **deterministic template fallback** for 100%-offline mode.
- **MVP vs enrichment:** MVP feeds *aggregate deltas* (passed/stopped/attention) —
  zero pipeline changes. Richer narratives ("most arrived from the left", "they stop
  but don't look") need **per-track summary events** (dwell, looked, entry_side)
  emitted on `pipeline.expire()` (`pipeline.py` ~L182). Start with deltas.
- ☐ Not started. MVP is ~low-medium effort; do only if a pilot wants live insight.

## Deferred until after August (scope discipline)
Stripe/billing · OTA auto-updates · self-service multi-tenant signup · model retrain from scratch.

## Human-in-the-loop (what only the founder can do)
- Provide a **recorded test clip** so Claude can verify the pipeline without a live camera.
- Run commands on Txema's VM (founder pastes output) — *Phase 3, not before*.
- Physically install the edge box + run calibration in real stores.
