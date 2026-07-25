# training/ — collecting data and (re)training the engagement model

The model is a tiny MLP: **(yaw, pitch, distance) → P(looking)**. It is defined
once in `edge/agent/classifier.py` and reused here, so training and the live agent
can never drift apart.

## Honest note — why collect glasses / caps / distances if the model only sees 3 numbers?
The classifier never *sees* glasses or a cap. But those conditions change the
**distribution and noise** of the three numbers: glasses confuse the eye/cheekbone
landmarks, a cap shifts the forehead/top landmark, far faces are noisier all round.
So collecting them teaches the classifier the realistic spread it must be robust to,
and lets us **measure accuracy per condition** (not just on easy close-up faces).
This is the right next step; the bigger long-term lever (eye-gaze) is in `ROADMAP.md`.

## The workflow (who does what)

**Hector — collect (on the camera). No git, no GitHub.** Download the repo once and
run the collector. Record everyone in a SINGLE run; as different people walk up,
press **G** to mark glasses and **H** to mark a cap/hat — the on-screen label updates
and every captured row is tagged. Walk from near to far so all distances fill up:

```bash
python -m visionmetrics.training.collect --collector hector
# L = mark LOOKING (you decide)   A = mark AWAY   T = continuous capture on/off
# M = switch the continuous label  G = cycle gafas  H = cycle gorra  Q = quit + save
```

When he quits, the tool prints the **full path of the one file** it saved. Hector
just **sends that file to you by WhatsApp or email** — nothing else.

> **What does "looking" mean during collection?** There's no window yet — the
> **camera stands in for the window**, so "looking" = head pointed **at the camera**
> (yaw/pitch ≈ 0). The model learns "head on target = looking"; each real store's
> actual window direction is mapped later at calibration (GazeReference re-centring).
> Note this is *head* direction, not eye gaze (a known ceiling — see ROADMAP).

**You — merge + retrain (you own the code).** Drop the file(s) he sends into
`data/raw_sessions/`, then:

```bash
python -m visionmetrics.training.build_dataset   # merge sessions -> data/engagement_dataset.csv + coverage report
python -m visionmetrics.training.train           # retrain -> models/engagement_model.pth + metrics
```

`build_dataset` prints a **coverage report** (looking/away counts per distance tier
and per condition) and flags where you're thin — i.e. exactly what to collect next.
`train` prints overall accuracy **and a per-condition / per-distance breakdown**, and
writes `models/engagement_metrics.json` so you can compare runs over time. Commit the
new `models/engagement_model.pth` to ship it to the edge boxes.

## Tips for a strong dataset
- **Balance** looking vs away, and cover each distance tier (near / mid / far / very-far).
- **Tag as you go**: press G/H to set glasses/cap for whoever is in front right now.
- **Diversity** beats volume: several people, glasses/caps, lighting, angles — including
  the off-axis/corner camera position real stores use.
- Keep an **independent eval session** you never train on, to measure honest generalisation
  (`train --data <that file>` to score, or keep it out of `raw_sessions/`).

---

# Improving the *detector* (YOLO), not just the engagement model

Everything above trains the engagement MLP — it assumes YOLO already found the
people. But YOLO (`yolov8n.pt`) is a generic pretrained model: in a real
storefront it misses small/far people and sometimes fires on posters,
mannequins or reflections. Those errors are upstream of everything else, so
fixing them is the highest-value data you can collect. YOLO **can** be
fine-tuned on your own footage; `detector/` is the pipeline that does it.

## Where the labels come from

Both labeling surfaces already write a `*_detections.csv` in one shared schema
(`frame_idx, track_id, x1..y2, conf, verdict, …`), where **verdict 1 = real
person, 0 = not a person**:

- **Offline labeler** — `prep.py` samples one **whole frame every 5 s** from a
  recording and bakes it into a self-contained `<clip>_review.html`. Open it,
  and for each drawn box press **Y** (approve — it is a person) / **N** (dismiss
  — not a person), **L**/**A** for looking/not-looking, and **drag on empty
  space** to add a person YOLO missed. It downloads two files: the engagement
  CSV (feeds the workflow above) *and* `<clip>_detections.csv` (feeds this one).
- **Live dashboard** — `webserver.py`'s "Detections" mode writes the same
  `*_detections.csv` into `data/raw_sessions/` as you review, next to the
  auto-recorded `.mp4`.

```bash
python -m visionmetrics.training.prep clip.mp4        # -> clip.json + clip_review.html (auto-opens)
python -m visionmetrics.training.label_latest         # same, on the most recent live recording
```

## Build the dataset, then fine-tune

```bash
python -m visionmetrics.training.detector.build_yolo_dataset --scan .
python -m visionmetrics.training.detector.finetune_detector --data data/detector_dataset/data.yaml
```

`build_yolo_dataset` joins each `*_detections.csv` back to its frame pixels
(embedded in the prep `.json`, or seeked from the video) and writes an
Ultralytics dataset (`images/`, `labels/`, `data.yaml`). The three verdict
types map exactly how a detector needs to learn:

| you did | becomes |
| --- | --- |
| **approve** a box (or **L/A**) | a `person` label — a true positive to keep |
| **dismiss** a box | *nothing* at that spot → trains it as background (kills that false positive) |
| **drag** a missed person | a `person` label YOLO didn't have — the highest-value label |

A reviewed frame with **no** real people becomes an empty-label background
image on purpose (also reduces false positives).

`finetune_detector` continues training from the agent's *current* weights
(`config.models.yolo`), so it keeps YOLO's general "what a person looks like"
prior and only adapts to your store. On a small fresh dataset it **freezes the
backbone** (`--freeze 10`) and trains few epochs to avoid overfitting; it trains
at `--imgsz 960` because bigger images are what recover the far/small people the
nano model drops. It copies the result to `models/yolo_finetuned.pt`.

**Ship it:** point `models.yolo` in your `device.yaml` at that file and restart
the agent — no code change. Keep the old weights around so you can A/B them.

## Honest caveats
- **You need real data first.** A handful of frames won't move YOLO; aim for a
  spread of the false positives and misses you actually see in the store.
- **Training wants compute.** It runs on CPU but is slow; a GPU (or Apple
  `--device mps`) is strongly preferred.
- **Measure before you swap.** Ultralytics prints val mAP per run — don't deploy
  a fine-tune that didn't beat the base on a held-out set (raise `--val-frac`, or
  keep an untouched session to eval on).

## The whole loop
```
record → prep.py (5 s frames) → review.html (approve / dismiss / looking / draw missed)
   → build_yolo_dataset → finetune_detector → best.pt → config.models.yolo → better detector ↺
```
