"""Fine-tune YOLOv8 on your own labeled frames, then drop the weights into the agent.

Continues training from the SAME base weights the agent runs today
(`config.models.yolo`, default yolov8n.pt), so the model keeps its general
"what a person looks like" prior and only adapts to your store: your camera
angle, lighting, and the specific false positives you dismissed. On a small,
freshly-collected dataset that adaptation is the whole point — but a small
dataset also risks overfitting / catastrophic forgetting, so by default we
freeze the backbone (`--freeze 10`) and train few epochs.

Build the dataset first:
    python -m visionmetrics.training.detector.build_yolo_dataset --scan .
Then:
    python -m visionmetrics.training.detector.finetune_detector --data data/detector_dataset/data.yaml

The result is a `best.pt`. Point the agent at it by setting `models.yolo` in
your device.yaml to that path (or use the copy this script drops in models/).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def resolve_base(base: str | None, config_path: str | None) -> str:
    if base:
        return base
    from ...edge.agent.config import DeviceConfig

    config = DeviceConfig.load(config_path) if config_path else DeviceConfig()
    return config.models.yolo


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8 on collected detection labels.")
    ap.add_argument("--data", default="data/detector_dataset/data.yaml",
                    help="dataset YAML from build_yolo_dataset")
    ap.add_argument("--base", default=None,
                    help="base weights to start from (default: the agent's config.models.yolo)")
    ap.add_argument("--config", default=None, help="device.yaml, only to read the default base weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="train image size — bigger helps the small/far people the nano model misses")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=10,
                    help="freeze the first N layers (backbone) to avoid overfitting tiny datasets; "
                         "set 0 to train the whole network")
    ap.add_argument("--mosaic", type=float, default=1.0,
                    help="mosaic aug prob: stitches 4 frames so people recur across contexts "
                         "and scales — the effective amplifier for rare far/small people on a "
                         "bbox dataset")
    ap.add_argument("--close-mosaic", type=int, default=10,
                    help="turn mosaic OFF for the final N epochs so training ends on real, "
                         "un-stitched frames — steadier convergence on a small set")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="random resize gain (±fraction); helps span the near/far scales the "
                         "store camera sees")
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="copy-paste aug prob. NOTE: Ultralytics only applies this with "
                         "segmentation masks, so a bbox-only dataset ignores it — left off by "
                         "default; mosaic + the oversampled hard-example tiles do this job here")
    ap.add_argument("--device", default=None, help="cuda device / 'cpu' / 'mps' (default: auto)")
    ap.add_argument("--project", default="runs/detector", help="where Ultralytics writes the run")
    ap.add_argument("--name", default="finetune", help="run name under --project")
    ap.add_argument("--out", default="models/yolo_finetuned.pt",
                    help="copy the resulting best.pt here for easy reuse")
    a = ap.parse_args()

    if not Path(a.data).exists():
        print(f"[finetune] dataset YAML not found: {a.data}\n"
              "[finetune] run build_yolo_dataset first.")
        return 1

    from ultralytics import YOLO  # lazy: heavy import, and keeps --help fast

    base = resolve_base(a.base, a.config)
    print(f"[finetune] base weights: {base}")
    print(f"[finetune] dataset:      {a.data}")
    print(f"[finetune] {a.epochs} epochs · imgsz {a.imgsz} · freeze {a.freeze} layers")
    print(f"[finetune] aug: mosaic {a.mosaic} (off last {a.close_mosaic} ep) · "
          f"scale ±{a.scale} · copy_paste {a.copy_paste}")

    model = YOLO(base)
    train_kwargs = dict(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        freeze=a.freeze, project=a.project, name=a.name, exist_ok=True,
        mosaic=a.mosaic, close_mosaic=a.close_mosaic, scale=a.scale, copy_paste=a.copy_paste,
    )
    if a.device is not None:
        train_kwargs["device"] = a.device
    model.train(**train_kwargs)

    # Ask the trainer where it ACTUALLY saved, don't guess. Ultralytics resolves a
    # relative `project` under its global runs_dir (settings.yaml), so the run can
    # land at e.g. ~/AI-AD/runs/detector/finetune instead of ./runs/detector/finetune.
    # The old code assumed the CWD-relative path and reported best.pt "missing"
    # whenever a global runs_dir was set — even though training fully succeeded.
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", "")
                    or (Path(a.project) / a.name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        alt = Path(a.project) / a.name / "weights" / "best.pt"  # legacy fallback
        if alt.exists():
            best = alt
    if not best.exists():
        print(f"[finetune] training finished but {best} is missing — check the run output above.")
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, a.out)
    print(f"\n[finetune] best weights -> {best}")
    print(f"[finetune] copied to     -> {a.out}")
    print("[finetune] to use it, set  models.yolo  in your device.yaml to this path, "
          "then restart the agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
