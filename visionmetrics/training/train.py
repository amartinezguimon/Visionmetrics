"""Train the engagement classifier on the master dataset, with honest per-condition eval.

    python -m visionmetrics.training.train
    python -m visionmetrics.training.train --data data/engagement_dataset.csv --epochs 50

Key choices:
* Reuses the SAME network the agent runs (`edge.agent.classifier.EngagementNet`),
  so training and serving can never drift apart.
* Train/val/test are split by IDENTITY (subject, or session when subject wasn't
  recorded — dataset.group_key), not by row: a plain row-level split lets
  different frames of the SAME person land in more than one pile, so the model
  can partly win by recognising THAT PERSON's own numbers instead of learning
  gaze in general, inflating every metric below.
* Real rows are split BEFORE far-distance augmentation, and only the TRAIN
  slice is augmented — val and test stay 100% real, so a synthetic sample can
  never leak into a number we report as "how well does this generalise".
* The loss reweights the rarer class (data/README: "away" and "looking" are
  rarely balanced), so the model can't win by mostly guessing the majority
  class — see dataset.class_weights.
* Early stopping on a held-out validation set: trains until val loss stops
  improving (patience) instead of a fixed epoch count, and keeps the BEST
  epoch's weights, not just whatever epoch happened to be last.
* The decision threshold is CHOSEN on validation (maximising F1 on "looking"),
  never on test — tuning a hyperparameter against test data is the same kind
  of leakage as the train/test split itself, just one level up.
* Reports overall accuracy/precision/recall AND a breakdown by distance tier and
  by condition (glasses / headwear) — that's how you tell whether new data made
  the model better *where it was weak*, not just on easy close-up faces.
  precision/recall are `null` (not a misleading 0.0) where a slice has zero
  real examples of that class — e.g. "far" has no real "looking" rows yet.
  Writes the report to models/engagement_metrics.json next to the weights.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

LEGACY = "data/engagement_data.csv"


def _resolve_data(path: str) -> str:
    if Path(path).exists():
        return path
    if Path(LEGACY).exists():
        print(f"[train] {path} not found; falling back to legacy {LEGACY}.")
        return LEGACY
    raise SystemExit(f"No dataset at {path} (or {LEGACY}). Run build_dataset first.")


def _fmt(v):
    """None -> 'n/a' for console printing; leaves real numbers untouched."""
    return "n/a" if v is None else v


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the engagement model.")
    ap.add_argument("--data", default="data/engagement_dataset.csv")
    ap.add_argument("--out", default="models/engagement_model.pth")
    ap.add_argument("--epochs", type=int, default=50, help="MAXIMUM epochs (early stopping usually stops sooner)")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.15,
                    help="fraction of the TRAIN pool (post test-split) held out for early stopping + threshold tuning")
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many epochs with no validation-loss improvement")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from . import dataset
    from ..edge.agent.classifier import CURRENT_FEATURE_SCHEMA, EngagementNet

    path = _resolve_data(a.data)
    df = dataset.load_csv(path)
    print(f"[train] {len(df)} samples from {path}")
    print(dataset.coverage_text(df))

    # 1) Grouped by identity, NOT by row (see module docstring). Carve val out
    # of the TRAIN pool only, so test never influences model selection at all.
    train_df, test_df = dataset.group_train_test_split(df, test_size=a.test_size, seed=a.seed)
    have_val = a.val_size > 0 and dataset.group_key(train_df).nunique() >= 2
    if have_val:
        train_df, val_df = dataset.group_train_test_split(train_df, test_size=a.val_size, seed=a.seed)
    else:
        val_df = train_df.iloc[0:0]   # empty — too few identities to carve out a val split
        print("[train] WARNING: too few distinct identities for a validation split — "
              "training the full --epochs with NO early stopping and a fixed 0.5 threshold. "
              "Collect data from more people/sessions to enable this.")

    n_train_ids = dataset.group_key(train_df).nunique()
    n_val_ids = dataset.group_key(val_df).nunique()
    n_test_ids = dataset.group_key(test_df).nunique()
    print(f"[train] grouped split: {n_train_ids} train / {n_val_ids} val / "
          f"{n_test_ids} test identities (0 shared between any two)")

    # 2) Augment ONLY the train slice; val and test stay 100% real rows.
    X_tr = train_df[dataset.FEATURES].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    aug_X, aug_y = dataset.augment_far(X_tr, y_tr, seed=a.seed)
    X_train = np.vstack([X_tr, aug_X])
    y_train = np.concatenate([y_tr, aug_y])
    X_val = val_df[dataset.FEATURES].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=int)
    X_test = test_df[dataset.FEATURES].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)
    print(f"[train] rows: train {len(X_train)} (real {len(X_tr)} + aug {len(aug_X)})  "
          f"| val (real only) {len(X_val)}  | test (real only) {len(X_test)}")

    torch.manual_seed(a.seed)
    model = EngagementNet()
    model.set_feature_schema(CURRENT_FEATURE_SCHEMA)   # tags this checkpoint so a
    # future mismatched pipeline (or an old model loaded here by mistake) is caught
    # loudly at load time instead of silently serving garbage — see classifier.py.
    # Fit the input z-score on the TRAIN set only (never val/test) and store it
    # INSIDE the model (buffers -> saved in the .pth). forward() then standardizes
    # automatically, so train, eval and the live agent all apply the identical
    # transform to raw (yaw, pitch, distance) — no train/serve skew, no side file.
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    model.set_standardization(feat_mean, feat_std)
    print(f"[train] input z-score  mean={np.round(feat_mean, 4).tolist()}  "
          f"std={np.round(feat_std, 4).tolist()}")

    # 3) Class-balanced loss: the rarer class (whichever it is, on THIS data)
    # gets a bigger weight, so the model can't win by mostly guessing the
    # majority class. Computed on the actual sampled training distribution
    # (post-augmentation), since that IS what the loss averages over.
    cw = dataset.class_weights(y_train)
    print(f"[train] class weights  away(0)={cw[0]:.3f}  looking(1)={cw[1]:.3f}")
    weight_lookup = torch.tensor([cw[0], cw[1]], dtype=torch.float32)
    loss_fn = nn.BCELoss(reduction="none")
    opt = optim.Adam(model.parameters(), lr=a.lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32).view(-1, 1)),
        batch_size=8, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # 4) Train with early stopping on validation loss; keep the BEST epoch's
    # weights (not whatever the last epoch happened to produce).
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    stopped_early = False
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for bx, by in loader:
            pred = model(bx)
            sample_w = weight_lookup[by.long().view(-1)].view(-1, 1)
            loss = (loss_fn(pred, by) * sample_w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / len(loader)

        if have_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = nn.functional.binary_cross_entropy(val_pred, y_val_t).item()
            improved = val_loss < best_val_loss - 1e-5
            if improved:
                best_val_loss, best_epoch, no_improve = val_loss, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
            if (epoch + 1) % 10 == 0 or improved:
                print(f"  epoch {epoch + 1}/{a.epochs}  train_loss {train_loss:.4f}  "
                      f"val_loss {val_loss:.4f}{'  *best*' if improved else ''}")
            if no_improve >= a.patience:
                print(f"[train] early stop at epoch {epoch + 1} "
                      f"(best was epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                stopped_early = True
                break
        else:
            best_epoch = epoch + 1
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1}/{a.epochs}  train_loss {train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)   # roll back to the best validation epoch

    # 5) Choose the decision threshold on VALIDATION (never test), then apply
    # that fixed threshold once to the held-out test set for reporting.
    model.eval()
    if have_val:
        with torch.no_grad():
            val_probs = model(X_val_t).numpy().ravel()
        threshold, val_at_threshold = dataset.best_threshold(y_val, val_probs)
        print(f"\n[train] threshold {threshold:.2f} chosen on validation "
              f"(val F1={val_at_threshold['f1']}  precision={_fmt(val_at_threshold['precision'])}  "
              f"recall={_fmt(val_at_threshold['recall'])})")
    else:
        threshold, val_at_threshold = 0.5, None
        print("\n[train] no validation data — using the default threshold 0.5 (not tuned)")

    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()
    preds = (probs >= threshold).astype(int)

    overall = dataset.classification_metrics(y_test, preds)
    print(f"\n[eval] overall (threshold={threshold:.2f}): acc {overall['accuracy']}%  "
          f"precision {_fmt(overall['precision'])}  recall {_fmt(overall['recall'])}  (n={len(y_test)})")

    report = {
        "data": path,
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "n_train_identities": int(n_train_ids), "n_val_identities": int(n_val_ids),
        "n_test_identities": int(n_test_ids),
        "class_weights": {"away(0)": round(cw[0], 4), "looking(1)": round(cw[1], 4)},
        "standardization": {"features": dataset.FEATURES,
                            "mean": np.round(feat_mean, 6).tolist(),
                            "std": np.round(feat_std, 6).tolist()},
        "training": {"max_epochs": a.epochs, "stopped_at_epoch": best_epoch,
                     "early_stopped": stopped_early, "had_validation_split": have_val},
        "threshold": {"value": round(threshold, 3),
                      "chosen_on": "validation" if have_val else "default (no validation data)",
                      "val_metrics_at_threshold": val_at_threshold},
        "overall": overall, "by": {},
    }
    ev = test_df.copy()
    ev["true"] = y_test
    ev["pred"] = preds
    for dim in ["distance_tier", "glasses", "headwear"]:
        report["by"][dim] = {}
        print(f"[eval] by {dim}:")
        for val, g in ev.groupby(dim):
            m = dataset.classification_metrics(g["true"].to_numpy(), g["pred"].to_numpy())
            report["by"][dim][str(val)] = {**m, "n": int(len(g))}
            print(f"  {str(val):<16} acc {m['accuracy']}%  recall {_fmt(m['recall']):<5} "
                  f"(n={len(g)}, real looking={m['n_pos']})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    metrics_path = Path(a.out).with_name("engagement_metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[train] saved model   -> {a.out}")
    print(f"[train] saved metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
