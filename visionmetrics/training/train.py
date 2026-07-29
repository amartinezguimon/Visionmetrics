"""Train the engagement classifier on the master dataset, with honest per-condition eval.

    python -m visionmetrics.training.train
    python -m visionmetrics.training.train --data data/engagement_dataset.csv --epochs 50

Key choices:
* Reuses the SAME network the agent runs (`edge.agent.classifier.EngagementNet`),
  so training and serving can never drift apart.
* Real rows are split into train/test BEFORE far-distance augmentation, so the
  same pose can't leak into both and inflate accuracy.
* Reports overall accuracy/precision/recall AND a breakdown by distance tier and
  by condition (glasses / headwear) — that's how you tell whether new data made the
  model better *where it was weak*, not just on easy close-up faces. Writes the
  report to models/engagement_metrics.json next to the weights.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the engagement model.")
    ap.add_argument("--data", default="data/engagement_dataset.csv")
    ap.add_argument("--out", default="models/engagement_model.pth")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    from . import dataset
    from ..edge.agent.classifier import EngagementNet

    path = _resolve_data(a.data)
    df = dataset.load_csv(path)
    print(f"[train] {len(df)} samples from {path}")
    print(dataset.coverage_text(df))

    # Split the DataFrame (not just arrays) so the test rows keep their condition
    # metadata for the per-condition breakdown below.
    train_df, test_df = train_test_split(
        df, test_size=a.test_size, random_state=a.seed, stratify=df["label"])

    X_tr = train_df[dataset.FEATURES].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy(dtype=int)
    aug_X, aug_y = dataset.augment_far(X_tr, y_tr, seed=a.seed)
    X_train = np.vstack([X_tr, aug_X])
    y_train = np.concatenate([y_tr, aug_y])
    X_test = test_df[dataset.FEATURES].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)
    print(f"[train] train rows {len(X_train)} (real {len(X_tr)} + aug {len(aug_X)})  "
          f"| test (real only) {len(X_test)}")

    torch.manual_seed(a.seed)
    model = EngagementNet()
    # Fit the input z-score on the TRAIN set only (never the test set) and store it
    # INSIDE the model (buffers -> saved in the .pth). forward() then standardizes
    # automatically, so train, eval and the live agent all apply the identical
    # transform to raw (yaw, pitch, distance) — no train/serve skew, no side file.
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    model.set_standardization(feat_mean, feat_std)
    print(f"[train] input z-score  mean={np.round(feat_mean, 4).tolist()}  "
          f"std={np.round(feat_std, 4).tolist()}")
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=a.lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32).view(-1, 1)),
        batch_size=8, shuffle=True)

    model.train()
    for epoch in range(a.epochs):
        total = 0.0
        for bx, by in loader:
            pred = model(bx)
            loss = loss_fn(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{a.epochs}  loss {total / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()
    preds = (probs >= 0.5).astype(int)

    overall = dataset.classification_metrics(y_test, preds)
    print(f"\n[eval] overall: acc {overall['accuracy']}%  "
          f"precision {overall['precision']}  recall {overall['recall']}  (n={len(y_test)})")

    report = {"data": path, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
              "standardization": {"features": dataset.FEATURES,
                                  "mean": np.round(feat_mean, 6).tolist(),
                                  "std": np.round(feat_std, 6).tolist()},
              "overall": overall, "by": {}}
    ev = test_df.copy()
    ev["true"] = y_test
    ev["pred"] = preds
    for dim in ["distance_tier", "glasses", "headwear"]:
        report["by"][dim] = {}
        print(f"[eval] by {dim}:")
        for val, g in ev.groupby(dim):
            m = dataset.classification_metrics(g["true"].to_numpy(), g["pred"].to_numpy())
            report["by"][dim][str(val)] = {**m, "n": int(len(g))}
            print(f"  {str(val):<16} acc {m['accuracy']}%  (n={len(g)})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    metrics_path = Path(a.out).with_name("engagement_metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[train] saved model   -> {a.out}")
    print(f"[train] saved metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
