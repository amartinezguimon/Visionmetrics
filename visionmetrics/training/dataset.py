"""Dataset schema, merging, coverage and augmentation — the shared core for
collecting, building and training the engagement model.

Why metadata columns when the model only uses (yaw, pitch, distance)?
The classifier never *sees* glasses or a cap. But those conditions — and distance
— change the *distribution and noise* of the three numbers (glasses confuse the
eye/cheekbone landmarks, a cap shifts the forehead/top landmark, far faces are
noisier all round). Recording the condition lets us (a) check we have COVERAGE of
each, and (b) evaluate accuracy PER condition, so we know the model is robust where
it matters — not just on easy, close, bare-faced subjects.

Pure (pandas/numpy) so it is unit-testable without a camera or torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .collect import SESSION_COLUMNS, tier_for

# Canonical column order for any dataset file we read or write.
CANONICAL = SESSION_COLUMNS
# The only columns the model actually trains on.
FEATURES = ["yaw", "pitch", "distance"]
META = ["glasses", "headwear", "subject", "collector", "session", "captured_at"]
_REQUIRED = ["yaw", "pitch", "distance", "label"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any (old 4-column or new rich) frame to the canonical schema.

    Missing metadata becomes "unknown"; a missing/blank distance_tier is derived
    from the distance. Rows missing a feature or label are dropped.
    """
    df = df.copy()
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    for c in META:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str)

    if "distance_tier" not in df.columns:
        df["distance_tier"] = ""
    df["distance_tier"] = df["distance_tier"].fillna("").astype(str)
    blank = df["distance_tier"].isin(["", "unknown", "nan", "None"])
    df.loc[blank, "distance_tier"] = df.loc[blank, "distance"].map(tier_for)

    df = df.dropna(subset=_REQUIRED)
    # Validate the label is strictly binary. A stray value (a typo, a 2, a 0.5, a
    # non-numeric cell from a hand-edited CSV) blindly cast with astype(int) would
    # silently poison BCELoss, so coerce to numeric and DROP anything that isn't
    # exactly {0, 1} rather than trusting the cast.
    label_num = pd.to_numeric(df["label"], errors="coerce")
    valid = label_num.isin([0, 1])
    dropped = int((~valid).sum())
    if dropped:
        bad = sorted(df.loc[~valid, "label"].astype(str).unique())[:5]
        print(f"[dataset] dropped {dropped} row(s) with a non-binary label "
              f"(e.g. {bad}); labels must be 0 or 1.")
    df = df.loc[valid].copy()
    df["label"] = label_num.loc[valid].astype(int)
    return df[CANONICAL].reset_index(drop=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop EXACT duplicate rows (identical features, label AND all metadata).

    Guards against re-ingesting the same data twice — e.g. a session file copied
    under two names, or the legacy flat CSV overlapping rows already in
    raw_sessions/. Only byte-identical rows are removed, so two genuinely distinct
    captures are never merged (they differ in captured_at/session at minimum).
    """
    return df.drop_duplicates().reset_index(drop=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    return normalize(pd.read_csv(path))


def merge(paths: list[str | Path]) -> pd.DataFrame:
    """Read + normalise several CSVs and concatenate them into one dataset."""
    frames = [load_csv(p) for p in paths if Path(p).exists()]
    if not frames:
        return pd.DataFrame(columns=CANONICAL)
    return pd.concat(frames, ignore_index=True)


def augment_far(X: np.ndarray, y: np.ndarray, *, scales=(0.6, 0.35, 0.15),
                noise_std: float = 0.005, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise far-distance samples by scaling the distance column.

    Yaw/pitch are scale-invariant (a straight-ahead look reads ~0 at any range),
    so scaling only the distance proxy (with a touch of noise) generates realistic
    far examples. Apply to TRAIN rows ONLY — never the held-out test set.
    """
    rng = np.random.default_rng(seed)
    ax, ay = [], []
    for (yaw, pitch, dist), label in zip(X, y):
        for s in scales:
            n = rng.normal(0, noise_std, 2)
            ax.append([yaw + n[0], pitch + n[1], dist * s])
            ay.append(label)
    return np.asarray(ax, dtype=float), np.asarray(ay)


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy (%), precision and recall for the positive ("looking") class.

    Pure numpy; denominators guarded so a single-class slice never divides by zero.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    n = len(yt)
    acc = float((yt == yp).mean() * 100) if n else 0.0
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"accuracy": round(acc, 1), "precision": round(precision, 3),
            "recall": round(recall, 3)}


def coverage_text(df: pd.DataFrame, *, thin: int = 40) -> str:
    """Human-readable coverage report: where you have data and where you're thin.

    Cells with fewer than `thin` samples are flagged — that's where to collect more.
    """
    if df.empty:
        return "Dataset is empty."
    lines = [f"Total samples: {len(df)}",
             f"  looking(1): {(df.label == 1).sum()}    away(0): {(df.label == 0).sum()}"]
    warnings: list[str] = []
    for dim in ["distance_tier", "glasses", "headwear"]:
        lines.append(f"\nBy {dim}:")
        ct = (df.groupby([dim, "label"]).size().unstack(fill_value=0)
              .reindex(columns=[0, 1], fill_value=0))
        for value, row in ct.iterrows():
            away, look = int(row.get(0, 0)), int(row.get(1, 0))
            lines.append(f"  {value:<16} looking={look:<5} away={away:<5} total={look + away}")
            if look < thin or away < thin:
                warnings.append(f"{dim}={value} (looking={look}, away={away})")
    if warnings:
        lines.append("\nThin coverage (collect more here):")
        lines.extend(f"  - {w}" for w in warnings)
    return "\n".join(lines)
