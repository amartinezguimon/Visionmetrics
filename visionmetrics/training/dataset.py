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


_UNKNOWN_TOKENS = {"", "unknown", "nan", "none"}


def group_key(df: pd.DataFrame) -> pd.Series:
    """One identity per row for a leak-free train/test split: the real
    ``subject`` when it was actually recorded, otherwise the ``session`` they
    were collected in (a session is one sitting in front of one camera — almost
    always one real person — even on rows where nobody typed a name in).

    Why this matters: a plain row-level split lets different frames of the SAME
    person land in both train and test. The model then has 3 numbers (yaw,
    pitch, distance) that repeat with that person's own idiosyncrasies across
    both piles, so it can partly "recognise" them rather than learning gaze in
    general — inflating every reported metric. Grouping by identity keeps a
    person's rows entirely on one side.

    Never returns the bare literal "unknown" — that would silently collapse
    every subject-less row onto one fake shared "person" and defeat the whole
    point. Rows with neither a real subject nor a real session (the oldest,
    barest legacy data) do still collapse together under one group; there is no
    metadata left to tell those individuals apart, which is a genuine ceiling
    of that data, not a bug in this function.
    """
    def _clean(col: str, prefix: str) -> pd.Series:
        s = df[col].astype(str).str.strip()
        known = ~s.str.lower().isin(_UNKNOWN_TOKENS)
        return known, prefix + s

    subj_known, subj_val = _clean("subject", "subject:")
    sess_known, sess_val = _clean("session", "session:")
    fallback = sess_val.where(sess_known, "legacy:no-metadata")
    return subj_val.where(subj_known, fallback)


def group_train_test_split(
    df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split with NO shared identity (see `group_key`) between the
    two sides, keeping the label balance close to `test_size` when there are
    enough distinct identities to do so.

    Uses ``StratifiedGroupKFold`` (grouped AND label-aware) when there are
    enough distinct groups for it to run; a dataset too small/thin for that
    (say, a handful of people) falls back to a plain ``GroupShuffleSplit`` —
    still zero leakage, just without the label-balance guarantee — rather than
    raising on a small real-world dataset.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    labels = df["label"].to_numpy()
    groups = group_key(df).to_numpy()
    n_groups = len(set(groups))
    n_splits = max(2, round(1.0 / test_size))

    if n_groups < n_splits:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, labels, groups))
    else:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(skf.split(df, labels, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


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

    `precision`/`recall` are ``None`` — NOT a misleading ``0.0`` — when there is
    no support for the relevant class in this slice. E.g. the "far" distance
    tier currently has zero REAL "looking" test rows: a bare 0.0 there would
    read as "the model always misses far lookers", when the honest answer is
    "we have never actually tested that, there is nothing to measure yet".
    `n_pos`/`n_neg` are always included so a caller can tell the two cases
    apart (a real 0% recall vs. no data) without re-deriving it.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    n = len(yt)
    acc = float((yt == yp).mean() * 100) if n else 0.0
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    n_pos = int((yt == 1).sum())
    n_neg = int((yt == 0).sum())
    precision = round(tp / (tp + fp), 3) if (tp + fp) else None
    recall = round(tp / (tp + fn), 3) if (tp + fn) else None
    return {"accuracy": round(acc, 1), "precision": precision, "recall": recall,
            "n_pos": n_pos, "n_neg": n_neg}


def class_weights(y) -> dict[int, float]:
    """Balanced per-class loss weight: ``n_samples / (n_classes * class_count)``
    (the same formula as sklearn's ``class_weight='balanced'``). The rarer
    class gets a bigger weight, so a loss averaged over samples can't win just
    by favouring whichever class happens to be more numerous. A class with 0
    samples gets weight 1.0 (nothing to reweight, and it avoids a div-by-zero).
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    counts = {0: int((y == 0).sum()), 1: int((y == 1).sum())}
    return {c: (n / (2.0 * cnt)) if cnt > 0 else 1.0 for c, cnt in counts.items()}


def best_threshold(y_true, probs, *, grid: np.ndarray | None = None) -> tuple[float, dict]:
    """Sweep decision thresholds and return the one maximising F1 on the
    positive ("looking") class, plus the metrics at that threshold.

    MUST be called on a validation set, never on the held-out test set —
    tuning the threshold against test data leaks test information into a
    number then reported as "test performance", the same leakage class as the
    train/test split itself, just one level up (a hyperparameter, not weights).
    """
    yt = np.asarray(y_true).astype(int)
    p = np.asarray(probs).astype(float)
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_t, best_f1, best_m = 0.5, -1.0, classification_metrics(yt, (p >= 0.5).astype(int))
    for t in grid:
        m = classification_metrics(yt, (p >= t).astype(int))
        prec, rec = m["precision"], m["recall"]
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec and (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_t, best_f1, best_m = float(t), f1, m
    return best_t, {**best_m, "f1": round(best_f1, 3)}


def coverage_text(df: pd.DataFrame, *, thin: int = 40) -> str:
    """Human-readable coverage report: where you have data and where you're thin.

    Two severities, kept apart on purpose:
      * ZERO  — no real example of that class at all in that slice. For
        "distance_tier", this also means every "looking" row the model trains
        on there is 100% synthetic (see `augment_far`) — there is nothing real
        behind that number yet, which a generic "thin" warning would bury.
      * thin  — some real examples, just fewer than `thin`. Collect more, but
        it isn't fabricated-from-nothing the way a zero slice is.
    """
    if df.empty:
        return "Dataset is empty."
    lines = [f"Total samples: {len(df)}",
             f"  looking(1): {(df.label == 1).sum()}    away(0): {(df.label == 0).sum()}"]
    zero: list[str] = []
    thin_warnings: list[str] = []
    for dim in ["distance_tier", "glasses", "headwear"]:
        lines.append(f"\nBy {dim}:")
        ct = (df.groupby([dim, "label"]).size().unstack(fill_value=0)
              .reindex(columns=[0, 1], fill_value=0))
        for value, row in ct.iterrows():
            away, look = int(row.get(0, 0)), int(row.get(1, 0))
            lines.append(f"  {value:<16} looking={look:<5} away={away:<5} total={look + away}")
            if look == 0 or away == 0:
                missing = "looking" if look == 0 else "away"
                note = " (trained ONLY on synthetic/augmented rows there, if any)" \
                    if dim == "distance_tier" and missing == "looking" else ""
                zero.append(f"{dim}={value}: ZERO real '{missing}' examples{note}")
            elif look < thin or away < thin:
                thin_warnings.append(f"{dim}={value} (looking={look}, away={away})")
    if zero:
        lines.append("\nNO REAL DATA (collect this before trusting that number):")
        lines.extend(f"  - {w}" for w in zero)
    if thin_warnings:
        lines.append("\nThin coverage (collect more here):")
        lines.extend(f"  - {w}" for w in thin_warnings)
    return "\n".join(lines)
