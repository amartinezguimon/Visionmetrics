"""Tests for the training dataset core (schema, merge, augmentation, metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from visionmetrics.training import dataset


def _legacy_df():
    return pd.DataFrame({
        "yaw": [0.0, 0.1], "pitch": [0.0, -0.1],
        "distance": [0.30, 0.06], "label": [1, 0],
    })


def test_normalize_adds_metadata_and_derives_tier():
    out = dataset.normalize(_legacy_df())
    assert list(out.columns) == dataset.CANONICAL
    assert (out["glasses"] == "unknown").all()
    assert (out["collector"] == "unknown").all()
    assert out.loc[0, "distance_tier"] == "near <0.5m"
    assert out.loc[1, "distance_tier"] == "far 1.5-3.5m"


def test_normalize_keeps_existing_metadata():
    df = _legacy_df()
    df["glasses"] = "yes"
    df["headwear"] = "cap"
    out = dataset.normalize(df)
    assert (out["glasses"] == "yes").all() and (out["headwear"] == "cap").all()


def test_normalize_requires_feature_columns():
    bad = pd.DataFrame({"yaw": [0.0], "pitch": [0.0]})   # no distance/label
    try:
        dataset.normalize(bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_normalize_drops_non_binary_labels():
    df = pd.DataFrame({
        "yaw": [0.0, 0.1, 0.2, 0.3, 0.4],
        "pitch": [0.0, 0.0, 0.0, 0.0, 0.0],
        "distance": [0.3, 0.3, 0.3, 0.3, 0.3],
        "label": [1, 0, 2, 0.5, "x"],   # 2, 0.5 and "x" are invalid
    })
    out = dataset.normalize(df)
    assert len(out) == 2                       # only the 1 and 0 survive
    assert sorted(out["label"].tolist()) == [0, 1]
    assert out["label"].dtype.kind == "i"      # stays integer


def test_normalize_accepts_float_binary_labels():
    df = _legacy_df()
    df["label"] = [1.0, 0.0]                    # floats that ARE 0/1
    out = dataset.normalize(df)
    assert sorted(out["label"].tolist()) == [0, 1]


def test_merge_concatenates(tmp_path):
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _legacy_df().to_csv(a, index=False)
    _legacy_df().to_csv(b, index=False)
    merged = dataset.merge([a, b, tmp_path / "missing.csv"])
    assert len(merged) == 4
    assert list(merged.columns) == dataset.CANONICAL


def test_dedupe_drops_exact_duplicates_only():
    df = dataset.normalize(pd.DataFrame({
        "yaw": [0.1, 0.1, 0.1], "pitch": [0.0, 0.0, 0.0],
        "distance": [0.3, 0.3, 0.3], "label": [1, 1, 0],
    }))
    out = dataset.dedupe(df)
    assert len(out) == 2               # the two identical looking rows collapse to 1
    assert sorted(out["label"].tolist()) == [0, 1]


def test_augment_far_scales_distance_only():
    X = np.array([[0.1, 0.2, 0.20]])
    y = np.array([1])
    ax, ay = dataset.augment_far(X, y, scales=(0.5,), noise_std=0.0)
    assert ax.shape == (1, 3) and ay.tolist() == [1]
    assert abs(ax[0, 2] - 0.10) < 1e-9          # distance halved
    assert abs(ax[0, 0] - 0.1) < 1e-9           # yaw unchanged (no noise)


def test_classification_metrics():
    m = dataset.classification_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert m["accuracy"] == 75.0
    assert m["precision"] == 1.0      # 1 TP, 0 FP
    assert m["recall"] == 0.5         # 1 TP, 1 FN
    assert m["n_pos"] == 2 and m["n_neg"] == 2


def test_classification_metrics_recall_is_none_without_support():
    # No REAL positive examples at all (e.g. the "far" tier today) — recall
    # must read as "not applicable", not a misleading 0.0.
    m = dataset.classification_metrics([0, 0, 0], [1, 0, 0])
    assert m["recall"] is None
    assert m["n_pos"] == 0


def test_classification_metrics_precision_is_none_without_positive_predictions():
    m = dataset.classification_metrics([1, 0], [0, 0])   # model never predicts "looking"
    assert m["precision"] is None
    assert m["recall"] == 0.0   # this one IS a real, meaningful 0 — it had a chance and missed


def test_class_weights_favours_the_rare_class():
    w = dataset.class_weights([1] * 90 + [0] * 10)   # 90 looking, 10 away
    assert w[0] > w[1]              # the rare class (away=0) gets the bigger weight
    assert w[0] == 100 / (2 * 10)
    assert w[1] == 100 / (2 * 90)


def test_class_weights_missing_class_is_neutral():
    w = dataset.class_weights([1, 1, 1])   # no 0s at all in this slice
    assert w[0] == 1.0


def test_best_threshold_beats_naive_half_on_skewed_probs():
    # A model that outputs LOW probabilities for real positives (badly
    # calibrated, but rank-ordered correctly) — 0.5 misses almost everyone;
    # a lower threshold recovers them without inventing false positives.
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    probs = [0.30, 0.32, 0.28, 0.31, 0.05, 0.06, 0.04, 0.07]
    t, m = dataset.best_threshold(y, probs)
    assert t < 0.5
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_best_threshold_returns_valid_metrics_dict():
    t, m = dataset.best_threshold([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert 0.0 < t < 1.0
    assert "f1" in m and "recall" in m and "precision" in m


def _person_df(subject, session, n, *, label=1):
    return pd.DataFrame({
        "yaw": np.linspace(0.0, 0.05, n), "pitch": [0.0] * n, "distance": [0.3] * n,
        "label": [label] * n, "subject": [subject] * n, "session": [session] * n,
    })


def test_group_key_prefers_real_subject():
    df = dataset.normalize(_person_df("hector", "sess-1", 3))
    keys = dataset.group_key(df)
    assert (keys == "subject:hector").all()


def test_group_key_falls_back_to_session_when_subject_unknown():
    df = pd.DataFrame({
        "yaw": [0.0, 0.1], "pitch": [0.0, 0.0], "distance": [0.3, 0.3], "label": [1, 0],
        "session": ["live_20260101-000000", "live_20260101-000000"],
    })
    df = dataset.normalize(df)   # subject column defaults to "unknown"
    keys = dataset.group_key(df)
    assert (keys == "session:live_20260101-000000").all()
    assert "unknown" not in keys.iloc[0]


def test_group_key_never_bare_unknown_literal():
    df = dataset.normalize(pd.DataFrame({
        "yaw": [0.0], "pitch": [0.0], "distance": [0.3], "label": [1],
    }))
    assert dataset.group_key(df).iloc[0] != "unknown"


def test_group_train_test_split_keeps_each_subject_on_one_side():
    # 20 distinct subjects, 10 rows each, mixed labels — enough groups for the
    # stratified group split to actually run (not the small-N fallback path).
    frames = []
    for i in range(20):
        n_look = 5 if i % 2 == 0 else 3
        frames.append(_person_df(f"person{i}", f"sess{i}", n_look, label=1))
        frames.append(_person_df(f"person{i}", f"sess{i}", 10 - n_look, label=0))
    df = dataset.normalize(pd.concat(frames, ignore_index=True))

    train_df, test_df = dataset.group_train_test_split(df, test_size=0.25, seed=7)

    train_subjects = set(dataset.group_key(train_df))
    test_subjects = set(dataset.group_key(test_df))
    assert train_subjects.isdisjoint(test_subjects), "same person appears on both sides"
    assert len(train_df) + len(test_df) == len(df)


def test_group_train_test_split_falls_back_when_too_few_groups():
    # Only 2 distinct people — too few for StratifiedGroupKFold at test_size=0.2
    # (n_splits=5 > n_groups=2); must not raise, must still not leak.
    df = dataset.normalize(pd.concat([
        _person_df("a", "sess-a", 6, label=1), _person_df("a", "sess-a", 6, label=0),
        _person_df("b", "sess-b", 6, label=1), _person_df("b", "sess-b", 6, label=0),
    ], ignore_index=True))

    train_df, test_df = dataset.group_train_test_split(df, test_size=0.2, seed=1)

    assert set(dataset.group_key(train_df)).isdisjoint(set(dataset.group_key(test_df)))
    assert len(train_df) + len(test_df) == len(df)


def test_coverage_text_flags_thin():
    df = dataset.normalize(pd.DataFrame({
        "yaw": [0.0] * 5, "pitch": [0.0] * 5, "distance": [0.3] * 5,
        "label": [1, 1, 1, 1, 0],
    }))
    text = dataset.coverage_text(df, thin=3)
    assert "Total samples: 5" in text
    assert "Thin coverage" in text   # away count (1) is below thin=3
    assert "NO REAL DATA" not in text   # 1 real away example exists — not zero


def test_coverage_text_flags_zero_separately_from_thin():
    # far tier (small normalised face width, 0.04-0.10): 0 real "looking" rows
    # at all (today's actual situation) — must land under the ZERO header, not
    # lumped in with merely-thin cells.
    df = dataset.normalize(pd.DataFrame({
        "yaw": [0.0] * 6, "pitch": [0.0] * 6,
        "distance": [0.06] * 5 + [0.30] * 1,   # 5 far (0.06), 1 near (0.30)
        "label": [0, 0, 0, 0, 0, 1],
    }))
    text = dataset.coverage_text(df, thin=40)
    assert "NO REAL DATA" in text
    assert "ZERO real 'looking'" in text
    assert "synthetic/augmented" in text   # distance_tier zero-looking gets the augment_far note
