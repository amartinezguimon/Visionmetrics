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


def test_coverage_text_flags_thin():
    df = dataset.normalize(pd.DataFrame({
        "yaw": [0.0] * 5, "pitch": [0.0] * 5, "distance": [0.3] * 5,
        "label": [1, 1, 1, 1, 0],
    }))
    text = dataset.coverage_text(df, thin=3)
    assert "Total samples: 5" in text
    assert "Thin coverage" in text   # away count (1) is below thin=3
