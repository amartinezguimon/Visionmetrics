"""Tests for EngagementNet/EngagementClassifier — standardization + the
feature-schema safety tag that stops a model trained under the old flat-ratio
yaw/pitch from being silently served predictions computed under the new
solvePnP-degrees pipeline (or vice versa)."""

from __future__ import annotations

import torch

from visionmetrics.edge.agent.classifier import (
    CURRENT_FEATURE_SCHEMA,
    EngagementClassifier,
    EngagementNet,
)


def test_fresh_net_defaults_to_schema_1_legacy():
    # A checkpoint saved before this buffer existed loads via strict=False and
    # must land here — schema 1, NOT silently treated as "current".
    net = EngagementNet()
    assert int(net.feature_schema.item()) == 1


def test_set_feature_schema_updates_the_buffer():
    net = EngagementNet()
    net.set_feature_schema(CURRENT_FEATURE_SCHEMA)
    assert int(net.feature_schema.item()) == CURRENT_FEATURE_SCHEMA


def test_classifier_exposes_loaded_schema(tmp_path):
    net = EngagementNet()
    net.set_feature_schema(2)
    p = tmp_path / "model.pth"
    torch.save(net.state_dict(), p)

    clf = EngagementClassifier.load(p)
    assert clf.feature_schema == 2


def test_classifier_loading_a_legacy_checkpoint_reports_schema_1(tmp_path):
    # Simulate a REAL legacy .pth: no feature_schema key in the state_dict at
    # all (as if saved before this buffer existed), same as `strict=False`
    # already handles for feat_mean/feat_std.
    net = EngagementNet()
    state = net.state_dict()
    del state["feature_schema"]
    p = tmp_path / "legacy_model.pth"
    torch.save(state, p)

    clf = EngagementClassifier.load(p)
    assert clf.feature_schema == 1


def test_probability_still_works_after_setting_schema(tmp_path):
    # The schema tag is metadata only — must never affect the actual forward pass.
    net = EngagementNet()
    net.set_feature_schema(2)
    p = tmp_path / "model.pth"
    torch.save(net.state_dict(), p)
    clf = EngagementClassifier.load(p)
    prob = clf.probability(0.0, 0.0, 0.15)
    assert 0.0 <= prob <= 1.0
