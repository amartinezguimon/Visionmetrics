"""PyTorch engagement classifier — network definition + loading + scoring.

The architecture MUST stay in lock-step with ml/train.py (3 -> 16 -> 8 -> 1,
Sigmoid). It is defined here once and imported by both the agent and training,
so they can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class EngagementNet(nn.Module):
    """MLP: (yaw, pitch, face_width_norm) -> P(engaged) in [0, 1].

    Inputs are z-scored (``(x - mean) / std``) before the MLP. The mean/std are
    registered BUFFERS, so they are saved and loaded inside the ``state_dict``
    (the .pth) — the exact same standardization travels with the weights and is
    applied identically in training and in the live agent, with no separate file
    to keep in sync (that would risk train/serve skew). Defaults are the identity
    transform (mean 0, std 1), so an unfitted or legacy model behaves as if fed
    raw inputs.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(3))
        self.register_buffer("feat_std", torch.ones(3))
        self.network = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def set_standardization(self, mean, std) -> None:
        """Fit the input scaler from TRAIN stats. Zero-variance columns fall back
        to std=1 so a constant feature never divides by zero."""
        m = torch.as_tensor(mean, dtype=torch.float32).reshape(3)
        s = torch.as_tensor(std, dtype=torch.float32).reshape(3)
        s = torch.where(s < 1e-6, torch.ones_like(s), s)
        self.feat_mean.copy_(m)
        self.feat_std.copy_(s)

    def forward(self, x):
        x = (x - self.feat_mean) / self.feat_std
        return self.network(x)


class EngagementClassifier:
    """Loads trained weights and scores head-pose feature triples."""

    def __init__(self, model: EngagementNet):
        self._model = model
        self._model.eval()

    @classmethod
    def load(cls, weights_path: str | Path) -> "EngagementClassifier":
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Engagement model not found: {weights_path}. Train it with ml/train.py."
            )
        net = EngagementNet()
        # strict=False: a legacy .pth trained before standardization has no
        # feat_mean/feat_std buffers; they stay at the identity defaults so the
        # old model keeps behaving exactly as raw-input. New models carry them.
        net.load_state_dict(torch.load(weights_path, weights_only=True), strict=False)
        return cls(net)

    def probability(self, yaw: float, pitch: float, distance: float) -> float:
        """Return P(engaged) for one feature triple."""
        with torch.no_grad():
            x = torch.tensor([[yaw, pitch, distance]], dtype=torch.float32)
            return float(self._model(x).item())
