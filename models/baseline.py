"""
MonotoneStep: the baseline model. Memoryless -- x(t) itself is the latent
(role 3 in the assignment's framing: "an auditable record of what the model
believes about this patient right now"). No history, no learned
representation to collapse or go off-manifold. This is the natural x(t)-as-
latent peer the memo weighs the JEPA against.

predict(x_t, context_features_t, ercp_flag_t) -> x_{t+1}
via a small MLP producing raw ConstraintHead inputs.
"""
import torch
import torch.nn as nn
from models.constraints import ConstraintHead


class MonotoneStep(nn.Module):
    def __init__(self, ctx_dim=8, hidden=64):
        super().__init__()
        in_dim = 8 + ctx_dim  # x_t + context features (includes on-treatment flag, time)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ConstraintHead.RAW_DIM),
        )
        self.constraint_head = ConstraintHead()

    def forward(self, x_t, ctx_feat_t, ercp_flag_t):
        inp = torch.cat([x_t, ctx_feat_t], dim=1)
        raw = self.net(inp)
        return self.constraint_head(raw, x_t, ercp_flag_t)

    def rollout(self, x0, ctx_all, ercp_all, T):
        """Free rollout (no teacher forcing) from x0 for T steps.
        ctx_all: (B, T, ctx_dim) precomputed per-step context features
        ercp_all: (B, T) ERCP flags
        Returns (B, T, 8) including x0 at t=0."""
        B = x0.shape[0]
        xs = [x0]
        x = x0
        for t in range(1, T):
            x = self.forward(x, ctx_all[:, t, :], ercp_all[:, t])
            xs.append(x)
        return torch.stack(xs, dim=1)
