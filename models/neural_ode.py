"""
Continuous-time variant: instead of one discrete MLP step per month, learn
a derivative field dx/dt and integrate it with fixed-step RK4 across
several sub-monthly steps, then apply the ERCP relief as a discrete jump
at the month boundary (ERCP is a real clinical event, not a continuous
process, so modeling it as a jump rather than smearing it into the ODE
is the more faithful choice, not a simplification of convenience).

Constraint guarantee under integration: for every ratchet channel, the
derivative is `softplus(raw) * scale`, i.e. non-negative at every
evaluated point along the RK4 stages. A weighted average (with
non-negative weights summing to 1, which is exactly what RK4's b_i
weights are) of non-negative numbers is non-negative, so the accumulated
increment over the sub-steps stays non-negative -- the by-construction
guarantee survives integration, not just a single discrete step. A final
clamp to bounds is kept anyway as a numerical safety net (RK4 with a
coarse step size could in principle overshoot a bound even with a
correctly-signed derivative), matching the same clamp-as-projection
approach already used in models/constraints.py.

forward() intentionally matches models.baseline.MonotoneStep's signature
exactly -- (x_t, ctx_feat_t, ercp_flag_t) -> x_next -- so it is a drop-in
replacement everywhere the baseline is used (compare.py's
baseline_rollout, eval.py, explain.py) without any harness changes.
"""
import torch
import torch.nn as nn

F, D, S, P, A, C, M, FLARE = range(8)
STEP_SCALE = 0.35
RELAX_RATE = 3.0  # fast-variable relaxation rate for A, C, flare toward their driven target


class ODEFunc(nn.Module):
    RAW_DIM = 8  # [F_rate, D_rate, S_creep_rate, P_rate, A_target, C_target, M_rate, flare_target]

    def __init__(self, ctx_dim=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8 + ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.RAW_DIM),
        )

    def forward(self, x, ctx_feat):
        raw = self.net(torch.cat([x, ctx_feat], dim=-1))
        F_rate = STEP_SCALE * torch.nn.functional.softplus(raw[:, 0])
        D_rate = STEP_SCALE * torch.nn.functional.softplus(raw[:, 1])
        S_creep_rate = STEP_SCALE * torch.nn.functional.softplus(raw[:, 2])
        P_rate = STEP_SCALE * torch.nn.functional.softplus(raw[:, 3])
        A_target = torch.sigmoid(raw[:, 4])
        C_target = torch.sigmoid(raw[:, 5])
        M_rate_raw = STEP_SCALE * torch.nn.functional.softplus(raw[:, 6])
        flare_target = torch.sigmoid(raw[:, 7])

        dA = RELAX_RATE * (A_target - x[:, A])
        dC = RELAX_RATE * (C_target - x[:, C])
        dflare = RELAX_RATE * (flare_target - x[:, FLARE])
        gate = 0.3 + 1.7 * (x[:, F] * x[:, C])
        dM = M_rate_raw * gate

        return torch.stack([F_rate, D_rate, S_creep_rate, P_rate, dA, dC, dM, dflare], dim=1)


class NeuralODEStep(nn.Module):
    """Drop-in replacement for MonotoneStep. Integrates one month with
    fixed-step RK4 (n_substeps sub-steps), then applies the ERCP relief as
    a discrete jump, then clamps to bounds as a numerical safety net."""

    def __init__(self, ctx_dim=8, hidden=64, n_substeps=4):
        super().__init__()
        self.ode_func = ODEFunc(ctx_dim, hidden)
        self.n_substeps = n_substeps
        self.relief_net = nn.Sequential(nn.Linear(8 + ctx_dim, 16), nn.ReLU(), nn.Linear(16, 1))

        lo = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.0])
        hi = torch.tensor([1, 1, 1, 1, 1, 1, 2, 1.0])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def integrate(self, x0, ctx_feat):
        h = 1.0 / self.n_substeps
        x = x0
        for _ in range(self.n_substeps):
            k1 = self.ode_func(x, ctx_feat)
            k2 = self.ode_func(torch.clamp(x + h / 2 * k1, min=0), ctx_feat)
            k3 = self.ode_func(torch.clamp(x + h / 2 * k2, min=0), ctx_feat)
            k4 = self.ode_func(torch.clamp(x + h * k3, min=0), ctx_feat)
            dx = (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            x = x + dx
        return x

    def forward(self, x_t, ctx_feat_t, ercp_flag_t):
        x_next = self.integrate(x_t, ctx_feat_t)
        if ercp_flag_t.dim() == 1:
            ercp_flag_t = ercp_flag_t.unsqueeze(-1)
        relief = STEP_SCALE * torch.nn.functional.softplus(self.relief_net(torch.cat([x_t, ctx_feat_t], dim=-1))) * 2.0
        x_next = x_next.clone()
        x_next[:, S:S+1] = x_next[:, S:S+1] - ercp_flag_t * relief
        x_next = torch.clamp(x_next, self.lo, self.hi)
        return x_next
