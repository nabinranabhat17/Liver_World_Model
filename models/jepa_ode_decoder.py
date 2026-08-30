"""
LatentODEDecoder: drop-in replacement for models.jepa.LatentDecoder that
integrates each decoded month with fixed-step RK4 (4 sub-steps) instead of
taking one discrete step, mirroring models/neural_ode.py's argument for
the baseline -- folded here into TS-JEPA's decode-anchor chain instead.

Same interface as LatentDecoder: forward(z, x_prev, ercp_flag) -> x_next,
decode_chain(z_list, x_anchor, ercp_seq) -> (B,k,8). The one structural
difference from LatentDecoder: instead of raw = net(z) feeding a single
ConstraintHead step, the derivative field is a function of BOTH the
current sub-step state x AND the (fixed, per-month) latent z, so unlike
plain LatentDecoder -- whose raw output depends only on z, never on
x_prev -- this decoder's derivative can depend on x_prev through every
RK4 stage, closer to how the baseline's Neural-ODE variant works.

Ratchet derivatives are softplus(raw)*scale at every RK4 stage, so (per
the same argument as models/neural_ode.py) the accumulated increment over
the sub-steps stays non-negative -- the monotonicity guarantee survives
integration here too. ERCP relief is applied as a discrete jump at the
month boundary, gated on z (the closest available per-month signal to
"context"), not folded into the ODE, for the same reason as the baseline
variant: it's a real clinical event, not a continuous process.
"""
import torch
import torch.nn as nn

F, D, S, P, A, C, M, FLARE = range(8)
STEP_SCALE = 0.35
RELAX_RATE = 3.0


class LatentODEFunc(nn.Module):
    RAW_DIM = 8  # [F_rate, D_rate, S_creep_rate, P_rate, A_target, C_target, M_rate, flare_target]

    def __init__(self, latent_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8 + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.RAW_DIM),
        )

    def forward(self, x, z):
        raw = self.net(torch.cat([x, z], dim=-1))
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


class LatentODEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden=32, n_substeps=4):
        super().__init__()
        self.ode_func = LatentODEFunc(latent_dim, hidden)
        self.n_substeps = n_substeps
        self.relief_net = nn.Sequential(nn.Linear(8 + latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))

        lo = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.0])
        hi = torch.tensor([1, 1, 1, 1, 1, 1, 2, 1.0])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def integrate(self, x0, z):
        h = 1.0 / self.n_substeps
        x = x0
        for _ in range(self.n_substeps):
            k1 = self.ode_func(x, z)
            k2 = self.ode_func(torch.clamp(x + h / 2 * k1, min=0), z)
            k3 = self.ode_func(torch.clamp(x + h / 2 * k2, min=0), z)
            k4 = self.ode_func(torch.clamp(x + h * k3, min=0), z)
            dx = (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            x = x + dx
        return x

    def forward(self, z, x_prev, ercp_flag):
        x_next = self.integrate(x_prev, z)
        if ercp_flag.dim() == 1:
            ercp_flag = ercp_flag.unsqueeze(-1)
        relief = STEP_SCALE * torch.nn.functional.softplus(
            self.relief_net(torch.cat([x_prev, z], dim=-1))) * 2.0
        x_next = x_next.clone()
        x_next[:, S:S + 1] = x_next[:, S:S + 1] - ercp_flag * relief
        x_next = torch.clamp(x_next, self.lo, self.hi)
        return x_next

    def decode_chain(self, z_list, x_anchor, ercp_seq):
        x = x_anchor
        xs = []
        for s, z in enumerate(z_list):
            x = self.forward(z, x, ercp_seq[:, s])
            xs.append(x)
        return torch.stack(xs, dim=1)
