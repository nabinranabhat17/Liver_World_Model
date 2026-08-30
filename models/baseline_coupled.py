"""
MonotoneStepCoupled: the structural (hard-guarantee) counterpart to
scripts/train_baseline_jacobian.py's soft double-backward penalty (see
DECISIONS.md D9). D9 found d(F_next)/d(A_prev) and d(F_next)/d(C_prev)
wrong-signed on 86-93% of samples despite the generator's true dF being
non-negative in A, C, and fixed it with a training-time hinge loss --
100% sign-correct, but ~5.3% accuracy cost from fighting the primary
objective.

This variant removes the degree of freedom that caused the wrong sign in
the first place, instead of penalising it: F_inc's and D_inc's raw
scores are each split into a "free" component that has NO access at all
to their coupled driver field(s), plus a `MonotonicCoupling` (see
models/monotonic.py) that is provably non-decreasing in those drivers
for any trained weights. The same treatment is applied to D's sensitivity
to S, A (dD ~ 0.7*S + 0.3*A in generator.py) -- structurally identical to
F's case, and not previously measured.

No training-time penalty is needed: the guarantee holds under random,
UNTRAINED weights (see scripts/test_invariants.py's
test_coupled_monotonicity_random_weights). Same forward() signature as
MonotoneStep, so it is a drop-in for every existing rollout helper.
"""
import torch
import torch.nn as nn
from models.constraints import ConstraintHead
from models.monotonic import MonotonicCoupling

F_IDX, D_IDX, S_IDX, P_IDX, A_IDX, C_IDX, M_IDX, FLARE_IDX = range(8)


class MonotoneStepCoupled(nn.Module):
    TRUNK_RAW_DIM = 7  # S_creep, S_relief_raw, P_inc, A_logit, C_logit, M_inc, flare_logit

    def __init__(self, ctx_dim=8, hidden=64, free_head_hidden=32, coupling_hidden=16):
        super().__init__()
        full_dim = 8 + ctx_dim  # x_t (8) + ctx_feat_t (ctx_dim)

        self.trunk = nn.Sequential(
            nn.Linear(full_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.TRUNK_RAW_DIM),
        )

        # F head: excludes A_prev, C_prev from its free input; sees them
        # only through the monotone coupling term.
        self.F_excl = [A_IDX, C_IDX]
        self.register_buffer("_F_keep", self._keep_idx(full_dim, self.F_excl))
        self.F_free = nn.Sequential(
            nn.Linear(full_dim - len(self.F_excl), free_head_hidden), nn.ReLU(),
            nn.Linear(free_head_hidden, 1),
        )
        self.F_coupling = MonotonicCoupling(n_drivers=2, hidden=coupling_hidden)

        # D head: excludes S_prev, A_prev likewise. disease_class (ctx
        # column 0) is wired in as free_feat since the generator scales
        # dD's magnitude by is_psc -- this can't break the guarantee
        # (see models/monotonic.py) and gives back a real capacity gap
        # the old monolithic trunk had access to.
        self.D_excl = [S_IDX, A_IDX]
        self.register_buffer("_D_keep", self._keep_idx(full_dim, self.D_excl))
        self.D_free = nn.Sequential(
            nn.Linear(full_dim - len(self.D_excl), free_head_hidden), nn.ReLU(),
            nn.Linear(free_head_hidden, 1),
        )
        self.D_coupling = MonotonicCoupling(n_drivers=2, n_free=1, hidden=coupling_hidden)

        self.constraint_head = ConstraintHead()

    @staticmethod
    def _keep_idx(full_dim, exclude_x_idx, x_dim=8):
        keep = [i for i in range(x_dim) if i not in exclude_x_idx] + list(range(x_dim, full_dim))
        return torch.tensor(keep, dtype=torch.long)

    def forward(self, x_t, ctx_feat_t, ercp_flag_t):
        full = torch.cat([x_t, ctx_feat_t], dim=1)  # (B, 8+ctx_dim)

        trunk_out = self.trunk(full)  # (B, 7)

        F_restricted = full.index_select(1, self._F_keep)
        F_drivers = x_t[:, self.F_excl]  # (B, 2) = [A_prev, C_prev]
        F_inc_raw = self.F_free(F_restricted) + self.F_coupling(F_drivers)

        D_restricted = full.index_select(1, self._D_keep)
        D_drivers = x_t[:, self.D_excl]  # (B, 2) = [S_prev, A_prev]
        disease_class = ctx_feat_t[:, 0:1]
        D_inc_raw = self.D_free(D_restricted) + self.D_coupling(D_drivers, disease_class)

        raw = torch.cat([
            F_inc_raw, D_inc_raw,          # 0 F_inc, 1 D_inc
            trunk_out[:, 0:1],             # 2 S_creep
            trunk_out[:, 1:2],             # 3 S_relief_raw
            trunk_out[:, 2:3],             # 4 P_inc
            trunk_out[:, 3:4],             # 5 A_logit
            trunk_out[:, 4:5],             # 6 C_logit
            trunk_out[:, 5:6],             # 7 M_inc
            trunk_out[:, 6:7],             # 8 flare_logit
        ], dim=1)

        return self.constraint_head(raw, x_t, ercp_flag_t)

    def rollout(self, x0, ctx_all, ercp_all, T):
        """Free rollout (no teacher forcing) from x0 for T steps. Identical
        to MonotoneStep.rollout."""
        B = x0.shape[0]
        xs = [x0]
        x = x0
        for t in range(1, T):
            x = self.forward(x, ctx_all[:, t, :], ercp_all[:, t])
            xs.append(x)
        return torch.stack(xs, dim=1)
