"""
The by-construction constraint head.

Given a raw vector of network pre-activations `raw` (R^8) and the previous
state x_prev (R^8), produce the next state x_next such that:

  - F, D, P are non-decreasing: implemented as x_prev + softplus(raw) * step_scale,
    then clamped to [0,1]. A decrease is UNREPRESENTABLE (softplus >= 0 always),
    not merely discouraged -- this is what makes the violation-rate guarantee a
    property of the parameterisation rather than of training.
  - S is non-decreasing EXCEPT it may drop at an ERCP event: the creep term uses
    the same softplus construction as F/D/P, but an ERCP flag (fed in as part of
    the action/context) gates in a *separate*, explicitly signed relief term.
    Without the flag, S can only creep up. This mirrors the generator's own
    "creep unless ERCP" structure -- see DECISIONS.md D-constraints.
  - A, C, flare are fast/bounded, not monotone: sigmoid to [0,1], no ratchet.
  - M is a non-decreasing hazard accumulator, [0,2]: same softplus-increment
    trick as F/D/P, but its increment is additionally GATED by prev F*C (the
    coupling from the spec: "M accumulates as a hazard of sustained F*C").
    This is the one channel where getting the per-field guarantee right is not
    enough on its own -- see the coupling note below.

Coupling note (the "interesting part" per the assignment):
  Per-field monotonicity alone does not capture the *interactions* the spec
  calls out (M<-F*C, flares hitting A and C together, treatment suppressing
  A/C). This constraint head enforces the hard one-directional guarantees
  per-field, and layers the coupling ON TOP as a soft architectural bias (the
  F*C gate on M's increment) rather than a hard constraint, because F*C
  interaction strength is something the model should be free to learn the
  *magnitude* of from data -- only the *sign* (non-negative increment) needs
  to be hard. This is a deliberate, named compromise: we get zero constraint
  violations by construction, and we get the coupling direction by
  construction, but the coupling's learned strength is only as good as
  training teaches it (see DECISIONS.md D-coupling for the alternative we
  rejected: a fully hand-specified M update, which would have made "did the
  model learn the hazard" untestable).
"""
from __future__ import annotations
import torch
import torch.nn as nn

# indices into the 8-D state, matching generator.py
F, D, S, P, A, C, M, FLARE = range(8)
RATCHET_UP_IDX = [F, D, P]     # strictly non-decreasing, no exceptions
RATCHET_S_IDX = S              # non-decreasing except ERCP-gated relief
RATCHET_M_IDX = M              # non-decreasing, F*C-gated increment
FAST_IDX = [A, C, FLARE]       # bounded, not monotone

STEP_SCALE = 0.35  # caps how much a single softplus unit can move a ratchet in one month


class ConstraintHead(nn.Module):
    """Maps raw network output (B, 9) -> constrained x_next (B, 8).

    Raw layout: [F_inc, D_inc, S_creep, S_relief_raw, P_inc, A_logit, C_logit, M_inc, flare_logit]
    (9 raw numbers because S needs two: a creep term and a relief term.)
    """
    RAW_DIM = 9

    def __init__(self):
        super().__init__()

    def forward(self, raw: torch.Tensor, x_prev: torch.Tensor, ercp_flag: torch.Tensor) -> torch.Tensor:
        """
        raw:       (B, 9)
        x_prev:    (B, 8)
        ercp_flag: (B,) or (B,1) in {0,1} -- whether an ERCP event occurs this step
        """
        if ercp_flag.dim() == 1:
            ercp_flag = ercp_flag.unsqueeze(-1)

        F_inc = STEP_SCALE * torch.nn.functional.softplus(raw[:, 0:1])
        D_inc = STEP_SCALE * torch.nn.functional.softplus(raw[:, 1:2])
        S_creep = STEP_SCALE * torch.nn.functional.softplus(raw[:, 2:3])
        S_relief = STEP_SCALE * torch.nn.functional.softplus(raw[:, 3:4]) * 2.0  # bigger scale, relief is a bigger jump
        P_inc = STEP_SCALE * torch.nn.functional.softplus(raw[:, 4:5])
        A_new = torch.sigmoid(raw[:, 5:6])
        C_new = torch.sigmoid(raw[:, 6:7])
        M_inc_raw = STEP_SCALE * torch.nn.functional.softplus(raw[:, 7:8])
        flare_new = torch.sigmoid(raw[:, 8:9])

        Fp = x_prev[:, F:F+1]
        Dp = x_prev[:, D:D+1]
        Sp = x_prev[:, S:S+1]
        Pp = x_prev[:, P:P+1]
        Mp = x_prev[:, M:M+1]
        Cp = x_prev[:, C:C+1]

        F_next = torch.clamp(Fp + F_inc, 0.0, 1.0)
        D_next = torch.clamp(Dp + D_inc, 0.0, 1.0)
        # S: creep always applied; relief only where ERCP flag is set, and relief
        # is a genuinely signed decrease (not run through softplus-add), gated
        # multiplicatively by the flag so absent-ERCP => zero relief, always.
        S_next = torch.clamp(Sp + S_creep - ercp_flag * S_relief, 0.0, 1.0)
        P_next = torch.clamp(Pp + P_inc, 0.0, 1.0)
        # M: hazard increment gated by prev F*C (the coupling), then softplus'd
        # increment itself is still guaranteed >= 0, so gating only modulates
        # MAGNITUDE, never sign -- the non-decreasing guarantee is untouched.
        coupling_gate = (Fp * Cp).detach() * 0 + (Fp * Cp)  # keep gradient path
        M_inc = M_inc_raw * (0.3 + 1.7 * coupling_gate)  # gate in [0.3,2.0]x, never zero (avoid dead grad)
        M_next = torch.clamp(Mp + M_inc, 0.0, 2.0)

        x_next = torch.cat([F_next, D_next, S_next, P_next, A_new, C_new, M_next, flare_new], dim=1)
        return x_next


def cirrhosis_stage_torch(F_val: torch.Tensor) -> torch.Tensor:
    """Same derived function as generator.cirrhosis_stage, differentiable-ish
    (used only for reporting, not backprop)."""
    return torch.clamp(torch.floor(F_val * 3), 0, 2)
