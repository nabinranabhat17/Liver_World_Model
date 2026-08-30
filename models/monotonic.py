"""
Building block for hard (by-construction) monotonicity in specific
INPUTS, not just outputs. `ConstraintHead` already guarantees ratchet
INCREMENTS are non-negative (softplus) and, for M, hard-gates the
increment's *magnitude* by a non-negative F*C product -- but nothing in
this repo constrains a raw score's *sensitivity to a specific input* the
way `PosLinear`/`MonotonicCoupling` do here. That gap is what let
d(F_next)/d(A_prev), d(F_next)/d(C_prev) come out wrong-signed on a
plain MLP despite the generator's true dF being non-negative in A, C
(see DECISIONS.md D9): softplus only protects the increment's own sign,
never the sign of its derivative with respect to any one input.

PosLinear + a non-decreasing activation composed together are
non-decreasing in every input whose weight is non-negative, for ANY
value the (unconstrained) underlying parameters take -- a guarantee
provable before a single gradient step, exactly like ConstraintHead's
own ratchet guarantees.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PosLinear(nn.Module):
    """Linear layer whose weight matrix is forced non-negative via
    w = softplus(raw_weight). Bias is left UNCONSTRAINED: a bias shifts
    the pre-activation additively and never appears in a partial
    derivative w.r.t. the input, so it can't break monotonicity -- only
    the weight's sign matters.

    softplus over abs()/square() for the reparametrisation: square(w) is
    non-identifiable (+-w map to the same effective weight) and has a
    zero-gradient saddle at 0; abs(w) is non-smooth at 0 with constant
    +-1 gradient regardless of magnitude, which tends to make near-zero
    weights oscillate under Adam instead of settling. softplus(w) is
    smooth, strictly increasing (a unique raw_weight per effective
    weight), can represent an "off" (~0) weight without a singularity,
    and is already this repo's own reparametrisation of choice for every
    ratchet increment in ConstraintHead -- reusing the same primitive
    elsewhere is a stated value here, not just taste.

    Init scale: Xavier/He assume a zero-mean weight distribution, where
    fan_in terms in a sum mostly CANCEL and only the variance needs
    controlling. Positive weights never cancel -- a sum of fan_in
    same-signed terms grows in direct proportion to fan_in unless each
    weight shrinks as ~1/fan_in. So `init_scale` targets a MEAN weight of
    `init_scale / in_features`, not a variance-matched one; this keeps a
    layer's total output near `init_scale` (for typical unit-scale
    inputs) regardless of width, which plain Xavier does not.
    """

    def __init__(self, in_features, out_features, init_scale=0.5):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        with torch.no_grad():
            mean_w = init_scale / max(in_features, 1)
            target = torch.empty(out_features, in_features).uniform_(0.5 * mean_w, 1.5 * mean_w)
            target = target.clamp_min(1e-4)
            # numerically stable softplus^{-1}(t) = t + log(1 - exp(-t))
            self.raw_weight.copy_(target + torch.log(-torch.expm1(-target)))

    def forward(self, x):
        w = F.softplus(self.raw_weight)
        return F.linear(x, w, self.bias)


class MonotonicCoupling(nn.Module):
    """Provably non-decreasing in every `driver` input, for ANY trainable
    parameters -- before OR after training, not a property training has
    to learn. Optionally also takes `free_feat`: extra conditioning
    inputs (e.g. context) with UNRESTRICTED weights. Because free_feat
    only enters the pre-activation additively and softplus's derivative
    is strictly positive everywhere, mixing it in can never zero out or
    flip the sign of d(output)/d(driver_i):

        d(output)/d(driver_i)
          = w_skip_i                                       (>= 0, PosLinear)
          + sum_h  w2_h * softplus'(z_h) * w1_drivers[h, i] (each factor >= 0)
        where z_h = (drivers @ w1_drivers.T)_h + (free_feat @ w1_free.T)_h + bias_h.

    A plain PosLinear -> softplus -> PosLinear stack can only represent
    *convex* non-decreasing shapes in the drivers (a non-negative sum of
    convex non-decreasing terms). The generator's true couplings here are
    exactly linear in the driver (dF ~ 0.6*A + 0.6*C, dD ~ 0.7*S + 0.3*A)
    -- both convex and concave, so representable -- and the extra
    `PosLinear(n_drivers, 1)` skip below makes that straight-line shape
    cheaply reachable rather than relying on the deeper path to
    approximate it.

    Output is intentionally unconstrained in overall sign/range (only
    monotonic *in the drivers*) -- it feeds into ConstraintHead's own
    softplus downstream, exactly like every other raw channel, so it
    must not be pre-squashed here.
    """

    def __init__(self, n_drivers, n_free=0, hidden=16):
        super().__init__()
        self.fc1_drivers = PosLinear(n_drivers, hidden)
        self.fc1_free = nn.Linear(n_free, hidden, bias=False) if n_free > 0 else None
        self.fc2 = PosLinear(hidden, 1)
        self.skip = PosLinear(n_drivers, 1)

    def forward(self, drivers, free_feat=None):
        z = self.fc1_drivers(drivers)
        if self.fc1_free is not None and free_feat is not None:
            z = z + self.fc1_free(free_feat)
        h = F.softplus(z)
        return self.fc2(h) + self.skip(drivers)
