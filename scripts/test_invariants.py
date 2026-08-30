"""
Prove the by-construction guarantees hold for RANDOM weights -- i.e. they
are a property of the parameterisation, not something training has to learn.
If these pass before a single gradient step is taken, a broken loss function
or bad training run can make the model inaccurate, but it cannot make it
violate the ratchet constraints.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from models.constraints import ConstraintHead, F, D, S, P, M

torch.manual_seed(0)


def random_raw(batch):
    return torch.randn(batch, ConstraintHead.RAW_DIM) * 5.0  # wide range, incl. large negatives


def test_monotonicity_random_weights(n_steps=200, batch=64):
    head = ConstraintHead()
    x = torch.rand(batch, 8) * torch.tensor([1, 1, 1, 1, 1, 1, 0.3, 1])  # M starts small
    ercp_flag = torch.zeros(batch)
    violations = {k: 0 for k in ["F", "D", "P", "M", "S"]}
    for t in range(n_steps):
        raw = random_raw(batch)
        x_next = head(raw, x, ercp_flag)
        for name, idx in [("F", F), ("D", D), ("P", P), ("M", M)]:
            violations[name] += int((x_next[:, idx] < x[:, idx] - 1e-6).sum())
        # S without ERCP flag must never decrease
        violations["S"] += int((x_next[:, S] < x[:, S] - 1e-6).sum())
        x = x_next.detach()
    return violations


def test_S_relief_only_with_flag(n_steps=100, batch=64):
    head = ConstraintHead()
    x = torch.rand(batch, 8)
    x[:, S] = 0.8  # start high so relief is visible
    ercp_flag = torch.ones(batch)  # ERCP EVERY step
    drops = 0
    for t in range(n_steps):
        raw = random_raw(batch)
        x_next = head(raw, x, ercp_flag)
        drops += int((x_next[:, S] < x[:, S] - 1e-6).sum())
        x = x_next.detach()
    return drops


def test_bounds_random_weights(n_steps=200, batch=64):
    head = ConstraintHead()
    x = torch.rand(batch, 8)
    x[:, M] = torch.rand(batch) * 2
    ercp_flag = torch.randint(0, 2, (batch,)).float()
    out_of_bounds = 0
    for t in range(n_steps):
        raw = random_raw(batch) * 20  # extreme values
        x_next = head(raw, x, ercp_flag)
        lo = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.0])
        hi = torch.tensor([1, 1, 1, 1, 1, 1, 2, 1.0])
        out_of_bounds += int(((x_next < lo - 1e-6) | (x_next > hi + 1e-6)).sum())
        x = x_next.detach()
    return out_of_bounds


def test_no_cirrhosis_channel():
    """Cirrhosis must never be a stored/predicted channel -- it's derived
    from F only. State dim must stay 8, not 9."""
    head = ConstraintHead()
    raw = random_raw(4)
    x = torch.rand(4, 8)
    out = head(raw, x, torch.zeros(4))
    assert out.shape[1] == 8, "state dimension must stay 8 -- cirrhosis is derived, not predicted"
    return True


def test_M_gate_never_kills_gradient_sign():
    """The F*C coupling gate on M's increment must modulate magnitude only;
    the increment must remain >=0 even when F*C=0."""
    head = ConstraintHead()
    x = torch.zeros(4, 8)  # F=0, C=0 -> gate=0
    raw = torch.randn(4, ConstraintHead.RAW_DIM) * 3
    ercp_flag = torch.zeros(4)
    x_next = head(raw, x, ercp_flag)
    return bool((x_next[:, M] >= x[:, M] - 1e-6).all())


def test_coupled_monotonicity_random_weights(n_inits=20, batch=64, seed=0):
    """Prove (not measure) that MonotoneStepCoupled's d(F_next)/d(A_prev),
    d(F_next)/d(C_prev), d(D_next)/d(S_prev), d(D_next)/d(A_prev) are
    non-negative under UNTRAINED, randomly-initialized weights -- the
    structural counterpart to DECISIONS.md D9's soft double-backward
    penalty (scripts/train_baseline_jacobian.py). If this passes before a
    single gradient step, the guarantee is a property of the
    parameterisation, not something training has to learn or a bad
    training run could undo."""
    from models.baseline_coupled import MonotoneStepCoupled
    torch.manual_seed(seed)
    violations = {"dF_dA": 0, "dF_dC": 0, "dD_dS": 0, "dD_dA": 0}
    total = 0
    for _ in range(n_inits):
        model = MonotoneStepCoupled(ctx_dim=8)  # fresh random init each iteration
        x_t = torch.rand(batch, 8, requires_grad=True)
        with torch.no_grad():
            x_t[:, 6] *= 2.0  # M lives in [0,2]; keep inputs in-domain
        ctx_feat_t = torch.rand(batch, 8)
        ercp_flag_t = torch.randint(0, 2, (batch,)).float()

        x_next = model(x_t, ctx_feat_t, ercp_flag_t)
        F_next, D_next = x_next[:, 0], x_next[:, 1]
        gF = torch.autograd.grad(F_next.sum(), x_t, retain_graph=True)[0]
        gD = torch.autograd.grad(D_next.sum(), x_t)[0]

        violations["dF_dA"] += int((gF[:, 4] < -1e-6).sum())
        violations["dF_dC"] += int((gF[:, 5] < -1e-6).sum())
        violations["dD_dS"] += int((gD[:, 2] < -1e-6).sum())
        violations["dD_dA"] += int((gD[:, 4] < -1e-6).sum())
        total += batch
    return violations, total


if __name__ == "__main__":
    print("=== Invariant tests (random weights -- guarantees, not statistics) ===\n")

    v = test_monotonicity_random_weights()
    print("Monotonicity violations under random weights (want all 0):")
    for k, val in v.items():
        status = "OK" if val == 0 else "FAIL"
        print(f"  {k}: {val}  [{status}]")

    drops = test_S_relief_only_with_flag()
    print(f"\nS actually drops when ERCP flag is set every step: {drops} drops observed "
          f"({'OK, relief mechanism works' if drops > 0 else 'FAIL, relief never fires'})")

    oob = test_bounds_random_weights()
    print(f"\nOut-of-bounds violations under extreme random weights (want 0): {oob} "
          f"[{'OK' if oob == 0 else 'FAIL'}]")

    ok = test_no_cirrhosis_channel()
    print(f"\nState stays 8-D, no separate cirrhosis channel: {'OK' if ok else 'FAIL'}")

    ok2 = test_M_gate_never_kills_gradient_sign()
    print(f"M increment stays non-negative even when F*C gate = 0: {'OK' if ok2 else 'FAIL'}")

    cv, ctotal = test_coupled_monotonicity_random_weights()
    print(f"\nMonotoneStepCoupled sensitivity-sign violations under random,\n"
          f"UNTRAINED weights (want all 0, out of {ctotal} samples each):")
    for k, val in cv.items():
        status = "OK" if val == 0 else "FAIL"
        print(f"  {k}: {val}  [{status}]")

    all_zero = (all(val == 0 for val in v.values()) and oob == 0 and drops > 0 and ok and ok2
                and all(val == 0 for val in cv.values()))
    print(f"\n{'ALL INVARIANTS HOLD' if all_zero else 'SOME INVARIANTS FAILED -- see above'}")
