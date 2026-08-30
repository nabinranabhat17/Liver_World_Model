"""
Evaluation harness. At minimum (per assignment):
  - predictive accuracy on held-out trajectories (free rollout, several K)
  - constraint-violation rate
  - the generalisation probe(s), with failures shown, not hidden

Also reports the "ratchet MAE": MAE specifically on the non-decreasing
channels (F,D,S,P,M), since those are the clinically load-bearing fields
and the ones the constraint mechanism targets -- a model that nails A/C
but blows up on F is not the win an aggregate MAE would suggest.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import (make_train_val, make_probe_held_out_susceptibility,
                   make_probe_unseen_treatment_timing, make_probe_long_rollout,
                   make_context_features, action_dim)
from models.baseline import MonotoneStep
from models.constraints import RATCHET_UP_IDX, RATCHET_S_IDX, RATCHET_M_IDX
from generator import check_constraints, FIELD_NAMES

T = 60
RATCHET_IDX = RATCHET_UP_IDX + [RATCHET_S_IDX, RATCHET_M_IDX]  # F,D,P,S,M


def build_ctx_and_ercp(ctx_np, ercp_np, T):
    N = ctx_np.shape[0]
    ctx_feats = np.stack([make_context_features(ctx_np, t, T) for t in range(T)], axis=1)
    return torch.tensor(ctx_feats, dtype=torch.float32), torch.tensor(ercp_np, dtype=torch.float32)


@torch.no_grad()
def free_rollout(model, X, ctx_feats, ercp, T, start_frac=0.3):
    """Roll out from a fraction of the way through each trajectory (using
    the true state as the seed, as a real deployment would: 'given the
    trajectory so far, predict forward'), free-running (no teacher forcing)
    to the end. Returns predicted (B,T,8) with the pre-start segment equal
    to ground truth (not counted in loss) and post-start segment predicted."""
    B = X.shape[0]
    start_t = int(T * start_frac)
    x = X[:, start_t, :].clone()
    preds = X.clone()
    for t in range(start_t + 1, T):
        x = model(x, ctx_feats[:, t, :], ercp[:, t])
        preds[:, t, :] = x
    return preds, start_t


def ratchet_mae_at_K(preds, X, start_t, K):
    """MAE on ratchet channels at horizon K steps past start_t."""
    t = min(start_t + K, X.shape[1] - 1)
    err = (preds[:, t, RATCHET_IDX] - X[:, t, RATCHET_IDX]).abs()
    return err.mean().item()


def full_mae_at_K(preds, X, start_t, K):
    t = min(start_t + K, X.shape[1] - 1)
    err = (preds[:, t, :] - X[:, t, :]).abs()
    return err.mean().item()


def constraint_violation_rate(preds_np, ercp_np):
    """Reuse generator.check_constraints on model OUTPUT, treating the
    model's own predicted S trajectory to decide which drops are 'allowed'
    is wrong -- instead we check against ground-truth ERCP timing, since
    that's the actual mechanism the model was given (ercp_flag input)."""
    v = check_constraints(preds_np, ercp_np)
    ratchet_viol = v["F"] + v["D"] + v["P"] + v["M"] + v["S_unexpected_drop"]
    total = v["total_steps"] * 5  # 5 ratchet-constrained channels
    return ratchet_viol, total, v


def evaluate_model(model, dataset, T, label, start_frac=0.3, Ks=(4, 8, 16, 24)):
    X = torch.tensor(dataset["X"], dtype=torch.float32)
    ctx_feats, ercp = build_ctx_and_ercp(dataset["ctx"], dataset["ercp"], T)
    preds, start_t = free_rollout(model, X, ctx_feats, ercp, T, start_frac)

    print(f"\n--- {label} (N={X.shape[0]}, T={T}, rollout from t={start_t}) ---")
    for K in Ks:
        if start_t + K >= T:
            continue
        rmae = ratchet_mae_at_K(preds, X, start_t, K)
        fmae = full_mae_at_K(preds, X, start_t, K)
        print(f"  K={K:3d}  ratchet MAE {rmae:.4f}   full-state MAE {fmae:.4f}")

    viol, total, detail = constraint_violation_rate(preds.numpy(), dataset["ercp"])
    print(f"  constraint violation rate: {viol}/{total} = {viol/total:.8f}")
    print(f"    breakdown: {detail}")
    return preds, start_t


def main():
    train, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    model = MonotoneStep(ctx_dim=action_dim())
    model.load_state_dict(torch.load("checkpoints/baseline.pt"))
    model.eval()

    print("=" * 70)
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)

    evaluate_model(model, val, T, "held-out validation (in-distribution)")

    probe1 = make_probe_held_out_susceptibility(n=400)
    evaluate_model(model, probe1, T, "PROBE 1: held-out susceptibility (1.0-1.6, never trained on)")

    probe2 = make_probe_unseen_treatment_timing(n=400)
    evaluate_model(model, probe2, T, "PROBE 2: unseen treatment timing (UDCA start month 45-59)")

    probe3 = make_probe_long_rollout(n=200, T=90)
    evaluate_model(model, probe3, 90, "PROBE 3: long rollout (T=90, 30mo beyond training horizon)",
                   Ks=(4, 8, 16, 24, 40, 55))


if __name__ == "__main__":
    main()
