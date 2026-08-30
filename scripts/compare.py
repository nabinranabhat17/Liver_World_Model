"""
Head-to-head: baseline vs TS-JEPA.

Axes tested:
  1. clean, in-distribution (the sanitised comparison point)
  2. PROBE: held-out susceptibility
  3. PROBE: unseen treatment timing
  4. PROBE: long rollout (T=90)
  5. STRESS: sensor noise added to the observed history before the model
     ever sees it (ground truth used only for scoring). This is where a
     memoryless baseline (x_prev IS its belief state) has no mechanism to
     denoise, while JEPA's encoder integrates the whole window and can, in
     principle, average noise out.
  6. STRESS: stale last visit (large gap since the last observation,
     simulated by holding x fixed for a stretch before rollout starts).
     Tests whether JEPA's history integration helps it "know" more about
     where the patient really is than the last raw reading suggests.

We report ratchet MAE at K=24 throughout, and are explicit that a win on
axes 5-6 for JEPA (if it happens) is not free -- see the honesty checks in
DECISIONS.md: this is not evidence that JEPA "understands" the disease
better in-distribution, only that its architecture has a mechanism the
baseline structurally lacks.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import (make_train_val, make_probe_held_out_susceptibility,
                   make_probe_unseen_treatment_timing, make_probe_long_rollout,
                   make_context_features, action_dim)
from models.baseline import MonotoneStep
from models.jepa import TSJEPA
from eval import build_ctx_and_ercp, ratchet_mae_at_K, full_mae_at_K, constraint_violation_rate

T = 60
K_REPORT = 24


@torch.no_grad()
def baseline_rollout(model, X, ctx_feats, ercp, T, start_t):
    x = X[:, start_t, :].clone()
    preds = X.clone()
    for t in range(start_t + 1, T):
        x = model(x, ctx_feats[:, t, :], ercp[:, t])
        preds[:, t, :] = x
    return preds


@torch.no_grad()
def jepa_rollout(model, X, ctx_feats, ercp, T, start_t):
    """Encode history up to start_t (causal), then roll the latent forward
    with the predictor using only action features, decode-anchor chained."""
    z_seq = model.online_encoder(X[:, :start_t + 1, :], ctx_feats[:, :start_t + 1, :])
    z_t = z_seq[:, -1, :]
    k = T - start_t - 1
    action_future = ctx_feats[:, start_t + 1:T, :]
    ercp_future = ercp[:, start_t + 1:T]
    z_preds = model.predictor.rollout(z_t, action_future)
    x_anchor = X[:, start_t, :]
    x_hats = model.decoder.decode_chain(z_preds, x_anchor, ercp_future)  # (B,k,8)
    preds = X.clone()
    preds[:, start_t + 1:T, :] = x_hats
    return preds


def add_sensor_noise(X, sigma, rng):
    noisy = X + rng.normal(0, sigma, size=X.shape)
    return np.clip(noisy, [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 2, 1])


def stale_last_visit(X, stale_months, start_t):
    """Simulate a patient whose last observed visit was `stale_months` ago:
    freeze x at (start_t - stale_months) and hold it flat up to start_t,
    as if that were the last real reading -- ground truth X is untouched
    for scoring."""
    Xs = X.copy()
    src_t = max(0, start_t - stale_months)
    Xs[:, src_t:start_t + 1, :] = Xs[:, src_t:src_t + 1, :]
    return Xs


def run_axis(label, X_np, ctx_np, ercp_np, base_model, jepa_model, start_frac=0.3, K=K_REPORT):
    X = torch.tensor(X_np, dtype=torch.float32)
    ctx_feats, ercp = build_ctx_and_ercp(ctx_np, ercp_np, T)
    start_t = int(T * start_frac)
    if start_t + K >= T:
        K = T - start_t - 1

    Xtrue = torch.tensor(X_np, dtype=torch.float32)  # clean ground truth for scoring
    base_preds = baseline_rollout(base_model, X, ctx_feats, ercp, T, start_t)
    jepa_preds = jepa_rollout(jepa_model, X, ctx_feats, ercp, T, start_t)

    b_rmae = ratchet_mae_at_K(base_preds, Xtrue, start_t, K)
    b_fmae = full_mae_at_K(base_preds, Xtrue, start_t, K)
    j_rmae = ratchet_mae_at_K(jepa_preds, Xtrue, start_t, K)
    j_fmae = full_mae_at_K(jepa_preds, Xtrue, start_t, K)

    # Only score the ROLLED-OUT segment (start_t onward) for constraint violations --
    # the pre-start segment is raw (possibly noise-perturbed) ground truth, not a
    # model prediction, and would trivially "violate" monotonicity under noise.
    b_viol, b_total, _ = constraint_violation_rate(base_preds.numpy()[:, start_t:, :], ercp_np[:, start_t:])
    j_viol, j_total, _ = constraint_violation_rate(jepa_preds.numpy()[:, start_t:, :], ercp_np[:, start_t:])

    print(f"\n--- {label} (K={K}) ---")
    print(f"  baseline : ratchet MAE {b_rmae:.4f}  full MAE {b_fmae:.4f}  violations {b_viol}/{b_total}")
    print(f"  TS-JEPA  : ratchet MAE {j_rmae:.4f}  full MAE {j_fmae:.4f}  violations {j_viol}/{j_total}")
    winner = "baseline" if b_rmae < j_rmae else "TS-JEPA"
    margin = abs(b_rmae - j_rmae) / max(b_rmae, j_rmae) * 100
    print(f"  -> {winner} wins on ratchet MAE by {margin:.1f}%")
    return dict(label=label, baseline_ratchet_mae=b_rmae, jepa_ratchet_mae=j_rmae,
                baseline_full_mae=b_fmae, jepa_full_mae=j_fmae)


def main():
    global T
    base_model = MonotoneStep(ctx_dim=action_dim())
    base_model.load_state_dict(torch.load("checkpoints/baseline.pt"))
    base_model.eval()

    jepa_model = TSJEPA(action_dim=action_dim())
    jepa_model.load_state_dict(torch.load("checkpoints/jepa.pt"))
    jepa_model.eval()

    results = []
    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    results.append(run_axis("Clean, in-distribution", val["X"], val["ctx"], val["ercp"], base_model, jepa_model))

    p1 = make_probe_held_out_susceptibility(n=400)
    results.append(run_axis("PROBE: held-out susceptibility", p1["X"], p1["ctx"], p1["ercp"], base_model, jepa_model))

    p2 = make_probe_unseen_treatment_timing(n=400)
    results.append(run_axis("PROBE: unseen treatment timing", p2["X"], p2["ctx"], p2["ercp"], base_model, jepa_model))

    p3 = make_probe_long_rollout(n=200, T=90)
    ctx_feats3, ercp3 = build_ctx_and_ercp(p3["ctx"], p3["ercp"], 90)
    T_saved = T
    T = 90
    results.append(run_axis("PROBE: long rollout (T=90)", p3["X"], p3["ctx"], p3["ercp"], base_model, jepa_model, K=55))
    T = T_saved

    rng = np.random.default_rng(42)
    for sigma in [0.03, 0.06, 0.10]:
        noisy_X = add_sensor_noise(val["X"], sigma, rng)
        results.append(run_axis(f"STRESS: sensor noise sigma={sigma}", noisy_X, val["ctx"], val["ercp"], base_model, jepa_model))

    for stale in [6, 12, 18]:
        stale_X = stale_last_visit(val["X"], stale, start_t=int(T * 0.3))
        results.append(run_axis(f"STRESS: stale last visit {stale}mo", stale_X, val["ctx"], val["ercp"], base_model, jepa_model))

    print("\n" + "=" * 70)
    print("SUMMARY (ratchet MAE, lower is better)")
    print("=" * 70)
    print(f"{'axis':45s} {'baseline':>10s} {'TS-JEPA':>10s} {'winner':>10s}")
    for r in results:
        w = "baseline" if r["baseline_ratchet_mae"] < r["jepa_ratchet_mae"] else "TS-JEPA"
        print(f"{r['label']:45s} {r['baseline_ratchet_mae']:10.4f} {r['jepa_ratchet_mae']:10.4f} {w:>10s}")


if __name__ == "__main__":
    main()
