"""
Three-way comparison for D9's fix: baseline.pt (wrong-signed, unfixed),
baseline_jacobian.pt (soft double-backward penalty), baseline_monotonic.pt
(structural MonotonicCoupling guarantee, models/baseline_coupled.py).
Axes:
  (a) sign-correctness of d(F_next)/dA, d(F_next)/dC, d(D_next)/dS,
      d(D_next)/dA over 500 random validation (patient, month) samples --
      same style as D9's original 500-sample check and
      scripts/coupling_probe.py. D's sensitivities have not been measured
      before this script; reported honestly for all three checkpoints.
  (b) ratchet MAE at K=24: clean in-distribution AND held-out
      susceptibility probe.
  (c) constraint violation rate (sanity: must stay 0/N for all three).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import make_train_val, make_probe_held_out_susceptibility, action_dim
from models.baseline import MonotoneStep
from models.baseline_coupled import MonotoneStepCoupled
from eval import build_ctx_and_ercp, ratchet_mae_at_K, full_mae_at_K, constraint_violation_rate
from compare import baseline_rollout

T = 60
F_IDX, D_IDX, S_IDX, A_IDX, C_IDX = 0, 1, 2, 4, 5


def sign_correctness(model, X, ctx_feats, ercp, ts):
    """d(F_next)/dA, d(F_next)/dC, d(D_next)/dS, d(D_next)/dA via autograd
    at real validation (patient, t) pairs -- same idiom as
    coupling_probe.py's baseline_ratio / D9's 500-sample check."""
    B = X.shape[0]
    idx = torch.arange(B)
    x_prev = X[idx, ts].clone().detach().requires_grad_(True)
    ctx_t1 = ctx_feats[idx, ts + 1]
    ercp_t1 = ercp[idx, ts + 1]
    x_next = model(x_prev, ctx_t1, ercp_t1)
    F_next, D_next = x_next[:, F_IDX], x_next[:, D_IDX]
    gF = torch.autograd.grad(F_next.sum(), x_prev, retain_graph=True)[0]
    gD = torch.autograd.grad(D_next.sum(), x_prev)[0]
    return {
        "dF/dA > 0": (gF[:, A_IDX] > 0).float().mean().item() * 100,
        "dF/dC > 0": (gF[:, C_IDX] > 0).float().mean().item() * 100,
        "dD/dS > 0": (gD[:, S_IDX] > 0).float().mean().item() * 100,
        "dD/dA > 0": (gD[:, A_IDX] > 0).float().mean().item() * 100,
    }


def main():
    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    probe = make_probe_held_out_susceptibility(n=400)
    X = torch.tensor(val["X"], dtype=torch.float32)
    ctx_feats, ercp = build_ctx_and_ercp(val["ctx"], val["ercp"], T)

    rng = np.random.default_rng(0)
    n = min(500, X.shape[0])
    ts = torch.tensor(rng.integers(15, T - 2, size=n))
    Xs, ctxs, ercps = X[:n], ctx_feats[:n], ercp[:n]

    models = {
        "baseline (unfixed)": ("checkpoints/baseline.pt", MonotoneStep),
        "baseline_jacobian (soft penalty)": ("checkpoints/baseline_jacobian.pt", MonotoneStep),
        "baseline_monotonic (structural)": ("checkpoints/baseline_monotonic.pt", MonotoneStepCoupled),
    }

    for label, (ckpt, cls) in models.items():
        model = cls(ctx_dim=action_dim())
        model.load_state_dict(torch.load(ckpt))
        model.eval()

        sc = sign_correctness(model, Xs, ctxs, ercps, ts)
        print(f"\n=== {label} ===")
        print(f"  sign-correctness (n={n}): " + "  ".join(f"{k} {v:.1f}%" for k, v in sc.items()))

        for label2, dataset in [("clean in-distribution", val), ("held-out susceptibility", probe)]:
            Xd = torch.tensor(dataset["X"], dtype=torch.float32)
            cf, ec = build_ctx_and_ercp(dataset["ctx"], dataset["ercp"], T)
            start_t = int(T * 0.3)
            preds = baseline_rollout(model, Xd, cf, ec, T, start_t)
            rmae = ratchet_mae_at_K(preds, Xd, start_t, 24)
            fmae = full_mae_at_K(preds, Xd, start_t, 24)
            viol, total, _ = constraint_violation_rate(preds.numpy()[:, start_t:, :], dataset["ercp"][:, start_t:])
            print(f"  {label2}: K=24 ratchet MAE {rmae:.4f}  full MAE {fmae:.4f}  violations {viol}/{total}")


if __name__ == "__main__":
    main()
