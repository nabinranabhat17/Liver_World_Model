"""
Does continuous-time integration (finer sub-monthly resolution for the
fast A/C dynamics) help over the discrete one-step-per-month baseline?
Same interface, same rollout function (baseline_rollout works unmodified
since NeuralODEStep.forward matches MonotoneStep.forward exactly).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data import make_train_val, make_probe_held_out_susceptibility, action_dim
from models.baseline import MonotoneStep
from models.neural_ode import NeuralODEStep
from compare import baseline_rollout, build_ctx_and_ercp
from eval import ratchet_mae_at_K, full_mae_at_K, constraint_violation_rate

T = 60


def main():
    base = MonotoneStep(ctx_dim=action_dim())
    base.load_state_dict(torch.load("checkpoints/baseline.pt")); base.eval()

    ode = NeuralODEStep(ctx_dim=action_dim(), n_substeps=4)
    ode.load_state_dict(torch.load("checkpoints/neural_ode.pt")); ode.eval()

    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    probe = make_probe_held_out_susceptibility(n=400)

    for label, dataset in [("clean in-distribution", val), ("held-out susceptibility", probe)]:
        X = torch.tensor(dataset["X"], dtype=torch.float32)
        ctx_feats, ercp = build_ctx_and_ercp(dataset["ctx"], dataset["ercp"], T)
        start_t = int(T * 0.3)

        bp = baseline_rollout(base, X, ctx_feats, ercp, T, start_t)
        op = baseline_rollout(ode, X, ctx_feats, ercp, T, start_t)  # same rollout fn, drop-in

        print(f"\n--- {label} (K=24) ---")
        for name, preds in [("baseline (discrete)", bp), ("Neural-ODE (RK4, 4 substeps)", op)]:
            for K in [8, 24]:
                rmae = ratchet_mae_at_K(preds, X, start_t, K)
                fmae = full_mae_at_K(preds, X, start_t, K)
                print(f"  {name:32s} K={K:2d}  ratchet MAE {rmae:.4f}  full MAE {fmae:.4f}")
            viol, total, _ = constraint_violation_rate(preds.numpy()[:, start_t:, :], dataset["ercp"][:, start_t:])
            print(f"  {name:32s} violations {viol}/{total}")


if __name__ == "__main__":
    main()
