"""
Does the graph-attention encoder (causal edges as an attention mask)
actually help over the plain concat-GRU encoder? Same axes as compare.py,
same checkpoints for baseline/plain-JEPA, freshly trained graph-JEPA.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data import make_train_val, make_probe_held_out_susceptibility, action_dim
from models.baseline import MonotoneStep
from models.jepa import TSJEPA
from models.graph_encoder import GraphHistoryEncoder
from compare import jepa_rollout, baseline_rollout, build_ctx_and_ercp
from eval import ratchet_mae_at_K, full_mae_at_K, constraint_violation_rate

T = 60


def main():
    base = MonotoneStep(ctx_dim=action_dim())
    base.load_state_dict(torch.load("checkpoints/baseline.pt")); base.eval()

    jepa_plain = TSJEPA(action_dim=action_dim())
    jepa_plain.load_state_dict(torch.load("checkpoints/jepa.pt")); jepa_plain.eval()

    jepa_graph = TSJEPA(action_dim=action_dim(),
                         encoder=GraphHistoryEncoder(state_dim=8, action_dim=action_dim(), latent_dim=16))
    jepa_graph.load_state_dict(torch.load("checkpoints/jepa_graph.pt")); jepa_graph.eval()

    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    probe = make_probe_held_out_susceptibility(n=400)

    for label, dataset in [("clean in-distribution", val), ("held-out susceptibility", probe)]:
        X = torch.tensor(dataset["X"], dtype=torch.float32)
        ctx_feats, ercp = build_ctx_and_ercp(dataset["ctx"], dataset["ercp"], T)
        start_t = int(T * 0.3)

        bp = baseline_rollout(base, X, ctx_feats, ercp, T, start_t)
        jp = jepa_rollout(jepa_plain, X, ctx_feats, ercp, T, start_t)
        jg = jepa_rollout(jepa_graph, X, ctx_feats, ercp, T, start_t)

        print(f"\n--- {label} (K=24) ---")
        for name, preds in [("baseline", bp), ("TS-JEPA (plain)", jp), ("TS-JEPA (graph-attn)", jg)]:
            rmae = ratchet_mae_at_K(preds, X, start_t, 24)
            fmae = full_mae_at_K(preds, X, start_t, 24)
            viol, total, _ = constraint_violation_rate(preds.numpy()[:, start_t:, :], dataset["ercp"][:, start_t:])
            print(f"  {name:24s} ratchet MAE {rmae:.4f}  full MAE {fmae:.4f}  violations {viol}/{total}")


if __name__ == "__main__":
    main()
