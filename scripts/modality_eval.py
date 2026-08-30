"""
Uses models/modalities.py to answer the spec's implicit question directly:
"an error in x shows up across every modality at once" -- by how much, in
each modality's own units, for the baseline's and TS-JEPA's actual free
rollouts?

This is diagnostic, not a new accuracy metric to optimise against: it
translates the already-measured state-space MAE (eval.py, compare.py)
into clinically legible units (mg/dL, kPa, discrete stage) so a reviewer
can judge whether a given ratchet MAE is a big deal or a rounding error
in practice.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import make_train_val, action_dim
from models.baseline import MonotoneStep
from models.jepa import TSJEPA
from models.modalities import modality_error_report
from compare import baseline_rollout, jepa_rollout, build_ctx_and_ercp

T = 60


def main():
    base = MonotoneStep(ctx_dim=action_dim())
    base.load_state_dict(torch.load("checkpoints/baseline.pt")); base.eval()
    jepa = TSJEPA(action_dim=action_dim())
    jepa.load_state_dict(torch.load("checkpoints/jepa.pt")); jepa.eval()

    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    X = torch.tensor(val["X"], dtype=torch.float32)
    ctx_feats, ercp = build_ctx_and_ercp(val["ctx"], val["ercp"], T)
    start_t = int(T * 0.3)
    K = 24
    t_eval = min(start_t + K, T - 1)

    bp = baseline_rollout(base, X, ctx_feats, ercp, T, start_t)
    jp = jepa_rollout(jepa, X, ctx_feats, ercp, T, start_t)

    x_true = val["X"][:, t_eval, :]
    x_base = bp.numpy()[:, t_eval, :]
    x_jepa = jp.numpy()[:, t_eval, :]

    report_base = modality_error_report(x_true, x_base)
    report_jepa = modality_error_report(x_true, x_jepa)

    print(f"Per-modality MAE at K={K} (clean rendering, no observation noise added)\n")
    print(f"{'observable':30s} {'baseline':>10s} {'TS-JEPA':>10s}")
    for k in report_base:
        print(f"{k:30s} {report_base[k]:10.4f} {report_jepa[k]:10.4f}")

    print("\nContext for interpreting these numbers:")
    print("  bilirubin MAE ~0.1-0.2 mg/dL is clinically negligible (normal range ~0.3-1.2 mg/dL)")
    print("  liver_stiffness MAE ~0.5-1 kPa is small relative to fibrosis-stage cutoffs (~2 kPa apart)")
    print("  ishak_fibrosis_stage is discrete 0-6; MAE <0.5 means rollout error rarely crosses a stage boundary")


if __name__ == "__main__":
    main()
