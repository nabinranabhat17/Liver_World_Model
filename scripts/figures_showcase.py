"""Generate the memo's summary figures from the real trained checkpoints
(no fabricated numbers -- every figure here re-runs the actual models)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.baseline import MonotoneStep
from models.jepa import TSJEPA
from data import make_train_val, make_probe_held_out_susceptibility, action_dim
from compare import run_axis, add_sensor_noise, stale_last_visit, baseline_rollout, jepa_rollout
from eval import build_ctx_and_ercp, ratchet_mae_at_K

T = 60


def load_models():
    base = MonotoneStep(ctx_dim=action_dim())
    base.load_state_dict(torch.load("checkpoints/baseline.pt"))
    base.eval()
    jepa = TSJEPA(action_dim=action_dim())
    jepa.load_state_dict(torch.load("checkpoints/jepa.pt"))
    jepa.eval()
    jepa_dn = TSJEPA(action_dim=action_dim())
    jepa_dn.load_state_dict(torch.load("checkpoints/jepa_denoise.pt"))
    jepa_dn.eval()
    return base, jepa, jepa_dn


def fig_scorecard(base, jepa, val, probe_susc):
    axes_labels = ["Clean\nin-dist.", "Held-out\nsusceptibility", "Unseen\ntreatment\ntiming", "Long\nrollout\n(T=90)"]
    b_vals, j_vals = [], []

    ctx_feats, ercp = build_ctx_and_ercp(val["ctx"], val["ercp"], T)
    X = torch.tensor(val["X"], dtype=torch.float32)
    start_t = int(T * 0.3)
    bp = baseline_rollout(base, X, ctx_feats, ercp, T, start_t)
    jp = jepa_rollout(jepa, X, ctx_feats, ercp, T, start_t)
    b_vals.append(ratchet_mae_at_K(bp, X, start_t, 24))
    j_vals.append(ratchet_mae_at_K(jp, X, start_t, 24))

    ctx_feats2, ercp2 = build_ctx_and_ercp(probe_susc["ctx"], probe_susc["ercp"], T)
    X2 = torch.tensor(probe_susc["X"], dtype=torch.float32)
    bp2 = baseline_rollout(base, X2, ctx_feats2, ercp2, T, start_t)
    jp2 = jepa_rollout(jepa, X2, ctx_feats2, ercp2, T, start_t)
    b_vals.append(ratchet_mae_at_K(bp2, X2, start_t, 24))
    j_vals.append(ratchet_mae_at_K(jp2, X2, start_t, 24))

    from data import make_probe_unseen_treatment_timing, make_probe_long_rollout
    p2 = make_probe_unseen_treatment_timing(n=400)
    ctx_feats3, ercp3 = build_ctx_and_ercp(p2["ctx"], p2["ercp"], T)
    X3 = torch.tensor(p2["X"], dtype=torch.float32)
    bp3 = baseline_rollout(base, X3, ctx_feats3, ercp3, T, start_t)
    jp3 = jepa_rollout(jepa, X3, ctx_feats3, ercp3, T, start_t)
    b_vals.append(ratchet_mae_at_K(bp3, X3, start_t, 24))
    j_vals.append(ratchet_mae_at_K(jp3, X3, start_t, 24))

    p3 = make_probe_long_rollout(n=200, T=90)
    ctx_feats4, ercp4 = build_ctx_and_ercp(p3["ctx"], p3["ercp"], 90)
    X4 = torch.tensor(p3["X"], dtype=torch.float32)
    start_t4 = int(90 * 0.3)
    bp4 = baseline_rollout(base, X4, ctx_feats4, ercp4, 90, start_t4)
    jp4 = jepa_rollout(jepa, X4, ctx_feats4, ercp4, 90, start_t4)
    b_vals.append(ratchet_mae_at_K(bp4, X4, start_t4, 55))
    j_vals.append(ratchet_mae_at_K(jp4, X4, start_t4, 55))

    x = np.arange(len(axes_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, b_vals, width, label="baseline", color="tab:blue")
    ax.bar(x + width/2, j_vals, width, label="TS-JEPA", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(axes_labels)
    ax.set_ylabel("ratchet MAE (K=24, K=55 for long rollout)")
    ax.set_title("Baseline vs TS-JEPA: accuracy across generalisation probes")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig_scorecard.png", dpi=120)
    plt.close()
    print("saved figures/fig_scorecard.png")
    return b_vals, j_vals


def fig_noise(base, jepa, jepa_dn, val, n_seeds=8):
    """Multi-seed averaged (a single-seed version of this experiment showed
    an apparent crossover at high noise for the denoised JEPA; that did NOT
    replicate across seeds -- see DECISIONS.md. This figure reports the
    honest, averaged result.)"""
    sigmas = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25]
    ctx_feats, ercp = build_ctx_and_ercp(val["ctx"], val["ercp"], T)
    Xtrue = torch.tensor(val["X"], dtype=torch.float32)
    start_t = int(T * 0.3)
    b_res, j_res, jd_res = [], [], []
    b_std, jd_std = [], []
    for sigma in sigmas:
        b_seed, j_seed, jd_seed = [], [], []
        n_s = 1 if sigma == 0 else n_seeds
        for seed in range(n_s):
            rng = np.random.default_rng(seed)
            Xn = val["X"] if sigma == 0 else add_sensor_noise(val["X"], sigma, rng)
            X = torch.tensor(Xn, dtype=torch.float32)
            bp = baseline_rollout(base, X, ctx_feats, ercp, T, start_t)
            jp = jepa_rollout(jepa, X, ctx_feats, ercp, T, start_t)
            jdp = jepa_rollout(jepa_dn, X, ctx_feats, ercp, T, start_t)
            b_seed.append(ratchet_mae_at_K(bp, Xtrue, start_t, 24))
            j_seed.append(ratchet_mae_at_K(jp, Xtrue, start_t, 24))
            jd_seed.append(ratchet_mae_at_K(jdp, Xtrue, start_t, 24))
        b_res.append(np.mean(b_seed)); j_res.append(np.mean(j_seed)); jd_res.append(np.mean(jd_seed))
        b_std.append(np.std(b_seed)); jd_std.append(np.std(jd_seed))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sigmas, b_res, yerr=b_std, fmt="-o", label="baseline", color="tab:blue", capsize=3)
    ax.plot(sigmas, j_res, "-o", label="TS-JEPA (plain)", color="tab:orange")
    ax.errorbar(sigmas, jd_res, yerr=jd_std, fmt="-o", label="TS-JEPA (noise-augmented)", color="tab:green", capsize=3)
    ax.set_xlabel("sensor noise sigma")
    ax.set_ylabel(f"ratchet MAE (K=24), mean +/- std over {n_seeds} noise seeds")
    ax.set_title("Accuracy under sensor noise (multi-seed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig_noise.png", dpi=120)
    plt.close()
    print("saved figures/fig_noise.png")
    return sigmas, b_res, j_res, jd_res


def fig_staleness(base, jepa, val):
    stales = [0, 3, 6, 9, 12, 15, 18]
    b_res, j_res = [], []
    for stale in stales:
        Xs = stale_last_visit(val["X"], stale, start_t=int(T * 0.3)) if stale > 0 else val["X"]
        ctx_feats, ercp = build_ctx_and_ercp(val["ctx"], val["ercp"], T)
        X = torch.tensor(Xs, dtype=torch.float32)
        start_t = int(T * 0.3)
        bp = baseline_rollout(base, X, ctx_feats, ercp, T, start_t)
        jp = jepa_rollout(jepa, X, ctx_feats, ercp, T, start_t)
        Xtrue = torch.tensor(val["X"], dtype=torch.float32)
        b_res.append(ratchet_mae_at_K(bp, Xtrue, start_t, 24))
        j_res.append(ratchet_mae_at_K(jp, Xtrue, start_t, 24))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(stales, b_res, "-o", label="baseline", color="tab:blue")
    ax.plot(stales, j_res, "-o", label="TS-JEPA", color="tab:orange")
    ax.set_xlabel("months since last real visit (staleness)")
    ax.set_ylabel("ratchet MAE (K=24)")
    ax.set_title("Accuracy under stale last observation")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig_staleness.png", dpi=120)
    plt.close()
    print("saved figures/fig_staleness.png")
    return stales, b_res, j_res


def fig_training_curves():
    """Re-plot loss curves already logged during training runs (re-run
    briefly at reduced epochs here for a clean, fast figure)."""
    import subprocess
    fig, ax = plt.subplots(figsize=(7, 5))
    # illustrative: use the already-trained checkpoints' final numbers is not
    # a curve; instead note this needs re-instrumented training to log
    # per-epoch val loss arrays. Skipped here to avoid re-training cost;
    # see make_training_curves.py for the instrumented version.
    plt.close()


if __name__ == "__main__":
    base, jepa, jepa_dn = load_models()
    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    probe_susc = make_probe_held_out_susceptibility(n=400)

    fig_scorecard(base, jepa, val, probe_susc)
    fig_noise(base, jepa, jepa_dn, val)
    fig_staleness(base, jepa, val)
    print("\nAll figures generated.")
