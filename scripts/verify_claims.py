"""
Recomputes every load-bearing number cited in memo.md, from the actual
saved checkpoints, so the memo's claims are never "trust me" -- run this
and compare.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import make_train_val, make_probe_held_out_susceptibility, action_dim
from models.baseline import MonotoneStep
from models.jepa import TSJEPA
from compare import baseline_rollout, jepa_rollout, add_sensor_noise
from eval import build_ctx_and_ercp, ratchet_mae_at_K, constraint_violation_rate
from generator import check_constraints, generate_dataset

T = 60


def main():
    print("=" * 70)
    print("CLAIM 1: generator self-check -- 0 constraint violations in raw data")
    print("=" * 70)
    X, ctx, ercp, meta = generate_dataset(2000, T=T, seed=999)
    v = check_constraints(X, ercp)
    total_ratchet = v["total_steps"] * 5
    viol = v["F"] + v["D"] + v["P"] + v["M"] + v["S_unexpected_drop"]
    print(f"  {viol} / {total_ratchet} = {viol/total_ratchet:.8f}  (claim: 0.000000)")

    base = MonotoneStep(ctx_dim=action_dim())
    base.load_state_dict(torch.load("checkpoints/baseline.pt")); base.eval()
    jepa = TSJEPA(action_dim=action_dim())
    jepa.load_state_dict(torch.load("checkpoints/jepa.pt")); jepa.eval()
    jepa_dn = TSJEPA(action_dim=action_dim())
    jepa_dn.load_state_dict(torch.load("checkpoints/jepa_denoise.pt")); jepa_dn.eval()

    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    ctx_feats, ercp_v = build_ctx_and_ercp(val["ctx"], val["ercp"], T)
    Xv = torch.tensor(val["X"], dtype=torch.float32)
    start_t = int(T * 0.3)

    print("\n" + "=" * 70)
    print("CLAIM 2: clean in-distribution ratchet MAE @ K=24: baseline < TS-JEPA")
    print("=" * 70)
    bp = baseline_rollout(base, Xv, ctx_feats, ercp_v, T, start_t)
    jp = jepa_rollout(jepa, Xv, ctx_feats, ercp_v, T, start_t)
    b_mae = ratchet_mae_at_K(bp, Xv, start_t, 24)
    j_mae = ratchet_mae_at_K(jp, Xv, start_t, 24)
    print(f"  baseline: {b_mae:.4f}   TS-JEPA: {j_mae:.4f}   "
          f"baseline wins by {(j_mae-b_mae)/j_mae*100:.1f}%")

    print("\n" + "=" * 70)
    print("CLAIM 3: model rollout has 0 constraint violations (both models)")
    print("=" * 70)
    bv, bt, _ = constraint_violation_rate(bp.numpy()[:, start_t:, :], val["ercp"][:, start_t:])
    jv, jt, _ = constraint_violation_rate(jp.numpy()[:, start_t:, :], val["ercp"][:, start_t:])
    print(f"  baseline: {bv}/{bt}   TS-JEPA: {jv}/{jt}")

    print("\n" + "=" * 70)
    print("CLAIM 4: held-out susceptibility probe -- both models degrade, baseline less")
    print("=" * 70)
    probe = make_probe_held_out_susceptibility(n=400)
    ctx_p, ercp_p = build_ctx_and_ercp(probe["ctx"], probe["ercp"], T)
    Xp = torch.tensor(probe["X"], dtype=torch.float32)
    bp2 = baseline_rollout(base, Xp, ctx_p, ercp_p, T, start_t)
    jp2 = jepa_rollout(jepa, Xp, ctx_p, ercp_p, T, start_t)
    b_mae2 = ratchet_mae_at_K(bp2, Xp, start_t, 24)
    j_mae2 = ratchet_mae_at_K(jp2, Xp, start_t, 24)
    print(f"  in-dist baseline {b_mae:.4f} -> OOD baseline {b_mae2:.4f}  "
          f"({(b_mae2/b_mae-1)*100:+.0f}% relative degradation)")
    print(f"  in-dist TS-JEPA  {j_mae:.4f} -> OOD TS-JEPA  {j_mae2:.4f}  "
          f"({(j_mae2/j_mae-1)*100:+.0f}% relative degradation)")

    print("\n" + "=" * 70)
    print("CLAIM 5: noise-augmented JEPA narrows (but does not close) the noise-axis gap")
    print("=" * 70)
    for sigma in [0.10, 0.25]:
        b_seed, jp_seed, jd_seed = [], [], []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            Xn = add_sensor_noise(val["X"], sigma, rng)
            Xt = torch.tensor(Xn, dtype=torch.float32)
            bpn = baseline_rollout(base, Xt, ctx_feats, ercp_v, T, start_t)
            jpn = jepa_rollout(jepa, Xt, ctx_feats, ercp_v, T, start_t)
            jdn = jepa_rollout(jepa_dn, Xt, ctx_feats, ercp_v, T, start_t)
            b_seed.append(ratchet_mae_at_K(bpn, Xv, start_t, 24))
            jp_seed.append(ratchet_mae_at_K(jpn, Xv, start_t, 24))
            jd_seed.append(ratchet_mae_at_K(jdn, Xv, start_t, 24))
        b_gap = np.mean(jp_seed) - np.mean(b_seed)
        jd_gap = np.mean(jd_seed) - np.mean(b_seed)
        print(f"  sigma={sigma}: plain-JEPA gap over baseline = {b_gap:+.4f}, "
              f"denoise-aug gap = {jd_gap:+.4f}  "
              f"(gap narrowed: {'yes' if abs(jd_gap) < abs(b_gap) else 'no'}, "
              f"crossover: {'yes' if jd_gap < 0 else 'no'})")


if __name__ == "__main__":
    main()
