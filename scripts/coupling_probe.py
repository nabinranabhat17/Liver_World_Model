"""
Coupling-strength perturbation probe (D14 item 2 / D19): does either
model's LEARNED sensitivity of M's increment to F and C respect the
generator's true multiplicative F*C coupling shape?

Because M_inc = M_inc_raw * (0.3 + 1.7 * F_prev * C_prev) is
architecturally gated (ConstraintHead), and the generator's true
dM ~ susceptibility * 0.09 * F_prev * C_prev is also multiplicative in
F, C, the RATIO (dM/dF_prev) / (dM/dC_prev) should equal C_prev / F_prev
regardless of the hidden susceptibility value -- it's a common
multiplicative factor that cancels in the ratio. That makes the ratio a
susceptibility-free, directly testable invariant: exactly the probe D14
flagged as not having had time to run. No retraining needed -- pure
autograd through each model's own one-step forward, at real validation
trajectories.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import make_train_val, action_dim
from eval import build_ctx_and_ercp
from models.baseline import MonotoneStep
from models.jepa import TSJEPA

T = 60
F_IDX, D_IDX, S_IDX, P_IDX, A_IDX, C_IDX, M_IDX, FLARE_IDX = range(8)


def baseline_ratio(model, X, ctx_feats, ercp, ts):
    """dM/dF and dM/dC via autograd through one baseline step, batched
    over (patient, t) pairs (one t per row of X, given in `ts`)."""
    B = X.shape[0]
    idx = torch.arange(B)
    x_prev = X[idx, ts].clone().detach().requires_grad_(True)
    ctx_t1 = ctx_feats[idx, ts + 1]
    ercp_t1 = ercp[idx, ts + 1]
    x_next = model(x_prev, ctx_t1, ercp_t1)
    M_next = x_next[:, M_IDX]
    grad = torch.autograd.grad(M_next.sum(), x_prev)[0]
    return (grad[:, F_IDX].detach(), grad[:, C_IDX].detach(),
            x_prev[:, F_IDX].detach(), x_prev[:, C_IDX].detach())


def jepa_ratio(model, X, ctx_feats, ercp, ts):
    """Same, through the full encoder -> z_t -> decode chain, so the
    gradient captures whatever influence x_prev has on M_next through
    both the encoder's causal history window and the decode-anchor's
    direct ConstraintHead usage -- the fair end-to-end analogue of the
    baseline test above."""
    B = X.shape[0]
    dM_dF, dM_dC, Fp, Cp = [], [], [], []
    for b in range(B):
        t = int(ts[b])
        x_leaf = X[b, t].clone().detach().requires_grad_(True)
        hist = torch.cat([X[b, :t], x_leaf.unsqueeze(0)], dim=0).unsqueeze(0)
        ctx_hist = ctx_feats[b, :t + 1].unsqueeze(0)
        z_seq = model.online_encoder(hist, ctx_hist)
        z_t = z_seq[:, -1, :]
        ercp_t1 = ercp[b, t + 1:t + 2]
        x_next = model.decoder(z_t, x_leaf.unsqueeze(0), ercp_t1)
        M_next = x_next[0, M_IDX]
        grad = torch.autograd.grad(M_next, x_leaf)[0]
        dM_dF.append(grad[F_IDX].item()); dM_dC.append(grad[C_IDX].item())
        Fp.append(x_leaf[F_IDX].item()); Cp.append(x_leaf[C_IDX].item())
    return (torch.tensor(dM_dF), torch.tensor(dM_dC),
            torch.tensor(Fp), torch.tensor(Cp))


def jepa_decoder_only_ratio(model, X, ctx_feats, ercp, ts):
    """z held fixed (detached): isolates the ConstraintHead's hard F*C
    gate from any learned dependence, since LatentDecoder.net(z) never
    sees x_prev at all."""
    with torch.no_grad():
        z_seq_full = model.online_encoder(X, ctx_feats)
    dM_dF, dM_dC, Fp, Cp = [], [], [], []
    for b in range(X.shape[0]):
        t = int(ts[b])
        z_t = z_seq_full[b, t:t + 1, :].detach()
        x_leaf = X[b, t].clone().detach().requires_grad_(True)
        ercp_t1 = ercp[b, t + 1:t + 2]
        x_next = model.decoder(z_t, x_leaf.unsqueeze(0), ercp_t1)
        M_next = x_next[0, M_IDX]
        grad = torch.autograd.grad(M_next, x_leaf)[0]
        dM_dF.append(grad[F_IDX].item()); dM_dC.append(grad[C_IDX].item())
        Fp.append(x_leaf[F_IDX].item()); Cp.append(x_leaf[C_IDX].item())
    return (torch.tensor(dM_dF), torch.tensor(dM_dC),
            torch.tensor(Fp), torch.tensor(Cp))


def report(name, dM_dF, dM_dC, Fp, Cp, eps=1e-3):
    # dM_dF/dM_dC are gradients (O(1e-3)), not state values (O(0.1-1)) --
    # they need their own, much smaller floor so the mask threshold does
    # the "avoid dividing by ~0" job instead of an epsilon sized for Fp/Cp.
    mask = (Fp > 0.08) & (Cp > 0.08) & (dM_dC.abs() > 1e-5)
    dM_dF, dM_dC, Fp, Cp = dM_dF[mask], dM_dC[mask], Fp[mask], Cp[mask]
    pred_ratio = (dM_dF / dM_dC).numpy()
    true_ratio = (Cp / (Fp + eps)).numpy()
    corr = np.corrcoef(pred_ratio, true_ratio)[0, 1]
    log_err = np.abs(np.log(np.clip(pred_ratio, 1e-3, 1e3)) - np.log(np.clip(true_ratio, 1e-3, 1e3)))
    print(f"\n{name} (n={int(mask.sum())} samples, F_prev>0.08 & C_prev>0.08):")
    print(f"  dM/dF sign correct (>0): {(dM_dF.numpy() > 0).mean()*100:.0f}%")
    print(f"  dM/dC sign correct (>0): {(dM_dC.numpy() > 0).mean()*100:.0f}%")
    print(f"  ratio (dM/dF)/(dM/dC) vs true C_prev/F_prev -- correlation: {corr:.3f}")
    print(f"  median |log(pred_ratio) - log(true_ratio)|: {np.median(log_err):.3f}")


def main():
    torch.manual_seed(0)
    _, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    X = torch.tensor(val["X"], dtype=torch.float32)
    ctx_feats, ercp = build_ctx_and_ercp(val["ctx"], val["ercp"], T)

    base = MonotoneStep(ctx_dim=action_dim())
    base.load_state_dict(torch.load("checkpoints/baseline.pt")); base.eval()
    jepa = TSJEPA(action_dim=action_dim())
    jepa.load_state_dict(torch.load("checkpoints/jepa.pt")); jepa.eval()

    rng = np.random.default_rng(0)
    N = X.shape[0]
    ts = torch.tensor(rng.integers(15, T - 2, size=N))

    dM_dF, dM_dC, Fp, Cp = baseline_ratio(base, X, ctx_feats, ercp, ts)
    report("baseline", dM_dF, dM_dC, Fp, Cp)

    sub = torch.tensor(rng.choice(N, size=150, replace=False))
    Xs, ctxs, ercps, tss = X[sub], ctx_feats[sub], ercp[sub], ts[sub]

    dM_dF_j, dM_dC_j, Fp_j, Cp_j = jepa_ratio(jepa, Xs, ctxs, ercps, tss)
    report("TS-JEPA (encoder+decoder, full path)", dM_dF_j, dM_dC_j, Fp_j, Cp_j)

    dM_dF_d, dM_dC_d, Fp_d, Cp_d = jepa_decoder_only_ratio(jepa, Xs, ctxs, ercps, tss)
    report("TS-JEPA (decoder only, z held fixed)", dM_dF_d, dM_dC_d, Fp_d, Cp_d)


if __name__ == "__main__":
    main()
