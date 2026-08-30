"""
Baseline training plus a Jacobian-sign auxiliary loss targeting D9:
gradient attribution through the plain baseline (`explain.py`) found
WRONG-SIGNED sensitivities d(F_next)/dA and d(F_next)/dC on the large
majority of samples, even though the generator's true structure has F's
increment driven POSITIVELY by both (dF ~ susceptibility * (A + C),
susceptibility > 0, so always >= 0 in A and C). The comparison against
the structural fix (models/baseline_coupled.py) then found D has the
exact same problem for its own drivers S, A (dD ~ susceptibility *
(0.7*S + 0.3*A)) -- in fact worse: d(D_next)/dS was wrong-signed on
100% of samples, never previously diagnosed. Both are covered here so
the two fixes (soft penalty vs. structural) are compared on equal
footing.

The fix tested here: a lightweight double-backward penalty, computed on
a fresh minibatch every training step, that supervises the SIGN (not
magnitude -- magnitude depends on the hidden susceptibility scalar,
which the model can't know) of d(F_next)/dA, d(F_next)/dC, d(D_next)/dS,
and d(D_next)/dA, added on top of the existing one-step +
annealed-multistep loss with weight `jac_w`. Everything else
(architecture, schedule, optimizer) is identical to `train_baseline.py`,
imported directly, so any difference in the resulting attribution is
attributable to this one loss term.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import make_train_val, action_dim
from models.baseline import MonotoneStep
from train_baseline import build_ctx_and_ercp, one_step_loss, multistep_loss

torch.manual_seed(0)
np.random.seed(0)

T = 60
F_IDX, D_IDX, S_IDX, A_IDX, C_IDX = 0, 1, 2, 4, 5


def jacobian_sign_loss(model, X, ctx_feats, ercp):
    B, Tn, _ = X.shape
    t_idx = torch.randint(0, Tn - 1, (B,))
    x_t = X[torch.arange(B), t_idx].clone().detach().requires_grad_(True)
    ctx_t1 = ctx_feats[torch.arange(B), t_idx + 1]
    ercp_t1 = ercp[torch.arange(B), t_idx + 1]
    x_next = model(x_t, ctx_t1, ercp_t1)
    F_next, D_next = x_next[:, F_IDX], x_next[:, D_IDX]
    grad_F = torch.autograd.grad(F_next.sum(), x_t, retain_graph=True, create_graph=True)[0]
    grad_D = torch.autograd.grad(D_next.sum(), x_t, create_graph=True)[0]
    dF_dA, dF_dC = grad_F[:, A_IDX], grad_F[:, C_IDX]
    dD_dS, dD_dA = grad_D[:, S_IDX], grad_D[:, A_IDX]
    return (torch.relu(-dF_dA).mean() + torch.relu(-dF_dC).mean()
            + torch.relu(-dD_dS).mean() + torch.relu(-dD_dA).mean())


def main(jac_w=1.0, n_epochs=60, batch_size=128):
    train, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    Xtr = torch.tensor(train["X"], dtype=torch.float32)
    Xva = torch.tensor(val["X"], dtype=torch.float32)
    ctx_tr, ercp_tr = build_ctx_and_ercp(train["ctx"], train["ercp"], T)
    ctx_va, ercp_va = build_ctx_and_ercp(val["ctx"], val["ercp"], T)

    model = MonotoneStep(ctx_dim=action_dim())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n_train_pts = Xtr.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train_pts)
        multi_weight = min(1.0, epoch / 20.0)
        epoch_loss, epoch_jac, n_batches = 0.0, 0.0, 0
        for i in range(0, n_train_pts, batch_size):
            idx = perm[i:i + batch_size]
            Xb, ctxb, ercpb = Xtr[idx], ctx_tr[idx], ercp_tr[idx]
            opt.zero_grad()
            l1 = one_step_loss(model, Xb, ctxb, ercpb)
            l2 = multistep_loss(model, Xb, ctxb, ercpb, k=8)
            l3 = jacobian_sign_loss(model, Xb, ctxb, ercpb)
            loss = l1 + multi_weight * l2 + jac_w * l3
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            epoch_jac += l3.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                val_l1 = one_step_loss(model, Xva, ctx_va, ercp_va).item()
                val_l2 = multistep_loss(model, Xva, ctx_va, ercp_va, k=8).item()
            print(f"epoch {epoch:3d}  train_loss {epoch_loss/n_batches:.5f}  "
                  f"val_1step {val_l1:.5f}  val_8step {val_l2:.5f}  "
                  f"jac_penalty {epoch_jac/n_batches:.5f}  multi_w {multi_weight:.2f}")

    torch.save(model.state_dict(), "checkpoints/baseline_jacobian.pt")
    print("saved checkpoints/baseline_jacobian.pt")


if __name__ == "__main__":
    main()
