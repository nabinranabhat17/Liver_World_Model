"""
Train MonotoneStepCoupled (models/baseline_coupled.py) -- the structural
counterpart to scripts/train_baseline_jacobian.py's soft penalty (see
DECISIONS.md D9). Same one-step + annealed-multistep loss and schedule
as scripts/train_baseline.py, imported directly rather than
reimplemented, so any accuracy delta vs. baseline.pt / baseline_jacobian.pt
is attributable to the architecture alone. Unlike train_baseline_jacobian.py,
there is no auxiliary loss term: the sign guarantee here is structural
(see scripts/test_invariants.py's test_coupled_monotonicity_random_weights,
which must pass under random weights before this script is ever run), so
nothing needs to be added to the training objective to obtain it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from data import make_train_val, action_dim
from models.baseline_coupled import MonotoneStepCoupled
from train_baseline import build_ctx_and_ercp, one_step_loss, multistep_loss

torch.manual_seed(0)
np.random.seed(0)

T = 60


def main():
    train, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    Xtr = torch.tensor(train["X"], dtype=torch.float32)
    Xva = torch.tensor(val["X"], dtype=torch.float32)
    ctx_tr, ercp_tr = build_ctx_and_ercp(train["ctx"], train["ercp"], T)
    ctx_va, ercp_va = build_ctx_and_ercp(val["ctx"], val["ercp"], T)

    model = MonotoneStepCoupled(ctx_dim=action_dim())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    n_epochs = 60
    batch_size = 128
    n_train_pts = Xtr.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train_pts)
        multi_weight = min(1.0, epoch / 20.0)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train_pts, batch_size):
            idx = perm[i:i + batch_size]
            Xb, ctxb, ercpb = Xtr[idx], ctx_tr[idx], ercp_tr[idx]
            opt.zero_grad()
            l1 = one_step_loss(model, Xb, ctxb, ercpb)
            l2 = multistep_loss(model, Xb, ctxb, ercpb, k=8)
            loss = l1 + multi_weight * l2
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                val_l1 = one_step_loss(model, Xva, ctx_va, ercp_va).item()
                val_l2 = multistep_loss(model, Xva, ctx_va, ercp_va, k=8).item()
            print(f"epoch {epoch:3d}  train_loss {epoch_loss/n_batches:.5f}  "
                  f"val_1step {val_l1:.5f}  val_8step {val_l2:.5f}  multi_w {multi_weight:.2f}")

    torch.save(model.state_dict(), "checkpoints/baseline_monotonic.pt")
    print("saved checkpoints/baseline_monotonic.pt")


if __name__ == "__main__":
    main()
