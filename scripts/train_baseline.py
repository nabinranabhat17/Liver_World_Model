"""
Train the baseline MonotoneStep model.

Two loss terms:
  1. one-step teacher-forced MSE: predict x_{t+1} from true x_t. Easy, fast
     to converge, but doesn't penalise compounding rollout error.
  2. multistep free-rollout MSE (k=8 steps): start from a true x_t, roll
     forward WITHOUT teacher forcing for k steps, compare to ground truth.
     This is what actually gets tested at eval time (free rollout to K=24+),
     so training on it closes the train/eval mismatch.

We anneal from pure one-step to a mix, since pure multistep from
random-init weights is unstable (early rollouts diverge before the model
has learned anything).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from data import make_train_val, make_context_features, action_dim
from models.baseline import MonotoneStep

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"
T = 60


def build_ctx_and_ercp(ctx_np, ercp_np, T):
    """Precompute per-step context features and ERCP flags as tensors."""
    N = ctx_np.shape[0]
    ctx_feats = np.stack([make_context_features(ctx_np, t, T) for t in range(T)], axis=1)  # (N,T,8)
    return torch.tensor(ctx_feats, dtype=torch.float32), torch.tensor(ercp_np, dtype=torch.float32)


def one_step_loss(model, X, ctx_feats, ercp):
    B, T, _ = X.shape
    t_idx = torch.randint(0, T - 1, (B,))
    x_t = X[torch.arange(B), t_idx]
    x_next_true = X[torch.arange(B), t_idx + 1]
    ctx_t1 = ctx_feats[torch.arange(B), t_idx + 1]
    ercp_t1 = ercp[torch.arange(B), t_idx + 1]
    x_next_pred = model(x_t, ctx_t1, ercp_t1)
    return nn.functional.mse_loss(x_next_pred, x_next_true)


def multistep_loss(model, X, ctx_feats, ercp, k=8):
    B, T, _ = X.shape
    start = torch.randint(0, T - k - 1, (B,))
    x = X[torch.arange(B), start]
    loss = 0.0
    for step in range(1, k + 1):
        idx = start + step
        ctx_t = ctx_feats[torch.arange(B), idx]
        ercp_t = ercp[torch.arange(B), idx]
        x = model(x, ctx_t, ercp_t)
        x_true = X[torch.arange(B), idx]
        loss = loss + nn.functional.mse_loss(x, x_true)
    return loss / k


def main():
    train, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    Xtr = torch.tensor(train["X"], dtype=torch.float32)
    Xva = torch.tensor(val["X"], dtype=torch.float32)
    ctx_tr, ercp_tr = build_ctx_and_ercp(train["ctx"], train["ercp"], T)
    ctx_va, ercp_va = build_ctx_and_ercp(val["ctx"], val["ercp"], T)

    model = MonotoneStep(ctx_dim=action_dim())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    n_epochs = 60
    batch_size = 128
    n_train_pts = Xtr.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train_pts)
        # anneal multistep weight in over first 20 epochs
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

    torch.save(model.state_dict(), "checkpoints/baseline.pt")
    print("saved checkpoints/baseline.pt")


if __name__ == "__main__":
    main()
