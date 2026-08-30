"""
Train TS-JEPA: latent prediction loss (vs EMA target encoder) + VICReg
(prevents representation collapse) + decode-anchor loss (keeps the latent
decodable to a constrained, on-manifold state -- without this the latent
could satisfy the prediction+VICReg losses while encoding something that
doesn't decode back to anything clinically sensible).

VICReg terms (Bardes et al. 2022, adapted): computed on the ONLINE
encoder's z's across a batch of (different patient, different time) pairs.
  variance: hinge loss pushing std of each latent dim above a floor (1.0)
            -- directly penalises collapse to a constant.
  covariance: off-diagonal covariance pushed toward 0 -- spreads
            information across dimensions instead of one dominant axis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from data import make_train_val, make_context_features, action_dim
from models.jepa import TSJEPA

torch.manual_seed(0)
np.random.seed(0)

T = 60
K = 8              # multistep prediction horizon during training
LATENT_DIM = 16


def build_ctx_and_ercp(ctx_np, ercp_np, T):
    N = ctx_np.shape[0]
    ctx_feats = np.stack([make_context_features(ctx_np, t, T) for t in range(T)], axis=1)
    return torch.tensor(ctx_feats, dtype=torch.float32), torch.tensor(ercp_np, dtype=torch.float32)


def vicreg_loss(z, var_floor=1.0, eps=1e-4):
    """z: (N, D) flattened across batch and time."""
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.relu(var_floor - std).mean()

    N, D = z.shape
    cov = (z.T @ z) / (N - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = (off_diag ** 2).sum() / D
    return var_loss, cov_loss


def effective_rank(z):
    """Effective rank of the latent covariance -- collapse diagnostic.
    Effective rank near 1 = collapsed onto (almost) one direction.
    Effective rank near D = using the full latent capacity."""
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (z.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = eigvals / eigvals.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


def train_one_seed(seed, n_epochs=80, batch_size=128, jepa_w=1.0, var_w=15.0, cov_w=1.0, dec_w=5.0,
                    encoder_factory=None, decoder_factory=None, anneal_vicreg=False):
    torch.manual_seed(seed)
    train, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    Xtr = torch.tensor(train["X"], dtype=torch.float32)
    Xva = torch.tensor(val["X"], dtype=torch.float32)
    ctx_tr, ercp_tr = build_ctx_and_ercp(train["ctx"], train["ercp"], T)
    ctx_va, ercp_va = build_ctx_and_ercp(val["ctx"], val["ercp"], T)

    encoder = encoder_factory() if encoder_factory is not None else None
    decoder = decoder_factory() if decoder_factory is not None else None
    model = TSJEPA(action_dim=action_dim(), latent_dim=LATENT_DIM, encoder=encoder, decoder=decoder)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    # once effective rank first reaches this, VICReg weights relax toward a
    # floor instead of continuing to push rank up indefinitely -- tests the
    # D11 diagnosis (see train_jepa_anneal.py)
    erank_target = 5.3
    var_w_floor, cov_w_floor = var_w * 0.4, cov_w * 0.4
    relaxed = False

    n_train_pts = Xtr.shape[0]

    for epoch in range(n_epochs):
        if anneal_vicreg:
            if not relaxed:
                with torch.no_grad():
                    z_check = model.online_encoder(Xva, ctx_va)
                    erank_now = effective_rank(z_check.reshape(-1, LATENT_DIM))
                if erank_now >= erank_target:
                    relaxed = True
            cur_var_w = var_w_floor if relaxed else var_w
            cur_cov_w = cov_w_floor if relaxed else cov_w
        else:
            cur_var_w, cur_cov_w = var_w, cov_w

        perm = torch.randperm(n_train_pts)
        tot = {"jepa": 0.0, "var": 0.0, "cov": 0.0, "dec": 0.0}
        n_batches = 0
        for i in range(0, n_train_pts, batch_size):
            idx = perm[i:i + batch_size]
            Xb, ctxb, ercpb = Xtr[idx], ctx_tr[idx], ercp_tr[idx]
            B = Xb.shape[0]

            opt.zero_grad()
            z_online_seq = model.online_encoder(Xb, ctxb)             # (B,T,D)
            with torch.no_grad():
                z_target_seq = model.target_encoder(Xb, ctxb)         # (B,T,D)

            start = torch.randint(0, T - K - 1, (B,))
            z_t = z_online_seq[torch.arange(B), start]                # (B,D)
            action_future = torch.stack([ctxb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            ercp_future = torch.stack([ercpb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            x_anchor = Xb[torch.arange(B), start]                     # (B,8) real state at t

            z_preds = model.predictor.rollout(z_t, action_future)     # list of K (B,D)
            jepa_loss = 0.0
            for s in range(K):
                target = z_target_seq[torch.arange(B), start + s + 1].detach()
                jepa_loss = jepa_loss + nn.functional.mse_loss(z_preds[s], target)
            jepa_loss = jepa_loss / K

            var_loss, cov_loss = vicreg_loss(z_online_seq.reshape(-1, LATENT_DIM))

            x_hats = model.decoder.decode_chain(z_preds, x_anchor, ercp_future)  # (B,K,8)
            x_true_future = torch.stack([Xb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            dec_loss = nn.functional.mse_loss(x_hats, x_true_future)

            loss = jepa_w * jepa_loss + cur_var_w * var_loss + cur_cov_w * cov_loss + dec_w * dec_loss
            loss.backward()
            opt.step()
            model.update_target()

            tot["jepa"] += jepa_loss.item()
            tot["var"] += var_loss.item()
            tot["cov"] += cov_loss.item()
            tot["dec"] += dec_loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                z_val = model.online_encoder(Xva, ctx_va)
                erank = effective_rank(z_val.reshape(-1, LATENT_DIM))
            anneal_note = f"  var_w {cur_var_w:.1f} cov_w {cur_cov_w:.1f} relaxed {relaxed}" if anneal_vicreg else ""
            print(f"[seed {seed}] epoch {epoch:3d}  "
                  f"jepa {tot['jepa']/n_batches:.5f}  var {tot['var']/n_batches:.5f}  "
                  f"cov {tot['cov']/n_batches:.5f}  dec {tot['dec']/n_batches:.5f}  "
                  f"eff_rank {erank:.2f}/{LATENT_DIM}{anneal_note}")

    return model


def main():
    model = train_one_seed(seed=0)
    torch.save(model.state_dict(), "checkpoints/jepa.pt")
    print("saved checkpoints/jepa.pt")


if __name__ == "__main__":
    main()
