"""
Experiment: does training-time noise augmentation give TS-JEPA an actual
denoising advantage over the baseline, or is the plain JEPA (train_jepa.py)
just structurally incapable of one?

Hypothesis (from compare.py's honest negative result): the plain JEPA's
decode-anchor loss supervises decode(z_t) to match the (clean, at train
time) x_t directly, giving the encoder no incentive to average out noise --
it only ever sees clean inputs, so it has no reason to be robust to dirty
ones. Fix: corrupt the ENCODER's input with Gaussian noise during training
(the online encoder only ever sees noisy x), while keeping the JEPA
prediction target (from the target encoder) and the decode supervision
target both CLEAN. This forces the online encoder to map noisy
observations toward the same latent a clean observation would produce --
i.e. to denoise, not just to compress.

This is a real experiment with an uncertain outcome, not a foregone
conclusion -- see DECISIONS.md for the result and the honest interpretation
either way.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from data import make_train_val, make_context_features, action_dim
from models.jepa import TSJEPA
from train_jepa import build_ctx_and_ercp, vicreg_loss, effective_rank, T, K, LATENT_DIM

torch.manual_seed(0)
np.random.seed(0)


def train_denoised(seed=0, n_epochs=80, batch_size=128, jepa_w=1.0, var_w=15.0, cov_w=1.0,
                    dec_w=5.0, noise_std=0.06):
    torch.manual_seed(seed)
    train, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    Xtr = torch.tensor(train["X"], dtype=torch.float32)
    ctx_tr, ercp_tr = build_ctx_and_ercp(train["ctx"], train["ercp"], T)

    model = TSJEPA(action_dim=action_dim(), latent_dim=LATENT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n_train_pts = Xtr.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train_pts)
        tot = {"jepa": 0.0, "dec": 0.0}
        n_batches = 0
        for i in range(0, n_train_pts, batch_size):
            idx = perm[i:i + batch_size]
            Xb, ctxb, ercpb = Xtr[idx], ctx_tr[idx], ercp_tr[idx]
            B = Xb.shape[0]

            # online encoder sees a NOISY version; target encoder + decode
            # supervision both stay CLEAN.
            Xb_noisy = Xb + torch.randn_like(Xb) * noise_std
            Xb_noisy = torch.clamp(Xb_noisy, min=torch.tensor([0,0,0,0,0,0,0,0.]),
                                              max=torch.tensor([1,1,1,1,1,1,2,1.]))

            opt.zero_grad()
            z_online_seq = model.online_encoder(Xb_noisy, ctxb)
            with torch.no_grad():
                z_target_seq = model.target_encoder(Xb, ctxb)  # clean target

            start = torch.randint(0, T - K - 1, (B,))
            z_t = z_online_seq[torch.arange(B), start]
            action_future = torch.stack([ctxb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            ercp_future = torch.stack([ercpb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            # anchor uses the NOISY current reading (that's the realistic deployment case:
            # you only ever observe a noisy sensor), decode targets stay clean.
            x_anchor = Xb_noisy[torch.arange(B), start]

            z_preds = model.predictor.rollout(z_t, action_future)
            jepa_loss = sum(nn.functional.mse_loss(z_preds[s], z_target_seq[torch.arange(B), start + s + 1].detach())
                             for s in range(K)) / K
            var_loss, cov_loss = vicreg_loss(z_online_seq.reshape(-1, LATENT_DIM))

            x_hats = model.decoder.decode_chain(z_preds, x_anchor, ercp_future)
            x_true_future = torch.stack([Xb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)  # clean target
            dec_loss = nn.functional.mse_loss(x_hats, x_true_future)

            loss = jepa_w * jepa_loss + var_w * var_loss + cov_w * cov_loss + dec_w * dec_loss
            loss.backward()
            opt.step()
            model.update_target()

            tot["jepa"] += jepa_loss.item()
            tot["dec"] += dec_loss.item()
            n_batches += 1

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"[denoise-aug] epoch {epoch:3d}  jepa {tot['jepa']/n_batches:.5f}  dec {tot['dec']/n_batches:.5f}")

    return model


if __name__ == "__main__":
    model = train_denoised()
    torch.save(model.state_dict(), "checkpoints/jepa_denoise.pt")
    print("saved checkpoints/jepa_denoise.pt")
