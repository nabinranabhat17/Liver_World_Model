"""
TS-JEPA trained with an added counterfactual-consistency loss targeting
D18: the plain JEPA's implied treatment-timing effect is uncorrelated to
inversely correlated with the generator's true matched-pair effect
(counterfactual.py). Diagnosis there was that nothing in training rewards
getting the marginal effect of the on-treatment action dimension right,
inside a 16-D latent shared with seven other action dimensions.

The fix tested here: build a small, fixed set of matched (factual,
counterfactual) patient pairs -- identical exogenous randomness, treated
responders only, differing only in udca_start -- using the exact same
construction as counterfactual.py's own make_matched_pair. Once per
epoch, roll the model forward TWICE from an identical anchor state (once
under the factual treatment timing, once under the counterfactual one,
exactly like counterfactual.py's own probe) and add a hinge loss pushing
the model's own implied effect on F at +24 months to have the same SIGN
as the generator's true matched-pair effect -- not the exact magnitude,
which the model has no way to know without being told the hidden
susceptibility, but the direction, which is stably determined by
responder status and treatment timing alone.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import torch
import torch.nn as nn
from data import make_train_val, make_context_features, action_dim
from generator import simulate, _sample_patient, T_DEFAULT
from models.jepa import TSJEPA, LATENT_DIM
from train_jepa import build_ctx_and_ercp, vicreg_loss, effective_rank

torch.manual_seed(0)
np.random.seed(0)

T = T_DEFAULT
K = 8
HORIZON = 24
SHIFT = 6
N_PAIRS = 64


def build_cf_pairs(n_pairs=N_PAIRS, shift_months=SHIFT, seed_base=90_000):
    """Matched (factual, counterfactual) responder pairs, built exactly as
    counterfactual.make_matched_pair -- kept to responders only, since
    non-responders have no true treatment effect to train toward."""
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 20:
        tries += 1
        base_seed = seed_base + tries
        sample_rng = np.random.default_rng(base_seed)
        patient = _sample_patient(sample_rng, (0.4, 1.0))
        if patient.udca_start < 0:
            patient.udca_start = int(sample_rng.integers(10, 30))
        if not patient.responder or patient.udca_start < shift_months + 1:
            continue

        patient_cf = copy.deepcopy(patient)
        patient_cf.udca_start = max(0, patient.udca_start - shift_months)

        rng_f = np.random.default_rng(base_seed + 500000)
        x_factual, _ = simulate(patient, T, rng_f)
        rng_cf = np.random.default_rng(base_seed + 500000)
        x_cf, _ = simulate(patient_cf, T, rng_cf)

        anchor_t = max(0, patient_cf.udca_start - 1)
        if anchor_t + HORIZON >= T:
            continue
        true_effect = float(x_cf[anchor_t + HORIZON, 0] - x_factual[anchor_t + HORIZON, 0])

        def ctx_row(p):
            return np.array([p.disease_class, p.age / 100.0, p.sex, p.responder,
                              p.udca_start / T, 1.0])

        ctx_f = ctx_row(patient)
        ctx_cf = ctx_row(patient_cf)
        ctx_feats_f = np.stack([make_context_features(ctx_f[None, :], t, T) for t in range(T)], axis=1)[0]
        ctx_feats_cf = np.stack([make_context_features(ctx_cf[None, :], t, T) for t in range(T)], axis=1)[0]

        pairs.append(dict(
            x_factual=torch.tensor(x_factual, dtype=torch.float32),
            ctx_feats_f=torch.tensor(ctx_feats_f, dtype=torch.float32),
            ctx_feats_cf=torch.tensor(ctx_feats_cf, dtype=torch.float32),
            ercp_flags=torch.zeros(T, dtype=torch.float32),
            anchor_t=anchor_t,
            true_effect=true_effect,
        ))
    return pairs


def counterfactual_consistency_loss(model, cf_pairs, horizon=HORIZON, margin=0.0015):
    hinge_terms = []
    for item in cf_pairs:
        t0 = item["anchor_t"]
        x_hist = item["x_factual"][:t0 + 1].unsqueeze(0)
        ctx_hist_f = item["ctx_feats_f"][:t0 + 1].unsqueeze(0)
        z_seq = model.online_encoder(x_hist, ctx_hist_f)
        z_t = z_seq[:, -1, :]
        x_anchor = item["x_factual"][t0:t0 + 1]

        action_future_f = item["ctx_feats_f"][t0 + 1:t0 + 1 + horizon].unsqueeze(0)
        action_future_cf = item["ctx_feats_cf"][t0 + 1:t0 + 1 + horizon].unsqueeze(0)
        ercp_future = item["ercp_flags"][t0 + 1:t0 + 1 + horizon].unsqueeze(0)

        z_preds_f = model.predictor.rollout(z_t, action_future_f)
        x_hats_f = model.decoder.decode_chain(z_preds_f, x_anchor, ercp_future)
        z_preds_cf = model.predictor.rollout(z_t, action_future_cf)
        x_hats_cf = model.decoder.decode_chain(z_preds_cf, x_anchor, ercp_future)

        implied_effect = x_hats_cf[0, horizon - 1, 0] - x_hats_f[0, horizon - 1, 0]
        true_sign = -1.0 if item["true_effect"] < 0 else 1.0
        hinge_terms.append(torch.relu(margin - true_sign * implied_effect))
    return torch.stack(hinge_terms).mean()


def train(seed=0, n_epochs=80, batch_size=128, jepa_w=1.0, var_w=15.0, cov_w=1.0, dec_w=5.0, cf_w=8.0):
    torch.manual_seed(seed)
    train_data, val = make_train_val(n_train=1500, n_val=400, T=T, seed=0)
    Xtr = torch.tensor(train_data["X"], dtype=torch.float32)
    Xva = torch.tensor(val["X"], dtype=torch.float32)
    ctx_tr, ercp_tr = build_ctx_and_ercp(train_data["ctx"], train_data["ercp"], T)
    ctx_va, ercp_va = build_ctx_and_ercp(val["ctx"], val["ercp"], T)

    cf_pairs = build_cf_pairs()
    print(f"built {len(cf_pairs)} matched counterfactual pairs "
          f"(mean true effect {np.mean([p['true_effect'] for p in cf_pairs]):+.4f})")

    model = TSJEPA(action_dim=action_dim(), latent_dim=LATENT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n_train_pts = Xtr.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train_pts)
        tot = {"jepa": 0.0, "var": 0.0, "cov": 0.0, "dec": 0.0}
        n_batches = 0
        for i in range(0, n_train_pts, batch_size):
            idx = perm[i:i + batch_size]
            Xb, ctxb, ercpb = Xtr[idx], ctx_tr[idx], ercp_tr[idx]
            B = Xb.shape[0]

            opt.zero_grad()
            z_online_seq = model.online_encoder(Xb, ctxb)
            with torch.no_grad():
                z_target_seq = model.target_encoder(Xb, ctxb)

            start = torch.randint(0, T - K - 1, (B,))
            z_t = z_online_seq[torch.arange(B), start]
            action_future = torch.stack([ctxb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            ercp_future = torch.stack([ercpb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            x_anchor = Xb[torch.arange(B), start]

            z_preds = model.predictor.rollout(z_t, action_future)
            jepa_loss = 0.0
            for s in range(K):
                target = z_target_seq[torch.arange(B), start + s + 1].detach()
                jepa_loss = jepa_loss + nn.functional.mse_loss(z_preds[s], target)
            jepa_loss = jepa_loss / K

            var_loss, cov_loss = vicreg_loss(z_online_seq.reshape(-1, LATENT_DIM))

            x_hats = model.decoder.decode_chain(z_preds, x_anchor, ercp_future)
            x_true_future = torch.stack([Xb[torch.arange(B), start + s + 1] for s in range(K)], dim=1)
            dec_loss = nn.functional.mse_loss(x_hats, x_true_future)

            loss = jepa_w * jepa_loss + var_w * var_loss + cov_w * cov_loss + dec_w * dec_loss
            loss.backward()
            opt.step()
            model.update_target()

            tot["jepa"] += jepa_loss.item()
            tot["var"] += var_loss.item()
            tot["cov"] += cov_loss.item()
            tot["dec"] += dec_loss.item()
            n_batches += 1

        # once per epoch: counterfactual-consistency step, its own backward
        opt.zero_grad()
        cf_loss = counterfactual_consistency_loss(model, cf_pairs)
        (cf_w * cf_loss).backward()
        opt.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                z_val = model.online_encoder(Xva, ctx_va)
                erank = effective_rank(z_val.reshape(-1, LATENT_DIM))
            print(f"[seed {seed}] epoch {epoch:3d}  "
                  f"jepa {tot['jepa']/n_batches:.5f}  var {tot['var']/n_batches:.5f}  "
                  f"cov {tot['cov']/n_batches:.5f}  dec {tot['dec']/n_batches:.5f}  "
                  f"cf_hinge {cf_loss.item():.5f}  eff_rank {erank:.2f}/{LATENT_DIM}")

    return model


if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "checkpoints/jepa_counterfactual.pt")
    print("saved checkpoints/jepa_counterfactual.pt")
