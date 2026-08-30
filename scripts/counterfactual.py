"""
Counterfactual rollout: "what if UDCA had started 6 months earlier?" --
validated against a matched re-run of the generator, per the assignment's
optional "toe into the out-of-scope dimension" seed.

Matched-pair construction: generator.simulate() consumes its rng in a
fixed sequence (flare draw, then A-noise, then C-noise, every month,
regardless of treatment status) -- treat_gain only rescales a target
value, it never changes which rng calls happen or in what order. So
re-seeding an independent rng with the SAME seed for two calls to
simulate() on two Patient copies that differ ONLY in udca_start produces
a true matched pair: identical exogenous randomness (flare timing, noise
realizations), differing only in the one intervention we changed. This is
what makes "true_effect = counterfactual_trajectory - factual_trajectory"
an honest ground-truth causal effect, not a confound of separately-drawn
noise.

What this validates, precisely: whether the model's OWN implied response
to changing the on_treatment action feature -- computed the same way,
by rolling out twice from an identical anchor state with only the action
sequence's treatment timing changed -- tracks the generator's true,
matched-pair effect.

What this does NOT validate (stated per the assignment's own scoping of
this as out-of-scope-for-context-only): this is not a test of the model
under REFERRAL-STRATIFIED confounding ("this patient reached us, therefore
they are sick"), because our training cohort is not referral-biased in
the first place -- treatment assignment here is exogenous (random
udca_start, independent of disease severity) by construction of
generator.py. A model could pass this check while still failing to
respect interventional semantics under a referral-biased cohort; that is
a materially harder and different validation, correctly scoped out.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generator import simulate, _sample_patient, T_DEFAULT
from data import make_context_features, action_dim
from models.baseline import MonotoneStep
from models.jepa import TSJEPA
from compare import baseline_rollout, jepa_rollout

T = T_DEFAULT


def make_matched_pair(base_seed, shift_months=6, susceptibility_range=(0.4, 1.0)):
    """One patient, simulated twice with identical exogenous randomness:
    factual (real udca_start) and counterfactual (udca_start shifted
    `shift_months` earlier, clamped at 0)."""
    sample_rng = np.random.default_rng(base_seed)
    patient = _sample_patient(sample_rng, susceptibility_range)
    if patient.udca_start < 0:
        patient.udca_start = int(sample_rng.integers(10, 30))  # force a treated patient for this probe

    patient_cf = copy.deepcopy(patient)
    patient_cf.udca_start = max(0, patient.udca_start - shift_months)

    rng_factual = np.random.default_rng(base_seed + 500000)
    x_factual, ercp_f = simulate(patient, T, rng_factual)
    rng_cf = np.random.default_rng(base_seed + 500000)  # SAME seed -> matched exogenous noise
    x_cf, ercp_cf = simulate(patient_cf, T, rng_cf)

    return patient, patient_cf, x_factual, x_cf


def context_row(patient, T):
    has_udca = 1.0
    return np.array([patient.disease_class, patient.age / 100.0, patient.sex, patient.responder,
                      patient.udca_start / T, has_udca])


def model_rollout_from_anchor(model, x_anchor, ctx_row, ercp_flags, T, anchor_t, is_jepa=False, history=None):
    """Roll a model forward from anchor_t using a GIVEN context timeline
    (so we can feed it the factual OR counterfactual treatment schedule)."""
    ctx_feats = np.stack([make_context_features(ctx_row[None, :], t, T) for t in range(T)], axis=1)[0]
    ctx_feats_t = torch.tensor(ctx_feats, dtype=torch.float32)
    ercp_t = torch.tensor(ercp_flags, dtype=torch.float32)

    if not is_jepa:
        x = torch.tensor(x_anchor, dtype=torch.float32).unsqueeze(0)
        preds = [x_anchor.copy()]
        for t in range(anchor_t + 1, T):
            x = model(x, ctx_feats_t[t:t+1, :], ercp_t[t:t+1])
            preds.append(x.detach().numpy()[0])
        return np.stack(preds, axis=0)  # (T-anchor_t, 8)
    else:
        # history: (anchor_t+1, 8) real observed history up to and including anchor_t
        Xh = torch.tensor(history[None, :anchor_t+1, :], dtype=torch.float32)
        ctxh = ctx_feats_t[None, :anchor_t+1, :]
        z_seq = model.online_encoder(Xh, ctxh)
        z_t = z_seq[:, -1, :]
        action_future = ctx_feats_t[None, anchor_t+1:T, :]
        ercp_future = ercp_t[None, anchor_t+1:T]
        z_preds = model.predictor.rollout(z_t, action_future)
        x_anchor_t = torch.tensor(x_anchor, dtype=torch.float32).unsqueeze(0)
        x_hats = model.decoder.decode_chain(z_preds, x_anchor_t, ercp_future)
        out = np.concatenate([x_anchor[None, :], x_hats.detach().numpy()[0]], axis=0)
        return out


def run_probe(n_patients=100, shift_months=6):
    base = MonotoneStep(ctx_dim=action_dim())
    base.load_state_dict(torch.load("checkpoints/baseline.pt")); base.eval()
    jepa = TSJEPA(action_dim=action_dim())
    jepa.load_state_dict(torch.load("checkpoints/jepa.pt")); jepa.eval()

    true_effects, base_effects, jepa_effects = [], [], []
    HORIZON = 24  # months past anchor to evaluate the effect at

    for i in range(n_patients):
        patient, patient_cf, x_factual, x_cf = make_matched_pair(base_seed=10_000 + i, shift_months=shift_months)
        anchor_t = max(0, patient_cf.udca_start - 1)  # last month before EITHER timeline has started treatment
        if anchor_t + HORIZON >= T or patient.udca_start < shift_months + 1:
            continue

        # ground truth: the generator's own matched-pair effect on F at anchor_t+HORIZON
        true_effect = x_cf[anchor_t + HORIZON, 0] - x_factual[anchor_t + HORIZON, 0]  # effect on F

        ctx_factual = context_row(patient, T)
        ctx_cf = context_row(patient_cf, T)
        ercp_flags = np.zeros(T)  # ERCP only relevant for PSC; keep both arms' ERCP off for a clean isolated effect
        x_anchor = x_factual[anchor_t]  # identical in both arms by construction up to anchor_t

        # baseline: two rollouts from the same anchor, two different context timelines
        roll_f = model_rollout_from_anchor(base, x_anchor, ctx_factual, ercp_flags, T, anchor_t)
        roll_cf = model_rollout_from_anchor(base, x_anchor, ctx_cf, ercp_flags, T, anchor_t)
        base_effect = roll_cf[HORIZON, 0] - roll_f[HORIZON, 0]

        roll_f_j = model_rollout_from_anchor(jepa, x_anchor, ctx_factual, ercp_flags, T, anchor_t,
                                              is_jepa=True, history=x_factual)
        roll_cf_j = model_rollout_from_anchor(jepa, x_anchor, ctx_cf, ercp_flags, T, anchor_t,
                                               is_jepa=True, history=x_factual)
        jepa_effect = roll_cf_j[HORIZON, 0] - roll_f_j[HORIZON, 0]

        true_effects.append(true_effect)
        base_effects.append(base_effect)
        jepa_effects.append(jepa_effect)

    true_effects = np.array(true_effects)
    base_effects = np.array(base_effects)
    jepa_effects = np.array(jepa_effects)

    print(f"Matched pairs used: {len(true_effects)} (shift={shift_months}mo earlier UDCA, effect on F at +{HORIZON}mo)")
    print(f"\nTrue (generator) effect on F: mean={true_effects.mean():+.4f}  std={true_effects.std():.4f}")
    print(f"  (negative = earlier treatment reduces fibrosis accumulation, as expected for responders)")

    corrs = {}
    for name, eff in [("baseline", base_effects), ("TS-JEPA", jepa_effects)]:
        mae = np.mean(np.abs(eff - true_effects))
        same_sign = np.mean((np.sign(eff) == np.sign(true_effects)) | (np.abs(true_effects) < 1e-4))
        corr = np.corrcoef(eff, true_effects)[0, 1] if eff.std() > 0 and true_effects.std() > 0 else float("nan")
        corrs[name] = corr
        print(f"\n{name} implied effect: mean={eff.mean():+.4f}  std={eff.std():.4f}")
        print(f"  MAE vs true effect: {mae:.4f}")
        print(f"  same sign as true effect: {same_sign*100:.0f}% of patients")
        print(f"  correlation with true effect across patients: {corr:.3f}")

    fig_counterfactual(true_effects, base_effects, jepa_effects, corrs, shift_months)


def fig_counterfactual(true_effects, base_effects, jepa_effects, corrs, shift_months):
    """Per-patient scatter: true (generator) matched-pair effect on F vs.
    each model's own implied effect from rolling out twice. Baseline
    should track the diagonal; TS-JEPA should show no consistent trend --
    the visual version of D12's headline correlation numbers."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lims = [min(true_effects.min(), base_effects.min(), jepa_effects.min()),
            max(true_effects.max(), base_effects.max(), jepa_effects.max())]

    for ax, eff, name, color in [
        (axes[0], base_effects, "baseline", "tab:blue"),
        (axes[1], jepa_effects, "TS-JEPA", "tab:orange"),
    ]:
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.plot(lims, lims, ls="--", color="gray", lw=1, label="perfect agreement")
        ax.scatter(true_effects, eff, s=18, alpha=0.6, color=color)
        ax.set_xlabel("true (generator) effect on F at +24mo")
        ax.set_title(f"{name}  (r={corrs[name]:.2f})")
        ax.set_xlim(lims); ax.set_ylim(lims)
    axes[0].set_ylabel("model's implied effect on F")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Counterfactual validation: implied vs. true treatment effect (shift={shift_months}mo)")
    plt.tight_layout()
    plt.savefig("figures/fig_counterfactual.png", dpi=120)
    plt.close()
    print("saved figures/fig_counterfactual.png")


if __name__ == "__main__":
    run_probe(n_patients=150, shift_months=6)
