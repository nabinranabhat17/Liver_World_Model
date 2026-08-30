# Extensions — four lines of investigation beyond the core comparison

Detailed backing material for `DECISIONS.md` D1, D10-D12 (see also `memo.md`
Sec. 8), of different provenance relative to the assignment: the
graph-attention encoder (E1) and counterfactual validation (E4) are two
of the assignment's own "try something new" seeds; the continuous-time
Neural-ODE (E3) is mentioned in the assignment as something a JEPA-style
model "pairs naturally with," not one of the four bulleted seeds; the
modality decoders (E2) aren't from the seed list at all — they're
motivated by the assignment's own framing that every observable is "a
pure function of x(t)," used here as a diagnostic, not a requirement.
All four are built and evaluated with real numbers, in the order tackled
below. Every number here is reproducible by running the corresponding
script.

## E1 — Graph-attention encoder: a genuine negative result

**Built:** `models/graph_encoder.py` — a `GraphHistoryEncoder` that
replaces TS-JEPA's plain concat-GRU input encoding with a single
attention layer masked to the causal edges read directly off
`generator.py`'s own update equations (e.g. `A -> F`, `C -> F`, `F,C -> M`,
`A -> S`, `S -> D`, full edge list in the module docstring). Same
predictor, decoder, VICReg weights, and training schedule as plain
TS-JEPA — only the input representation changes, isolating its effect.

**Result: it underperforms the plain encoder, substantially and
robustly.** Trained at the same 80-epoch schedule (`checkpoints/jepa_graph.pt`):
clean in-distribution ratchet MAE 0.0566 vs. plain TS-JEPA's 0.0291 —
roughly double the error. Two follow-up checks ruled out the obvious
confounds before accepting this as a real finding:

- **Not an over-training artifact** (the same pattern DECISIONS.md D7
  found for training TS-JEPA longer): early-stopped at 25 epochs, matching plain TS-JEPA's effective
  rank (~5.2/16) at its best checkpoint. Still 0.068 ratchet MAE — no
  better.
- **Not an under-capacity artifact**: increased the per-node embedding
  dimension from 8 to 24 (3x). Still 0.054 at 40 epochs — better than the
  8-dim version, but still roughly double the plain encoder's error.

**Honest interpretation:** at this state size (8 channels), a plain GRU
over the concatenated raw state already has enough capacity to implicitly
learn whatever cross-channel structure helps prediction, without needing
an explicit sparse prior. The causal mask, if anything, actively hurts
here — most likely because it excludes non-causal but still statistically
useful correlations (e.g. two ratchets that move together because they
share an upstream driver, even without a direct edge between them), and
because the extra attention layer is additional capacity that must be
learned from scratch inside the same training budget that already stresses
the plain encoder's own collapse/predictability tradeoff (memo Sec. 4).
This is exactly the kind of documented negative result the assignment
invites ("tell us why your idea might be wrong") — the idea is principled,
and it didn't pay off at this scale.

Run: `python scripts/train_jepa_graph.py && python scripts/compare_graph.py`

## E2 — Modality decoders: state error in clinically legible units

**Built:** `models/modalities.py` — four fixed (non-learned) rendering
functions treating every observable as a pure function of x(t), per the
spec's framing: `blood_panel` (bilirubin, albumin, platelets, ALT, ALP),
`imaging` (liver stiffness kPa, duct dilation score), `histology` (Ishak
fibrosis stage 0-6, inflammation grade 0-4), `shape_3d` (nodularity,
volume ratio). Sanity-checked: perturbing F alone changes exactly the
F-dependent observables (albumin, liver stiffness, nodularity) and none
of the A/C-dependent ones, confirming the rendering functions are wired
correctly to the intended channels, not leaking cross-dependencies.

**Used for:** `scripts/modality_eval.py` translates the already-measured
state-space rollout error into each observable's own units. At K=24,
clean in-distribution: baseline's smaller state error propagates into
smaller error in every single modality (e.g. liver stiffness MAE 0.63 kPa
vs. TS-JEPA's 0.79 kPa; Ishak stage MAE 0.16 vs. 0.20) — a clinically
legible confirmation of the same ranking already established in
state-space, with the useful side effect of calibrating what a given MAE
number actually means in practice (a bilirubin MAE of ~0.15-0.25 mg/dL is
within normal-range noise; a liver-stiffness MAE under 1 kPa is small next
to the ~2 kPa gaps between clinical fibrosis-stage cutoffs).

Run: `python -m models.modalities && python scripts/modality_eval.py`

## E3 — Continuous-time Neural-ODE: the one clean positive result

**Built:** `models/neural_ode.py` — `NeuralODEStep`, a drop-in
replacement for the baseline (identical `forward(x_t, ctx_feat_t,
ercp_flag_t)` signature) that learns a derivative field and integrates it
with fixed-step RK4 (4 sub-steps per month) instead of taking one discrete
MLP step. Ratchet derivatives are `softplus(raw) * scale` at every RK4
stage — since RK4's stage weights are non-negative and sum to 1, a
weighted average of non-negative derivatives stays non-negative, so the
monotonicity guarantee survives integration, not just a single discrete
step (this is a real mathematical argument, not an empirical hope — and
it held: 0/N constraint violations after training, same as the discrete
baseline). ERCP relief is modeled as a discrete jump at the month boundary
(a real clinical event, not something to smear into a continuous
derivative) rather than folded into the ODE.

Trained with the exact same one-step + annealed-multistep loss schedule
as the discrete baseline (imported directly from `scripts/train_baseline.py`, not
reimplemented), so the comparison isolates the integration scheme alone.

**Result: a genuine, if modest, win.** Clean in-distribution: 0.0249 vs.
baseline's 0.0246 (essentially tied, very slightly worse). **Held-out
susceptibility probe: 0.0773 vs. baseline's 0.0791 — the Neural-ODE
generalizes slightly better** under an out-of-range disease-speed
parameter. Plausible mechanism, not fully isolated: finer sub-monthly
resolution may capture the fast A/C dynamics' contribution to the
ratchets' accumulated increment more accurately than a single coarse
discrete step, especially where susceptibility scales that accumulation
outside the range the discrete step size was implicitly tuned against.
This is the first clearly positive result among the four extensions —
modest, but real and reproducible.

Run: `python scripts/train_neural_ode.py && python scripts/compare_ode.py`

## E4 — Counterfactual validation: the most consequential finding

**Built:** `scripts/counterfactual.py`. Matched-pair construction exploits a
property of `generator.py`'s own implementation: `simulate()` consumes its
random-number generator in a fixed sequence (flare draw, then A-noise,
then C-noise, every month) regardless of treatment status — `treat_gain`
only rescales a target value, it never changes which `rng` calls happen or
in what order. So re-seeding an independent `rng` with the *same* seed for
two `simulate()` calls on two `Patient` copies that differ **only** in
`udca_start` produces a true matched pair: identical exogenous randomness,
differing only in the one intervention changed. This makes
`true_effect = counterfactual_trajectory - factual_trajectory` an honest
ground-truth causal effect, not a confound of independently-drawn noise.

The probe: for 100+ treated patients, simulate the factual trajectory and
a counterfactual where UDCA starts 3, 6, or 9 months earlier. Separately,
roll each trained model forward twice from an identical anchor state
(before either timeline's treatment start) — once fed the factual
treatment-timing action sequence, once fed the counterfactual one — and
compare the model's own *implied* treatment effect
(`counterfactual_rollout - factual_rollout`) to the generator's true
effect on fibrosis (F) at +24 months.

**Result, at shift=6 months (consistent at 3 and 9 months too, see
below):**

| | mean implied effect | MAE vs. true effect | same sign as true | correlation with true effect |
|---|---|---|---|---|
| true (generator) | −0.0066 | — | — | — |
| **baseline** | −0.0009 | **0.0059** | **100%** | **0.851** |
| **TS-JEPA** | +0.0002 | 0.0138 | 64% | **−0.284** |

**The baseline reliably tracks the true causal direction of the treatment
effect** (100% sign agreement, correlation 0.85 across patients) despite
never being trained on counterfactual pairs — it generalizes the
treatment-timing feature's marginal effect correctly. **TS-JEPA's implied
effect is unreliable and, on average, points the wrong way**
(correlation −0.28) — worse than having no signal at all. This replicated
at shift=3 months (baseline corr 0.80, JEPA corr −0.18) and shift=9 months
(baseline corr 0.90, JEPA corr −0.24), so it is not a one-off artifact of
a single shift size.

**Honest interpretation, and what this does and doesn't establish** (the
assignment's own scoping matters here): this validates whether each
model's response to changing the on-treatment action feature matches a
matched generator re-run — it does **not** validate behavior under a
referral-stratified, confounded cohort ("this patient reached us,
therefore they are sick"), because treatment assignment in `generator.py`
is exogenous by construction, not referral-biased. A model could pass this
check while still failing under real confounding; that's a materially
harder, correctly out-of-scope validation. Within what it does test,
though, this is the most consequential finding across all four
extensions: it's a second, independent, mechanistically different line of
evidence (not just accuracy-under-noise) pointing at the same conclusion
as the original memo — **ship the baseline**. A predictive latent that
can't reliably tell you which direction an intervention pushes the
patient is a real liability for a system whose stated clinical value is
counterfactual reasoning ("what if UDCA had started six months earlier").

**Plausible mechanism for the gap** (not fully isolated, offered as the
most likely explanation): TS-JEPA's predictor does receive the on-treatment
flag every step (it's part of the 8-D action vector fed to the `GRUCell`
each step), so it isn't structurally blind to it — but training only ever
saw the *factual* range of treatment timings mixed with seven other action
dimensions (disease class, age, sex, responder, ERCP flag, time, UDCA
start fraction) competing for representation capacity inside a 16-D
latent, with no loss term that specifically rewards getting the marginal
effect of *this one dimension* right. The baseline's direct, un-bottlenecked
mapping from `(x_t, action_t)` to `x_{t+1}` has an easier time keeping that
one feature's effect isolated and correctly signed.

Run: `python scripts/counterfactual.py`

## Summary

| extension | result |
|---|---|
| Graph-attention encoder | **Negative** — underperforms plain encoder by ~2x, robust to epoch/capacity checks |
| Modality decoders | Confirms the existing ranking in clinical units; useful calibration tool |
| Continuous-time Neural-ODE | **Modest positive** — ties in-distribution, wins slightly on held-out susceptibility |
| Counterfactual validation | **Most consequential** — baseline tracks true treatment-effect direction (corr 0.85), TS-JEPA does not (corr −0.28) |

Taken together, these sharpen rather than overturn the original memo's
recommendation: the baseline wins on accuracy, robustness, *and* now
causal-direction fidelity. The one place a more sophisticated mechanism
helped (continuous time) is a small, targeted change to the winning
baseline architecture, not a case for JEPA. If continuing, the natural
next step is folding E3's RK4 integration into the shipped baseline (it's
a strict improvement on the one axis tested) and using E4's
matched-pair-counterfactual method as a permanent regression test — a
model with good aggregate accuracy but a wrong-signed treatment effect
would be a serious clinical liability that ratchet-MAE alone would never
surface.
