# Decisions

Full reasoning trail, in twelve entries: the core baseline-vs-JEPA build,
including two dead ends and one corrected mistake (D1-D9), then three
further lines of investigation beyond the assignment's core ask
(D10-D12) — each carrying its own later verification or fix, where one
was tried, folded into the same entry rather than left as a separate
follow-up. Read `memo.md` first; this is the detail behind it.

## D1 — State representation: x(t) as the organizing construct, modality decoders deferred then built

Followed the spec directly: 8-D bounded state, monthly timesteps. No
modality decoders built originally — the spec treats every modality as a
pure function of x(t) and asks us to model the clean version; building
them would have spent the 6-8 hour budget on rendering functions with
zero bearing on accuracy, constraints, or explainability.

Built later anyway, as a diagnostic rather than a requirement:
`models/modalities.py`, four fixed rendering functions (blood panel,
imaging, histology, 3-D shape) treating every observable as a pure
function of x(t). Sanity-checked: perturbing F alone changes only
F-dependent observables. `scripts/modality_eval.py` translates
state-space error into each unit — at K=24, baseline's smaller error
propagates into smaller error everywhere (liver stiffness 0.63 vs 0.79
kPa; Ishak stage 0.16 vs 0.20), confirming the ranking already known in
state-space and calibrating what a given MAE means clinically (bilirubin
MAE ~0.15-0.25 mg/dL is within normal-range noise). Run:
`python -m models.modalities && python scripts/modality_eval.py`.

## D2 — Generator design: ratchets, coupling, hidden context, and tuning

Every ratchet field is generated as `prev + max(increment, 0)`, never an
unconstrained update that happens to usually go up — this makes the
generator's self-check a tautology-verifier for bugs, not a statistical
claim, which matters because the *model* needs the same property by
construction. **Rejected alternative:** clipping a cumulative sum
post-hoc (`np.maximum.accumulate`) — changes the innovation's statistics
and would let coupling bugs hide behind the clip.

The coupling, all in `generator.py`: F driven by sustained A+C; D by
sustained S+A (PSC-weighted); P by sustained F (a consequence, not an
independent ratchet); M by sustained F·C; S creeps from A, relieved only
at ERCP (gated, not clamped); flares perturb A and C together then
decay; treatment suppresses A/C for responders. A *designed* interaction
graph, not five independent OU processes — a model that only nails
per-field monotonicity will show it in the coupling-sensitive probes
(D6, D7).

`susceptibility` scales every ratchet's rate but is **never given to the
model** — only `disease_class`, `age`, `sex`, `responder`, and the
treatment timeline are, exactly as spec'd. It's the one variable a model
has to infer from the trajectory itself, which is what makes "held-out
susceptibility" a real test of learned dynamics.

Tuned once: first pass had F topping out at 0.52 over 60 months and ERCP
firing only ~0.05x/PSC patient — undersold the dynamic range and left
S-relief nearly untestable. Increased rate constants ~1.5-2x and
loosened the ERCP trigger (S>0.3→S>0.15, 6→14 candidate months).
Re-checked: still 0 violations, F now reaches 0.77, ERCP fires
~1.7x/patient. Verified visually (`figures/generator_sanity_check.png`)
before building anything downstream.

## D3 — Constraint head: hard sign guarantee, soft coupling strength, later verified

Per-field monotonicity: `prev + softplus(raw) * scale`, clamped —
architecturally non-negative for any weights, proven under random init
(`scripts/test_invariants.py`) before training. S's relief is a
separately-signed decrease, multiplicatively gated by an ERCP flag —
zero relief when absent, by construction.

M's increment is gated by `F_prev * C_prev` into a `[0.3, 2.0]x`
multiplier on the softplus increment — gets the coupling's **sign**
right by construction (never shrinks) but leaves its **strength** for
training to learn. Deliberate: a fully hand-specified M update would
make "did it learn the hazard" true by definition. The 0.3x floor (not
0) avoids a dead-gradient region.

**Verified later** (`scripts/coupling_probe.py`): since `M_inc =
M_inc_raw * (0.3 + 1.7 * F_prev * C_prev)` and the true `dM ~
susceptibility * 0.09 * F_prev * C_prev` are both multiplicative, the
ratio `(dM/dF_prev) / (dM/dC_prev)` should equal `C_prev / F_prev`
regardless of hidden susceptibility — a testable invariant, no
retraining needed (pure autograd, 388 baseline / 144 JEPA samples,
`F_prev, C_prev > 0.08`). Both models get the *sign* right everywhere
(100%). The *ratio* differs: baseline correlates 0.753 with true C/F;
TS-JEPA's full path correlates 0.914; isolating the decoder alone (z
fixed) gives **1.000, exact to numerical precision** — because
`LatentDecoder.net(z)` never takes `x_prev` at all, so TS-JEPA's
decode-step ratio is architecturally pinned regardless of training,
while the baseline's is a real, only-moderately-accurate *learned*
property. The probe shows the two architectures put this guarantee in
different places, not that JEPA "understood" it better. Run:
`python scripts/coupling_probe.py`.

## D4 — Baseline model: x(t) itself as the latent

`MonotoneStep`: a plain MLP mapping `(x_t, context_t)` to raw
constraint-head inputs — no history, no learned representation. The
"x(t) as the latent" peer the assignment asks to weigh JEPA against, and
the delivered prototype (memo.md for why).

Trained with one-step MSE plus an annealed-in 8-step free-rollout MSE
(anneal over 20 epochs) — pure multistep loss from random init diverges
before the model learns anything.

## D5 — TS-JEPA: what was actually built, and what wasn't

Built: causal GRU history encoder (online + EMA target, tau=0.99),
action-conditioned `GRUCell` predictor rolled forward on action/context
only (never ground truth), VICReg variance+covariance regularisation,
decode-anchor chain (`LatentDecoder` + the same `ConstraintHead`) so
every decoded step is on-manifold the same way as the baseline.

**Sequenced deliberately**: the assignment's seeds are optional.
Priority was getting action-conditioning + EMA + VICReg + decode-anchor
right and *measured* first. Two of the four seeds were pursued after:
the causal graph as an attention mask (D10), counterfactual validation
(D12). Continuous-time integration (mentioned as something JEPA "pairs
naturally with," not a bulleted seed) was tried both on the baseline and
folded into TS-JEPA's own decode step (D11 — a win one way, a loss the
other) — real evidence, not just intuition, for where it actually pays
off.

**Not pursued:** a learned "is-this-state-on-manifold" critic (the third
seed) — the constraint mechanism already makes off-manifold states
unreachable, so a violation-detecting critic has nothing to catch; a
*plausibility* critic for technically-valid-but-implausible trajectories
is a different, untried question. The fourth seed's referral-bias half
is also untried, for the same reason the assignment's own "out of
scope" section gives.

## D6 — Head-to-head result: baseline wins, and why that's the honest finding

`scripts/compare.py`, head-to-head across clean in-distribution, all
three probes, noise stress (σ=0.03-0.25), stale-visit stress (6-18mo):
**baseline wins every axis**, by 2-40% ratchet MAE at K=24
(`figures/fig_scorecard.png`, `fig_noise.png`, `fig_staleness.png`).
Reported as the honest result — two follow-ups (D7, D8) tried to
understand *why*, not to manufacture a JEPA win.

Along the way, a harness bug: the noise-stress axis initially scored
constraint violations over the *entire* trajectory, including the
un-rolled noisy prefix — raw corrupted ground truth, not a prediction —
inflating violations to ~16,000/118,000. Fixed by scoring only the
rolled-out segment; after the fix, 0/N everywhere, as the guarantee
predicts. A harness bug, not a model failure — worth stating precisely,
since it's the kind of bug that could otherwise get reported as a false
finding either direction.

## D7 — JEPA training dynamics: two attempts to close the gap

Hypothesis: TS-JEPA was under-trained. Ran 150 epochs (up from 80) plus
a higher decode weight (10 vs 5). Result: clean MAE at K=24 got
**worse** (0.0308 vs 0.0291), while `jepa_loss` quintupled (0.03→0.15)
as effective rank climbed 5.1→8.7/16.

Diagnosis: VICReg's covariance term decorrelates indefinitely under
fixed weights, but a more diverse latent is a harder multistep target —
past ~epoch 40 (where rank had already stabilized) this trades net
negative. Kept the 80-epoch checkpoint; this finding, not "train
longer," is the real answer to "how would you close the gap."

**Tried the natural fix later:** relax VICReg once effective rank
reaches its useful range instead of leaving weights fixed.
`scripts/train_jepa_anneal.py` trains both halves with an identical
harness: the 150-epoch setup fresh as a control (reproduces cleanly:
rank 8.74/16, MAE 0.0311) and the fix (`var_w`/`cov_w` relax to 40% once
rank crosses 5.3/16 — final rank 6.89/16, `jepa_loss` 0.047 vs. 0.147).
**Result: real but partial, and mixed — not a fix.** Clean MAE 0.0302
with annealing vs. 0.0311 without, recovering ~45% of the gap to
80-epoch's 0.0291. Held-out susceptibility goes the *other* way: 0.0882,
worse than both the no-anneal control (0.0785) and the 80-epoch
checkpoint (0.0807) — a real generalisation cost for a partial
in-distribution gain. Single-seed, reported as such. Checkpoints:
`checkpoints/jepa_150.pt` (control), `checkpoints/jepa_anneal.pt`
(annealed).

## D8 — Honest correction: an apparent noise-robustness crossover did not replicate

`scripts/jepa_denoise.py`: online encoder trained only on corrupted
input (decode/EMA targets stay clean), hypothesizing this teaches
denoising. A single-seed test showed it edging ahead of baseline by
0.9-1.9% at σ≥0.15 — reported as a crossover.

**Didn't replicate.** 8 independent seeds: baseline won 8/8, 8/8, 5/8 at
σ=0.15/0.20/0.25 — the crossover was noise-draw variance
(`scripts/verify_claims.py` CLAIM 5). What held up: the augmentation
narrows the gap monotonically (σ=0.25: +0.0034→+0.0006, an 82%
reduction) without closing it — directionally real, insufficient to
flip the comparison.

## D9 — Explainability: a wrong-signed attribution, found and fixed

`scripts/explain.py` uses plain autograd, not a surrogate explainer —
both models keep an explicit decoded state at every step, so a direct
gradient is already the honest explanation. Finding: one-step
attribution for the selected patient shows `d(F_next)/d(F_prev) ≈ 1.0`
(expected) but small, **wrong-signed** sensitivities to A and C
(`-0.006`, `-0.005`), where the generator's true structure is positive
in both. Priority from here: diagnose the wrong sign first (a trust
problem independent of which architecture wins), then verify the F·C
coupling's learned strength (D3), then try the causal graph as an
attention mask (D10).

**Diagnosed and fixed.** Broader check first: across 500 random
samples, the baseline's `d(F_next)/dA` is positive only **13.8%** of the
time, `d(F_next)/dC` only **7.4%** — the wrong sign is the *typical*
case, not a one-off. `scripts/train_baseline_jacobian.py`: identical to
the baseline, plus a double-backward term each step —
`relu(-dF/dA) + relu(-dF/dC)` — supervising sign only (magnitude needs
the hidden susceptibility), weight 1.0. **Completely fixed, at a small
honest cost.** Sign-correctness: 13.8%/7.4% → **100%/100%**. At the
original patient/month: dF/dA `-0.006→+0.003`, dF/dC `-0.005→+0.004`.
Cost: clean K=24 MAE 0.0246→**0.0259** (+5.3%), held-out
0.0791→**0.0844** (+6.7%). Violations untouched (0/N). Checkpoint:
`checkpoints/baseline_jacobian.pt`.

## D10 — Graph-attention encoder: a genuine negative result

`models/graph_encoder.py`: `GraphHistoryEncoder` replaces the concat-GRU
with one attention layer masked to causal edges read off
`generator.py`'s own equations. Same predictor/decoder/VICReg/schedule —
only the input representation changes.

**Underperforms, substantially and robustly.** Clean MAE 0.0566 vs.
plain TS-JEPA's 0.0291 — ~2x worse. Ruled out confounds: early-stopped
at 25 epochs (matching plain JEPA's best effective rank) still 0.068;
tripling embedding dim (8→24) still 0.054 at 40 epochs.

Diagnosis: at 8 channels, a plain GRU already has enough capacity to
implicitly learn cross-channel structure without an explicit sparse
prior; the mask likely excludes useful non-causal correlations, and the
extra attention layer is capacity that has to be learned from scratch in
the same budget. Run:
`python scripts/train_jepa_graph.py && python scripts/compare_graph.py`.

## D11 — Continuous-time integration: a split verdict

`models/neural_ode.py`: `NeuralODEStep`, a drop-in baseline replacement
integrating a learned derivative field with RK4 (4 sub-steps/month).
Ratchet derivatives are `softplus(raw) * scale` at every stage — RK4's
non-negative, summing-to-1 weights keep the accumulated increment
non-negative, so monotonicity survives integration (0/N violations
held). ERCP relief stays a discrete jump, not smeared into the ODE.
Trained with the exact baseline schedule, isolating the integration
scheme.

**On the baseline: a genuine, modest win.** Clean: 0.0249 vs. baseline's
0.0246 (essentially tied). Held-out susceptibility: 0.0773 vs. 0.0791 —
generalizes slightly better, plausibly because finer sub-monthly
resolution captures fast A/C dynamics more accurately than one coarse
step. Run: `python scripts/train_neural_ode.py && python scripts/compare_ode.py`.

**Folded into TS-JEPA's decode step instead: worse, not better.**
`models/jepa_ode_decoder.py` (`LatentODEDecoder`), a drop-in for
`LatentDecoder` integrating each month with 4 RK4 sub-steps instead of
one `ConstraintHead` call — same encoder/predictor/VICReg/schedule, only
the decode step changes. Clean MAE 0.0365 vs. 0.0291 (+25%); held-out
0.1015 vs. 0.0807 (+26%). Violations stay 0/N — monotonicity does
survive integration here too, the only part of the hypothesis that
held. Diagnosis: the baseline's ODE integrates the same quantity it
already predicted; TS-JEPA's decoder instead has to fit a derivative
field from a single *static* per-month `z`, evaluated at four
sub-states, learned jointly with three other competing loss terms in the
same 80-epoch budget — more capacity without more budget. Well-motivated,
didn't pay off. Run: `python scripts/train_jepa_ode.py`.

## D12 — Counterfactual validation: the most consequential finding, and a partial fix

`scripts/counterfactual.py` exploits that `simulate()` consumes its RNG
in a fixed sequence regardless of treatment status — re-seeding
identically for two `Patient` copies differing only in `udca_start`
gives a true matched pair, so `true_effect = cf_trajectory -
factual_trajectory` is an honest causal effect. Probe: for 100+ treated
patients, simulate factual + counterfactual (UDCA 3/6/9 months earlier);
roll each model forward twice from an identical anchor and compare its
*implied* effect on F at +24mo to the true effect.

**At shift=6mo (consistent at 3, 9):** baseline tracks the true
direction reliably (100% sign agreement, corr 0.851) despite never
training on counterfactual pairs. TS-JEPA's implied effect points the
wrong way on average (corr −0.284) — worse than no signal. Replicates
at shift=3 (base 0.80, JEPA −0.18) and shift=9 (base 0.90, JEPA −0.24).

**Scope:** validates response to the on-treatment action feature against
a matched re-run — not behavior under referral-stratified confounding,
since treatment assignment here is exogenous by construction. Within
what it tests, this is the most consequential of the extended lines of
investigation: a second, mechanistically independent line of evidence
for D6 — ship the baseline.

**A targeted fix was tried:** `scripts/train_jepa_counterfactual.py`
rolls the model forward twice per epoch from an identical anchor
(factual vs. counterfactual treatment timing, 64 treated-responder
matched pairs, shift=6mo) and adds a hinge loss pushing the implied
effect at +24mo to the true effect's *sign* (weight 8.0, margin 0.0015).
**A real, substantial fix on one axis — none on the one flagged as most
damning.** Same-sign agreement, shift=3/6/9mo: **58%/64%/65% →
88%/94%/94%**, and the *mean* implied effect now tracks the true
direction and scale (shift=6mo: true −0.0066, plain +0.0002, fixed
−0.0081; MAE 0.0138→0.0112). But **per-patient correlation barely moves
and stays negative**: −0.18→−0.20, −0.28→−0.24, −0.24→−0.25 — the hinge
rewards each patient crossing a margin independently, with no term for
*relative* effect size, so the model learns to shift its average
response without learning to rank patients. Side effects: clean MAE
0.0291→0.0307 (+5.5%), held-out *improves* 0.0807→0.0720 (−10.8%).
Violations stay 0/N. Checkpoint: `checkpoints/jepa_counterfactual.pt`.
Closes the population-level question, not the per-patient one.
