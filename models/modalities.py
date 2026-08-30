"""
Modality decoders: every observable (blood panel, MRCP/ultrasound imaging
score, histology, 3-D liver shape) as a fixed, deterministic-plus-noise
function of x(t), per the spec ("every observable modality is treated as
a pure function of x(t)... an error in x shows up across every modality
at once").

These are NOT learned by either world model -- they are a separate
"rendering" module applied identically to ground-truth x(t) and to a
model's predicted x_hat(t), so we can measure how a given state-space
error propagates into each observable's units and answer "if the model's
fibrosis prediction is off by 0.05, how far off is the predicted platelet
count / liver stiffness / histology stage as a result?" This is exactly
what the spec's framing implies should be checked, and it was not
originally built (out of scope for the 6-8hr budget) -- this is the
first of the four follow-up extensions.

Modalities implemented (loosely realistic mappings, not clinical fact):
  blood_panel:  bilirubin (S,C), albumin (F, inverse), platelets (P, inverse),
                ALT (A), ALP (S,C)
  imaging:      liver_stiffness_kPa (F,P), duct_dilation_score (S,D)
  histology:    ishak_fibrosis_stage (F, 0-6 discrete), inflammation_grade (A, 0-4 discrete)
  shape_3d:     nodularity_score (F), volume_ratio (P, atrophy proxy)

Each function is fixed (no learned parameters) and includes a small,
fixed observation-noise term, exactly mirroring the "clean version is
what you model" simplification the spec asks for -- the noise here
represents each modality's own measurement noise on top of a perfectly
known x(t), not disagreement between modalities.
"""
import numpy as np

F, D, S, P, A, C, M, FLARE = range(8)


def blood_panel(x, rng=None, noise=True):
    """x: (...,8) -> dict of (...,) arrays."""
    bilirubin = 0.3 + 4.0 * x[..., C] + 2.0 * x[..., S]           # mg/dL, cholestasis/stricture driven
    albumin = 4.5 - 2.0 * x[..., F]                                 # g/dL, falls with fibrosis
    platelets = 250 - 180 * x[..., P]                               # x10^3/uL, falls with portal HTN
    alt = 20 + 120 * x[..., A]                                      # U/L, inflammation driven
    alp = 80 + 300 * x[..., S] + 100 * x[..., C]                    # U/L, cholestatic marker
    out = dict(bilirubin=bilirubin, albumin=albumin, platelets=platelets, alt=alt, alp=alp)
    if noise and rng is not None:
        sigma = dict(bilirubin=0.15, albumin=0.1, platelets=8.0, alt=5.0, alp=10.0)
        out = {k: v + rng.normal(0, sigma[k], size=v.shape) for k, v in out.items()}
    return out


def imaging(x, rng=None, noise=True):
    liver_stiffness = 4.0 + 20.0 * x[..., F] + 6.0 * x[..., P]      # kPa, elastography-like
    duct_dilation = 5.0 * x[..., S] + 2.0 * x[..., D]                # arbitrary severity units
    out = dict(liver_stiffness_kpa=liver_stiffness, duct_dilation_score=duct_dilation)
    if noise and rng is not None:
        out["liver_stiffness_kpa"] = out["liver_stiffness_kpa"] + rng.normal(0, 1.0, size=liver_stiffness.shape)
        out["duct_dilation_score"] = out["duct_dilation_score"] + rng.normal(0, 0.3, size=duct_dilation.shape)
    return out


def histology(x, rng=None, noise=True):
    ishak = np.clip(np.round(x[..., F] * 6), 0, 6)                  # discrete 0-6 fibrosis stage
    inflammation_grade = np.clip(np.round(x[..., A] * 4), 0, 4)     # discrete 0-4
    return dict(ishak_fibrosis_stage=ishak, inflammation_grade=inflammation_grade)


def shape_3d(x, rng=None, noise=True):
    nodularity = x[..., F] ** 1.5                                    # surface nodularity, nonlinear in F
    volume_ratio = 1.0 - 0.3 * x[..., P]                              # atrophy proxy, shrinks with portal HTN
    out = dict(nodularity_score=nodularity, volume_ratio=volume_ratio)
    if noise and rng is not None:
        out["nodularity_score"] = np.clip(out["nodularity_score"] + rng.normal(0, 0.03, size=nodularity.shape), 0, 1)
        out["volume_ratio"] = out["volume_ratio"] + rng.normal(0, 0.02, size=volume_ratio.shape)
    return out


ALL_MODALITIES = {
    "blood_panel": blood_panel,
    "imaging": imaging,
    "histology": histology,
    "shape_3d": shape_3d,
}


def render_all(x, rng=None, noise=True):
    """x: (...,8) -> flat dict of every observable across all 4 modalities."""
    out = {}
    for name, fn in ALL_MODALITIES.items():
        for k, v in fn(x, rng=rng, noise=noise).items():
            out[f"{name}.{k}"] = v
    return out


def modality_error_report(x_true, x_pred, rng=None):
    """For each observable, report MAE between the CLEAN rendering of
    x_true and x_pred (no observation noise -- isolating the effect of
    state-space error, not adding rendering noise on top of it)."""
    true_obs = render_all(x_true, rng=None, noise=False)
    pred_obs = render_all(x_pred, rng=None, noise=False)
    report = {}
    for k in true_obs:
        report[k] = float(np.mean(np.abs(true_obs[k] - pred_obs[k])))
    return report


if __name__ == "__main__":
    import numpy as np
    from generator import generate_dataset

    X, ctx, ercp, meta = generate_dataset(200, T=60, seed=0)
    rng = np.random.default_rng(1)
    obs = render_all(X, rng=rng, noise=True)
    print("Rendered modalities (mean, std) across dataset:")
    for k, v in obs.items():
        print(f"  {k:28s} mean={v.mean():8.3f}  std={v.std():7.3f}")

    # sanity: a small perturbation in F should show up in albumin, liver
    # stiffness, ishak stage, and nodularity -- but NOT in alt/inflammation
    x0 = X[0:1, 30, :].copy()
    x1 = x0.copy()
    x1[..., F] += 0.1
    o0 = render_all(x0, noise=False)
    o1 = render_all(x1, noise=False)
    print("\nSanity: +0.1 to F alone changes:")
    for k in o0:
        delta = float(np.ravel(o1[k] - o0[k])[0])
        if abs(delta) > 1e-6:
            print(f"  {k:28s} delta={delta:+.4f}")
