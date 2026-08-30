"""
Deterministic train/val split, plus the generalisation probe cohorts.

Design (see DECISIONS.md D-probe):
  Training data is generated with susceptibility in-range (0.4-1.0).
  Three probes are generated OUTSIDE the training distribution to test
  whether a model learned the *dynamics* or just interpolated the
  training generator's typical trajectories:

    1. held_out_susceptibility : susceptibility in (1.0, 1.6], a faster
       disease-progression regime never seen in training.
    2. unseen_treatment_timing : UDCA start restricted to very late
       (month 45-59), a timing regime rarely seen in training (training
       draws start uniformly in [0,30)).
    3. long_rollout : same in-range susceptibility, but T=90 instead of
       T=60 -- 30 months beyond the horizon the model was ever trained
       to look at (relevant for the JEPA's learned absolute positions).

  We are explicit that this cannot fully settle "world model vs.
  generator-inverter" (see memo.md Sec 6): every probe is still a
  trajectory from the *same* generator family. What it CAN show is
  whether the model's error grows gracefully or catastrophically when
  pushed off the training distribution along a specific, named axis.
"""
from __future__ import annotations
import numpy as np
from generator import generate_dataset, T_DEFAULT

TRAIN_SUSCEPTIBILITY_RANGE = (0.4, 1.0)
OOD_SUSCEPTIBILITY_RANGE = (1.0, 1.6)


def make_train_val(n_train=1500, n_val=400, T=T_DEFAULT, seed=0):
    Xtr, Ctr, Etr, Mtr = generate_dataset(n_train, T=T, seed=seed,
                                           susceptibility_range=TRAIN_SUSCEPTIBILITY_RANGE)
    Xva, Cva, Eva, Mva = generate_dataset(n_val, T=T, seed=seed + 1,
                                           susceptibility_range=TRAIN_SUSCEPTIBILITY_RANGE)
    return dict(X=Xtr, ctx=Ctr, ercp=Etr, meta=Mtr), dict(X=Xva, ctx=Cva, ercp=Eva, meta=Mva)


def make_probe_held_out_susceptibility(n=400, T=T_DEFAULT, seed=100):
    return _pack(*generate_dataset(n, T=T, seed=seed, susceptibility_range=OOD_SUSCEPTIBILITY_RANGE))


def make_probe_unseen_treatment_timing(n=400, T=T_DEFAULT, seed=101):
    """Re-generate then override: force UDCA start late (45-59) for all
    treated patients, keep everything else in-distribution."""
    X, ctx, ercp, meta = generate_dataset(n, T=T, seed=seed,
                                           susceptibility_range=TRAIN_SUSCEPTIBILITY_RANGE)
    rng = np.random.default_rng(seed + 1)
    for i, p in enumerate(meta):
        if p.udca_start >= 0:
            new_start = int(rng.integers(45, T - 1))
            p.udca_start = new_start
            ctx[i, 4] = new_start / T
    # NOTE: ctx/meta now reflect the new (late) start, but X was simulated
    # with the OLD start. To make this probe internally consistent we must
    # re-simulate with the new timeline.
    from generator import simulate
    rng2 = np.random.default_rng(seed + 2)
    for i, p in enumerate(meta):
        traj, hits = simulate(p, T, rng2)
        X[i] = traj
        ercp[i] = False
        for h in hits:
            ercp[i, h] = True
    return _pack(X, ctx, ercp, meta)


def make_probe_long_rollout(n=200, T=90, seed=102):
    return _pack(*generate_dataset(n, T=T, seed=seed, susceptibility_range=TRAIN_SUSCEPTIBILITY_RANGE))


def _pack(X, ctx, ercp, meta):
    return dict(X=X, ctx=ctx, ercp=ercp, meta=meta)


def context_dim():
    return 6  # [disease_class, age/100, sex, responder, udca_start_frac(-1 if none), has_udca]


def make_context_features(ctx: np.ndarray, t: int, T: int) -> np.ndarray:
    """Per-timestep context features fed to the model at step t:
    the 6 static context features + treatment-active flag + normalised time."""
    N = ctx.shape[0]
    has_udca = ctx[:, 5]
    udca_frac = ctx[:, 4]
    on_treatment = ((has_udca == 1) & (udca_frac >= 0) & (t >= udca_frac * T)).astype(np.float64)
    time_feat = np.full((N,), t / T)
    return np.concatenate([ctx, on_treatment[:, None], time_feat[:, None]], axis=1)  # -> 8 dims


def action_dim():
    return 8


if __name__ == "__main__":
    train, val = make_train_val(n_train=200, n_val=50, seed=0)
    print("train X:", train["X"].shape, "val X:", val["X"].shape)
    p1 = make_probe_held_out_susceptibility(n=50)
    p2 = make_probe_unseen_treatment_timing(n=50)
    p3 = make_probe_long_rollout(n=50)
    print("probe susceptibility X:", p1["X"].shape,
          "susc range:", min(m.susceptibility for m in p1["meta"])),
    print("  max susc:", max(m.susceptibility for m in p1["meta"]))
    print("probe treatment timing: udca starts (treated only):",
          sorted(m.udca_start for m in p2["meta"] if m.udca_start >= 0)[:10])
    print("probe long rollout X:", p3["X"].shape)
