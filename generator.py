"""
Synthetic generator for the Digital Liver 8-D clinical state.

State x(t) in R^8, monthly timesteps:
  0 F  fibrosis                 [0,1]  ratchet, non-decreasing
  1 D  ductopenia               [0,1]  ratchet, non-decreasing
  2 S  biliary strictures       [0,1]  ratchet, non-decreasing EXCEPT may
                                        step down at an ERCP event month
  3 P  portal hypertension      [0,1]  ratchet, non-decreasing
  4 A  inflammatory activity    [0,1]  fast, mean-reverting
  5 C  cholestasis              [0,1]  fast, with flares
  6 M  malignancy hazard accum. [0,2]  monotone non-decreasing
  7 flare  acute cholangitis flare [0,1] transient, decays

Context (supplied to the model, not predicted):
  disease_class in {0: PSC-like, 1: PBC-like}
  age (years), sex in {0,1}, responder in {0,1}
  treatment timeline: udca_start_month (or -1 if never treated),
                       ercp_months (list, PSC-like only)

Hidden generator parameter (NOT supplied to any model):
  susceptibility  -- per-patient scalar controlling ratchet speed.
  This is exactly what the held-out-susceptibility generalisation probe
  in data.py withholds a range of at training time.

Design choices (see DECISIONS.md):
  - All ratchets are produced as x_prev + nonneg_increment, so
    non-decreasingness is true by construction in the generator itself,
    not just "usually true". This makes the self-check in main() a
    tautology-checker for bugs, not a statistical claim.
  - The coupling is explicit and intentional, not incidental:
      F,D driven by sustained A/C/S (inflammation begets fibrosis/ductopenia)
      P driven by sustained F (portal HTN is a consequence of fibrosis)
      M driven by sustained F*C (hazard of fibrosis-times-cholestasis)
      S creeps upward from A, relieved only at ERCP events
    This interaction structure is exactly what a per-field-only constraint
    mechanism risks breaking if not modeled carefully (see models/constraints.py).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

T_DEFAULT = 60  # months; matches the learned positional cap used downstream

FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
RATCHET_UP = [0, 1, 3, 6]     # F, D, P, M: strictly non-decreasing
RATCHET_S = 2                 # S: non-decreasing except at ERCP
FAST = [4, 5]                 # A, C
BOUNDS = {0: (0, 1), 1: (0, 1), 2: (0, 1), 3: (0, 1),
          4: (0, 1), 5: (0, 1), 6: (0, 2), 7: (0, 1)}


@dataclass
class Patient:
    disease_class: int          # 0 = PSC-like, 1 = PBC-like
    age: float
    sex: int
    responder: int
    susceptibility: float       # HIDDEN from the model
    udca_start: int             # -1 if never treated
    ercp_candidates: list


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _sample_patient(rng: np.random.Generator, susceptibility_range=(0.4, 1.6)) -> Patient:
    disease_class = int(rng.random() < 0.5)          # 0=PSC-like, 1=PBC-like
    age = float(rng.uniform(30, 70))
    sex = int(rng.random() < 0.5)
    responder = int(rng.random() < 0.6)
    susceptibility = float(rng.uniform(*susceptibility_range))

    ever_treated = rng.random() < 0.75
    udca_start = int(rng.integers(0, 30)) if ever_treated else -1

    # ERCP events are only meaningful for PSC-like (stricture-forming) disease,
    # and are triggered opportunistically once S is clinically significant --
    # we pre-sample *candidate* months; whether they actually fire depends on
    # S(t) at simulation time (handled in simulate()).
    ercp_candidates = sorted(rng.choice(np.arange(4, T_DEFAULT), size=14, replace=False).tolist()) \
        if disease_class == 0 else []

    return Patient(disease_class, age, sex, responder, susceptibility, udca_start, ercp_candidates)


def simulate(patient: Patient, T: int, rng: np.random.Generator, noise_std: float = 0.01):
    """Roll out one patient's clean trajectory. Returns (T, 8) array and the
    realised ERCP months (subset of candidates that actually fired)."""
    x = np.zeros((T, 8), dtype=np.float64)
    realised_ercp = []

    susc = patient.susceptibility
    is_psc = patient.disease_class == 0
    base_A = 0.35 if is_psc else 0.20
    base_C = 0.25 if is_psc else 0.30

    for t in range(T):
        if t == 0:
            x[t, 0] = rng.uniform(0.02, 0.08)   # F
            x[t, 1] = rng.uniform(0.0, 0.05)    # D
            x[t, 2] = rng.uniform(0.0, 0.10) if is_psc else 0.0  # S
            x[t, 3] = rng.uniform(0.0, 0.05)    # P
            x[t, 4] = base_A + rng.normal(0, 0.03)   # A
            x[t, 5] = base_C + rng.normal(0, 0.03)   # C
            x[t, 6] = 0.0                        # M
            x[t, 7] = 0.0                        # flare
            x[t] = np.clip(x[t], [b[0] for b in BOUNDS.values()], [b[1] for b in BOUNDS.values()])
            continue

        F, D, S, P, A, C, M, flare = x[t - 1]
        on_treatment = patient.udca_start >= 0 and t >= patient.udca_start
        treat_gain = 0.55 if (on_treatment and patient.responder) else 1.0

        # --- flare dynamics (transient) ---
        flare_prob = 0.06 if is_psc else 0.04
        if rng.random() < flare_prob:
            flare_new = float(rng.uniform(0.5, 1.0))
        else:
            flare_new = flare * 0.65
        flare_new = float(np.clip(flare_new, 0, 1))

        # --- fast, mean-reverting A, C ---
        A_target = (base_A + 0.5 * flare_new) * treat_gain
        A_new = A + 0.5 * (A_target - A) + rng.normal(0, noise_std)
        A_new = float(np.clip(A_new, 0, 1))

        C_target = (base_C + 0.4 * flare_new + 0.3 * S) * treat_gain
        C_new = C + 0.45 * (C_target - C) + rng.normal(0, noise_std)
        C_new = float(np.clip(C_new, 0, 1))

        # --- S: ratchet creep, ERCP-gated relief ---
        ercp_now = is_psc and (t in patient.ercp_candidates) and S > 0.15
        if ercp_now:
            S_new = max(0.0, S - float(rng.uniform(0.3, 0.5)))
            realised_ercp.append(t)
        else:
            dS = susc * 0.010 * A_new * (1.6 if is_psc else 0.4)
            S_new = float(np.clip(S + dS, 0, 1))

        # --- ratchets: F, D, P, M (all nonneg increments -> non-decreasing) ---
        dF = susc * 0.016 * (0.6 * A_new + 0.6 * C_new)
        F_new = float(np.clip(F + max(dF, 0.0), 0, 1))

        dD = susc * 0.018 * (0.7 * S_new + 0.3 * A_new) * (1.2 if is_psc else 0.5)
        D_new = float(np.clip(D + max(dD, 0.0), 0, 1))

        dP = 0.05 * F_new  # portal HTN is a consequence of sustained fibrosis
        P_new = float(np.clip(P + max(dP, 0.0), 0, 1))

        dM = susc * 0.09 * F_new * C_new  # hazard of sustained F*C
        M_new = float(min(M + max(dM, 0.0), 2.0))

        x[t] = [F_new, D_new, S_new, P_new, A_new, C_new, M_new, flare_new]

    return x, realised_ercp


def cirrhosis_stage(F: np.ndarray) -> np.ndarray:
    """Fixed monotone function of F. Derived, never stored, can never
    disagree with F because it IS a function of F."""
    return np.clip(np.floor(F * 3), 0, 2).astype(int)  # stages 0,1,2


def generate_dataset(n_patients: int, T: int = T_DEFAULT, seed: int = 0,
                      susceptibility_range=(0.4, 1.6)):
    """Returns:
      X          (N, T, 8) float64  clean state trajectories
      context    (N, 6) float64     [disease_class, age/100, sex, responder, udca_start/T, has_udca]
      ercp_mask  (N, T) bool        True where an ERCP event actually fired
      meta       list[Patient]
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_patients, T, 8))
    context = np.zeros((n_patients, 6))
    ercp_mask = np.zeros((n_patients, T), dtype=bool)
    meta = []

    for i in range(n_patients):
        p = _sample_patient(rng, susceptibility_range)
        traj, ercp_hits = simulate(p, T, rng)
        X[i] = traj
        for m in ercp_hits:
            ercp_mask[i, m] = True
        has_udca = 1.0 if p.udca_start >= 0 else 0.0
        context[i] = [p.disease_class, p.age / 100.0, p.sex, p.responder,
                       (p.udca_start / T) if p.udca_start >= 0 else -1.0, has_udca]
        meta.append(p)

    return X, context, ercp_mask, meta


def check_constraints(X: np.ndarray, ercp_mask: np.ndarray) -> dict:
    """Self-check: verify constraints hold. Returns violation counts."""
    N, T, _ = X.shape
    violations = {"F": 0, "D": 0, "P": 0, "M": 0, "S_unexpected_drop": 0, "bounds": 0}
    for idx in RATCHET_UP:
        name = FIELD_NAMES[idx]
        diffs = np.diff(X[:, :, idx], axis=1)
        violations[name] = int(np.sum(diffs < -1e-9))

    S = X[:, :, RATCHET_S]
    dS = np.diff(S, axis=1)
    dropped = dS < -1e-9
    # a drop is only allowed where ercp_mask is True at the arrival timestep (t+1)
    allowed = ercp_mask[:, 1:]
    unexpected = dropped & (~allowed)
    violations["S_unexpected_drop"] = int(np.sum(unexpected))

    for idx, (lo, hi) in BOUNDS.items():
        vals = X[:, :, idx]
        violations["bounds"] += int(np.sum((vals < lo - 1e-9) | (vals > hi + 1e-9)))

    violations["total_steps"] = int(N * (T - 1))
    return violations


if __name__ == "__main__":
    X, context, ercp_mask, meta = generate_dataset(n_patients=500, T=T_DEFAULT, seed=0)
    print("X shape:", X.shape)
    v = check_constraints(X, ercp_mask)
    print("Constraint self-check (should all be 0 except total_steps):")
    for k, val in v.items():
        print(f"  {k}: {val}")
    print("\nField ranges (min, max) across dataset:")
    for i, name in enumerate(FIELD_NAMES):
        print(f"  {name}: [{X[:,:,i].min():.4f}, {X[:,:,i].max():.4f}]")
    print("\nFraction PSC-like (disease_class=0):", (context[:, 0] == 0).mean())
    print("Fraction ever treated:", (context[:, 5] == 1).mean())
    print("Mean ERCP events per PSC patient:", ercp_mask[context[:, 0] == 0].sum(1).mean())
