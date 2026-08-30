"""
D22's paired D11 ablation, in one script since it's inherently a
control/treatment comparison. D11 found that training longer (150 epochs
vs 80) and with a higher decode weight (10.0 vs 5.0) made clean accuracy
WORSE (0.0308 vs 0.0291), diagnosed as: VICReg's covariance term keeps
decorrelating latent dimensions indefinitely under fixed weights, and a
higher-rank latent is a harder multistep target for the recurrent
predictor to hit.

Trains two checkpoints with an identical harness (same encoder,
predictor, decoder, 150 epochs, dec_w=10.0 -- D11's exact setup):

  jepa_150.pt    -- fresh replica of D11 (no VICReg annealing), the
                    matched control.
  jepa_anneal.pt -- the fix tested: VICReg's variance+covariance weights
                    relax to 40% of their starting value once effective
                    rank first crosses 5.3/16 (about where D11 found the
                    useful range had already stabilized), instead of
                    letting them keep pushing decorrelation for the
                    remaining ~100+ epochs.

See DECISIONS.md D22 for the result: annealing recovers part of D11's
degradation in-distribution but costs held-out-susceptibility accuracy
in return -- a real, partial, and mixed outcome, not a clean fix.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from train_jepa import train_one_seed

if __name__ == "__main__":
    control = train_one_seed(seed=0, n_epochs=150, dec_w=10.0, anneal_vicreg=False)
    torch.save(control.state_dict(), "checkpoints/jepa_150.pt")
    print("saved checkpoints/jepa_150.pt (D11 replica, no anneal)")

    annealed = train_one_seed(seed=0, n_epochs=150, dec_w=10.0, anneal_vicreg=True)
    torch.save(annealed.state_dict(), "checkpoints/jepa_anneal.pt")
    print("saved checkpoints/jepa_anneal.pt (VICReg annealed)")
