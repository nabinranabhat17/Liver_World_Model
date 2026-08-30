"""
TS-JEPA with the RK4 latent-ODE decoder (models/jepa_ode_decoder.py) in
place of the plain single-step LatentDecoder. Everything else (encoder,
predictor, VICReg weights, epoch count, optimizer) is identical to
train_jepa.py's default, so compare_jepa_ode.py isolates the effect of
folding continuous-time integration into the decode-anchor chain.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data import action_dim
from models.jepa import LATENT_DIM
from models.jepa_ode_decoder import LatentODEDecoder
from train_jepa import train_one_seed


def make_ode_decoder():
    return LatentODEDecoder(latent_dim=LATENT_DIM)


if __name__ == "__main__":
    model = train_one_seed(seed=0, decoder_factory=make_ode_decoder)
    torch.save(model.state_dict(), "checkpoints/jepa_ode_decoder.pt")
    print("saved checkpoints/jepa_ode_decoder.pt")
