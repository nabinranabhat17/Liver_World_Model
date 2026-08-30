"""
TS-JEPA with the graph-attention encoder (models/graph_encoder.py) in
place of the plain concat-GRU encoder. Everything else (predictor,
decoder, VICReg weights, epoch count, optimizer) is identical to
train_jepa.py's default, so the comparison in compare_graph.py isolates
the effect of the input representation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data import action_dim
from models.jepa import LATENT_DIM
from models.graph_encoder import GraphHistoryEncoder
from train_jepa import train_one_seed


def make_graph_encoder():
    return GraphHistoryEncoder(state_dim=8, action_dim=action_dim(), latent_dim=LATENT_DIM)


if __name__ == "__main__":
    model = train_one_seed(seed=0, encoder_factory=make_graph_encoder)
    torch.save(model.state_dict(), "checkpoints/jepa_graph.pt")
    print("saved checkpoints/jepa_graph.pt")
