"""
Graph-attention encoder: the causal graph as an attention mask, per the
assignment's "try something new" seed. The edges below are read directly
off generator.py's own update equations, not guessed at:

  flare -> A     (A_target includes flare)
  flare -> C     (C_target includes flare)
  S     -> C     (C_target includes 0.3*S)
  A     -> F     (dF includes 0.6*A)
  C     -> F     (dF includes 0.6*C)
  A     -> S     (dS ~ A_new)
  S     -> D     (dD includes 0.7*S)
  A     -> D     (dD includes 0.3*A)
  F     -> P     (dP ~ F_new)
  F     -> M     (dM ~ F_new*C_new)
  C     -> M     (dM ~ F_new*C_new)

Every field also has a self-loop (its own previous value always matters --
it's an increment on top of itself). No other edges exist in the
generator, so the mask below is a faithful, falsifiable encoding of the
data-generating causal structure, not a soft prior.

This replaces the plain "concatenate x_t and action_t, feed to GRU" input
encoding in HistoryEncoder with one attention layer that mixes information
ONLY along these edges before handing off to the same causal GRU used by
the plain TS-JEPA -- everything downstream (predictor, decoder,
constraint head, EMA target, VICReg) is unchanged, so any accuracy
difference is attributable to the input representation alone.
"""
import torch
import torch.nn as nn

FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
IDX = {name: i for i, name in enumerate(FIELD_NAMES)}

EDGES = [
    ("flare", "A"), ("flare", "C"),
    ("S", "C"),
    ("A", "F"), ("C", "F"),
    ("A", "S"),
    ("S", "D"), ("A", "D"),
    ("F", "P"),
    ("F", "M"), ("C", "M"),
]


def build_causal_mask():
    """(8,8) boolean mask, mask[target, source] = True if source is
    allowed to inform target (plus self-loops)."""
    mask = torch.eye(8, dtype=torch.bool)  # self-loops
    for src, tgt in EDGES:
        mask[IDX[tgt], IDX[src]] = True
    return mask


class GraphAttentionLayer(nn.Module):
    """Single-head scaled dot-product attention over the 8 state channels,
    masked to the causal graph above. Each channel is first embedded from
    a scalar into a small vector so attention has something to compute
    over; the output is one updated vector per channel, flattened."""

    def __init__(self, node_dim=8, n_heads=2):
        super().__init__()
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.embed = nn.Linear(1, node_dim)
        self.q = nn.Linear(node_dim, node_dim)
        self.k = nn.Linear(node_dim, node_dim)
        self.v = nn.Linear(node_dim, node_dim)
        self.out = nn.Linear(node_dim, node_dim)
        mask = build_causal_mask()  # (8,8) target x source
        self.register_buffer("attn_mask", mask)

    def forward(self, x):
        """x: (B, 8) raw state values -> (B, 8*node_dim) attended, flattened."""
        B = x.shape[0]
        nodes = self.embed(x.unsqueeze(-1))  # (B, 8, node_dim)
        q, k, v = self.q(nodes), self.k(nodes), self.v(nodes)
        scores = torch.einsum("bid,bjd->bij", q, k) / (self.node_dim ** 0.5)  # (B,8,8) target i attends to source j
        scores = scores.masked_fill(~self.attn_mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bij,bjd->bid", attn, v)  # (B,8,node_dim)
        out = self.out(attended)  # (B,8,node_dim)
        return out.reshape(B, -1)  # (B, 8*node_dim)


class GraphHistoryEncoder(nn.Module):
    """Drop-in replacement for models.jepa.HistoryEncoder: same interface
    (x_seq, action_seq) -> z_seq, but the state channels are first passed
    through the causal graph-attention layer before the GRU."""

    def __init__(self, state_dim=8, action_dim=8, latent_dim=16, hidden=32, node_dim=8):
        super().__init__()
        self.graph_attn = GraphAttentionLayer(node_dim=node_dim)
        gru_input_dim = state_dim * node_dim + action_dim
        self.gru = nn.GRU(input_size=gru_input_dim, hidden_size=hidden, batch_first=True)
        self.proj = nn.Linear(hidden, latent_dim)

    def forward(self, x_seq, action_seq):
        B, T, D = x_seq.shape
        x_flat = x_seq.reshape(B * T, D)
        attended = self.graph_attn(x_flat).reshape(B, T, -1)  # (B,T,8*node_dim)
        inp = torch.cat([attended, action_seq], dim=-1)
        h_seq, _ = self.gru(inp)
        return self.proj(h_seq)
