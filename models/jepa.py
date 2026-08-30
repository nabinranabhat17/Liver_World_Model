"""
TS-JEPA: action-conditioned, latent time-series JEPA for the Digital Liver
state. Predicts the REPRESENTATION of future states (role: "the thing a
learned world model decodes from"), not raw values directly -- the
assignment's recommended route.

Components:
  HistoryEncoder (online):  causal GRU over (x_t, action_feat_t) -> z_t.
                             Causal because GRU only sees x_0..x_t.
  HistoryEncoder (target):  EMA copy of the online encoder, no gradient.
                             Encodes the TRUE future to produce the
                             prediction target z_{t+s}, exactly like
                             I-JEPA/V-JEPA's teacher network -- this is
                             what lets the predictor be graded on
                             "predict the representation" rather than
                             "reconstruct the pixels/raw state", which is
                             the whole point of the JEPA framing.
  LatentPredictor:           GRUCell that rolls z_t forward autoregressively
                             using only the ACTION features (context,
                             on-treatment flag, ERCP flag, time) -- never
                             the ground-truth state -- for k steps.
  LatentDecoder + ConstraintHead: decodes each predicted latent back to a
                             constrained x_hat, CHAINED (dec-anchor): x_hat_s
                             is decoded using x_hat_{s-1} as x_prev, not the
                             ground truth. This is what makes the decoded
                             rollout an honest free rollout and keeps every
                             decoded step on the constraint manifold by the
                             same construction as the baseline.

Collapse prevention: VICReg-style variance + covariance regularisation on
the online encoder's outputs (see train_jepa.py). We use VICReg rather than
a contrastive loss because there's no natural negative-pair structure here
(every trajectory is a plausible liver trajectory) -- VICReg only needs the
positive/predictive pairs already implied by the JEPA objective, plus a
batch to decorrelate against. The diagnostic that would catch collapse is
the effective rank of the latent covariance (see explain_jepa collapse
check in eval_jepa.py): a collapsed representation has effective rank 1,
i.e. all patients map to (near-)the same latent regardless of state.

Scope note (see DECISIONS.md): we do NOT implement input masking (randomly
hiding history timesteps) or a graph-attention encoder / Neural-ODE, both
listed as seeds/extras in the assignment, not requirements. What we DO
implement -- action-conditioning, an EMA target, VICReg, and a decode-
anchor for on-manifold constrained rollout -- is the load-bearing core of
the JEPA route as it applies to this 8-D bounded state, and is what the
memo's collapse and on-manifold arguments rest on.
"""
import copy
import torch
import torch.nn as nn
from models.constraints import ConstraintHead

LATENT_DIM = 16


class HistoryEncoder(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, latent_dim=LATENT_DIM, hidden=32):
        super().__init__()
        self.gru = nn.GRU(input_size=state_dim + action_dim, hidden_size=hidden, batch_first=True)
        self.proj = nn.Linear(hidden, latent_dim)

    def forward(self, x_seq, action_seq):
        """x_seq: (B,T,8), action_seq: (B,T,8) -> z_seq: (B,T,latent_dim).
        Causal: z_seq[:,t,:] depends only on x_seq[:, :t+1, :]."""
        inp = torch.cat([x_seq, action_seq], dim=-1)
        h_seq, _ = self.gru(inp)
        return self.proj(h_seq)


class LatentPredictor(nn.Module):
    def __init__(self, action_dim=8, latent_dim=LATENT_DIM):
        super().__init__()
        self.cell = nn.GRUCell(input_size=action_dim, hidden_size=latent_dim)

    def forward(self, z_t, action_next):
        """One step: z_t, action_{t+1} -> predicted z_{t+1}."""
        return self.cell(action_next, z_t)

    def rollout(self, z_t, action_seq_future):
        """action_seq_future: (B, k, action_dim) for steps t+1..t+k.
        Returns list of predicted latents [z_{t+1}, ..., z_{t+k}]."""
        z = z_t
        preds = []
        for s in range(action_seq_future.shape[1]):
            z = self.forward(z, action_seq_future[:, s, :])
            preds.append(z)
        return preds


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ConstraintHead.RAW_DIM),
        )
        self.constraint_head = ConstraintHead()

    def forward(self, z, x_prev, ercp_flag):
        raw = self.net(z)
        return self.constraint_head(raw, x_prev, ercp_flag)

    def decode_chain(self, z_list, x_anchor, ercp_seq):
        """z_list: list of k predicted latents. x_anchor: (B,8) real state at
        step t (the anchor). ercp_seq: (B,k). Chains decode-anchor: each
        x_hat becomes the x_prev for the next decode, exactly mirroring how
        the baseline's free rollout works, so the two models are evaluated
        on a like-for-like footing."""
        x = x_anchor
        xs = []
        for s, z in enumerate(z_list):
            x = self.forward(z, x, ercp_seq[:, s])
            xs.append(x)
        return torch.stack(xs, dim=1)  # (B,k,8)


class TSJEPA(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, latent_dim=LATENT_DIM, ema_tau=0.99,
                 encoder=None, decoder=None):
        super().__init__()
        self.online_encoder = encoder if encoder is not None else HistoryEncoder(state_dim, action_dim, latent_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = LatentPredictor(action_dim, latent_dim)
        self.decoder = decoder if decoder is not None else LatentDecoder(latent_dim)
        self.ema_tau = ema_tau

    @torch.no_grad()
    def update_target(self):
        for p_t, p_o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            p_t.data.mul_(self.ema_tau).add_(p_o.data, alpha=1 - self.ema_tau)
