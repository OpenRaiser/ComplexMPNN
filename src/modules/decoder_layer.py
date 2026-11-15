# -*- coding: utf-8 -*-
"""One decoder layer: a message over every edge, then a feed-forward update.

ProteinMPNN's decoder layer, indexed by the encoder's edge list instead of a
neighbour axis. The parameter names are theirs, so a state dict transfers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.tools.segment import segment_sum

# ProteinMPNN divides the summed messages by a constant instead of by the
# number of neighbours. The trained weights are scaled to it.
MPNN_SCALE = 30.0


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        return self.W_out(self.act(self.W_in(h_V)))


class DecoderLayer(nn.Module):
    """A message from the receiver, the edge, and the sender with its token."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1,
                 scale: float = MPNN_SCALE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = float(scale)
        self.W1 = nn.Linear(4 * hidden_dim, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dense = PositionWiseFeedForward(hidden_dim, hidden_dim * 4)

    def forward(self, h: torch.Tensor, h_E: torch.Tensor,
                s_embed: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        recv, send = edge_idx[0], edge_idx[1]
        pair = torch.cat([h[recv], h_E, s_embed[send], h[send]], dim=-1)
        msg = self.W3(self.act(self.W2(self.act(self.W1(pair)))))
        delta = segment_sum(msg, recv, h.shape[0]) / self.scale
        h = self.norm1(h + self.dropout1(delta))
        return self.norm2(h + self.dropout2(self.dense(h)))
