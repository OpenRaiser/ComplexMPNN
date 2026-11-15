# -*- coding: utf-8 -*-
"""The coupled discrete flow decoder.

Three layers over the encoder's graph, called once per generation step. Each
call reads the structural encoding, the sequence as it stands, the flow time,
and the probabilities the previous step predicted. Both residues of an edge
contribute their sequence information, so a prediction can respond to what was
chosen at the positions it contacts.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.modules.decoder_layer import MPNN_SCALE, DecoderLayer

# How the decoder reads a position's state: 0 fixed, 1 revealed, 2 masked.
FIXED = 0


class FlowDecoder(nn.Module):
    def __init__(self, hidden_dim: int, vocab: int, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, hidden_dim)
        self.source_emb = nn.Embedding(3, hidden_dim)
        self.time_in = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        # A residue reads its own sequence state over a self-edge, which needs
        # an edge representation of its own.
        self.self_edge_feat = nn.Parameter(torch.zeros(hidden_dim))
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, dropout=dropout, scale=MPNN_SCALE)
            for _ in range(n_layers)])
        self.readout = nn.Linear(hidden_dim, vocab)
        # The previous step's probabilities, and whether there were any.
        self.feedback_in = nn.Sequential(
            nn.Linear(vocab + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.state_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(5 * hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in self.layers])

    def sequence_state(self, state, design_mask, mask_id, feedback):
        """The sequence representation, with the feedback vector separate.

        A position is fixed, revealed, or masked and awaiting a prediction.
        The feedback is only read at masked design positions: a revealed or
        fixed identity is already known and is not up for revision.
        """
        design = design_mask.bool()
        source = torch.where(design, 1 + (state == mask_id).long(), FIXED)
        s_embed = self.token_emb(state) + self.source_emb(source)
        vocab = self.token_emb.num_embeddings
        if feedback is None:
            probs = s_embed.new_zeros((len(state), vocab))
            available = torch.zeros_like(state, dtype=torch.bool)
        else:
            if feedback.shape != (len(state), vocab):
                raise ValueError('the feedback does not match the alphabet: '
                                 f'{tuple(feedback.shape)} for {len(state)} '
                                 f'position(s) and {vocab} token(s)')
            probs = feedback.detach().to(s_embed.dtype)
            available = design & (state == mask_id)
        probs = probs * available[:, None]
        g = self.feedback_in(
            torch.cat((probs, available[:, None].to(probs.dtype)), -1))
        g = g * available[:, None]
        return s_embed + g, g

    def forward(self, h_V, h_E, edge_idx, state, time, design_mask, mask_id,
                feedback=None):
        own = torch.arange(len(h_V), device=h_V.device)
        edges = torch.cat((edge_idx, torch.stack((own, own))), dim=1)
        h_E = torch.cat((h_E, self.self_edge_feat.expand(len(h_V), -1)))

        s_embed, g = self.sequence_state(state, design_mask, mask_id, feedback)
        clock = torch.stack((time, 1 - time, (math.pi * time).sin(),
                             (math.pi * time).cos()), -1)
        h = h_V + self.time_in(clock.to(h_V.dtype)) + self.state_in(s_embed) + g

        recv, send = edges[0], edges[1]
        for layer, update in zip(self.layers, self.edge_updates):
            h_E = h_E + update(torch.cat(
                (h_E, h[recv], h[send], s_embed[recv], s_embed[send]), -1))
            h = layer(h, h_E, s_embed, edges)
        return self.readout(h)
