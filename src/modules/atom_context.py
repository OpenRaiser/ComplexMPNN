# -*- coding: utf-8 -*-
"""Ligand and metal-ion context for each residue.

Every residue reads the atoms of the retained cofactors near it: their
distances, their elements, and how many there are. The result is added to the
residue's representation before the first encoder block. The atoms themselves
are never designed and never move.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.tools.geometry import (
    PROTEIN, atom_table, atom_validity, node_centers, residue_frames,
)

# The role id for an atom that is fixed context. Roles 0 to 11 belong to the
# polymer backbone atoms that backbone_atoms labels.
CONTEXT_ROLE = 12
N_ROLES = 13
# Radial basis functions per distance, over the cutoff radius.
N_RADIAL = 16


def _mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(),
                         nn.Linear(hidden, n_out))


def backbone_atoms(batch):
    """Per-residue atoms with their roles and elements, and the node centres.

    Returns (xyz, ok, roles, elements, R, frame_ok, centre) over eight slots:
    the seven supplied slots and a virtual CB, which is constructed for every
    protein residue by ProteinMPNN's formula. A protein uses the first four
    slots, a nucleotide all seven and no CB.
    """
    X = batch['X'].float()
    valid = batch['slot_valid'].bool() & torch.isfinite(X).all(-1)
    protein = batch['geo_cls'] == PROTEIN
    valid = valid & (~protein[:, None] | (torch.arange(7, device=X.device) < 4))
    X = torch.where(valid[..., None], X, torch.zeros_like(X))
    R, _, frame_ok = residue_frames(X[:, 0], X[:, 1], X[:, 2], valid)
    cb = atom_table(X, PROTEIN, X.new_empty((0, 3)))[:, -1]
    cb_ok = atom_validity(valid, PROTEIN, frame_ok, 0)[:, -1] & protein
    xyz = torch.cat([X, cb[:, None]], 1)
    ok = torch.cat([valid, cb_ok[:, None]], 1)
    p_roles = torch.tensor([0, 1, 2, 3, 0, 0, 0, 4], device=X.device)
    n_roles = torch.tensor([5, 6, 7, 8, 9, 10, 11, 0], device=X.device)
    roles = torch.where(protein[:, None], p_roles, n_roles)
    p_elements = torch.tensor([7, 6, 6, 8, 0, 0, 0, 6], device=X.device)
    n_elements = torch.tensor([15, 6, 7, 8, 8, 6, 6, 0], device=X.device)
    elements = torch.where(protein[:, None], p_elements, n_elements)
    return xyz, ok, roles, elements, R, frame_ok, node_centers(X, valid)


class AtomWeights(nn.Module):
    """One atom's contribution to one residue.

    The distance goes through radial basis functions; the element, the atom's
    role and the receiving residue's polymer type through embeddings. The
    cosine factor takes the contribution smoothly to zero at the cutoff, so an
    atom entering or leaving the neighbourhood does not step the output.
    """

    def __init__(self, channels: int, radius: float):
        super().__init__()
        self.radius = float(radius)
        self.role = nn.Embedding(N_ROLES, 8)
        self.element = nn.Embedding(119, 8)        # element 0 is unresolved
        self.receiver_type = nn.Embedding(3, 4)
        self.register_buffer('centres', torch.linspace(0, radius, N_RADIAL))
        width = N_RADIAL + 8 + 8 + 4
        self.net = nn.Sequential(nn.Linear(width, 48), nn.SiLU(),
                                 nn.Linear(48, channels))

    def forward(self, distance, role, element, receiver_type):
        radial = torch.exp(-((distance[..., None] - self.centres)
                             / (self.radius / N_RADIAL)) ** 2)
        typ = self.receiver_type(receiver_type)[:, None].expand(
            -1, distance.shape[1], -1)
        x = torch.cat([radial, self.role(role), self.element(element), typ], -1)
        cutoff = 0.5 * (1 + torch.cos(math.pi
                                      * (distance / self.radius).clamp(0, 1)))
        return F.softplus(self.net(x)) * cutoff[..., None]


class CofactorContext(nn.Module):
    """The cofactor atoms around each residue, as one vector per residue.

    The per-atom contributions are averaged over the neighbourhood and the
    count is handed over alongside, so a residue with one nearby atom and a
    residue surrounded by them are distinguishable. A residue with no atom
    nearby gets zeros.
    """

    def __init__(self, hidden: int, channels: int = 8, radius: float = 16.0):
        super().__init__()
        self.weights = AtomWeights(channels, radius)
        self.out = _mlp(channels + 1, hidden, hidden)

    def forward(self, batch, centre):
        h = centre.new_zeros((len(centre), self.out[-1].out_features))
        edge = batch.get('context_edge_idx')
        if edge is None or not edge.numel():
            return h
        recv, atom = edge
        delta = batch['context_xyz'][atom].float() - centre[recv]
        dist = delta.norm(dim=-1)
        valid = torch.isfinite(delta).all(-1) & (dist < self.weights.radius)
        dist = torch.where(valid, dist, torch.zeros_like(dist))
        role = torch.full_like(atom[:, None], CONTEXT_ROLE)
        w = self.weights(dist[:, None], role,
                         batch['context_element'][atom, None],
                         batch['type_vec'][recv])[:, 0].float()
        total = w.new_zeros((len(centre), w.shape[-1])).index_add(
            0, recv, w * valid[:, None])
        count = dist.new_zeros(len(centre)).index_add(0, recv, valid.float())
        feat = torch.cat([total / count.clamp_min(1)[:, None],
                          count.log1p()[:, None]], -1)
        return self.out(feat).to(h.dtype) * (count > 0)[:, None]
