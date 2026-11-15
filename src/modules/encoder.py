# -*- coding: utf-8 -*-
"""The encoder: geometric edge features, then six residue and fragment blocks.

The encoder sees coordinates only. Sequence enters the model in the decoder,
so this output can be computed once and reused at every generation step.

One block is a residue layer followed by a fragment layer. The residue layer
passes messages along the k-nearest-neighbour graph. The fragment layer groups
up to eight consecutive residues, lets the groups exchange messages over the
same graph, and returns the result to the residues.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.modules.atom_context import CofactorContext, backbone_atoms
from src.tools.geometry import (
    NUCLEIC, PROTEIN, SLOTS, anchor_pairs, atom_table, atom_validity,
    init_virtual_coeffs, intra_pairs, local_directions, n_anchors, n_atoms,
    pair_rbf, relative_position, residue_frames, torsion_width, virtual_pairs,
)
from src.tools.segment import segment_softmax, segment_sum

# Protein and nucleic acid are the two geometry classes: RNA and DNA share one
# set of atoms and one set of projections.
CLASSES = (PROTEIN, NUCLEIC)
N_RELATIONS = len(CLASSES) ** 2

FRAGMENT_SIZE = 8


def relation_index(cls_recv: torch.Tensor, cls_send: torch.Tensor) -> torch.Tensor:
    """The ordered class pair as one index: pp, pn, np or nn."""
    return cls_recv * len(CLASSES) + cls_send


def build_MLP(n_layers: int, d_in: int, d_hidden: int, d_out: int,
              activation=nn.GELU) -> nn.Sequential:
    if n_layers < 1:
        raise ValueError('an MLP needs at least one layer')
    layers = []
    d = d_in
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, d_hidden), activation()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


def node_feature_width(cls: int, n_rbf: int) -> int:
    """The width of the per-residue features.

    The released weights hold a projection of this width but do not read it:
    they were trained with the nodes starting empty, and the geometry reaching
    them through the edges. The width is still needed to build the projection
    that a strict load has to find.
    """
    return (len(intra_pairs(cls)) * n_rbf + len(SLOTS[cls]) + 1
            + torsion_width(cls))


def edge_feature_width(cls_recv: int, cls_send: int, n_virtual: int,
                       n_rbf: int) -> int:
    """Distances between every atom pair, plus directions and validity flags."""
    n_pairs = (len(anchor_pairs(cls_recv, cls_send))
               + len(virtual_pairs(n_virtual)))
    n_dirs = n_atoms(cls_send, n_virtual)
    return n_pairs * (n_rbf + 1) + n_dirs * 4


class GeometryEmbedding(nn.Module):
    """Edge representations from coordinates, computed once per structure.

    Each edge carries the distances between all atom pairs of its two
    residues, the directions of the sender's atoms in the receiver's local
    frame, a validity flag per distance and direction, the clipped sequence
    offset and whether the two residues share a chain. Protein-protein,
    protein-nucleotide, nucleotide-protein and nucleotide-nucleotide contacts
    have separate projections, because their atom sets differ.
    """

    def __init__(self, hidden_dim: int = 128, n_virtual: int = 3,
                 n_rbf: int = 16, d_min: float = 2.0, d_max: float = 22.0,
                 pos_clip: int = 32, n_polymer_types: int = 3):
        super().__init__()
        self.hidden_dim, self.n_virtual, self.n_rbf = hidden_dim, n_virtual, n_rbf
        self.d_min, self.d_max, self.pos_clip = d_min, d_max, pos_clip
        # Coefficients placing the virtual points, one set for protein and one
        # shared by RNA and DNA. Learned, following PiFold.
        self.virtual = nn.ParameterList([
            nn.Parameter(init_virtual_coeffs(n_virtual)) for _ in CLASSES])
        self.edge_proj = nn.ModuleList([
            build_MLP(2, edge_feature_width(a, b, n_virtual, n_rbf),
                      hidden_dim, hidden_dim)
            for a in CLASSES for b in CLASSES])
        self.pos_embed = nn.Embedding(2 * pos_clip + 1, hidden_dim)
        self.same_chain_embed = nn.Embedding(2, hidden_dim)
        # Built because the released state dict holds them. The forward below
        # does not read them: see node_feature_width.
        self.node_proj = nn.ModuleList([
            build_MLP(2, node_feature_width(c, n_rbf), hidden_dim, hidden_dim)
            for c in CLASSES])
        self.type_embed = nn.Embedding(n_polymer_types, hidden_dim)
        self.frame_embed = nn.Embedding(2, hidden_dim)

    def _edge_features(self, X, valid, R, t, frame_ok, cls_r, cls_s,
                       recv, send) -> torch.Tensor:
        a_r = atom_table(X, cls_r, self.virtual[cls_r].to(X.dtype))
        a_s = atom_table(X, cls_s, self.virtual[cls_s].to(X.dtype))
        ok_r = atom_validity(valid, cls_r, frame_ok, self.n_virtual)
        ok_s = atom_validity(valid, cls_s, frame_ok, self.n_virtual)
        anchors_r, anchors_s = n_anchors(cls_r), n_anchors(cls_s)
        pairs = anchor_pairs(cls_r, cls_s) + tuple(
            (anchors_r + i, anchors_s + j)
            for i, j in virtual_pairs(self.n_virtual))
        d, dmask = pair_rbf(a_r, a_s, ok_r, ok_s, recv, send, pairs,
                            self.n_rbf, self.d_min, self.d_max)
        dirs, vmask = local_directions(R, t, a_s, ok_s, recv, send, frame_ok)
        return torch.cat([d, dmask.to(d.dtype), dirs, vmask.to(d.dtype)], dim=-1)

    def frames(self, batch: Dict[str, torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The local frame of every residue, from slots 0, 1 and 2."""
        return residue_frames(batch['X'][:, 0], batch['X'][:, 1],
                              batch['X'][:, 2], batch['slot_valid'])

    def forward(self, batch: Dict[str, torch.Tensor]):
        """Empty nodes, geometric edges, and the relation index per edge."""
        X, valid = batch['X'], batch['slot_valid']
        geo_cls, edge_idx = batch['geo_cls'], batch['edge_idx']
        recv, send = edge_idx[0], edge_idx[1]
        N, E, H = X.shape[0], recv.shape[0], self.hidden_dim

        R, t, frame_ok = self.frames(batch)
        h_V = X.new_zeros((N, H))
        h_E = X.new_zeros((E, H))

        relation = relation_index(geo_cls[recv], geo_cls[send])
        for cls_r in CLASSES:
            for cls_s in CLASSES:
                rel = cls_r * len(CLASSES) + cls_s
                sel = (relation == rel).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                feats = self._edge_features(X, valid, R, t, frame_ok, cls_r,
                                            cls_s, recv[sel], send[sel])
                h_E[sel] = self.edge_proj[rel](feats).to(h_E.dtype)
        off, same = relative_position(batch['res_idx'], batch['chain_idx'],
                                      recv, send, self.pos_clip)
        h_E = h_E + self.pos_embed(off) + self.same_chain_embed(same.long())
        return h_V, h_E, relation, frame_ok


class ResidueLayer(nn.Module):
    """Attention over a residue's neighbours, then an edge update.

    The message and its weight are read from the receiver, the edge and the
    sender together. The weights are a softmax over each residue's incoming
    edges, in four heads, so a residue can weight a contact inside its chain
    differently from a contact with a partner chain.
    """

    def __init__(self, hidden_dim: int = 128, n_heads: int = 4,
                 dropout: float = 0.1, n_relations: int = N_RELATIONS):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError('hidden_dim must divide by n_heads')
        self.n_heads, self.hidden_dim = n_heads, hidden_dim
        self.value = build_MLP(3, 3 * hidden_dim, hidden_dim, hidden_dim)
        self.bias = build_MLP(3, 3 * hidden_dim, hidden_dim, n_heads)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn = build_MLP(2, hidden_dim, hidden_dim * 4, hidden_dim)
        self.edge = build_MLP(3, 3 * hidden_dim, hidden_dim, hidden_dim)
        # Which polymer types the edge connects, receiver first, added to the
        # edge update.
        self.relation_bias = nn.Embedding(n_relations, hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])
        self.drop = nn.Dropout(dropout)
        # In the released state dict and not read here: it belongs to a
        # training-time masking of residue identities.
        self.mask_token = nn.Embedding(2, hidden_dim)

    def forward(self, h_V, h_E, edge_idx, relation):
        recv, send = edge_idx[0], edge_idx[1]
        N, E = h_V.shape[0], h_E.shape[0]
        d = self.hidden_dim // self.n_heads

        pair = torch.cat([h_V[recv], h_E, h_V[send]], dim=-1)
        value = self.value(pair).view(E, self.n_heads, d)
        weight = segment_softmax(
            self.bias(pair).view(E, self.n_heads, 1) / math.sqrt(d), recv, N)
        agg = segment_sum(weight * value, recv, N).reshape(N, self.hidden_dim)

        h_V = self.norms[0](h_V + self.drop(self.out(agg)))
        h_V = self.norms[1](h_V + self.drop(self.ffn(h_V)))
        h_E = self.norms[2](h_E + self.drop(
            self.edge(torch.cat([h_V[recv], h_E, h_V[send]], dim=-1))
            + self.relation_bias(relation)))
        h_V = self.norms[3](h_V)
        return h_V, h_E


def fragment_plan(batch, size: int = FRAGMENT_SIZE):
    """Group consecutive residues, and connect the groups.

    A group holds at most `size` residues and never crosses a complex, a chain
    or a backbone break: the featuriser raises the residue index by 100 across
    a break, so a break shows up here as a gap in the index.

    Two groups are connected when any residue edge runs between them, keeping
    the direction of that edge. Groups that follow one another in an unbroken
    chain are connected both ways. Duplicates and self-edges are dropped.

    Returns (owner, slot, count, edges): the group each residue belongs to,
    its position inside that group, the number of groups, and the group edges.
    """
    r, c, b = batch['res_idx'], batch['chain_idx'], batch['batch_id']
    n = len(r)
    start = torch.ones(n, dtype=torch.bool, device=r.device)
    start[1:] = (b[1:] != b[:-1]) | (c[1:] != c[:-1]) | (r[1:] != r[:-1] + 1)
    idx = torch.arange(n, device=r.device)
    offset = idx - torch.where(start, idx, torch.zeros_like(idx)).cummax(0).values
    first = start | (offset.remainder(size) == 0)
    owner = first.long().cumsum(0) - 1
    slot = offset.remainder(size)
    count = int(owner[-1]) + 1 if n else 0

    edges = batch['edge_idx']
    src = torch.stack((owner[edges[0]], owner[edges[1]]))
    joins = torch.where(first & ~start)[0]
    seq = torch.stack((owner[joins], owner[joins - 1]))
    src = torch.cat((src, seq, seq.flip(0)), 1)
    src = torch.unique(src[:, src[0] != src[1]], dim=1)
    return owner, slot, count, src


class FragmentLayer(nn.Module):
    """Message passing between groups of consecutive residues.

    A group's representation is built from its residues in chain order, with
    the unused slots zeroed and a flag vector marking which are filled. Groups
    then exchange messages, and each residue reads back its own group's
    representation together with its position inside it.
    """

    def __init__(self, h: int, dropout: float, size: int = FRAGMENT_SIZE):
        super().__init__()
        self.size = size
        self.pack = mlp(size * h + size, 2 * h, h)
        self.message = mlp(2 * h, h, h)
        self.unpack = mlp(2 * h + size, 2 * h, h)
        self.edge = mlp(3 * h, h, h)
        self.norm = nn.LayerNorm(h)
        self.edge_norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)

    def forward(self, nodes, edge_state, edges, plan):
        owner, slot, count, graph = plan
        packed = nodes.new_zeros((count, self.size, nodes.shape[-1]))
        packed[owner, slot] = nodes
        filled = nodes.new_zeros((count, self.size))
        filled[owner, slot] = 1
        p = self.pack(torch.cat((packed.flatten(1), filled), -1))

        recv, send = graph
        msg = self.message(torch.cat((p[recv], p[send]), -1))
        total = msg.new_zeros((count, msg.shape[-1])).index_add(0, recv, msg)
        degree = torch.bincount(recv, minlength=count).clamp_min(1)[:, None]
        p = p + total / degree.to(msg.dtype)

        pos = torch.nn.functional.one_hot(slot, self.size).to(nodes.dtype)
        nodes = self.norm(nodes + self.drop(
            self.unpack(torch.cat((nodes, p[owner], pos), -1))))
        recv, send = edges
        edge_state = self.edge_norm(edge_state + self.drop(self.edge(
            torch.cat((nodes[recv], edge_state, nodes[send]), -1))))
        return nodes, edge_state


class EncoderStack(nn.Module):
    """The encoder blocks: a residue layer and a fragment layer each.

    The attribute holding this module is named `research_graph` in the
    Encoder, because that is the prefix the released state dict uses.
    """

    def __init__(self, h: int, depth: int, dropout: float, heads: int = 4):
        super().__init__()
        self.local = nn.ModuleList([ResidueLayer(h, heads, dropout)
                                    for _ in range(depth)])
        self.layers = nn.ModuleList([FragmentLayer(h, dropout)
                                     for _ in range(depth)])

    def forward(self, nodes, edges, batch, relation):
        plan = fragment_plan(batch)
        for residue, fragment in zip(self.local, self.layers):
            nodes, edges = residue(nodes, edges, batch['edge_idx'], relation)
            nodes, edges = fragment(nodes, edges, batch['edge_idx'], plan)
        return nodes, edges


class Encoder(nn.Module):
    """Geometric features, the cofactor context, then the encoder blocks."""

    def __init__(self, hidden_dim: int = 128, n_layers: int = 6,
                 n_heads: int = 4, dropout: float = 0.1, n_virtual: int = 3,
                 n_rbf: int = 16, pos_clip: int = 32,
                 n_polymer_types: int = 3,
                 fixed_atom_context: bool = False):
        super().__init__()
        self.geometry = GeometryEmbedding(hidden_dim, n_virtual, n_rbf,
                                          pos_clip=pos_clip,
                                          n_polymer_types=n_polymer_types)
        self.research_graph = EncoderStack(hidden_dim, n_layers, dropout,
                                          n_heads)
        self.hidden_dim = hidden_dim
        self.fixed_atom_context = (CofactorContext(hidden_dim)
                                   if fixed_atom_context else None)

    def forward(self, batch: Dict[str, torch.Tensor]):
        h_V, h_E, relation, _ = self.geometry(batch)
        if self.fixed_atom_context is not None:
            centre = backbone_atoms(batch)[6]
            h_V = h_V + self.fixed_atom_context(batch, centre).to(h_V.dtype)
        return self.research_graph(h_V, h_E, batch, relation)

    def parameter_report(self) -> Dict[str, int]:
        geometry = sum(p.numel() for p in self.geometry.parameters())
        blocks = sum(p.numel() for p in self.research_graph.parameters())
        context = sum(p.numel() for p in self.parameters()) - geometry - blocks
        return dict(geometry=geometry, blocks=blocks, cofactor_context=context,
                    n_blocks=len(self.research_graph.layers))
