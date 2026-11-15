# -*- coding: utf-8 -*-
"""From parsed chains to the tensors the model reads.

A structure becomes one node per residue. Each node carries up to seven atom
slots, which atoms those are depends on the polymer type, a validity flag per
slot, a token, a polymer type, a chain index and a position index. The
position index jumps at a chain break, so nothing downstream treats the two
sides of a break as neighbours in sequence.

`collate_complex` concatenates the nodes of several structures, builds the
k-nearest-neighbour graph within each one, and attaches the cofactor atoms.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from src.tools.geometry import (
    TYPE_TO_CLASS, NUCLEIC, PROTEIN, geometry_class, knn_edges, node_centers,
)

PRIME_IN_MAP = {
    "P": ("P",),
    "O5": ("O5", "O5'", "O5*"),
    "C5": ("C5", "C5'", "C5*"),
    "C4": ("C4", "C4'", "C4*"),
    "C3": ("C3", "C3'", "C3*"),
    "O3": ("O3", "O3'", "O3*"),
    "N": ("N_nuc", "N'", "N*", "N"),
}

SLOT_NAMES: Dict[str, Tuple[Optional[str], ...]] = {
    'protein': ('N', 'CA', 'C', 'O', None, None, None),
    'rna': ('P', 'C4', 'N', 'O5', 'O3', 'C5', 'C3'),
    'dna': ('P', 'C4', 'N', 'O5', 'O3', 'C5', 'C3'),
}
N_SLOTS = 7
REFERENCE_SLOT = 1          # CA for a protein, C4' for a nucleic acid
NUCLEIC_KINDS = ('rna', 'dna')

TYPE_ID = {'protein': 0, 'rna': 1, 'dna': 2}
assert tuple(TYPE_TO_CLASS) == (PROTEIN, NUCLEIC, NUCLEIC)

BREAK_CUTOFF = {'protein': 4.5, 'rna': 7.5, 'dna': 7.5}
BREAK_OFFSET = 100          # added to the position index at an inferred break


def chain_slots(kind: str, coords: Dict[str, Sequence], length: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """[L, 6, 3] coordinates and [L, 6] validity for one chain."""
    names = SLOT_NAMES.get(kind)
    if names is None:
        raise ValueError(f'unknown chain kind: {kind}')
    X = torch.zeros(length, N_SLOTS, 3, dtype=torch.float32)
    valid = torch.zeros(length, N_SLOTS, dtype=torch.bool)
    for s, name in enumerate(names):
        if name is None:
            continue
        arr = coords.get(name)
        if arr is None:
            continue
        a = torch.as_tensor(np.asarray(arr, dtype=np.float32))
        if a.shape != (length, 3):
            raise ValueError(f'{kind} atom {name}: expected ({length}, 3), got {tuple(a.shape)}')
        ok = torch.isfinite(a).all(dim=-1)
        X[:, s, :] = torch.where(ok[:, None], a, torch.zeros_like(a))
        valid[:, s] = ok
    return X, valid


def segment_index(x_ref: torch.Tensor, valid_ref: torch.Tensor, kind: str
                  ) -> torch.Tensor:
    """A position index that jumps at an inferred chain break. [L] long."""
    L = x_ref.shape[0]
    if L == 0:
        return torch.zeros(0, dtype=torch.long)
    cutoff = BREAK_CUTOFF[kind]
    d = (x_ref[1:] - x_ref[:-1]).norm(dim=-1)
    joined = valid_ref[1:] & valid_ref[:-1] & (d <= cutoff)
    step = torch.ones(L, dtype=torch.long)
    step[0] = 0
    step[1:] = torch.where(joined, torch.ones(L - 1, dtype=torch.long),
                           torch.full((L - 1,), BREAK_OFFSET, dtype=torch.long))
    return torch.cumsum(step, dim=0)


NODE_KEYS = ('X', 'slot_valid', 'S', 'type_vec', 'res_idx', 'design_mask',
             'loss_mask')


def collate_complex(samples: Sequence[dict], knn_k: int = 32) -> dict:
    """Concatenate along the node axis and build the neighbour graph."""
    samples = [s for s in samples if s is not None]
    if not samples:
        raise ValueError('an empty batch reached the collate function')
    out = {k: torch.cat([s[k] for s in samples]) for k in NODE_KEYS}
    chains, batch, offset = [], [], 0
    for b, s in enumerate(samples):
        chains.append(s['chain_idx'] + offset)
        offset += int(s['chain_idx'].max()) + 1
        batch.append(torch.full((s['S'].shape[0],), b, dtype=torch.long))
    out['chain_idx'] = torch.cat(chains)
    out['batch_id'] = torch.cat(batch)
    if any('context_xyz' in s for s in samples):
        from src.datasets.fixed_atom_context import attach_context_edges
        context_xyz, context_element, context_batch = [], [], []
        for b, s in enumerate(samples):
            xyz = s.get('context_xyz', torch.empty((0, 3)))
            context_xyz.append(xyz)
            context_element.append(s.get('context_element', torch.empty(0, dtype=torch.long)))
            context_batch.append(torch.full((len(xyz),), b, dtype=torch.long))
        out['context_xyz'] = torch.cat(context_xyz)
        out['context_element'] = torch.cat(context_element)
        out['context_batch_id'] = torch.cat(context_batch)
    out['geo_cls'] = geometry_class(out['type_vec'])
    out['center'] = node_centers(out['X'], out['slot_valid'])
    out['edge_idx'] = knn_edges(out['center'], out['batch_id'], knn_k)
    if 'context_xyz' in out:
        attach_context_edges(out)
    out['ids'] = [s['example_id'] for s in samples]
    return out
