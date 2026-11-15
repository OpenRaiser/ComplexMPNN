# -*- coding: utf-8 -*-
"""Fitting a complex into a residue budget.

Training cropped every example to at most 3,000 residues. Evaluation keeps the
same budget, so an entry larger than that is cropped the same way it was
during training rather than being handled by a second code path. Anything
inside the budget passes through whole, which is every target in the released
benchmarks.

The designed chains are placed first, then the chains the task needs as
context, then whatever else is in contact, nearest first. A chain that does
not fit whole contributes a window around its closest approach to what is
already kept.
"""
from __future__ import annotations

from collections import Counter
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.datasets.structure_input import NUCLEIC_KINDS

REQUIRED_CONTEXT = {
    'protein_given_nucleic': NUCLEIC_KINDS,
    'nucleic_given_protein': ('protein',),
}


class ConditionDropped(ValueError):
    """The budget cannot hold the designed chains and the task's partner."""


def required_context_flags(task: str, kinds: Sequence[str],
                           designed: Sequence[bool]) -> List[bool]:
    """Per chain, whether the task requires it as context."""
    want = REQUIRED_CONTEXT.get(task, ())
    return [bool(not d and k in want) for k, d in zip(kinds, designed)]


def _share_out(lengths: Sequence[int], budget: int,
               floors: Sequence[int]) -> List[int]:
    """Budget split in proportion to length, each at least its own floor."""
    if len(floors) != len(lengths):
        raise ValueError('floors and lengths disagree on length')
    total = sum(lengths)
    if total <= budget:
        return list(lengths)
    if sum(floors) > budget:
        raise ConditionDropped(
            f'budget {budget} cannot hold {len(lengths)} chains at their '
            f'floors {list(floors)}')
    out = [max(f, int(budget * n / total)) for n, f in zip(lengths, floors)]
    # Rounding can overshoot. Take the excess off whichever allocation has the
    # most room above its own floor, so the partner is not the one that pays.
    over = sum(out) - budget
    while over > 0:
        i = max(range(len(out)), key=lambda j: (out[j] - floors[j], lengths[j]))
        if out[i] <= floors[i]:
            raise ConditionDropped('cannot fit the chains at their floors')
        take = min(over, out[i] - floors[i])
        out[i] -= take
        over -= take
    return out


def _closest(ref: List[torch.Tensor], ok: List[torch.Tensor],
             anchor: torch.Tensor, candidates: Sequence[int]) -> Tuple[float, int]:
    """The candidate chain closest to the anchor residues, and how close."""
    best = (float('inf'), -1)
    if anchor.numel() == 0:
        return (float('inf'), min(candidates)) if candidates else best
    for i in sorted(candidates):
        c = ref[i][ok[i]]
        if c.numel() == 0:
            continue
        d = float(torch.cdist(c, anchor).min())
        if d < best[0]:
            best = (d, i)
    return best if best[1] >= 0 else (float('inf'), min(candidates))


def _window_near(c: torch.Tensor, ok: torch.Tensor, anchor: torch.Tensor,
                 width: int) -> Tuple[int, int]:
    """A contiguous window of `width` residues around the closest approach."""
    L = c.shape[0]
    if width >= L:
        return 0, L
    centre = L // 2
    if ok.any() and anchor.numel():
        d = torch.full((L,), float('inf'))
        d[ok] = torch.cdist(c[ok], anchor).min(dim=1).values
        centre = int(torch.argmin(d))
    start = max(0, min(centre - width // 2, L - width))
    return start, start + width


def crop_to_budget(kinds: Sequence[str], ref: List[torch.Tensor],
                   ref_ok: List[torch.Tensor], designed: Sequence[bool],
                   budget: int, rng: random.Random,
                   contact_cutoff: float = 12.0, scarce_first: bool = True,
                   required_context: Optional[Sequence[bool]] = None,
                   min_required: int = 4
                   ) -> Tuple[List[Tuple[int, int, int]], Dict[str, int]]:
    """Which residues of which chains to keep, and a record of what it cost."""
    if budget <= 0:
        raise ValueError('the node budget must be positive')
    n = len(kinds)
    if not (len(ref) == len(ref_ok) == len(designed) == n):
        raise ValueError('chain lists disagree on length')
    if not any(designed):
        raise ValueError('every sample must design at least one chain')
    required_context = list(required_context or [False] * n)
    if len(required_context) != n:
        raise ValueError('required_context disagrees on the chain count')
    if any(r and d for r, d in zip(required_context, designed)):
        raise ValueError('a chain cannot be both designed and required context')

    design_idx = [i for i in range(n) if designed[i]]
    rng.shuffle(design_idx)
    if scarce_first:
        design_idx.sort(key=lambda i: 0 if kinds[i] in NUCLEIC_KINDS else 1)
    partner_idx = [i for i in range(n) if required_context[i]]

    rec = Counter()
    order = design_idx + partner_idx
    # A designed chain only has to exist; a chain the task is defined by has to
    # be present in a useful amount.
    floors = [1] * len(design_idx) + [min_required] * len(partner_idx)
    widths = _share_out([ref[i].shape[0] for i in order], budget, floors)
    if sum(widths) < sum(ref[i].shape[0] for i in order):
        rec['required_set_over_budget'] = 1

    kept: List[Tuple[int, int, int]] = []
    anchor_parts: List[torch.Tensor] = []
    for i, width in zip(order, widths):
        length = ref[i].shape[0]
        if width >= length:
            a, b = 0, length
        elif anchor_parts:
            a, b = _window_near(ref[i], ref_ok[i], torch.cat(anchor_parts), width)
        else:
            a = rng.randrange(0, length - width + 1)
            b = a + width
        kept.append((i, a, b))
        if b - a < length:
            key = 'designed' if designed[i] else 'required'
            rec[f'{key}_chains_cropped'] += 1
            rec[f'{key}_cropped_{kinds[i]}'] += 1
        anchor_parts.append(ref[i][a:b][ref_ok[i][a:b]])

    remaining = budget - sum(b - a for _, a, b in kept)
    optional = {i for i in range(n) if not designed[i] and not required_context[i]}
    while optional:
        anchor = torch.cat(anchor_parts) if anchor_parts else torch.zeros(0, 3)
        d, i = _closest(ref, ref_ok, anchor, optional)
        if remaining <= 0:
            rec['context_chains_dropped'] += len(optional)
            for j in optional:
                rec[f'context_dropped_{kinds[j]}'] += 1
            break
        if d > contact_cutoff:
            # Distances to the kept set only shrink as it grows, so if the
            # nearest remaining chain is out of contact, every other one is too.
            rec['context_chains_out_of_contact'] += len(optional)
            for j in optional:
                rec[f'context_out_of_contact_{kinds[j]}'] += 1
            break
        optional.discard(i)
        length = ref[i].shape[0]
        if length <= remaining:
            kept.append((i, 0, length))
        else:
            a, b = _window_near(ref[i], ref_ok[i], anchor, remaining)
            kept.append((i, a, b))
            rec['context_chains_cropped'] += 1
            rec[f'context_cropped_{kinds[i]}'] += 1
        remaining -= kept[-1][2] - kept[-1][1]
        anchor_parts.append(ref[i][kept[-1][1]:kept[-1][2]][ref_ok[i][kept[-1][1]:kept[-1][2]]])

    rec['chains_kept'] = len(kept)
    for i, a, b in kept:
        rec[f'kept_{kinds[i]}'] += 1
        rec[f'kept_nodes_{kinds[i]}'] += b - a
    rec['nodes'] = sum(b - a for _, a, b in kept)
    return kept, dict(rec)
