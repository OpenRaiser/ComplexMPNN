#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which chains a task designs.

The four tasks the benchmarks use: design everything, design the protein
against fixed nucleic acids, design the nucleic acids against a fixed
protein, or design one chain. Evaluation fixes one task for the whole split;
the weights below are the mix the released model was trained on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

NUCLEIC = ("rna", "dna")

# The tasks the benchmarks ask for, and the whole-complex task the model was
# trained on. Weights are relative and need not sum to one.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "all": 0.4,
    "protein_given_nucleic": 0.25,
    "nucleic_given_protein": 0.15,
    "one_chain": 0.2,
}


def applicable(task: str, kinds: Sequence[str]) -> bool:
    """Whether a task can be asked of this complex at all."""
    has_prot = any(k == "protein" for k in kinds)
    has_na = any(k in NUCLEIC for k in kinds)
    if task == "all":
        return len(kinds) > 0
    if task == "protein_given_nucleic":
        return has_prot and has_na
    if task == "nucleic_given_protein":
        return has_prot and has_na
    if task == "one_chain":
        return len(kinds) >= 2
    return False


def chains_for_task(task: str, kinds: Sequence[str], rng,
                    seqs: Optional[Sequence[str]] = None) -> List[bool]:
    """Which chains the task designs, one flag per chain."""
    if task == "all":
        return [True] * len(kinds)
    if task == "protein_given_nucleic":
        return [k == "protein" for k in kinds]
    if task == "nucleic_given_protein":
        return [k in NUCLEIC for k in kinds]
    if task == "one_chain":
        pick = int(rng.integers(len(kinds))) if hasattr(rng, "integers") \
            else int(rng.randrange(len(kinds)))
        if seqs is None:
            return [i == pick for i in range(len(kinds))]
        if len(seqs) != len(kinds):
            raise ValueError("seqs and kinds disagree on the chain count")
        target = seqs[pick]
        return [s == target for s in seqs]
    raise ValueError(f"unknown task: {task}")


def sample_design_mask(kinds: Sequence[str], rng,
                       weights: Dict[str, float] = None,
                       seqs: Optional[Sequence[str]] = None) -> (str, List[bool]):
    """(task, one flag per chain) for one training example."""
    if not kinds:
        return "all", []
    weights = DEFAULT_WEIGHTS if weights is None else weights
    live = [(t, w) for t, w in weights.items() if w > 0 and applicable(t, kinds)]
    if not live:
        return "all", [True] * len(kinds)
    total = sum(w for _, w in live)
    draw = float(rng.random()) * total
    for task, w in live:
        draw -= w
        if draw <= 0:
            return task, chains_for_task(task, kinds, rng, seqs)
    task = live[-1][0]
    return task, chains_for_task(task, kinds, rng, seqs)
