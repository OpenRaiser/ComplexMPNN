#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input collection, design masks, logit bias and output writing."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".mmcif", ".mmcif.gz", ".pdb", ".pdb.gz")

# ------------------------------------------------------------------ chain layout

def _spans_to_list(spans: Any, n_nodes: int) -> List[Tuple[int, int]]:
    """Normalize the featurizer's chain_spans into [(start, end), ...]."""
    if isinstance(spans, list) and len(spans) == 1:
        spans = spans[0]
    if spans is None:
        return [(0, n_nodes)]
    if isinstance(spans, torch.Tensor):
        arr = spans.detach().cpu().long().numpy()
    else:
        arr = np.asarray(spans)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return [(0, n_nodes)]
    return [(int(s), int(e)) for s, e in arr.tolist() if int(e) > int(s)]

# ------------------------------------------------------------------- design pass

FORBID = -1e9

# Lehninger's nonpolar side chains, the ones whose cost on the surface is paid
# in solubility rather than in fold.
NONPOLAR = "FILMVWY"
# What counts as exposed: fewer than SURFACE_NEIGHBOURS representative atoms of
# other residues within SURFACE_RADIUS angstroms.
SURFACE_NEIGHBOURS = 14.0
SURFACE_RADIUS = 10.0

def burial_counts(batch, radius: float):
    """Representative atoms of other residues within `radius` of each residue."""
    X = batch["X"].detach().cpu().float()
    rep = X[:, 1, :]
    good = rep.abs().sum(-1) > 0
    counts = torch.full((rep.shape[0],), float("nan"))
    pts = rep[good]
    if pts.shape[0] < 2:
        # a residue that has its atom and no neighbour has none, which is a
        # count of zero and not an unknown
        counts[good] = 0.0
        return counts
    step = max(1, int(4e7 // pts.shape[0]))
    hits = torch.zeros(pts.shape[0])
    for i in range(0, pts.shape[0], step):
        d = torch.cdist(pts[i:i + step], pts)
        hits[i:i + step] = (d <= radius).sum(1).float() - 1.0   # drop itself
    counts[good] = hits
    return counts

def build_logit_bias(batch, tokenizer, args, device):
    """A [nodes, vocab] bias added to the logits before the temperature."""
    omit = [c for c in (args.omit_AA or "").upper() if c.strip()]
    pairs = []
    for part in (args.bias_AA or "").split(","):
        part = part.strip()
        if not part:
            continue
        aa, _, val = part.partition(":")
        aa = aa.strip().upper()
        if not aa or not val.strip():
            raise SystemExit(f"--bias_AA wants LETTER:VALUE, got {part!r}")
        pairs.append((aa, float(val)))
    surface = float(args.surface_bias or 0.0)
    if not omit and not pairs and surface == 0.0:
        return None

    n_nodes = int(batch["S"].shape[0])
    vocab = int(tokenizer.vocab_size)
    bias = torch.zeros(n_nodes, vocab)
    is_prot = (batch["type_vec"].detach().cpu().long() == 0)

    def tok(aa):
        if aa not in tokenizer.protein_to_id:
            raise SystemExit(f"{aa!r} is not one of the twenty amino acids")
        return tokenizer.protein_to_id[aa]

    for aa in omit:
        bias[is_prot, tok(aa)] = FORBID
    for aa, val in pairs:
        bias[is_prot, tok(aa)] += val

    n_exposed = 0
    if surface != 0.0:
        counts = burial_counts(batch, SURFACE_RADIUS)
        exposed = is_prot & (counts < SURFACE_NEIGHBOURS)
        n_exposed = int(exposed.sum())
        for aa in NONPOLAR:
            bias[exposed, tok(aa)] += surface

    print(f"[bias] omit={''.join(omit) or '-'} bias_AA={len(pairs)} "
          f"surface={surface:+g} on {n_exposed} exposed position(s) "
          f"(< {SURFACE_NEIGHBOURS:g} neighbours within "
          f"{SURFACE_RADIUS:g} A)")
    return bias.to(device)

# ------------------------------------------------------------------ input files

def collect_inputs(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.name.lower().endswith(STRUCTURE_SUFFIXES) else []
    if p.is_dir():
        out = [
            str(Path(d) / fn)
            for d, _, fns in os.walk(p)
            for fn in fns
            if fn.lower().endswith(STRUCTURE_SUFFIXES)
        ]
        return sorted(out)
    return []

def input_stem(path: str) -> str:
    name = Path(path).name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    for ext in (".cif", ".mmcif", ".pdb"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem

# -------------------------------------------------------------- design-mask file

def parse_design_mask_file(path: str) -> List[Tuple[str, str, List[int]]]:
    """Read a design-mask TSV: <input_prefix> <chain> <positions>."""
    rules: List[Tuple[str, str, List[int]]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split(None, 2)
            if len(parts) < 3:
                print(f"[mask] line {lineno}: expected 3 columns, got {len(parts)}, skipped")
                continue
            prefix, chain = parts[0].strip(), parts[1].strip().upper()
            positions: List[int] = []
            for tok in parts[2].split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok[1:]:
                    lo, hi = tok.split("-", 1)
                    try:
                        positions.extend(range(int(lo) - 1, int(hi)))
                    except ValueError:
                        print(f"[mask] line {lineno}: bad range '{tok}', skipped")
                    continue
                try:
                    positions.append(int(tok) - 1)
                except ValueError:
                    print(f"[mask] line {lineno}: bad position '{tok}', skipped")
            if positions:
                rules.append((prefix, chain, sorted(set(positions))))
    return rules

def lookup_design_positions(
    rules: Sequence[Tuple[str, str, List[int]]], name: str
) -> Optional[Dict[str, List[int]]]:
    """Positions requested for this input, as {chain: [pos_0based]}. None means no."""
    result: Dict[str, List[int]] = {}
    for prefix, chain, positions in rules:
        if prefix == "*" or name.startswith(prefix):
            result[chain] = sorted(set(result.get(chain, []) + list(positions)))
    return result or None

# ----------------------------------------------------------------------- outputs

def _near_other(X, valid, mine, theirs, cutoff):
    """Which of `mine`'s nodes have an atom within cutoff of one of `theirs`."""
    out = torch.zeros(X.shape[0], dtype=torch.bool)
    their_pts = X[theirs][valid[theirs]]
    if not their_pts.numel() or not bool(mine.any()):
        return out
    Xm, vm = X[mine], valid[mine]
    P, A = vm.shape
    flat, flatv = Xm.reshape(P * A, 3), vm.reshape(P * A)
    pts = flat[flatv]
    if not pts.numel():
        return out
    hit = torch.zeros(pts.shape[0], dtype=torch.bool)
    step = max(1, int(4e7 // max(1, their_pts.shape[0])))   # cap the distance block
    for i in range(0, pts.shape[0], step):
        hit[i:i + step] = (torch.cdist(pts[i:i + step], their_pts) <= cutoff).any(1)
    res = torch.zeros(P * A, dtype=torch.bool)
    res[flatv] = hit
    out[mine] = res.reshape(P, A).any(1)
    return out

def interface_nodes(batch, cutoff: float, partner: str = "nucleic"):
    """Polymer nodes with an atom within `cutoff` of the chosen partner."""
    X = batch["X"].detach().cpu().float()
    tv = batch["type_vec"].detach().cpu().long()
    out = torch.zeros(tv.shape[0], dtype=torch.bool)
    if cutoff <= 0:
        return out
    valid = X.abs().sum(-1) > 0

    if partner == "chain":
        polymer = tv != 3
        spans = _spans_to_list(batch.get("chain_spans"), tv.shape[0])
        if len(spans) < 2:
            return out
        for start, end in spans:
            mine = torch.zeros_like(polymer)
            mine[start:end] = True
            mine &= polymer
            theirs = polymer & ~mine
            out |= _near_other(X, valid, mine, theirs, cutoff)
        return out

    prot = tv == 0
    na = (tv == 1) | (tv == 2)
    if not bool(prot.any()) or not bool(na.any()):
        return out
    return _near_other(X, valid, prot, na, cutoff)

def write_outputs(records: List[Dict[str, Any]], out_dir: str, meta: Dict[str, Any]) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fasta_path = Path(out_dir) / "designed_sequences.fasta"
    with open(fasta_path, "w", encoding="utf-8") as fh:
        for rec in records:
            header = (
                f">{rec['input_name']}|chain={rec['chain']}|type={rec['type']}"
                f"|sample={rec['sample']}|score={rec['score']:.4f}"
                f"|recovery={rec['recovery']:.4f}|designed={rec['n_designed']}"
            )
            seq = rec["sequence"]
            fh.write(header + "\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")

    cols = [
        "input_name", "chain", "type", "sample", "score", "recovery",
        "n_designed", "recovery_interface", "n_interface", "length", "sequence",
        "native",
        "input_path",
    ]
    tsv_path = Path(out_dir) / "summary.tsv"
    with open(tsv_path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for rec in records:
            fh.write("\t".join(str(rec.get(c, "")) for c in cols) + "\n")

    recoveries = [r["recovery"] for r in records]
    weight = sum(r["n_designed"] for r in records)
    meta = dict(meta)
    meta.update(
        {
            "n_designed_chains": len(records),
            "n_inputs": len({r["input_name"] for r in records}),
            "mean_recovery": (sum(recoveries) / len(recoveries)) if recoveries else None,
            "mean_recovery_interface": (
                sum(r["recovery_interface"] for r in records
                    if r.get("recovery_interface") is not None)
                / max(1, sum(1 for r in records if r.get("recovery_interface") is not None))
            ) if any(r.get("recovery_interface") is not None for r in records) else None,
            "weighted_recovery": (
                sum(r["recovery"] * r["n_designed"] for r in records) / weight
            ) if weight else None,
            "fasta": str(fasta_path),
            "summary": str(tsv_path),
        }
    )
    metrics_path = Path(out_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\n=========== summary ===========")
    print(f"inputs designed   : {meta['n_inputs']}")
    if meta.get("refine_fraction") is not None:
        print(f"refine replaced   : {100 * meta['refine_fraction']:.2f}% of "
              f"{meta['refine_opportunities']} position-sweeps")
    print(f"chain records     : {meta['n_designed_chains']}")
    if meta["mean_recovery"] is not None:
        print(f"mean recovery     : {100 * meta['mean_recovery']:.2f}%")
        print(f"weighted recovery : {100 * meta['weighted_recovery']:.2f}%")
    if meta.get("mean_recovery_interface") is not None:
        n_if = sum(1 for r in records if r.get("recovery_interface") is not None)
        print(f"interface recovery: {100 * meta['mean_recovery_interface']:.2f}%  "
              f"({n_if} chain record(s) with an interface)")
    print(f"fasta             : {fasta_path}")
    print(f"summary           : {tsv_path}")
    print(f"metrics           : {metrics_path}")
