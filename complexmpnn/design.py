#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Design sequences for a backbone."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from complexmpnn import common           # shared with evaluate.py
from src.datasets.tokenizer_complex import Unitokenizer_Complex
from src.models.complexmpnn import ComplexMPNN

try:
    from src.tools.geometry import FEATURE_VERSION
except ImportError:
    FEATURE_VERSION = 1
from src.datasets.structure_input import (
    REFERENCE_SLOT, SLOT_NAMES, TYPE_ID, chain_slots, collate_complex,
    segment_index,
)


# ---------------------------------------------------------------- model loading

ARCHITECTURE = {
    'hidden_dim': 'hidden_dim',
    'encoder_layers': 'encoder_layers',
    'ar_layers': 'ar_layers',
    'heads': 'n_heads',
    'virtual_points': 'n_virtual',
    'flow_remask': 'flow_remask',
    'fixed_atom_context': 'fixed_atom_context',
}
# Settings this code does not implement, with the value it assumes for each.
# A checkpoint that sets one of them is a different network: most of its
# weights would still fit, so it would load and answer as something it is not.
# Refused by name instead.
REMOVED = {
    'decoder': 'flow',
    'decoder_recv_encoder': 0,
    'flow_variant': 'coupled',
    'encoder': 'v6',
    'encoder_operator': 'hierarchy',
    'node_init': 'zero',
    'node_geometry': 'full',
    'edge_virtual_mode': 'all',
    'base_method': 'ar',
    'potts': False,
    'joint_geometry': False,
    'global_geo': False,
    'chain_mixer': 'none',
    'graph_design': 'none',
    'fragment_pair': False,
    'sparse_triangle': False,
    'vector_channels': 0,
    'vector_magnitude': False,
    'decoder_atom_mode': 'none',
    'decoder_atom_readout': 'none',
    'decoder_unary_mode': 'none',
    'decoder_geo_queries': 0,
    'decoder_pair_dim': 0,
    'decoder_rel_aggregate': False,
    'decoder_fragment_exchange': False,
    'decoder_triangle': False,
}
# Saved as 0 or 1 and taken by the network as a flag.
BOOLEAN = ("fixed_atom_context",)


def load_model(ckpt: str, device: str):
    """Build the network the checkpoint was trained as and load its weights."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise SystemExit(f"{ckpt} is not a checkpoint dictionary")
    saved = dict(blob.get("args") or {})
    on = {k: saved[k] for k, off in REMOVED.items()
          if saved.get(k) is not None and saved[k] != off}
    if on:
        raise SystemExit(
            f"{ckpt} was trained with " +
            ", ".join(f"{k}={v!r}" for k, v in sorted(on.items())) +
            ". This release builds the network the published weights were "
            "trained as and does not carry those branches, so it cannot load "
            "this checkpoint.")

    kwargs: Dict[str, Any] = {}
    for arg, name in ARCHITECTURE.items():
        if arg in saved and saved[arg] is not None:
            kwargs[name] = bool(saved[arg]) if name in BOOLEAN else saved[arg]
    # Training-only settings the network must not be built with here.
    kwargs.update(dropout=0.0)
    saved_features = blob.get("feature_version", 1)
    if saved_features != FEATURE_VERSION:
        raise SystemExit(
            f"{ckpt} was written with feature version {saved_features} and "
            f"this code builds version {FEATURE_VERSION}. The node and edge "
            f"feature widths differ, so the weights do not fit.")

    model = ComplexMPNN(**kwargs)
    state = blob.get("model_state_dict") or blob.get("state_dict") or blob
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"{ckpt} does not fit the network it names:\n{exc}")

    dev = torch.device(device)
    model = model.to(dev).eval()
    report = model.parameter_report()
    shown = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    print(f"[model] {ckpt}")
    print(f"[model] {shown}")
    info = dict(args=saved, feature_version=saved_features)
    # A checkpoint may also carry where in training it came from. The released
    # one does not, and nothing here needs it, so it is reported when present
    # and left out otherwise.
    for field in ("epoch", "step", "best_valid_nll", "world_size"):
        if blob.get(field) is not None:
            info[field] = blob[field]
    training = "".join(f"  {k}={info[k]}" for k in
                       ("epoch", "step", "best_valid_nll") if k in info)
    print(f"[model] params={report['total']}  features=v{saved_features}"
          f"{training}  device={dev}")
    return model, dev, info


# ---------------------------------------------------------- structure to input

# Sugar atoms are named without their primes here and with them in the parsed
# structure. Two spellings of one atom, nothing more.
PRIMED = {'O5': "O5'", 'C5': "C5'", 'C4': "C4'", 'C3': "C3'", 'O3': "O3'"}

DEOXY_NAMES = {'DA', 'DC', 'DG', 'DT', 'DU', 'DI'}
RIBO_NAMES = {'A', 'C', 'G', 'U', 'I', 'URA', 'ADE', 'GUA', 'CYT'}
THYMINE_NAMES = {'DT', 'T', 'THY'}
RIBOSE_ATOMS = ("O2'", 'O2*', 'O2\u2019')


def nucleic_kind(chain) -> Tuple[str, str]:
    """RNA or DNA for one parsed chain, and what decided it."""
    deoxy = ribo = thymine = hydroxyl = 0
    for res in chain.get_residues():
        if res.id[0] != ' ':
            continue
        name = res.get_resname().strip().upper()
        deoxy += name in DEOXY_NAMES
        ribo += name in RIBO_NAMES
        thymine += name in THYMINE_NAMES
        hydroxyl += any(res.has_id(a) for a in RIBOSE_ATOMS)
    if deoxy > ribo:
        return 'dna', f'{deoxy} residue name(s) carry the deoxy prefix'
    if hydroxyl:
        return 'rna', f"{hydroxyl} residue(s) have a 2' hydroxyl"
    if thymine:
        return 'dna', f'{thymine} thymine residue(s)'
    return 'rna', 'nothing in the structure says which; read as RNA'


def parse_structure(path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Read a .pdb/.cif/.gz into the chain records the design path takes."""
    import re

    from inference.helpers import (extract_chain_seq_and_backbone,
                                   parse_structure_file)

    structure, _fmt, _text = parse_structure_file(Path(path))
    model = next(structure.get_models())

    chains: List[Dict[str, Any]] = []
    for chain in model:
        seq1, raw, is_na = extract_chain_seq_and_backbone(
            chain, ndigits=3, validate=True, drop_all_none_keys=False)
        if not is_na:
            seq = re.sub(r'[^A-Za-z]', '', seq1 or '').upper()
            kind, why = 'protein', ''
        else:
            kind, why = nucleic_kind(chain)
            upper = (seq1 or '').upper()
            if kind == 'dna':
                seq = ''.join(c if c in 'ATGCN' else 'N'
                              for c in upper.replace('U', 'T'))
            else:
                seq = ''.join(c if c in 'AUGCN' else 'N'
                              for c in upper.replace('T', 'U'))
        if not seq:
            continue
        if why and verbose:
            print(f'[parse] chain {chain.id}: {kind}, {why}')

        coords: Dict[str, np.ndarray] = {}
        for slot in SLOT_NAMES[kind]:
            if slot is None:
                continue
            column = raw.get(PRIMED.get(slot, slot))
            if column is None:
                continue
            arr = np.full((len(seq), 3), np.nan, dtype=np.float32)
            for i, xyz in enumerate(column[:len(seq)]):
                if xyz is not None:
                    arr[i] = xyz
            coords[slot] = arr
        chains.append(dict(kind=kind, chain_id=str(chain.id), seq=seq,
                           coords=coords))
    return chains


def build_sample(chains: Sequence[Dict[str, Any]], designed: Sequence[bool],
                 tokenizer: Unitokenizer_Complex, example_id: str) -> Dict[str, Any]:
    """One sample holding every residue of every chain, in file order."""
    X, valid, S, type_vec, chain_idx, res_idx, design = [], [], [], [], [], [], []
    for slot, chain in enumerate(chains):
        kind, seq = chain["kind"], chain["seq"]
        length = len(seq)
        Xi, vi = chain_slots(kind, chain["coords"], length)
        X.append(Xi)
        valid.append(vi)
        S.append(torch.tensor(tokenizer.encode(seq, kind), dtype=torch.long))
        type_vec.append(torch.full((length,), TYPE_ID[kind], dtype=torch.long))
        chain_idx.append(torch.full((length,), slot, dtype=torch.long))
        res_idx.append(segment_index(Xi[:, REFERENCE_SLOT, :],
                                     vi[:, REFERENCE_SLOT], kind))
        design.append(torch.full((length,), bool(designed[slot])))

    out = dict(X=torch.cat(X), slot_valid=torch.cat(valid), S=torch.cat(S),
               type_vec=torch.cat(type_vec), chain_idx=torch.cat(chain_idx),
               res_idx=torch.cat(res_idx), design_mask=torch.cat(design))
    unknown = (tokenizer.protein_X_id, tokenizer.nuc_N_id, tokenizer.mask_id)
    known = ~sum((out["S"] == t) for t in unknown).bool()
    out["loss_mask"] = (out["design_mask"] & known
                        & out["slot_valid"][:, REFERENCE_SLOT])
    out["example_id"] = example_id
    out["task"] = "design"
    out["crop"] = {}
    return out


def chain_layout(chains: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Where each chain's nodes are. Node order is file order, uncropped."""
    layout, start = [], 0
    for index, chain in enumerate(chains):
        end = start + len(chain["seq"])
        layout.append(dict(index=index, name=chain["chain_id"],
                           kind=chain["kind"], start=start, end=end,
                           is_cofactor=False))
        start = end
    return layout


def design_scope(chains: Sequence[Dict[str, Any]],
                 design_chains: Optional[Sequence[str]],
                 design_types: Optional[Sequence[str]]) -> List[bool]:
    """Which chains are in scope, one flag per chain."""
    want_chains = {c.strip().upper() for c in design_chains} if design_chains else None
    want_types = {t.strip().lower() for t in design_types} if design_types else None
    flags = []
    for chain in chains:
        if want_chains is None and want_types is None:
            flags.append(True)
            continue
        flags.append(bool(
            (want_chains is not None and chain["chain_id"].upper() in want_chains)
            or (want_types is not None and chain["kind"] in want_types)))
    return flags


def restrict_to_positions(sample: Dict[str, Any], layout: Sequence[Dict[str, Any]],
                          positions: Dict[str, List[int]]) -> int:
    """Narrow the design mask to the positions a --design_mask rule listed."""
    mask = sample["design_mask"]
    kept = torch.zeros_like(mask)
    out_of_range = 0
    for entry in layout:
        if not bool(mask[entry["start"]:entry["end"]].any()):
            continue
        wanted = sorted(set(positions.get(entry["name"].upper(), []))
                        | set(positions.get("*", [])))
        for pos in wanted:
            node = entry["start"] + pos
            if entry["start"] <= node < entry["end"]:
                kept[node] = True
            else:
                out_of_range += 1
    sample["design_mask"] = kept
    sample["loss_mask"] = sample["loss_mask"] & kept
    return out_of_range


# ------------------------------------------------------------------- design pass

def design_one(model, device: torch.device, path: str, args: argparse.Namespace,
               tokenizer: Unitokenizer_Complex,
               mask_rules: Optional[List[Tuple[str, str, List[int]]]],
               knn: int) -> List[Dict[str, Any]]:
    """Design one structure. Returns one record per designed chain per sample."""
    name = common.input_stem(path)
    chains = parse_structure(path)
    if not chains:
        print(f"[skip] {name}: no protein, RNA or DNA chain parsed from {path}")
        return []

    total = sum(len(c["seq"]) for c in chains)

    designed = design_scope(chains, args.design_chains, args.design_types)
    if not any(designed):
        parsed_list = ", ".join(f"{c['chain_id']}:{c['kind']}" for c in chains)
        print(f"[skip] {name}: nothing selected for design")
        print(f"        chains parsed : {parsed_list}")
        print(f"        asked for     : types={args.design_types} "
              f"chains={args.design_chains}")
        return []

    sample = build_sample(chains, designed, tokenizer, name)
    if model.fixed_atom_context:
        from src.datasets.fixed_atom_context import context_from_structure
        context = context_from_structure(path)
        sample.update(context)
        n_context = int(context['context_element'].shape[0])
        print(f"[ligand] {name}: {n_context} cofactor atom(s) in context"
              if n_context else
              f"[ligand] {name}: no ligand or metal atom found")
    layout = chain_layout(chains)
    reference_S = sample["S"].detach().clone()

    positions = common.lookup_design_positions(mask_rules, name) if mask_rules else None
    if mask_rules and not positions:
        print(f"[skip] {name}: --design_mask names no rule that matches it, "
              f"and designing the whole complex instead would be a "
              f"different run")
        return []
    if positions:
        lost = restrict_to_positions(sample, layout, positions)
        if lost:
            print(f"[mask] {name}: {lost} position(s) outside the chain they name")
        if not bool(sample["design_mask"].any()):
            print(f"[skip] {name}: the design mask selected no position")
            return []

    batch = collate_complex([sample], knn_k=knn)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    design_mask = batch["design_mask"].bool()
    n_design = int(design_mask.sum())
    scope = ", ".join(f"{e['name']}({e['kind']})" for e in layout
                      if bool(design_mask[e["start"]:e["end"]].any()))
    print(f"[design] {name}: {n_design} of {total} positions across {scope}")

    logit_bias = common.build_logit_bias(batch, tokenizer, args, device)

    out = model.sample(batch, temperature=args.temperature,
                       num_samples=args.num_samples, logit_bias=logit_bias,
                       flow_steps=args.flow_steps)
    # What the model makes of each designed token, with every design position
    # masked: one decoder call, shared by all the samples of this input.
    with torch.no_grad():
        marginal = model.marginal_log_probs(batch).cpu()

    cut = float(args.interface_cutoff or 0.0)
    iface = None
    if cut > 0:
        # interface_nodes reads the chain spans off the batch; the collate
        # carries chain indices instead, so they are handed over here.
        spans = np.asarray([[e["start"], e["end"]] for e in layout], dtype=np.int64)
        # A protein-protein complex has no nucleic acid to be near, so the
        # partner is whatever the complex holds: nucleic if there is any.
        partner = ("nucleic" if any(c["kind"] in ("rna", "dna") for c in chains)
                   else "chain")
        iface = common.interface_nodes(dict(batch, chain_spans=spans), cut,
                                       partner)
        if bool(iface.any()):
            near = ("a nucleic acid" if partner == "nucleic"
                    else "a different chain")
            print(f"[interface] {name}: {int(iface.sum())} residue(s) within "
                  f"{cut:g} A of {near}")

    # The positions that are not designed keep the sequence the structure
    # came with, and it is what recovery is measured against.
    native = batch["S"].detach().cpu().long()
    reference = reference_S.detach().cpu().long()
    n_nodes = native.shape[0]
    if reference.shape[0] != n_nodes:
        raise SystemExit(f"{name}: {reference.shape[0]} reference tokens "
                         f"against {n_nodes} nodes")
    tokens = out["S"].detach().cpu().long()
    n_got = max(1, tokens.shape[0] // n_nodes)
    mask_cpu = design_mask.detach().cpu()

    records: List[Dict[str, Any]] = []
    for s_idx in range(n_got):
        lo, hi = s_idx * n_nodes, (s_idx + 1) * n_nodes
        sampled = native.clone()
        sampled[mask_cpu] = tokens[lo:hi][mask_cpu]
        log_p = marginal.gather(1, sampled[:, None]).squeeze(1)

        for entry in layout:
            start, end = entry["start"], entry["end"]
            chain_mask = mask_cpu[start:end]
            if not bool(chain_mask.any()):
                continue
            kind = entry["kind"]
            pred_ids, true_ids = sampled[start:end], reference[start:end]
            recovery = float((pred_ids[chain_mask] == true_ids[chain_mask]).float().mean())
            score = float(log_p[start:end][chain_mask].mean())
            rec_if, n_if = None, 0
            if iface is not None:
                im = chain_mask & iface[start:end]
                n_if = int(im.sum())
                if n_if:
                    rec_if = float((pred_ids[im] == true_ids[im]).float().mean())
            records.append({
                "input_name": name,
                "input_path": path,
                "chain": entry["name"],
                "type": kind,
                "sample": s_idx,
                "score": score,
                "recovery": recovery,
                "n_designed": int(chain_mask.sum()),
                "recovery_interface": rec_if,
                "n_interface": n_if,
                "length": int(end - start),
                "sequence": tokenizer.decode(pred_ids.tolist(), kind=kind),
                "native": tokenizer.decode(true_ids.tolist(), kind=kind),
            })
    return records


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Design sequences for a backbone with ComplexMPNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Structure file (.pdb/.cif, optionally .gz) or a directory of them")
    p.add_argument("--out_dir", required=True, help="Directory for fasta/tsv/json outputs")
    p.add_argument("--ckpt", default=os.environ.get("COMPLEXMPNN_CKPT"),
                   help="Model checkpoint (or set COMPLEXMPNN_CKPT). The "
                        "architecture is read out of it")

    g = p.add_argument_group("design scope")
    g.add_argument("--design_chains", type=str, default=None,
                   help='Chains to design, e.g. "A,B". Default: all polymer chains')
    g.add_argument("--design_types", type=str, default=None,
                   help="Molecule types to design: protein, rna, dna (comma separated)")
    g.add_argument("--design_mask", type=str, default=None,
                   help="TSV of per-chain positions to design (see examples/design_mask.tsv)")
    g.add_argument("--interface_cutoff", type=float, default=0.0,
                   help="Also report recovery restricted to residues within this "
                        "many angstroms of a partner chain. 0 disables it")
    g.add_argument("--omit_AA", default="",
                   help="Amino acids the design may not use, as letters, e.g. C")
    g.add_argument("--bias_AA", default="",
                   help='Per-amino-acid shift added to every protein position, '
                        'as "C:-2.0,W:-0.5". Added before the temperature')
    g.add_argument("--surface_bias", type=float, default=0.0,
                   help="Shift added to the nonpolar side chains at exposed "
                        "positions only. Negative discourages them. 0 disables it")

    s = p.add_argument_group("sampling")
    s.add_argument("--num_samples", type=int, default=1, help="Sequences to sample per input")
    s.add_argument("--temperature", type=float, default=0.1,
                   help="Sampling temperature; 0 is greedy. Raise it for diversity")
    s.add_argument('--flow-steps', type=int, default=50,
                   dest='flow_steps',
                   help='Generation steps, one decoder call each. 50 is what '
                        'the benchmarks use: against 100 it differs by at most '
                        '0.004 recovery and takes half the time')
    s.add_argument("--seed", type=int, default=None, help="Random seed")
    s.add_argument("--device", default=None, help='e.g. "cuda", "cuda:0", "cpu"')
    return p


def run_metadata(args: argparse.Namespace, info: Dict[str, Any],
                 knn: int) -> Dict[str, Any]:
    """Everything about this run that changes what it produced."""
    saved = dict(info.get("args") or {})
    return {
        "model": "complexmpnn",
        "checkpoint": args.ckpt,
        "feature_version": info["feature_version"],
        # The settings the network was built from, out of the checkpoint.
        "hidden_dim": saved.get("hidden_dim"),
        "encoder_layers": saved.get("encoder_layers"),
        "ar_layers": saved.get("ar_layers"),
        "heads": saved.get("heads"),
        "virtual_points": saved.get("virtual_points"),
        "fixed_atom_context": saved.get("fixed_atom_context"),
        "decoder_recv_encoder": saved.get("decoder_recv_encoder"),
        "decoder": "flow",
        "flow_variant": "coupled",
        "flow_steps": args.flow_steps,
        "flow_remask": saved.get('flow_remask', 1.0),
        "score_semantics": 'mean_log_prob_of_designed_tokens_all_masked',
        "knn": knn,
        "ligand_context": bool(saved.get("fixed_atom_context")),
        "design_chains": args.design_chains,
        "design_types": args.design_types,
        "design_mask": args.design_mask,
        "omit_AA": args.omit_AA,
        "bias_AA": args.bias_AA,
        "surface_bias": args.surface_bias,
        "surface_neighbours": common.SURFACE_NEIGHBOURS,
        "surface_radius": common.SURFACE_RADIUS,
        # Which residues the surface bias is applied to, so a nonzero
        # surface_bias is reproducible.
        "surface_nonpolar": common.NONPOLAR,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        # recovery_interface and n_interface in summary.tsv mean nothing
        # without it.
        "interface_cutoff": args.interface_cutoff,
    }


def main() -> None:
    args = build_parser().parse_args()

    if not args.ckpt:
        raise SystemExit("--ckpt is required (or set COMPLEXMPNN_CKPT)")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    args.design_chains = ([x.strip() for x in args.design_chains.split(",") if x.strip()]
                          if args.design_chains else None)
    args.design_types = ([x.strip() for x in args.design_types.split(",") if x.strip()]
                         if args.design_types else None)

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    inputs = common.collect_inputs(args.input)
    if not inputs:
        raise SystemExit(f"no structure files found at: {args.input}")
    print(f"[input] {len(inputs)} structure(s) from {args.input}")

    mask_rules = None
    if args.design_mask:
        if not os.path.exists(args.design_mask):
            raise SystemExit(f"design mask file not found: {args.design_mask}")
        mask_rules = common.parse_design_mask_file(args.design_mask)
        print(f"[mask] {len(mask_rules)} rule(s) from {args.design_mask}")

    model, device, info = load_model(args.ckpt, args.device)
    print('[flow] each sample is an independent trajectory; the score beside '
          'it is the mean log-probability of its designed tokens')
    knn = int(info["args"].get("knn", 32) or 32)
    print(f"[features] knn={knn}  ligand context: "
          + ("read from the structure" if model.fixed_atom_context
             else "not read by this model"))
    tokenizer = Unitokenizer_Complex()

    # Seeded here rather than at startup, so the sampler's draws depend on the
    # seed and the inputs alone and not on how the network was built.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for path in inputs:
            try:
                records.extend(design_one(model, device, path, args, tokenizer,
                                          mask_rules, knn))
            except Exception as exc:          # keep going over a batch of inputs
                print(f"[error] {path}: {type(exc).__name__}: {exc}")

    if not records:
        raise SystemExit("no sequences were designed")

    meta = run_metadata(args, info, knn)
    common.write_outputs(records, args.out_dir, meta)


if __name__ == "__main__":
    main()
