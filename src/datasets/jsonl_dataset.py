# -*- coding: utf-8 -*-
"""Reading a benchmark split from JSONL.

One line is one structure: a PROTEIN, RNA or DNA section per polymer chain,
each with a sequence and its atom coordinates, and a LIGAND section with the
cofactor atoms. `JsonlRows` indexes the file by byte offset so a line can be
read without holding the split in memory, and the index is cached beside it.

Two details of the files themselves are handled here. Polymer residues are
sometimes stored in the LIGAND section instead of their chain, and are moved
back to the chain they bond to. Sequences and coordinates are checked against
each other, and an entry whose chains disagree is dropped with a reason.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.crop import (
    ConditionDropped, crop_to_budget, required_context_flags,
)
from src.datasets.ligand_context import get_ligand_section
from src.datasets.structure_input import (
    NUCLEIC_KINDS, PRIME_IN_MAP, REFERENCE_SLOT, SLOT_NAMES, TYPE_ID,
    chain_slots, segment_index,
)
from src.datasets.task_sampling import (
    applicable, chains_for_task, sample_design_mask,
)
from src.datasets.tokenizer_complex import Unitokenizer_Complex


def example_seed(base: int, epoch: int, index: int) -> int:
    """A seed for one entry, from its index rather than the reading order."""
    return (base + 1000003 * epoch + 9176 * index) % (2 ** 32)


@contextmanager
def sample_seed(seed: int):
    """Seed the crop and the task of one entry, then restore the caller's."""
    py, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


class SampleBuilder:
    """Turn a list of chains plus a task into the tensors the model reads."""

    def __init__(self, budget: int = 3000, tokenizer: Unitokenizer_Complex = None,
                 contact_cutoff: float = 12.0, scarce_first: bool = True):
        self.budget = int(budget)
        self.contact_cutoff = float(contact_cutoff)
        self.scarce_first = bool(scarce_first)
        self.tok = tokenizer or Unitokenizer_Complex()
        self.unknown = {self.tok.protein_X_id, self.tok.nuc_N_id, self.tok.mask_id}

    def build(self, chains: Sequence[dict], designed: Sequence[bool],
              task: str, rng: random.Random, example_id: str = '') -> dict:
        kinds = [c['kind'] for c in chains]
        lengths = [len(c['seq']) for c in chains]
        slots = [chain_slots(k, c['coords'], L) for k, c, L in zip(kinds, chains, lengths)]
        ref = [X[:, REFERENCE_SLOT, :] for X, _ in slots]
        ref_ok = [v[:, REFERENCE_SLOT] for _, v in slots]

        required = required_context_flags(task, kinds, designed)
        try:
            kept, rec = crop_to_budget(kinds, ref, ref_ok, designed, self.budget,
                                       rng, self.contact_cutoff,
                                       self.scarce_first, required)
        except ConditionDropped:
            downgraded = task
            task = 'all'
            designed = [True] * len(kinds)
            kept, rec = crop_to_budget(kinds, ref, ref_ok, designed, self.budget,
                                       rng, self.contact_cutoff,
                                       self.scarce_first, [False] * len(kinds))
            rec = dict(rec)
            rec[f'task_downgraded_from_{downgraded}'] = 1
        if not kept:
            raise ValueError(f'{example_id}: nothing survived the budget')

        X, valid, S, type_vec, chain_idx, res_idx, design = [], [], [], [], [], [], []
        for slot, (i, a, b) in enumerate(kept):
            kind = kinds[i]
            Xi, vi = slots[i]
            X.append(Xi[a:b])
            valid.append(vi[a:b])
            S.append(torch.tensor(self.tok.encode(chains[i]['seq'][a:b], kind), dtype=torch.long))
            type_vec.append(torch.full((b - a,), TYPE_ID[kind], dtype=torch.long))
            chain_idx.append(torch.full((b - a,), slot, dtype=torch.long))
            res_idx.append(segment_index(ref[i][a:b], ref_ok[i][a:b], kind))
            design.append(torch.full((b - a,), bool(designed[i])))

        out = dict(X=torch.cat(X), slot_valid=torch.cat(valid), S=torch.cat(S),
                   type_vec=torch.cat(type_vec), chain_idx=torch.cat(chain_idx),
                   res_idx=torch.cat(res_idx), design_mask=torch.cat(design))
        if out['S'].shape[0] != out['X'].shape[0]:
            raise ValueError(f'{example_id}: sequence and coordinates disagree in length')

        known = ~sum((out['S'] == t) for t in self.unknown).bool()
        out['loss_mask'] = out['design_mask'] & known & out['slot_valid'][:, REFERENCE_SLOT]
        out['example_id'] = example_id
        out['task'] = task
        out['crop'] = rec
        out['kept_chains'] = [(kinds[i], a, b, bool(designed[i])) for i, a, b in kept]
        return out


SECTIONS = (('PROTEIN', 'protein'), ('RNA', 'rna'), ('DNA', 'dna'))


def _section(row: dict, name: str):
    """One section of a row, whatever case it was written in."""
    block = row.get(name) or row.get(name.lower()) or row.get(name.capitalize())
    return block if isinstance(block, dict) else {}


def _atom_array(coords: dict, slot: str, length: int):
    """One atom's coordinates for a chain, or None where it is absent."""
    if not isinstance(coords, dict):
        return None
    for cand in PRIME_IN_MAP.get(slot, (slot,)):
        raw = coords.get(cand)
        if isinstance(raw, (list, tuple)) and len(raw):
            out = np.full((length, 3), np.nan, dtype=np.float32)
            for i, a in enumerate(raw[:length]):
                if isinstance(a, (list, tuple)) and len(a) == 3:
                    out[i] = a
            return out
    return None


def row_chains(row: dict) -> List[dict]:
    """Every polymer chain in a row, uncropped."""
    out = []
    for name, kind in SECTIONS:
        for chain_id, rec in _section(row, name).items():
            seq = (rec or {}).get('seq') or ''
            if not seq:
                continue
            raw = (rec or {}).get('coords') or {}
            coords = {}
            for slot in SLOT_NAMES[kind]:
                if slot is None:
                    continue
                arr = _atom_array(raw, slot, len(seq))
                if arr is not None:
                    coords[slot] = arr
            out.append(dict(kind=kind, chain_id=str(chain_id), seq=seq, coords=coords))
    return out


# -------------------------------------- residues parked in the LIGAND section

PARKED_SLOTS = {
    'nucleic': {'P': ('P',), 'O5': ("O5'", 'O5*', 'O5'), 'C5': ("C5'", 'C5*', 'C5'),
                'C4': ("C4'", 'C4*', 'C4'), 'C3': ("C3'", 'C3*', 'C3'),
                'O3': ("O3'", 'O3*', 'O3'), 'N': ('N9', 'N1', 'N')},
    'protein': {'N': ('N',), 'CA': ('CA',), 'C': ('C',), 'O': ('O',)},
}
PARKED_LINKS = {
    'nucleic': (('P', 'O3', 'after'), ('O3', 'P', 'before')),
    'protein': (('N', 'C', 'after'), ('C', 'N', 'before')),
}
PARKED_TOKEN = {'nucleic': 'N', 'protein': 'X'}


PARKED_MIN_BACKBONE = {'nucleic': 4, 'protein': 3}
PARKED_REQUIRED = {'nucleic': ('C4',), 'protein': ('N', 'CA', 'C')}


def _parked_atom(names, coords, candidates):
    for cand in candidates:
        for i, n in enumerate(names):
            if n.strip().upper() == cand.upper():
                return coords[i]
    return None


def _nearest_under(a, b, bond_max):
    """(index into b, distance) for the closest point of b to a, or None."""
    if a is None or b is None or not len(b):
        return None
    d = np.linalg.norm(np.asarray(b, dtype=np.float64)
                       - np.asarray(a, dtype=np.float64)[None, :], axis=1)
    if not np.isfinite(d).any():
        return None
    j = int(np.nanargmin(d))
    return (j, float(d[j])) if np.isfinite(d[j]) and d[j] <= bond_max else None


def parked_residues(row: dict, chains: Sequence[dict], bond_max: float = 2.0,
                    report: Optional[Counter] = None
                    ) -> List[Tuple[int, int, List[dict], List[str]]]:
    """LIGAND entries that are really residues of these chains."""
    from src.datasets.ligand_context import extract_ligand_atoms_detailed

    rep = report if report is not None else Counter()
    det = extract_ligand_atoms_detailed(row, drop_cryo=False)
    if not len(det['coords']):
        return []
    names, xyz, inst = det['atom_names'], det['coords'], det['instance']

    # One atom dict per candidate, keyed by the slot it would occupy, and the
    # polymer family it would belong to.
    cand = []
    for k, lig_name in enumerate(det['names']):
        rows = inst == k
        if not rows.any():
            continue
        mine = [n for n, r in zip(names, rows) if r]
        mxyz = xyz[rows]
        for family in ('nucleic', 'protein'):
            atoms = {}
            for slot, cands in PARKED_SLOTS[family].items():
                v = _parked_atom(mine, mxyz, cands)
                if v is not None:
                    atoms[slot] = np.asarray(v, dtype=np.float32)
            if all(r in atoms for r in PARKED_REQUIRED[family]) \
                    and len(atoms) >= PARKED_MIN_BACKBONE[family]:
                cand.append(dict(family=family, atoms=atoms, name=str(lig_name)))
                break
        else:
            rep['parked_no_backbone'] += 1
    if not cand:
        return []

    # The chain atoms each link can reach.
    chain_xyz = []
    for chain in chains:
        family = 'protein' if chain['kind'] == 'protein' else 'nucleic'
        got = {}
        for slot, arr in chain['coords'].items():
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 2 and a.shape[1] == 3:
                got[slot] = a
        chain_xyz.append((family, got))

    # Links, both to the chains and between candidates. PARKED_LINKS reads as
    # (this residue's atom, the other's slot, which side this residue is on).
    for c in cand:
        c['prev_chain'] = c['next_chain'] = None
        c['prev_parked'] = c['next_parked'] = None
    for i, c in enumerate(cand):
        for lig_atom, slot, side in PARKED_LINKS[c['family']]:
            mine = c['atoms'].get(lig_atom)
            key = 'prev' if side == 'after' else 'next'
            best = None
            for ci, (family, got) in enumerate(chain_xyz):
                if family != c['family']:
                    continue
                hit = _nearest_under(mine, got.get(slot), bond_max)
                if hit is not None and (best is None or hit[1] < best[2]):
                    best = (ci, hit[0], hit[1])
            if best is not None:
                c[f'{key}_chain'] = (best[0], best[1])
            other = None
            for j, d in enumerate(cand):
                if j == i or d['family'] != c['family']:
                    continue
                theirs = d['atoms'].get(slot)
                if theirs is None or mine is None:
                    continue
                dist = float(np.linalg.norm(mine.astype(np.float64)
                                            - theirs.astype(np.float64)))
                if dist <= bond_max and (other is None or dist < other[1]):
                    other = (j, dist)
            if other is not None:
                c[f'{key}_parked'] = other[0]

    # Directed runs: start at a candidate with no predecessor among the others.
    runs, seen = [], set()
    for i, c in enumerate(cand):
        if i in seen or c['prev_parked'] is not None:
            continue
        run, j = [], i
        while j is not None and j not in seen:
            seen.add(j)
            run.append(j)
            j = cand[j]['next_parked']
        runs.append(run)
    for i in range(len(cand)):           # a cycle, which cannot be ordered
        if i not in seen:
            rep['parked_cyclic'] += 1
            seen.add(i)

    out, taken = [], {}
    for run in runs:
        head, tail = cand[run[0]], cand[run[-1]]
        a = head['prev_chain']
        b = tail['next_chain']
        where = None
        if a is not None and b is not None:
            if a[0] == b[0] and a[1] + 1 == b[1]:
                where, kind = (a[0], a[1] + 1), 'parked_confident'
            else:
                rep['parked_ambiguous'] += len(run)
                continue
        elif a is not None:
            where, kind = (a[0], a[1] + 1), 'parked_single_ended'
        elif b is not None:
            where, kind = (b[0], b[1]), 'parked_single_ended'
        else:
            rep['parked_unbonded'] += len(run)
            continue
        if where in taken:               # two runs claiming one hole
            rep['parked_ambiguous'] += len(run) + taken[where]
            out = [o for o in out if (o[0], o[1]) != where]
            continue
        taken[where] = len(run)
        rep[kind] += len(run)
        ci = where[0]
        keep = [{s: v for s, v in cand[j]['atoms'].items()
                 if s in chains[ci]['coords']} for j in run]
        out.append((ci, where[1], keep, [cand[j]['name'] for j in run]))
    return out


def apply_parked_residues(chains: List[dict],
                          parked: Sequence[Tuple[int, int, List[dict], List[str]]]
                          ) -> List[dict]:
    """Put the parked runs back, in place of the holes they left."""
    if not parked:
        return chains
    out = [dict(c, coords=dict(c['coords'])) for c in chains]
    for ci, at, run, _names in sorted(parked, key=lambda p: (p[0], -p[1])):
        chain = out[ci]
        family = 'protein' if chain['kind'] == 'protein' else 'nucleic'
        length = len(chain['seq'])
        at = max(0, min(int(at), length))
        chain['seq'] = (chain['seq'][:at] + PARKED_TOKEN[family] * len(run)
                        + chain['seq'][at:])
        for slot, arr in chain['coords'].items():
            a = np.asarray(arr, dtype=np.float32)
            block = np.stack([
                (np.full(3, np.nan, dtype=np.float32) if atoms.get(slot) is None
                 else np.asarray(atoms[slot], dtype=np.float32))
                for atoms in run])
            chain['coords'][slot] = np.concatenate([a[:at], block, a[at:]], axis=0)
    return out


class JsonlRows:
    """Byte offsets into a JSONL file, so a row is read without holding the file."""

    def __init__(self, path: str):
        self.path = str(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self._not_found_message(self.path))
        self.offsets: List[int] = []
        h = hashlib.sha256()
        with open(self.path, 'rb') as fh:
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    break
                h.update(line)
                if line.strip():
                    self.offsets.append(pos)
        self.sha256 = h.hexdigest()
        self._fh = None

    @staticmethod
    def _not_found_message(path: str) -> str:
        """Name what was looked for and what is actually there."""
        directory = os.path.dirname(path) or '.'
        lines = [f'no such file: {path}']
        for where in (directory, os.path.join(directory, 'pdb_complex')):
            if os.path.isdir(where):
                found = sorted(f for f in os.listdir(where) if f.endswith('.jsonl'))
                lines.append(f'  {where} holds: ' +
                             (', '.join(found) if found else 'no .jsonl files'))
            else:
                lines.append(f'  {where} is not a directory')
        lines.append('  pass --train-jsonl and --valid-jsonl, or point '
                     'DATA_ROOT at the directory that holds them')
        return '\n'.join(lines)

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_fh'] = None
        return d

    def __getitem__(self, index: int) -> dict:
        if self._fh is None:
            self._fh = open(self.path, 'rb')
        self._fh.seek(self.offsets[index])
        return json.loads(self._fh.readline().decode('utf-8', errors='ignore'))

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


INDEX_VERSION = 3
TASKS = ('all', 'protein_given_nucleic', 'nucleic_given_protein', 'one_chain')


def length_stats(lengths: Sequence[int]) -> Dict[str, float]:
    """The node-count distribution, which is what sizes a batch."""
    xs = sorted(lengths)
    n = len(xs)
    if n == 0:
        return dict(entries=0, mean=0.0, median=0, p90=0, p99=0, max=0)

    def q(f):
        return xs[min(n - 1, int(f * n))]

    return dict(entries=n, mean=round(sum(xs) / n, 1), median=q(0.5),
                p90=q(0.90), p99=q(0.99), max=xs[-1])


def _sequence_keys(row: dict) -> Tuple[str, str]:
    """Two hashes of a row's sequences, for deduplication without an aligner."""
    seqs = []
    for name, _ in SECTIONS:
        for _, rec in _section(row, name).items():
            seq = (rec or {}).get('seq') or ''
            if seq:
                seqs.append(seq)
    if not seqs:
        return '', ''
    longest = max(seqs, key=len)
    whole = '\x00'.join(sorted(seqs))
    return (hashlib.sha1(longest.encode()).hexdigest()[:16],
            hashlib.sha1(whole.encode()).hexdigest()[:16])


def row_summary(row: dict) -> Optional[Tuple[List[str], List[int]]]:
    """The chain kinds and sequence lengths of a row, and nothing else."""
    kinds, lengths = [], []
    for name, kind in SECTIONS:
        for _, rec in _section(row, name).items():
            seq = (rec or {}).get('seq') or ''
            if seq:
                kinds.append(kind)
                lengths.append(len(seq))
    return (kinds, lengths) if kinds else None


class ComplexDataset(Dataset):
    """One split, read by index, with a design task per entry.

    The file is scanned once on construction for the chains of every entry,
    their lengths and their names, and the result is cached beside it under
    `<path>.index.json`, keyed by the file's own hash. The scan is what lets
    `__len__` and the task counts be answered without reading the file again.
    """

    def __init__(self, path: str, budget: int = 3000, contact_cutoff: float = 12.0,
                 task_weights: Optional[Dict[str, float]] = None, seed: int = 111,
                 training: bool = True, cache: bool = True,
                 progress_every: int = 20000, log=print,
                 mixed_task_weights: Optional[Dict[str, float]] = None,
                 fixed_mixed_tasks: Optional[Sequence[str]] = None,
                 scarce_first: bool = True, repair_parked: bool = False,
                 fixed_atom_context: bool = False):
        self.repair_parked = bool(repair_parked)
        self.fixed_atom_context = bool(fixed_atom_context)
        self.rows = JsonlRows(path)
        self.builder = SampleBuilder(budget, contact_cutoff=contact_cutoff,
                                     scarce_first=scarce_first)
        self.task_weights = dict(task_weights) if task_weights else None
        self.mixed_task_weights = dict(mixed_task_weights) if mixed_task_weights else None
        self.fixed_mixed_tasks = tuple(fixed_mixed_tasks or ())
        self.seed, self.training, self.epoch = int(seed), bool(training), 0
        built = None
        # The repair changes the chain lengths the index records, so the two
        # settings cannot share a cache.
        suffix = '_repaired' if self.repair_parked else ''
        cache_path = f'{path}.index{suffix}.json'
        cache_state = 'disabled'
        if cache:
            built, cache_state = self._load_cache(cache_path, self.rows.sha256)
        if built is None:
            built = self._scan(path, progress_every, log)
            if cache:
                cache_state = '%s; %s' % (
                    cache_state,
                    self._save_cache(cache_path, self.rows.sha256, built))
        self.rows.close()

        self.index = built['index']
        self.names = built['names']
        self.is_mixed = built['is_mixed']
        self.seq_keys = built['seq_keys']
        self.set_keys = built['set_keys']
        self.mixed_rank: List[int] = []
        seen = 0
        for flag in self.is_mixed:
            self.mixed_rank.append(seen)
            seen += int(bool(flag))
        self.lengths = [min(n, budget) for n in built['sums']]
        if not self.index:
            raise ValueError(f'no usable entries in {path}: {built["counts"]}')
        counts = dict(built['counts'])
        counts['selected_over_budget'] = sum(n > budget for n in built['sums'])
        self.report = dict(path=str(path), sha256=self.rows.sha256,
                           entries=len(self.index), counts=counts,
                           budget=budget, contact_cutoff=contact_cutoff,
                           task_weights=self.task_weights,
                           mixed_task_weights=self.mixed_task_weights,
                           fixed_mixed_tasks=list(self.fixed_mixed_tasks),
                           scarce_first=bool(scarce_first),
                           task_support=built['task_support'],
                           length_stats=length_stats(self.lengths),
                           index_cache=cache_state,
                           residue_numbering='inferred from reference-atom '
                                             'spacing; the JSONL stores none')

    # ------------------------------------------------------- the single scan

    def _scan(self, path, progress_every, log) -> dict:
        suffix = '_repaired' if self.repair_parked else ''
        index, sums, names, is_mixed = [], [], [], []
        seq_keys, set_keys = [], []
        counts, support = Counter(), Counter()
        total = len(self.rows)
        if progress_every and log:
            log(f'[data] indexing {total} rows of {path} '
                f'(cached afterwards as <file>.index{suffix}.json)'
                + ('; parked residues are put back' if self.repair_parked else ''))
        for i in range(total):
            row = self.rows[i]
            summary = row_summary(row)
            if summary is None:
                counts['no_polymer_chain'] += 1
                continue
            kinds, lengths = summary
            if self.repair_parked and get_ligand_section(row):
                here = row_chains(row)
                put = parked_residues(row, here, report=counts)
                if put:
                    here = apply_parked_residues(here, put)
                    kinds = [c['kind'] for c in here]
                    lengths = [len(c['seq']) for c in here]
                    counts['repaired_entries'] += 1
                    counts['repaired_runs'] += len(put)
                    for _ci, _at, _atoms, _nms in put:
                        counts['repaired_residues'] += len(_nms)
                        counts[f'repaired_run_of_{len(_nms)}'] += 1
                        for _nm in _nms:
                            counts[f'repaired_{_nm}'] += 1
            title = row.get('name') or row.get('title')
            if not title:
                raise ValueError(f'{path}:{i + 1}: no name or title. Every '
                                 f'entry needs one, since it is what the '
                                 f'results are reported against')
            has_prot = any(k == 'protein' for k in kinds)
            has_na = any(k in NUCLEIC_KINDS for k in kinds)
            longest_key, whole_key = _sequence_keys(row)
            index.append(i)
            sums.append(sum(lengths))
            names.append(str(title).rsplit('.', 1)[0].lower())
            is_mixed.append(has_prot and has_na)
            seq_keys.append(longest_key)
            set_keys.append(whole_key)
            counts['selected'] += 1
            counts['selected_mixed'] += int(has_prot and has_na)
            counts['selected_protein_only'] += int(has_prot and not has_na)
            counts['selected_nucleic_only'] += int(has_na and not has_prot)
            # How many entries each task could be applied to, counted in this
            # same pass: four counters do not justify a second scan.
            for task in TASKS:
                support[task] += int(applicable(task, kinds))
            if progress_every and log and (i + 1) % progress_every == 0:
                log(f'[data]   {i + 1}/{total} rows, {len(index)} selected')
        return dict(index=index, sums=sums, names=names, is_mixed=is_mixed,
                    seq_keys=seq_keys, set_keys=set_keys,
                    counts=dict(counts), task_support=dict(support))

    # -------------------------------------------------------------- the cache

    @staticmethod
    def _load_cache(cache_path, sha256) -> Tuple[Optional[dict], str]:
        if not os.path.isfile(cache_path):
            return None, 'absent, will be written'
        try:
            with open(cache_path) as fh:
                blob = json.load(fh)
        except (OSError, ValueError) as exc:
            return None, f'unreadable ({exc}), rebuilding'
        if blob.get('version') != INDEX_VERSION:
            return None, 'older version, rebuilding'
        if blob.get('sha256') != sha256:
            return None, 'data file changed, rebuilding'
        need = ('index', 'sums', 'names', 'is_mixed', 'seq_keys', 'set_keys',
                'counts', 'task_support')
        if any(k not in blob for k in need):
            return None, 'incomplete, rebuilding'
        built = {k: blob[k] for k in need}
        built['is_mixed'] = [bool(x) for x in built['is_mixed']]
        return built, f'loaded from {os.path.basename(cache_path)}'

    @staticmethod
    def _save_cache(cache_path, sha256, built) -> str:
        blob = dict(version=INDEX_VERSION, sha256=sha256, **built)
        temp = f'{cache_path}.tmp'
        try:
            with open(temp, 'w') as fh:
                json.dump(blob, fh)
            os.replace(temp, cache_path)
            return f'written to {os.path.basename(cache_path)}'
        except OSError as exc:
            try:
                os.unlink(temp)
            except OSError:
                pass
            return f'not saved ({exc}); every run will rescan'

    # ------------------------------------------------------------ per example

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        seed = example_seed(self.seed, self.epoch if self.training else 0, i)
        row = self.rows[self.index[i]]
        chains = row_chains(row)
        if self.repair_parked:
            put = parked_residues(row, chains)
            if put:
                chains = apply_parked_residues(chains, put)
        kinds = [c['kind'] for c in chains]
        rng = random.Random(seed)
        with sample_seed(seed):
            seqs = [c['seq'] for c in chains]
            if self.fixed_mixed_tasks and self.is_mixed[i]:
                task = self.fixed_mixed_tasks[
                    self.mixed_rank[i] % len(self.fixed_mixed_tasks)]
                if not applicable(task, kinds):
                    raise ValueError(f'{self.names[i]} is mixed but cannot be '
                                     f'asked {task}')
                designed = chains_for_task(task, kinds,
                                           np.random.default_rng(seed), seqs)
            else:
                weights = self.task_weights
                if self.mixed_task_weights is not None and self.is_mixed[i]:
                    weights = self.mixed_task_weights
                task, designed = sample_design_mask(
                    kinds, np.random.default_rng(seed), weights, seqs)
            sample = self.builder.build(chains, designed, task, rng, self.names[i])
            if self.fixed_atom_context:
                from src.datasets.fixed_atom_context import context_from_row
                sample.update(context_from_row(row))
            return sample

    def task_support(self) -> Dict[str, int]:
        """How many entries each task can be asked of at all."""
        return dict(self.report['task_support'])
