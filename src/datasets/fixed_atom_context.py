# -*- coding: utf-8 -*-
"""The cofactor atoms near each residue.

Reads the retained ligand atoms and metal ions, from a parsed structure or
from a dataset row, and connects each residue to the sixteen nearest of them
within sixteen angstroms. The network reads those edges in
src/modules/atom_context.py.
"""
from pathlib import Path
from functools import lru_cache
from io import StringIO

import numpy as np
import torch

from src.datasets.ligand_context import (
    CRYO_BLACKLIST, extract_ligand_atoms_detailed, get_ligand_section,
)
from src.tools.geometry import node_centers

ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe '
            'Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In '
            'Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu '
            'Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu '
            'Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv '
            'Ts Og').split()
ATOMIC_NUMBER = {e.upper(): i + 1 for i, e in enumerate(ELEMENTS)}


COFACTORS = {
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'UTP', 'UDP', 'UMP',
    'CTP', 'CDP', 'CMP', 'TTP', 'ITP', 'IDP',
    'ANP', 'GNP', 'ACP', 'AGS', 'GSP',          # non-hydrolysable analogues
    'NAD', 'NAI', 'NAP', 'NDP', 'NAJ',          # NAD(P)(H)
    'FAD', 'FMN', 'FDA',
    'SAM', 'SAH', 'COA', 'ACO', 'COO',
    'TPP', 'PLP', 'THF', 'MTA', 'BTN', 'B12', 'COB', 'GSH', 'GDU',
}


@lru_cache(maxsize=8192)
def allowed_component(name):
    from Bio.Data.PDBData import protein_letters_3to1_extended, nucleic_letters_3to1_extended
    name = name.strip().upper()
    if name in CRYO_BLACKLIST:
        return False
    if name in COFACTORS:
        return True
    polymers = {k.strip().upper() for k in protein_letters_3to1_extended}
    polymers.update(k.strip().upper() for k in nucleic_letters_3to1_extended)
    polymers.update(('A', 'C', 'G', 'U', 'T', 'DA', 'DC', 'DG', 'DT', 'DU', 'N', 'DN', 'UNK'))
    return name not in polymers


def _pack(xyz, elements):
    xyz = torch.as_tensor(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
    elem = torch.tensor([ATOMIC_NUMBER.get(str(e).strip().upper(), 0) for e in elements], dtype=torch.long)
    valid = torch.isfinite(xyz).all(-1) & (elem != 1)  # heavy atoms only
    return dict(context_xyz=xyz[valid], context_element=elem[valid])


def context_from_row(row):
    ligands = [l for l in get_ligand_section(row)
               if isinstance(l, dict) and allowed_component(str(l.get('ligand_name', '')))]
    data = extract_ligand_atoms_detailed({'LIGAND': ligands}, contact_cutoff=0)
    return _pack(data['coords'], data['elements'])


def context_from_structure(path):
    # Same format detection as design, parsed in memory: reading compressed
    # context must not write another PDB/CIF beside the user's input file.
    from Bio.PDB import MMCIFParser, PDBParser
    from inference.helpers import detect_format, read_text
    text = read_text(Path(path))
    parser = (MMCIFParser(QUIET=True) if detect_format(text) == 'cif'
              else PDBParser(QUIET=True))
    structure = parser.get_structure('context', StringIO(text))
    xyz, elements = [], []
    for chain in next(iter(structure)):
        for residue in chain:
            if not str(residue.id[0]).strip() or not allowed_component(residue.resname):
                continue
            for atom in residue:
                xyz.append(atom.coord)
                elements.append(atom.element)
    return _pack(xyz, elements)


def attach_context_edges(batch, max_neighbours=16, radius=16.0, chunk=128):
    """Bounded K edges, confined to each complex. No full N-by-A distance map."""
    centre = node_centers(batch['X'], batch['slot_valid'])
    xyz = batch['context_xyz']
    receivers, senders = [], []
    for b in torch.unique(batch['batch_id']):
        rows = (batch['batch_id'] == b).nonzero().flatten()
        atoms = (batch['context_batch_id'] == b).nonzero().flatten()
        if not len(atoms):
            continue
        k = min(max_neighbours, len(atoms))
        for start in range(0, len(rows), chunk):
            rr = rows[start:start + chunk]
            best_d = centre.new_full((len(rr), k), float('inf'))
            best_i = torch.zeros((len(rr), k), dtype=torch.long, device=centre.device)
            # Bound both axes even for an unusually large fixed component.
            for offset in range(0, len(atoms), 1024):
                aa = atoms[offset:offset + 1024]
                dist = torch.cdist(centre[rr].float(), xyz[aa].float(),
                                   compute_mode='donot_use_mm_for_euclid_dist')
                ids = aa[None].expand(len(rr), -1)
                all_d, all_i = torch.cat([best_d, dist], -1), torch.cat([best_i, ids], -1)
                best_d, take = all_d.topk(k, largest=False)
                best_i = all_i.gather(1, take)
            keep = best_d < radius
            receivers.append(rr[:, None].expand(-1, k)[keep])
            senders.append(best_i[keep])
    batch['context_edge_idx'] = (torch.stack([torch.cat(receivers), torch.cat(senders)])
                                 if receivers else torch.empty((2, 0), dtype=torch.long,
                                                                device=centre.device))
    return batch
