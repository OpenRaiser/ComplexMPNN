# -*- coding: utf-8 -*-
"""Reading a structure file into sequences and backbone coordinates.

Accepts PDB or mmCIF, plain or gzipped, and works out which from the content
rather than the extension. A chain is read as nucleic acid when at least half
of its residues have nucleotide names, and as protein otherwise.

The coordinate lists this returns have one entry per residue, in chain order,
and None where the atom is absent. Nothing here drops a residue: the caller
decides what missing atoms mean.
"""
from __future__ import annotations

import gzip
import math
import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser

# The atoms read per residue. For a nucleotide, N is the base atom bonded to
# the sugar, N9 in purines and N1 in pyrimidines.
NA_BACKBONE_ORDER = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'",
                     "C2'", "N"]
PROT_BACKBONE_ORDER = ["N", "CA", "C", "O"]

# Sugar atoms are written with a prime or, in older files, a star.
NA_BACKBONE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "P": ("P",),
    "O5'": ("O5'", "O5*"),
    "C5'": ("C5'", "C5*"),
    "C4'": ("C4'", "C4*"),
    "C3'": ("C3'", "C3*"),
    "O3'": ("O3'", "O3*"),
    "O4'": ("O4'", "O4*"),
    "C1'": ("C1'", "C1*"),
    "C2'": ("C2'", "C2*"),
}

NA3_TO_1 = {
    "DA": "A", "DT": "T", "DG": "G", "DC": "C", "DI": "I", "DU": "U",
    "A": "A", "T": "T", "G": "G", "C": "C", "U": "U", "I": "I",
    "URA": "U", "ADE": "A", "THY": "T", "GUA": "G", "CYT": "C",
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # Selenocysteine, pyrrolysine, and the protonation and force-field
    # variants of histidine, all read as their parent amino acid.
    "SEC": "U", "PYL": "O", "MSE": "M",
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    # Ambiguous or unknown: the model reads these as an unknown identity.
    "ASX": "B", "GLX": "Z", "XLE": "J", "UNK": "X", "XXX": "X",
}


def detect_format(text: str) -> str:
    """'cif' or 'pdb', from the first few kilobytes of the file."""
    head = text[:4000]
    return "cif" if ("_atom_site." in head or "data_" in head) else "pdb"


def read_text(path: Path) -> str:
    """The file's text, decompressed if it is gzipped."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_structure_file(path: Path) -> Tuple[Any, str, str]:
    """(structure, format, text). Parsed in memory, so nothing is written."""
    text = read_text(path)
    fmt = detect_format(text)
    parser = MMCIFParser(QUIET=True) if fmt == "cif" else PDBParser(QUIET=True)
    return parser.get_structure("structure", StringIO(text)), fmt, text


def canon_atom_name(name: str) -> str:
    """One spelling for the prime, whichever character the file used."""
    s = name.strip()
    for odd in ("’", "′", "`", "*"):
        s = s.replace(odd, "'")
    return s


def sanitize_coord(atom, ndigits: int) -> Optional[List[float]]:
    """An atom's coordinate, rounded, or None if it is missing or not finite."""
    coord = getattr(atom, "coord", None)
    if coord is None or len(coord) != 3:
        return None
    try:
        x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return [round(x, ndigits), round(y, ndigits), round(z, ndigits)]


def build_residue_atom_index(res) -> Dict[str, Any]:
    """{atom name: atom} for one residue, keeping the highest occupancy."""
    index: Dict[str, Any] = {}
    for atom in res.get_atoms():
        name = canon_atom_name(atom.get_name())
        if atom.is_disordered():
            alternatives = atom.disordered_get_list()
            atom = max(alternatives, key=lambda a: a.occupancy or 0.0)
        index.setdefault(name, atom)
    return index


def guess_is_nucleic(chain) -> bool:
    """Whether at least half of the chain's residues are nucleotides."""
    total = na_like = 0
    for res in chain.get_residues():
        if res.id[0] != " ":          # a hetero residue, not part of the chain
            continue
        total += 1
        name = res.get_resname().strip().upper()
        if name in NA3_TO_1 or (name.startswith("D")
                                and name[1:] in ("A", "T", "G", "C", "U", "I")):
            na_like += 1
    return total > 0 and na_like / max(total, 1) >= 0.5


def na3_to1_safe(resname: str) -> str:
    """The one-letter base, or N where the residue name is not a nucleotide."""
    name = resname.strip().upper()
    if name in NA3_TO_1:
        return NA3_TO_1[name]
    if name.startswith("D") and name[1:] in ("A", "T", "G", "C", "U", "I"):
        return name[1:]
    return "N"


def prot3_to1_safe(resname: str) -> str:
    """The one-letter amino acid, or X where the residue name is unknown."""
    name = re.sub(r"[^A-Z]", "", resname.strip().upper())
    return AA3_TO_1.get(name, "X")


def extract_chain_seq_and_backbone(
    chain,
    ndigits: int = 3,
    validate: bool = True,
    drop_all_none_keys: bool = False,
) -> Tuple[str, Dict[str, List[Optional[List[float]]]], bool]:
    """(sequence, {atom name: coordinates per residue}, is nucleic acid)."""
    is_na = guess_is_nucleic(chain)
    order = NA_BACKBONE_ORDER if is_na else PROT_BACKBONE_ORDER

    residues = [res for res in chain.get_residues() if res.id[0] == " "]
    seq = "".join(na3_to1_safe(r.get_resname()) if is_na
                  else prot3_to1_safe(r.get_resname()) for r in residues)

    coords: Dict[str, List[Optional[List[float]]]] = {a: [] for a in order}
    for res in residues:
        atom_index = build_residue_atom_index(res)
        for atom_name in order:
            atom = atom_index.get(canon_atom_name(atom_name))
            if atom_name == "N" and is_na:
                # The base atom bonded to the sugar: N9 in a purine, N1 in a
                # pyrimidine. Fall back to whatever is called N.
                base = na3_to1_safe(res.get_resname())
                if base in {"A", "G"}:
                    atom = atom_index.get("N9") or atom
                elif base in {"C", "T", "U"}:
                    atom = atom_index.get("N1") or atom
            if atom is None and is_na:
                for alias in NA_BACKBONE_ALIASES.get(atom_name, (atom_name,)):
                    atom = atom_index.get(canon_atom_name(alias))
                    if atom is not None:
                        break
            coords[atom_name].append(
                sanitize_coord(atom, ndigits) if atom is not None else None)

    if drop_all_none_keys:
        coords = {k: v for k, v in coords.items() if any(c is not None for c in v)}

    if validate:
        for name, arr in coords.items():
            if len(arr) != len(seq):
                raise ValueError(f'atom {name}: {len(arr)} coordinate(s) for '
                                 f'{len(seq)} residue(s)')
            for i, item in enumerate(arr):
                if item is not None and not (isinstance(item, list)
                                             and len(item) == 3):
                    raise ValueError(f'atom {name} at residue {i}: '
                                     f'{item!r} is not a coordinate')

    return seq, coords, is_na
