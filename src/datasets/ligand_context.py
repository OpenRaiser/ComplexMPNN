# -*- coding: utf-8 -*-
"""Which non-polymer components count as cofactors.

Crystallisation additives, cryoprotectants, free halides and heavy-atom
phasing derivatives are dropped; functional metal ions and cofactors are
kept. The same lists were used to build the dataset and are applied again to
a structure handed to the model, so a design run and the training data treat
a component the same way.
"""
import re
from typing import Any, Dict, List, Tuple

import numpy as np

# Crystallisation, buffer and cryoprotectant components, free halides, and
# heavy atoms used for phasing rather than bound as cofactors.
CRYO_BLACKLIST = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "1PE", "P6G", "PGE",
    "ACT", "FMT", "CIT", "FLC", "TRS", "TAR", "MPD", "IPA", "DMS", "BME", "MES",
    "EPE", "HEPES", "BU3", "MRD", "NO3", "NH4", "IOD", "MSE", "SEP", "TPO", "PTR",
    "CL", "BR", "IOD", "F", "OXY", "PEO", "SIN", "MLI", "GAI", "UNX", "UNL",
    "2PE", "12P", "15P", "PE4", "PE8", "144", "OLC", "BOG", "LDA", "SPD",
    # heavy-atom phasing derivatives / non-functional heavy substitutes (NOT cofactors)
    "HG", "PT", "AU", "AG", "PB", "TL", "CD", "CS", "RB", "BA", "SR",
}
METAL_CODES = {
    "ZN", "MG", "MN", "FE", "FE2", "CA", "K", "NA", "CU", "NI", "CO", "3CO",
    "NCO", "MO", "W", "V", "LI", "AL",
}
# two-letter element symbols for parsing organic atom names
_ELEM2 = {"CL", "BR", "FE", "ZN", "MG", "MN", "CA", "NA", "CU", "NI", "CO", "CD",
          "HG", "MO", "SE", "LI", "AL", "BA", "SR", "CS", "RB", "PB", "PT", "AU",
          "AG", "SI", "AS", "TE", "IR", "RU", "RH", "OS", "RE", "TL", "GA"}




def get_ligand_section(obj: Dict[str, Any]):
    for k, v in obj.items():
        if k.lower() == "ligand" and isinstance(v, list):
            return v
    return []


def _macromolecule_coords(obj: Dict[str, Any]) -> np.ndarray:
    """All valid protein/RNA/DNA backbone coordinates in obj as [M,3] float32."""
    pts: List[List[float]] = []
    for k, v in obj.items():
        if k.lower() not in ("protein", "rna", "dna"):
            continue
        if isinstance(v, dict):
            recs = list(v.values())
        elif isinstance(v, list):
            recs = v
        else:
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            co = rec.get("coords")
            if not isinstance(co, dict):
                continue
            for arr in co.values():
                if not isinstance(arr, (list, tuple)):
                    continue
                for xyz in arr:
                    if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                        continue
                    try:
                        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                    except (TypeError, ValueError):
                        continue
                    if x == x and y == y and z == z:   # not NaN
                        pts.append([x, y, z])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


GREEK = "ABGDEZH"
AMBIGUOUS_TWO = frozenset(
    f'{a}{g}' for a in "CNOPS" for g in GREEK) - {"SE"}
ONE_LETTER = frozenset("HBCNOFPSKVWIUY")
TWO_LETTER = frozenset((
    "HE", "LI", "BE", "NE", "NA", "MG", "AL", "SI", "CL", "AR", "CA", "SC",
    "TI", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE",
    "BR", "KR", "RB", "SR", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG",
    "CD", "IN", "SN", "SB", "TE", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
    "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF",
    "TA", "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT",
    "RN", "FR", "RA", "AC", "TH", "PA", "NP", "PU", "AM", "CM", "BK", "CF",
)) - AMBIGUOUS_TWO


COMPONENT_ONLY = frozenset((
    "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
    "TM", "YB", "LU", "AC", "TH", "PA", "NP", "PU", "AM", "CM", "BK", "CF",
    "ES", "FM", "MD", "NO", "LR", "AT", "PO", "IN",
    "HE", "HF", "HG", "HS",
))


def _element_of(atom_name: str, ligand_name: str):
    """Element symbol for a ligand atom, or None when it cannot be resolved."""
    ln = str(ligand_name or "").upper().strip()
    raw = str(atom_name or "").strip()
    an = re.sub(r"['\"*+\-]", "", raw).upper()
    if ln in METAL_CODES:                    # single-atom metal ion
        return (ln[:2] if ln[:2] in _ELEM2 else ln[:1]).title()
    lead = re.match(r"[A-Z]+", an)
    if not lead:
        return None
    lead = lead.group(0)
    if (len(lead) == 2 and lead in TWO_LETTER
            and (lead not in COMPONENT_ONLY or ln == lead)):
        return lead.title()
    if lead[0] in ONE_LETTER:
        return lead[0].title()
    return None


def extract_ligand_atoms(obj: Dict[str, Any],
                         drop_cryo: bool = True,
                         max_atoms: int = 0,
                         contact_cutoff: float = 0.0) -> Tuple[np.ndarray, List[str]]:
    """-> (coords [N,3] float32, elements [N] list[str]). Empty if no ligand."""
    coords: List[List[float]] = []
    elems: List[str] = []
    for lig in get_ligand_section(obj):
        if not isinstance(lig, dict):
            continue
        name = str(lig.get("ligand_name") or "").upper().strip()
        if drop_cryo and name in CRYO_BLACKLIST:
            continue
        for atom_name, xyz in (lig.get("coords") or {}).items():
            if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                continue
            try:
                x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            except (TypeError, ValueError):
                continue
            if x != x or y != y or z != z:   # NaN
                continue
            el = _element_of(atom_name, name)
            if el is None:
                continue
            coords.append([x, y, z])
            elems.append(el)
    if not coords:
        return np.zeros((0, 3), dtype=np.float32), []
    coords = np.asarray(coords, dtype=np.float32)

    # distance/contact filter: drop ligand atoms not near any protein/NA backbone
    # atom (unbound buffer/solvent ions floating in the crystal channels).
    if contact_cutoff and contact_cutoff > 0:
        macro = _macromolecule_coords(obj)
        if len(macro):
            d2min = np.full(len(coords), np.inf, dtype=np.float32)
            for s in range(0, len(macro), 20000):          # chunk to bound memory
                mb = macro[s:s + 20000]
                d2 = ((coords[:, None, :] - mb[None, :, :]) ** 2).sum(-1)  # [n,m]
                d2min = np.minimum(d2min, d2.min(axis=1))
            keep = d2min <= (contact_cutoff * contact_cutoff)
            coords = coords[keep]
            elems = [e for e, k in zip(elems, keep) if k]
        if len(coords) == 0:
            return np.zeros((0, 3), dtype=np.float32), []

    if max_atoms and len(coords) > max_atoms:
        coords, elems = coords[:max_atoms], elems[:max_atoms]
    return coords, elems


def extract_ligand_atoms_detailed(obj: Dict[str, Any],
                                  drop_cryo: bool = True,
                                  contact_cutoff: float = 0.0
                                  ) -> Dict[str, Any]:
    """Like extract_ligand_atoms, keeping the identity it throws away."""
    coords: List[List[float]] = []
    elems: List[str] = []
    inst: List[int] = []
    atom_names: List[str] = []
    names: List[str] = []
    unresolved = 0
    for lig in get_ligand_section(obj):
        if not isinstance(lig, dict):
            continue
        name = str(lig.get("ligand_name") or "").upper().strip()
        if drop_cryo and name in CRYO_BLACKLIST:
            continue
        here = len(names)
        kept_any = False
        for atom_name, xyz in (lig.get("coords") or {}).items():
            if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                continue
            el = _element_of(atom_name, name)
            if el is None:
                unresolved += 1
                continue
            coords.append([float(v) for v in xyz])
            elems.append(el)
            inst.append(here)
            atom_names.append(str(atom_name))
            kept_any = True
        if kept_any:
            names.append(name)

    xyz = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    inst_a = np.asarray(inst, dtype=np.int64)
    if contact_cutoff > 0 and len(xyz):
        env = _macromolecule_coords(obj)
        if len(env):
            d = np.linalg.norm(xyz[:, None, :] - env[None, :, :], axis=-1).min(axis=1)
            keep = d <= float(contact_cutoff)
            xyz, inst_a = xyz[keep], inst_a[keep]
            elems = [e for e, k in zip(elems, keep) if k]
            atom_names = [a for a, k in zip(atom_names, keep) if k]
    return dict(coords=xyz, elements=elems, instance=inst_a, names=names,
                atom_names=atom_names, unresolved=unresolved)


# slots 0,1,2 are the frame atoms (always finite); 3,4,5 are unused (NaN)


def summarize(obj: Dict[str, Any]) -> str:
    ligs = get_ligand_section(obj)
    kept_c, kept_e = extract_ligand_atoms(obj, drop_cryo=True)
    all_c, _ = extract_ligand_atoms(obj, drop_cryo=False)
    names = [str(l.get("ligand_name")) for l in ligs if isinstance(l, dict)]
    return (f"name={obj.get('name')}  ligands={names}  "
            f"atoms(all)={len(all_c)}  atoms(kept)={len(kept_c)}  "
            f"elems(kept)={sorted(set(kept_e))}")


if __name__ == "__main__":                    # quick self-test on a file
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "re_train_N.jsonl"
    n = 0
    with open(path) as fh:
        for line in fh:
            o = json.loads(line)
            if get_ligand_section(o):
                print(summarize(o))
                n += 1
                if n >= 10:
                    break
