# -*- coding: utf-8 -*-
"""Geometry computed from coordinates at the point of use."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

PROTEIN, NUCLEIC = 0, 1

SLOTS: Dict[int, Tuple[int, ...]] = {
    PROTEIN: (0, 1, 2, 3),              # N, CA, C, O
    NUCLEIC: (0, 1, 2, 3, 4, 5, 6),     # P, C4', N, O5', O3', C5', C3'
}

TYPE_TO_CLASS = (PROTEIN, NUCLEIC, NUCLEIC)


def geometry_class(type_vec: torch.Tensor) -> torch.Tensor:
    """Geometry class per residue, from the polymer type id."""
    table = torch.tensor(TYPE_TO_CLASS, device=type_vec.device, dtype=torch.long)
    if bool((type_vec < 0).any()) or bool((type_vec >= table.shape[0]).any()):
        raise ValueError('a polymer type id has no geometry class')
    return table[type_vec]


def n_atoms(cls: int, n_virtual: int) -> int:
    """Real slots, then the fixed virtual atoms, then the learnable ones."""
    return n_anchors(cls) + n_virtual


FIXED_VIRTUAL: Dict[int, Tuple[Tuple[float, float, float], ...]] = {
    PROTEIN: ((-0.58273431, 0.56802827, -0.54067466),),
    NUCLEIC: (),
}


def n_fixed(cls: int) -> int:
    return len(FIXED_VIRTUAL[cls])


def fixed_coeffs(cls: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """[n_fixed(cls), 3]. Empty where the class has none."""
    rows = FIXED_VIRTUAL[cls]
    if not rows:
        return torch.zeros(0, 3, device=device, dtype=dtype)
    return torch.tensor(rows, device=device, dtype=dtype)


def n_anchors(cls: int) -> int:
    """Real slots plus the fixed virtual atoms, which pair like real ones."""
    return len(SLOTS[cls]) + n_fixed(cls)


def anchor_pairs(cls_a: int, cls_b: int) -> Tuple[Tuple[int, int], ...]:
    """The complete ordered product over both classes' anchor atoms."""
    return tuple((a, b) for a in range(n_anchors(cls_a))
                 for b in range(n_anchors(cls_b)))


def intra_pairs(cls: int) -> Tuple[Tuple[int, int], ...]:
    """Unordered pairs among one residue's own real slots, for node features."""
    n = len(SLOTS[cls])
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def virtual_pairs(n_virtual: int) -> Tuple[Tuple[int, int], ...]:
    """Virtual point i to i, then every ordered cross pair, as PiFold does."""
    out = [(i, i) for i in range(n_virtual)]
    out += [(i, j) for i in range(n_virtual) for j in range(n_virtual) if i != j]
    return tuple(out)


# Backbone torsions a residue would carry in the node branch: phi, psi and
# omega for a protein, alpha to zeta for a nucleotide, each as a sine, a
# cosine and a validity flag. The released weights hold a node projection
# sized for them and never read it, so only the count is needed here.
N_TORSIONS: Dict[int, int] = {PROTEIN: 3, NUCLEIC: 6}


def torsion_width(cls: int) -> int:
    return 3 * N_TORSIONS[cls]


FEATURE_VERSION = 2

def rbf(d: torch.Tensor, n_rbf: int = 16, d_min: float = 2.0,
        d_max: float = 22.0) -> torch.Tensor:
    """The radial basis ProteinMPNN uses, same centres and same width."""
    mu = torch.linspace(d_min, d_max, n_rbf, device=d.device, dtype=d.dtype)
    sigma = (d_max - d_min) / n_rbf
    return torch.exp(-(((d[..., None] - mu) / sigma) ** 2))


def residue_frames(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                   valid: torch.Tensor,
                   eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gram-Schmidt frame per residue from slots 0, 1 and 2."""
    if valid.shape[0] != x0.shape[0]:
        raise ValueError('valid and coordinates disagree on the residue count')
    v = valid[:, 0] & valid[:, 1] & valid[:, 2]

    e1 = x2 - x1
    n1 = e1.norm(dim=-1, keepdim=True)
    u = x0 - x1
    # Project before normalising, so a collinear triple is caught by the norm of
    # the residual rather than producing an arbitrary perpendicular.
    e1s = e1 / n1.clamp_min(eps)
    w = u - (u * e1s).sum(-1, keepdim=True) * e1s
    n2 = w.norm(dim=-1, keepdim=True)

    ok = v & (n1.squeeze(-1) > eps) & (n2.squeeze(-1) > eps)
    e2s = w / n2.clamp_min(eps)
    e3s = torch.linalg.cross(e1s, e2s, dim=-1)
    R = torch.stack([e1s, e2s, e3s], dim=-1)          # columns, local -> global
    eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
    R = torch.where(ok[:, None, None], R, eye)
    t = torch.where(ok[:, None], x1, torch.zeros_like(x1))
    return R, t, ok


def init_virtual_coeffs(n_virtual: int, device=None,
                        dtype=torch.float32) -> torch.Tensor:
    """Starting coefficients for the virtual points, in the residue frame."""
    prior = torch.tensor([
        [-0.58273431, 0.56802827, -0.54067466],
        [0.0, 0.83867057, -0.54463904],
        [0.01984028, -0.78380804, -0.54183614],
    ], device=device, dtype=dtype)
    if n_virtual <= prior.shape[0]:
        return prior[:n_virtual].clone()
    extra = torch.rand(n_virtual - prior.shape[0], 3, device=device, dtype=dtype)
    return torch.cat([prior, extra], dim=0)


def virtual_positions(coeffs: torch.Tensor, x0: torch.Tensor,
                      x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Place virtual points for every residue. [n, n_virtual, 3]."""
    b = x1 - x0
    c = x2 - x1
    a = torch.linalg.cross(b, c, dim=-1)
    basis = torch.stack([a, b, c], dim=1)                  # [n, 3, 3]
    return x1[:, None, :] + torch.einsum('vk,nkj->nvj', coeffs.to(basis.dtype), basis)


def atom_table(X: torch.Tensor, cls: int, coeffs: torch.Tensor) -> torch.Tensor:
    """[n, n_atoms(cls), 3]: real slots, then fixed virtual, then learnable."""
    blocks = [X[:, SLOTS[cls], :]]
    fixed = fixed_coeffs(cls, X.device, X.dtype)
    if fixed.shape[0]:
        blocks.append(virtual_positions(fixed, X[:, 0], X[:, 1], X[:, 2]))
    if coeffs.shape[0]:
        blocks.append(virtual_positions(coeffs, X[:, 0], X[:, 1], X[:, 2]))
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)


def atom_validity(valid: torch.Tensor, cls: int,
                  frame_ok: torch.Tensor, n_virtual: int) -> torch.Tensor:
    """[n, n_atoms(cls)] bool, lining up with atom_table."""
    real = valid[:, SLOTS[cls]]
    n_virt = n_fixed(cls) + n_virtual
    if n_virt == 0:
        return real
    virt = frame_ok[:, None].expand(-1, n_virt)
    return torch.cat([real, virt], dim=1)


def pair_rbf(atoms_recv: torch.Tensor, atoms_send: torch.Tensor,
             ok_recv: torch.Tensor, ok_send: torch.Tensor,
             recv: torch.Tensor, send: torch.Tensor,
             pairs: Sequence[Tuple[int, int]], n_rbf: int = 16,
             d_min: float = 2.0, d_max: float = 22.0
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Distances for one relation's edge subset, with their validity."""
    if not pairs:
        z = atoms_recv.new_zeros((recv.shape[0], 0))
        return z, z.to(torch.bool)
    feats, masks = [], []
    for a, b in pairs:
        pa, pb = atoms_recv[recv, a], atoms_send[send, b]
        m = ok_recv[recv, a] & ok_send[send, b]
        d = (pa - pb).norm(dim=-1)
        feats.append(rbf(d, n_rbf, d_min, d_max) * m[:, None].to(pa.dtype))
        masks.append(m)
    return torch.cat(feats, dim=-1), torch.stack(masks, dim=-1)


def local_directions(R_recv: torch.Tensor, t_recv: torch.Tensor,
                     atoms_send: torch.Tensor, ok_send: torch.Tensor,
                     recv: torch.Tensor, send: torch.Tensor,
                     frame_ok: torch.Tensor, eps: float = 1e-6
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit vectors from the receiver to each of the sender's atoms, in the."""
    A = atoms_send.shape[1]
    d = atoms_send[send] - t_recv[recv][:, None, :]
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(eps)
    local = torch.einsum('eij,eaj->eai', R_recv[recv].transpose(-1, -2), d)
    m = ok_send[send] & frame_ok[recv][:, None]
    local = local * m[..., None].to(local.dtype)
    return local.reshape(-1, A * 3), m


def relative_position(res_idx: torch.Tensor, chain_idx: torch.Tensor,
                      recv: torch.Tensor, send: torch.Tensor,
                      clip: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clipped sequence offset and a same-chain flag for every edge."""
    same = chain_idx[recv] == chain_idx[send]
    off = (res_idx[recv] - res_idx[send]).clamp(-clip, clip)
    off = torch.where(same, off, torch.zeros_like(off))
    return off + clip, same

def node_centers(X: torch.Tensor, valid: torch.Tensor,
                 reference_slot: int = 1) -> torch.Tensor:
    """One position per residue, for building the neighbour graph. [n, 3]."""
    ref_ok = valid[:, reference_slot]
    w = valid.to(X.dtype)
    count = w.sum(dim=1)
    if bool((count == 0).any()):
        bad = int((count == 0).sum())
        raise ValueError(f'{bad} residues have no valid atom and cannot be placed')
    centroid = (X * w[..., None]).sum(dim=1) / count[:, None]
    return torch.where(ref_ok[:, None], X[:, reference_slot, :], centroid)


def knn_edges(center: torch.Tensor, batch_id: torch.Tensor, k: int = 32
              ) -> torch.Tensor:
    """[2, n_edges] with row 0 the receiver and row 1 the sender."""
    if k < 1:
        raise ValueError('k must be positive')
    if center.shape[0] != batch_id.shape[0]:
        raise ValueError('centers and batch ids disagree on the residue count')
    recv_all, send_all = [], []
    for b in torch.unique(batch_id, sorted=True):
        idx = (batch_id == b).nonzero(as_tuple=False).squeeze(-1)
        m = idx.numel()
        if m < 2:
            continue
        d = torch.cdist(center[idx], center[idx])
        d.fill_diagonal_(float('inf'))
        kk = min(k, m - 1)
        nbr = d.topk(kk, dim=1, largest=False).indices
        recv_all.append(idx.view(-1, 1).expand(-1, kk).reshape(-1))
        send_all.append(idx[nbr].reshape(-1))
    if not recv_all:
        return center.new_zeros((2, 0), dtype=torch.long)
    edge = torch.stack([torch.cat(recv_all), torch.cat(send_all)])
    # Sorted by receiver then sender, so a segment reduction over the receiver
    # reads contiguous memory and two runs give identical edge order.
    order = torch.argsort(edge[0] * center.shape[0] + edge[1])
    return edge[:, order]
