"""Backbone frames, nearest neighbours and relative frames, without ESM3.

What it does
------------
The 256-D descriptor (`ensembits/descriptor.py`) needs three geometric operations:

  1. `backbone_frames(coords)`: a rigid frame T_r = (R_r, t_r) per residue from its
     N, CA, C atoms by Gram-Schmidt: origin t_r = CA, x axis along CA - C, y axis
     from N - CA made orthogonal to x, z = x cross y. Residues with missing or
     non-finite coordinates get the frame of the per-protein average N/CA/C.
  2. `knn_edges(coords, mask, k)`: the k residues nearest to each residue by CA
     distance, closest first; the residue itself comes first (distance 0). Pairs
     involving a residue without coordinates are ranked after all real distances,
     by sequence separation.
  3. `relative_frames(rot, trans, edges)`: T_r^{-1} o T_j for every neighbour j,
     flattened to 12 numbers (3x3 rotation, row-major, then translation).

These follow, operation by operation, the conventions of the ESM3 structure
utilities that the descriptor originally used (`build_affine3d_from_coordinates`,
`find_knn_edges` / `knn_graph`, `Affine3D.invert` / `compose`, `node_gather`), so
the descriptor is bit-identical (verified by tests/test_geometry.py and
checks/check_geometry_vs_esm.py). They are an independent implementation: no ESM
code is imported and no model weights are downloaded. This removes the need to
load the ESM3-open structure encoder, whose weights are under a separate
(non-commercial) licence, just to call its kNN routine.

Paper
-----
  * `appendix.tex` \\label{app:descriptor-esm3}: per-residue local frame,
    K = 16 nearest residues by C-alpha distance (no sequence filter, the residue
    itself first), relative frame T^{p}_{r}^{-1} o T^{p}_{j} flattened to 12 numbers.
    (`appendix.tex` still names ESM3's routines; see require_fix #6 for the axis
    order of the frame.)

Origin
------
New file (code_fix T3.1, 2026-10-15). Written from the mathematical definitions,
matching the floating-point operation order of ESM 3.2.1 (`esm.utils.structure.
affine3d`, `esm.utils.misc`, `esm.models.vqvae.StructureTokenEncoder.find_knn_edges`).

Used by the reported model: yes (through `ensembits/descriptor.py`).
"""
from __future__ import annotations

import torch

MAX_SUPPORTED_DISTANCE = 1e6      # coordinates at or above this are treated as missing
_GS_EPS = 1e-10                   # epsilon of the Gram-Schmidt normalisations


def _gram_schmidt(x_axis: torch.Tensor, xy_plane: torch.Tensor, eps: float) -> torch.Tensor:
    """Orthonormal axes (..., 3, 3), columns = (x, y, z), from an x direction and a
    vector in the xy plane. Same operation order as ESM's `_graham_schmidt`."""
    e1 = xy_plane
    denom = torch.sqrt((x_axis ** 2).sum(dim=-1, keepdim=True) + eps)
    x_axis = x_axis / denom
    dot = (x_axis * e1).sum(dim=-1, keepdim=True)
    e1 = e1 - x_axis * dot
    denom = torch.sqrt((e1 ** 2).sum(dim=-1, keepdim=True) + eps)
    e1 = e1 / denom
    e2 = torch.cross(x_axis, e1, dim=-1)
    return torch.stack([x_axis, e1, e2], dim=-1)


def _frame_from_atoms(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor):
    """(rot (..., 3, 3), trans (..., 3)) with origin CA, x along CA - C, xy plane through N."""
    rot = _gram_schmidt(ca - c, n - ca, _GS_EPS).to(torch.float32)
    return rot, ca


def _flat(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Frame -> 12 numbers: rotation flattened row-major, then translation."""
    return torch.cat([rot.flatten(-2), trans], dim=-1)


def _apply_rot(rot: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotate points p (..., 3) by rot (..., 3, 3). The two branches mirror ESM's
    `RotationMatrix.apply` (matmul when the third-last dim is 1, else einsum), which
    round differently; keeping both makes the result bit-identical."""
    if rot.shape[-3] == 1:
        return p @ rot.transpose(-1, -2).squeeze(-3)
    return torch.einsum("...ij,...j", rot, p)


def backbone_frames(coords: torch.Tensor):
    """Per-residue backbone frames.

    coords: (B, L, 3, 3) atoms N, CA, C.
    Returns (rot (B, L, 3, 3) float32, trans (B, L, 3), mask (B, L) bool), where mask
    marks residues with finite coordinates below MAX_SUPPORTED_DISTANCE.
    """
    mask = torch.all(torch.all(torch.isfinite(coords) & (coords < MAX_SUPPORTED_DISTANCE),
                               dim=-1), dim=-1)                                    # (B, L)
    coords = coords.clone().float()
    coords[~mask] = 0
    B, S, _, _ = coords.shape
    # Frame of the per-protein average N/CA/C, used for residues without coordinates.
    avg = coords.masked_fill(~mask[..., None, None], 0).sum(1) / (
        mask.sum(-1)[..., None, None] + 1e-8)                                       # (B, 3, 3)
    avg_rot, avg_trans = _frame_from_atoms(*avg.float().unbind(dim=-2))
    fill_rot = avg_rot.flatten(-2)[..., None, :].expand(B, S, 9)
    fill_trans = avg_trans[..., None, :].expand(B, S, 3)
    eye = torch.eye(3, dtype=torch.float32, device=coords.device).view(1, 1, 3, 3)
    eye = eye.expand(B, S, 3, 3).flatten(-2)
    fill_rot = fill_rot.where(mask.any(-1)[..., None, None], eye)                    # identity if none valid
    fill = torch.cat([fill_rot, fill_trans], dim=-1)                                 # (B, S, 12)
    rot, trans = _frame_from_atoms(*coords.float().unbind(dim=-2))
    flat = _flat(rot, trans).where(mask[..., None], fill)                            # (B, S, 12)
    return flat[..., :9].unflatten(-1, (3, 3)).to(torch.float32), flat[..., 9:], mask


def knn_edges(coords: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Indices (B, L, min(k, L)) of the k nearest residues by CA distance, closest first.

    coords (B, L, 3, 3) N/CA/C; mask (B, L) from `backbone_frames`. Pairs with a masked
    residue are ranked by 100 x sequence separation + MAX_SUPPORTED_DISTANCE, i.e. after
    every real distance. Same ranking as ESM's `knn_graph` (including tie order, since
    the same `torch.sort` call is used).
    """
    coords = coords.clone()
    coords[~mask] = 0
    ca = coords[..., 1, :]                                                           # (B, L, 3)
    L = ca.shape[-2]
    n_keep = min(k, L)
    ca = ca.nan_to_num()
    pair_masked = ~(mask[..., None, :] & mask[..., :, None])                         # (B, L, L)
    dists = (ca.unsqueeze(-2) - ca.unsqueeze(-3)).norm(dim=-1)
    ar = torch.arange(L, device=ca.device)
    seq_dists = (ar.unsqueeze(-1) - ar.unsqueeze(-2)).abs()
    ranked = (seq_dists.to(dists.dtype).mul(1e2).add(MAX_SUPPORTED_DISTANCE)
              .where(pair_masked, dists))
    _, edges = ranked.sort(dim=-1, descending=False)
    return edges[..., :n_keep]


def gather_neighbours(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """x (B, L, D), edges (B, L, K) -> (B, L, K, D) with out[b, i, k] = x[b, edges[b, i, k]]."""
    B, L, K = edges.shape
    idx = edges.reshape(B, L * K, 1).expand(B, L * K, x.shape[-1])
    return torch.gather(x, 1, idx).reshape(B, L, K, x.shape[-1])


def relative_frames(rot: torch.Tensor, trans: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """T_r^{-1} o T_j for every residue r and neighbour j = edges[r, k], as 12 numbers.

    rot (B, L, 3, 3), trans (B, L, 3), edges (B, L, K) -> (B, L, K, 12).
    """
    B, L, K = edges.shape
    flat = _flat(rot, trans)                                                         # (B, L, 12)
    nb = gather_neighbours(flat, edges)                                              # (B, L, K, 12)
    nb_rot = nb[..., :9].unflatten(-1, (3, 3)).to(torch.float32)
    nb_trans = nb[..., 9:]
    # Inverse of T_r: rotation R^T, translation -(R^T t).
    inv_rot = rot.transpose(-1, -2)
    inv_trans = -_apply_rot(inv_rot, trans)
    inv = _flat(inv_rot, inv_trans).unsqueeze(2).expand(B, L, K, 12).contiguous()
    q_rot = inv[..., :9].unflatten(-1, (3, 3)).to(torch.float32)
    q_trans = inv[..., 9:]
    # Composition (R_q, t_q) o (R_j, t_j) = (R_q R_j, R_q t_j + t_q).
    new_rot = (q_rot @ nb_rot).to(torch.float32)
    new_trans = _apply_rot(q_rot, nb_trans) + q_trans
    return _flat(new_rot, new_trans)
