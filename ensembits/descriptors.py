"""ESM3-style geometric descriptor (192-D, K=16 kNN of Affine3D relative frames).

For each residue r in each frame:

    T^p_r            local frame at residue r (Gram-Schmidt on N, Cα, C)
    rel_{r → j}      = T^p_r^{-1} ∘ T^p_j         for j in kNN(r), |kNN|=K
    desc_r           = concat( rel_{r → j}.tensor ) ∈ ℝ^{K·12}

The descriptor is SE(3)-invariant by construction (any global rigid
motion cancels in the composition). The shipped tokenizer uses K = 16
neighbours, giving a 192-D per-residue, per-frame descriptor.

Input:
    coords_ncac : np.ndarray (P, L, 3, 3) float32 — N/CA/C per frame.

Output:
    desc        : np.ndarray (P, L, K·12) float32.

ESM3 weights are loaded once per process (module-level singleton). CPU
inference is plenty fast for whole-corpus tokenization (~0.1 s per
mid-sized protein on CPU); on a 24 GB GPU it's a few ms per protein.
"""
from __future__ import annotations

import numpy as np
import torch

K = 16
T_AFF = 12
DESC_DIM = K * T_AFF  # 192

_MODEL = None


def _get_model(device: str = "cpu"):
    """Lazy-load + cache the ESM3 structure encoder.

    Requires the ``esm`` package (`pip install esm`) and a Hugging Face
    token if the model weights are gated (``HF_TOKEN`` env var).
    """
    global _MODEL
    if _MODEL is None:
        from esm.pretrained import ESM3_structure_encoder_v0
        _MODEL = ESM3_structure_encoder_v0().to(device).eval()
        for p_ in _MODEL.parameters():
            p_.requires_grad_(False)
    return _MODEL


@torch.no_grad()
def compute_esm3_descriptor(
    coords_ncac: np.ndarray,
    k: int = K,
    dihedral: bool = False,   # accepted for signature parity; ignored
    device: str | None = None,
) -> np.ndarray:
    """Compute the 192-D ESM3 relative-frame descriptor.

    Args:
        coords_ncac: ``(P, L, 3, 3)`` float32 N/CA/C coordinates.
        k: pinned to 16 (matches the shipped tokenizer); raises otherwise.
        dihedral: accepted for parity with other descriptor families;
            this descriptor doesn't use dihedrals.
        device: ``cuda`` if available, else ``cpu``.

    Returns:
        ``(P, L, K·12)`` float32 descriptor.
    """
    if k != K:
        raise ValueError(f"esm3desc is pinned to K={K}; got k={k}")
    from esm.utils.structure.affine3d import (
        Affine3D, build_affine3d_from_coordinates)
    from esm.utils.misc import node_gather

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_model(device)

    coords = torch.as_tensor(coords_ncac, dtype=torch.float32, device=device)
    if coords.ndim != 4 or coords.shape[-2:] != (3, 3):
        raise ValueError(
            f"coords_ncac must have shape (P, L, 3, 3); got {tuple(coords.shape)}")
    B, L, _, _ = coords.shape

    affine, affine_mask = build_affine3d_from_coordinates(coords=coords)
    attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
    sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)
    knn_edges, _ = model.find_knn_edges(
        coords, ~attention_mask, coord_mask=affine_mask,
        sequence_id=sequence_id, knn=k)                       # (B, L, K)

    aff_t = affine.tensor                                     # (B, L, T_AFF)
    knn_aff_t = node_gather(aff_t, knn_edges)                 # (B, L, K, T_AFF)
    inv_query = Affine3D.from_tensor(aff_t).invert()          # (B, L)
    neighbor_aff = Affine3D.from_tensor(knn_aff_t)            # (B, L, K)
    inv_query_bcast = Affine3D.from_tensor(
        inv_query.tensor.unsqueeze(2).expand(B, L, k, T_AFF).contiguous())
    rel = inv_query_bcast.compose(neighbor_aff)               # (B, L, K)
    desc = rel.tensor.reshape(B, L, k * T_AFF)                # (B, L, K·12)
    return desc.cpu().numpy().astype(np.float32)
