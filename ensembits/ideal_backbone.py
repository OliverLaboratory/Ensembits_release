"""Rebuild backbone N and C atoms from a C-alpha trace with ideal geometry.

What it does
------------
The descriptor (`ensembits/descriptor.py`) needs N, CA and C for every residue.
When a structure only has C-alpha atoms, this module places N and C at
standard bond lengths and angle:

    t  = unit vector from CA_{i-1} to CA_{i+1}           (local chain direction)
    n  = direction from the midpoint of CA_{i-1}, CA_{i+1} towards CA_i,
         made perpendicular to t (fixed fallback axis if the chain is straight)
    N_i = CA_i - 1.191 t + 0.844 n
    C_i = CA_i + 1.264 t + 0.844 n

which gives |N-CA| = 1.460 A, |CA-C| = 1.520 A and angle N-CA-C = 110.9 deg.
Chain ends use a linearly extrapolated phantom CA.

It fills missing *atoms*, not missing *residues*: residues absent from a
structure are dropped by the cache builders, never re-inserted.

Paper
-----
  * `appendix.tex`, \\subsection{Conventions} ("reconstructed from consecutive
    C-alpha positions using ideal peptide geometry ... 1.46 A, 1.52 A, 111 deg").
  * `appendix.tex`, \\label{app:descriptor-esm3} ("When only C-alpha is
    available, N and C are reconstructed via ideal peptide geometry").

Origin
------
Copied from `Ensembits/ensembits/data/descriptors_3di.py`
(`_reconstruct_n_c_ideal` and its constants). Logic unchanged; comments added.
A longer write-up is in `camera_ready/ideal_geometry_backbone.md`.

Used by the reported model: no. BioEmu-CATH and MISATO provide real backbones for
every residue, so the training corpus never needed it. It is kept because the
descriptor builders fall back to it when a real-backbone cache is missing.
"""
from __future__ import annotations

import numpy as np

# Ideal-geometry Cα→backbone constants (solved from |N-CA|=1.46 Å,
# |CA-C|=1.52 Å, ∠N-CA-C=111°). N and C are both offset along the same in-plane
# direction n (the original comment said "symmetrically out of the plane", which
# is not literal).
_BB_ALPHA_N = 1.191       # how far N sits back along the chain direction t (Å)
_BB_ALPHA_C = 1.264       # how far C sits forward along t (Å)
_BB_BETA = 0.844          # offset of both N and C along n (Å)
_BB_EPS = 1e-8


def _reconstruct_n_c_ideal(ca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C-alpha trace (L, 3) -> ideal-geometry (N, C) positions, each (L, 3)."""
    ca = np.asarray(ca, dtype=np.float32)
    L = ca.shape[0]
    if L < 2:
        n = ca.copy(); c = ca.copy()
        return n, c
    # Pad both ends with a linearly extrapolated phantom CA so every residue has
    # a predecessor and a successor.
    ca_ext = np.empty((L + 2, 3), dtype=np.float32)
    ca_ext[1:-1] = ca
    ca_ext[0] = 2.0 * ca[0] - ca[1]
    ca_ext[-1] = 2.0 * ca[-1] - ca[-2]
    # t: local chain direction CA_{i-1} -> CA_{i+1}
    t = ca_ext[2:] - ca_ext[:-2]
    t /= (np.linalg.norm(t, axis=-1, keepdims=True) + _BB_EPS)
    # n: from the neighbours' midpoint towards CA_i, orthogonalised against t
    mid = 0.5 * (ca_ext[:-2] + ca_ext[2:])
    n_raw = ca - mid
    n = n_raw - (n_raw * t).sum(-1, keepdims=True) * t
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    # Fallback axis for (near) straight chains where n is undefined.
    fb = np.zeros_like(n); fb[..., 1] = 1.0
    fb_par = np.abs((fb * t).sum(-1, keepdims=True)) > 0.9
    fb = np.where(fb_par, np.array([0.0, 0.0, 1.0], dtype=np.float32), fb)
    fb = fb - (fb * t).sum(-1, keepdims=True) * t
    fb /= (np.linalg.norm(fb, axis=-1, keepdims=True) + _BB_EPS)
    n = np.where(nn < 1e-3, fb, n / (nn + _BB_EPS))
    # Place the atoms.
    n_atom = ca - _BB_ALPHA_N * t + _BB_BETA * n
    c_atom = ca + _BB_ALPHA_C * t + _BB_BETA * n
    return n_atom.astype(np.float32), c_atom.astype(np.float32)


reconstruct_n_c_ideal = _reconstruct_n_c_ideal   # readable public alias
