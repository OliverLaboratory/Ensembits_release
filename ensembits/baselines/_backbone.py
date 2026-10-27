"""Rebuild N / C / O backbone atoms from a C-alpha chain (baseline helper).

What it does
------------
Two reconstructions live here:

- `reconstruct_n_c` / `reconstruct_o`: the legacy *midpoint* rule
  (N[i] = C[i-1] = midpoint of CA[i-1], CA[i]). Degenerate (phi/psi
  collapse to constants); only used here as the fallback for chains with
  fewer than 2 residues.
- `reconstruct_backbone_ideal`: *ideal-geometry* N and C (|N-CA| = 1.46 A,
  |CA-C| = 1.52 A, angle N-CA-C = 111 deg) plus a carbonyl O placed in the
  trigonal plane at C using the next residue's N. Used by the ProToken
  wrapper, by `_real_or_ideal_bb.py` when no real backbone matches, and by
  the ESM3struct / AminoAseed token builders when they are called without a
  real backbone (the frame-pooled features, see
  `data_prep/baselines/build_pooled_baseline_features.py`).
- `atom37_from_ca`: atom37-shaped array with ideal N / CA / C.

Duplication: the N / C part of `reconstruct_backbone_ideal` is the same
construction, with the same constants (1.191, 1.264, 0.844, fallback axis), as
`ensembits/ideal_backbone.py` `_reconstruct_n_c_ideal`, which the descriptor
uses. It is kept here unchanged so the baseline token caches are reproduced
with the code that built them; this module adds the O atom.

Paper
-----
`appendix.tex` \\label{app:baselines}, first paragraph ("missing backbone atoms
are reconstructed with ideal geometry (|N-CA| = 1.46 A, |CA-C| = 1.52 A,
N-CA-C ~ 111 deg). For datasets we used in this paper, such approximation is
not used."), and \\subsection{Conventions}.

Origin
------
Copied from `Ensembits/utils/baselines/_backbone.py`. Logic unchanged; docstring
rewritten. (The original comment said N and C are placed "symmetrically out of
the local plane"; both are offset along the same in-plane direction n, as noted
in `ensembits/ideal_backbone.py`.)

Used by the reported model: no (baselines only).

Inputs / outputs
----------------
none (library)

How to run
----------
    from ensembits.baselines._backbone import reconstruct_backbone_ideal
    n, c, o = reconstruct_backbone_ideal(ca)     # ca: (L, 3)
"""
from __future__ import annotations

import numpy as np


def reconstruct_n_c(ca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cα `(L, 3)` → `(N, C)` placeholder atoms via consecutive Cα midpoints.

    Legacy: forces `N[i] = C[i-1] = midpoint(CA[i-1], CA[i])`, which
    makes φ/ψ torsion features degenerate. Use
    `reconstruct_backbone_ideal` for any consumer that reads torsion
    angles (e.g. ProToken).
    """
    n = np.zeros_like(ca); n[1:] = (ca[:-1] + ca[1:]) / 2.0; n[0] = ca[0]
    c = np.zeros_like(ca); c[:-1] = (ca[:-1] + ca[1:]) / 2.0; c[-1] = ca[-1]
    return n.astype(ca.dtype, copy=False), c.astype(ca.dtype, copy=False)


def reconstruct_o(ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Approximate carbonyl-O at 1.23 Å along the C → Cα reverse direction.

    Legacy companion to `reconstruct_n_c`. Geometrically wrong (real O
    is not on the C–CA line) but ProToken's preprocessor doesn't read O
    for torsion features, so it only affects the encoder's raw atom
    channel.
    """
    eps = 1e-8
    cca = ca - c
    cca = cca / (np.linalg.norm(cca, axis=-1, keepdims=True) + eps)
    return (c - 1.23 * cca).astype(ca.dtype, copy=False)


def atom37_from_ca(ca: np.ndarray) -> np.ndarray:
    """Cα `(L, 3)` → atom37-shaped tensor `(L, 37, 3)` with N / Cα / C
    in slots [0, 1, 2]; remaining 34 slots zeroed (unused by the
    structural-token encoders we wrap).

    Uses `reconstruct_backbone_ideal` so the rigid frames the encoder
    builds from N/CA/C aren't degenerate (the legacy midpoint rule
    forces N[i] = C[i-1], collapsing those frames to a line).
    """
    L = ca.shape[0]
    n, c, _ = reconstruct_backbone_ideal(ca)
    out = np.zeros((L, 37, 3), dtype=np.float32)
    out[:, 0, :] = n
    out[:, 1, :] = ca
    out[:, 2, :] = c
    return out


# --- Ideal-geometry Cα → backbone reconstruction --------------------------
# Solved offline from |N-CA|=1.46 Å, |CA-C|=1.52 Å, ∠N-CA-C=111° with N
# and C placed symmetrically out of the local Cα plane. β is shared so
# both atoms sit on the same side of the Cα–Cα–Cα plane.
_ALPHA_N = 1.191
_ALPHA_C = 1.264
_BETA = 0.844
_O_BOND_LEN = 1.23
_EPS = 1e-8


def reconstruct_backbone_ideal(
        ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cα `(L, 3)` → real-valued `(N, C, O)` triple via ideal geometry.

    Each residue's local frame is built from `t = unit(CA[i+1] - CA[i-1])`
    (chain tangent) and `n` (in-plane unit normal pointing away from
    `½(CA[i-1] + CA[i+1])`). N and C are placed at:

        N[i] = CA[i] - α_N · t  +  β · n
        C[i] = CA[i] + α_C · t  +  β · n

    so that |N–CA| = 1.46 Å, |CA–C| = 1.52 Å and ∠N–CA–C = 111°. Termini
    are handled by linear extrapolation of the Cα chain (`ca[-1]` ←
    `2·ca[0] - ca[1]`, mirror at the C end). When CA[i-1], CA[i],
    CA[i+1] are colinear (Cα chain locally straight), `n` is replaced by
    a deterministic perpendicular fallback so the frame stays
    well-defined.

    Carbonyl O is placed in the trigonal-planar plane at C using the
    next residue's reconstructed N: `unit(O - C) ≈ -(unit(CA - C) +
    unit(N_{i+1} - C))`. The C-terminal O extrapolates from the previous
    residue's geometry.

    Crucially this avoids the degeneracy in the midpoint rule
    (`N[i] = C[i-1]`), so φ and ψ are real, conformation-dependent
    dihedrals.
    """
    ca = np.asarray(ca, dtype=np.float32)
    if ca.ndim != 2 or ca.shape[-1] != 3:
        raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")

    L = ca.shape[0]
    if L < 2:
        # Too short for any meaningful frame; fall back to the legacy rule.
        n_leg, c_leg = reconstruct_n_c(ca)
        o_leg = reconstruct_o(ca, c_leg)
        return n_leg, c_leg, o_leg

    # Pad Cα chain by linear extrapolation so every residue has CA[i-1],
    # CA[i+1] available.
    ca_ext = np.empty((L + 2, 3), dtype=np.float32)
    ca_ext[1:-1] = ca
    ca_ext[0] = 2.0 * ca[0] - ca[1]
    ca_ext[-1] = 2.0 * ca[-1] - ca[-2]

    # Chain tangent t at each residue.
    t = ca_ext[2:] - ca_ext[:-2]
    t /= (np.linalg.norm(t, axis=-1, keepdims=True) + _EPS)

    # In-plane perpendicular: project (CA - midpoint) onto plane ⊥ t.
    mid = 0.5 * (ca_ext[:-2] + ca_ext[2:])
    n_raw = ca - mid
    n = n_raw - (n_raw * t).sum(-1, keepdims=True) * t
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)

    # Fallback perpendicular for colinear Cα stretches: pick any axis
    # not parallel to t, then orthogonalize.
    fallback_axis = np.zeros_like(n)
    fallback_axis[..., 1] = 1.0
    parallel = np.abs((fallback_axis * t).sum(-1, keepdims=True)) > 0.9
    fallback_axis = np.where(parallel,
                             np.array([0.0, 0.0, 1.0], dtype=np.float32),
                             fallback_axis)
    fallback = fallback_axis - (fallback_axis * t).sum(-1, keepdims=True) * t
    fallback /= (np.linalg.norm(fallback, axis=-1, keepdims=True) + _EPS)

    use_fallback = n_norm < 1e-3
    n = np.where(use_fallback, fallback, n / (n_norm + _EPS))

    n_atom = ca - _ALPHA_N * t + _BETA * n
    c_atom = ca + _ALPHA_C * t + _BETA * n

    # Carbonyl O: trigonal-planar at C using the next residue's N.
    # For the last residue we extrapolate by mirroring the previous
    # geometry (so |O[-1] - C[-1]| stays ~1.23 Å).
    n_next = np.empty_like(n_atom)
    n_next[:-1] = n_atom[1:]
    n_next[-1] = c_atom[-1] + (n_atom[-1] - ca[-1])

    v_ca = ca - c_atom
    v_ca /= (np.linalg.norm(v_ca, axis=-1, keepdims=True) + _EPS)
    v_nn = n_next - c_atom
    v_nn /= (np.linalg.norm(v_nn, axis=-1, keepdims=True) + _EPS)

    v_o = -(v_ca + v_nn)
    v_o /= (np.linalg.norm(v_o, axis=-1, keepdims=True) + _EPS)
    o_atom = c_atom + _O_BOND_LEN * v_o

    return (n_atom.astype(ca.dtype, copy=False),
            c_atom.astype(ca.dtype, copy=False),
            o_atom.astype(ca.dtype, copy=False))
