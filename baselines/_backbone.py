"""Shared backbone helpers used by every structural baseline.

Provides:

- ``reconstruct_backbone_ideal(ca)`` — Cα → real-valued (N, C, O) via
  ideal peptide geometry. Avoids the algebraic degeneracy in the older
  midpoint rule (which forces ``N[i] = C[i-1]``, collapsing the local
  N–CA–C frame to a line and making φ/ψ torsion features constant).
- ``atom37_from_ca(ca)`` / ``atom37_from_bb(bb)`` — assemble the
  (L, 37, 3) atom37 tensor consumed by AminoAseed / ESM3Struct (only
  N/CA/C/O slots are populated).
- ``load_bb_for_pid(pid, dataset, P)`` — read the per-pid ``bb_{P}``
  array from the real-backbone cache; returns ``(P, L, 3, 3)`` N/CA/C.
- ``cb_from_bb_or_real(bb, cb_real)`` — resolve Cβ for mini3di
  consumers; uses real Cβ where available and mini3di's tetrahedral
  helper for GLY (or when real Cβ is missing).
- ``split_chains_by_peptide_bond(bb)`` — chain boundaries within a
  concatenated multi-chain protein via the |C → N| peptide-bond test;
  every structural tokenizer wrapper uses this so multi-chain inputs
  (52 % of MISATO PDBs) tokenize correctly.

All helpers are pure NumPy and take/return ``float32`` arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Ideal-geometry constants, solved offline from
#   |N–CA| = 1.46 Å,  |CA–C| = 1.52 Å,  ∠N–CA–C = 111°
# with N and C placed symmetrically out of the local Cα plane.
_ALPHA_N = 1.191
_ALPHA_C = 1.264
_BETA = 0.844
_O_BOND_LEN = 1.23
_EPS = 1e-8


# ============================================================
# Cα → ideal-geometry backbone
# ============================================================

def reconstruct_backbone_ideal(
        ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cα ``(L, 3)`` → real-valued ``(N, C, O)`` triple.

    Each residue's local frame is built from
    ``t = unit(CA[i+1] - CA[i-1])`` (chain tangent) and the in-plane
    perpendicular ``n``; N and C are placed at
    ``N[i] = CA[i] - α_N · t + β · n`` and
    ``C[i] = CA[i] + α_C · t + β · n``. Carbonyl O is placed in the
    trigonal-planar plane at C using the next residue's reconstructed N.

    The torsion features φ and ψ remain real, conformation-dependent
    dihedrals (unlike the midpoint rule).
    """
    ca = np.asarray(ca, dtype=np.float32)
    if ca.ndim != 2 or ca.shape[-1] != 3:
        raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")
    L = ca.shape[0]
    if L < 2:
        # Degenerate input: any reconstruction is essentially arbitrary.
        n = ca.copy(); c = ca.copy()
        o = ca + np.array([_O_BOND_LEN, 0, 0], dtype=np.float32)
        return n, c, o

    ca_ext = np.empty((L + 2, 3), dtype=np.float32)
    ca_ext[1:-1] = ca
    ca_ext[0] = 2.0 * ca[0] - ca[1]
    ca_ext[-1] = 2.0 * ca[-1] - ca[-2]

    t = ca_ext[2:] - ca_ext[:-2]
    t /= (np.linalg.norm(t, axis=-1, keepdims=True) + _EPS)

    mid = 0.5 * (ca_ext[:-2] + ca_ext[2:])
    n_raw = ca - mid
    n = n_raw - (n_raw * t).sum(-1, keepdims=True) * t
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)

    # Fallback perpendicular for colinear Cα stretches.
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

    # Trigonal-planar carbonyl O at C using the next residue's N.
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


# ============================================================
# atom37 assembly
# ============================================================

def atom37_from_ca(ca: np.ndarray) -> np.ndarray:
    """Cα ``(L, 3)`` → ``(L, 37, 3)`` atom37 with N/CA/C/O in slots 0/1/2/4.

    Reconstructs N, C, O via ideal geometry. The unused slots are zero.
    """
    n, c, o = reconstruct_backbone_ideal(ca)
    L = ca.shape[0]
    out = np.zeros((L, 37, 3), dtype=np.float32)
    out[:, 0, :] = n
    out[:, 1, :] = ca
    out[:, 2, :] = c
    out[:, 4, :] = o
    return out


def atom37_from_bb(bb: np.ndarray) -> np.ndarray:
    """``(L, 4, 3) [N, CA, C, O]`` → ``(L, 37, 3)`` atom37.

    Suitable for AminoAseed and ESM3Struct, which read N/CA/C from
    slots 0/1/2 (and O from slot 4 when present).
    """
    bb = np.asarray(bb, dtype=np.float32)
    L = bb.shape[0]
    out = np.zeros((L, 37, 3), dtype=np.float32)
    out[:, 0, :] = bb[:, 0]   # N
    out[:, 1, :] = bb[:, 1]   # CA
    out[:, 2, :] = bb[:, 2]   # C
    out[:, 4, :] = bb[:, 3]   # O
    return out


# ============================================================
# Real-backbone cache lookup
# ============================================================

def load_bb_for_pid(pid: str, dataset: str = "mdcath", P: int = 10,
                    data_dir: Path | str = DEFAULT_DATA_DIR
                    ) -> np.ndarray | None:
    """Load ``(P, L, 3, 3)`` float32 N/CA/C frames for ``pid``.

    Reads ``{data_dir}/{dataset}_real_bb/<pid>.npz`` (key ``bb_{P}``,
    shape ``(P, L, 4, 3)`` N/CA/C/O). Returns ``None`` if the file or
    key is absent. The 4th atom (O) is dropped from the return; for
    pipelines that need O, call ``np.load(...)[bb_key]`` directly.
    """
    data_dir = Path(data_dir)
    if dataset not in ("mdcath", "misato"):
        raise ValueError(f"Unknown dataset {dataset!r}; expected mdcath or misato")
    path = data_dir / f"{dataset}_real_bb" / f"{pid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    key = f"bb_{P}"
    if key not in d.files:
        return None
    bb = np.asarray(d[key], dtype=np.float32)
    return np.stack([bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]], axis=2)


def load_bb4_for_pid(pid: str, dataset: str = "mdcath", P: int = 10,
                     data_dir: Path | str = DEFAULT_DATA_DIR
                     ) -> np.ndarray | None:
    """Same as ``load_bb_for_pid`` but keeps O — returns ``(P, L, 4, 3)``."""
    data_dir = Path(data_dir)
    path = data_dir / f"{dataset}_real_bb" / f"{pid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    key = f"bb_{P}"
    if key not in d.files:
        return None
    return np.asarray(d[key], dtype=np.float32)


def load_cb_for_pid(pid: str, dataset: str = "mdcath", P: int = 10,
                    data_dir: Path | str = DEFAULT_DATA_DIR
                    ) -> np.ndarray | None:
    """Load ``(P, L, 3)`` real Cβ from the cache (NaN at GLY)."""
    data_dir = Path(data_dir)
    path = data_dir / f"{dataset}_real_bb" / f"{pid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    key = f"cb_{P}"
    if key not in d.files:
        return None
    return np.asarray(d[key], dtype=np.float32)


# ============================================================
# Cβ resolution (for mini3di consumers)
# ============================================================

def cb_from_bb_or_real(bb: np.ndarray, cb_real: np.ndarray | None
                       ) -> np.ndarray:
    """Per-residue Cβ array, mini3di-canonical at GLY (or any NaN row).

    For non-GLY residues with real Cβ available, returns the real
    coordinate. For GLY (Cβ NaN) — and the entire array when ``cb_real``
    is ``None`` — substitutes the **tetrahedral** Cβ produced by
    ``mini3di.VirtualCenterEncoder._approximate_cb_position``. This is
    chiral and out-of-plane, matching mini3di's training convention; the
    older trigonal-planar substitution silently broke mini3di's
    virtual-center computation, propagating into wrong tokens at GLY *and*
    its non-GLY neighbours via partner-index lookups.
    """
    bb = np.asarray(bb, dtype=np.float32)
    n_atom = bb[:, 0]
    ca_atom = bb[:, 1]
    c_atom = bb[:, 2]
    try:
        import mini3di
        vce = mini3di.VirtualCenterEncoder()
        cb_tetra = np.asarray(
            vce._approximate_cb_position(ca_atom, n_atom, c_atom),
            dtype=np.float32)
    except ImportError:
        # mini3di not installed — fall back to trigonal-planar so the
        # function still returns something usable, but flag with a
        # warning since no real mini3di consumer should hit this branch.
        import warnings
        warnings.warn("mini3di not installed; using trigonal-planar Cβ "
                      "(NOT mini3di-canonical).")
        eps = _EPS
        n_ca = n_atom - ca_atom
        n_ca = n_ca / (np.linalg.norm(n_ca, axis=-1, keepdims=True) + eps)
        c_ca = c_atom - ca_atom
        c_ca = c_ca / (np.linalg.norm(c_ca, axis=-1, keepdims=True) + eps)
        cb_dir = -(n_ca + c_ca)
        cb_dir = cb_dir / (np.linalg.norm(cb_dir, axis=-1, keepdims=True) + eps)
        cb_tetra = (ca_atom + 1.522 * cb_dir).astype(np.float32)

    if cb_real is None:
        return cb_tetra
    cb_real = np.asarray(cb_real, dtype=np.float32)
    nan_mask = np.isnan(cb_real).any(axis=-1)
    return np.where(nan_mask[:, None], cb_tetra, cb_real).astype(np.float32)


# ============================================================
# Multi-chain split
# ============================================================

def split_chains_by_peptide_bond(bb: np.ndarray, threshold: float = 5.0
                                  ) -> list[tuple[int, int]]:
    """Split a concatenated multi-chain ``bb (L, 4, 3)`` into chains.

    Returns ``[(start, end), ...]`` half-open intervals — one per chain —
    that together cover ``[0, L)``. Chain boundaries are flagged where
    ``|C[i] − N[i+1]| > threshold`` (real peptide bonds are ~1.33 Å;
    fictitious cross-chain bonds in concatenated multi-chain inputs are
    typically 5+ Å).

    Every structural tokenizer wrapper uses this — mini3di, AminoAseed,
    ESM3Struct, and ProToken all assume single-chain inputs and produce
    wrong boundary tokens for naively concatenated multi-chain proteins
    (52 % of MISATO).
    """
    L = bb.shape[0]
    if L < 2:
        return [(0, L)]
    c_to_n = np.linalg.norm(bb[:-1, 2] - bb[1:, 0], axis=-1)
    breaks = (np.where(c_to_n > threshold)[0] + 1).tolist()
    bounds = [0] + breaks + [L]
    return list(zip(bounds[:-1], bounds[1:]))
