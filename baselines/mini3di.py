"""Mini3Di-based baselines.

Three flavours, all derived from
`mini3di <https://github.com/althonos/mini3di>`_ — Foldseek's 20-letter
structural alphabet plus its 2-D centroid embedding:

- :func:`three_di_tokens` — single-frame tokens
- :func:`vote_3di` — multi-frame per-residue plurality vote
- :func:`protprofile_k` — multi-frame histogram (60-D continuous feature)

The single- and multi-frame functions both accept optional real backbone
atoms (N, C, Cβ); when present, no Cα-only ideal-geometry reconstruction
is needed and the GLY Cβ uses mini3di's tetrahedral helper (not the
trigonal-planar substitution that the older code path used).

Multi-chain proteins must be split before calling — the encoder assumes
single-chain input.

``mini3di`` is an optional dependency: ``have_mini3di()`` returns
``False`` if the import fails, and the functions raise a clear
ImportError when called.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._backbone import reconstruct_backbone_ideal, split_chains_by_peptide_bond


def have_mini3di() -> bool:
    """``True`` iff the ``mini3di`` package can be imported."""
    try:
        import mini3di  # noqa: F401
        return True
    except Exception:
        return False


def _import_mini3di():
    try:
        import mini3di
    except Exception as e:
        raise ImportError(
            "The `mini3di` package is required for 3di-based baselines. "
            "Install with `pip install mini3di` and re-run."
        ) from e
    return mini3di


def mini3di_centroids() -> np.ndarray:
    """``(20, 2)`` float32 — mini3di's 2-D centroid embedding of the 20 states.

    Suitable as ``--codebook`` for ``probes/rmsf.py`` when the feature is
    a ``(L,)`` int64 mini3di-token sequence.
    """
    m = _import_mini3di()
    return np.asarray(m.Encoder._CENTROIDS, dtype=np.float32)


def _backbone_from_ca(ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct N/C via ideal geometry, then Cβ via mini3di's tetrahedral
    helper. Returns ``(N, C, Cβ)``."""
    m = _import_mini3di()
    n_atom, c_atom, _ = reconstruct_backbone_ideal(ca)
    vce = m.VirtualCenterEncoder()
    cb = vce._approximate_cb_position(ca.astype(np.float32),
                                       n_atom.astype(np.float32),
                                       c_atom.astype(np.float32))
    return (n_atom.astype(ca.dtype, copy=False),
            c_atom.astype(ca.dtype, copy=False),
            np.asarray(cb, dtype=np.float32))


def _three_di_tokens_single(ca: np.ndarray,
                              n: Optional[np.ndarray] = None,
                              c: Optional[np.ndarray] = None,
                              cb: Optional[np.ndarray] = None) -> np.ndarray:
    """Single-chain tokenization (no multi-chain logic)."""
    m = _import_mini3di()
    if n is None or c is None or cb is None:
        n_r, c_r, cb_r = _backbone_from_ca(ca.astype(np.float32))
        if n is None: n = n_r
        if c is None: c = c_r
        if cb is None: cb = cb_r
    enc = m.Encoder()
    states = enc.encode_atoms(
        ca=ca.astype(np.float32),
        cb=np.asarray(cb, dtype=np.float32),
        n=np.asarray(n, dtype=np.float32),
        c=np.asarray(c, dtype=np.float32))
    # The masked-array .filled() returns _INVALID_STATE=2 at the
    # terminals (and at any residue with NaN backbone) — mirrors
    # upstream's Encoder.build_sequence. Without .filled() the masked
    # entries leak uninitialised VAE garbage.
    return np.asarray(states.filled(), dtype=np.int64).reshape(-1)


def three_di_tokens(ca: np.ndarray,
                    n: Optional[np.ndarray] = None,
                    c: Optional[np.ndarray] = None,
                    cb: Optional[np.ndarray] = None,
                    *,
                    bb: Optional[np.ndarray] = None,
                    chain_split: bool = True) -> np.ndarray:
    """Single-frame Foldseek 3Di tokens.

    Args:
        ca: ``(L, 3)`` Cα coordinates (Å).
        n, c, cb: optional real ``(L, 3)`` N, C, and Cβ atoms. When
            provided, no Cα-only reconstruction is needed.
        bb: optional ``(L, 4, 3)`` real backbone ``[N, CA, C, O]``;
            convenience alternative to passing ``n``/``c`` separately.
            If supplied, ``cb`` should still be passed (or
            :func:`cb_from_bb_or_real` should populate it).
        chain_split: if True (default) and ``bb`` is provided, split the
            protein on peptide-bond breaks (>5 Å) and tokenize each
            chain independently, then concatenate.

    Returns: ``(L,) int64`` token IDs in ``[0, 20)``.
    """
    if ca.ndim != 2 or ca.shape[-1] != 3:
        raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")

    # Coerce optional bb into per-atom args.
    if bb is not None:
        bb_arr = np.asarray(bb, dtype=np.float32)
        n = bb_arr[:, 0] if n is None else n
        c = bb_arr[:, 2] if c is None else c
    else:
        bb_arr = None

    # Chain-split path
    if chain_split and bb_arr is not None:
        chains = split_chains_by_peptide_bond(bb_arr)
        if len(chains) > 1:
            outs = []
            for s, e in chains:
                cb_s = cb[s:e] if cb is not None else None
                outs.append(_three_di_tokens_single(
                    ca[s:e],
                    n=None if n is None else n[s:e],
                    c=None if c is None else c[s:e],
                    cb=cb_s))
            return np.concatenate(outs)
    return _three_di_tokens_single(ca, n=n, c=c, cb=cb)


def vote_3di(ca_K: np.ndarray,
             bb_K: Optional[np.ndarray] = None,
             cb_K: Optional[np.ndarray] = None,
             *, chain_split: bool = True) -> np.ndarray:
    """Multi-frame mini3di token, picked by per-residue plurality.

    Args:
        ca_K: ``(K, L, 3)`` Cα for K conformations of the same protein.
        bb_K: optional ``(K, L, 4, 3)`` real backbone.
        cb_K: optional ``(K, L, 3)`` real Cβ.

    For each residue, the winning letter is:
    - the strict plurality among the K tokens (count ≥ 2 and unique), or
    - the centroid-closest letter to the K-frame centroid mean (tiebreak).

    Returns ``(L,) int64``.
    """
    if ca_K.ndim != 3 or ca_K.shape[-1] != 3:
        raise ValueError(f"expected ca_K shape (K, L, 3), got {ca_K.shape}")
    K, L, _ = ca_K.shape
    centroids = mini3di_centroids()
    M = centroids.shape[0]
    per_frame = np.stack(
        [three_di_tokens(
            ca_K[k],
            bb=None if bb_K is None else bb_K[k],
            cb=None if cb_K is None else cb_K[k],
            chain_split=chain_split)
         for k in range(K)],
        axis=-1)                          # (L, K)
    out = np.zeros(L, dtype=np.int64)
    for i in range(L):
        toks = per_frame[i]
        counts = np.bincount(toks, minlength=M)
        mx = int(counts.max())
        n_mx = int((counts == mx).sum())
        if mx >= 2 and n_mx == 1:
            out[i] = int(np.argmax(counts))
        else:
            avg = centroids[toks].mean(axis=0)
            out[i] = int(np.argmin(np.linalg.norm(centroids - avg, axis=-1)))
    return out


def protprofile_k(ca_K: np.ndarray,
                   bb_K: Optional[np.ndarray] = None,
                   cb_K: Optional[np.ndarray] = None,
                   *, chain_split: bool = True) -> np.ndarray:
    """Multi-frame mini3di histogram + centroid concat (60-D per residue).

    For each residue r, ``h[r] ∈ Δ^{19}`` is the empirical distribution of
    mini3di tokens across the K frames. The feature interleaves each
    histogram entry with its fixed 2-D centroid coordinates, producing
    ``(L, 60)`` float32.

    NB: this is *algorithmically* equivalent to the upstream ProtProfile
    target, but not *bit*-equivalent. For bit-equivalence with the
    Foldseek + mmseqs profile pipeline, use
    ``scripts/build_protprofile_foldseek.py`` (foldseek 11.79cd10b +
    mmseqs 17.b804f static binaries required).
    """
    if ca_K.ndim != 3 or ca_K.shape[-1] != 3:
        raise ValueError(f"expected ca_K shape (K, L, 3), got {ca_K.shape}")
    K, L, _ = ca_K.shape
    centroids = mini3di_centroids()                   # (20, 2)
    per_frame = np.stack(
        [three_di_tokens(
            ca_K[k],
            bb=None if bb_K is None else bb_K[k],
            cb=None if cb_K is None else cb_K[k],
            chain_split=chain_split)
         for k in range(K)],
        axis=-1)                                       # (L, K)
    hist = np.zeros((L, 20), dtype=np.float32)
    for i in range(L):
        np.add.at(hist[i], per_frame[i], 1.0 / K)
    cb = np.broadcast_to(centroids, (L, 20, 2)).copy()
    return np.concatenate([hist[..., None], cb], axis=-1).reshape(L, 60)
