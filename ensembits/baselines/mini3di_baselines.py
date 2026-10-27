"""mini3di (Foldseek 3Di) baselines from C-alpha coordinates, as library functions.

What it does
------------
Three flavours, all from the `mini3di` package (Foldseek's 20-letter
structural alphabet and the 2-D centroid of each letter):

    three_di_tokens(ca)       single-frame
    vote_3di(ca_K)            multi-frame plurality (nearest-centroid tie break)
    protprofile_k(ca_K)       multi-frame histogram + centroids (60-D)

N and C are rebuilt with ideal geometry (`_backbone.reconstruct_backbone_ideal`)
when not given. `mini3di` is optional: `have_mini3di()` reports whether it is
installed.

Paper
-----
`appendix.tex` \\label{app:baselines}: *3Di_tokens*, *Vote_3di*,
*ProtProfileMD_K*.

Origin
------
Copied from `Ensembits/utils/baselines/mini3di_baselines.py`. Logic unchanged;
docstring rewritten. Not called by the scripts of this codebase (the MISATO
caches are built from the real backbone by `data_prep/baselines/build_mini3di_*`
and `build_protprofile_K_misato_foldseek.py`); copied because
`ensembits/baselines/__init__.py` imports it. 2026-10-18 (anova_audit): the voting
rule of `vote_3di` is split out unchanged as `vote_from_frame_tokens`, used by
`downstream/analyses/token_anova.py` on the cached K = 10 mini3di tokens.

Used by the reported model: no (the Vote_3Di row of the token ANOVA uses the rule).

Inputs / outputs
----------------
none (library)

How to run
----------
    from ensembits.baselines import three_di_tokens, vote_3di, protprofile_k
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# mini3di's centroids are part of its public encoder; we re-load them
# at call time so this module imports cleanly without mini3di present.
_MINI3DI_VOCAB = 20
_CENTROID_DIM = 2


def have_mini3di() -> bool:
    """True iff `mini3di` is importable."""
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
    """`(20, 2) float32` — the 2-D centroid embedding of the 20 mini3di
    states. Suitable as `--codebook` for `probe_rmsf.py` when the
    feature is a `(L,) int64` mini3di-token sequence."""
    m = _import_mini3di()
    return np.asarray(m.Encoder._CENTROIDS, dtype=np.float32)


def _backbone_from_ca(ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct N / C from a Cα-only chain via ideal geometry, then
    derive Cβ via mini3di's reference frame helper. Returns
    `(N, C, Cb)` aligned with the input `ca` (L, 3).

    Uses the shared `reconstruct_backbone_ideal` so the N–CA–C frame is
    non-degenerate (the legacy midpoint rule made the frame colinear,
    which collapsed the mini3di-derived Cβ direction onto the local
    Cα-Cα-Cα plane).
    """
    from ._backbone import reconstruct_backbone_ideal
    m = _import_mini3di()
    n_atom, c_atom, _ = reconstruct_backbone_ideal(ca)
    vce = m.VirtualCenterEncoder()
    cb = vce._approximate_cb_position(ca.astype(np.float32),
                                       n_atom.astype(np.float32),
                                       c_atom.astype(np.float32))
    return (n_atom.astype(ca.dtype, copy=False),
            c_atom.astype(ca.dtype, copy=False),
            np.asarray(cb, dtype=np.float32))


def three_di_tokens(ca: np.ndarray,
                    n: np.ndarray | None = None,
                    c: np.ndarray | None = None,
                    cb: np.ndarray | None = None) -> np.ndarray:
    """Single-frame Foldseek 3Di tokens.

    `ca`: `(L, 3)` Cα coordinates (Å). When real `n` / `c` / `cb` are
    available (e.g. all-atom mdCATH frames), pass them in to skip the
    Cα-only reconstruction. Otherwise N / C are filled in via ideal
    geometry and Cβ is approximated via mini3di's reference frame
    helper.

    Returns `(L,) int64` token IDs in `[0, 20)`.
    """
    if ca.ndim != 2 or ca.shape[-1] != 3:
        raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")
    L = ca.shape[0]
    m = _import_mini3di()
    if n is None or c is None or cb is None:
        n_recon, c_recon, cb_recon = _backbone_from_ca(ca.astype(np.float32))
        if n is None: n = n_recon
        if c is None: c = c_recon
        if cb is None: cb = cb_recon
    enc = m.Encoder()
    states = enc.encode_atoms(ca=ca.astype(np.float32),
                                cb=np.asarray(cb, dtype=np.float32),
                                n=np.asarray(n, dtype=np.float32),
                                c=np.asarray(c, dtype=np.float32))
    # `enc.encode_atoms` returns a masked array — positions 0 and L-1
    # (and any residue with NaN backbone) are masked. Upstream's
    # `Encoder.build_sequence` consumes via `.filled()` so masked
    # positions become `_INVALID_STATE = 2`. We must do the same here;
    # `np.asarray(states)` would silently discard the mask and leak
    # the (uninitialised-descriptor) garbage that the VAE produced.
    return np.asarray(states.filled(), dtype=np.int64).reshape(L)


def vote_3di(ca_K: np.ndarray) -> np.ndarray:
    """Multi-frame mini3di token, picked by per-residue plurality.

    `ca_K`: `(K, L, 3)` Cα coordinates for K conformations of the
    same protein. Tokenizes each frame independently, then for each
    residue:
      * if some letter has a strict plurality (count ≥ 2 and unique),
        that letter wins;
      * else fall back to the letter whose centroid is nearest to the
        mean of the K per-frame centroids (so ties resolve in the
        natural geometry of the alphabet).

    Returns `(L,) int64`.
    """
    if ca_K.ndim != 3 or ca_K.shape[-1] != 3:
        raise ValueError(f"expected ca_K shape (K, L, 3), got {ca_K.shape}")
    K, L, _ = ca_K.shape
    per_frame = np.stack([three_di_tokens(ca_K[k]) for k in range(K)], axis=-1)
    # per_frame: (L, K)
    return vote_from_frame_tokens(per_frame, mini3di_centroids())


def vote_from_frame_tokens(per_frame: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """The Vote_3di rule on precomputed per-frame tokens `(L, K)` (e.g. a K-frame cache).

    Strict plurality with count >= 2 wins; otherwise the letter whose centroid is nearest
    to the mean of the K per-frame centroids; among letters at the same distance (within
    1e-6) the lowest letter id. Returns `(L,) int64`.
    Split out of `vote_3di` (anova_audit, 2026-10-18) so `token_anova.py` applies the paper
    rule to the cached K = 10 tokens.
    FIXED(anova_audit fix round 2, 2026-10-18): distances in float64 and an explicit
    lowest-id tie-break (old: float32 argmin, which picked either of two exactly tied
    centroids depending on rounding; on mdCATH-div 4,785 residues have exactly tied nearest
    centroids in float64, the next-closest gap is > 1e-5).
    """
    per_frame = np.asarray(per_frame, dtype=np.int64)
    centroids = np.asarray(centroids, dtype=np.float64)
    M = centroids.shape[0]
    L = per_frame.shape[0]
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
            d = np.linalg.norm(centroids - avg, axis=-1)
            out[i] = int(np.flatnonzero(d <= d.min() + 1e-6)[0])
    return out


def protprofile_k(ca_K: np.ndarray) -> np.ndarray:
    """Multi-frame mini3di histogram + centroid concat (60-D feature
    per residue), as defined in the manuscript's app:baselines.

    For each residue r, let `h[r] ∈ Δ^{19}` be the empirical
    distribution of mini3di tokens across the K frames. The feature
    is the flattened
        `[h_0, c_{0,x}, c_{0,y}, h_1, c_{1,x}, c_{1,y}, ..., h_{19}, c_{19,x}, c_{19,y}]`
    so the 20 histogram entries are interleaved with their fixed 2-D
    centroid coordinates → `(L, 60) float32`.
    """
    if ca_K.ndim != 3 or ca_K.shape[-1] != 3:
        raise ValueError(f"expected ca_K shape (K, L, 3), got {ca_K.shape}")
    K, L, _ = ca_K.shape
    centroids = mini3di_centroids()                        # (20, 2)
    per_frame = np.stack([three_di_tokens(ca_K[k]) for k in range(K)], axis=-1)
    hist = np.zeros((L, 20), dtype=np.float32)
    for i in range(L):
        np.add.at(hist[i], per_frame[i], 1.0 / K)
    # Interleave histogram with its centroid annotation → (L, 20, 3) → (L, 60)
    cb = np.broadcast_to(centroids, (L, 20, _CENTROID_DIM)).copy()
    out = np.concatenate([hist[..., None], cb], axis=-1).reshape(L, 60)
    return out
