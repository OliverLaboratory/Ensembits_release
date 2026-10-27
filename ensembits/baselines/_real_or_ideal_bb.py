"""Load the real backbone of one MD frame, or rebuild it from C-alpha (baseline helper).

What it does
------------
  * `load_bb_for_frame(dataset, pid, frame_idx, ca_frame)`: opens the
    real-backbone cache of the dataset (`mdcath_real_bb/<pid>.npz` or
    `misato_real_bb/<pid>.npz`, keys `bb_10`/`cb_10` or `bb_8`/`cb_8`), takes
    frame `frame_idx` and returns it if its C-alpha matches `ca_frame` within
    `match_atol` (1e-3 A). Otherwise (no file, wrong length, no match) it
    rebuilds N / C / O from `ca_frame` with ideal geometry. Returns
    `(bb (L, 4, 3) [N, CA, C, O], cb (L, 3) or None, 'real' | 'ideal')`.
  * `atom37_from_bb`: (L, 4, 3) -> atom37 layout (N, CA, C in slots 0-2, O in 4).
  * `cb_from_bb_or_real`: C-beta per residue for mini3di; real C-beta where
    present, else mini3di's own tetrahedral C-beta
    (`VirtualCenterEncoder._approximate_cb_position`).
  * `split_chains_by_peptide_bond`: chain boundaries where |C_i - N_{i+1}| > 2.5 A
    (5 A before 2026-10-16, code_fix P5), so every baseline encoder is run per chain.
    Real backbones only: callers treat ideal-geometry frames as one chain.
  * `refuse_existing_outputs`, `refuse_ideal_misato_frame`: guards shared by the
    baseline cache builders (code_fix P15).

Every MISATO baseline token builder calls this module for frame 0 (or all 8
frames), so the MISATO caches it reads must come from the fixed parser
(code_fix §4); the builders check that with
`ensembits.data_versions.require_misato_parser` (code_fix C5).

Paper
-----
`appendix.tex` \\label{app:baselines}, first paragraph (ideal-geometry
reconstruction only when N/C atoms are missing; "For datasets we used in this
paper, such approximation is not used").

Origin
------
Copied from `submission_exp/src/_real_or_ideal_bb.py`. Changed: `ROOT` from
`ensembits.paths.REPO_ROOT`; `_ideal_n_c_o` imports
`ensembits.baselines._backbone` instead of putting `REPO_ROOT/Ensembits` on
sys.path and importing `utils.baselines._backbone` (same file, copied). Logic
unchanged. The ideal-geometry path duplicates `ensembits/ideal_backbone.py`
(see `_backbone.py`); `split_chains_by_peptide_bond` is duplicated as
`_split_chains` in `data_prep/baselines/build_protprofile_K_misato_foldseek.py`.
Approved fixes 2026-10-16: chain-break threshold 2.5 Å (code_fix P5); the
builder guards `refuse_existing_outputs` / `refuse_ideal_misato_frame` (P15).

Used by the reported model: no (baseline token builders only).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads  data/cached_descriptors/mdcath_real_bb/<pid>.npz
         data/cached_descriptors/misato_real_bb/<pid>.npz

How to run
----------
    from ensembits.baselines._real_or_ideal_bb import load_bb_for_frame
    bb, cb, source = load_bb_for_frame('misato', pid, 0, ca_frame)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from ensembits.paths import REPO_ROOT as ROOT   # data/ stays in the research repo
CACHE = ROOT / 'data' / 'cached_descriptors'
MDCATH_REAL_BB = CACHE / 'mdcath_real_bb'
MISATO_REAL_BB = CACHE / 'misato_real_bb'


def _ideal_n_c_o(ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cα `(L, 3)` → ideal-geometry `(N, C, O)`."""
    from ensembits.baselines._backbone import reconstruct_backbone_ideal
    return reconstruct_backbone_ideal(ca)


def load_bb_for_frame(
        dataset: str, pid: str, frame_idx: int,
        ca_frame: np.ndarray,
        match_atol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return `(bb, cb, source)` for the given (dataset, pid, frame_idx).

    `bb`: `(L, 4, 3)` `[N, CA, C, O]`.
    `cb`: `(L, 3)` Cβ when `source='real'` (NaN rows for GLY); `None` for
          `source='ideal'` — the caller should derive Cβ from N/CA/C
          using whatever convention the encoder expects.
    `source`: `'real'` if loaded from the dataset's real-bb cache,
              `'ideal'` if the cache was missing or the cache's frame's
              Cα didn't match `ca_frame` within `match_atol`.
    """
    cache_dir = MDCATH_REAL_BB if dataset == 'mdcath' else MISATO_REAL_BB
    p = cache_dir / f'{pid}.npz'
    if p.exists():
        d = np.load(p, allow_pickle=False)
        # FIXED(code_fix C11, 2026-10-15): try every frame set in the file (bb_10,
        #   bb_8) and use the one whose Cα matches. Before, bb_10 was always preferred,
        #   so frames 1..7 of the 8-frame MISATO set (coords_ca8) never matched and
        #   fell back to ideal geometry. Frame 0 is shared (both FPS runs start there).
        for bb_key, cb_key in (('bb_10', 'cb_10'), ('bb_8', 'cb_8')):
            if bb_key not in d.files:
                continue
            bb_all = d[bb_key]
            cb_all = d[cb_key]
            if frame_idx < bb_all.shape[0] and bb_all.shape[1] == ca_frame.shape[0]:
                bb_f = bb_all[frame_idx].astype(np.float32)
                cb_f = cb_all[frame_idx].astype(np.float32)
                if np.allclose(bb_f[:, 1], ca_frame, atol=match_atol):
                    return bb_f, cb_f, 'real'

    # Fallback: ideal-geometry reconstruction from Cα.
    # Paper: appendix.tex app:baselines says this approximation is not used for
    # the paper's datasets; the builders print how many frames were 'ideal'.
    n, c, o = _ideal_n_c_o(ca_frame)
    bb = np.stack([n, ca_frame.astype(np.float32), c, o], axis=1).astype(np.float32)
    return bb, None, 'ideal'


def atom37_from_bb(bb: np.ndarray) -> np.ndarray:
    """`(L, 4, 3) [N, CA, C, O]` → `(L, 37, 3)` with N/CA/C in slots
    [0, 1, 2] and O in slot 4 (atom37 ordering). Other slots zeroed.

    Suitable for AminoAseed and ESM3Struct (which read N/CA/C from
    atom37[:, 0:3]).
    """
    L = bb.shape[0]
    out = np.zeros((L, 37, 3), dtype=np.float32)
    out[:, 0, :] = bb[:, 0]   # N
    out[:, 1, :] = bb[:, 1]   # CA
    out[:, 2, :] = bb[:, 2]   # C
    out[:, 4, :] = bb[:, 3]   # O
    return out


def cb_from_bb_or_real(bb: np.ndarray, cb_real: np.ndarray | None) -> np.ndarray:
    """Resolve a Cβ array per residue, **for mini3di consumers**.

    For non-GLY residues with real Cβ available, use real Cβ. For GLY
    (or any residue with NaN Cβ) and the entire array when `cb_real`
    is None, return the **mini3di-canonical tetrahedral** Cβ produced
    by `mini3di.VirtualCenterEncoder._approximate_cb_position` — chiral,
    out-of-plane, ~1.530 Å bond. This is the convention mini3di was
    trained on.

    Earlier versions returned an in-plane *trigonal-planar* Cβ at GLY
    via `Cβ = CA - 1.522·unit(unit(N-CA)+unit(C-CA))`, which differs
    from mini3di's expected position and shifts mini3di's virtual-
    center computation, producing wrong tokens at GLY *and* at
    non-GLY neighbours via partner-index propagation. The fix below
    pulls mini3di's own Cβ helper through, so a non-NaN return is
    bit-for-bit equivalent to mini3di replacing the NaN internally.
    """
    L = bb.shape[0]
    n_atom = bb[:, 0].astype(np.float32)
    ca_atom = bb[:, 1].astype(np.float32)
    c_atom = bb[:, 2].astype(np.float32)
    # Tetrahedral Cβ via mini3di's helper (cached at module level).
    try:
        import mini3di
        vce = mini3di.VirtualCenterEncoder()
        cb_tetra = np.asarray(
            vce._approximate_cb_position(ca_atom, n_atom, c_atom),
            dtype=np.float32)
    except ImportError:
        # mini3di not available in this env — no consumer would call us
        # for mini3di tokens, but provide a sane fallback for downstream
        # robustness (consumers that don't care about chirality).
        eps = 1e-8
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
    out = np.where(nan_mask[:, None], cb_tetra, cb_real)
    return out.astype(np.float32)


# FIXED(code_fix P5, 2026-10-16): default threshold 2.5 Å (was 5.0 Å), the cutoff of the
#   Ensembits descriptor (ensembits/descriptor.py DEFAULT_CHAIN_BREAK_CUTOFF, DSSP's
#   chain-break distance). With 5.0 Å the baselines treated the 2.5–5 Å gaps between
#   MISATO molecules (16–19% of complexes) as peptide bonds while Ensembits broke there.
#   Authors' decision (preretrain_audit P5). Ideal-geometry backbones have C–N 1.71–4.42 Å
#   and are never split by the callers (they pass one chunk), as the descriptor treats
#   them as connected.
CHAIN_BREAK_THRESHOLD = 2.5


def split_chains_by_peptide_bond(bb: np.ndarray, threshold: float = CHAIN_BREAK_THRESHOLD
                                  ) -> list[tuple[int, int]]:
    """Detect chain boundaries within a concatenated `bb (L, 4, 3)` by
    flagging adjacent residue pairs with `|C[i] − N[i+1]| > threshold`
    (real peptide bonds are ~1.33 Å; measured bonded C–N <= 1.50 Å, breaks
    >= 2.81 Å; threshold 2.5 Å as in the Ensembits descriptor). Returns
    `[(start, end), ...]` half-open intervals — one per chain — that together
    cover `[0, L)`. Meant for real backbones; do not call it on ideal-geometry
    frames (their C–N reaches 4.42 Å).

    Used by tokenizer builders to call the per-chain encoders the
    upstream packages were trained on (mini3di, AminoAseed, ESM3,
    ProToken all assume single-chain inputs and produce wrong
    boundary tokens for concatenated multi-chain inputs).
    """
    L = bb.shape[0]
    if L < 2:
        return [(0, L)]
    c_to_n = np.linalg.norm(bb[:-1, 2] - bb[1:, 0], axis=-1)
    breaks = (np.where(c_to_n > threshold)[0] + 1).tolist()
    bounds = [0] + breaks + [L]
    return list(zip(bounds[:-1], bounds[1:]))


# FIXED(code_fix P15, 2026-10-16): guards shared by the baseline cache builders.
#   Old: a builder silently skipped (or overwrote) an existing output, so stale caches
#   built from the old MISATO parser could be reused, and MISATO frames that fell back to
#   ideal geometry were only counted in the log. New: exit 1 on an existing output unless
#   --overwrite; exit 1 on the first MISATO frame that falls back to ideal geometry unless
#   --allow-ideal (after the rebuild every MISATO frame has a real backbone).
def refuse_existing_outputs(paths, overwrite: bool = False) -> None:
    """Exit 1 if any of `paths` exists, unless `overwrite`."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        print('output already exists (move it aside, see data_prep/baselines/README.md, '
              'or pass --overwrite):\n  ' + '\n  '.join(existing), flush=True)
        raise SystemExit(1)


def refuse_ideal_misato_frame(dataset: str, pid: str, frame_idx: int, source: str,
                              allow_ideal: bool = False) -> None:
    """Exit 1 if a MISATO frame fell back to ideal geometry, unless `allow_ideal`.

    Raises SystemExit (not an Exception), so the builders' per-protein
    `except Exception` does not swallow it.
    """
    if dataset == 'misato' and source == 'ideal' and not allow_ideal:
        print(f'MISATO {pid} frame {frame_idx} fell back to ideal geometry: no frame of '
              f'misato_real_bb/{pid}.npz matches its C-alpha. Every MISATO frame has a real '
              f'backbone after the rebuild (code_fix E0); rebuild the coords pickle from the '
              f'same misato_real_bb, or pass --allow-ideal.', flush=True)
        raise SystemExit(1)
