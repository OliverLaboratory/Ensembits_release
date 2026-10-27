"""The 256-D per-residue, per-frame descriptor that Ensembits tokenizes.

What it computes
----------------
For one protein conformation (frame) and every residue r, the descriptor
describes the backbone neighbourhood of r:

  1. Build a rigid backbone frame T_r for every residue from its N, CA, C atoms
     (ESM3's `build_affine3d_from_coordinates`: origin CA, x axis along CA - C,
     xy plane containing N).
  2. Find the K = 16 residues nearest to r by CA distance (ESM3's
     `find_knn_edges`). The list starts with r itself (distance 0).
  3. For each of those 16 entries j, express its frame in r's frame,
     T_r^{-1} o T_j, flattened to 12 numbers (3x3 rotation + translation), and
     append j's own backbone dihedrals as 4 numbers (sin phi, cos phi, sin psi,
     cos psi).
  4. Concatenate the 16 entries: 16 x (12 + 4) = 256 numbers per residue.

The descriptor has no learnable parameters; it only borrows ESM3's frame and
neighbour routines. It is computed independently for each frame (the
"dynamical" neighbour mode), so neighbours can change between frames.

Paper
-----
  * `appendix.tex`, \\subsection{ESM3-style relative-frame descriptor}
    (\\label{app:descriptor-esm3}): per-residue local frame, per-neighbour block
    g^p_{r,j} in R^16, descriptor f^p_r in R^256, self slot n^p_{r,1} = r.
  * `appendix.tex`, \\subsection{Resulting dimensions}: D_f = 256.
  * `1_method.tex`, \\paragraph{Dynamical mode.}: neighbour set recomputed per frame.

Origin
------
Copied from `submission_exp/src/build_esm3desc_dihedral.py` (functions
`_get_model`, `_dihedral`, `_phi_psi_sincos`, `_descriptor`). Changed
2026-10-15: dihedral sign convention (code_fix §1; "legacy_plus180" reproduces
the old features) and the ESM3-free geometry backend (`ensembits/geometry.py`,
bit-identical; "esm" backend kept for checks). The corpus-building `main()` of
that file lives in `data_prep/build_esm3desc_dihedral.py`.

Used by the reported model: yes, for training data, inference, every downstream
task and ProteinGym.
"""
from __future__ import annotations

import torch

K = 16                      # neighbours per residue (paper: K = 16), incl. the residue itself
P = 10                      # frames per protein in the training corpus (paper: P_max = 10)
T_AFF = 12                  # numbers per relative frame: 3x3 rotation (9) + translation (3)
PER_NB = T_AFF + 4          # + (sin phi, cos phi, sin psi, cos psi)  -> 16 per entry
DESC_DIM = K * PER_NB       # 256 = D_f in the paper

_MODEL = None               # ESM3 structure encoder, loaded once; only its kNN routine is used

# Dihedral sign convention (code_fix §1).
#   "iupac"          : standard phi/psi (paper, `appendix.tex` app:descriptor-esm3). DEFAULT.
#   "legacy_plus180" : the convention every checkpoint trained before 2026-10-15 was trained
#                      with (angles shifted by 180 deg, i.e. features (-sin, -cos)). Use it only
#                      to reproduce results of those checkpoints; `featurize.load_model` picks it
#                      automatically from the checkpoint config.
DIHEDRAL_CONVENTIONS = ("iupac", "legacy_plus180")
DEFAULT_DIHEDRAL_CONVENTION = "iupac"

# Chain breaks (code_fix §6, user decision 2026-10-15). phi_i and psi_{i-1} are undefined
# when residues i-1 and i are not bonded (MISATO chain junctions, residues missing in PDB
# or MD inputs): they are set to (0, 0) like the chain ends when the C(i-1)–N(i) distance
# exceeds the cutoff. Measured C–N: bonded <= 1.50 Å, breaks >= 2.81 Å (BioEmu-CATH,
# mdCATH, MISATO); 2.5 Å is DSSP's chain-break distance.
# Backbones whose N/C were rebuilt from Cα with ideal geometry have no real peptide bond
# (C–N 1.71–4.42 Å measured), so callers pass chain_break_cutoff=None for them: treated
# as connected.
# FIXED(code_fix P29, 2026-10-16): comment only; old text said "C–N 2.4–4.2 Å" for the
#   ideal-geometry backbones, the measured range is 1.71–4.42 Å.
# "auto" resolves by convention: iupac -> 2.5 Å; legacy_plus180 -> None (no detection,
# which reproduces the features of every checkpoint trained before the fix).
DEFAULT_CHAIN_BREAK_CUTOFF = 2.5


def resolve_chain_break_cutoff(cutoff, convention: str = DEFAULT_DIHEDRAL_CONVENTION):
    """'auto' -> 2.5 Å for "iupac", None for "legacy_plus180"; numbers and None pass."""
    if cutoff == "auto":
        return None if convention == "legacy_plus180" else DEFAULT_CHAIN_BREAK_CUTOFF
    if cutoff in (None, "none", "None"):
        return None
    return float(cutoff)


def with_convention(filename: str, convention: str) -> str:
    """Cache file name for a dihedral convention, so new caches never overwrite old ones.

    "legacy_plus180" keeps the original names (e.g. esm3desc_dihedral_K16_P10.npy);
    "iupac" inserts "_iupac" (esm3desc_dihedral_iupac_K16_P10.npy).
    Names without "esm3desc_dihedral" get the tag before the suffix
    (x.npz -> x_iupac.npz), so no name can silently stay the same.
    """
    if convention not in DIHEDRAL_CONVENTIONS:
        raise ValueError(f"unknown dihedral convention {convention!r}")
    if convention == "legacy_plus180":
        return filename
    if "esm3desc_dihedral" in filename:
        return filename.replace("esm3desc_dihedral", f"esm3desc_dihedral_{convention}", 1)
    stem, dot, suffix = filename.rpartition(".")
    return f"{stem}_{convention}.{suffix}" if dot else f"{filename}_{convention}"


def _get_model(device):
    """Load ESM3's structure encoder once and return (model, K).

    Only `model.find_knn_edges` and `model.knn` (= 16) are used; no ESM3 weights
    take part in the descriptor values.
    """
    global _MODEL
    if _MODEL is None:
        from esm.pretrained import ESM3_structure_encoder_v0
        _MODEL = ESM3_structure_encoder_v0().to(device).eval()
        for p_ in _MODEL.parameters():
            p_.requires_grad_(False)
    return _MODEL, _MODEL.knn


def _dihedral(p0, p1, p2, p3, convention: str = DEFAULT_DIHEDRAL_CONVENTION):
    """Dihedral angle (radians) of four points, each (..., 3), via atan2.

    Paper: phi_j and psi_j in `appendix.tex` \\label{app:descriptor-esm3}.
    """
    # FIXED(code_fix §1, 2026-10-15): the standard (IUPAC) formula uses
    #   b0 = p0 - p1. The original code used p1 - p0, which negates x and y below,
    #   so every angle was the true angle + 180 deg (features (-sin, -cos); checked
    #   against mdtraj). "iupac" is now the default; "legacy_plus180" reproduces the
    #   old features for checkpoints trained with them (selected from the
    #   checkpoint config by `featurize.load_model`). An old checkpoint can also be
    #   converted exactly instead: checks/check_dihedral_reparam.py.
    if convention not in DIHEDRAL_CONVENTIONS:
        raise ValueError(f"unknown dihedral convention {convention!r}; use one of {DIHEDRAL_CONVENTIONS}")
    b0 = (p1 - p0) if convention == "legacy_plus180" else (p0 - p1)
    b1 = p2 - p1; b2 = p3 - p2
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-8)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n        # b0 projected off the central bond
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n        # b2 projected off the central bond
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


def _phi_psi_sincos(coords, convention: str = DEFAULT_DIHEDRAL_CONVENTION,
                    chain_break_cutoff="auto"):
    """Backbone dihedrals of every residue, as sine/cosine pairs.

    coords: (B, L, 3, 3) with atoms ordered N, CA, C.
    Returns (B, L, 4) = [sin phi, cos phi, sin psi, cos psi], where
        phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
        psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})
    phi of the first residue and psi of the last are undefined and set to (0, 0),
    and so are phi_i and psi_{i-1} across a chain break (C_{i-1}–N_i farther than
    `chain_break_cutoff` Å in that frame; None = no detection, for ideal-geometry
    backbones and legacy checkpoints; see `resolve_chain_break_cutoff`).

    Paper: `appendix.tex` \\label{app:descriptor-esm3} ("the undefined phi of the
    first residue and psi of the last are set to (0, 0)").
    """
    B, L, _, _ = coords.shape
    N = coords[:, :, 0, :]; CA = coords[:, :, 1, :]; C = coords[:, :, 2, :]
    phi = torch.zeros(B, L, device=coords.device, dtype=coords.dtype)
    psi = torch.zeros(B, L, device=coords.device, dtype=coords.dtype)
    if L >= 2:
        phi[:, 1:] = _dihedral(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:], convention)
        psi[:, :-1] = _dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:], convention)
    feats = torch.stack([phi.sin(), phi.cos(), psi.sin(), psi.cos()], dim=-1)  # (B,L,4)
    # zero-out the two undefined endpoints (sin/cos both 0 as a sentinel)
    feats[:, 0, 0] = 0; feats[:, 0, 1] = 0       # phi[0] undefined
    feats[:, -1, 2] = 0; feats[:, -1, 3] = 0     # psi[L-1] undefined
    # FIXED(code_fix §6, 2026-10-15): chain breaks are treated like chain ends.
    cutoff = resolve_chain_break_cutoff(chain_break_cutoff, convention)
    if cutoff is not None and L >= 2:
        brk = (N[:, 1:] - C[:, :-1]).norm(dim=-1) > cutoff   # (B, L-1): no bond i-1 -> i
        feats[:, 1:, 0:2][brk] = 0                            # phi_i
        feats[:, :-1, 2:4][brk] = 0                           # psi_{i-1}
    return feats


def _descriptor(coords, device, convention: str = DEFAULT_DIHEDRAL_CONVENTION,
                backend: str = "native", chain_break_cutoff="auto"):
    """Compute the 256-D descriptor for a batch of frames.

    coords: (B, L, 3, 3) float tensor, B frames of one protein, atoms N, CA, C.
    convention: dihedral sign convention, "iupac" (default) or "legacy_plus180"
        (for checkpoints trained before the code_fix §1 fix).
    chain_break_cutoff: C–N distance (Å) above which phi/psi across the gap are set
        to (0, 0); "auto" = 2.5 for iupac, None for legacy_plus180. Pass None for
        backbones rebuilt from Cα with ideal geometry (treated as connected).
    backend: "native" (default) uses `ensembits/geometry.py` (no ESM dependency);
        "esm" uses ESM3's structure utilities as the original code did (kept only to
        cross-check; requires the `esm` package and downloads ESM3-open weights).
        Both give bit-identical results (tests/test_geometry.py).
    Returns (B, L, 256) on CPU. Frames are processed independently, so the
    neighbour list of a residue can differ between frames (dynamical mode).

    Paper: f^p_r = [g^p_{r,n_1}, ..., g^p_{r,n_K}] in `appendix.tex`
    \\label{app:descriptor-esm3}.
    """
    coords = coords.to(device).float()
    B, L, _, _ = coords.shape
    # FIXED(code_fix P29, 2026-10-16): old: a protein with fewer residues than K = 16
    #   failed later with an unrelated reshape / top-k error. New: a clear ValueError.
    if L < K:
        raise ValueError(f"the descriptor needs at least K = {K} residues (one kNN slot "
                         f"per neighbour), got L = {L}")
    knn_k = K
    if backend == "native":
        from ensembits import geometry as G
        # Step 1 — per-residue backbone frames T_r (paper: "Per-residue local frame"):
        # Gram-Schmidt with origin CA, x axis along CA - C, xy plane through N.
        # FIXED(require_fix #6, 2026-10-15): appendix now states "x along CA − C".
        rot, trans, mask = G.backbone_frames(coords)
        # Step 2 — the K = 16 nearest residues by CA distance, sorted closest first.
        # The residue itself is at distance 0, so edges[..., 0] == r (the "self slot"
        # of the paper, n^p_{r,1} = r). No sequence separation filter.
        knn_edges = G.knn_edges(coords, mask, knn_k)                              # (B, L, K)
        # Step 3a — relative frames T_r^{-1} o T_j for every neighbour j (12 numbers).
        rel = G.relative_frames(rot, trans, knn_edges)                            # (B, L, K, 12)
        gather = G.gather_neighbours
    elif backend == "esm":
        # Original implementation (ESM3 utilities); kept for verification only.
        from esm.utils.structure.affine3d import Affine3D, build_affine3d_from_coordinates
        from esm.utils.misc import node_gather
        model, knn_k = _get_model(device)
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)
        attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)
        knn_edges, _ = model.find_knn_edges(coords, ~attention_mask, coord_mask=affine_mask,
                                            sequence_id=sequence_id, knn=knn_k)   # (B, L, K)
        aff_t = affine.tensor
        knn_aff_t = node_gather(aff_t, knn_edges)                                 # (B, L, K, 12)
        inv_query = Affine3D.from_tensor(aff_t).invert()
        neighbor_aff = Affine3D.from_tensor(knn_aff_t)
        inv_q_b = Affine3D.from_tensor(
            inv_query.tensor.unsqueeze(2).expand(B, L, knn_k, T_AFF).contiguous())
        rel = inv_q_b.compose(neighbor_aff).tensor                               # (B, L, K, 12)
        gather = node_gather
    else:
        raise ValueError(f"backend must be 'native' or 'esm', got {backend!r}")

    # Step 3b — each neighbour's own phi/psi, gathered along the same kNN edges
    # (the self slot therefore carries r's own phi/psi).
    dih = _phi_psi_sincos(coords, convention, chain_break_cutoff)            # (B, L, 4)
    knn_dih = gather(dih, knn_edges)                                         # (B, L, K, 4)

    # Step 4 — concatenate the 16 entries of 16 numbers each.
    per_nb = torch.cat([rel, knn_dih], dim=-1)                               # (B, L, K, 16)
    return per_nb.reshape(B, L, knn_k * PER_NB).cpu()                        # (B, L, 256)


# Public aliases with readable names (the underscored names are kept because the
# original downstream scripts import them).
compute_descriptor = _descriptor
phi_psi_sincos = _phi_psi_sincos
