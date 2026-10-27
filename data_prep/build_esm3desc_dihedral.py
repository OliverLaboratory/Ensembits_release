"""Build the 256-D descriptor cache of the old mdCATH-div + MISATO corpus (and hold its constants).

What it does
------------
Computes the 256-D ESM3-style relative-frame + dihedral descriptor for every
chain of the combined mdCATH-div + MISATO training corpus of the earlier
flagship model, in the row order of that corpus's `layout.json`, so the
existing splits (splits_amino2M.json, splits_collapse_test.json, ...) apply
unchanged. Output is a flat (N_residues, P, 256) float32 array.

Per residue, per frame:
    neighbor j (j=0..15):  [ 12-D relative Affine3D | 4-D neighbor's phi/psi sincos ]
    descriptor = concat over 16 neighbors  ->  16 * (12 + 4) = 256-D

The 192-D variant (esm3desc) has only the 16 x 12 frame part. The dihedrals
are intrinsic (SE(3)-invariant) scalars, gathered along the same kNN edges as
the frames.

Coordinates come from the real-backbone caches (`bb_10`, first P = 10 frames).
For an mdCATH-div chain without a usable `mdcath_real_bb` file, N and C are
rebuilt from the CA trace (`ca_10` in MDCATH_SRC) with ideal peptide geometry so
the cache stays row-aligned with the layout, without a flag (recorded only in
the sidecar's IDEAL_PIDS). This silent fallback is LEGACY ONLY: it reproduces
the earlier flagship corpus, built when 8 chains were missing from
`mdcath_real_bb`. Since data_prep/build_mdcath_real_bb.py (2026-10-17) all
2,442 representatives have a real-backbone file, so the fallback is not used
on current data; every other builder and consumer refuses ideal geometry
unless `--allow-ideal` (build_esm3desc_dihedral_perdataset.py, token_anova.py).

This file also defines the corpus constants (MDCATH_REAL_BB, MISATO_REAL_BB,
MDCATH_SRC, P, K, DESC_DIM, ...) that `build_esm3desc_dihedral_perdataset.py`
and some downstream scripts import.

Paper
-----
  * `appendix.tex`, \\subsection{ESM3-style relative-frame descriptor}
    (\\label{app:descriptor-esm3}): the 256-D descriptor itself (implemented in
    `ensembits/descriptor.py`).
  * `appendix.tex`, \\subsection{Conventions}: N/C reconstructed from CA with
    ideal peptide geometry when absent (the mdCATH fallback in `_coords_for`;
    legacy only, see above).
  * The corpus built by `main()` (mdCATH-div + MISATO) is the earlier flagship's
    training corpus and is NOT the one the paper describes in
    \\subsection{Dataset} (\\label{app:dataset}); see
    `build_bioemu_misato_descriptor.py` for that.

Origin
------
Copied from `submission_exp/src/build_esm3desc_dihedral.py`. Changed:
  * the four descriptor functions `_get_model`, `_dihedral`, `_phi_psi_sincos`
    and `_descriptor` were REMOVED from this file and are now imported from
    `ensembits/descriptor.py`, which holds the same code verbatim (with
    comments). `from build_esm3desc_dihedral import _descriptor` therefore
    still works and returns the same function;
  * the `ROOT` line now comes from `ensembits.paths.REPO_ROOT` (same directory
    as before) and `sys.path` holds this codebase instead of the research repo;
  * `from ensembits.data.descriptors_3di import _reconstruct_n_c_ideal`
    (research-repo `Ensembits/` package, added to sys.path inline) became
    `from ensembits.ideal_backbone import _reconstruct_n_c_ideal`, and the
    inline `sys.path.insert(0, ROOT / "Ensembits")` was dropped (it would shadow
    this codebase's `ensembits` package);
  * comments and this docstring.
Everything else (constants, `_coords_for`, `main`) is unchanged, except the
provenance sidecar written by `main` (code_fix P13, 2026-10-16).

Used by the reported model: partly. `main()` builds the cache of the earlier
mdCATH-div + MISATO flagship, not the reported model's corpus. The constants
(MDCATH_REAL_BB, MISATO_REAL_BB, MDCATH_SRC, P, K, DESC_DIM) are used by the
per-dataset eval caches (`build_esm3desc_dihedral_perdataset.py`), including
the mdCATH-div RMSF probe cache evaluated with the reported model.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/combined_mdcath_misato_train_kabsch/cached_desc/aminoaseed_K16_P10.layout.json
          data/cached_descriptors/mdcath_real_bb/<pid>.npz   (bb_10)
          data/cached_descriptors/misato_real_bb_kabsch/<pid>.npz   (bb_10)
          data/external/mdcath_div/<pid>.npz (ca_10, fallback; ENSEMBITS_MDCATH_DIV_SRC)
  writes  output/combined_mdcath_misato_train_kabsch/cached_desc/esm3desc_dihedral[_iupac]_K16_P10.npy
          (or --out), and <output>.provenance.json (code_fix P13, since 2026-10-16)

How to run
----------
GPU recommended:
    python data_prep/build_esm3desc_dihedral.py --device cuda:0 [--limit N] [--out PATH]
"""
from __future__ import annotations
import os
# Limit CPU threads used by numpy/torch.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo

# The descriptor functions formerly defined here (see Origin above).
# FIXED(code_fix §1, 2026-10-15): `_dihedral` now defaults to the IUPAC sign;
# caches written before the fix hold (−sin φ, −cos φ, −sin ψ, −cos ψ)
# (convention "legacy_plus180", still selectable for old checkpoints).
from ensembits.descriptor import _get_model, _dihedral, _phi_psi_sincos, _descriptor
from ensembits.descriptor import resolve_chain_break_cutoff   # noqa: E402
from ensembits.data_versions import write_provenance         # noqa: E402

# Descriptor sizes (identical to the values in ensembits/descriptor.py).
# Paper: appendix.tex app:descriptor-esm3 — K = 16 entries x (12 + 4) = 256 = D_f.
K = 16                      # kNN entries per residue (the residue itself + 15 neighbours)
P = 10                      # frames per chain (paper: P_max = 10)
T_AFF = 12                  # relative frame: 3x3 rotation + translation
PER_NB = T_AFF + 4          # 12 frame + 4 dihedral sincos
DESC_DIM = K * PER_NB        # 256

# Corpus of the earlier mdCATH-div + MISATO flagship (not the reported model).
CACHE = ROOT / "output" / "combined_mdcath_misato_train_kabsch" / "cached_desc"
LAYOUT = CACHE / "aminoaseed_K16_P10.layout.json"     # row order of the corpus: [{pid, corpus, L}, ...]
# Real-backbone caches (bb_10 = (10, L, 4, 3) N/CA/C/O), shared with other builders.
MDCATH_REAL_BB = ROOT / "data" / "cached_descriptors" / "mdcath_real_bb"
# code_fix §4 / T1.1, T1.6 (2026-10-15): after the MISATO rebuild only the Kabsch
# cache has bb_10 (the downstream cache misato_real_bb holds bb_8). The reported
# model's corpus is built by build_bioemu_misato_descriptor.py, not here.
MISATO_REAL_BB = ROOT / "data" / "cached_descriptors" / "misato_real_bb_kabsch"
from ensembits.paths import MDCATH_DIV_SRC as MDCATH_SRC   # data/external/mdcath_div (ca_10 fallback)
# (absolute path on the original cluster, kept as is: the mdCATH-div source
#  npz files with ca_10 and _manifest.json)
OUT = CACHE / "esm3desc_dihedral_K16_P10.npy"

# Leftover module global from the original file. The descriptor's model cache
# now lives in ensembits.descriptor._MODEL; this one is not used.
_MODEL = None


# pids whose N/C were rebuilt with ideal geometry (code_fix §6).
IDEAL_PIDS: set = set()


def _coords_for(row):
    """Return (P, L, 3, 3) N/CA/C coordinates for one layout row, or None."""
    is_mdcath = row["corpus"].startswith("mdcath")   # 'mdcath' or 'mdcath_MISSING'
    base = MDCATH_REAL_BB if is_mdcath else MISATO_REAL_BB
    p = base / f"{row['pid']}.npz"
    # Preferred: real N/CA/C from the first P frames of bb_10.
    if p.exists():
        d = np.load(p, allow_pickle=False)
        if "bb_10" in d.files:
            bb = np.asarray(d["bb_10"], dtype=np.float32)    # (>=P, L, 4, 3)
            if bb.shape[0] >= P:
                bb = bb[:P]
                return np.stack([bb[:, :, 0, :], bb[:, :, 1, :], bb[:, :, 2, :]], axis=2)
    # mdcath fallback: ca_10 from MDCATH_SRC + ideal-geometry N/C reconstruction
    # (matches the original esm3desc build's _process_pid_mdcath fallback, so the
    #  8 chains then missing from mdcath_real_bb stayed in the cache and alignment
    #  held). LEGACY ONLY (2026-10-17): silent, recorded only in IDEAL_PIDS; since
    #  data_prep/build_mdcath_real_bb.py every representative has a real file, and
    #  build_esm3desc_dihedral_perdataset.py needs --allow-ideal for this fallback.
    # Paper: appendix.tex \subsection{Conventions} — N/C reconstructed from
    # consecutive Cα positions with ideal peptide geometry.
    if is_mdcath:
        src = MDCATH_SRC / f"{row['pid']}.npz"
        if not src.exists():
            return None
        ca = np.asarray(np.load(src, allow_pickle=False)["ca_10"], dtype=np.float32)
        if ca.shape[0] < P:
            return None
        ca = ca[:P]
        from ensembits.ideal_backbone import _reconstruct_n_c_ideal
        n_list, c_list = [], []
        for fp in range(P):
            n_, c_ = _reconstruct_n_c_ideal(ca[fp])
            n_list.append(n_); c_list.append(c_)
        n_all = np.stack(n_list, 0); c_all = np.stack(c_list, 0)
        IDEAL_PIDS.add(row['pid'])   # code_fix §6: no chain-break detection on ideal geometry
        return np.stack([n_all, ca, c_all], axis=2)          # (P, L, 3, 3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    # code_fix §1: dihedral convention ("legacy_plus180" reproduces the old cache).
    ap.add_argument("--dihedral-convention", default="iupac",
                    choices=("iupac", "legacy_plus180"))
    args = ap.parse_args()
    rows = json.loads(LAYOUT.read_text())["rows"]
    if args.limit:
        rows = rows[:args.limit]
    # Default output name follows the dihedral convention so an "iupac" build never
    # overwrites the legacy cache (code_fix §1).
    if args.out == str(OUT):
        from ensembits.descriptor import with_convention
        args.out = str(OUT.with_name(with_convention(OUT.name, args.dihedral_convention)))
    print(f"building {Path(args.out).name}: {len(rows)} chains, dim={DESC_DIM}, dev={args.device}", flush=True)
    # One chain at a time: coordinates -> (P, L, 256) descriptor -> (L, P, 256)
    # rows appended in layout order. Chains whose length does not match the
    # layout are dropped (and would break row alignment, hence the log line).
    parts, dropped, t0 = [], 0, time.time()
    dropped_pids = []          # code_fix P13: recorded in the sidecar
    for i, row in enumerate(rows):
        coords = _coords_for(row)
        if coords is None or coords.shape[1] != row["L"]:
            dropped += 1
            dropped_pids.append(row["pid"])
            print(f"  DROP {row['pid']} (corpus={row['corpus']}, "
                  f"got L={None if coords is None else coords.shape[1]} vs layout {row['L']})", flush=True)
            continue
        # Paper: appendix.tex app:descriptor-esm3 — f^p_r in R^256 per residue and frame.
        # code_fix §1: φ/ψ in --dihedral-convention; the output name records it.
        with torch.no_grad():
            # code_fix §6: chain breaks detected on real backbones only.
            desc = _descriptor(torch.from_numpy(coords), args.device,
                               convention=args.dihedral_convention,
                               chain_break_cutoff=None if row["pid"] in IDEAL_PIDS else "auto").numpy()  # (P, L, 256)
        parts.append(desc.transpose(1, 0, 2))            # (L, P, 256)
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.0f}s ETA={ (len(rows)-i-1)/max(rate,1e-6):.0f}s "
                  f"dropped={dropped}", flush=True)
    # Flat (N_residues, P, 256) float32 cache, one .npy.
    flat = np.concatenate(parts, axis=0).astype(np.float32)
    print(f"final {flat.shape} (dropped {dropped})", flush=True)
    np.save(args.out, flat)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e9:.1f} GB)", flush=True)
    # FIXED(code_fix P13, 2026-10-16): provenance sidecar next to the cache (old: none).
    side = write_provenance(args.out, {
        "builder": "data_prep/build_esm3desc_dihedral.py", "output": str(args.out),
        "dihedral_convention": args.dihedral_convention,
        "chain_break_cutoff": resolve_chain_break_cutoff("auto", args.dihedral_convention),
        "chain_break_cutoff_ideal_pids": None,
        "IDEAL_PIDS": sorted(IDEAL_PIDS),
        "n_rows": int(flat.shape[0]), "n_pids": len(rows) - dropped,
        "dropped_pids": dropped_pids, "limit": args.limit,
        "P": P, "K": K, "desc_dim": DESC_DIM,
        "source_files": {"layout": str(LAYOUT), "mdcath_real_bb": str(MDCATH_REAL_BB),
                         "misato_real_bb": str(MISATO_REAL_BB),
                         "mdcath_div_src": str(MDCATH_SRC)},
        "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"wrote {side.name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
