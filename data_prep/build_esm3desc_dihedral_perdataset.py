"""Build a per-dataset 256-D descriptor cache for evaluation (mdCATH-div or MISATO).

What it does
------------
Same descriptor as `build_esm3desc_dihedral.py` (16 x [12 rel-Affine3D | 4 φ/ψ
sincos] = 256-D), but for a single dataset's pid list, in the same order the
original `build_esm3desc_realbb.py` used, so the output cache is row-aligned
to the existing 192-D eval cache at the same path:

    --dataset mdcath  -> output/phase_b/cached_desc/esm3desc_dihedral_k16_P10.npy
                         (2,442 mdCATH-div representatives, manifest order)
    --dataset misato  -> output/misato_train/cached_desc/esm3desc_dihedral_k16_P10.npy
                         (all misato_real_bb_kabsch pids, sorted)
    --dataset bioemu  -> output/phase_b/cached_desc/bioemu/esm3desc_dihedral_iupac_k16_P10.npy
                         (the 1,090 systems of bioemu_cath_real_bb/layout_rows.json, in
                         that order; real N/CA/C of bb_10; 2026-10-16, phase 2)

(Lowercase 'k' matches `probe_rmsf_eval.py`'s expected `{mode}_k{k}_P{P_trained}.npy`.)

For each pid it takes the first P = 10 frames of the real backbone `bb_10`
(N/CA/C), computes the descriptor per frame and appends (L, P, 256) rows.
For mdCATH-div chains without a real-backbone file it rebuilt N and C from
the CA trace with ideal peptide geometry; since 2026-10-17 every domain has a real
file (data_prep/build_mdcath_real_bb.py) and that fallback needs --allow-ideal. Pids without coordinates are
skipped (and logged); for --dataset mdcath a skipped pid aborts the build
(code_fix P13), since it would shift every later row against the RMSF labels.
Since 2026-10-16 the mdcath output must have exactly as many rows as the
manifest layout (308,293; and the all-frames label when built; G2 2026-10-17: was
rmsf_div.npy), an existing output is refused unless --overwrite, the write is atomic,
and every output gets a
provenance sidecar `<output>.provenance.json` (dihedral convention, chain-break
cutoff, ideal-geometry pids, row / pid counts, source files).

Paper
-----
  * `appendix.tex`, \\subsection{ESM3-style relative-frame descriptor}
    (\\label{app:descriptor-esm3}): the descriptor.
  * `appendix.tex`, \\subsection{Dataset}, \\paragraph{mdCATH-div (evaluation).}
    (2,442 ensembles, one representative per CATH H superfamily) and
    `2_results.tex` \\label{result:rmsf}: the mdcath cache is the input of the
    flexibility (RMSF) probe.
  * `appendix.tex`, \\subsection{Conventions}: ideal-geometry N/C fallback.

Origin
------
Copied from `submission_exp/src/build_esm3desc_dihedral_perdataset.py`.
Changed:
  * `ROOT` now comes from `ensembits.paths.REPO_ROOT` (same directory as before);
  * `from build_esm3desc_dihedral import _descriptor, MDCATH_REAL_BB, ...` was
    split into `from ensembits.descriptor import _descriptor` and
    `from build_esm3desc_dihedral import MDCATH_REAL_BB, ...` (the sibling
    script in `data_prep/`, whose `_descriptor` is the same function);
  * `from ensembits.data.descriptors_3di import _reconstruct_n_c_ideal` (with
    an inline `sys.path.insert(0, ROOT / "Ensembits")`) became
    `from ensembits.ideal_backbone import _reconstruct_n_c_ideal`;
  * the research repo and `submission_exp/src` are no longer put on sys.path
    (nothing else is imported from them);
  * comments and this docstring.
Logic unchanged, except the approved fixes of 2026-10-16 (code_fix P13: abort on
a dropped mdcath pid, row-count check, provenance sidecar) and the new `--dataset
bioemu` (phase 2: BioEmu-CATH RMSF probe; iupac default, chain-break cutoff "auto" =
2.5 Å as in the corpus; any dropped system aborts; the row count must equal the sum
of the layout L and, if built, the BioEmu label).

Used by the reported model: yes for `--dataset mdcath` (the mdCATH-div RMSF
probe reads output/phase_b/cached_desc/esm3desc_dihedral_k16_P10.npy for any
256-D checkpoint, including the reported one). No reported-model script was found that reads
the misato cache (the MISATO downstream scripts compute descriptors from
misato_real_bb on the fly).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/mdcath_real_bb/<pid>.npz (bb_10)
          data/cached_descriptors/misato_real_bb_kabsch/<pid>.npz (bb_10)
          data/external/mdcath_div/_manifest.json and <pid>.npz
          (ca_10 fallback; absolute path)
          output/phase_b/cached_desc/rmsf_div_allframes.npy   (mdcath: row count check, if built)
  writes  output/phase_b/cached_desc/esm3desc_dihedral[_iupac]_k16_P10.npy        (mdcath)
          output/misato_train/cached_desc/esm3desc_dihedral[_iupac]_k16_P10.npy   (misato)
          output/phase_b/cached_desc/bioemu/esm3desc_dihedral[_iupac]_k16_P10.npy (bioemu;
              reads data/cached_descriptors/bioemu_cath_real_bb/{layout_rows.json,<system>.npz})
          <each output>.provenance.json                                        (sidecar)

How to run
----------
    python data_prep/build_esm3desc_dihedral_perdataset.py --dataset mdcath --device cuda:0
    python data_prep/build_esm3desc_dihedral_perdataset.py --dataset misato --device cuda:0
    python data_prep/build_esm3desc_dihedral_perdataset.py --dataset bioemu --device cuda:0
"""
from __future__ import annotations
import os
# Limit CPU threads used by numpy/torch.
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in data_prep/
# Reuse the descriptor + helpers from the combined builder.
# FIXED(code_fix §1, 2026-10-15): `_descriptor` takes the dihedral convention
# (--dihedral-convention, default iupac; legacy_plus180 for old checkpoints).
from ensembits.descriptor import _descriptor
from build_esm3desc_dihedral import MDCATH_REAL_BB, MISATO_REAL_BB, MDCATH_SRC, P, K, DESC_DIM
from ensembits.descriptor import resolve_chain_break_cutoff   # noqa: E402
from ensembits.data_versions import write_provenance         # noqa: E402

# Per-residue mdCATH-div RMSF labels; the mdcath descriptor cache must be row-aligned
# with them (code_fix P13).
RMSF_DIV = ROOT / "output" / "phase_b" / "cached_desc" / "rmsf_div.npy"   # legacy label (old check)
# FIXED(preretrain_audit_r2 G2, 2026-10-17): the mdcath row check uses the manifest layout
#   (sum of the ca_5 lengths) and, when built, the all-frames label (was the legacy
#   rmsf_div.npy); both paths are recorded in the sidecar.
RMSF_DIV_ALLFRAMES = ROOT / "output" / "phase_b" / "cached_desc" / "rmsf_div_allframes.npy"
# BioEmu-CATH (phase 2, 2026-10-16): system-keyed real backbones, layout_rows.json order;
# the cache goes next to the BioEmu label / splits (rmsf_common.BIOEMU_DIR).
BIOEMU_REAL_BB = ROOT / "data" / "cached_descriptors" / "bioemu_cath_real_bb"
BIOEMU_OUT_DIR = ROOT / "output" / "phase_b" / "cached_desc" / "bioemu"
RMSF_BIOEMU = BIOEMU_OUT_DIR / "rmsf_bioemu_allframes.npy"


# pids whose N/C were rebuilt with ideal geometry (code_fix §6).
IDEAL_PIDS: set = set()


ALLOW_IDEAL = False   # --allow-ideal (main)


def _coords_mdcath(pid):
    """(P, L, 3, 3) N/CA/C for one mdCATH-div pid, or None.

    FIXED(mdCATH re-extraction, 2026-10-17): old = ca_10 with ideal-geometry N/C whenever
    no usable real file existed (the 8 domains without a file). New = every domain has a
    real file (data_prep/build_mdcath_real_bb.py): a real file that exists but cannot be
    used gives None (the build aborts), and the ideal fallback for a domain without a
    file needs --allow-ideal (recorded in IDEAL_PIDS of the sidecar; the probes refuse a
    residue-set protein listed there).
    """
    # Preferred: real backbone from mdcath_real_bb (first P frames of bb_10).
    rb = MDCATH_REAL_BB / f"{pid}.npz"
    if rb.exists():
        d = np.load(rb, allow_pickle=False)
        if "bb_10" in d.files:
            bb = np.asarray(d["bb_10"], dtype=np.float32)
            if bb.shape[0] >= P:
                bb = bb[:P]
                return np.stack([bb[:, :, 0, :], bb[:, :, 1, :], bb[:, :, 2, :]], axis=2)
        return None
    if not ALLOW_IDEAL:
        return None
    # MDCATH_SRC ca_10 + ideal-N/C fallback
    # Paper: appendix.tex \subsection{Conventions} — N/C reconstructed from Cα
    # with ideal peptide geometry.
    src = MDCATH_SRC / f"{pid}.npz"
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
    IDEAL_PIDS.add(pid)   # code_fix §6: no chain-break detection on ideal geometry
    return np.stack([np.stack(n_list, 0), ca, np.stack(c_list, 0)], axis=2)


def _coords_misato(pid):
    """(P, L, 3, 3) N/CA/C for one MISATO pid from misato_real_bb_kabsch bb_10, or None."""
    rb = MISATO_REAL_BB / f"{pid}.npz"
    if not rb.exists():
        return None
    d = np.load(rb, allow_pickle=False)
    if "bb_10" not in d.files:
        return None
    bb = np.asarray(d["bb_10"], dtype=np.float32)
    if bb.shape[0] < P:
        return None
    bb = bb[:P]
    return np.stack([bb[:, :, 0, :], bb[:, :, 1, :], bb[:, :, 2, :]], axis=2)


def _coords_bioemu(system):
    """(P, L, 3, 3) N/CA/C for one BioEmu-CATH system from bioemu_cath_real_bb bb_10, or None."""
    rb = BIOEMU_REAL_BB / f"{system}.npz"
    if not rb.exists():
        return None
    with np.load(rb, allow_pickle=False) as d:
        bb = np.asarray(d["bb_10"], dtype=np.float32)
    if bb.shape[0] < P:
        return None
    bb = bb[:P]
    return np.stack([bb[:, :, 0, :], bb[:, :, 1, :], bb[:, :, 2, :]], axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("mdcath", "misato", "bioemu"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    # code_fix §1: dihedral convention; "iupac" caches get a distinct file name
    # (esm3desc_dihedral_iupac_k16_P10.npy) so the legacy cache is never overwritten.
    # G2 (2026-10-17): an existing cache is refused unless --overwrite (it may be in use).
    ap.add_argument("--allow-ideal", action="store_true",
                    help="mdcath: ideal-geometry N/C for a domain without a real-backbone file "
                         "(none after data_prep/build_mdcath_real_bb.py; default: abort)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output (default: exit 1 if it exists)")
    ap.add_argument("--dihedral-convention", default="iupac",
                    choices=("iupac", "legacy_plus180"))
    args = ap.parse_args()
    global ALLOW_IDEAL
    ALLOW_IDEAL = args.allow_ideal

    # Pid order and output path per dataset. The order must match the existing
    # 192-D eval cache at the same location (row alignment with its layout).
    # FIXED(code_fix E0, 2026-10-20): eval caches regenerated with iupac descriptor.
    if args.dataset == "mdcath":
        # mdCATH-div: the 2,442 H-superfamily representatives, manifest order.
        # Paper: appendix.tex \paragraph{mdCATH-div (evaluation).}
        manifest = json.loads((MDCATH_SRC / "_manifest.json").read_text())
        pids = list(manifest["representatives"].keys())
        out = ROOT / "output" / "phase_b" / "cached_desc" / "esm3desc_dihedral_k16_P10.npy"
        get_coords = _coords_mdcath
    elif args.dataset == "bioemu":
        # BioEmu-CATH: the systems of layout_rows.json, in that order (the row order of the
        # BioEmu RMSF label and splits).
        pids = [r["system"] for r in json.loads((BIOEMU_REAL_BB / "layout_rows.json").read_text())]
        out = BIOEMU_OUT_DIR / "esm3desc_dihedral_k16_P10.npy"
        get_coords = _coords_bioemu
    else:
        # MISATO: every complex in MISATO_REAL_BB (Kabsch cache), sorted by pid.
        pids = sorted(p.stem for p in MISATO_REAL_BB.glob("*.npz"))
        out = ROOT / "output" / "misato_train" / "cached_desc" / "esm3desc_dihedral_k16_P10.npy"
        get_coords = _coords_misato
    if args.limit:
        pids = pids[: args.limit]
    from ensembits.descriptor import with_convention
    out = out.with_name(with_convention(out.name, args.dihedral_convention))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to rebuild (never while probes read it)", flush=True)
        return 1
    print(f"building {out.name}: {len(pids)} pids, dim={DESC_DIM}, dev={args.device}", flush=True)

    # One pid at a time: (P, L, 256) descriptor -> (L, P, 256) rows, appended in
    # pid order. A dropped pid shifts all later rows (only the first 8 are logged).
    parts, dropped, t0 = [], 0, time.time()
    dropped_pids = []          # code_fix P13: every dropped pid (abort for mdcath)
    for i, pid in enumerate(pids):
        c = get_coords(pid)
        if c is None:
            dropped += 1
            dropped_pids.append(pid)
            if dropped <= 8: print(f"  DROP {pid}", flush=True)
            continue
        # code_fix §1: φ/ψ in --dihedral-convention; the output name records it.
        # code_fix §6: chain breaks detected on real backbones only; ideal-geometry
        # backbones (no real peptide bond) are treated as connected.
        with torch.no_grad():
            d = _descriptor(torch.from_numpy(c), args.device,
                            convention=args.dihedral_convention,
                            chain_break_cutoff=None if pid in IDEAL_PIDS else "auto").numpy()  # (P, L, 256)
        parts.append(d.transpose(1, 0, 2))  # (L, P, 256)
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{i+1}/{len(pids)}] {time.time()-t0:.0f}s ETA={(len(pids)-i-1)/max(rate,1e-6):.0f}s dropped={dropped}",
                  flush=True)
    # Flat (N_residues, P, 256) float32 cache.
    flat = np.concatenate(parts, axis=0).astype(np.float32)
    print(f"final {flat.shape} (dropped {dropped})", flush=True)
    # FIXED(code_fix P13, 2026-10-16): old: dropped pids were only logged and the cache
    #   was written anyway (rows shifted against rmsf_div.npy). New: for mdcath, any
    #   dropped pid aborts and the row count must equal len(rmsf_div.npy) (308,293);
    #   nothing is written on failure.
    if args.dataset == "mdcath":
        if dropped_pids:
            print(f"ABORT: {len(dropped_pids)} mdCATH-div pid(s) without coordinates "
                  f"(rows would shift against the manifest layout and {RMSF_DIV_ALLFRAMES.name}): "
                  f"{dropped_pids}", flush=True)   # K5 (2026-10-17): was rmsf_div.npy
            return 1
        if args.limit:
            print(f"--limit {args.limit}: row-count check skipped", flush=True)
        else:
            sys.path.insert(0, str(CODE_ROOT / "downstream"))
            from rmsf_common import mdcath_div_layout
            n_exp = mdcath_div_layout(MDCATH_SRC)[2]
            if flat.shape[0] != n_exp:
                print(f"ABORT: {flat.shape[0]} rows != {n_exp} (manifest layout, ca_5 lengths)",
                      flush=True)
                return 1
            if RMSF_DIV_ALLFRAMES.exists() and \
                    int(np.load(RMSF_DIV_ALLFRAMES, mmap_mode="r").shape[0]) != flat.shape[0]:
                print(f"ABORT: {flat.shape[0]} rows != {RMSF_DIV_ALLFRAMES.name}", flush=True)
                return 1
    if args.dataset == "bioemu":
        # Same guard as mdcath: a dropped system would shift every later row.
        layout = json.loads((BIOEMU_REAL_BB / "layout_rows.json").read_text())
        n_exp = sum(int(r["L"]) for r in layout[: len(pids)])
        if dropped_pids or flat.shape[0] != n_exp:
            print(f"ABORT: dropped {dropped_pids[:8]} / {flat.shape[0]} rows != {n_exp} "
                  f"(layout_rows.json L)", flush=True)
            return 1
        if RMSF_BIOEMU.exists() and not args.limit and \
                int(np.load(RMSF_BIOEMU, mmap_mode="r").shape[0]) != flat.shape[0]:
            print(f"ABORT: {flat.shape[0]} rows != {RMSF_BIOEMU.name}", flush=True)
            return 1
    # G2 (2026-10-17): atomic write (temporary name, then rename)
    tmp = out.with_name(f".{out.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, flat)
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size/1e9:.1f} GB)", flush=True)
    # FIXED(code_fix P13, 2026-10-16): provenance sidecar (old: none), so the probe can
    #   check the convention and the row count.
    sources = ({"mdcath_real_bb": str(MDCATH_REAL_BB), "mdcath_div_src": str(MDCATH_SRC),
                "manifest": str(MDCATH_SRC / "_manifest.json"),
                "row_check": "manifest layout (ca_5 lengths)",
                "rmsf_labels": str(RMSF_DIV_ALLFRAMES) if RMSF_DIV_ALLFRAMES.exists() else None}
               if args.dataset == "mdcath" else
               {"bioemu_real_bb": str(BIOEMU_REAL_BB),
                "layout": str(BIOEMU_REAL_BB / "layout_rows.json")} if args.dataset == "bioemu"
               else {"misato_real_bb": str(MISATO_REAL_BB)})
    side = write_provenance(out, {
        "builder": "data_prep/build_esm3desc_dihedral_perdataset.py",
        "dataset": args.dataset, "output": str(out),
        "dihedral_convention": args.dihedral_convention,
        "chain_break_cutoff": resolve_chain_break_cutoff("auto", args.dihedral_convention),
        "chain_break_cutoff_ideal_pids": None,
        "IDEAL_PIDS": sorted(IDEAL_PIDS),
        "n_rows": int(flat.shape[0]), "n_pids": len(pids) - dropped,
        "dropped_pids": dropped_pids, "limit": args.limit,
        "P": P, "K": K, "desc_dim": DESC_DIM, "source_files": sources,
        "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"wrote {side.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
