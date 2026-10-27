"""Tokenize one protein conformational ensemble with an Ensembits checkpoint.

What it does
------------
Input:  backbone coordinates of P conformations of ONE chain, as N/CA/C atoms.
Output: a per-residue token sequence (L,) at codebook level 0, plus the full
        3-level RVQ tokens (L, 3).

Accepts either:
  --coords  a .npy of shape (P, L, 3, 3)  [P frames, L residues, N/CA/C, xyz]
            (or (P, L, 4, 3) with N/CA/C/O — the O is ignored), OR
  --pdb     a multi-MODEL PDB file (needs mdtraj); each MODEL = one frame.

Steps: compute the 256-D descriptor of every frame, apply the checkpoint's
descriptor normalisation (identity for the 256-D models), reshape to
(L, P, 256) and run the encoder + residual quantizer; level 0 is the 2048-way
primary token, levels 1-2 refine it (128 codes each).

Coordinates must be in Angstrom (the PDB path converts mdtraj's nm).

Default model = ckpt/bioemu_cath_misato_lam10 in this codebase (`ensembits.paths.CKPT_DIR`),
the model reported in the paper (replaced by the retrained checkpoint, code_fix E0).
The original script defaulted to the earlier mdCATH-div + MISATO tokenizer
(output/dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT); pass --model-dir for others.

Paper
-----
  * `1_method.tex` (tokenizer) and `appendix.tex` \\label{app:model-impl}: the
    encoder and the RVQ codebooks [2048, 128, 128].
  * `appendix.tex`, \\subsection{ESM3-style relative-frame descriptor}
    (\\label{app:descriptor-esm3}): the 256-D input descriptor.

Origin
------
Copied from `submission_exp/src/quantize_ensemble.py`. Changed:
  * `ROOT` now comes from `ensembits.paths.REPO_ROOT` (same directory as before,
    so DEFAULT_MODEL is unchanged);
  * `from build_esm3desc_dihedral import _descriptor` became
    `from ensembits.descriptor import _descriptor` (same function);
  * `from ensembits_featurize import load_model` became
    `from ensembits.featurize import load_model`;
  * the research-repo folders (repo root, submission_exp/src, scripts/baselines)
    are no longer put on sys.path (nothing is imported from them);
  * comments and this docstring.
Logic, defaults and CLI flags unchanged.
Fixes 2026-10-16 (FIXED markers): protein-only backbone atom selection and a clear
error below 16 residues (code_fix P29).

Used by the reported model: yes, as the inference entry point (with
--model-dir ckpt/bioemu_cath_misato_lam10).

Inputs / outputs
----------------
  reads   --coords .npy or --pdb file; <model-dir>/config.json, stats.npz, best.pt
  writes  --out .npz (default ./ensemble_tokens.npz) with
          tokens_L0 (L,) int64 and tokens_rvq (L, 3) int64

How to run
----------
IMPORTANT — if PyTorch uses pip CUDA wheels, put their libraries on the path or
torch may silently fall back to CPU (SP = site-packages/nvidia of your environment):

  SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
  export LD_LIBRARY_PATH="$(find $SP -name lib -type d | tr '\\n' ':')$LD_LIBRARY_PATH"
  python \\
    inference/quantize_ensemble.py \\
    --coords my_ensemble.npy --model-dir ckpt/bioemu_cath_misato_lam10 --out my_tokens.npz
"""
from __future__ import annotations
import os
# Limit CPU threads; silence TensorFlow logging pulled in by the ESM/ProToken stack.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse, sys
from pathlib import Path
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo

# FIXED(code_fix T2.6, 2026-10-15): the default model is the checkpoint the paper
# reports, `ensembits.paths.CKPT_DIR` (previously the old mdCATH+MISATO flagship
# `output/dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT`).
# FIXED(code_fix E0 / LAM, 2026-10-23): CKPT_DIR = ckpt/bioemu_cath_misato_lam10 (λ=10).
from ensembits.paths import CKPT_DIR                          # noqa: E402
DEFAULT_MODEL = CKPT_DIR


def load_coords(args) -> np.ndarray:
    """Return (P, L, 3, 3) N/CA/C float32."""
    # .npy input: keep the first three atoms (N, CA, C) of each residue.
    if args.coords:
        a = np.load(args.coords).astype(np.float32)
        if a.ndim != 4 or a.shape[2] < 3 or a.shape[3] != 3:
            raise SystemExit(f"--coords must be (P,L,>=3,3); got {a.shape}")
        return np.ascontiguousarray(a[:, :, :3, :])
    # Multi-MODEL PDB: every MODEL is one conformation. Atoms are selected by
    # name over the protein residues, so the file should hold a complete N/CA/C
    # backbone (the three counts must agree).
    if args.pdb:
        import mdtraj as md
        t = md.load(args.pdb)                                   # all MODELs as frames
        top = t.topology
        # FIXED(code_fix P29, 2026-10-16): old: "name CA" etc. over the whole topology, which
        #   also matched calcium ions (atom name CA) and ligand atoms. New: protein atoms only,
        #   and the N / CA / C counts must agree (one complete backbone per residue).
        sel = {at: top.select(f"protein and name {at}") for at in ("N", "CA", "C")}
        if not all(len(v) for v in sel.values()):
            raise SystemExit("PDB missing protein N/CA/C atoms")
        if len({len(v) for v in sel.values()}) != 1:
            raise SystemExit(f"PDB backbone incomplete: {len(sel['N'])} N, {len(sel['CA'])} CA, "
                             f"{len(sel['C'])} C protein atoms (every residue needs N, CA and C)")
        xyz = t.xyz * 10.0                                      # nm -> Angstrom
        coords = np.stack([xyz[:, sel["N"]], xyz[:, sel["CA"]], xyz[:, sel["C"]]], axis=2)
        return coords.astype(np.float32)                        # (P, L, 3, 3)
    raise SystemExit("pass --coords <npy> or --pdb <multi-model pdb>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", help="(P,L,3,3) N/CA/C npy")
    ap.add_argument("--pdb", help="multi-MODEL pdb (one MODEL per conformation)")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="ensemble_tokens.npz")
    args = ap.parse_args()

    # FIXED(code_fix §1, 2026-10-15): the descriptor is computed with the dihedral
    # convention the checkpoint was trained with (`ens.dihedral_convention`).
    from ensembits.descriptor import _descriptor          # 256-D frame+phi/psi
    from ensembits.featurize import load_model

    coords = load_coords(args)                                # (P, L, 3, 3)
    P, L = coords.shape[:2]
    # FIXED(code_fix P29, 2026-10-16): old: L < 16 crashed inside the descriptor (top-16
    #   neighbour search). New: a clear error.
    if L < 16:
        raise SystemExit(f"{L} residues: the descriptor needs at least 16 (it uses the 16 "
                         f"nearest residues of each residue)")
    print(f"[quantize] {P} conformations x {L} residues, model={Path(args.model_dir).name}",
          flush=True)

    # Load the tokenizer; only 256-D (dihedral descriptor) checkpoints are accepted.
    ens = load_model(args.model_dir, device=args.device)
    if ens.config.get("input_dim") != 256:
        raise SystemExit(f"model input_dim={ens.config.get('input_dim')} != 256 "
                         "(not a dihedral tokenizer)")

    with torch.no_grad():
        # Paper: appendix.tex app:descriptor-esm3 — f^p_r in R^256 per residue and frame.
        desc = _descriptor(torch.from_numpy(coords), args.device,
                           convention=ens.dihedral_convention,
                           chain_break_cutoff=ens.chain_break_cutoff)         # (P, L, 256) on cpu
        desc = desc.to(args.device)
        # config is --no-standardize -> mean 0 / std 1 (identity), but apply for safety
        mu = torch.as_tensor(ens.mean, dtype=desc.dtype, device=desc.device)
        sd = torch.as_tensor(ens.std, dtype=desc.dtype, device=desc.device)
        # Residues become the batch; the P frames form the set the encoder pools.
        desc = ((desc - mu) / sd).transpose(0, 1).contiguous()        # (L, P, 256)
        # Level-0 (primary) token and the tokens of all 3 RVQ levels.
        tok_L0 = ens.model.encode(desc).cpu().numpy().astype(np.int64)        # (L,)
        tok_all = np.stack([t.cpu().numpy() for t in ens.model.encode_all(desc)],
                            axis=-1).astype(np.int64)                          # (L, 3)

    np.savez(args.out, tokens_L0=tok_L0, tokens_rvq=tok_all)
    print(f"[quantize] wrote {args.out}: tokens_L0 {tok_L0.shape} (codebook 0..2047), "
          f"tokens_rvq {tok_all.shape}", flush=True)
    print(f"  first 20 L0 tokens: {tok_L0[:20].tolist()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
