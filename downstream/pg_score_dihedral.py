"""ProteinGym scoring for the 256-D dihedral-augmented ESM3 tokenizers.

What it does
------------
Thin wrapper over `pg_score.py`: reuses its entire scoring core (WT-vs-variant
token-distance scores a/b/c/d, ESM2 z-mix α-sweep, per-assay output schema,
multi-GPU threading) verbatim, and only swaps:

  * the structure cache  -> `esmfold_cache_atom14` (N/CA/C atom14), and
  * the descriptor       -> `build_esm3desc_dihedral._descriptor` (256-D:
                             16 × [12 rel-Affine3D frame ‖ 4 φ/ψ sincos]),
                             here `ensembits.descriptor._descriptor`,

so the output `per_assay_pg_<tag>.json` is directly comparable to the baseline
tokenizers already scored in `output/proteingym/per_assay_pg_*`.

The command line is `pg_score.main()`'s, unchanged and passed through as is:
`--model-dir`, `--gpus`, `--assays`, `--assay-list` (one assay id per line;
default = every folder of the atom14 cache, i.e. the paper's 96 assays),
`--out-tag`, `--force`. Per-variant scores go to `per_variant_pg_<out-tag>/`
and the scored assay list to `per_assay_pg_<out-tag>.assays_scored.json`
(both added in `pg_score.py` on 2026-10-15; scores unchanged).

Each ESMFold structure (WT or mutant) is one frame: its N/CA/C atoms give a
(1, L, 3, 3) input, the 256-D descriptor is computed, standardised with the
model's mean/std (identity for the reported model) and encoded at P = 1; all RVQ
levels are returned so `pg_score.evaluate_assay` can compute every score.

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool alphabet carry functional
    information?} \\label{sec:downstreams}, \\paragraph{Mutation Effect
    Prediction}: "fold the wild-type sequence and every unique mutant sequence
    with ESMFold and tokenize each predicted single-frame structure with our
    model at P=1"; Table \\label{tab:proteingym_ensemble} caption: "All
    tokenizers consume ESMFold-predicted atom14 backbones (real N/Cα/C/O/Cβ);
    no ideal-geometry reconstruction" (this model reads N/Cα/C only).
  * Descriptor: `appendix.tex`, \\subsection{ESM3-style relative-frame
    descriptor} \\label{app:descriptor-esm3}.
  * `appendix.tex`, \\subsection{Compute Requirements} \\label{sec:compute}:
    "ESMFold atom14 cache for ProteinGym (~400k variants)".

Origin
------
Copied from `submission_exp/src/pg_score_dihedral.py`. Changes: ROOT is now
`ensembits.paths.REPO_ROOT`; sys.path holds this codebase and this script's
folder (so `import pg_score` is the copy in `downstream/`) instead of
`submission_exp/src` and `scripts/baselines`; `from build_esm3desc_dihedral
import _descriptor` -> `from ensembits.descriptor import _descriptor`; comments
and this docstring. No logic changed here (the 2026-10-15 output-naming and
`--assay-list` changes live in `pg_score.py` and apply through `pg_score.main`).
2026-10-19 (benchmark_audit D9): `_tokenize_dihedral` takes the sequence as an
unused third argument; the D9 scoring fixes (float64, common variant mask, n_mut /
mutant in the per-variant file, provenance, fixed 96-assay list by default, blind
console) live in `pg_score.py` and apply through `pg_score.main`. alpha is chosen
afterwards by `pg_aggregate.py` (leave-one-UniProt-out CV).

Used by the reported model: yes. Step C of `_run_model_battery.py`; the
reported model's PG tag is `dih256_fapeH_bioemu_misato` (row "↳ BioEmu+MISATO
(dataset abl.)" of `rebuttal/results_bf65.md`).

Inputs / outputs
----------------
Reads:
  data/proteingym/esmfold_cache_atom14/<assay>/{manifest.json,<hash>.npy}  (L, 14, 3) atom14
  data/external/proteingym_zero_shot_scores/<assay>.csv     (external)
  <model-dir>/{config.json,best.pt,stats.npz}
  [--assay-list file]                                                        optional
Writes (REPO_ROOT): see `pg_score.py` (output/proteingym/per_assay_pg_<tag>{/,.json},
  output/proteingym/per_assay_pg_<tag>.assays_scored.json,
  output/proteingym/per_variant_pg_<tag>/<assay>.npz).

Dependencies left in the research repo: none in code. The atom14 cache is built
by the research repo's `submission_exp/src/pg_esmfold_cache.py`.

How to run
----------
Run from REPO_ROOT (relative --model-dir), single-frame structures, P=1:
    python \\
        camera_ready/ensembits_code/downstream/pg_score_dihedral.py \\
        --model-dir output/dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
        --out-tag dih256_fapeH_bioemu_misato --gpus 1,2 [--assay-list <file>]
Camera-ready (D9) run of one retrained variant on one GPU (default assay list =
downstream/pg_assays_wt_le200.txt; no Spearman printed unless --print-scores):
    CUDA_VISIBLE_DEVICES=2 python camera_ready/ensembits_code/downstream/pg_score_dihedral.py \\
        --model-dir output/ensembits_bm_lam0p25_rv --out-tag bm_lam0p25_rv --gpus 0
(original example: --model-dir output/dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT
 --out-tag dih256_fapeH, the flagship)
"""
from __future__ import annotations
import os
# Cap BLAS threads (one worker thread per GPU shares the CPU).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys
from pathlib import Path

import numpy as np
import torch

# Codebase root and this script's folder on sys.path (`import pg_score` resolves
# to downstream/pg_score.py); data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pg_score                                   # the scoring core we reuse
from ensembits.descriptor import _descriptor as _dihedral_descriptor

# Point the reused core at the atom14 structure cache (N/CA/C in slots 0,1,2).
pg_score.CACHE_DIR = ROOT / 'data' / 'proteingym' / 'esmfold_cache_atom14'


def _tokenize_dihedral(ens, a14, seq=None):
    """a14: (L, 14, 3) atom14 -> (L,) L0 tokens AND (L, n_levels) all-level
    tokens, treating the single ESMFold structure as P=1.

    `seq` (the structure's sequence) is unused; pg_score.evaluate_assay passes it
    to every tokenizer since 2026-10-19 (FIXED(benchmark_audit D9): the hook gained
    a third argument for ProToken in pg_score_baselines.py).

    Builds the 256-D dihedral descriptor from N/CA/C (slots 0/1/2), matching
    `_pretokenize_misato_dihedral.py` exactly (identity standardize under
    --no-standardize)."""
    a14 = np.asarray(a14, dtype=np.float32)
    if a14.ndim != 3 or a14.shape[1] < 3:
        raise ValueError(f'bad atom14 shape {a14.shape}')
    bb = a14[None, :, :3, :]                                   # (1, L, 3, 3) N/CA/C, P=1
    dev = getattr(ens, 'device', None) or next(ens.model.parameters()).device
    # Paper: appendix.tex app:descriptor-esm3 — 16 neighbours × [12-D relative
    # frame ‖ sin/cos φ, ψ] = 256-D per residue.
    # code_fix §1: descriptor in the dihedral convention the checkpoint was trained with.
    desc = _dihedral_descriptor(torch.from_numpy(bb), str(dev),
                                convention=ens.dihedral_convention,
                                chain_break_cutoff=ens.chain_break_cutoff)  # (1, L, 256) on cpu
    desc = desc.to(dev)
    mu = torch.as_tensor(ens.mean, dtype=desc.dtype, device=dev)
    std = torch.as_tensor(ens.std, dtype=desc.dtype, device=dev)
    desc = ((desc - mu) / std).transpose(0, 1).contiguous()    # (L, 1, 256)
    # Paper: 2_results.tex, Mutation Effect Prediction — "tokenize each predicted
    # single-frame structure with our model at P=1".
    tok_L0 = ens.model.encode(desc).cpu().numpy().astype(np.int64)
    if hasattr(ens.model, 'encode_all'):
        with torch.no_grad():
            tok_list = ens.model.encode_all(desc)
        tok_all = np.stack([t.cpu().numpy().astype(np.int64) for t in tok_list],
                           axis=-1)
    else:
        tok_all = tok_L0[:, None]
    return tok_L0, tok_all


# Swap the descriptor-specific function; evaluate_assay() / main() are agnostic.
pg_score.tokenize_ca = _tokenize_dihedral


if __name__ == '__main__':
    sys.exit(pg_score.main())
