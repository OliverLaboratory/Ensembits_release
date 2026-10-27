"""Build the MISATO level-1 token cache for one 256-D Ensembits checkpoint.

What it does
------------
For every MISATO complex that has a real-backbone cache
(`data/cached_descriptors/misato_real_bb/<pid>.npz`, key `bb_8` = 8 MD frames of
N/CA/C/O), it:
  1. takes the first P frames (P = 8 by default, i.e. the full MISATO ensemble;
     `--p-inference 1` keeps only the first frame, the "Ensembits, P=1" mode),
  2. computes the 256-D descriptor per frame and residue
     (16 neighbours x [12-D relative frame | sin/cos phi | sin/cos psi]),
  3. standardises it with the checkpoint's stored mean/std (identity for the
     reported model, which was trained without standardisation),
  4. runs the set encoder + residual quantizer and keeps the level-1 token per
     residue (`model.encode`), and
  5. writes all tokens to one `.npz` keyed by pid.

The EC / GO / binding / MISATO-RMSF scripts read this cache through
`tokenize_misato()` in `tokenize_misato_and_eval.py`; they only tokenize on the
fly when the cache file is missing (and that fallback cannot build 256-D tokens,
see the note there). So this script must be run first, once for P = full and
once for P = 1.

The descriptor's frame block is bit-identical to the old 192-D descriptor, so a
192-D cache and this cache differ only by the per-neighbour dihedrals.

The tokenizer's training corpus (BioEmu-CATH + MISATO) contains most of the
MISATO downstream test complexes; this is accepted because the tokenizer never
sees the downstream labels.

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool alphabet carry functional
    information?} \\label{sec:downstreams}: the MISATO downstream suite that
    consumes these tokens.
  * `appendix.tex`, \\section{Baselines} \\label{app:baselines}: "\\tool, P=1"
    (single conformation through the set encoder) and "\\tool, P=full"
    (P = 8 on MISATO).
  * `appendix.tex`, \\subsection{ESM3-style relative-frame descriptor}
    \\label{app:descriptor-esm3}: the 256-D descriptor.
  * `1_method.tex`, \\paragraph{Residual quantizer.}: only level-1 tokens
    (M_1 = 2048) are used downstream.

Origin
------
Copied from `submission_exp/src/_pretokenize_misato_dihedral.py`. Changes:
imports and sys.path setup only (`build_esm3desc_dihedral` ->
`ensembits.descriptor`, `ensembits_featurize` -> `ensembits.featurize`; the
research-repo entries `scripts/baselines` and `submission_exp/src` were dropped
from sys.path because nothing is imported from them any more), plus comments.
Logic, defaults, file names and CLI flags are unchanged.
Fixes 2026-10-16 (FIXED markers): exit 1 (was 0) without config.json or for an
unfinished training run unless --allow-unfinished (code_fix P12); provenance sidecar
written and checked on reuse, --force (P17).

Used by the reported model: yes. The battery
(`downstream/_run_model_battery.py`, phase A) runs it for the reported checkpoint
with P = full and with `--p-inference 1`.

Inputs / outputs
----------------
Read  : `output/<variant>/{config.json,best.pt,stats.npz}` (the checkpoint)
        `data/cached_descriptors/misato_real_bb/<pid>.npz` (key `bb_8`; built by
        `data_prep/build_misato_real_bb.py`)
Write : `data/cached_descriptors/ensembits_<tag>_tokens_misato.npz`
        (`<tag>` = variant name, plus `_P<n>inf` with `--p-inference n`)
        `data/cached_descriptors/ensembits_<tag>_tokens_misato.npz.provenance.json`
        (checkpoint block: variant, sha256 of best.pt, code_sha256; code_fix P17)
All paths are relative to REPO_ROOT (the research repo). An existing cache is
never overwritten, except with --force; it is reused only if its provenance
sidecar names the same checkpoint (2026-10-16, code_fix P17).
Exit status: 0 = cache written or a matching cache exists; 1 = refused (no
config.json, unfinished training run, provenance mismatch, wrong input_dim).

How to run
----------
  python downstream/_pretokenize_misato_dihedral.py \\
      --variant dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
      --device cuda:0 [--p-inference 1] [--allow-unfinished] [--force]
"""
from __future__ import annotations

import os
# Keep CPU thread pools small (several of these jobs run in parallel).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

from ensembits.descriptor import _descriptor, DESC_DIM  # noqa: E402

# Real-backbone MISATO frames (one .npz per pid) and the token cache folder.
REAL_BB = ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'
TOK_CACHE_DIR = ROOT / 'data' / 'cached_descriptors'
P = 8     # misato downstream uses 8 frames (matches coords_ca8_misato.pkl)
# Paper: appendix.tex \section{Baselines} \label{app:baselines} — "\tool, P=full" uses P=8 on MISATO.


def tag_of(name: str) -> str:
    """Cache tag of a checkpoint folder name (strips the old `final_model` prefix)."""
    return name.replace('final_model_', '').replace('final_model', '') or name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--p-inference', type=int, default=None,
                     help='Use only the first P_inference frames at inference.')
    ap.add_argument('--allow-old-parser', action='store_true',
                    help='accept MISATO files written before the code_fix §4 parser fix '
                         '(only to reproduce checkpoints trained before 2026-10-15)')
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='accept a checkpoint whose training run has not finished '
                         '(config.json "status": "running"; code_fix P12)')
    ap.add_argument('--force', action='store_true',
                    help='rebuild an existing cache even if its provenance names another '
                         'checkpoint (code_fix P17)')
    args = ap.parse_args()

    # ---- Locate the checkpoint and the output cache file ----
    # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv.
    variant_dir = ROOT / 'output' / args.variant
    # FIXED(code_fix P12, 2026-10-16): old: a folder without config.json printed "[skip]"
    #   and exited 0, so the battery went on with a missing cache. New: exit 1 for a missing
    #   config.json or an unfinished training run (config "status": "running"), unless
    #   --allow-unfinished (which only covers the unfinished case).
    from ensembits.featurize import (load_model, check_checkpoint_finished,
                                     UnfinishedCheckpointError, checkpoint_provenance,
                                     check_provenance_sidecar, write_provenance_sidecar,
                                     ProvenanceMismatchError)
    try:
        check_checkpoint_finished(variant_dir, allow_unfinished=args.allow_unfinished)
    except UnfinishedCheckpointError as e:
        print(f'[abort] {e}', file=sys.stderr); return 1
    tag = tag_of(args.variant)
    if args.p_inference is not None:
        tag = f'{tag}_P{args.p_inference}inf'
    cache_path = TOK_CACHE_DIR / f'ensembits_{tag}_tokens_misato.npz'
    # FIXED(code_fix P17, 2026-10-16): old: an existing cache was reused by name only. New:
    #   it is reused only if its <cache>.provenance.json names this checkpoint (best.pt
    #   sha256 + code_sha256); a mismatch (or a missing sidecar for a checkpoint of the
    #   fixed trainer) exits 1, unless --force, which rebuilds the cache.
    prov = checkpoint_provenance(variant_dir)
    if cache_path.exists():
        try:
            state = check_provenance_sidecar(cache_path, prov, force=False)
            print(f'[skip] cache exists ({state} provenance): {cache_path.name}'); return 0
        except ProvenanceMismatchError as e:
            if not args.force:
                print(f'[abort] {e}', file=sys.stderr); return 1
            print(f'[force] rebuilding {cache_path.name}: {e}', flush=True)

    # ---- Load the tokenizer and its descriptor standardisation stats ----
    ens = load_model(str(variant_dir), device=args.device,
                     allow_unfinished=args.allow_unfinished)
    cfg = ens.config
    if cfg.get('input_dim') != DESC_DIM:
        print(f'[abort] expected input_dim={DESC_DIM} for dihedral, '
              f'got {cfg.get("input_dim")}'); return 1
    # stats.npz holds mean 0 / std 1 for the reported model (trained with
    # --no-standardize), so this step is the identity there.
    # Paper: appendix.tex \label{app:model-impl} — descriptors used without standardization.
    mu = torch.tensor(ens.mean.astype(np.float32), device=args.device)
    std = torch.tensor(ens.std.astype(np.float32), device=args.device)

    # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
    from ensembits.data_versions import require_misato_parser
    require_misato_parser(REAL_BB, allow_old=args.allow_old_parser)
    pids = sorted(p.stem for p in REAL_BB.glob('*.npz'))
    P_eff = args.p_inference if args.p_inference is not None else P
    print(f'[tokenize] {len(pids)} misato pids on {args.device}, '
          f'desc=dihedral {DESC_DIM}-D P={P_eff}', flush=True)

    # ---- Tokenize every MISATO complex ----
    tokens = {}
    t0 = time.time()
    for i, pid in enumerate(pids):
        rb_path = REAL_BB / f'{pid}.npz'
        try:
            d = np.load(rb_path, allow_pickle=False)
            if 'bb_8' not in d.files:
                continue
            bb = np.asarray(d['bb_8'], dtype=np.float32)        # (8, L, 4, 3)
            if bb.shape[0] < P_eff:
                continue
            # Keep the first P_eff frames (P=1 -> the first MD frame only).
            # Paper: appendix.tex \label{app:baselines} — "\tool, P=1" from a single conformation.
            bb = bb[:P_eff]
            # Drop O: the descriptor needs N, CA, C only.
            coords = np.stack([bb[:, :, 0, :], bb[:, :, 1, :], bb[:, :, 2, :]],
                              axis=2)                            # (P, L, 3, 3)
            with torch.no_grad():
                # 256-D descriptor per frame and residue.
                # Paper: appendix.tex \label{app:descriptor-esm3} — 16 neighbours x [12-D frame | phi/psi sin/cos].
                # FIXED(code_fix §1, 2026-10-15): the descriptor uses the dihedral
                # convention the checkpoint was trained with (legacy +180 deg for
                # old checkpoints, IUPAC for retrained ones), read from its config.
                # FIXED(require_fix #17 / D.14, tex pass 2026-10-26): 256-D stated in method.
                desc = _descriptor(torch.from_numpy(coords), args.device,
                                   convention=ens.dihedral_convention,
                                   chain_break_cutoff=ens.chain_break_cutoff)  # (P, L, 256)
                desc = desc.to(args.device).transpose(0, 1).contiguous()    # (L, P, 256)
                desc = (desc - mu) / std
                # Set encoder + RVQ; `encode` returns the level-1 token only.
                # Paper: 1_method.tex \paragraph{Residual quantizer.} — only level-1 tokens (M_1 = 2048) used downstream.
                toks = ens.model.encode(desc)                              # (L,)
            tokens[pid] = toks.cpu().numpy().astype(np.int64)
        except Exception as exc:
            print(f'  {pid}: err {type(exc).__name__} {str(exc)[:80]}', flush=True)
            continue
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = (len(pids) - i - 1) / max((i + 1) / elapsed, 1e-9)
            print(f'  {i+1:>5}/{len(pids)}  {elapsed:.0f}s ETA={eta:.0f}s '
                  f'(ok={len(tokens):,})', flush=True)

    # ---- Write the cache ----
    print(f'[done] {len(tokens):,} pids tokenized in {time.time()-t0:.0f}s',
          flush=True)
    # FIXED(code_fix P14, 2026-10-16): refuse to write a cache that covers fewer than
    #   95% of the MISATO complexes (e.g. a wrong or half-built backbone cache), instead
    #   of writing a near-empty cache that every downstream task would read silently.
    if len(tokens) < 0.95 * len(pids):
        print(f'[error] only {len(tokens):,} of {len(pids):,} pids tokenized (< 95%); '
              f'not writing {cache_path.name}', flush=True)
        return 1
    np.savez_compressed(cache_path, **tokens)
    from ensembits.featurize import file_sha256
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rmsf_common import dataset_input_record
    # FIXED(code_fix P17, 2026-10-16): new (old: none): provenance sidecar (never a key inside the pid-keyed npz).
    write_provenance_sidecar(cache_path, prov, p_inference=args.p_inference, P=P_eff,
                             n_pids=len(tokens), n_pids_total=len(pids),
                             source=str(REAL_BB.relative_to(ROOT)) if REAL_BB.is_relative_to(ROOT)
                             else str(REAL_BB),
                             dihedral_convention=ens.dihedral_convention,
                             chain_break_cutoff=ens.chain_break_cutoff,
                             written_by='downstream/_pretokenize_misato_dihedral.py',
                             # FIXED(preretrain_audit_r2 H1, 2026-10-16): new: the inputs of the
                             #   cache (MISATO parser versions, K, FPS rule of misato_real_bb),
                             #   the frames used and the cache's sha256; the RMSF probe requires
                             #   the current parser, K = 8 and Kabsch FPS.
                             inputs=dataset_input_record('misato'),
                             frames=f'first {P_eff} of bb_8',
                             cache_sha256=file_sha256(cache_path))
    print(f'wrote {cache_path}  ({cache_path.stat().st_size/1e6:.1f} MB) + provenance sidecar',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
