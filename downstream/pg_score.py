"""ProteinGym zero-shot scoring core: WT-vs-mutant token distance, blended with ESM2.

What it does
------------
For each ProteinGym DMS assay that has a cached ESMFold structure folder
(`data/proteingym/esmfold_cache/<assay>/`, with `manifest.json` naming the WT
sequence and one `<sha1[:16]>.npy` structure per unique sequence):
  1. tokenizes the WT structure and every mutant structure as a single frame
     (P = 1), keeping all RVQ levels;
  2. computes four disruption scores per variant (below);
  3. negates them ("higher = more WT-like = predicted fitter") and reports the
     per-assay Spearman against `DMS_score`, ESM2-650M's own Spearman, and the
     within-assay z-score blends alpha * z(ours) + (1 - alpha) * z(ESM2) for
     alpha in {0.1, ..., 0.5};
  4. writes one JSON per assay (resumable), then one aggregate JSON, plus the
     per-variant scores in a folder tagged by `--out-tag`.
Assays are spread over GPUs with one worker thread per GPU.

Assay set: `--assays a,b,...` or `--assay-list <file>` (one assay id per line,
blank lines and `#` comments ignored) name the assays explicitly; with neither,
the fixed list `downstream/pg_assays_wt_le200.txt` is scored (the paper's 96
assays with WT length <= 200, built and checked by `pg_assay_list.py`;
`--all-cached` restores the old default "every cache folder with a manifest").
The assays actually scored (those with a per-assay JSON after the run) are
written to `per_assay_pg_<tag>.assays_scored.json` and their count is printed.

Camera-ready fixes (benchmark_audit D9, 2026-10-19; FIXED markers below):
  * float64 throughout (DMS, ESM2, scores); the float32 cast created ties that
    changed the ESM2 Spearman of some assays between the Ensembits and baseline files;
  * one variant set per assay for every score: variants without a structure, with a
    failed tokenization or a non-finite DMS / ESM2 value are dropped for all keys
    (old: a single NaN ESM2 removed every blend key of the assay, so the blend mean
    ran over fewer assays than the standalone mean); a constant score has Spearman 0
    (flagged in `constant_scores`), not NaN;
  * new key `spearman_ranksum_a` (parameter-free rank-sum blend of score (a) with
    ESM2; it existed only for score (d));
  * per-variant npz gain `mutant` and `n_mut` (mutated positions), so
    `pg_aggregate.py` can align methods variant by variant, build the -n_mut control
    and choose alpha by leave-one-UniProt-out CV;
  * provenance: every per-assay JSON has an `inputs` block (CSV, manifest and
    structure-content sha256, codebook sha256) and the sha256 of its per-variant npz;
    the assays_scored sidecar has the assay list sha256 and the structure set sha256;
    all files are written atomically;
  * the tokenizer hook is `tokenize_ca(model, structure, seq)` (the sequence is needed
    by ProToken; `pg_score_baselines.py` plugs the baseline tokenizers in through
    `main(argv, baseline=...)`);
  * blind by default: the console shows no Spearman values (per-assay or mean) unless
    `--print-scores` (lambda / model selection must not see ProteinGym).

Scores per variant (code calls RVQ level 1 "L0"):
  (a) sum over all positions of ‖cb_L0[wt_i] − cb_L0[mut_i]‖₂ (positive = disruption)
  (b) sum over MUTATED positions only of the same
  (c) Hamming: fraction of residues where L0 token changed
  (d) sum over all positions of ‖[cb_L0; cb_L1; cb_L2][wt_i] − [...][mut_i]‖₂ (3-level)

For Spearman: we negate so "higher score = more preserved = predicted higher fitness".

The paper's numbers use score (a) (`spearman_score_a_allL0`) and its blend at
alpha = 0.3 (`spearman_zmix_a_a03`); see `_compile_rebuttal_results.py`.
Camera-ready (D9): alpha is chosen by leave-one-UniProt-out CV in
`pg_aggregate.py` from the per-variant files; the per-assay `zmix` keys stay as
raw per-assay values.

The tokenizer used here (`tokenize_ca`) is the legacy Cα-only path. The
reported 256-D model needs real N/CA/C, so `pg_score_dihedral.py` replaces
`tokenize_ca` and `CACHE_DIR` before calling `main()`; that wrapper is what the
battery runs.

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool alphabet carry functional
    information?} \\label{sec:downstreams}, \\paragraph{Mutation Effect
    Prediction} and Table \\label{tab:proteingym_ensemble}: ProteinGym
    substitutions, 96 DMS assays with WT length <= 200, ESMFold structures,
    P = 1, score = -sum_i ‖C^1[c_wt] − C^1[c_mut]‖₂ from the L_1 codeword only
    (= score (a)), within-assay z-score blend with ESM2-650M at alpha = 0.3.
  * `appendix.tex`, \\subsection{Compute Requirements} \\label{sec:compute}:
    "ProteinGym scoring (all 96 assays, one tokenizer) ~40 min CPU".

Origin
------
Copied from `submission_exp/src/pg_score.py`. Changes: ROOT is now
`ensembits.paths.REPO_ROOT`; sys.path holds this codebase and this script's
folder instead of `REPO_ROOT`, `submission_exp/src` and `scripts/baselines`;
`from ensembits_featurize import load_model / _descriptor` ->
`from ensembits.featurize import ...`; comments and this docstring.
Changed on 2026-10-15 (approved, code_fix §R): the per-variant folder is now
`per_variant_pg_<out-tag>/` (was the untagged `per_variant_pg/`, overwritten by
every run); new `--assay-list`; the scored assay list is written to a sidecar
JSON. Scores and the per-assay / aggregate JSONs are unchanged.
Fixes 2026-10-16 (FIXED markers): refuses an unfinished checkpoint up front
(--allow-unfinished) (code_fix P12); `checkpoint` block in every per-assay JSON and
in the assays_scored file, checked when resuming (P17).

Used by the reported model: partly. `evaluate_assay`, `main` and
`load_variant_model` are called through `pg_score_dihedral.py` (the wrapper
for the reported 256-D model); `tokenize_ca`, the default `--model-dir`
(old 192-D model) and the default `CACHE_DIR` are not.

Inputs / outputs
----------------
Reads:
  data/external/proteingym_zero_shot_scores/<assay>.csv (ENSEMBITS_PROTEINGYM_DIR)
      (external, another user's directory: mutated_sequence, DMS_score, ESM2_650M)
  data/proteingym/esmfold_cache[_atom14]/<assay>/{manifest.json,<hash>.npy}   (REPO_ROOT)
  <model-dir>/{config.json,best.pt,stats.npz}
  [--assay-list file]                                  one assay id per line (optional)
Writes (REPO_ROOT):
  output/proteingym/per_assay_pg_<tag>/<assay>.json   per-assay Spearman table
  output/proteingym/per_assay_pg_<tag>.json           aggregate over assays {assay: result}
  output/proteingym/per_assay_pg_<tag>.assays_scored.json
      {assays_scored, n_assays_scored, assays_requested, assays_not_scored,
       assay_source, cache_dir}; a sidecar, because the aggregate is a flat
       {assay: result} map that the table compilers iterate over
  output/proteingym/per_variant_pg_<tag>/<assay>.npz  per-variant scores
      {mutant, n_mut, fitness, esm2, score_a, score_b, score_c, score_d} (float64;
       mutant / n_mut since 2026-10-19). Runs before 2026-10-15 wrote the untagged
       `per_variant_pg/`, which holds whichever tokenizer ran last.

Dependencies left in the research repo: none in code. The ESMFold structure
caches are built by `data_prep/build_pg_esmfold_cache.py` (ported from the
research repo's `submission_exp/src/pg_esmfold_cache.py`).

How to run
----------
Not run directly for the reported model; use `pg_score_dihedral.py` (Ensembits)
or `pg_score_baselines.py` (baselines), e.g.
    python downstream/pg_score_dihedral.py \\
        --model-dir output/<variant> --out-tag <tag> --gpus 0
then `pg_aggregate.py` for the table (alpha by leave-one-UniProt-out CV).
"""
from __future__ import annotations
import os
# Cap BLAS threads (one worker thread per GPU shares the CPU).
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, csv, json, queue, sys, threading, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, rankdata

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
# External dependency: ProteinGym substitution CSVs with the precomputed
# zero-shot scores (ESM2_650M column), under data/external/ (ensembits.paths).
from ensembits.paths import PROTEINGYM_ZERO_SHOT_DIR as PG_DIR   # noqa: E402
# Per-assay ESMFold structures (Cα only here; pg_score_dihedral.py repoints this
# to the atom14 cache). Without --assays / --assay-list, which assays have a
# cache decides the assay set: the paper's "96 assays with WT length <= 200" is
# not filtered in this file (pass --assay-list to fix it explicitly; the scored
# list is written to per_assay_pg_<tag>.assays_scored.json either way).
CACHE_DIR = ROOT / 'data' / 'proteingym' / 'esmfold_cache'
OUT_DIR = ROOT / 'output' / 'proteingym'
# Paper: 2_results.tex tab:proteingym_ensemble — "ESM2-650M" baseline column.
ESM_COLUMN = 'ESM2_650M'

# This script's folder on sys.path (sibling imports resolve to downstream/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pg_common                                   # noqa: E402  D9 helpers (float64, sha, CV)

import hashlib
# File name of a sequence's cached structure: first 16 hex chars of its SHA-1.
def hash_seq(s): return hashlib.sha1(s.encode()).hexdigest()[:16]

# FIXED(benchmark_audit D9, 2026-10-19): new. Version tag of the per-assay JSON schema
#   (float64, common variant mask, inputs block). A resumed per-assay JSON with another
#   version is refused (old files cannot be mixed with new ones under one tag).
PG_VERSION = 'd9-2026-10-19'
# Printing Spearman values on the console (per assay and means) is off by default:
# lambda selection is blind to ProteinGym (--print-scores turns it on).
PRINT_SCORES = False


def load_variant_model(model_dir, device, allow_unfinished=False):
    """Load our shipped model + 3-level codebook."""
    # Returns the Ensembits bundle and the list of RVQ codebooks
    # [(2048,128), (128,128), (128,128)] for the reported model.
    from ensembits.featurize import load_model
    # FIXED(code_fix P12, 2026-10-16): new (old: none): load_model refuses an unfinished run unless allow_unfinished.
    ens = load_model(model_dir, device=device, allow_unfinished=allow_unfinished)
    state = ens.model.state_dict()
    cbs = []
    for lvl in range(3):
        k = f'vq_levels.{lvl}.codebook.weight'
        if k not in state:
            break
        cbs.append(state[k].detach().cpu().numpy().astype(np.float32))
    print(f'[load] codebooks: {[c.shape for c in cbs]}', flush=True)
    return ens, cbs


def tokenize_ca(ens, ca, seq=None):
    """ca: (L, 3) → (L,) primary token AND (L, n_levels) all-level tokens.
    Treats the structure as P=1 (single frame)."""
    # Legacy Cα-only path for the old descriptor modes. It is overridden for the
    # reported 256-D model by pg_score_dihedral._tokenize_dihedral. Note that
    # ensembits.featurize._descriptor (this codebase) only carries the 256-D
    # descriptor and raises without bb_all, so this function no longer works
    # here; it is kept unchanged for reference.
    # FIXED(benchmark_audit D9, 2026-10-19): third argument `seq` (the sequence of the
    #   structure; unused here, needed by ProToken in pg_score_baselines.py).
    from ensembits.featurize import _descriptor
    ca_p1 = ca[None, :, :]   # (1, L, 3)
    desc = _descriptor(ens, ca_p1.astype(np.float32))   # (L, 1, D) standardized
    tok_L0 = ens.model.encode(desc).cpu().numpy().astype(np.int64)  # (L,)
    # All levels via encode_all (RVQ → list of (L,) tensors per level)
    if hasattr(ens.model, 'encode_all'):
        with torch.no_grad():
            tok_list = ens.model.encode_all(desc)
        tok_all = np.stack([t.cpu().numpy().astype(np.int64) for t in tok_list],
                            axis=-1)   # (L, n_levels)
    else:
        tok_all = tok_L0[:, None]
    return tok_L0, tok_all


def _rel(p):
    """Path relative to REPO_ROOT when possible (provenance without machine paths)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def evaluate_assay(assay, ens, cbs, scope='all', out_tag=None):
    """Return per-variant scores + DMS + ESM2 baseline.

    `scope`: 'all' = score over every residue, 'mut' = only mutated positions.
    (`scope` is not used: scores (a) and (b) are always both computed.)
    `out_tag`: per-variant scores go to `per_variant_pg_<out_tag>/` (main()
    always passes --out-tag); None keeps the old untagged `per_variant_pg/`.

    Returns None (assay skipped) when the structure cache, the CSV, the manifest
    or the WT structure is missing, or fewer than 10 usable variants remain.
    Raises ValueError when a CSV row disagrees with the WT (mutant string vs
    sequence) or a mutant id repeats (pg_common.load_assay_rows).
    """
    # ---- WT structure and tokens ------------------------------------------
    cdir = CACHE_DIR / assay
    if not cdir.exists():
        return None
    csv_path = PG_DIR / f'{assay}.csv'
    if not csv_path.exists():
        return None

    # Load WT
    man_path = cdir / 'manifest.json'
    if not man_path.exists():
        return None
    man_bytes = man_path.read_bytes()
    man = json.loads(man_bytes)
    wt_seq = man['wt']
    wt_hash = man['wt_hash']
    wt_ca_path = cdir / f'{wt_hash}.npy'
    if not wt_ca_path.exists():
        return None
    # FIXED(benchmark_audit D9, 2026-10-19): every structure read is hashed (content
    #   sha256), so the result records exactly which ESMFold structures it used (old: none).
    file_shas = {}
    wt_ca, file_shas[wt_hash] = pg_common.read_structure(wt_ca_path)
    wt_tok, wt_tok_all = tokenize_ca(ens, wt_ca, wt_seq)

    # ---- Variants from the ProteinGym CSV ----------------------------------
    # (substitutions only: rows whose mutated sequence differs in length from
    # the WT are dropped; the ESM2_650M column is the ESM2 baseline score)
    # FIXED(benchmark_audit D9, 2026-10-19): rows read by pg_common.load_assay_rows in
    #   float64 (old: float32 below, which created ties), with n_mut = number of mutated
    #   positions, and refused if a mutant string disagrees with its sequence.
    rows, row_stats = pg_common.load_assay_rows(csv_path, wt_seq)
    if row_stats['n_inconsistent'] or row_stats['n_dup_mutant']:
        raise ValueError(f'{assay}: {row_stats["n_inconsistent"]} rows disagree with the WT, '
                         f'{row_stats["n_dup_mutant"]} repeated mutant ids')

    if len(rows) < 10:
        return None

    # ---- Score every variant against the WT ---------------------------------
    n = len(rows)
    # FIXED(benchmark_audit D9, 2026-10-19): float64 scores and codebooks (old: float32).
    score_a = np.zeros(n, dtype=np.float64)   # all positions, L0 only
    score_b = np.zeros(n, dtype=np.float64)   # mut positions, L0 only
    score_c = np.zeros(n, dtype=np.float64)   # hamming over all positions
    score_d = np.zeros(n, dtype=np.float64)   # all positions, all-levels concat
    cbs = [np.asarray(c, dtype=np.float64) for c in cbs]

    cb0 = cbs[0]   # (M0, d0)
    if len(cbs) >= 3:
        # 384-D combined embedding per token-tuple: concat the 3 level embeddings
        wt_emb_full = np.concatenate(
            [cbs[lvl][wt_tok_all[:, lvl]] for lvl in range(3)], axis=-1)  # (L, d0+d1+d2)
    else:
        wt_emb_full = cb0[wt_tok]   # fallback

    wt_emb_L0 = cb0[wt_tok]   # (L, d0)

    valid = np.ones(n, dtype=bool)
    n_no_struct = n_tok_fail = 0
    tok_cache = {wt_hash: (wt_tok, wt_tok_all)}   # one tokenization per unique structure
    for i, r in enumerate(rows):
        h = hash_seq(r['mut_seq'])
        if h not in tok_cache:
            ca_path = cdir / f'{h}.npy'
            if not ca_path.exists():
                valid[i] = False; n_no_struct += 1; continue
            ca, file_shas[h] = pg_common.read_structure(ca_path)
            try:
                tok_cache[h] = tokenize_ca(ens, ca, r['mut_seq'])
            except Exception as exc:
                # B.17: slot errors are fatal (was: silently counted as drops)
                raise RuntimeError(
                    f'[{assay}] tokenize fail variant {i}: '
                    f'{type(exc).__name__}: {exc}') from exc
        if tok_cache[h] is None:
            valid[i] = False; n_tok_fail += 1; continue
        mut_tok, mut_tok_all = tok_cache[h]
        if len(mut_tok) != len(wt_tok):
            valid[i] = False; n_tok_fail += 1; continue

        # diff positions
        ms = r['mut_seq']
        diff_pos = np.array([j for j, (a, b) in enumerate(zip(wt_seq, ms))
                              if a != b], dtype=np.int64)

        # (a) all-positions L0 distance
        # Paper: 2_results.tex, Mutation Effect Prediction — "the sum, across
        # positions, of the Euclidean distance between the wild-type and mutant
        # L_1 codewords" (the paper's L_1 = level 0 here).
        d_l0_per = np.linalg.norm(wt_emb_L0 - cb0[mut_tok], axis=-1)  # (L,)
        score_a[i] = d_l0_per.sum()

        # (b) mut-positions L0 distance
        score_b[i] = d_l0_per[diff_pos].sum() if len(diff_pos) else 0.0

        # (c) Hamming over all positions
        score_c[i] = float((wt_tok != mut_tok).mean())

        # (d) all-positions, all-levels distance
        if len(cbs) >= 3:
            mut_emb_full = np.concatenate(
                [cbs[lvl][mut_tok_all[:, lvl]] for lvl in range(3)], axis=-1)
            d_full_per = np.linalg.norm(wt_emb_full - mut_emb_full, axis=-1)
            score_d[i] = d_full_per.sum()
        else:
            score_d[i] = score_a[i]

    n_dropped = n_no_struct + n_tok_fail
    if n_dropped:
        print(f'  [{assay}] skipped {n_no_struct} variants without a structure, '
              f'{n_tok_fail} with a failed tokenization', flush=True)
    # FIXED(pg_audit, 2026-10-21): cap on silently dropped variants.
    drop_frac = n_dropped / max(n, 1)
    if drop_frac > 0.5:
        raise RuntimeError(
            f'[{assay}] {n_dropped}/{n} variants ({drop_frac:.0%}) dropped by '
            f'tokenization failures / missing structures; refusing to score. '
            f'Check the ESMFold cache and the tokenizer.')

    # ---- Spearman of each score and of ESM2 against the DMS fitness ---------
    fitness = np.array([r['DMS'] for r in rows], dtype=np.float64)
    esm2 = np.array([r['ESM2'] for r in rows], dtype=np.float64)
    # FIXED(benchmark_audit D9, 2026-10-19): one variant mask for every key, including a
    #   finite ESM2 (old: ESM2 was not in the mask, and one NaN ESM2 removed all blend keys
    #   of the assay while the standalone keys stayed).
    finite = (valid & np.isfinite(fitness) & np.isfinite(esm2) & np.isfinite(score_a) &
              np.isfinite(score_b) & np.isfinite(score_c) & np.isfinite(score_d))
    if finite.sum() < 10:
        return None
    fa = fitness[finite]; ea = esm2[finite]
    sa = score_a[finite]; sb = score_b[finite]; sc = score_c[finite]; sd = score_d[finite]
    mutants = np.array([r['mutant'] for r in rows])[finite]
    n_mut = np.array([r['n_mut'] for r in rows], dtype=np.int64)[finite]

    constant = []
    def neg_spear(x, y, name):
        # higher x (disruption) = lower fitness predicted; report -corr so positive = good
        # Paper: 2_results.tex — "negate so that larger values predict more
        # wild-type-like (and presumed more fit) variants".
        # FIXED(benchmark_audit D9, 2026-10-19): pg_common.spearman (float64; a constant
        #   score -> 0.0 and listed in `constant_scores`; old: NaN, dropped from the mean).
        rho, const = pg_common.spearman(-x, y)
        if const: constant.append(name)
        return rho

    out = {
        'n_variants': int(finite.sum()),
        'spearman_score_a_allL0':  neg_spear(sa, fa, 'score_a'),
        'spearman_score_b_mutL0':  neg_spear(sb, fa, 'score_b'),
        'spearman_score_c_hamming': neg_spear(sc, fa, 'score_c'),
        'spearman_score_d_allLvl':  neg_spear(sd, fa, 'score_d'),
        'spearman_esm2': pg_common.spearman(ea, fa)[0],
    }

    # ---- z-score blends with ESM2 -------------------------------------------
    # Combined ensembles (using score_d as our default)
    # (3-level score (d); not the variant reported in the paper)
    # Paper: 2_results.tex, Mutation Effect Prediction and tab:proteingym_ensemble —
    # "alpha * z(\tool) + (1 - alpha) * z(ESM2) at alpha = 0.3"; the reported key is
    # spearman_zmix_a_a03. These per-assay keys are raw values on the alpha grid
    # 0.1-0.5; the camera-ready alpha is chosen by leave-one-UniProt-out CV in
    # pg_aggregate.py from the per-variant file (benchmark_audit D9).
    grid = (0.1, 0.2, 0.3, 0.4, 0.5)
    for key, s in (('d', sd), ('a', sa)):
        rhos = pg_common.blend_spearman(-s, ea, fa, grid)
        for alpha, rho in zip(grid, rhos):
            out[f'spearman_zmix_{key}_a{int(round(alpha*10)):02d}'] = float(rho)
        # FIXED(benchmark_audit D9, 2026-10-19): parameter-free rank-sum also for score (a)
        #   (old: only `spearman_ranksum_d`).
        out[f'spearman_ranksum_{key}'] = pg_common.ranksum_spearman(-s, ea, fa)
    out['constant_scores'] = constant
    out['n_dropped'] = {'no_structure': int(n_no_struct), 'tokenize_fail': int(n_tok_fail),
                        'nonfinite': int(valid.sum() - finite.sum())}
    out['n_multi'] = int((n_mut > 1).sum())

    # Save per-variant scores for downstream remix
    # FIXED(code_fix §R, 2026-10-15): the folder was the untagged `per_variant_pg/`,
    #   so every tokenizer run overwrote the previous one's per-variant scores
    #   (the reported model's files there may come from a later run). It is now
    #   `per_variant_pg_<out_tag>/`. The per-assay JSONs were already tagged.
    # FIXED(benchmark_audit D9, 2026-10-19): float64 arrays plus `mutant` (variant id) and
    #   `n_mut` (mutated positions); atomic write; its sha256 goes into the per-assay JSON.
    var_dir = OUT_DIR / (f'per_variant_pg_{out_tag}' if out_tag else 'per_variant_pg')
    var_path = var_dir / f'{assay}.npz'
    pg_common.atomic_savez(var_path, mutant=mutants, n_mut=n_mut,
                           fitness=fa, esm2=ea,
                           score_a=sa, score_b=sb, score_c=sc, score_d=sd)
    # FIXED(benchmark_audit D9, 2026-10-19): new `inputs` block (provenance of the inputs).
    out['inputs'] = {
        'csv': _rel(csv_path), 'csv_sha256': pg_common.file_sha256(csv_path),
        'cache_dir': _rel(CACHE_DIR), 'manifest_sha256': hashlib.sha256(man_bytes).hexdigest(),
        'n_structures': len(file_shas),
        'structure_sha256': pg_common.structure_digest(file_shas),
        'codebook_sha256': hashlib.sha256(b''.join(
            np.ascontiguousarray(c, dtype=np.float32).tobytes() for c in cbs)).hexdigest(),
        'row_stats': row_stats}
    out['per_variant_npz'] = _rel(var_path)
    out['per_variant_sha256'] = pg_common.file_sha256(var_path)
    out['pg_version'] = PG_VERSION
    return out


def build_parser():
    # ---- CLI -------------------------------------------------------------
    # The default --model-dir / --out-tag are the old 192-D preprint model
    # ("ship"); the battery always passes the variant and its PG tag.
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=str(ROOT / 'submission_exp' / 'ship' / 'model'),
                    help='Tokenizer model dir (default: shipped)')
    ap.add_argument('--gpus', default='0,1,2')
    # Assay set: --assays or --assay-list name it explicitly; default = the fixed
    # list downstream/pg_assays_wt_le200.txt (96 assays); --all-cached = every assay
    # folder of CACHE_DIR with a manifest (the old default).
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument('--assays', default=None,
                     help='Comma-separated subset (default: the fixed 96-assay list)')
    sel.add_argument('--assay-list', default=None,
                     help='file with one assay id per line (blank lines and #-comments '
                          'ignored); default: downstream/pg_assays_wt_le200.txt')
    sel.add_argument('--all-cached', action='store_true',
                     help='old default: every assay folder of the structure cache')
    ap.add_argument('--out-tag', default='ship',
                    help='Output filename suffix (default: ship)')
    ap.add_argument('--force', action='store_true',
                    help='Recompute even if per-assay JSON already exists.')
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='accept a checkpoint whose training run has not finished (code_fix P12)')
    ap.add_argument('--print-scores', action='store_true',
                    help='print per-assay and mean Spearman values (off: blind run)')
    ap.add_argument('--out-dir', default=None,
                    help='output folder (default REPO_ROOT/output/proteingym; smoke runs)')
    return ap


def main(argv=None, baseline=None):
    """Score the assays with the Ensembits checkpoint (default) or a baseline.

    `baseline` (pg_score_baselines.py): {'load': f(device) -> (tokenizer, [codebook]),
    'provenance': dict} replaces the checkpoint load and the checkpoint block; the
    module-level `tokenize_ca` must be swapped by the caller.
    """
    global PRINT_SCORES, OUT_DIR
    args = build_parser().parse_args(argv)
    PRINT_SCORES = args.print_scores
    if args.out_dir:                      # FIXED(benchmark_audit D9, 2026-10-19): new option
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # FIXED(code_fix P12 / P17, 2026-10-16): old: the checkpoint was only loaded inside the
    #   worker threads (an error there killed the thread silently) and the outputs carried
    #   no provenance. New: refuse an unfinished / config-less checkpoint up front (exit 1);
    #   every per-assay JSON gets a "checkpoint" block (variant, best.pt sha256,
    #   code_sha256); a resumed per-assay JSON must carry the same block (exit 1
    #   otherwise; --force recomputes it).
    from ensembits.featurize import (check_checkpoint_finished, UnfinishedCheckpointError,
                                     checkpoint_provenance, check_recorded_checkpoint,
                                     ProvenanceMismatchError)
    if baseline is None:
        try:
            check_checkpoint_finished(args.model_dir, allow_unfinished=args.allow_unfinished)
        except UnfinishedCheckpointError as e:
            print(f'[abort] {e}', file=sys.stderr); return 1
        ckpt_block = checkpoint_provenance(args.model_dir)
        def load(device):
            return load_variant_model(args.model_dir, device,
                                      allow_unfinished=args.allow_unfinished)
    else:
        # FIXED(benchmark_audit D9, 2026-10-19): new: baselines through the same scorer.
        ckpt_block = baseline['provenance']
        load = baseline['load']

    # ---- Assay list: --assays, --assay-list, the fixed list, or every cache folder ---
    # FIXED(benchmark_audit D9, 2026-10-19): the default is the fixed list (was: every
    #   cache folder with a manifest, which made the assay set depend on the cache).
    if args.assays:
        assays = args.assays.split(',')
        assay_source = '--assays'
    elif args.all_cached:
        assays = sorted(d.name for d in CACHE_DIR.iterdir()
                        if d.is_dir() and (d / 'manifest.json').exists())
        assay_source = f'cache folders with manifest.json in {_rel(CACHE_DIR)}'
    else:
        lst = Path(args.assay_list) if args.assay_list else pg_common.ASSAY_LIST
        assays = pg_common.read_assay_list(lst)
        assay_source = f'--assay-list {lst.name} (sha256 {pg_common.file_sha256(lst)[:12]})'
    print(f'scoring {len(assays)} assays (from {assay_source})', flush=True)

    # Per-assay JSONs make us resumable.
    per_assay_dir = OUT_DIR / f'per_assay_pg_{args.out_tag}'
    per_assay_dir.mkdir(parents=True, exist_ok=True)

    # Queue of pending assays (skipping already-scored)
    pending = []
    for assay in assays:
        a_path = per_assay_dir / f'{assay}.json'
        if a_path.exists() and not args.force:
            rec = json.loads(a_path.read_text())
            try:
                check_recorded_checkpoint(rec.get('checkpoint'),
                                          ckpt_block, f'{per_assay_dir.name}/{a_path.name}')
            except ProvenanceMismatchError as e:
                print(f'[abort] {e}', file=sys.stderr); return 1
            # FIXED(benchmark_audit D9, 2026-10-19): a per-assay JSON of another schema
            #   version (e.g. the float32 pre-D9 files) is not resumed silently.
            if rec.get('pg_version') != PG_VERSION:
                print(f'[abort] {per_assay_dir.name}/{a_path.name} has pg_version '
                      f'{rec.get("pg_version")!r} (expected {PG_VERSION!r}); use a new '
                      f'--out-tag or --force', file=sys.stderr)
                return 1
            continue
        pending.append(assay)
    print(f'  {len(assays) - len(pending)} already done, {len(pending)} pending',
          flush=True)

    # ---- Score assays in parallel ---------------------------------------------
    # Multi-GPU: one worker per GPU, each loads the variant model + cbs once.
    gpus = [int(g) for g in args.gpus.split(',') if g.strip()]
    q = queue.Queue()
    for a in pending: q.put(a)

    lock = threading.Lock()
    counters = {'new': 0, 'skip': 0}

    def worker(gpu):
        device = f'cuda:{gpu}'
        ens, cbs = load(device)
        print(f'[GPU {gpu}] ready', flush=True)
        while True:
            try:
                assay = q.get(timeout=2)
            except queue.Empty:
                return
            try:
                t0 = time.time()
                res = evaluate_assay(assay, ens, cbs, out_tag=args.out_tag)
                if res is None:
                    print(f'[GPU {gpu}] SKIP {assay}', flush=True)
                    with lock: counters['skip'] += 1
                else:
                    res['checkpoint'] = ckpt_block          # FIXED(code_fix P17, 2026-10-16): new key (old: none)
                    # FIXED(benchmark_audit D9, 2026-10-19): atomic write.
                    pg_common.atomic_write_json(per_assay_dir / f'{assay}.json', res)
                    with lock: counters['new'] += 1
                    msg = (f'[GPU {gpu}] OK {assay}: (n={res["n_variants"]}, '
                           f'{time.time()-t0:.0f}s)')
                    if PRINT_SCORES:
                        msg += (f' a={res["spearman_score_a_allL0"]:+.3f} '
                                f'd={res["spearman_score_d_allLvl"]:+.3f} '
                                f'esm2={res["spearman_esm2"]:+.3f}')
                    print(msg, flush=True)
            except Exception as e:
                print(f'[GPU {gpu}] FAIL {assay}: {type(e).__name__} {e}',
                      flush=True)
                with lock: counters['skip'] += 1
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=False)
               for g in gpus]
    for t in threads: t.start()
    for t in threads: t.join()

    # ---- Aggregate ------------------------------------------------------------
    # Aggregate per-assay JSONs into one file
    # FIXED(require_fix #17, tex pass 2026-10-26): PG table updated with 256-D numbers.
    summary = {}
    for a in assays:
        p = per_assay_dir / f'{a}.json'
        if p.exists():
            try: summary[a] = json.loads(p.read_text())
            except Exception: pass
    out_path = OUT_DIR / f'per_assay_pg_{args.out_tag}.json'
    pg_common.atomic_write_json(out_path, summary)
    print(f'\nfinished: {counters["new"]} new / {counters["skip"]} skipped, '
          f'{len(summary)} total in aggregate', flush=True)
    print(f'wrote aggregate to {out_path}', flush=True)
    # The assay set made explicit (the paper's "96 assays"): which requested
    # assays ended up with a result. Kept out of the aggregate JSON, whose
    # values the compilers expect to be per-assay dicts.
    # FIXED(benchmark_audit D9, 2026-10-19): + assay list sha256 (requested and scored),
    #   structure set sha256 (over the per-assay structure digests), CSV set sha256,
    #   pg_version; atomic write.
    scored = sorted(summary)
    list_path = OUT_DIR / f'per_assay_pg_{args.out_tag}.assays_scored.json'
    pg_common.atomic_write_json(list_path, {
        'assays_scored': scored,
        'n_assays_scored': len(scored),
        'assays_requested': list(assays),
        'assays_not_scored': sorted(set(assays) - set(summary)),
        'assay_source': assay_source,
        'assay_list_sha256': pg_common.assay_list_sha256(assays),
        'assays_scored_sha256': pg_common.assay_list_sha256(scored),
        'structure_set_sha256': pg_common.structure_set_sha256(
            {a: summary[a].get('inputs', {}).get('structure_sha256') for a in scored}),
        'csv_set_sha256': pg_common.structure_set_sha256(
            {a: summary[a].get('inputs', {}).get('csv_sha256') for a in scored}),
        'cache_dir': _rel(CACHE_DIR),
        'pg_version': PG_VERSION,
        'checkpoint': ckpt_block})
    print(f'assays scored: {len(scored)} of {len(assays)} requested '
          f'(list in {list_path.name})', flush=True)

    # Summary statistics (only with --print-scores: lambda selection is blind)
    if summary and PRINT_SCORES:
        for key in ('spearman_score_a_allL0', 'spearman_score_b_mutL0',
                     'spearman_score_c_hamming', 'spearman_score_d_allLvl',
                     'spearman_esm2'):
            vals = [v[key] for v in summary.values()
                    if np.isfinite(v.get(key, float('nan')))]
            if vals:
                print(f'  mean({key}) = {np.mean(vals):.3f}  '
                      f'(n={len(vals)})', flush=True)
    return 0 if len(scored) == len(assays) else 2


if __name__ == '__main__': sys.exit(main())
