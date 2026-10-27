"""EC classification on MISATO for the baselines (Conv1d head on fixed per-residue features).

What it does
------------
  1. `load_features(spec)` builds one (L, D) float32 feature matrix per MISATO
     protein for a baseline: aa one-hot (21-D), ESM2, mini3di 10-D, 3Di tokens
     -> frozen 2-D mini3di centroid, ProtProfileMD_K8 (20-bin histogram
     interleaved with the centroids, 60-D), Vote_3di (2-D centroid), PCA
     eigenvalues, a random 2-D vector, the frozen codebook row of the
     AminoAseed / ProToken / ESM3struct token, the frame-pooled `*_pooled`
     features and the rebuttal `*_ourdata` tokenizers. No learnable embedding.
     It is imported by `conv1d_GO_misato.py` and `conv1d_binding_baselines.py`.
  2. EC numbers are truncated to depth 1, 2, 3; the classes are those seen in
     the train split (multi-label).
  3. `Conv1DClassifier` (3 x [Conv1d(k=5) + GELU + Dropout] -> masked mean+max
     pooling -> 2-layer MLP) is trained with BCE, AdamW lr 1e-3, batch 32, up
     to `--epochs` epochs, optional warmup / cosine schedule, early stopping on
     val BCE (`--patience`, `--min-epochs`); the best-val weights are scored on
     test: top-1 hit, macro / micro AP, macro / micro F1@0.5.
  4. One JSON per depth plus a summary JSON per (feature, split, tag, seed).

Pre-run fixes 2026-10-19 (camera_ready/function_reaudit.md; FIXED markers):
`load_features(spec, pids)` keeps the split-file complexes and builds `aa` from the
misato_real_bb sequences (old: the pairs pickle, 3,903 complexes); `load_features_checked`
also checks the cache's provenance sidecar (rmsf_common.check_cache_provenance; aa: current
MISATO parser; random: coords_ca8 sidecar) and requires a feature for every complex; the EC
items / classes come from `function_common.run_ec_depth` (classes from all labelled
training complexes, missing feature = error, sorted order, class list and item hashes);
validation metrics at the selected epoch in r['val']; Vote_3di =
mini3di_baselines.vote_from_frame_tokens; token ids out of range are an error (was clip);
result JSONs carry function_common.protocol_record (protocol tag, input records) and are
written atomically; test metrics are no longer printed.

Function CV 2026-10-19 (authors' decision option A; FIXED markers): `--cv-mode
{sequence,structure} --fold r` scores rotation r of the EC cross-validation
(data/misato_function_cv_v1.json, downstream/function_cv.py: distinct proteins, grouped
label-stratified 5 folds; test = fold r, val = fold r+1 mod 5); class lists checked against
the CV file; `function_cv` record in every JSON. `--split` is kept for the old runs.

EC/GO capacity check B 2026-10-19 (FIXED marker; non-paper row): feature
`aminoaseed_pca128` = the AminoAseed token's row of its 512 x 1024 codebook projected to
128-D by PCA fitted on the codebook rows (`codebook_pca`; label-free, deterministic), so
every token baseline feeds <= 128-D codebook vectors (ESM3struct and Ensembits are 128-D).
Input standardisation 2026-10-19 (authors' decision; FIXED markers; default for every
feature): lookup features (aa one-hot = rows of the 21 x 21 identity, the AminoAseed /
AminoAseed-PCA128 / ESM3struct / ProToken / *_ourdata / other codebooks, the mini3di centroids
of 3di_tokens and vote_3di) are looked up in their table standardised per dimension over its
rows (function_common.standardize_lookup_table; LOOKUP_SPECS); continuous features
(protprofile_K8 / K5, random, pca_eig, esm2, mini3di, *_pooled; CONTINUOUS_SPECS) are
standardised per dimension over the residues of the training items of each split / CV
rotation / EC depth (function_common.InputStandardisation.apply). `load_features_checked`
returns the method's InputStandardisation; every result JSON records `input_standardisation`.
`--no-input-standardisation` = raw features (old protocol, tag + "rawin"). The temporary
non-paper features `aminoaseed_std` / `esm3struct_std` of the EC/GO conditioning check were
removed (their rule is now the default of `aminoaseed` / `esm3struct`).

The same head and protocol as the Ensembits EC probe
(`downstream/tokenize_misato_and_eval.py`), except that the input is a fixed
feature instead of the frozen Ensembits codebook embedding.

Paper
-----
  * `appendix.tex` \\section{Baselines} \\label{app:baselines} (the feature of
    each baseline row);
  * `appendix.tex` \\label{app:downstreams}, \\paragraph{EC Function Prediction}
    (Table \\label{tab:ec_d1_structure}, baseline rows);
  * `appendix.tex` \\label{app:experiments}, \\subsection{EC Function
    Prediction} (depths 1-3, structure / sequence / random splits, 10 seeds);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar}).

Origin
------
Copied from `submission_exp/src/ec_classify_conv1d_misato.py`. Changed: `ROOT`
from `ensembits.paths.REPO_ROOT` (sys.path setup as in `downstream/`); the
usage lines with a personal interpreter path were replaced by "How to run";
comments and this docstring. Logic, defaults, file names and CLI flags are
unchanged. (The original docstring said the splits come from the pairs pickle;
the code read `data/misato_splits.json`; since benchmark_audit D3,
2026-10-18, `data/misato_splits_v2.json` = ensembits.paths.MISATO_SPLITS.)

Used by the reported model: no (baseline rows of the EC tables). Run by
`downstream/baselines/run_misato_baselines.py`.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/annotations/misato_ec_v2.json (D8, 2026-10-18; old: misato_ec.json),
          data/misato_splits_v2.json,
          data/cached_descriptors/<feature cache> (+ .provenance.json; see `FEATURE_CACHES`;
          built by data_prep/baselines/ and data_prep/build_coords_ca8_misato.py),
          data/cached_descriptors/misato_real_bb/<pid>.npz (aa)
          data/misato_function_cv_v1.json (+ sidecar; --cv-mode)
  writes  output/ec_conv1d_misato/<feature>[__<split>][__<tag>][__seed<s>]__d{1,2,3}.json
          output/ec_conv1d_misato/<feature>[__<split>][__<tag>][__seed<s>]__summary.json
          (no split suffix for the sequence split, no seed suffix for seed 0; with --cv-mode
          the split suffix is __cv_<mode>_f<r>)
          Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): <depth json stem>.preds.npz per EC depth
          (class scores of the selected model, labels, class list, val and test), written next to
          the JSON (downstream/protein_preds.py) and recorded with its sha256 in
          `per_protein_preds` (depth JSON and summary); read only by downstream/protein_level_stats.py.

How to run
----------
    python downstream/baselines/ec_classify_conv1d_misato.py --feature esm3struct \\
        --split structure --seed 0 --depths 1,2,3 \\
        --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine --device cuda:0
    # paper rows (2026-10-19): one rotation of the EC cross-validation instead of --split
    python downstream/baselines/ec_classify_conv1d_misato.py --feature esm3struct \\
        --cv-mode structure --fold 0 --seed 0 --depths 1,2,3 \\
        --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine --device cuda:0
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))            # tokenize_misato_*_eval.py
sys.path.insert(0, str(Path(__file__).resolve().parent))    # sibling probes in downstream/baselines/
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
# FIXED(benchmark_audit D8, 2026-10-18): old = data/annotations/misato_ec.json (per-PDB SIFTS
#   union). New = downstream/ec_labels.EC_LABELS (data/annotations/misato_ec_v2.json) through
#   ec_labels.load_ec_labels; every result JSON records "ec_labels" {path, sha256, schema}.
from ec_labels import EC_LABELS, load_ec_labels  # noqa: E402  downstream/ec_labels.py
PAIRS = ROOT / 'data' / 'cached_descriptors' / 'aa2ensembits_pairs_misato_v2.pkl'
CACHE_DIR = ROOT / 'data' / 'cached_descriptors'
OUT_DIR = ROOT / 'output' / 'ec_conv1d_misato'
# FIXED(benchmark_audit D3, 2026-10-18): old = data/misato_splits.json (leaky). New =
#   ensembits.paths.MISATO_SPLITS (data/misato_splits_v2.json) through
#   data_versions.load_misato_splits; every result JSON records misato_splits_record().
from ensembits.paths import MISATO_SPLITS  # noqa: E402
from ensembits.data_versions import load_misato_splits, misato_splits_record  # noqa: E402
# Shared function-probe rules (function_reaudit pre-run fixes, 2026-10-19).
from function_common import (protocol_record, require_features, run_ec_depth,  # noqa: E402
                             atomic_write_json, add_input_std_arg, input_std_tag)
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import PREDS_KEY, items_preds, write_function_preds  # noqa: E402
# EC / GO cross-validation (authors' decision option A, 2026-10-19).
from function_cv import (add_cv_args, check_cv_args, load_function_cv, cv_split_map,  # noqa: E402
                         cv_result_block, check_classes, cv_tag)

AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWYX'
AA_IDX = {a: i for i, a in enumerate(AA_VOCAB)}


def load_split_map(split_name: str = 'sequence') -> dict:
    """Returns {pid_upper: 'train'|'val'|'test'} for the given misato split."""
    s = load_misato_splits(MISATO_SPLITS)
    if split_name not in s:
        raise ValueError(f'unknown split {split_name!r} (have {list(s)})')
    out = {}
    for sp in ('train', 'val', 'test'):
        for pid in s[split_name].get(sp, []):
            out[pid.upper()] = sp
    return out


# FIXED(benchmark_audit D8, 2026-10-18): old = the first `depth` fields whatever they were,
#   so a partial EC number became a class of its own ('3.1.-.-' at depth 3 -> '3.1.-',
#   '3.-.-.-' at depth 2 -> '3.-'; the comment in build_label_sets promised to drop such
#   classes but nothing did). New = a partial EC number contributes at depth d only if its
#   first d fields are all specified; no class ever contains '-'. Identical copy in
#   downstream/tokenize_misato_and_eval.py.
def truncate_ec(ec, depth):
    """EC number truncated to `depth` fields, or None.

    '3.4.21.4' at depth 2 -> '3.4'. A partial (wildcard) EC number counts only at depths
    whose fields are all specified: '3.1.-.-' -> '3' (d1), '3.1' (d2), None (d3);
    '3.-.-.-' -> '3' (d1), None (d2, d3). None also if the number has fewer than `depth`
    fields. Preliminary numbers ('1.1.1.n12') are specified fields.
    """
    parts = str(ec).strip().split('.')
    if len(parts) < depth:
        return None
    head = parts[:depth]
    if any(p in ('-', '') for p in head):
        return None
    return '.'.join(head)


def build_label_sets(ec_raw: dict, depth: int) -> dict:
    """pid -> set of class strings at given depth (partial EC numbers only at depths whose
    fields are all specified, see truncate_ec; pids without any class dropped)."""
    out: dict = {}
    for pid, ec_list in ec_raw.items():
        if not ec_list:
            continue
        s: set = set()
        for ec in ec_list:
            t = truncate_ec(ec, depth)
            if t is None:      # too short, or a wildcard field at this depth (D8)
                continue
            s.add(t)
        if s:
            out[pid] = s
    return out


# Paper: appendix.tex app:baselines — one fixed per-residue feature per baseline.
# FIXED(function_reaudit pre-run fixes 1 / 2, 2026-10-19): `load_features(spec, pids)` keeps
#   only `pids` (default: every complex of the MISATO split file) and `aa` is the one-hot of
#   the misato_real_bb sequence (function_common.misato_aa_onehot; old: the pairs-pickle
#   sequence, 3,903 of 4,894 complexes, CYX -> X; the old branch is kept as
#   `_aa_from_pairs_pickle`, not called). `load_features_checked` adds the provenance check
#   of the feature cache (rmsf_common.check_cache_provenance, as the RMSF probes) and returns
#   the records written into every result JSON.
def load_features(spec: str, pids=None, table_fn=None) -> tuple:
    """({PID: (L, D) float32}, D) of one baseline for `pids` (default: all split complexes).

    table_fn(table, name) -> table: applied to the lookup table of a LOOKUP_SPECS feature
    before the row lookup (2026-10-19: the input standardisation, see _TableStd); None = raw."""
    from function_common import misato_function_pids, misato_aa_onehot
    keep = set(misato_function_pids(MISATO_SPLITS) if pids is None else
               (str(p).upper() for p in pids))
    if spec == 'aa':
        feats, _ = misato_aa_onehot(keep)
        return _aa_lookup(feats, table_fn), len(AA_VOCAB)
    feats, D = _load_features_all(spec, keep, table_fn)
    return {p: v for p, v in feats.items() if p in keep}, D


def _aa_lookup(feats: dict, table_fn=None) -> dict:
    """aa one-hot rows looked up in table_fn(21 x 21 identity) (one-hot @ table; raw if None).

    FIXED(function input standardisation, 2026-10-19): new; the aa one-hot is a lookup of the
    identity table, standardised over its 21 rows like every other lookup table."""
    if table_fn is None:
        return feats
    T = np.asarray(table_fn(np.eye(len(AA_VOCAB), dtype=np.float32),
                            f'aa one-hot ({len(AA_VOCAB)} x {len(AA_VOCAB)} identity)'),
                   dtype=np.float32)
    return {p: (x @ T).astype(np.float32) for p, x in feats.items()}


# FIXED(function input standardisation, authors' decision 2026-10-19): which rule each feature
#   gets. Lookup features = a frozen table indexed by a per-residue id (standardised over the
#   table rows, label free); continuous features = per-residue values (standardised over the
#   residues of the training items). A spec in neither set is refused when standardising.
LOOKUP_SPECS = frozenset({
    'aa', 'esm3struct', 'aminoaseed', 'aminoaseed_pca128', 'protoken', '3di_tokens', 'vote_3di',
    'aminoaseed_ourdata', 'vanillavq_ourdata', 'esm3style_ourdata', 'ensembits_ifold',
    'ensembits_ifold_L0', 'big_divonly_p10', 'big_divafs', 'aug_rot6d',
    'aminoaseed_retrained', 'esm3struct_retrained'})
CONTINUOUS_SPECS = frozenset({
    'protprofile_K8', 'protprofile_K5', 'random', 'pca_eig', 'esm2', 'mini3di',
    'esm3struct_pooled', 'aminoaseed_pooled', '3di_pooled', 'protoken_pooled'})
# Removed 2026-10-19 (now the default rule of the base feature).
REMOVED_SPECS = {'aminoaseed_std': 'aminoaseed', 'esm3struct_std': 'esm3struct'}


class _TableStd:
    """table_fn of load_features: standardise each lookup table over its rows, keep records."""

    def __init__(self):
        self.records = []

    def __call__(self, table, name):
        from function_common import standardize_lookup_table
        t, rec = standardize_lookup_table(table, name)
        self.records.append(rec)
        return t


def feature_input_std(spec: str, D: int, table_records=(), enabled: bool = True):
    """InputStandardisation of baseline `spec` (lookup: its table records; continuous: the
    training-residue rule). FunctionInputError for a spec of neither kind."""
    from function_common import InputStandardisation, FunctionInputError
    if not enabled:
        return InputStandardisation.disabled(spec)
    if spec in LOOKUP_SPECS:
        return InputStandardisation.lookup(spec, D, list(table_records))
    if spec in CONTINUOUS_SPECS:
        if table_records:              # a continuous feature must not come out of a table
            raise FunctionInputError(f'{spec}: continuous feature but a lookup table was '
                                     f'standardised')
        return InputStandardisation.continuous(spec, D)
    raise FunctionInputError(f'{spec}: no input-standardisation rule (add it to LOOKUP_SPECS or '
                             f'CONTINUOUS_SPECS)')


def _aa_from_pairs_pickle() -> tuple:
    """Old aa source (before 2026-10-19): one-hot of the pairs-pickle sequence (21-D incl. X;
    the paper says 20-D, require_fix #22); 3,903 complexes. Kept for reference only."""
    d = pickle.load(open(PAIRS, 'rb'))
    feats = {}
    for k, v in d.items():
        pid = k.split(':')[-1]
        seq = v['seq']
        x = np.zeros((len(seq), len(AA_VOCAB)), dtype=np.float32)
        for i, a in enumerate(seq):
            x[i, AA_IDX.get(a, AA_IDX['X'])] = 1.0
        feats[pid] = x
    return feats, len(AA_VOCAB)


# Feature caches of the paper baselines: spec -> (cache, frames, uses_sequence, codebook),
# the values their builders record in the provenance sidecar (as in probe_rmsf_misato._use).
CENTROIDS_FILE = CACHE_DIR / 'mini3di_centroids.npy'
FEATURE_CACHES = {
    'esm3struct': ('esm3struct_tokens_misato.npz', 'frame 0', False, 'esm3struct_codebook.npy'),
    'aminoaseed': ('aminoaseed_tokens_misato.npz', 'frame 0', False, 'aminoaseed_codebook.npy'),
    'protoken': ('protoken_tokens_misato.npz', 'frame 0', False, 'protoken_codebook.npy'),
    '3di_tokens': ('mini3di_tokens_misato.npz', 'frame 0', False, None),
    'vote_3di': ('mini3di_tokens_K8_misato.npz', 'all 8', False, None),
    'protprofile_K8': ('protprofile_K8_misato.npz', 'first 8 of bb_8', True, None),
    # FIXED(EC/GO capacity check B, 2026-10-19): new, non-paper row. Same token cache and
    #   codebook as 'aminoaseed' (same provenance check); the codebook is projected to 128-D
    #   by `codebook_pca` before the row lookup (see _load_features_all).
    'aminoaseed_pca128': ('aminoaseed_tokens_misato.npz', 'frame 0', False,
                          'aminoaseed_codebook.npy'),
    # (aminoaseed_std / esm3struct_std of the EC/GO conditioning check removed 2026-10-19: the
    #  per-row codebook standardisation is now the default of every lookup feature.)
    # Retrained static tokenizers (P-17/P-26/P-42, camera-ready).
    'aminoaseed_retrained': ('aminoaseed_retrained_tokens_misato.npz', 'frame 0', False,
                             'aminoaseed_retrained_codebook.npy'),
    'esm3struct_retrained': ('esm3struct_retrained_tokens_misato.npz', 'frame 0', False,
                             'esm3struct_retrained_codebook.npy'),
}
USES_CENTROIDS = ('3di_tokens', 'vote_3di', 'protprofile_K8')


def codebook_pca(cb: np.ndarray, k: int) -> tuple:
    """(projected codebook (n_codes, k) float32, explained-variance fraction) of a codebook.

    FIXED(EC/GO capacity check B, 2026-10-19): new; used only by the non-paper feature
    'aminoaseed_pca128' (capacity-matched AminoAseed input, so every token baseline feeds a
    codebook of <= 128 dimensions like ESM3struct and Ensembits). PCA is fitted on the
    codebook vectors themselves (every code weighted once, no token frequencies, no labels):
    centre the rows, SVD in float64, keep the top-k right singular vectors, and fix each
    component's sign so that its largest-|value| entry is positive (deterministic).
    No whitening: the projected rows keep the codebook's variance along each component.
    """
    X = np.asarray(cb, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k].copy()
    flip = np.sign(V[np.arange(k), np.abs(V).argmax(axis=1)])
    V *= flip[:, None]
    ev = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return (Xc @ V.T).astype(np.float32), ev


def feature_inputs(spec: str, required, allow_unverified: bool = False, coords=None) -> list:
    """Records of the input files of baseline `spec` (path, sha256, sidecar facts).

    Paper baselines with a builder in this codebase are verified (CacheError /
    FunctionInputError otherwise); any other spec (side experiments without a builder,
    *_pooled) only with allow_unverified, recorded as 'verified': False.
    """
    sys.path.insert(0, str(CODE_ROOT / 'downstream'))
    from rmsf_common import check_cache_provenance, file_record
    from function_common import check_coords_ca8, FunctionInputError
    if spec in FEATURE_CACHES:
        name, frames, uses_seq, cb = FEATURE_CACHES[spec]
        recs = [{**check_cache_provenance(CACHE_DIR / name, 'misato', frames=frames,
                                          uses_sequence=uses_seq,
                                          codebook=(CACHE_DIR / cb) if cb else None,
                                          required=sorted(required)), 'verified': True}]
        if spec in USES_CENTROIDS:
            recs.append({**file_record(CENTROIDS_FILE), 'verified': True})
        return recs
    if spec in ('random', 'pca_eig'):
        return [check_coords_ca8(CACHE_DIR / 'coords_ca8_misato.pkl', coords)]
    if not allow_unverified:
        raise FunctionInputError(f'{spec}: no provenance-checked cache in this codebase; pass '
                                 f'--allow-unverified-cache (result marked unverified)')
    files = sorted(CACHE_DIR.glob(f'{spec}*misato*'))
    return [{**file_record(f), 'verified': False} for f in files] or [
        {'file': spec, 'verified': False}]


def load_features_checked(spec: str, allow_unverified: bool = False,
                          standardise: bool = True) -> tuple:
    """(feats, D, input records, InputStandardisation) of baseline `spec` on every complex of
    the split file.

    Every complex must have a feature (function_common.require_features).
    FIXED(function input standardisation, 2026-10-19): new 4th value and `standardise`
    (default True; False = --no-input-standardisation): lookup features come out of their
    row-standardised table; continuous features come out raw and are standardised per item
    set by the returned rule's apply() (training residues only)."""
    from function_common import misato_function_pids, misato_aa_onehot, require_features
    if spec in REMOVED_SPECS:
        raise ValueError(f'{spec} was removed on 2026-10-19: --feature {REMOVED_SPECS[spec]} is '
                         f'now standardised by default')
    pids = misato_function_pids(MISATO_SPLITS)
    tstd = _TableStd() if standardise else None
    if standardise and spec not in LOOKUP_SPECS | CONTINUOUS_SPECS:   # before loading it
        from function_common import FunctionInputError
        raise FunctionInputError(f'{spec}: no input-standardisation rule (add it to '
                                 f'LOOKUP_SPECS or CONTINUOUS_SPECS)')
    if spec == 'aa':
        feats, rec = misato_aa_onehot(pids, verify=not allow_unverified)
        feats = _aa_lookup(feats, tstd)
        recs, D = [rec], len(AA_VOCAB)
    else:
        recs = feature_inputs(spec, pids, allow_unverified)
        feats, D = load_features(spec, pids, tstd)
    require_features(pids, feats, spec, 'split-file')
    std = feature_input_std(spec, D, tstd.records if tstd else (), enabled=standardise)
    return feats, D, recs, std


def _sel(pids, keep):
    """The pids of a cache to load (all when keep is None), in cache order."""
    return [p for p in pids if keep is None or p in keep]


def _load_features_all(spec: str, keep=None, table_fn=None) -> tuple:
    """The research-repo loader (every pid of the cache, or only `keep`; 2026-10-19: `keep`
    added so the K-frame vote is not computed for complexes outside the probe split).

    FIXED(function input standardisation, 2026-10-19): `table_fn(table, name)` is applied to the
    lookup table of every LOOKUP_SPECS branch before the row lookup (None = raw table)."""
    tf = table_fn if table_fn is not None else (lambda t, name: t)
    if spec == 'aa':
        feats, D = _aa_from_pairs_pickle()
        return _aa_lookup(feats, table_fn), D

    if spec == 'esm2':
        d = pickle.load(open(CACHE_DIR / 'esm2_650M_embeddings_misato.pkl', 'rb'))
        feats = {pid: np.asarray(v, dtype=np.float32) for pid, v in d.items()}
        D = next(iter(feats.values())).shape[1]
        return feats, D

    if spec == 'mini3di':
        d = np.load(CACHE_DIR / 'mini3di_features_misato.npz')
        return {pid: d[pid].astype(np.float32) for pid in _sel(d.files, keep)}, 10

    # Frame-pooled static-baseline features (Reviewer szcw #4): per-residue
    # embedding averaged over all 8 MD frames. Built by
    # build_pooled_baseline_features.py. spec = '{esm3struct,aminoaseed,3di}_pooled'.
    if spec.endswith('_pooled'):
        d = np.load(CACHE_DIR / f'{spec}_misato.npz')
        feats = {pid: d[pid].astype(np.float32) for pid in _sel(d.files, keep)}
        D = next(iter(feats.values())).shape[1]
        return feats, D

    if spec == 'protprofile_K5':
        d = np.load(CACHE_DIR / 'protprofile_K5_misato.npz')
        return {pid: d[pid].astype(np.float32) for pid in _sel(d.files, keep)}, 20

    if spec == 'protprofile_K8':
        # Per-residue 20-bin mini3di histogram from 8 frames, with each slot
        # annotated by its 2-D mini3di centroid coordinates. Per-residue feature:
        #   [p_0, c_0_x, c_0_y,  p_1, c_1_x, c_1_y,  ...,  p_19, c_19_x, c_19_y]
        # i.e. (L, 60) — gives the conv1d the centroid geometry alongside the
        # K-frame probability distribution at every residue.
        d = np.load(CACHE_DIR / 'protprofile_K8_misato.npz')
        cent = np.load(CACHE_DIR / 'mini3di_centroids.npy').astype(np.float32)  # (20, 2)
        feats = {}
        for pid in _sel(d.files, keep):
            hist = d[pid].astype(np.float32)  # (L, 20)
            L, M = hist.shape
            cb = np.broadcast_to(cent, (L, M, 2))           # (L, 20, 2)
            comb = np.concatenate([hist[..., None], cb], axis=-1)  # (L, 20, 3)
            feats[pid] = comb.reshape(L, M * 3)             # (L, 60)
        return feats, 60

    if spec == '3di_tokens':
        # Single-frame mini3di token per residue. Per-residue feature is a
        # FIXED 2-D centroid lookup (no learnable embedding). Symmetric with
        # protprofile_K8: just the 1-frame analog of probability + centroid
        # (here, since there's only one frame, the "probability" is one-hot,
        # so the only nonzero slot's centroid is what survives).
        # FIXED(function_reaudit, 2026-10-19): ids outside [0, 20) are an error (old: a zero
        #   vector), as in the RMSF probes (preretrain_audit_r2 H4).
        from rmsf_common import check_token_ids
        d = np.load(CACHE_DIR / 'mini3di_tokens_misato.npz')
        cent = np.load(CACHE_DIR / 'mini3di_centroids.npy').astype(np.float32)  # (20, 2)
        table = np.asarray(tf(cent, 'mini3di centroids'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = check_token_ids(d[pid], cent.shape[0], f'3di_tokens {pid}')
            feats[pid] = table[tok].astype(np.float32)
        return feats, cent.shape[1]   # 2

    # Paper: app:baselines "Vote_3di" (ties: nearest centroid here; the paper
    # says mean-pooled to a 21-D protein feature, require_fix #22).
    # FIXED(function_reaudit pre-run fix 7, 2026-10-19): the vote is
    #   ensembits.baselines.mini3di_baselines.vote_from_frame_tokens (strict plurality >= 2,
    #   else the centroid nearest to the mean in float64, lowest letter id on ties), the rule
    #   of the ANOVA and the RMSF probes. Old: the same rule inline in float32 with argmin
    #   (either of two exactly tied centroids), and ids clipped to [0, 20) instead of an error.
    if spec == 'vote_3di':
        from rmsf_common import check_token_ids
        from ensembits.baselines.mini3di_baselines import vote_from_frame_tokens
        d = np.load(CACHE_DIR / 'mini3di_tokens_K8_misato.npz')
        cent = np.load(CACHE_DIR / 'mini3di_centroids.npy').astype(np.float32)
        M = cent.shape[0]
        # 2026-10-19: the vote uses the raw centroids (its rule); the voted letter is then
        #   looked up in the (standardised) centroid table.
        table = np.asarray(tf(cent, 'mini3di centroids'), dtype=np.float32)
        feats = {}
        for pid in _sel(d.files, keep):
            tk = check_token_ids(d[pid], M, f'vote_3di {pid}')
            feats[pid] = table[vote_from_frame_tokens(tk, cent)]   # (L, 2)
        return feats, cent.shape[1]

    if spec == 'pca_eig':
        # Per residue across K=8 frames: 8 Cα positions (8, 3) → mean-center
        # → 3×3 covariance → 3 eigenvalues sorted descending → (3,) feature.
        # Captures per-residue motion amplitude.
        coords = pickle.load(open(CACHE_DIR / 'coords_ca8_misato.pkl', 'rb'))
        feats = {}
        for pid, ca in ((q, coords[q]) for q in _sel(coords, keep)):
            ca = np.asarray(ca, dtype=np.float32)   # (8, L, 3)
            P, L, _ = ca.shape
            mean = ca.mean(axis=0, keepdims=True)
            cen = ca - mean                           # (P, L, 3)
            # cov_i = cen[:, i, :].T @ cen[:, i, :] / P, shape (3, 3)
            cov = np.einsum('pli,plj->lij', cen, cen) / P  # (L, 3, 3)
            evals = np.linalg.eigvalsh(cov)           # (L, 3) ascending
            evals_desc = evals[:, ::-1].astype(np.float32)
            feats[pid] = evals_desc                   # (L, 3)
        return feats, 3

    # Paper: app:baselines "Random" is a random token in {0..K-1}; here a 2-D
    # Gaussian vector per residue (require_fix #22).
    if spec == 'random':
        # Deterministic per-residue random (2,) vector seeded from
        # hash((pid, residue_index)). Reproducible across head seeds.
        coords = pickle.load(open(CACHE_DIR / 'coords_ca8_misato.pkl', 'rb'))
        D = 2
        feats = {}
        for pid, ca in ((q, coords[q]) for q in _sel(coords, keep)):
            L = ca.shape[1]
            # Use a stable per-protein seed (FNV-1a 32-bit)
            h = 2166136261
            for ch in pid:
                h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            rng = np.random.default_rng(h)
            feats[pid] = rng.standard_normal((L, D)).astype(np.float32)
        return feats, D

    if spec == 'aminoaseed':
        # AminoAseed VQ token (frame 0) per residue → frozen 1024-D codebook
        # row lookup. No learnable embedding.
        from rmsf_common import check_token_ids   # function_reaudit 2026-10-19: was np.clip
        d = np.load(CACHE_DIR / 'aminoaseed_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'aminoaseed_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'aminoaseed codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = check_token_ids(d[pid], cb.shape[0], f'aminoaseed {pid}')
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'aminoaseed_pca128':
        # FIXED(EC/GO capacity check B, 2026-10-19): new, non-paper row. The AminoAseed
        #   token (frame 0) -> row of its 512 x 1024 codebook projected to 128-D by PCA
        #   fitted on the codebook rows (codebook_pca; label-free, deterministic). Frozen
        #   lookup, no learnable embedding, as 'aminoaseed'.
        from rmsf_common import check_token_ids
        d = np.load(CACHE_DIR / 'aminoaseed_tokens_misato.npz')
        cb_raw = np.load(CACHE_DIR / 'aminoaseed_codebook.npy').astype(np.float32)
        cb, ev = codebook_pca(cb_raw, 128)
        print(f'[aminoaseed_pca128] codebook {cb_raw.shape} -> {cb.shape}, '
              f'explained variance {ev:.4f}', flush=True)
        cb = np.asarray(tf(cb, 'aminoaseed codebook PCA-128'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = check_token_ids(d[pid], cb.shape[0], f'aminoaseed {pid}')
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec.endswith('_ourdata'):
        # Static tokenizers RETRAINED BY US on the Ensembits corpus (rebuttal:
        # "you didn't train the baselines on your data"). Same convention as the
        # released-checkpoint baselines below: token id per residue → frozen
        # codebook row lookup, no learnable embedding. Built by
        # rebuttal_analyses/sametrain_baselines/build_sametrain_tokens.py, which
        # reuses the released-baseline tokenization path verbatim.
        # spec ∈ {aminoaseed_ourdata, vanillavq_ourdata, esm3style_ourdata}.
        # FIXED(C13, 2026-10-23): *_ourdata baselines dropped from the paper (method paper;
        #   compare only with the released checkpoints). Code kept for reference.
        d = np.load(CACHE_DIR / f'{spec}_tokens_misato.npz')
        cb = np.load(CACHE_DIR / f'{spec}_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, f'{spec} codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = d[pid].astype(np.int64)
            tok = np.clip(tok, 0, cb.shape[0] - 1)
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'protoken':
        # ProToken VQ token (frame 0, single-bucket pick) per residue →
        # frozen 32-D codebook row lookup.
        from rmsf_common import check_token_ids   # function_reaudit 2026-10-19: was np.clip
        d = np.load(CACHE_DIR / 'protoken_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'protoken_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'protoken codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = check_token_ids(d[pid], cb.shape[0], f'protoken {pid}')
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'esm3struct':
        # ESM3 (sm-open-v1) structural VQ token per residue → frozen 128-D
        # codebook row lookup. (4096 codes.)
        from rmsf_common import check_token_ids   # function_reaudit 2026-10-19: was np.clip
        d = np.load(CACHE_DIR / 'esm3struct_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'esm3struct_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'esm3struct codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = check_token_ids(d[pid], cb.shape[0], f'esm3struct {pid}')
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec in ('aminoaseed_retrained', 'esm3struct_retrained'):
        # Static tokenizers retrained on our corpus (P-17/P-26/P-42, camera-ready).
        from rmsf_common import check_token_ids
        d = np.load(CACHE_DIR / f'{spec}_tokens_misato.npz')
        cb = np.load(CACHE_DIR / f'{spec}_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, f'{spec} codebook'), dtype=np.float32)
        feats = {}
        for pid in _sel(d.files, keep):
            tok = check_token_ids(d[pid], cb.shape[0], f'{spec} {pid}')
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'ensembits_ifold':
        d = np.load(CACHE_DIR / 'ensembits_ifold_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'ensembits_ifold_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'ensembits_ifold codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = d[pid].astype(np.int64)
            tok = np.clip(tok, 0, cb.shape[0] - 1)
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'big_divonly_p10':
        d = np.load(CACHE_DIR / 'big_divonly_p10_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'big_divonly_p10_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'big_divonly_p10 codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = d[pid].astype(np.int64)
            tok = np.clip(tok, 0, cb.shape[0] - 1)
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'big_divafs':
        d = np.load(CACHE_DIR / 'big_divafs_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'big_divafs_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'big_divafs codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = d[pid].astype(np.int64)
            tok = np.clip(tok, 0, cb.shape[0] - 1)
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'ensembits_ifold_L0':
        d = np.load(CACHE_DIR / 'ensembits_ifold_L0_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'ensembits_ifold_L0_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'ensembits_ifold_L0 codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = d[pid].astype(np.int64)
            tok = np.clip(tok, 0, cb.shape[0] - 1)
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    if spec == 'aug_rot6d':
        # Our rot6d-augmented RVQ-VAE primary token → frozen latent codebook
        # row (typically 128-D). Same multi-frame pipeline as 'ours full',
        # exposed as a baseline-style feature for sweep parity.
        d = np.load(CACHE_DIR / 'aug_rot6d_tokens_misato.npz')
        cb = np.load(CACHE_DIR / 'aug_rot6d_codebook.npy').astype(np.float32)
        cb = np.asarray(tf(cb, 'aug_rot6d codebook'), dtype=np.float32)   # 2026-10-19
        feats = {}
        for pid in _sel(d.files, keep):
            tok = d[pid].astype(np.int64)
            tok = np.clip(tok, 0, cb.shape[0] - 1)
            feats[pid] = cb[tok]
        return feats, cb.shape[1]

    raise ValueError(f'Unknown feature spec: {spec!r}')


class Conv1DClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int,
                 hidden: int = 128, depth: int = 3, kernel: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(depth):
            layers.append(nn.Conv1d(prev, hidden, kernel, padding=kernel // 2))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        m = mask.unsqueeze(-1).float()
        avg = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        very_neg = torch.full_like(x, float('-inf'))
        x_max = torch.where(mask.unsqueeze(-1), x, very_neg).max(1).values
        x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
        return self.head(torch.cat([avg, x_max], dim=-1))


class ProtProfileConv1D(nn.Module):
    """ProtProfileMD with mini3di centroid projection.

    Input: per-residue (L, 20) histogram over the mini3di alphabet.
    Architecture: a learnable Linear(20→2) layer initialized from the mini3di
    centroids (so per-residue feature = histogram @ learnable_centroids), then
    the same conv1d body as Conv1DClassifier.

    This makes the comparison apples-to-apples with the 3di_tokens baseline,
    which uses a learnable Embedding(20, 2) initialized from the same centroids.
    The two differ only in: 3di_tokens picks one of 20 tokens per residue, this
    averages 20 tokens (weighted by the K-frame histogram).
    """
    def __init__(self, centroids: torch.Tensor, n_classes: int,
                 hidden: int = 128, depth: int = 3, kernel: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        M, d_emb = centroids.shape   # (20, 2)
        # Linear(M -> d_emb): weight shape (d_emb, M); we want output = hist @ centroids,
        # so weight should be centroids.T (shape (d_emb, M)).
        self.proj = nn.Linear(M, d_emb, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(centroids.T)
        layers = []
        prev = d_emb
        for _ in range(depth):
            layers.append(nn.Conv1d(prev, hidden, kernel, padding=kernel // 2))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 20)
        x = self.proj(x)              # (B, L, 2)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        m = mask.unsqueeze(-1).float()
        avg = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        very_neg = torch.full_like(x, float('-inf'))
        x_max = torch.where(mask.unsqueeze(-1), x, very_neg).max(1).values
        x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
        return self.head(torch.cat([avg, x_max], dim=-1))


def collate(batch):
    feats = [b['x'] for b in batch]
    Y = torch.stack([torch.as_tensor(b['y'], dtype=torch.float32) for b in batch], 0)
    L_max = max(f.shape[0] for f in feats)
    D = feats[0].shape[1]
    X = torch.zeros((len(feats), L_max, D), dtype=torch.float32)
    M = torch.zeros((len(feats), L_max), dtype=torch.bool)
    for i, f in enumerate(feats):
        L = f.shape[0]
        X[i, :L] = torch.as_tensor(f)
        M[i, :L] = True
    return X, M, Y


def evaluate(model, items, batch_size, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for s in range(0, len(items), batch_size):
            batch = items[s:s + batch_size]
            X, M, Y = collate(batch)
            X, M = X.to(device), M.to(device)
            logits = model(X, M)
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(Y.numpy())
    p = np.concatenate(all_logits, 0)
    y = np.concatenate(all_y, 0)
    return p, y


# FIXED(benchmark_audit D7 / Q-metrics, 2026-10-18): `metrics` moved to
#   downstream/multilabel_metrics.py (one copy for EC and GO, Ensembits and baselines;
#   old: two identical-by-intent copies here and in tokenize_misato_and_eval.py). Same five metrics
#   as before; new: protein-centric Fmax (CAFA, thresholds 0.00-1.00 step 0.01, over all
#   classes of the task) in `fmax` / `fmax_threshold`.
from multilabel_metrics import metrics  # noqa: E402,F401


def _make_sched(opt, warmup, total, cosine):
    """Per-epoch LR multiplier: linear warmup then optional cosine decay to 0.
    No-op when warmup=0 & cosine=False (reproduces the original protocol)."""
    import math
    def lr_lambda(ep):
        if warmup > 0 and ep < warmup:
            return (ep + 1) / warmup
        if cosine and total > warmup:
            prog = (ep - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# Paper: appendix.tex app:downstreams EC — 3 Conv1D layers, masked mean+max
# pooling, MLP head, multi-label BCE; five test metrics (`metrics`).
def train_classifier(items_tr, items_va, items_te, n_classes, in_dim,
                     device, epochs=40, batch_size=32, lr=1e-3,
                     weight_decay=1e-4, patience=8, seed=0,
                     min_epochs=0, warmup=0, cosine=False, return_preds=False) -> dict:
    """Train one Conv1DClassifier (BCE, early stopping on val BCE), score val and test.
    return_preds: also return the selected model's val / test class scores and labels per
    item under the private key '_preds' (final_fix_list A4 (a), 2026-10-19)."""
    torch.manual_seed(seed)
    model = Conv1DClassifier(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _make_sched(opt, warmup, epochs, cosine)
    g = torch.Generator(device='cpu').manual_seed(seed)
    n_tr = len(items_tr)
    best_va = float('inf'); best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=g).tolist()
        for s in range(0, n_tr, batch_size):
            batch = [items_tr[i] for i in perm[s:s + batch_size]]
            X, M, Y = collate(batch)
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(X, M), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        # val loss
        model.eval()
        va_loss = 0.0; n = 0
        with torch.no_grad():
            for s in range(0, len(items_va), batch_size):
                batch = items_va[s:s + batch_size]
                X, M, Y = collate(batch)
                X, M, Y = X.to(device), M.to(device), Y.to(device)
                l = F.binary_cross_entropy_with_logits(model(X, M), Y)
                va_loss += l.item() * X.shape[0]; n += X.shape[0]
        va_loss = va_loss / max(n, 1)
        sched.step()
        if va_loss < best_va - 1e-4:
            best_va = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep + 1 >= min_epochs and no_imp >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    p, y = evaluate(model, items_te, batch_size, device)
    m = metrics(p, y)
    m.update({'best_val_loss': best_va, 'n_train': len(items_tr),
              'n_val': len(items_va), 'n_test': len(items_te),
              'n_classes': int(n_classes), 'epochs_run': ep + 1})
    # FIXED(function_reaudit pre-run fix 5, 2026-10-19): new: the same metrics on the
    #   validation split with the selected (best validation BCE) weights, in m['val'].
    pv, yv = evaluate(model, items_va, batch_size, device)
    m['val'] = {**metrics(pv, yv), 'best_val_loss': best_va, 'n_val': len(items_va)}
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-item scores of the selected model.
    if return_preds:
        m['_preds'] = {'val': items_preds(items_va, pv, yv), 'test': items_preds(items_te, p, y)}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', required=True,
                    help='aa | esm3struct | aminoaseed | protoken | 3di_tokens | vote_3di | '
                         'protprofile_K8 | random | aminoaseed_pca128 (non-paper capacity '
                         'check) | ... (see load_features; inputs standardised by default)')
    ap.add_argument('--depths', default='1,2,3')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--split', default='sequence',
                    choices=['sequence', 'structure', 'random'],
                    help='fixed v2 split (old protocol; the paper rows use --cv-mode / --fold)')
    # FIXED(function CV, authors' decision option A, 2026-10-19): new --cv-mode / --fold
    #   (distinct proteins, grouped label-stratified 5-fold CV; downstream/function_cv.py).
    add_cv_args(ap)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--allow-unverified-cache', action='store_true',
                    help='accept a feature cache without a current provenance sidecar (the '
                         'result is marked unverified; compilers refuse it)')
    add_input_std_arg(ap)              # FIXED(function input standardisation, 2026-10-19)
    args = ap.parse_args()
    is_cv = check_cv_args(ap, args)
    input_std_tag(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[setup] feature={args.feature} '
          + (f'cv={args.cv_mode} fold={args.fold}' if is_cv else f'split={args.split}'),
          flush=True)
    # FIXED(function_reaudit pre-run fixes 1 / 2, 2026-10-19): features of every split-file
    #   complex, provenance-checked (old: load_features, no check, aa on 3,903 complexes).
    # FIXED(function input standardisation, 2026-10-19): + the feature's input rule (lookup:
    #   table standardised over its rows here; continuous: per EC depth over its training items).
    feats, in_dim, feat_recs, in_std = load_features_checked(
        args.feature, args.allow_unverified_cache, standardise=not args.no_input_standardisation)
    print(f'[setup] {len(feats)} proteins, in_dim={in_dim}; input-std: '
          f'{in_std.base_record()["rule"]}', flush=True)
    split_rec = misato_splits_record(MISATO_SPLITS)   # D3 (2026-10-18)
    cv_block = {}
    if is_cv:                                         # function CV (2026-10-19)
        cv, cv_rec = load_function_cv()
        split_map = cv_split_map(cv, 'ec', args.cv_mode, args.fold)
        cv_block = cv_result_block(cv_rec, cv, 'ec', args.cv_mode, args.fold)
    else:
        split_map = load_split_map(args.split)
    ec_raw, ec_rec = load_ec_labels(EC_LABELS)        # FIXED(benchmark_audit D8, 2026-10-18)
    summary = {'feature': args.feature,
               'split': f'cv/{args.cv_mode}/fold{args.fold}' if is_cv else args.split,
               'cv_mode': args.cv_mode, 'cv_fold': args.fold, 'in_dim': in_dim,
               'by_depth': {}, **split_rec, **cv_block, 'ec_labels': ec_rec,
               **protocol_record(feat_recs, input_std=in_std.run_record())}
    # Output stem: <feature>[__cv_<mode>_f<r> | __<split>][__<tag>][__seed<s>]
    sp_tag = (f'__{cv_tag(args.cv_mode, args.fold)}' if is_cv else
              '' if args.split == 'sequence' else f'__{args.split}')

    for depth in [int(x) for x in args.depths.split(',')]:
        r = run_ec_depth(depth, build_label_sets(ec_raw, depth), split_map, feats, args.feature,
                      lambda b_tr, b_va, b_te, n_cls: train_classifier(
                          b_tr, b_va, b_te, n_cls, in_dim, args.device,
                          epochs=args.epochs, batch_size=args.batch_size,
                          seed=args.seed, patience=args.patience,
                          min_epochs=args.min_epochs, warmup=args.warmup,
                          cosine=args.cosine, return_preds=True),
                      input_std=in_std)                # 2026-10-19: per-depth training stats
        preds = r.pop('_preds', None)                  # A4 (a): sidecar below
        r['seed'] = args.seed
        summary['by_depth'][depth] = r
        if 'error' in r:
            continue
        if is_cv:                                     # classes must equal the CV file's
            check_classes(cv, 'ec', args.cv_mode, args.fold, r['ec_classes'], depth)
        seed_tag = '' if args.seed == 0 else f'__seed{args.seed}'
        out_path = OUT_DIR / f'{args.feature}{sp_tag}{("__"+args.tag) if args.tag else ""}{seed_tag}__d{depth}.json'
        # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar <out>.preds.npz
        #   (class scores of the selected model + labels, val and test) in `per_protein_preds`.
        r[PREDS_KEY] = write_function_preds(
            out_path, preds, r, task='EC', classes=r['ec_classes'], method=args.feature,
            split=summary['split'], cv_mode=args.cv_mode, cv_fold=args.fold)
        atomic_write_json(out_path, {**r, **split_rec, **cv_block, 'ec_labels': ec_rec,   # D3 / D8 / CV
                                     **protocol_record(feat_recs,
                                                       input_std=r['input_standardisation'])})
        # (test metrics are not printed; 2026-10-19)
        print(f'  d{depth}: [ep={r["epochs_run"]}, {r["runtime_s"]}s, '
              f'n_test={r["n_test"]}/{r["n_classes"]} cls] wrote {out_path.name}', flush=True)

    seed_tag = '' if args.seed == 0 else f'__seed{args.seed}'
    summary['seed'] = args.seed
    # 2026-10-19: top-level rule record over the per-depth statistics (continuous features)
    summary['input_standardisation'] = in_std.run_record(
        [(f'd{k}', v['input_standardisation']) for k, v in summary['by_depth'].items()
         if 'input_standardisation' in v])
    summary_path = OUT_DIR / f'{args.feature}{sp_tag}{("__"+args.tag) if args.tag else ""}{seed_tag}__summary.json'
    atomic_write_json(summary_path, summary)
    print(f'[done] wrote {summary_path}', flush=True)


if __name__ == '__main__':
    main()
