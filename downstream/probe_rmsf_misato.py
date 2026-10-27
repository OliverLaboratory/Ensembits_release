"""RMSF probe on MISATO for one tokenizer or one baseline feature (per-residue MLP).

What it does
------------
  1. Loads the per-residue RMSF label of every MISATO protein
     (`data/cached_descriptors/misato_rmsf_allframes.npz`; `--rmsf-label frames8`:
     `misato_rmsf.npz`) and one split of the v2 split file
     `data/misato_splits_v2.json` (ensembits.paths.MISATO_SPLITS, benchmark_audit D3;
     `--split sequence | structure | random`). The v2 splits replace the original
     ProteinShake splits (D3 fix, 2026-10-18).
  2. Builds per-residue inputs for the chosen `--mode`:
       * `variant`: the tokenizer's level-1 tokens from the MISATO token cache
         `data/cached_descriptors/ensembits_<tag>[_P<n>inf]_tokens_misato.npz`
         (for the reported model this cache is written beforehand by
         `downstream/_pretokenize_misato_dihedral.py`, P = 8 real-backbone
         frames, or the first frame with `--p-inference 1`), embedded with the
         frozen level-1 codebook;
       * baselines: amino-acid one-hot, mini3di tokens, ProtProfileMD_K8,
         Vote_3Di, AminoAseed / ProToken / ESM3struct codebook lookups, a random
         vector, and pca_eig (diagnostic); the rebuttal-only / side-experiment modes were
         removed on 2026-10-17 (G14).
  3. Pools residues of all train proteins, fits an MLP (128 -> 128 -> 1, GELU,
     dropout 0.1) with MSE, AdamW lr 3e-4, weight decay 1e-4, batch 2048.
     Training runs until convergence: after each epoch it computes the Spearman
     on the pooled val residues and stops when that has not improved by more
     than 1e-4 for `--patience` epochs (default 20), or after `--max-epochs`
     (default 500). The model weights of the best validation epoch are kept.
  4. Evaluates that best-validation model once on the pooled residues of the
     split's TEST proteins; `test_spearman` is the headline metric.
     (Before 2026-10-15 the probe ran a fixed 40 epochs and reported the max
     over epochs of the val Spearman, `best_val_spearman`, with no test
     evaluation; code_fix §7.)
  5. Writes one JSON per (mode/variant, split, P, seed) to
     `output/probe_rmsf_misato/`, with the `__<tag>` suffix (default
     `__test`) so the old best-val JSONs used by the paper are not overwritten.

Since 2026-10-16 (preretrain_audit_r2 Q2 / Q3 / Q11 / Q20, user decisions):
  * label `--rmsf-label allframes` (default; all 100 MD frames, converged superposition,
    `misato_rmsf_allframes.npz`, sidecar required) or `frames8` (old 8-frame label);
  * every mode is trained / validated / tested on the common residue set
    (`data/cached_descriptors/rmsf_residue_set_misato.json`); a complex of the set
    without a feature, or a feature / label length mismatch, is an error (was: cut to
    the shorter, absent complexes dropped); `--residue-set all` = legacy coverage;
  * `aa` reads `misato_real_bb/<pid>.npz` `seq` (was the old pairs pickle: 3,903 of
    4,894 complexes, disulfide Cys as X);
  * per-protein Spearman of the selected model on val and test, next to the pooled one;
  * the coordinates hash used by the frames8 provenance check is recomputed from the
    file (the sidecar hash was trusted).
  * (F1 / H1 / H4 / H6 / H7) baseline caches and the Ensembits token cache are used only
    with a current provenance sidecar (current MISATO parser, K = 8, Kabsch FPS, frames,
    chain split, sequence source); their sha256 goes into `feature_caches`; 3Di ids
    outside [0, 20) are errors; an existing result is skipped only when it matches,
    refused otherwise; an unfinished checkpoint gives one [error] line.

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool encode flexibility?}
    \\label{result:rmsf}: the MISATO (seq) and MISATO (struct) columns of Table
    \\label{tab:rmsf} (P=8 frames; single-frame MLP head on the pooled training
    residues; "Spearman ... between predicted and ground-truth RMSF on held-out
    residues" = `test_spearman` on the split's test proteins, epoch selected on
    val; mean over 10 seeds, looped by `_run_model_battery.py`).
  * `appendix.tex`, \\section{Baselines} \\label{app:baselines}: aa, 3Di_tokens,
    Random, \\tool P=1, AminoAseed, ProToken, ESM3struct, ProtProfileMD_K (K=8 on
    MISATO), Vote_3di, \\tool P=full (P=8 on MISATO).
  * `appendix.tex`, \\subsection{Compute Requirements} \\label{sec:compute}:
    "RMSF probe (one split, per-residue MLP, 40 ep)".
    (require_fix #20, resolved in the text: early stopping, epoch selected on
    validation, Spearman reported on the test split.)

Origin
------
Copied from `submission_exp/src/probe_rmsf_misato.py`. Changes: ROOT is now
`ensembits.paths.REPO_ROOT`; the three `sys.path.insert(... ROOT /
'submission_exp' / 'src')` lines (module top, `build_variant_tokens`, `main`)
now insert this script's own folder, so `tokenize_misato_and_eval` resolves to
the copy in `downstream/`; comments and this docstring.
Logic changed on 2026-10-15 (approved by the authors, code_fix §7 / require_fix
#20): `train_and_eval` now does early stopping on the val Spearman, restores the
best-val weights and scores the test split; every mode passes the test residues;
new flags `--max-epochs` (alias `--epochs`), `--patience`, `--tag`; output names
get the `__<tag>` suffix; the JSON gains `test_spearman` and the selection
fields. Seeding is unchanged.
Logic changed on 2026-10-16 (code_fix P25, P28, P17, P12): torch / numpy seeded
at the start of main() (the probe's initial weights were unseeded); the label's
provenance sidecar is checked against coords_ca8_misato.pkl (refused on a
mismatch, loud warning without provenance); the variant JSON carries a
`checkpoint` block; `--allow-unfinished`. 2026-10-16 (later): see the list above
(FIXED markers; main() takes argv and returns the exit status).
2026-10-19 (function_reaudit pre-run fix 7): Vote_3Di = mini3di_baselines.vote_from_frame_tokens
(float64, lowest letter id on ties; the ANOVA's rule); vote_3di results record
rmsf_common.VOTE_RULE and older ones are refused / not reused.

Used by the reported model: yes (`--mode variant`, sequence and structure
splits, P=full and P=1, seeds 0-9). The baseline modes produce the baseline rows
of the same table.

Inputs / outputs
----------------
Reads (relative to REPO_ROOT):
  data/cached_descriptors/misato_rmsf_allframes.npz (+ sidecar)  {PID: (L,) RMSF}
      (--rmsf-label frames8: misato_rmsf.npz)
  data/cached_descriptors/rmsf_residue_set_misato.json        common residue set
  data/cached_descriptors/misato_real_bb/<PID>.npz            seq (aa mode)
  data/misato_splits_v2.json (MISATO_SPLITS, D3)             {split: {train, val, test}}
  data/cached_descriptors/ensembits_<tag>[_P<n>inf]_tokens_misato.npz   (variant mode)
  output/<variant>/{config.json,best.pt}                      (variant mode, codebook)
  data/cached_descriptors/{aa2ensembits_pairs_misato_v2.pkl, mini3di_tokens_misato.npz,
      mini3di_tokens_K8_misato.npz, mini3di_centroids.npy, protprofile_K8_misato.npz,
      coords_ca8_misato.pkl, <kind>_tokens_misato.npz, <kind>_codebook.npy}   (baselines)
Writes:
  output/probe_rmsf_misato/<vtag>[_P<n>inf][_<split>][_seed<s>]__<tag>.json   (variant)
  output/probe_rmsf_misato/baseline_<mode>[_<split>][_seed<s>]__<tag>.json   (baselines)
  Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): next to each JSON,
  <json stem>.preds.npz (downstream/protein_preds.py: val / test predictions of the
  selected epoch, labels, protein ids, residue index), recorded with its sha256 in
  the JSON's `per_protein_preds`; read only by downstream/protein_level_stats.py, which
  refuses results without it. --redo-missing-preds recomputes an existing result that
  lacks it (otherwise it is skipped with a note).
  (no split suffix = sequence split; <vtag> = tag_of(variant); <tag> = --tag,
  default "test". The files WITHOUT the `__<tag>` suffix are the old
  fixed-40-epoch best-val results behind the current paper table; they are
  never written by this version.)
  JSON fields: test_spearman (headline), best_val_spearman (val Spearman at the
  selected epoch), best_epoch (1-based), n_epochs_run, max_epochs, patience,
  min_delta, selection = "val_early_stopping", split, n_{train,val,test}_residues,
  {val,test}_spearman_per_protein_{mean,median}, n_{val,test}_proteins[_scored,
  _constant_pred], rmsf_label, rmsf_label_file, dataset, residue_set*,
  label_provenance, plus the old fields (variant, p_inference, M, d_emb / feature,
  in_dim; seed; checkpoint).

Dependencies left in the research repo: none in code; all data above lives
under REPO_ROOT.

How to run
----------
    python downstream/probe_rmsf_misato.py --mode variant \\
        --variant dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
        --split structure --seed 0 --device cuda:0 [--p-inference 1] \\
        [--max-epochs 500 --patience 20 --tag test]
    python downstream/probe_rmsf_misato.py --mode aminoaseed \\
        --split sequence --seed 0 [--rmsf-label allframes] [--residue-set common]
(the residue set must exist: data_prep/build_rmsf_residue_sets.py --dataset misato)
`--epochs N` is still accepted as an alias of `--max-epochs N` (old command
lines keep working; `--epochs 40` would cap early stopping at 40 epochs;
`_run_model_battery.py` no longer passes it).
`--redo-missing-preds` recomputes an existing result that lacks a per-protein
prediction sidecar (otherwise it is skipped with a note); required for the
significance tests of `downstream/protein_level_stats.py`.
"""
from __future__ import annotations
import os
# Cap BLAS threads: many probes run in parallel on one node.
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr

# Codebase root and this script's folder on sys.path (sibling imports such as
# tokenize_misato_and_eval resolve to downstream/); data/ and output/ stay in
# the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = ROOT / 'data' / 'cached_descriptors'
# Common residue set, per-protein Spearman (preretrain_audit_r2 Q2 / Q3, 2026-10-16).
from rmsf_common import (MISATO_REAL_BB_DIR, RMSF_LABEL_FILES, RMSF_LABEL_DEFAULT,  # noqa: E402
                         PER_PROTEIN_MIN_RESIDUES, ResidueCoverageError, per_protein_fields,
                         per_protein_spearman, resolve_residue_set, CacheError,
                         check_cache_provenance, check_token_ids, check_label_hash,
                         file_record, records_changed, strict_same_checkpoint,
                         check_misato_inputs, splits_record, misato_real_bb_scan,
                         constant_prediction_fields, VOTE_RULE, vote_rule_problem)
# Records of the feature caches read by the builders below (F1); reset by main().
_CACHES: list = []
_VERIFY = True
_REQUIRED = None


def _use(path, codebook=None, frames='frame 0', uses_sequence=False):
    """FIXED(preretrain_audit_r2 F1, 2026-10-16): a baseline cache is read only with a
    current provenance sidecar (current MISATO parser; rmsf_common.check_cache_provenance);
    its sha256 goes into the result JSON. Old: accepted by file name. G7 (2026-10-17): the
    proteins of the residue set (_REQUIRED, set by main()) may not come from ideal frames."""
    if _VERIFY:
        _CACHES.append(check_cache_provenance(path, 'misato', codebook=codebook, frames=frames,
                                              uses_sequence=uses_sequence, required=_REQUIRED))
    else:
        _CACHES.append({**file_record(path), 'verified': False})
    return path


def _use_coords(path):
    """coords_ca8_misato.pkl (pca_eig / random): its build_coords_ca8_misato.py sidecar must
    name the current parser only and match the content."""
    from ensembits.data_versions import MISATO_PARSER_VERSION, read_provenance
    if _VERIFY:
        prov = read_provenance(path)
        if prov is None or prov.get('parser_version') != [MISATO_PARSER_VERSION]:
            raise CacheError(f'{path.name}: no sidecar or not from parser {MISATO_PARSER_VERSION}')
        if prov.get('coords_sha256') not in _hashes_of_coords_file(path):
            raise CacheError(f'{path.name} does not match its sidecar')
    _CACHES.append({**file_record(path), **({} if _VERIFY else {'verified': False})})
    return path
# Per-residue RMSF label {pid: (L,)}, built by data_prep/build_misato_rmsf_labels.py:
# default (allframes) all 100 MD frames with the converged superposition; the old label
# (frames8) superposes the 8 frames of coords_ca8_misato.pkl on frame 0 and takes the
# per-residue Cα fluctuation sqrt(sum_xyz var over the 8 frames).
# (require_fix #21, resolved in the text: MISATO has no 320 K replicas; the label is the
#   RMSF over the frames of one trajectory, all 100 since 2026-10-16, was 8.)
# FIXED(user decision 2026-10-16, all-frames labels): old = one label file
#   (misato_rmsf.npz, 8 frames superposed on frame 0). New = --rmsf-label: "allframes"
#   (default; all 100 frames of MD.hdf5, converged superposition, misato_rmsf_allframes.npz)
#   or "frames8" (the old file, only to reproduce earlier numbers).
RMSF_LABEL_CHOICES = RMSF_LABEL_FILES['misato']
RMSF_LABELS = RMSF_LABEL_CHOICES['frames8']     # old default (kept for check_label_provenance)
# FIXED(benchmark_audit D3, 2026-10-18): old = data/misato_splits.json; new = ensembits.paths.MISATO_SPLITS
#   (data/misato_splits_v2.json) read through data_versions.load_misato_splits (v2 schema).
from ensembits.paths import MISATO_SPLITS as SPLITS  # noqa: E402
from ensembits.data_versions import load_misato_splits  # noqa: E402
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import (PREDS_KEY, needs_rerun, write_preds, sidecar_path,  # noqa: E402
                           residue_split_from_residues, contiguous_residue_index,
                           not_selected_record)
from function_common import atomic_write_json  # noqa: E402
PAIRS = CACHE / 'aa2ensembits_pairs_misato_v2.pkl'
TOKENS_3DI = CACHE / 'mini3di_tokens_misato.npz'
CENTROIDS = CACHE / 'mini3di_centroids.npy'
OUT_DIR = ROOT / 'output' / 'probe_rmsf_misato'
# 8-frame Cα coordinates the label (and the CA-based baseline features) come from.
COORDS_CA8 = CACHE / 'coords_ca8_misato.pkl'


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + '.provenance.json')


def _coords_hash_of_label(prov: dict):
    """Hash of the coordinates file recorded in the label's provenance (several key spellings)."""
    for k in ('coords_sha256', 'coords_hash', 'source_coords_sha256'):
        if prov.get(k):
            return prov[k]
    c = prov.get('coords')
    if isinstance(c, dict):
        return c.get('sha256') or c.get('coords_sha256')
    return None


def _hashes_of_coords_file(coords_path: Path) -> dict:
    """Hashes of the coordinates file, computed from its content: {hash: what it is}.

    FIXED(preretrain_audit_r2 Q20 / R2-C10, 2026-10-16): old = the hash recorded in
    coords_ca8_misato.pkl.provenance.json was trusted (a stale sidecar next to a rebuilt
    pickle would pass). New = both hashes are recomputed from the file: the sha256 of the
    bytes and `hash_arrays_by_pid` of the loaded {pid: array} (the hash the label builder
    records as coords_sha256). A sidecar hash that matches neither is reported as stale
    and not used.
    """
    out = {}
    if not coords_path.exists():
        return out
    from ensembits.featurize import file_sha256
    from ensembits.data_versions import hash_arrays_by_pid
    out[file_sha256(coords_path)] = f'sha256({coords_path.name})'
    with open(coords_path, 'rb') as f:
        out.setdefault(hash_arrays_by_pid(pickle.load(f)), f'content hash of {coords_path.name}')
    side = _sidecar(coords_path)
    if side.exists():
        d = json.loads(side.read_text())
        for k in ('sha256', 'coords_sha256', 'output_sha256', 'file_sha256'):
            if d.get(k) and d[k] not in out:
                print(f'[WARNING] {side.name}:{k} = {str(d[k])[:12]} does not match the content of '
                      f'{coords_path.name}; stale sidecar, ignored', flush=True)
    return out


def check_label_provenance(labels_path: Path = RMSF_LABELS, coords_path: Path = COORDS_CA8,
                           label_name: str = 'frames8') -> str:
    """Refuse an RMSF label built from other coordinates than the features' coordinates file.

    label_name "allframes" (user decision 2026-10-16): the label comes from MD.hdf5, not
    from coords_ca8; its sidecar is required and must name label "allframes" and the
    current MISATO parser version (refused otherwise). Returns "allframes-ok".
    "frames8" (the old label): the check below.

    FIXED(code_fix P28, 2026-10-16): new. If `<labels>.provenance.json` exists, the coords
    hash it records must equal the hash of `coords_ca8_misato.pkl` (from
    coords_ca8_misato.pkl.provenance.json, else computed from the file); a mismatch raises
    ValueError. Without label provenance a loud warning is printed (labels built before
    2026-10-16 carry none; the old label was stale, preretrain_audit P4).
    Returns "match", "no-provenance" or "no-coords-hash".
    """
    side = _sidecar(labels_path)
    if label_name == 'allframes':
        from ensembits.data_versions import MISATO_PARSER_VERSION
        if not side.exists():
            raise ValueError(f'{labels_path.name} has no provenance sidecar; rebuild it with '
                             f'data_prep/build_misato_rmsf_labels.py')
        prov = json.loads(side.read_text())
        if prov.get('label_name') != 'allframes':
            raise ValueError(f'{side.name} names label {prov.get("label_name")!r}, not allframes')
        if prov.get('parser_version') != MISATO_PARSER_VERSION:
            raise ValueError(f'{labels_path.name} was built with MISATO parser '
                             f'{prov.get("parser_version")!r}, expected {MISATO_PARSER_VERSION!r}')
        print(f'[labels] provenance ok: {labels_path.name} (all frames, parser '
              f'{MISATO_PARSER_VERSION})', flush=True)
        return 'allframes-ok'
    if not side.exists():
        bar = '!' * 78
        print(f'{bar}\n[WARNING] {labels_path.name} has no provenance sidecar ({side.name}); '
              f'cannot check that the label was built from {coords_path.name}.\n'
              f'  Rebuild it with data_prep/build_misato_rmsf_labels.py (code_fix P4 / P28).\n{bar}',
              flush=True)
        return 'no-provenance'
    rec = _coords_hash_of_label(json.loads(side.read_text()))
    if rec is None:
        print(f'[WARNING] {side.name} records no coords hash; label provenance not checked',
              flush=True)
        return 'no-coords-hash'
    cur = _hashes_of_coords_file(coords_path)
    if rec not in cur:
        raise ValueError(f'{labels_path.name} was built from coordinates with sha256 '
                         f'{str(rec)[:12]}, but {coords_path.name} has '
                         f'{[h[:12] for h in cur] or "no file"}; rebuild the label or the '
                         f'coordinates so they match')
    print(f'[labels] provenance ok: {labels_path.name} built from {coords_path.name} '
          f'({cur[rec]})', flush=True)
    return 'match'

# Amino-acid alphabet for the `aa` baseline: 20 standard residues + X (unknown).
# (require_fix #22, resolved in the text: 21 dimensions, the extra X slot.)
AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWYX')
AA_IDX = {a: i for i, a in enumerate(AA_VOCAB)}


# ---- Split and residue pooling ----------------------------------------------

def load_split_map(split_name='sequence'):
    # pid (upper case) -> 'train' | 'val' | 'test' for one ProteinShake split.
    s = load_misato_splits(SPLITS)                   # D3: v2 schema required
    out = {}
    for sp in ('train', 'val', 'test'):
        for pid in s[split_name].get(sp, []):
            # FIXED(preretrain_audit_r2 G4, 2026-10-17): old = a pid listed in two parts was
            #   silently assigned to the last one; new = error.
            if pid.upper() in out and out[pid.upper()] != sp:
                raise ResidueCoverageError(f'{pid} is in {out[pid.upper()]} and {sp} of the '
                                           f'{split_name} split')
            out[pid.upper()] = sp
    return out


def gather_residues(per_pid_feats, rmsf_per_pid, split_map, residue_set=None):
    """Stack per-residue (feature, RMSF, protein index) triples grouped by split.

    Returns {split: (X, y, groups)} ((None, None, None) for an empty split); groups =
    protein index of every residue (per-protein Spearman).
    With `residue_set` (the common residue set, default of the probe): exactly the
    proteins of the set, each with its full length; a set protein that is not in the
    split, has no label or no feature, or whose label / feature length differs from the
    set's is an error (ResidueCoverageError).
    Without it (legacy --residue-set all): every protein with a feature, a label and a
    split; proteins shorter than 5 residues are skipped (old rule).
    FIXED(preretrain_audit_r2 Q3/E4, 2026-10-16): old = feature and label cut to the
    shorter of the two, absent proteins dropped silently. New = a length mismatch is
    always an error; absent proteins are an error inside the common set.
    Paper: 2_results.tex result:rmsf — "fit on the pooled-residue training set".
    """
    bucket = {s: {'x': [], 'y': [], 'g': []} for s in ('train', 'val', 'test')}
    if residue_set is not None:
        items = [(pid, int(v['length'])) for pid, v in residue_set['proteins'].items()]
    else:
        # legacy: the feature dict's order, as before (pooling order = batch order)
        items = [(pid, None) for pid in per_pid_feats
                 if pid in rmsf_per_pid and pid in split_map]
    for gi, (pid, L_set) in enumerate(items):
        if pid not in split_map:
            raise ResidueCoverageError(f'{pid} (common residue set) is not in the split')
        if pid not in rmsf_per_pid:
            raise ResidueCoverageError(f'{pid} (common residue set) has no RMSF label')
        x = per_pid_feats.get(pid)
        if x is None:
            raise ResidueCoverageError(f'{pid} (common residue set) has no feature')
        y = rmsf_per_pid[pid]
        if len(x) != len(y) or (L_set is not None and len(y) != L_set):
            raise ResidueCoverageError(f'{pid}: feature {len(x)}, label {len(y)}'
                                       + (f', residue set {L_set}' if L_set is not None else '')
                                       + ' residues')
        if L_set is None and len(y) < 5:
            continue
        s = split_map[pid]
        bucket[s]['x'].append(np.asarray(x, dtype=np.float32) if np.asarray(x).dtype.kind == 'f'
                              else np.asarray(x))
        bucket[s]['y'].append(np.asarray(y, dtype=np.float32))
        bucket[s]['g'].append(np.full(len(y), gi, dtype=np.int64))
    out = {}
    for s in bucket:
        if not bucket[s]['x']:
            out[s] = (None, None, None)
        else:
            out[s] = tuple(np.concatenate(bucket[s][k], axis=0) for k in ('x', 'y', 'g'))
    return out


# ---- Baseline feature builders (each returns {pid: (L, d)} and d) -----------
# Paper: appendix.tex app:baselines (single-frame and multi-frame methods).

def build_aa(pids):
    """`aa` baseline: one-hot of the residue identity.

    FIXED(preretrain_audit_r2 Q11, 2026-10-16): old = sequences from the old pairs pickle
    (3,903 of 4,894 complexes; disulfide Cys "CYX" mapped to X). New = the `seq` of
    data/cached_descriptors/misato_real_bb/<pid>.npz first (fixed parser, CYX/HIE mapped
    to C/H by build_misato_real_bb.THREE_TO_ONE, as P30 did for ProToken), the pickle
    only for complexes without a backbone file. Lengths are checked against the label in
    gather_residues.
    """
    pairs = pickle.load(open(PAIRS, 'rb')) if PAIRS.exists() else {}
    seq_by_pid = {}
    for k, v in pairs.items():
        pid = k.split(':')[-1] if ':' in k else k
        seq_by_pid[pid] = v.get('seq', '')
    feats, versions = {}, set()
    for pid in sorted(pids):
        rb = MISATO_REAL_BB_DIR / f'{pid}.npz'
        if rb.exists():
            with np.load(rb, allow_pickle=False) as z:
                seq = str(z['seq'])
                pv = str(z['parser_version']) if 'parser_version' in z.files else None
            versions.add(pv)
        else:
            seq = seq_by_pid.get(pid)
        if not seq: continue
        x = np.zeros((len(seq), len(AA_VOCAB)), dtype=np.float32)
        for i, a in enumerate(seq):
            x[i, AA_IDX.get(a, AA_IDX['X'])] = 1.0
        feats[pid] = x
    # FIXED(preretrain_audit_r2 F1, 2026-10-16): the sequences must come from the current
    #   MISATO parser (refused otherwise unless --allow-unverified-cache).
    from ensembits.data_versions import MISATO_PARSER_VERSION
    if _VERIFY and versions - {MISATO_PARSER_VERSION}:
        raise CacheError(f'misato_real_bb sequences from parser(s) '
                         f'{sorted(versions - {MISATO_PARSER_VERSION}, key=str)}, expected '
                         f'{MISATO_PARSER_VERSION}; rebuild misato_real_bb')
    # G11 (2026-10-17): path-less record re-checked by records_changed (full parser scan)
    _CACHES.append({'file': 'misato_real_bb (seq)', 'kind': 'misato_real_bb_seq',
                    'parser_version': misato_real_bb_scan().get('parser_version')})
    return feats, len(AA_VOCAB)


def build_3di_tokens(pids, centroids):
    """Returns per-pid (L, 2) features = centroid embedding lookup of mini3di tokens.

    Paper: appendix.tex app:baselines, "3Di_tokens" (frozen mini3di centroids).
    """
    d = np.load(_use(TOKENS_3DI))
    feats = {}
    for pid in pids:
        if pid not in d.files: continue
        # FIXED(preretrain_audit_r2 H4, 2026-10-16): old = ids outside [0, 20) became a zero
        #   vector; new = error (mini3di's masked terminals are state 2, in range).
        feats[pid] = centroids[check_token_ids(d[pid], centroids.shape[0], f'3di_tokens {pid}')]
    return feats, centroids.shape[1]


def build_protprofile_K8(pids):
    """Per-residue (L, 60) feature: 20-bin histogram with each bin annotated by
    its 2-D mini3di centroid coordinates → flat (L, 60).

    Paper: appendix.tex app:baselines, "ProtProfileMD_K" (K=8 on MISATO, 60-D).
    """
    d = np.load(_use(CACHE / 'protprofile_K8_misato.npz', frames='first 8 of bb_8',
                     uses_sequence=True))
    cent = np.load(CENTROIDS).astype(np.float32)  # (20, 2)
    feats = {}
    for pid in pids:
        if pid not in d.files: continue
        hist = d[pid].astype(np.float32)  # (L, 20)
        L = hist.shape[0]
        cb = np.broadcast_to(cent, (L, 20, 2))
        comb = np.concatenate([hist[..., None], cb], axis=-1)  # (L, 20, 3)
        feats[pid] = comb.reshape(L, 60)
    return feats, 60



def build_vote_3di(pids):
    """K=8 misato: plurality vote with NN-fallback on tie/all-different.

    Per residue: the mini3di state that wins a strict plurality (count >= 2) over
    the 8 frames; on a tie or when all differ, the state whose centroid is
    nearest to the mean of the frames' centroids. Feature = that state's 2-D
    centroid.
    Paper: appendix.tex app:baselines, "Vote_3di".
    (require_fix #22, resolved in the text: per-residue 2-D centroid, ties resolved by
    the nearest centroid.)
    """
    # FIXED(function_reaudit pre-run fix 7, 2026-10-19): the vote is
    #   ensembits.baselines.mini3di_baselines.vote_from_frame_tokens (float64 distances, lowest
    #   letter id on ties), the ANOVA's rule; old: the same rule inline with float32 argmin
    #   (either of two exactly tied centroids). Results record rmsf_common.VOTE_RULE.
    from ensembits.baselines.mini3di_baselines import vote_from_frame_tokens
    d = np.load(_use(CACHE / 'mini3di_tokens_K8_misato.npz', frames='all 8'))
    cent = np.load(CENTROIDS).astype(np.float32)
    M = cent.shape[0]
    feats = {}
    for pid in pids:
        if pid not in d.files: continue
        # FIXED(preretrain_audit_r2 F4, 2026-10-16): old = np.clip of the ids; new = error.
        tk = check_token_ids(d[pid], M, f'vote_3di {pid}')
        feats[pid] = cent[vote_from_frame_tokens(tk, cent)]
    return feats, cent.shape[1]


def build_pca_eig(pids):
    """K=8 misato: per-residue 3 eigenvalues of the 8-frame Cα covariance.

    Not a baseline in the paper (no frame alignment; diagnostic only).
    """
    coords = pickle.load(open(_use_coords(CACHE / 'coords_ca8_misato.pkl'), 'rb'))
    feats = {}
    for pid in pids:
        if pid not in coords: continue
        ca = np.asarray(coords[pid], dtype=np.float32)
        P, L, _ = ca.shape
        cen = ca - ca.mean(axis=0, keepdims=True)
        cov = np.einsum('pli,plj->lij', cen, cen) / P
        evals = np.linalg.eigvalsh(cov)[:, ::-1]
        feats[pid] = evals.astype(np.float32)
    return feats, 3


def build_token_codebook(pids, kind: str):
    """kind: 'aminoaseed', 'protoken' or 'esm3struct'. Frozen codebook lookup per residue.

    Needs `<kind>_tokens_misato.npz` (with its provenance sidecar) and
    `<kind>_codebook.npy` in data/cached_descriptors, built by
    data_prep/baselines/build_<kind>_tokens.py (K5 2026-10-17: the `*_ourdata` and side
    variants mentioned here were removed, G14).
    Paper: appendix.tex app:baselines, "AminoAseed" (512 x 1024-D), "ProToken"
    (512 x 32-D), "ESM3struct" (4096 x 128-D).
    """
    cb_path = CACHE / f'{kind}_codebook.npy'
    # G6 (2026-10-17): no token baseline reads the sequence (ProToken's encoder has no aatype)
    d = np.load(_use(CACHE / f'{kind}_tokens_misato.npz', codebook=cb_path))
    cb = np.load(cb_path).astype(np.float32)
    feats = {}
    for pid in pids:
        if pid not in d.files: continue
        tok = d[pid].astype(np.int64)
        tok = check_token_ids(tok, cb.shape[0], f'{kind} {pid}')   # F4: was np.clip
        feats[pid] = cb[tok]
    return feats, cb.shape[1]


def build_random(pids):
    """Deterministic per-(pid, residue) random (2,) vector.

    The seed is an FNV-1a hash of the pid, so the features do not change between
    probe seeds.
    Paper: appendix.tex app:baselines, "Random".
    (require_fix #22, resolved in the text: a fixed random 2-D vector per residue.)
    """
    coords = pickle.load(open(_use_coords(CACHE / 'coords_ca8_misato.pkl'), 'rb'))
    D = 2
    feats = {}
    for pid in pids:
        if pid not in coords: continue
        L = coords[pid].shape[1]
        h = 2166136261
        for ch in pid:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        feats[pid] = rng.standard_normal((L, D)).astype(np.float32)
    return feats, D


def build_variant_tokens(variant: str):
    """Loads the variant's misato token cache. Returns per-pid (L,) int array
    + the codebook (M, d_emb) for embedding init.

    (Only the level-1 codebook is returned; not called by main().)
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tokenize_misato_and_eval import load_model_robust  # noqa
    variant_dir = ROOT / 'output' / variant
    ens = load_model_robust(variant_dir, device='cpu')
    state = ens.model.state_dict()
    cb = state.get('vq_levels.0.codebook.weight')
    if cb is None: cb = state['vq.codebook.weight']
    cb = cb.detach().cpu().numpy().astype(np.float32)
    return cb


# ---- Probe heads -------------------------------------------------------------
# Paper: 2_results.tex result:rmsf — "a single-frame MLP head" per residue.

class MLPProbe(nn.Module):
    # Baselines: dense per-residue feature -> MLP (128 -> 128 -> 1).
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.mlp(x).squeeze(-1)


class TokenMLP(nn.Module):
    """Token id → learnable embedding (init from codebook) → MLP.

    Note: `freeze=True`, so the codebook embedding is NOT trained; only the MLP
    learns (the "learnable" above is outdated). Used for `--mode variant`.
    """
    def __init__(self, codebook, hidden=128):
        super().__init__()
        cb = torch.as_tensor(codebook, dtype=torch.float32)
        self.emb = nn.Embedding.from_pretrained(cb, freeze=True)
        d = cb.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))
    def forward(self, t): return self.mlp(self.emb(t)).squeeze(-1)


class ProtProfileMLP(nn.Module):
    """histogram (20-D) → learnable centroid projection → MLP.

    Not used by main() (the ProtProfile baseline uses the 60-D feature + MLPProbe).
    """
    def __init__(self, centroids, hidden=128):
        super().__init__()
        cent = torch.as_tensor(centroids, dtype=torch.float32)
        M, d_emb = cent.shape
        self.proj = nn.Linear(M, d_emb, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(cent.T)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))
    def forward(self, x): return self.mlp(self.proj(x)).squeeze(-1)


MAX_EPOCHS = 500    # default cap of the early-stopping loop (--max-epochs)
PATIENCE = 20       # default epochs without val improvement before stopping (--patience)
MIN_DELTA = 1e-4    # a val Spearman gain must exceed this to count as an improvement


def train_and_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_te, dev, *,
                   max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA,
                   bs=2048, lr=3e-4, wd=1e-4, dtype_x=torch.float32, seed=0,
                   groups_va=None, groups_te=None, return_preds=False):
    """Train with MSE on the pooled train residues until convergence, select the
    epoch on val, score test. Returns a dict (see below).

    groups_va / groups_te: protein index of every val / test residue; when given, the
    per-protein Spearman of the selected model is added for val and test (selection
    stays on the pooled val Spearman).
    return_preds: also return the selected model's val / test predictions (input order) under
    the private key '_preds' (popped by main for the per-protein sidecar, A4 (a)); absent when
    no epoch was selected.

    After every epoch the Spearman on the pooled val residues is computed; an
    epoch counts as an improvement when it beats the best so far by more than
    `min_delta`, and its weights are kept. Training stops after `patience`
    epochs without improvement, or after `max_epochs`. The kept (best-val)
    weights are then evaluated ONCE on the pooled test residues.
    Seeding is as before: torch.manual_seed(seed) here, and a separate
    generator seeded with `seed` for the batch order.
    Paper: 2_results.tex result:rmsf — "Spearman ... between predicted and
    ground-truth RMSF on held-out residues" (= test_spearman).
    """
    # FIXED(code_fix §7; require_fix #20, 2026-10-15): previously trained a
    #   fixed 40 epochs and returned the max over epochs of the val Spearman (the
    #   epoch was selected on the residues being reported; test never used). Now
    #   early stopping on val (patience 20, min delta 1e-4, cap 500 epochs), the
    #   best-val weights are restored, and the test split is the reported number.
    torch.manual_seed(seed)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x_tr = torch.tensor(X_tr, device=dev, dtype=dtype_x)
    y_tr_t = torch.tensor(y_tr, device=dev)
    x_va = torch.tensor(X_va, device=dev, dtype=dtype_x)
    y_va_t = torch.tensor(y_va, device=dev)
    x_te = torch.tensor(X_te, device=dev, dtype=dtype_x)
    y_te_np = np.asarray(y_te, dtype=np.float32)
    y_va_np = y_va_t.cpu().numpy()
    g = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = float('-inf'), 0, None
    since_best, n_run = 0, 0
    all_const = True          # val predictions constant at every epoch (2026-10-18)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g)
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs].to(dev)
            loss = F.mse_loss(model(x_tr[idx]), y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        n_run = ep + 1
        model.eval()
        with torch.no_grad():
            pv = model(x_va).cpu().numpy()
        # FIXED(constant-prediction rule, 2026-10-18): record whether the val predictions were
        #   constant at every epoch (rmsf_common.constant_prediction_fields).
        all_const = all_const and bool(np.ptp(pv) == 0)
        sp = spearmanr(pv, y_va_np).correlation
        # Early stopping on the val Spearman (NaN, e.g. constant predictions,
        # never counts as an improvement).
        if sp is not None and np.isfinite(sp) and sp > best + min_delta:
            best, best_epoch, since_best = float(sp), ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    # Restore the best-val weights and evaluate the held-out test residues once.
    pp_va = pp_te = None
    preds = None
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pt = model(x_te).cpu().numpy()
        # FIXED(final_fix_list A4 (a), 2026-10-19): new: the selected model's val / test
        #   predictions for the per-protein sidecar; old: not kept.
        if return_preds:
            with torch.no_grad():
                preds = {'val': model(x_va).cpu().numpy(), 'test': pt}
        test_sp = spearmanr(pt, y_te_np).correlation
        test_sp = float(test_sp) if test_sp is not None else float('nan')
        # FIXED(preretrain_audit_r2 Q2, 2026-10-16): old = pooled Spearman only. New = also
        #   the per-protein Spearman of the selected model on val and test (eval mode is
        #   deterministic: these val predictions are those of the selected epoch).
        if groups_va is not None:
            with torch.no_grad():
                pv_best = model(x_va).cpu().numpy()
            pp_va = per_protein_spearman(pv_best, y_va_np, groups_va)
        if groups_te is not None:
            pp_te = per_protein_spearman(pt, y_te_np, groups_te)
    else:   # val Spearman was never finite
        best, test_sp = float('nan'), float('nan')
    pp = {}
    if groups_va is not None:
        pp.update(per_protein_fields('val', pp_va))
    if groups_te is not None:
        pp.update(per_protein_fields('test', pp_te))
    if pp:
        pp['per_protein_min_residues'] = PER_PROTEIN_MIN_RESIDUES
    return {'test_spearman': test_sp,
            **pp,
            'best_val_spearman': float(best),
            'best_epoch': int(best_epoch),          # 1-based; 0 = never improved
            'n_epochs_run': int(n_run),
            'max_epochs': int(max_epochs),
            'patience': int(patience),
            'min_delta': float(min_delta),
            'selection': 'val_early_stopping',
            'n_train_residues': int(len(X_tr)),
            'n_val_residues': int(len(X_va)),
            'n_test_residues': int(len(X_te)),
            **constant_prediction_fields(best_state is None, all_const, X_tr, X_va, X_te),
            **({'_preds': preds} if return_preds else {})}


def write_misato_rmsf_preds(out_path, preds, lab_by_split: dict, group_names, out: dict) -> dict:
    """Write <out>.preds.npz (kind residue_regression) of a MISATO RMSF run; return the record.

    lab_by_split: {'val' / 'test': (labels, group index per residue)} as returned by
    gather_residues (each complex contiguous, residues in label order).
    """
    from probe_rmsf_eval import rmsf_meta
    if preds is None:
        return not_selected_record('no epoch selected (val predictions constant or NaN)')
    names = np.asarray(group_names).astype(str)
    splits = {}
    for s in ('val', 'test'):
        y, g = lab_by_split[s]
        splits[s] = residue_split_from_residues(names[np.asarray(g)], preds[s], y,
                                                contiguous_residue_index(g))
    return write_preds(sidecar_path(out_path), kind='residue_regression', splits=splits,
                       meta={'task': 'rmsf', **rmsf_meta(out)}, json_dir=Path(out_path).parent)


def main(argv=None):
    # ---- CLI -------------------------------------------------------------
    # Modes in the paper: variant, aa, 3di_tokens, protprofile_K8, vote_3di,
    # random, aminoaseed, protoken, esm3struct; pca_eig is a diagnostic (not in the paper).
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    # FIXED(preretrain_audit_r2 G14, 2026-10-17): aug_rot6d, ensembits_ifold*,
                    #   big_* and *_ourdata removed: no builder of this codebase writes their
                    #   caches with a provenance sidecar, so they could never pass (F1).
                    choices=['variant', 'aa', '3di_tokens', 'protprofile_K8',
                             'vote_3di', 'pca_eig', 'random',
                             'aminoaseed', 'protoken', 'esm3struct',
                             'aminoaseed_retrained', 'esm3struct_retrained'])
    ap.add_argument('--variant', default=None,
                    help='Required when mode=variant')
    ap.add_argument('--p-inference', type=int, default=None,
                    help='When mode=variant, restrict to first P frames at inference')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    # Early stopping (code_fix §7). `--epochs` is kept as an alias of
    # --max-epochs so old command lines still parse (it was a fixed epoch count).
    ap.add_argument('--max-epochs', '--epochs', dest='max_epochs', type=int,
                    default=MAX_EPOCHS,
                    help=f'cap on training epochs (default {MAX_EPOCHS}); training '
                         'stops earlier when val Spearman stops improving. '
                         '--epochs is an alias kept for backward compatibility')
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help=f'epochs without val Spearman improvement > {MIN_DELTA:g} '
                         f'before stopping (default {PATIENCE})')
    ap.add_argument('--tag', default='test',
                    help='output name suffix "__<tag>" (default: test), so the '
                         'old best-val JSONs of the paper are not overwritten')
    ap.add_argument('--split', default='structure',
                    choices=['sequence', 'structure', 'random'],
                    help='misato split (default: structure, matches the EC/GO/binding tables)')
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='mode=variant: accept a checkpoint whose training run has not '
                         'finished (code_fix P12)')
    # FIXED(preretrain_audit_r2 Q3, 2026-10-16): old = each mode on the complexes it had
    #   features for (aa: 3,903 of 4,894), lengths cut to the shorter of feature / label.
    #   New = the common residue set (default; refused if absent); "all" = legacy.
    ap.add_argument('--residue-set', default='common',
                    help='"common" (default): the common residue set of every method; "all": '
                         'legacy, the complexes this mode covers; or a residue-set JSON path')
    ap.add_argument('--rmsf-label', default=RMSF_LABEL_DEFAULT, choices=tuple(RMSF_LABEL_CHOICES),
                    help='allframes (default): all 100 MD frames, converged superposition; '
                         'frames8: the old 8-frame label (reproduction only)')
    ap.add_argument('--allow-unverified-cache', action='store_true',
                    help='accept feature caches without a current provenance sidecar (side '
                         'experiments only; recorded, and the compilers refuse such rows)')
    # FIXED(final_fix_list A4 (a), 2026-10-19): new; as in probe_rmsf_eval.py.
    ap.add_argument('--redo-missing-preds', action='store_true',
                    help='recompute an existing result without a per-protein prediction sidecar')
    args = ap.parse_args(argv)
    global _VERIFY
    _VERIFY = not args.allow_unverified_cache
    _CACHES.clear()
    # FIXED(code_fix P25, 2026-10-16): old: torch was seeded only inside train_and_eval, after
    #   the probe model had been built (unseeded initial weights). New: torch and numpy are
    #   seeded here, before any model is built; train_and_eval still re-seeds as before.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.tag:
        print('--tag must be non-empty (an empty tag would overwrite the old '
              'best-val result JSONs)', file=sys.stderr); return 2
    # Shared early-stopping settings for every mode.
    es_kw = dict(max_epochs=args.max_epochs, patience=args.patience, seed=args.seed)

    try:
        rs, rs_record = resolve_residue_set(args.residue_set, 'misato')
    except (FileNotFoundError, ValueError) as e:
        print(f'[error] {e}', file=sys.stderr); return 1
    global _REQUIRED
    _REQUIRED = set(rs['proteins']) if rs is not None else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Labels and split ---------------------------------------------------
    # Paper: 2_results.tex result:rmsf — MISATO sequence (UniProt-clustered) and
    # structure (CATH-H clustered) v2 splits (benchmark_audit D3).
    # FIXED(code_fix P28, 2026-10-16): the label's provenance must match the coordinates
    #   file of the features (refused otherwise; loud warning without provenance).
    label_path = RMSF_LABEL_CHOICES[args.rmsf_label]
    if not label_path.exists():
        print(f'[error] RMSF label {label_path} missing (build it with '
              f'data_prep/build_misato_rmsf_labels.py)', file=sys.stderr); return 1
    try:
        label_prov = check_label_provenance(label_path, COORDS_CA8, args.rmsf_label)
    except ValueError as e:
        print(f'[error] {e}', file=sys.stderr); return 1
    rmsf_d = np.load(label_path)
    rmsf_per_pid = {pid: rmsf_d[pid] for pid in rmsf_d.files}
    # FIXED(preretrain_audit_r2 F5, 2026-10-16): new: the label's content hash (checked
    #   against its sidecar) goes into the JSON next to its name.
    try:
        label_sha = check_label_hash(label_path, rmsf_per_pid)
    except CacheError as e:
        print(f'[error] {e}', file=sys.stderr); return 1
    try:
        split_map = load_split_map(args.split)
    except ResidueCoverageError as e:
        print(f'[error] {e}', file=sys.stderr); return 1
    pids = set(split_map.keys()) & set(rmsf_per_pid.keys())
    print(f'[setup] mode={args.mode} split={args.split} #pids with RMSF+split={len(pids)}', flush=True)

    # Suffix non-default splits into the output filename so we don't overwrite
    # existing sequence-split artifacts. (sequence was the original default.)
    split_suffix = '' if args.split == 'sequence' else f'_{args.split}'
    seed_tag = '' if args.seed == 0 else f'_seed{args.seed}'
    seed_tag = split_suffix + seed_tag

    # ---- Output name; an existing result is kept only if it matches ------------
    # FIXED(preretrain_audit_r2 F6, 2026-10-16): old = the result JSON was always
    #   (over)written. New = same rule as the mdCATH-div probes: an existing result is kept
    #   (skip, exit 0) only if its label (name and content), residue set, checkpoint
    #   (variant) and feature caches match the current ones; otherwise exit 1 (move it aside).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tokenize_misato_and_eval import tag_of as _tag_of
    if args.mode == 'variant':
        if args.variant is None:
            print('--variant required for mode=variant', file=sys.stderr); return 2
        stem = (f'{_tag_of(args.variant)}'
                f'{(f"_P{args.p_inference}inf" if args.p_inference is not None else "")}{seed_tag}')
    else:
        stem = f'baseline_{args.mode}{seed_tag}'
    out_path = OUT_DIR / f'{stem}__{args.tag}.json'
    prov0 = None
    if args.mode == 'variant':
        # FIXED(preretrain_audit_r2 F9, 2026-10-16): the checkpoint hash is taken BEFORE the
        #   model is loaded and compared again before writing (a running training could
        #   replace best.pt in between).
        from ensembits.featurize import checkpoint_provenance
        prov0 = checkpoint_provenance(ROOT / 'output' / args.variant)
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        why = []
        if prev.get('rmsf_label') != args.rmsf_label or prev.get('rmsf_label_sha256') != label_sha:
            why.append('RMSF label (name or content)')
        if prev.get('residue_set_sha256') != rs_record['residue_set_sha256']:
            why.append('residue set')
        if prov0 is not None and not strict_same_checkpoint(prev.get('checkpoint'), prov0):
            why.append('checkpoint')
        if prev.get('cache_verified') is not (not args.allow_unverified_cache):
            why.append('cache verification')
        if vote_rule_problem(prev):                         # 2026-10-19
            why.append('vote rule')
        why += records_changed(prev.get('feature_caches'))
        if prev.get('splits_sha256') != splits_record(SPLITS)['splits_sha256']:
            why.append('split file')                         # G4 (2026-10-17)
        if why:
            print(f'[error] {out_path.name} exists and differs ({", ".join(why)}); move it aside',
                  file=sys.stderr); return 1
        _np = needs_rerun(prev.get(PREDS_KEY), out_path)      # A4 (a), 2026-10-19
        if _np is None or not args.redo_missing_preds:
            print(f'[skip] {out_path.name}' + (f' (note: {_np}; --redo-missing-preds recomputes it)'
                                               if _np else '')); return 0
        print(f'[redo] {out_path.name}: {_np}', flush=True)

    # ---- Per-residue inputs of the chosen mode -----------------------------
    # Every mode goes through gather_residues (common residue set, strict lengths) and the
    # same train_and_eval; only the inputs and the head differ (unchanged per mode).
    if args.mode == 'variant':
        # ---- Tokenizer (variant) mode: frozen level-1 codeword -> MLP --------
        # Trigger tokenization (uses coords_ca8 in misato; if cache exists it's reused)
        # For the reported (256-D) model the cache MUST already exist: it is
        # written by _pretokenize_misato_dihedral.py from real N/CA/C (bb_8,
        # P = 8, or the first frame for --p-inference 1). The CA-only fallback in
        # tokenize_misato cannot build the 256-D descriptor (ensembits.featurize
        # requires bb_all), so every protein would be skipped.
        # Paper: 2_results.tex result:rmsf — "MISATO ... P=8 frames"; P=1 row.
        # code_fix §1: cached tokens are per checkpoint; the retrained model
        #   writes its own (old caches hold legacy-convention tokens).
        # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tokenize_misato_and_eval import tokenize_misato, tag_of
        variant_dir = ROOT / 'output' / args.variant
        tok_cache = CACHE / (f'ensembits_{tag_of(args.variant)}'
                             f'{(f"_P{args.p_inference}inf" if args.p_inference is not None else "")}'
                             f'_tokens_misato.npz')
        # FIXED(preretrain_audit_r2 H1, 2026-10-16): old = the Ensembits token cache was used
        #   with a checkpoint check only (old caches of the legacy checkpoint have no sidecar
        #   and come from old-parser backbones). New = its sidecar must record MISATO inputs
        #   from the current parser with K = 8 and Kabsch FPS, and the frames of this P
        #   (unless --allow-unverified-cache); recorded in feature_caches.
        from ensembits.data_versions import read_provenance
        want_frames = f'first {args.p_inference if args.p_inference is not None else 8} of bb_8'
        side = read_provenance(tok_cache) if tok_cache.exists() else None
        try:
            if _VERIFY:
                if side is None:
                    raise CacheError(f'{tok_cache.name}: no provenance sidecar (built before '
                                     f'2026-10-16); rebuild it with _pretokenize_misato_dihedral.py')
                check_misato_inputs(side.get('inputs'), tok_cache.name)
                # FIXED(preretrain_audit_r2 G8, 2026-10-17): the cache must still be the file
                #   its sidecar describes.
                from ensembits.featurize import file_sha256
                if side.get('cache_sha256') != file_sha256(tok_cache):
                    raise CacheError(f'{tok_cache.name} changed after its sidecar was written '
                                     f'(or the sidecar records no cache_sha256); rebuild it')
                if side.get('frames') != want_frames:
                    raise CacheError(f'{tok_cache.name}: frames {side.get("frames")!r}, '
                                     f'expected {want_frames!r}')
        except CacheError as e:
            print(f'[error] {e}', file=sys.stderr); return 1
        if tok_cache.exists():
            _CACHES.append({**file_record(tok_cache), 'frames': want_frames,
                            'inputs': (side or {}).get('inputs'),
                            **({} if _VERIFY else {'verified': False})})
        # FIXED(preretrain_audit_r2 H7, 2026-10-16): an unfinished checkpoint gives one clean
        #   [error] line (was a traceback).
        from ensembits.featurize import UnfinishedCheckpointError, ProvenanceMismatchError
        try:
            feats_int, codebook, M = tokenize_misato(variant_dir, device=args.device,
                                                      p_inference=args.p_inference,
                                                      allow_unfinished=args.allow_unfinished)
        except ProvenanceMismatchError as e:        # G8 (2026-10-17): clean error
            print(f'[error] {e} -> the token cache belongs to another checkpoint; rebuild it with '
                  f'_pretokenize_misato_dihedral.py', file=sys.stderr)
            return 1
        except UnfinishedCheckpointError as e:
            print(f'[error] {e} -> pass --allow-unfinished, or probe a frozen copy made by '
                  f'downstream/snapshot_checkpoint.py', file=sys.stderr)
            return 1
        feats = {pid: feats_int[pid] for pid in feats_int}   # (L,) tokens; TokenMLP below
        cb = codebook.detach().cpu().numpy() if hasattr(codebook, 'detach') else np.asarray(codebook)
        make_model = lambda: TokenMLP(cb)
        dtype_x = torch.long
        from rmsf_common import snapshot_info
        out = {'variant': args.variant, 'p_inference': args.p_inference,
               'M': int(cb.shape[0]), 'd_emb': int(cb.shape[1]), 'seed': args.seed,
               'snapshot_of': (snapshot_info(variant_dir) or {}).get('snapshot_of'),
               'checkpoint': prov0}   # FIXED(code_fix P17, 2026-10-16): new key (old: none)
    else:
        # ---- Baseline modes: dense per-residue feature -> MLP ----------------
        try:
            if args.mode == 'aa':
                feats, in_dim = build_aa(pids)
            elif args.mode == '3di_tokens':
                cent = np.load(CENTROIDS).astype(np.float32)
                feats, in_dim = build_3di_tokens(pids, cent)   # already centroid-embedded
            elif args.mode == 'protprofile_K8':
                feats, in_dim = build_protprofile_K8(pids)     # in_dim=60 (histogram + centroids)
            elif args.mode in ('aminoaseed', 'protoken', 'esm3struct',
                               'aminoaseed_retrained', 'esm3struct_retrained'):
                feats, in_dim = build_token_codebook(pids, args.mode)
            elif args.mode in ('vote_3di', 'pca_eig', 'random'):
                builder = {'vote_3di':  build_vote_3di,
                           'pca_eig':     build_pca_eig,
                           'random':      build_random}[args.mode]
                feats, in_dim = builder(pids)
            else:
                print(f'unknown mode {args.mode}'); return 2
        except CacheError as e:
            print(f'[error] {e}', file=sys.stderr); return 1
        make_model = lambda: MLPProbe(in_dim)
        dtype_x = torch.float32
        out = {'feature': args.mode, 'in_dim': in_dim, 'seed': args.seed}

    try:
        bucket = gather_residues(feats, rmsf_per_pid, split_map, residue_set=rs)
    except ResidueCoverageError as e:
        print(f'[error] {e}', file=sys.stderr); return 1
    if any(bucket[s][0] is None for s in ('train', 'val', 'test')):
        print('empty split'); return 1
    (X_tr, y_tr, _), (X_va, y_va, g_va), (X_te, y_te, g_te) = (
        bucket['train'], bucket['val'], bucket['test'])
    print(f'[residues] set={rs_record["residue_set"]} train={len(X_tr)} val={len(X_va)} '
          f'test={len(X_te)}', flush=True)
    res = train_and_eval(make_model(), X_tr, y_tr, X_va, y_va, X_te, y_te, args.device,
                         dtype_x=dtype_x, groups_va=g_va, groups_te=g_te, return_preds=True,
                         **es_kw)
    preds = res.pop('_preds', None)                   # A4 (a): sidecar below
    # Protein id of each group index of gather_residues (same item order).
    group_names = (list(rs['proteins']) if rs is not None else
                   [pid for pid in feats if pid in rmsf_per_pid and pid in split_map])
    out = {**{k: v for k, v in out.items() if k not in ('seed', 'checkpoint')}, **res,
           'seed': args.seed, **({'checkpoint': out['checkpoint']} if 'checkpoint' in out else {})}

    # ---- Write the result JSON ----------------------------------------------
    # FIXED(require_fix #17, tex pass 2026-10-26): RMSF table updated with 256-D numbers.
    # The "__<tag>" suffix keeps the old best-val JSONs (same stem, no suffix)
    # untouched; `split` is recorded because the sequence split has no name suffix.
    out['split'] = args.split
    out['label_provenance'] = label_prov      # FIXED(code_fix P28, 2026-10-16): new key (old: none)
    # FIXED(preretrain_audit_r2 Q3 / Q17, 2026-10-16): new keys: the label name (one MISATO
    #   label, user decision Q1), the dataset and the common residue set (hash, size).
    out['rmsf_label'] = args.rmsf_label
    out['rmsf_label_file'] = label_path.name
    out['rmsf_label_sha256'] = label_sha             # FIXED(preretrain_audit_r2 F5, 2026-10-16)
    out['dataset'] = 'misato'
    out.update(splits_record(SPLITS))                 # G4 (2026-10-17): split-file hash
    out['batch_size'] = 2048                         # train_and_eval default (recorded, F10)
    out.update(rs_record)
    # FIXED(preretrain_audit_r2 F1, 2026-10-16): new keys: the caches read (sha256).
    out['feature_caches'] = list(_CACHES)
    out['cache_verified'] = not args.allow_unverified_cache
    if out.get('feature') == 'vote_3di':
        out['vote_rule'] = VOTE_RULE                  # 2026-10-19 (function_reaudit fix 7)
    if prov0 is not None:
        from ensembits.featurize import checkpoint_provenance
        if not strict_same_checkpoint(checkpoint_provenance(ROOT / 'output' / args.variant), prov0):
            print(f'[error] output/{args.variant}/best.pt changed while the probe ran; result '
                  f'not written (probe a snapshot: downstream/snapshot_checkpoint.py)',
                  file=sys.stderr); return 1
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar <out>.preds.npz (val and
    #   test predictions of the selected model, labels, residue index = position in the
    #   complex's residue list) recorded in `per_protein_preds`; atomic JSON write (old:
    #   write_text).
    out[PREDS_KEY] = write_misato_rmsf_preds(out_path, preds, {'val': (y_va, g_va),
                                                               'test': (y_te, g_te)},
                                             group_names, out)
    atomic_write_json(out_path, out)
    print(f'[done] {out_path.name}  test Spearman={out["test_spearman"]:.4f}  '
          f'(per protein {out["test_spearman_per_protein_mean"]:.4f}; '
          f'best val {out["best_val_spearman"]:.4f} at epoch {out["best_epoch"]}, '
          f'{out["n_epochs_run"]} epochs run)', flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
