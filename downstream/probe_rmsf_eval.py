"""RMSF probe on mdCATH-div (or BioEmu-CATH) for one tokenizer (frozen level-1 codebook + MLP).

What it does
------------
  1. Loads a trained tokenizer from `REPO_ROOT/output/<variant>/` (config.json +
     best.pt) through `tokenize_misato_and_eval.load_model_robust`.
  2. Loads the cached per-residue descriptor of every mdCATH-div residue,
     `output/phase_b/cached_desc/<mode>_k<k>_P<P>.npy`. For the reported model
     this is `esm3desc_dihedral_k16_P10.npy`, shape (308,293 residues, 10 frames,
     256), built by `data_prep/build_esm3desc_dihedral_perdataset.py --dataset
     mdcath`. It is standardised with the model's stored mean/std (identity for
     the reported model) and, with `--p-inference n`, cut to the first n frames
     (n = 1 gives the single-frame "P=1" row).
  3. Tokenizes every residue with the set encoder + residual quantizer and keeps
     the level-1 token (codebook M_1 = 2048).
  4. Trains the probe: token -> frozen level-1 codeword (128-D) -> MLP
     (128 -> 128 -> 1, GELU, dropout 0.1) -> predicted RMSF. MSE loss, AdamW
     (lr 3e-4, weight decay 1e-4), batch 2048, on the residues listed under
     "train" in `output/full_train/cached_desc/splits.json`, until convergence:
     after every epoch it computes the Spearman correlation between predicted
     and true RMSF on the "val" residues and stops when that has not improved
     by more than 1e-4 for `--patience` epochs (default 20), or after
     `--max-epochs` (default 500). The weights of the best val epoch are kept.
  5. Evaluates that best-val model once on the "test" residues of the same
     splits file; `test_spearman` is the headline metric. (Before 2026-10-15:
     fixed 40 epochs, reported the max over epochs of the val Spearman,
     `best_val_spearman`, test never used; code_fix §7.) The test residues
     (30,831 rows, 245 domains; the common residue set had 30,746 / 244 before the
     re-extraction of 2026-10-17 added 1ggpA02) are all in the BioEmu hold-out list, which
     also resolves code_fix §8 for the reported number.
  6. Writes one JSON per (variant, P, seed) to `output/probe_rmsf_div/`, with
     the `__<tag>` suffix (default `__test`) so the old best-val JSONs used by
     the paper are not overwritten.

Since 2026-10-16 (preretrain_audit_r2 Q2 / Q3, user decisions):
  * train / val / test are restricted to the common residue set
    (`data/cached_descriptors/rmsf_residue_set_<dataset>.json`, built by
    `data_prep/build_rmsf_residue_sets.py`; on mdCATH-div all 2,442 domains since the
    re-extraction of 2026-10-17, 2,434 before); the probe refuses to run without it unless
    `--residue-set all` (legacy, every label row); the set's sha256 and size are in
    the JSON (`residue_set*`);
  * next to the pooled Spearman (headline, unchanged) the JSON has the per-protein
    Spearman of the selected model on val and test (`val_/test_spearman_per_protein_
    mean / _median`, `n_val/test_proteins_scored`, proteins with >= 10 residues and
    non-constant labels; rmsf_common.per_protein_spearman); early stopping still uses
    the pooled val Spearman;
  * `--rmsf-label` allframes (default: every MD frame, converged superposition),
    converged5 or legacy; recorded and checked against an existing result;
  * `--dataset bioemu` (phase 2): the same probe on the 1,090 BioEmu-CATH systems
    (label, splits and descriptor cache in output/phase_b/cached_desc/bioemu/, results
    in output/probe_rmsf_bioemu/; P = full = the 10 bb_10 frames, P = 1 = frame 0).

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool encode flexibility?}
    \\label{result:rmsf}: the mdCATH-div column of Table \\label{tab:rmsf}
    (P=10 frames for "\\tool, P=full", one frame for "\\tool, P=1"; "a
    single-frame MLP head is fit on the pooled-residue training set";
    "Spearman ... between predicted and ground-truth RMSF on held-out residues"
    = `test_spearman` on the test split, epoch selected on val; mean over 10
    seeds; the seeds are looped by `_run_model_battery.py`, one seed per call
    here).
  * `appendix.tex`, \\section{Baselines} \\label{app:baselines}: "\\tool, P=1"
    and "\\tool, P=full" ($P{=}10$ on mdCATH-div).
  * `appendix.tex`, \\subsection{Compute Requirements} \\label{sec:compute}:
    "RMSF probe (one split, per-residue MLP, 40 ep)".
    (require_fix #20, resolved in the text: early stopping, epoch selected on
    validation, Spearman reported on the test split.)

Origin
------
Copied from `submission_exp/src/probe_rmsf_eval.py`. Changes: imports, sys.path
and ROOT setup (ROOT is now `ensembits.paths.REPO_ROOT`; this script's own
folder replaces `submission_exp/src` on sys.path so `tokenize_misato_and_eval`
is the copy in `downstream/`; `scripts/baselines` was dropped because
`ensembits_featurize` is now `ensembits.featurize`), comments and this
docstring. (The original docstring said the output goes to
`output/probe_rmsf_misato/`; the code writes to `output/probe_rmsf_div/`.)
Logic changed on 2026-10-15 (approved by the authors, code_fix §7/§8,
require_fix #20): the training loop moved into `fit_probe_early_stopping`
(early stopping on the val Spearman, best-val weights restored, test split
scored); the test indices are read from the splits file; new flags
`--max-epochs` (alias `--epochs`), `--patience`, `--tag`; output names get the
`__<tag>` suffix; the JSON gains `test_spearman` and the selection fields.
Seeding is unchanged.
Fixes 2026-10-16 (FIXED markers): the descriptor cache must have as many rows as
rmsf_div.npy and match its builder sidecar (convention, chain-break cutoff)
(code_fix P13); `checkpoint` block in the JSON (P17); --allow-unfinished (P12); the
exit status is main()'s return value; common residue set, per-protein Spearman,
label names and --dataset (see above; preretrain_audit_r2 Q2 / Q3).

Used by the reported model: yes. Produces the mdCATH-div RMSF cells (P=full and
P=1, seeds 0-9) for `BioEmu-CATH+MISATO` in `rebuttal/results_bf65.md`.

Inputs / outputs
----------------
Reads (relative to REPO_ROOT):
  output/<variant>/{config.json,best.pt,stats.npz}
  output/phase_b/cached_desc/<mode>_k<k>_P<P>.npy       descriptor cache (L_total, P, D)
  output/phase_b/cached_desc/rmsf_div_allframes.npy     RMSF label per residue (L_total,)
      (--rmsf-label converged5: rmsf_div_converged.npy; legacy: rmsf_div.npy)
  output/full_train/cached_desc/splits.json             residue indices {train, val, test}
  data/cached_descriptors/rmsf_residue_set_mdcath_div.json   common residue set
  data/external/mdcath_div/{_manifest.json,<pid>.npz}   row layout (--residue-set all only)
  --dataset bioemu: output/phase_b/cached_desc/bioemu/{<mode>_k16_P10.npy,
      rmsf_bioemu_allframes.npy, splits.json}, data/cached_descriptors/rmsf_residue_set_bioemu.json
Writes:
  output/probe_rmsf_div/<vtag>[_P<n>inf][_seed<s>]__<tag>.json   (bioemu: output/probe_rmsf_bioemu/)
      <vtag> = tag_of(variant), <tag> = --tag (default "test"). Fields:
      test_spearman (headline), best_val_spearman (val Spearman at the selected
      epoch), best_epoch (1-based), n_epochs_run, max_epochs, patience,
      min_delta, selection = "val_early_stopping", n_{train,val,test}_residues,
      {val,test}_spearman_per_protein_{mean,median}, n_{val,test}_proteins[_scored,
      _constant_pred], per_protein_min_residues, dataset, rmsf_label, rmsf_label_file,
      residue_set, residue_set_sha256, residue_set_file, residue_set_n_{proteins,residues},
      checkpoint, plus the old fields variant, p_inference, M, d_emb, seed.
  The files WITHOUT the `__<tag>` suffix are the old fixed-40-epoch best-val
  results behind the current paper table; this version never writes them.
  Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): next to each JSON,
  <json stem>.preds.npz (downstream/protein_preds.py: val / test predictions of the
  selected epoch, labels, protein ids, residue index and label row), recorded with its sha256 in
  the JSON's `per_protein_preds`; read only by downstream/protein_level_stats.py, which
  refuses results without it. --redo-missing-preds recomputes an existing result that
  lacks it (otherwise it is skipped with a note).

Dependencies left in the research repo: none in code; all data above lives under
REPO_ROOT. (The constant MDCATH_DIV is not used; the layout for --residue-set all
comes from rmsf_common.mdcath_div_layout.)

How to run
----------
    python downstream/probe_rmsf_eval.py \\
        --variant dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
        --seed 0 --device cuda:0 [--p-inference 1] \\
        [--max-epochs 500 --patience 20 --tag test] [--dataset bioemu] \\
        [--rmsf-label allframes] [--residue-set common]
(the residue sets must exist: data_prep/build_rmsf_residue_sets.py)
`--epochs N` is still accepted as an alias of `--max-epochs N` (old command
lines keep working, but `--epochs 40` would cap early stopping at 40 epochs).
"""
from __future__ import annotations

import os
# Cap BLAS threads: many probes run in parallel on one node.
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# Codebase root and this script's folder on sys.path (sibling imports such as
# tokenize_misato_and_eval resolve to downstream/); data/ and output/ stay in
# the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

from tokenize_misato_and_eval import load_model_robust  # noqa: E402
from ensembits.featurize import checkpoint_provenance  # noqa: E402  code_fix P17
# Common residue set, per-protein Spearman (preretrain_audit_r2 Q2 / Q3, 2026-10-16).
from rmsf_common import (PER_PROTEIN_MIN_RESIDUES, per_protein_fields,  # noqa: E402
                         per_protein_spearman, resolve_residue_set, residue_set_row_mask,
                         residue_set_groups, restrict_splits_to_rows, dataset_layout,
                         FLAT_DATASETS, RESULT_DIRS, check_layout_matches_set,
                         check_label_hash, CacheError, records_changed, large_file_record,
                         strict_same_checkpoint, snapshot_info, check_one_split,
                         splits_record, check_bioemu_split_layout, check_desc_sidecar_rows,
                         check_flat_label_sidecar,
                         row_groups, ResidueCoverageError, constant_prediction_fields)
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import PREDS_KEY, needs_rerun  # noqa: E402
from function_common import atomic_write_json  # noqa: E402

# External dependency: mdCATH-div source data in another user's directory
# (now data/external/mdcath_div via ensembits.paths). Not used below.
from ensembits.paths import MDCATH_DIV_SRC as MDCATH_DIV   # noqa: E402  (unused below)
# Residue-level train/val/test split of mdCATH-div (CATH-H structure split,
# per rebuttal/results_bf65.md); indices are rows of the descriptor cache.
# (mdCATH-div defaults; main() takes the files of --dataset from rmsf_common.FLAT_DATASETS.)
SPLITS = ROOT / 'output' / 'full_train' / 'cached_desc' / 'splits.json'
# Per-residue RMSF label, row-aligned with the descriptor cache (308,293 rows), built by
# data_prep/build_mdcath_div_rmsf_labels.py (2026-10-16; default: all 320 K frames of
# every replica, converged superposition).
from ensembits.paths import MDCATH_DIV_RMSF  # noqa: E402
# FIXED(user decision 2026-10-16, all-frames labels): old = {'converged', 'legacy'}; new =
#   rmsf_common.RMSF_LABEL_FILES['mdcath_div'] = {'allframes' (default), 'converged5' (the
#   former 'converged'), 'legacy'}.
from rmsf_common import RMSF_LABEL_FILES, RMSF_LABEL_DEFAULT, normalize_label_name  # noqa: E402
RMSF_LABELS = RMSF_LABEL_FILES['mdcath_div']  # code_fix P3
RMSF_CACHE = MDCATH_DIV_RMSF
OUT_DIR = ROOT / 'output' / 'probe_rmsf_div'


def tag_of(v): return v.replace('final_model_','').replace('final_model','') or v


def check_cache_rows(n_desc_rows: int, n_rmsf_rows: int, desc_name: str = 'descriptor cache',
                     rmsf_name: str = 'rmsf_div.npy') -> None:
    """Refuse a descriptor cache that is not row-aligned with the RMSF label.

    FIXED(code_fix P13, 2026-10-16): new. The split indices address rows of both files,
    so a cache with a different row count (e.g. a builder that dropped a domain) would
    pair descriptors with the wrong labels silently.
    """
    if int(n_desc_rows) != int(n_rmsf_rows):
        raise ValueError(f'{desc_name} has {n_desc_rows} rows but {rmsf_name} has '
                         f'{n_rmsf_rows}; they must be row-aligned (rebuild the cache)')


# Names a data builder may give the sidecar of a descriptor cache, in order of preference
# (the codebase convention is <file>.provenance.json).
_SIDECAR_NAMES = ('{name}.provenance.json', '{name}.json', '{stem}.json', '{stem}.meta.json',
                  '{stem}_meta.json')


def find_desc_sidecar(desc_path: Path) -> Path | None:
    """The sidecar JSON written next to a descriptor cache by its builder, or None."""
    desc_path = Path(desc_path)
    for pat in _SIDECAR_NAMES:
        q = desc_path.with_name(pat.format(name=desc_path.name, stem=desc_path.stem))
        if q.exists():
            return q
    return None


def check_desc_sidecar(side: dict, convention: str, chain_break_cutoff) -> None:
    """Refuse a cache whose builder settings differ from the checkpoint's.

    FIXED(code_fix P13, 2026-10-16): new. side = the sidecar JSON (keys
    dihedral_convention, chain_break_cutoff, written by the data builder); both cutoffs
    are resolved ("auto" -> 2.5 A for iupac, None for legacy) before comparing.
    """
    from ensembits.descriptor import resolve_chain_break_cutoff
    s_conv = side.get('dihedral_convention')
    if s_conv is not None and s_conv != convention:
        raise ValueError(f'descriptor cache dihedral_convention={s_conv!r}, checkpoint '
                         f'{convention!r}')
    if 'chain_break_cutoff' in side:
        a = resolve_chain_break_cutoff(side['chain_break_cutoff'], s_conv or convention)
        b = resolve_chain_break_cutoff(chain_break_cutoff, convention)
        if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > 1e-6):
            raise ValueError(f'descriptor cache chain_break_cutoff={side["chain_break_cutoff"]!r} '
                             f'(resolved {a}), checkpoint {chain_break_cutoff!r} (resolved {b})')


MAX_EPOCHS = 500    # default cap of the early-stopping loop (--max-epochs)
PATIENCE = 20       # default epochs without val improvement before stopping (--patience)
MIN_DELTA = 1e-4    # a val Spearman gain must exceed this to count as an improvement


def fit_probe_early_stopping(model, tok_tr, rmsf_tr, tok_va, rmsf_va, tok_te, rmsf_te,
                             *, device, seed, batch_size=2048, max_epochs=MAX_EPOCHS,
                             patience=PATIENCE, min_delta=MIN_DELTA,
                             groups_va=None, groups_te=None, return_preds=False):
    """Train `model` (MSE, AdamW lr 3e-4, wd 1e-4) until the val Spearman stops
    improving, restore the best-val weights, and score the test residues once.

    Inputs are tensors already on `device` (tokens long, RMSF float32). An epoch
    counts as an improvement when its val Spearman beats the best so far by more
    than `min_delta`; training stops after `patience` epochs without one, or
    after `max_epochs`. The batch order uses a generator seeded with `seed`, as
    before (the caller seeds torch before building the model).
    groups_va / groups_te: protein index of every val / test residue; when given, the
    per-protein Spearman of the selected (best-val) model is added for val and test
    (rmsf_common.per_protein_spearman; selection stays on the pooled val Spearman).
    return_preds: also return the selected model's val / test predictions (input order) under
    the private key '_preds' (popped by the caller for the per-protein sidecar, A4 (a));
    absent when no epoch was selected.
    Paper: 2_results.tex result:rmsf — "Spearman ... between predicted and
    ground-truth RMSF on held-out residues" (= test_spearman).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    y_va = rmsf_va.cpu().numpy(); y_te = rmsf_te.cpu().numpy()
    best_sp, best_epoch, best_state = float('-inf'), 0, None
    since_best, n_run = 0, 0
    all_const = True          # val predictions constant at every epoch (2026-10-18)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(tok_tr), generator=g)
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s+batch_size].to(device)
            loss = F.mse_loss(model(tok_tr[idx]), rmsf_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        n_run = ep + 1
        model.eval()
        with torch.no_grad(): pv = model(tok_va).cpu().numpy()
        # FIXED(constant-prediction rule, 2026-10-18): record whether the val predictions were
        #   constant at every epoch (rmsf_common.constant_prediction_fields).
        all_const = all_const and bool(np.ptp(pv) == 0)
        # Paper: 2_results.tex result:rmsf — Spearman rank correlation between
        # predicted and ground-truth RMSF.
        sp = spearmanr(pv, y_va).correlation
        # Early stopping on val (NaN, e.g. constant predictions, never improves).
        if sp is not None and np.isfinite(sp) and sp > best_sp + min_delta:
            best_sp, best_epoch, since_best = float(sp), ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    # Restore the best-val weights; evaluate the held-out test residues once.
    pp_va = pp_te = None
    preds = None
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad(): pt = model(tok_te).cpu().numpy()
        # FIXED(final_fix_list A4 (a), 2026-10-19): new: the selected model's val / test
        #   predictions for the per-protein sidecar (protein_preds.py); old: not kept.
        if return_preds:
            with torch.no_grad(): preds = {'val': model(tok_va).cpu().numpy(), 'test': pt}
        test_sp = spearmanr(pt, y_te).correlation
        test_sp = float(test_sp) if test_sp is not None else float('nan')
        # FIXED(preretrain_audit_r2 Q2, 2026-10-16): old = pooled Spearman only. New = also
        #   the per-protein Spearman (mean / median over proteins) of the selected model on
        #   val and test. Eval mode is deterministic, so these val predictions are the ones
        #   of the selected epoch.
        if groups_va is not None:
            with torch.no_grad(): pv_best = model(tok_va).cpu().numpy()
            pp_va = per_protein_spearman(pv_best, y_va, groups_va)
        if groups_te is not None:
            pp_te = per_protein_spearman(pt, y_te, groups_te)
    else:   # val Spearman was never finite
        best_sp, test_sp = float('nan'), float('nan')
    pp = {}
    if groups_va is not None:
        pp.update(per_protein_fields('val', pp_va))
    if groups_te is not None:
        pp.update(per_protein_fields('test', pp_te))
    if pp:
        pp['per_protein_min_residues'] = PER_PROTEIN_MIN_RESIDUES
    return {'test_spearman': test_sp,
            **pp,
            'best_val_spearman': float(best_sp),
            'best_epoch': int(best_epoch),          # 1-based; 0 = never improved
            'n_epochs_run': int(n_run),
            'max_epochs': int(max_epochs),
            'patience': int(patience),
            'min_delta': float(min_delta),
            'selection': 'val_early_stopping',
            'n_train_residues': int(len(tok_tr)),
            'n_val_residues': int(len(tok_va)),
            'n_test_residues': int(len(tok_te)),
            **constant_prediction_fields(best_state is None, all_const, tok_tr, tok_va, tok_te),
            **({'_preds': preds} if return_preds else {})}


# FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein prediction sidecar of the flat
#   RMSF probes (mdCATH-div, BioEmu-CATH; this probe and probe_rmsf_baselines.py).
def write_flat_rmsf_preds(out_path, preds, idx_by_split: dict, rmsf, groups, names,
                          row_starts, meta: dict) -> dict:
    """Write <out>.preds.npz (kind residue_regression) and return the JSON record.

    preds: fit_probe_early_stopping(..., return_preds=True)['_preds'] (None = no selected
    epoch); idx_by_split: {'val', 'test'}: label rows in the order the probe scored them;
    groups: protein index of every row; names / row_starts: id and first row of each protein
    index (residue_index = row - row_start).
    """
    from protein_preds import (write_preds, sidecar_path, residue_split_from_residues,
                               not_selected_record)
    if preds is None:
        return not_selected_record('no epoch selected (val predictions constant or NaN)')
    names = np.asarray(names).astype(str); row_starts = np.asarray(row_starts, dtype=np.int64)
    splits = {}
    for s in ('val', 'test'):
        rows = np.asarray(idx_by_split[s], dtype=np.int64)
        g = np.asarray(groups)[rows]
        splits[s] = residue_split_from_residues(names[g], preds[s], np.asarray(rmsf)[rows],
                                                rows - row_starts[g], row=rows)
    return write_preds(sidecar_path(out_path), kind='residue_regression', splits=splits,
                       meta={'task': 'rmsf', **meta}, json_dir=Path(out_path).parent)


def rmsf_meta(out: dict) -> dict:
    """Facts of an RMSF result copied into its sidecar (identity of items and labels)."""
    keys = ('dataset', 'variant', 'feature', 'mode', 'p_inference', 'seed', 'split',
            'rmsf_label', 'rmsf_label_sha256', 'residue_set_sha256', 'splits_sha256',
            'best_epoch', 'selection', 'per_protein_min_residues')
    m = {k: out.get(k) for k in keys if k in out}
    m['checkpoint_best_pt_sha256'] = (out.get('checkpoint') or {}).get('best_pt_sha256')
    return m


def main(argv=None):
    # ---- CLI -------------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--device', default='cuda:0')
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
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--p-inference', type=int, default=None)
    # FIXED(code_fix P3, 2026-10-16): which mdCATH-div RMSF label to use. "converged"
    #   (default) = data_prep/build_mdcath_div_rmsf_labels.py (superposition iterated to
    #   convergence, pose-independent); "legacy" = the old rmsf_div.npy (3 rounds from the
    #   mean of the raw frames), only to reproduce earlier numbers.
    #   FIXED(user decision 2026-10-16): default 'allframes' = every 320 K frame of the 5
    #   replicas, converged superposition; 'converged5' = the 5-frame label above (was
    #   'converged'); 'legacy' as before.
    ap.add_argument('--rmsf-label', default=RMSF_LABEL_DEFAULT,
                    choices=sorted({n for d in FLAT_DATASETS for n in RMSF_LABEL_FILES[d]}))
    # Phase 2 (2026-10-16): the same probe on BioEmu-CATH (labels, splits, descriptor
    # cache and result folder of rmsf_common.FLAT_DATASETS['bioemu']).
    ap.add_argument('--dataset', default='mdcath_div', choices=tuple(FLAT_DATASETS),
                    help='mdcath_div (default) or bioemu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--desc-p', type=int, default=None,
                    help='which cached div descriptor P to load '
                         '(default: the model num_prototypes)')
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='accept a checkpoint whose training run has not finished (code_fix P12)')
    # FIXED(preretrain_audit_r2 Q3, 2026-10-16): old = every row of the label array (incl.
    #   the 8 Cα-only domains, ideal-geometry descriptor). New = the common residue set of
    #   data_prep/build_rmsf_residue_sets.py (default; refused if absent); "all" keeps the
    #   old behaviour, only to reproduce earlier numbers. Recorded in the JSON.
    ap.add_argument('--residue-set', default='common',
                    help='"common" (default): the common residue set of every method; "all": '
                         'legacy, every label row; or the path of a residue-set JSON')
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: an existing result that passes every reuse
    #   check but has no usable per-protein sidecar is recomputed (and overwritten) instead of
    #   skipped; without the flag it is skipped with a note (the tables accept it, the
    #   significance step protein_level_stats.py refuses it).
    ap.add_argument('--redo-missing-preds', action='store_true',
                    help='recompute an existing result without a per-protein prediction sidecar')
    args = ap.parse_args(argv)
    if not args.tag:
        print('[error] --tag must be non-empty (an empty tag would overwrite the '
              'old best-val result JSONs)'); return 2
    ds = args.dataset
    if args.rmsf_label not in RMSF_LABEL_FILES[ds]:
        print(f'[error] --rmsf-label {args.rmsf_label} does not exist for {ds} '
              f'({sorted(RMSF_LABEL_FILES[ds])})'); return 2
    OUT_DIR = ROOT / 'output' / RESULT_DIRS[ds]
    SPLITS = FLAT_DATASETS[ds]['splits']
    try:
        rs, rs_record = resolve_residue_set(args.residue_set, ds)
    except (FileNotFoundError, ValueError) as e:
        print(f'[error] {e}'); return 1

    # ---- Output name; skip if this (variant, P, seed) is already done ------
    # The "__<tag>" suffix keeps the old best-val JSONs (same stem) untouched.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f'_P{args.p_inference}inf' if args.p_inference is not None else ''
    if args.seed != 0: suffix += f'_seed{args.seed}'
    out_path = OUT_DIR / f'{tag_of(args.variant)}{suffix}__{args.tag}.json'
    # FIXED(preretrain_audit_r2 F9, 2026-10-16): the checkpoint's hash is taken BEFORE the
    #   model is loaded and checked again before the result is written (old: computed at
    #   the end, so a training run that replaced best.pt meanwhile was recorded wrongly).
    prov0 = checkpoint_provenance(ROOT / 'output' / args.variant)
    _label_path = RMSF_LABEL_FILES[ds][args.rmsf_label]
    if out_path.exists():
        # FIXED(preretrain_audit_r2 F5, 2026-10-16): nor one made with other label contents,
        #   or (F1) from a descriptor cache that has changed since.
        _prev = json.loads(out_path.read_text())
        _why = records_changed(_prev.get('feature_caches'))
        if SPLITS.exists() and _prev.get('splits_sha256') != splits_record(SPLITS)['splits_sha256']:
            _why.append('split file')                       # G4 (2026-10-17)
        if _label_path.exists():
            try:
                if _prev.get('rmsf_label_sha256') != check_label_hash(_label_path, np.load(_label_path)):
                    _why.append('label contents')
            except CacheError as e:
                _why.append(str(e))
        if _why:
            print(f'[error] {out_path.name} exists and differs ({", ".join(_why)}); move it aside')
            return 1
        # FIXED(code_fix P3, 2026-10-16): a result made with another RMSF label is not reused.
        _prev_label = normalize_label_name(ds, json.loads(out_path.read_text()).get('rmsf_label', 'legacy'))
        if _prev_label != args.rmsf_label:
            print(f'[error] {out_path.name} was made with the {_prev_label} RMSF label, not '
                  f'{args.rmsf_label}; move it aside'); return 1
        # FIXED(preretrain_audit_r2 Q3, 2026-10-16): new: a result made on another residue
        #   set (or before the common set existed) is not reused either.
        _prev_rs = json.loads(out_path.read_text()).get('residue_set_sha256')
        if _prev_rs != rs_record['residue_set_sha256']:
            print(f'[error] {out_path.name} was made on residue set {_prev_rs!r}, not '
                  f'{rs_record["residue_set_sha256"]!r}; move it aside'); return 1
        # FIXED(code_fix P17, 2026-10-16): an existing result is only kept if it was written
        #   by this checkpoint (or, for a legacy checkpoint, carries no record).
        from ensembits.featurize import check_recorded_checkpoint, ProvenanceMismatchError
        try:
            check_recorded_checkpoint(json.loads(out_path.read_text()).get('checkpoint'),
                                      prov0, out_path.name)
        except ProvenanceMismatchError as e:
            print(f'[error] {e}'); return 1
        _np = needs_rerun(_prev.get(PREDS_KEY), out_path)     # A4 (a), 2026-10-19
        if _np is None or not args.redo_missing_preds:
            print(f'[skip] {out_path.name}' + (f' (note: {_np}; --redo-missing-preds recomputes it)'
                                               if _np else '')); return 0
        print(f'[redo] {out_path.name}: {_np}', flush=True)

    # ---- Load the tokenizer ---------------------------------------------
    # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv.
    variant_dir = ROOT / 'output' / args.variant
    # FIXED(preretrain_audit_r2 H7, 2026-10-16): an unfinished checkpoint gives one clean
    #   [error] line (was a traceback).
    from ensembits.featurize import UnfinishedCheckpointError
    try:
        ens = load_model_robust(variant_dir, device=args.device,
                                allow_unfinished=args.allow_unfinished)
    except UnfinishedCheckpointError as e:
        print(f'[error] {e} -> pass --allow-unfinished, or probe a frozen copy made by '
                  f'downstream/snapshot_checkpoint.py')
        return 1
    cfg = ens.config
    # Ensembits/scripts/train.py does not write descriptor_mode/k (the flagship's
    # config got them from an older trainer), so a run trained with it raises
    # KeyError here -- which previously killed EVERY RMSF-div job in four batteries
    # silently, while the other tasks succeeded. Infer instead of dying.
    # (The reported model has input_dim = 256 -> mode esm3desc_dihedral, k = 16.)
    P_trained = cfg['num_prototypes']
    if 'descriptor_mode' in cfg and 'k' in cfg:
        mode, k = cfg['descriptor_mode'], cfg['k']
    else:
        mode, k = ('esm3desc_dihedral', 16) if cfg.get('input_dim') == 256 else \
                  ('esm3desc', 16) if cfg.get('input_dim') == 192 else (None, None)
        if mode is None:
            print(f"[error] config has no descriptor_mode/k and input_dim="
                  f"{cfg.get('input_dim')} is not recognised; pass them in config.json")
            return 1
        print(f"[warn] config lacks descriptor_mode/k; inferred mode={mode} k={k} "
              f"from input_dim={cfg.get('input_dim')}", flush=True)
    print(f'[setup] {args.variant}  mode={mode} k={k} P={P_trained} '
          f'p_inf={args.p_inference}', flush=True)

    # ---- Pick the cached mdCATH-div descriptor file ------------------------
    # The cached div descriptor is selected by the model's num_prototypes, but a
    # model may have been trained with a decoder narrower than any cached P (e.g.
    # the single-frame trainer uses num_prototypes=1 while only a P10 cache exists).
    # The set encoder accepts any P, so fall back to the largest available cache
    # and let --p-inference decide how many frames are actually fed.
    # Paper: appendix.tex app:baselines — "P=10 on mdCATH-div" for \tool, P=full.
    # code_fix §1: the legacy cache (esm3desc_dihedral_k16_P10.npy) holds φ/ψ + 180°
    #   and serves checkpoints trained before the fix; an iupac checkpoint reads
    #   esm3desc_dihedral_iupac_k16_P10.npy (selected below).
    #   FIXED(code_fix E0, 2026-10-20): iupac eval caches built for the _rv retrain.
    # (require_fix #22, resolved:) the builder (data_prep/build_esm3desc_dihedral_perdataset.py)
    #   fell back to ideal-geometry N/C from Cα for the 8 mdCATH-div domains without a
    #   real backbone file. Since 2026-10-17 all 2,442 have one
    #   (data_prep/build_mdcath_real_bb.py) and the fallback needs --allow-ideal; the
    #   cache sidecar's IDEAL_PIDS must not meet the residue set (G1 check).
    desc_p = args.desc_p or P_trained
    desc_dir = FLAT_DATASETS[ds]['desc_dir']
    # code_fix §1: checkpoints trained with IUPAC dihedrals read the IUPAC cache
    # (esm3desc_dihedral_iupac_k16_P10.npy, built by
    # data_prep/build_esm3desc_dihedral_perdataset.py --dihedral-convention iupac);
    # older checkpoints read the legacy cache (unchanged name).
    if mode == 'esm3desc_dihedral':
        from ensembits.descriptor import with_convention
        mode = with_convention(mode, ens.dihedral_convention)
    desc_path = desc_dir / f'{mode}_k{k}_P{desc_p}.npy'
    # FIXED(preretrain_audit_r2 G1, 2026-10-17): old = a missing P<num_prototypes> cache fell
    #   back silently to the largest other P cache. New = refused; pass --desc-p explicitly
    #   to read another P cache (recorded as desc_cache).
    if not desc_path.exists():
        cands = sorted(q.name for q in desc_dir.glob(f'{mode}_k{k}_P*.npy'))
        print(f'[error] descriptor cache {desc_path.name} missing in {desc_dir}'
              + (f'; available: {cands} (choose one with --desc-p)' if cands and args.desc_p is None
                 else ''))
        return 1

    # ---- Labels and residue split -----------------------------------------
    # FIXED(code_fix §7, §8; require_fix #20, 2026-10-15): the probe used to
    #   report the best-epoch Spearman on "val" (epoch chosen on the residues it
    #   is reported on) and never used "test". Now "val" only selects the epoch
    #   (early stopping) and the reported number is on "test" (30,831 rows / 245 test
    #   domains once all 2,442 domains are in the common residue set, i.e. after
    #   data_prep/build_mdcath_real_bb.py, 2026-10-17; 30,746 / 244 before; see the
    #   module docstring; H9).
    #   This also settles §8: of the 244 val domains only 45 are in the
    #   675-domain BioEmu hold-out list (6 are in the BioEmu-CATH training
    #   corpus), but all 245 test domains are held out.
    splits = json.loads(SPLITS.read_text())
    train_idx = np.array(splits['train']); val_idx = np.array(splits['val'])
    if 'test' not in splits or not len(splits['test']):
        print(f'[error] {SPLITS} has no "test" split; cannot report held-out residues')
        return 1
    test_idx = np.array(splits['test'])
    RMSF_CACHE = RMSF_LABEL_FILES[ds][args.rmsf_label]
    if not RMSF_CACHE.exists():
        print(f'[error] RMSF label {RMSF_CACHE} missing (build it with '
              f'data_prep/build_{"bioemu" if ds == "bioemu" else "mdcath_div"}_rmsf_labels.py)'); return 1
    rmsf = np.load(RMSF_CACHE)
    try:
        label_sha = check_label_hash(RMSF_CACHE, rmsf)    # F5: content hash of the label
    except CacheError as e:
        print(f'[error] {e}'); return 1
    # FIXED(code_fix P13, 2026-10-16): old: no check. New: the cache must have one row per
    #   label row, and if its builder wrote a sidecar (dihedral_convention,
    #   chain_break_cutoff) those must match the checkpoint; refused otherwise.
    n_desc = np.load(desc_path, mmap_mode='r').shape[0]
    try:
        check_cache_rows(n_desc, len(rmsf), desc_path.name, RMSF_CACHE.name)
        # FIXED(preretrain_audit_r2 G1, 2026-10-17): old = without a sidecar the convention
        #   was taken from the file name. New = the builder's sidecar is required and must
        #   name this dataset, the label's row count, the checkpoint's convention / chain-
        #   break cutoff, and no ideal-geometry protein of the residue set.
        side_p = find_desc_sidecar(desc_path)
        if side_p is None:
            raise ValueError(f'{desc_path.name} has no provenance sidecar; rebuild it with '
                             f'data_prep/build_esm3desc_dihedral_perdataset.py')
        side = json.loads(side_p.read_text())
        check_desc_sidecar(side, ens.dihedral_convention, ens.chain_break_cutoff)
        check_desc_sidecar_rows(side, ds, len(rmsf), rs)
        print(f'[desc] sidecar {side_p.name}: dataset / rows / convention / cutoff match',
              flush=True)
        # FIXED(preretrain_audit_r2 G1, 2026-10-17): new: the label's sidecar (required for
        #   the all-frames label) must name this label and row count.
        check_flat_label_sidecar(RMSF_CACHE, ds, args.rmsf_label, len(rmsf))
    except ValueError as e:
        print(f'[error] {e}'); return 1

    # ---- Common residue set and residue -> protein map ----------------------
    # FIXED(preretrain_audit_r2 Q3, 2026-10-16): old = train / val / test = every row of
    #   the splits. New = the rows of the common residue set only (every set row must be
    #   in exactly one split); --residue-set all keeps every row (legacy).
    # FIXED(preretrain_audit_r2 Q2, 2026-10-16): new: protein index of every row, for the
    #   per-protein Spearman (from the residue set; for "all" from the manifest layout).
    try:
        if rs is not None:
            # FIXED(preretrain_audit_r2 F6, 2026-10-16): new: the set's rows must be the
            #   dataset's layout (as in probe_rmsf_baselines.py).
            _, _offsets, _ = dataset_layout(ds)
            check_layout_matches_set(rs, _offsets)
            mask = residue_set_row_mask(rs, len(rmsf))
            sp_r = restrict_splits_to_rows({'train': train_idx, 'val': val_idx,
                                            'test': test_idx}, mask)
            train_idx, val_idx, test_idx = sp_r['train'], sp_r['val'], sp_r['test']
            groups = residue_set_groups(rs, len(rmsf))
            group_names = list(rs['proteins'])            # K5: names in messages
            row_starts = [int(v['row_start']) for v in rs['proteins'].values()]   # A4 (a)
        else:
            _, offsets, total = dataset_layout(ds)
            if total != len(rmsf):
                raise ResidueCoverageError(f'layout has {total} rows, the label {len(rmsf)}')
            groups = row_groups(offsets, total)
            group_names = list(offsets)
            row_starts = [int(s0) for s0, _ in offsets.values()]                 # A4 (a)
    except ResidueCoverageError as e:
        print(f'[error] {e}'); return 1
    # FIXED(preretrain_audit_r2 G4, 2026-10-17): new: every protein in one split only;
    #   BioEmu split built from the current layout; split-file hash recorded.
    try:
        check_one_split(groups, {'train': train_idx, 'val': val_idx, 'test': test_idx},
                        group_names)                     # K5 (2026-10-17): names, not indices
        if ds == 'bioemu':
            check_bioemu_split_layout(SPLITS)
    except (ResidueCoverageError, CacheError) as e:
        print(f'[error] {e}'); return 1
    print(f'[residues] set={rs_record["residue_set"]} train={len(train_idx)} '
          f'val={len(val_idx)} test={len(test_idx)}', flush=True)

    # ---- Standardise and (optionally) keep only the first P frames -----------
    # Paper: 2_results.tex result:rmsf — P=1 feeds one frame through the same
    # set encoder (enabled by SFTD); P=full feeds all 10 mdCATH-div frames.
    desc = np.load(desc_path)
    desc_std = ((desc - ens.mean) / ens.std).astype(np.float32)
    if args.p_inference is not None:
        desc_std = desc_std[:, :args.p_inference, :]

    # ---- Tokenize every residue (level-1 token only) -------------------------
    print(f'[tokenize] residues={desc_std.shape[0]}, P_input={desc_std.shape[1]}', flush=True)
    tokens = np.zeros(desc_std.shape[0], dtype=np.int64)
    bs = 16384
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, desc_std.shape[0], bs):
            x = torch.tensor(desc_std[s:s+bs], device=args.device)
            tokens[s:s+bs] = ens.model.encode(x).cpu().numpy().astype(np.int64)
    print(f'  tokenized in {time.time()-t0:.0f}s', flush=True)

    # ---- Level-1 codebook (RVQ level 0 in the code; plain VQ fallback) -------
    state = ens.model.state_dict()
    cb = state.get('vq_levels.0.codebook.weight')
    if cb is None: cb = state['vq.codebook.weight']
    cb = cb.detach().cpu()
    M = cb.shape[0]

    # ---- Probe: frozen codeword embedding -> 2-hidden-layer MLP -> RMSF ------
    # Paper: 2_results.tex result:rmsf — "a single-frame MLP head is fit on the
    # pooled-residue training set" (per-residue regression, all proteins pooled).
    class Probe(nn.Module):
        def __init__(self, codebook, hidden=128):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(codebook.float(), freeze=True)
            d = codebook.shape[1]
            self.mlp = nn.Sequential(
                nn.Linear(d,hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden,hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden,1))
        def forward(self, t): return self.mlp(self.emb(t)).squeeze(-1)

    # ---- Train (MSE) with early stopping on val, then score test ------------
    # Paper: appendix.tex sec:compute — "RMSF probe (one split, per-residue MLP, 40 ep)"
    # (now: until convergence, require_fix #20 resolved).
    torch.manual_seed(args.seed)
    model = Probe(cb).to(args.device)
    tok_tr = torch.tensor(tokens[train_idx], device=args.device, dtype=torch.long)
    rmsf_tr = torch.tensor(rmsf[train_idx], device=args.device, dtype=torch.float32)
    tok_va = torch.tensor(tokens[val_idx], device=args.device, dtype=torch.long)
    rmsf_va = torch.tensor(rmsf[val_idx], device=args.device, dtype=torch.float32)
    tok_te = torch.tensor(tokens[test_idx], device=args.device, dtype=torch.long)
    rmsf_te = torch.tensor(rmsf[test_idx], device=args.device, dtype=torch.float32)
    res = fit_probe_early_stopping(model, tok_tr, rmsf_tr, tok_va, rmsf_va,
                                   tok_te, rmsf_te, device=args.device,
                                   seed=args.seed, batch_size=args.batch_size,
                                   max_epochs=args.max_epochs,
                                   patience=args.patience,
                                   groups_va=groups[val_idx], groups_te=groups[test_idx],
                                   return_preds=True)            # A4 (a): sidecar below
    preds = res.pop('_preds', None)

    # ---- Write the result JSON ----------------------------------------------
    # FIXED(require_fix #17, tex pass 2026-10-26): RMSF table updated with 256-D numbers.
    out = {'variant': args.variant, 'p_inference': args.p_inference,
           **res, 'M': int(M),
           'd_emb': int(cb.shape[1]), 'seed': args.seed,
           'desc_cache': desc_path.name,
           'rmsf_label': args.rmsf_label, 'rmsf_label_file': RMSF_CACHE.name,
           'dataset': ds, **splits_record(SPLITS),
           **rs_record,   # FIXED(preretrain_audit_r2 Q3, 2026-10-16): new keys residue_set*
           'rmsf_label_sha256': label_sha,               # FIXED(preretrain_audit_r2 F5, 2026-10-16)
           'feature_caches': [large_file_record(desc_path)],   # F1: descriptor cache identity
           'batch_size': args.batch_size,
           # F9: a frozen copy of a running training (downstream/snapshot_checkpoint.py)
           'snapshot_of': (snapshot_info(variant_dir) or {}).get('snapshot_of'),
           'checkpoint': prov0}   # FIXED(code_fix P17, 2026-10-16): new key (old: none)
    if not strict_same_checkpoint(checkpoint_provenance(variant_dir), prov0):
        print(f'[error] {variant_dir.name}/best.pt changed while the probe ran; result not '
              f'written (probe a frozen copy: downstream/snapshot_checkpoint.py)'); return 1
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar <out>.preds.npz (val and
    #   test predictions of the selected model, labels, residue rows) recorded with its sha256
    #   in `per_protein_preds`; the JSON is written atomically (old: write_text).
    out[PREDS_KEY] = write_flat_rmsf_preds(out_path, preds, {'val': val_idx, 'test': test_idx},
                                           rmsf, groups, group_names, row_starts, rmsf_meta(out))
    atomic_write_json(out_path, out)
    print(f'[done] {out_path.name}  test Spearman={res["test_spearman"]:.4f}  '
          f'(per protein {res["test_spearman_per_protein_mean"]:.4f}; '
          f'best val {res["best_val_spearman"]:.4f} at epoch {res["best_epoch"]}, '
          f'{res["n_epochs_run"]} epochs run)', flush=True)
    return 0


# FIXED(code_fix P13, 2026-10-16): old: main() (a refusal returned 1 but the process exited 0).
#   New: the return value is the exit status.
if __name__ == '__main__': sys.exit(main())
