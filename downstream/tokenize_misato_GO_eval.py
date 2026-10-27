"""GO term prediction on MISATO from level-1 Ensembits tokens.

What it does
------------
  1. Loads the level-1 token cache and the level-1 codebook through
     `tokenize_misato()` (from `tokenize_misato_and_eval.py`; the cache is built
     by `_pretokenize_misato_dihedral.py`).
  2. Builds the GO label set from the training proteins of the chosen split
     (`downstream/go_labels.py`, authors' decision D7 (a)): molecular-function terms
     of `data/annotations/misato_go_mf_v2.json` (propagated, cleaned) annotated in
     >= 10 and in < 50 % of the training proteins, ranked by (-count, GO id):
       - top50 : the first 50 (the setting reported in the paper)
       - top200: the first 200
       - all   : every such term
     Proteins with none of the selected terms are dropped (counts in the JSON).
  3. Trains the same multi-label 1D-CNN head as EC (`TokenConv1D` /
     `train_one`, multi-label BCE, early stopping on validation BCE) and scores
     the test split with top-1 hit, macro / micro AP, macro / micro F1@0.5 and
     protein-centric Fmax (`multilabel_metrics.metrics`); the frequency floor
     (training frequency as the prediction, labels only) is stored next to them.
  4. Writes one JSON per (variant, split, seed, tag, scope), with the label file's
     sha256, the class list and the sha256 of the test pid list.

The split comes from `ensembits.paths.MISATO_SPLITS` (data/misato_splits_v2.json, v2
since benchmark_audit D3, 2026-10-18; its path + sha256 go into the result JSON) via
`load_split_map` (the
original docstring said it came from `aa2ensembits_pairs_misato_v2.pkl`; the
code does not use that file).

Paper
-----
  * `2_results.tex`, \\subsection{Does \\tool alphabet carry functional
    information?} \\label{sec:downstreams}: "top-50 Gene Ontology term
    prediction".
  * `appendix.tex`, \\section{Downstream task experiments}
    \\label{app:downstreams}, \\paragraph{GO Label Prediction} (50 most
    frequent training terms, multi-label; architecture, optimizer, loss and
    metrics identical to EC; Table \\label{tab:go_structure}).
  * `appendix.tex`, \\section{All Experiments} \\label{app:experiments},
    \\subsection{GO Label Prediction} (top-50 x structure / sequence / random
    splits, mean +- std over 10 seeds).

Origin
------
Copied from `submission_exp/src/tokenize_misato_GO_eval.py`. Changes: sys.path
setup only (ROOT now comes from `ensembits.paths`; the script's own folder is put
on sys.path so the sibling `tokenize_misato_and_eval` import resolves), plus
comments. Logic, defaults, file names and CLI flags are unchanged.
Fixes 2026-10-16 (FIXED markers): `checkpoint` block in the result JSON (code_fix
P17); --allow-unfinished (P12).
FIXED(benchmark_audit D7, 2026-10-18): GO labels v2 (MF only, propagated, cleaned;
old: raw misato_go.json MF + BP + CC), class band and deterministic tie-break, Fmax and
frequency floor, label / class / test-pid provenance in the result JSON
(`go_labels.py`, `multilabel_metrics.py`).

FIXED(function CV, authors' decision option A, 2026-10-19): `--cv-mode {sequence,structure}
--fold r` scores rotation r of the GO cross-validation (data/misato_function_cv_v1.json via
downstream/function_cv.py: distinct proteins, grouped label-stratified 5 folds; classes from
the three training folds, checked against the CV file); the JSON records the CV file
(path, sha256), mode and fold (`function_cv`). `--split` is kept for the old runs.

FIXED(function_reaudit pre-run fixes, 2026-10-19): items from function_common.go_items
(missing tokens = error, sorted order), zero-padded embedding (TokenConv1D, Q23), item
hashes, validation metrics + validation frequency floor in r['val'], token-cache record and
protocol tag, atomic write, test metrics not printed.

FIXED(function input standardisation, authors' decision 2026-10-19): the frozen codebook is
standardised per dimension over its rows by default (tokenize_misato_and_eval
.codebook_input_std, the rule of every lookup feature of every method); the result JSON
records `input_standardisation`. `--no-input-standardisation` = raw codebook (old protocol,
tag + "rawin"). Old: the non-paper opt-in `--standardize-codebook` (removed).

Used by the reported model: yes. `downstream/_run_model_battery.py` runs it with
`--scope top50` for 3 splits x seeds 0-9 x {P=full, P=1} and `--tag stab
--warmup 5 --min-epochs 20 --patience 12 --cosine`.

Inputs / outputs
----------------
Read  : `output/<variant>/` (checkpoint), the token cache
        `data/cached_descriptors/ensembits_<tag>_tokens_misato.npz`,
        `data/annotations/misato_go_mf_v2.json` (+ sidecar; MF terms per pid),
        `data/misato_splits_v2.json`, `data/misato_function_cv_v1.json` (--cv-mode)
Write : `output/go_conv1d_misato/go_<tag><suffix>_<scope>.json` (suffix:
        [_P<k>inf][_cv_<mode>_f<r> | _<split>][_<tag>][_seed<s>])
        Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): <json stem>.preds.npz (class scores of the
        selected model, labels, class list, val and test), written next to
        the JSON (downstream/protein_preds.py) and recorded with its sha256 in
        `per_protein_preds`; read only by downstream/protein_level_stats.py.
All paths are relative to REPO_ROOT (the research repo).

How to run
----------
  python downstream/tokenize_misato_GO_eval.py \\
      --variant dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
      --device cuda:0 --scope top50 --split structure --seed 0 \\
      --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine \\
      [--p-inference 1]
  # paper rows (2026-10-19): one rotation of the GO cross-validation instead of --split
  python downstream/tokenize_misato_GO_eval.py --variant <variant> --device cuda:0 \\
      --scope top50 --cv-mode sequence --fold 0 --seed 0 --tag stab --warmup 5 \\
      --min-epochs 20 --patience 12 --cosine [--p-inference 1]
"""
from __future__ import annotations

import os
# Keep CPU thread pools small (several of these jobs run in parallel).
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, json, os, pickle, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

# Codebase root and this script's folder (for the sibling import below) on
# sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

# Reuse helpers + model from the EC pipeline
from tokenize_misato_and_eval import (  # noqa: E402
    load_model_robust, tokenize_misato, TokenConv1D, collate, evaluate, metrics,
    train_one, load_split_map, MISATO_SPLITS, misato_splits_record,  # D3: v2 split file
)
from ensembits.featurize import checkpoint_provenance  # noqa: E402  code_fix P17
from tokenize_misato_and_eval import token_cache_path, codebook_input_std  # noqa: E402  (2026-10-19)
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import PREDS_KEY, write_function_preds  # noqa: E402
from function_common import (go_items, go_val_floor, items_record, protocol_record,  # noqa: E402
                             token_cache_record, atomic_write_json, add_input_std_arg,
                             input_std_tag)
# EC / GO cross-validation (authors' decision option A, 2026-10-19).
from function_cv import (add_cv_args, check_cv_args, load_function_cv, cv_split_map,  # noqa: E402
                         cv_result_block, check_classes, cv_tag)

# FIXED(benchmark_audit D7, 2026-10-18): GO labels v2 = molecular function only,
#   propagated over is_a + part_of on a pinned go-basic.obo, obsolete / do-not-annotate /
#   root / protein binding removed (data_prep/build_misato_go_labels.py). Old: the raw
#   misato_go.json (MF + BP + CC, unpropagated), taken as is.
# Open(audit D.5/D.6): appendix.tex app:downstreams still describes the old GO labels
#   (union of MF / BP / CC). The code now uses GO MF only (v2). Text pass item.
from go_labels import GO_LABELS_V2 as GO_LABELS, load_go_labels, build_go_labelset, go_result_fields  # noqa: E402
EC_LABELS = ROOT / 'data' / 'annotations' / 'misato_ec.json'  # for split align (not used below)
OUT_DIR = ROOT / 'output' / 'go_conv1d_misato'


def tag_of(v: str) -> str:
    """Output tag of a checkpoint folder name (strips `final_model`)."""
    return v.replace('final_model_', '').replace('final_model','') or v


# FIXED(benchmark_audit D7, 2026-10-18): `build_go_labelset` moved to go_labels.py
#   (imported above). Old: Counter.most_common over the training proteins with features
#   (ties at rank 50 broken by read order), no frequency band, raw labels. New: classes
#   from every training pid of the split file (same classes for every method), terms in
#   >= 10 and < 50 % of the training proteins, ranked by (-count, GO id); it now returns
#   (labels, classes, selection info).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--scope', choices=['top50', 'top200', 'all'],
                    default='top50',
                    help='GO label scope: top-50/200 most common, or all with ≥10 train occurrences')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--p-inference', type=int, default=None)
    ap.add_argument('--random-init-embedding', action='store_true')
    ap.add_argument('--rand-emb-dim', type=int, default=128)
    ap.add_argument('--split', default='sequence',
                    choices=['sequence', 'structure', 'random'],
                    help='fixed v2 split (old protocol; the paper rows use --cv-mode / --fold)')
    # FIXED(function CV, authors' decision option A, 2026-10-19): new --cv-mode / --fold
    #   (distinct proteins, grouped label-stratified 5-fold CV; downstream/function_cv.py).
    #   Old (still available): --split.
    add_cv_args(ap)
    # Paper: appendix.tex \label{app:experiments} — mean +- std over 10 seeds (battery: --seed 0..9).
    ap.add_argument('--seed', type=int, default=0)
    # Head stabilization recipe. The defaults reproduce the original protocol;
    # the paper's numbers use --warmup 5 --min-epochs 20 --patience 12 --cosine
    # (RECIPE in _run_model_battery.py). FIXED(require_fix #19, 2026-10-15): appendix
    # "Probe training" paragraph now lists these values.
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--tag', default='')
    # FIXED(function input standardisation, 2026-10-19): codebook standardised by default;
    #   --no-input-standardisation = raw (old protocol). Old: opt-in --standardize-codebook.
    add_input_std_arg(ap)
    ap.add_argument('--allow-unfinished', action='store_true',
                    help='accept a checkpoint whose training run has not finished (code_fix P12)')
    args = ap.parse_args()
    is_cv = check_cv_args(ap, args)
    input_std_tag(args)
    stab = dict(min_epochs=args.min_epochs, warmup=args.warmup, cosine=args.cosine)

    # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variant_dir = ROOT / 'output' / args.variant
    if not variant_dir.exists():
        print(f'ERR: {variant_dir} does not exist'); return 2

    # Level-1 tokens per pid + the level-1 codebook used as the token embedding.
    feats, codebook, M = tokenize_misato(variant_dir, device=args.device,
                                          p_inference=args.p_inference,
                                          allow_unfinished=args.allow_unfinished)
    # FIXED(function input standardisation, 2026-10-19): codebook standardised over its rows.
    codebook, in_std = codebook_input_std(codebook, args.variant,
                                          enabled=not args.no_input_standardisation,
                                          random_init=args.random_init_embedding,
                                          rand_emb_dim=args.rand_emb_dim)
    print(f'[input-std] {in_std.base_record()["rule"]}', flush=True)
    in_dim = codebook.shape[1]
    print(f'[setup] variant={args.variant}  M={M}  d_emb={in_dim}  '
          f'n_proteins={len(feats)}  scope={args.scope}', flush=True)

    # Class list from the training proteins of this split.
    split_rec = misato_splits_record(MISATO_SPLITS)   # FIXED(benchmark_audit D3, 2026-10-18)
    cv_block = {}
    if is_cv:
        # FIXED(function CV, 2026-10-19): rotation `fold` of the GO CV (representatives only);
        #   the classes come from its three training folds.
        cv, cv_rec = load_function_cv()
        split_map = cv_split_map(cv, 'go', args.cv_mode, args.fold)
        cv_block = cv_result_block(cv_rec, cv, 'go', args.cv_mode, args.fold)
    else:
        split_map = load_split_map(args.split)
    go_raw, go_info = load_go_labels(GO_LABELS)

    # Classes from every training pid of the split (not only those with features).
    train_pids = [p for p, s in split_map.items() if s == 'train']
    labels, classes, go_sel = build_go_labelset(go_raw, train_pids, scope=args.scope)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    if is_cv and args.scope == 'top50':               # must equal the CV file's classes
        check_classes(cv, 'go', args.cv_mode, args.fold, classes)
    print(f'[scope={args.scope}] n_classes={n_classes}, '
          f'n_proteins_with_labels={len(labels)}', flush=True)

    # Multi-hot targets per protein, bucketed by split; val/test proteins
    # without any in-scope term are dropped.
    # FIXED(function_reaudit pre-run fixes 2 / 6, 2026-10-19): function_common.go_items: every
    #   labelled complex of the split must have tokens (old: dropped silently), sorted order.
    bucket = go_items(labels, classes, split_map, feats, f'Ensembits {args.variant}')
    bucket, std_rec = in_std.apply(bucket)            # 2026-10-19 (lookup: tokens unchanged)
    print(f'  splits: train={len(bucket["train"])} val={len(bucket["val"])} '
          f'test={len(bucket["test"])}', flush=True)

    if not bucket['train'] or not bucket['test']:
        print('ERR: empty split'); return 1

    # Majority top-1 baseline (predict the most common class for everyone)
    train_class_counts = np.zeros(n_classes)
    for it in bucket['train']: train_class_counts += it['y']
    maj_class = int(train_class_counts.argmax())
    maj_top1 = float(np.mean([it['y'][maj_class] > 0 for it in bucket['test']]))

    # Same head, optimizer, loss and metrics as EC.
    # Paper: appendix.tex \label{app:downstreams} \paragraph{GO Label Prediction} — "identical to EC".
    t0 = time.time()
    r = train_one(bucket['train'], bucket['val'], bucket['test'],
                   codebook, n_classes, args.device,
                   epochs=args.epochs, batch_size=args.batch_size,
                   random_init=args.random_init_embedding,
                   rand_emb_dim=args.rand_emb_dim, seed=args.seed,
                   patience=args.patience, return_preds=True, **stab)
    preds = r.pop('_preds', None)                     # A4 (a): sidecar below
    r['majority_top1_hit'] = maj_top1
    r['runtime_s'] = round(time.time() - t0, 1)
    r['scope'] = args.scope
    # Label file, classes, kept / dropped counts, test-pid hash, frequency floor (D7).
    r.update(go_result_fields(go_info, go_sel, classes, bucket, split_map, feats, go_raw))
    # function_reaudit 2026-10-19: item hashes, val frequency floor, inputs, protocol tag.
    r.update(items_record(bucket))
    r['val']['frequency_floor'] = go_val_floor(go_sel, classes, bucket)
    r.update(protocol_record([token_cache_record(token_cache_path(variant_dir, args.p_inference))],
                             input_std=std_rec))

    # Output file name encodes P mode, embedding init, split, tag and seed.
    suffix = ''
    if args.p_inference is not None: suffix += f'_P{args.p_inference}inf'
    if args.random_init_embedding: suffix += f'_randEmb{args.rand_emb_dim}'
    if is_cv: suffix += f'_{cv_tag(args.cv_mode, args.fold)}'        # function CV (2026-10-19)
    elif args.split != 'sequence': suffix += f'_{args.split}'
    if args.tag: suffix += f'_{args.tag}'
    if args.seed != 0: suffix += f'_seed{args.seed}'
    r['seed'] = args.seed
    out_path = OUT_DIR / f'go_{tag_of(args.variant)}{suffix}_{args.scope}.json'
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar <out>.preds.npz (class
    #   scores of the selected model + labels, val and test, class list) in `per_protein_preds`.
    ckpt_block = checkpoint_provenance(variant_dir)
    r[PREDS_KEY] = write_function_preds(
        out_path, preds, r, task='GO', classes=classes, method=f'Ensembits {args.variant}',
        p_inference=args.p_inference,
        split=f'cv/{args.cv_mode}/fold{args.fold}' if is_cv else args.split,
        cv_mode=args.cv_mode, cv_fold=args.fold,
        checkpoint_best_pt_sha256=(ckpt_block or {}).get('best_pt_sha256'))
    atomic_write_json(out_path, {**r, 'variant': args.variant,
                                     'p_inference': args.p_inference,
                                     'random_init_embedding': args.random_init_embedding,
                                     'rand_emb_dim': args.rand_emb_dim,
                                     'M': int(M), 'd_emb': int(in_dim),
                                     # FIXED(code_fix P17, 2026-10-16): checkpoint provenance.
                                     'checkpoint': ckpt_block,
                                     'split': (f'cv/{args.cv_mode}/fold{args.fold}' if is_cv
                                               else args.split),
                                     'cv_mode': args.cv_mode, 'cv_fold': args.fold,
                                     **split_rec, **cv_block})   # D3 split file; CV record
    # (test metrics are not printed; 2026-10-19)
    print(f'  GO/{args.scope}: [ep={r["epochs_run"]}, {r["runtime_s"]}s, n_test={r["n_test"]}]',
          flush=True)
    print(f'[done] {out_path}', flush=True)


if __name__ == '__main__':
    # FIXED(code_fix P19, 2026-10-16): propagate main()'s return code (refusals exit non-zero).
    sys.exit(main())
