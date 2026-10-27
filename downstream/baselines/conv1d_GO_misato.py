"""GO term prediction on MISATO for the baselines (Conv1d head on fixed per-residue features).

What it does
------------
Same Conv1d classifier and training loop as the baseline EC probe
(`load_features`, `Conv1DClassifier`, `train_classifier`, `metrics` imported
from `ec_classify_conv1d_misato.py` in this folder), on GO labels: the label set
is built by `build_go_labelset` of `downstream/go_labels.py` (same as the Ensembits
probe `tokenize_misato_GO_eval.py`; `--scope top50`: the 50 most frequent
molecular-function terms of `misato_go_mf_v2.json` among the train proteins of the
split, within the band >= 10 and < 50 % of them, ties by GO id).
Proteins without any selected label are dropped. Metrics: `multilabel_metrics.metrics`
(top-1, mAP, muAP, F1@0.5, muF1@0.5, Fmax) plus the label-only frequency floor. Writes
one JSON per (feature, scope, split, tag, seed), with the label sha256, class list and
test-pid sha256.

Paper
-----
  * `appendix.tex` \\label{app:downstreams}, \\paragraph{GO Label Prediction}
    (Table \\label{tab:go_structure}, baseline rows);
  * `appendix.tex` \\label{app:experiments}, \\subsection{GO Label Prediction}
    (structure / sequence / random splits, 10 seeds);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * baselines: `appendix.tex` \\label{app:baselines}.

Origin
------
Copied from `submission_exp/src/conv1d_GO_misato.py`. Changed: `ROOT` from
`ensembits.paths.REPO_ROOT`; `ec_classify_conv1d_misato` is imported from this
folder and `tokenize_misato_GO_eval` from `downstream/` (both were in
`submission_exp/src`); the usage line with a personal interpreter path was
replaced by "How to run"; comments and this docstring.
FIXED(function_reaudit pre-run fixes, 2026-10-19): provenance-checked features of every
split-file complex (`load_features_checked`; aa from misato_real_bb), items from
`function_common.go_items` (missing feature = error, sorted order), item hashes, validation
metrics and the validation frequency floor in r['val'], protocol record, atomic write.
FIXED(function CV, authors' decision option A, 2026-10-19): `--cv-mode {sequence,structure}
--fold r` scores rotation r of the GO cross-validation (data/misato_function_cv_v1.json,
downstream/function_cv.py); classes from its training folds, checked against the CV file;
`function_cv` record in the JSON. `--split` is kept for the old runs.
FIXED(EC/GO capacity check B, 2026-10-19): `--feature aminoaseed_pca128` (non-paper row)
= AminoAseed codebook projected to 128-D by PCA on the codebook rows; implemented in
`ec_classify_conv1d_misato.load_features` (`codebook_pca`), nothing changes here but help.
FIXED(function input standardisation, authors' decision 2026-10-19): the feature's input rule
from `load_features_checked` (lookup tables standardised over their rows; continuous features
over the residues of this split's / rotation's GO training items, applied after go_items);
the JSON records `input_standardisation`. `--no-input-standardisation` = raw (old protocol,
tag + "rawin").
FIXED(benchmark_audit D7, 2026-10-18): GO labels v2 and class selection from
`go_labels.py` (old: raw misato_go.json, top-50 by read order), Fmax, frequency floor
and label / class / test-pid provenance in the result JSON.

Used by the reported model: no (baseline rows of the GO tables). Run by
`downstream/baselines/run_misato_baselines.py`.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/annotations/misato_go_mf_v2.json (+ sidecar), the MISATO split file,
          data/cached_descriptors/<feature cache> (see `load_features`)
          data/misato_function_cv_v1.json (+ sidecar; --cv-mode)
  writes  output/go_conv1d_misato/go_<feature>_<scope>[_<split>][_<tag>][_seed<s>].json
          (with --cv-mode the split part is _cv_<mode>_f<r>)
          Per-protein sidecar (final_fix_list A4 (a), 2026-10-19): <json stem>.preds.npz (class scores of the
          selected model, labels, class list, val and test), written next to
          the JSON (downstream/protein_preds.py) and recorded with its sha256 in
          `per_protein_preds`; read only by downstream/protein_level_stats.py.

How to run
----------
    python downstream/baselines/conv1d_GO_misato.py --feature esm3struct --scope top50 \\
        --split structure --seed 0 \\
        --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine --device cuda:0
    # paper rows (2026-10-19): one rotation of the GO cross-validation instead of --split
    python downstream/baselines/conv1d_GO_misato.py --feature esm3struct --scope top50 \\
        --cv-mode sequence --fold 0 --seed 0 \\
        --tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine --device cuda:0
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))            # tokenize_misato_*_eval.py
sys.path.insert(0, str(Path(__file__).resolve().parent))    # sibling probes in downstream/baselines/
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

# Reuse continuous-feature loaders + Conv1DClassifier from the EC pipeline
from ec_classify_conv1d_misato import (  # noqa: E402
    load_features, Conv1DClassifier, collate, evaluate, metrics, train_classifier,
    load_split_map, misato_splits_record, MISATO_SPLITS, load_features_checked,
)
from function_common import (go_items, go_val_floor, items_record, protocol_record,  # noqa: E402
                             atomic_write_json, add_input_std_arg, input_std_tag)
# Per-protein prediction sidecars (final_fix_list A4 (a), 2026-10-19).
from protein_preds import PREDS_KEY, write_function_preds  # noqa: E402
# EC / GO cross-validation (authors' decision option A, 2026-10-19).
from function_cv import (add_cv_args, check_cv_args, load_function_cv, cv_split_map,  # noqa: E402
                         cv_result_block, check_classes, cv_tag)
# GO labels v2 + class selection shared with the Ensembits probe (benchmark_audit D7).
from go_labels import GO_LABELS_V2 as GO_LABELS, load_go_labels, build_go_labelset, go_result_fields  # noqa: E402
OUT_DIR = ROOT / 'output' / 'go_conv1d_misato'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature', required=True,
                    help='aa | esm2 | mini3di | protprofile_K5 | aminoaseed_pca128 (non-paper '
                         'capacity check) | any feature of ec_classify_conv1d_misato.load_features')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--scope', choices=['top50', 'top200', 'all'],
                    default='top50')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--split', default='sequence',
                    choices=['sequence', 'structure', 'random'],
                    help='fixed v2 split (old protocol; the paper rows use --cv-mode / --fold)')
    # FIXED(function CV, authors' decision option A, 2026-10-19): new --cv-mode / --fold.
    add_cv_args(ap)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--cosine', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--allow-unverified-cache', action='store_true',
                    help='accept a feature cache without a current provenance sidecar (the '
                         'result is marked unverified; compilers refuse it)')
    add_input_std_arg(ap)              # FIXED(function input standardisation, 2026-10-19)
    args = ap.parse_args()
    is_cv = check_cv_args(ap, args)
    input_std_tag(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # FIXED(function_reaudit pre-run fixes 1 / 2, 2026-10-19): every split-file complex,
    #   provenance-checked cache (old: load_features without checks; aa on 3,903 complexes).
    feats, in_dim, feat_recs, in_std = load_features_checked(
        args.feature, args.allow_unverified_cache, standardise=not args.no_input_standardisation)
    print(f'[setup] feature={args.feature}  in_dim={in_dim}  '
          f'n_proteins={len(feats)}  scope={args.scope} '
          + (f'cv={args.cv_mode} fold={args.fold}' if is_cv else f'split={args.split}'), flush=True)

    split_rec = misato_splits_record(MISATO_SPLITS)   # FIXED(benchmark_audit D3, 2026-10-18)
    cv_block = {}
    if is_cv:                                         # function CV (2026-10-19)
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
          f'n_pids_with_labels={len(labels)}', flush=True)

    # FIXED(function_reaudit pre-run fixes 2 / 6, 2026-10-19): function_common.go_items: every
    #   labelled complex of the split must have a feature (old: dropped silently), sorted order.
    bucket = go_items(labels, classes, split_map, feats, args.feature)
    # FIXED(function input standardisation, 2026-10-19): continuous features standardised over
    #   the residues of these training items (lookup features: already over their table rows).
    bucket, std_rec = in_std.apply(bucket)
    print(f'  splits: train={len(bucket["train"])} val={len(bucket["val"])} '
          f'test={len(bucket["test"])}', flush=True)
    if not bucket['train'] or not bucket['test']:
        print('ERR: empty split'); return 1

    train_class_counts = np.zeros(n_classes)
    for it in bucket['train']: train_class_counts += it['y']
    maj_class = int(train_class_counts.argmax())
    maj_top1 = float(np.mean([it['y'][maj_class] > 0 for it in bucket['test']]))

    t0 = time.time()
    r = train_classifier(bucket['train'], bucket['val'], bucket['test'],
                          n_classes, in_dim, args.device,
                          epochs=args.epochs, batch_size=args.batch_size,
                          seed=args.seed, patience=args.patience,
                          min_epochs=args.min_epochs, warmup=args.warmup,
                          cosine=args.cosine, return_preds=True)
    preds = r.pop('_preds', None)                     # A4 (a): sidecar below
    r['majority_top1_hit'] = maj_top1
    r['runtime_s'] = round(time.time() - t0, 1)
    r['scope'] = args.scope
    # Label file, classes, kept / dropped counts, test-pid hash, frequency floor (D7).
    r.update(go_result_fields(go_info, go_sel, classes, bucket, split_map, feats, go_raw))
    # function_reaudit 2026-10-19: item hashes, val frequency floor, inputs, protocol tag.
    r.update(items_record(bucket))
    r['val']['frequency_floor'] = go_val_floor(go_sel, classes, bucket)
    r.update(protocol_record(feat_recs, input_std=std_rec))
    sp_tag = (f'_{cv_tag(args.cv_mode, args.fold)}' if is_cv else
              '' if args.split == 'sequence' else f'_{args.split}')
    seed_tag = '' if args.seed == 0 else f'_seed{args.seed}'
    r['seed'] = args.seed
    out_path = OUT_DIR / f'go_{args.feature}_{args.scope}{sp_tag}{("_"+args.tag) if args.tag else ""}{seed_tag}.json'
    # FIXED(final_fix_list A4 (a), 2026-10-19): new: per-protein sidecar <out>.preds.npz (class
    #   scores of the selected model + labels, val and test, class list) in `per_protein_preds`.
    r[PREDS_KEY] = write_function_preds(
        out_path, preds, r, task='GO', classes=classes, method=args.feature,
        split=f'cv/{args.cv_mode}/fold{args.fold}' if is_cv else args.split,
        cv_mode=args.cv_mode, cv_fold=args.fold)
    atomic_write_json(out_path, {**r, 'feature': args.feature, 'in_dim': in_dim,
                                 'split': (f'cv/{args.cv_mode}/fold{args.fold}' if is_cv
                                           else args.split),
                                 'cv_mode': args.cv_mode, 'cv_fold': args.fold,
                                 **split_rec, **cv_block})   # D3 split file; CV record
    # (test metrics are not printed; 2026-10-19)
    print(f'  GO/{args.scope}: [ep={r["epochs_run"]}, {r["runtime_s"]}s, n_test={r["n_test"]}]',
          flush=True)
    print(f'[done] {out_path}', flush=True)


if __name__ == '__main__': main()
