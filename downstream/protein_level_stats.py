"""Protein-level significance tests of one method against another, from the prediction sidecars.

What it does
------------
Authors' decision A4 (a) (camera_ready/final_fix_list.md, 2026-10-19): the differences between
two methods on RMSF, binding site, affinity, EC and GO are tested over PROTEINS (not over
probe seeds), from the per-protein prediction sidecars that every probe writes next to its
result JSON (downstream/protein_preds.py). Nothing is re-run and no label file is read: the
labels come from the sidecars (checked to be identical for the two methods); only the cluster
definitions are read from the dataset files named below.

  1. Loading (`load_runs`): each result JSON must carry a `per_protein_preds` record whose
     sidecar exists with the recorded sha256 (else the comparison is REFUSED; the tables still
     accept such results). Runs of one method are its probe seeds; their item ids, residue
     indices, labels, classes and item / residue-set hashes must agree, the seeds must differ.
  2. Seed averaging first (`seed_average`): the predictions (RMSF values, binding
     probabilities, class scores, affinities) are averaged over the seeds of a method, then
     the metric is computed on the averaged predictions.
  3. Paired cluster bootstrap of Delta = metric(A) - metric(B) (`paired_cluster_bootstrap`):
     the two methods are scored on the same items; clusters are drawn with replacement and a
     drawn cluster brings all its items (B replicates, fixed seed). Reported: point Delta,
     percentile 95 % CI, bootstrap SE and a two-sided p-value from the centred bootstrap
     distribution, p = (1 + #{|Delta*_b - Delta| >= |Delta|}) / (B + 1).
     Clusters (`cluster_map`):
       * RMSF mdCATH-div: CATH topology of the domain (data/external/mdcath_div/_manifest.json
         representatives[pid].topology, as the ANOVA);
       * RMSF BioEmu-CATH: CATH topology (first three fields of the H superfamily in
         output/phase_b/cached_desc/bioemu/splits.json `h_superfamily`; system cath1_/cath2_
         <domain> -> domain); a domain without CATH is its own cluster;
       * RMSF MISATO and binding site / affinity: the group of the v2 split the run used
         (sequence split: sequence groups; structure split: structure groups; random split:
         sequence groups, i.e. the same protein), rebuilt from the chain table with the
         functions of data_prep/build_misato_splits_v2.py (the group counts equal the split
         file's stats, 764 sequence / 308 structure groups, and no group straddles two
         parts). For MISATO RMSF a CATH-topology rule (`misato_clusters('topology')`, the
         structure-group rule on topologies) is available but not the default: multi-domain
         complexes chain the topologies into one component of 3,009 / 4,894 complexes
         (199 / 489 structure-split test complexes), which would dominate every replicate;
         the structure groups (H level, 91 clusters on that test split, largest 38) are used;
       * EC / GO: the CV groups of the run's mode (data/misato_function_cv_v1.json
         tasks[task].modes[mode].groups); items are distinct proteins (representatives).
  4. Metrics (all on seed-averaged predictions):
       * RMSF: `spearman_per_protein` (mean over proteins with >= 10 residues and
         non-constant labels; constant predictions score 0, the rule of
         rmsf_common.per_protein_spearman; the table's per-protein number) and
         `spearman_pooled` (pooled residues of the drawn proteins);
       * binding: `auroc_per_protein` (mean over proteins with both classes), `auroc_pooled`,
         `ap_pooled` (residues of the drawn proteins);
       * affinity: `spearman`, `r2`, `mse` over complexes;
       * EC / GO: `protein_ap` (mean over proteins of the average precision of the protein's
         class scores against its labels, protein-centric) and `fmax`
         (multilabel_metrics.fmax, CAFA protein-centric). EC / GO are run per CV fold
         (rotation r: test = fold r): Delta per fold with its own bootstrap, and the
         fold-paired mean Delta (mean over the folds; each replicate resamples the clusters of
         every fold independently and averages the folds' Delta). Only the fold-paired mean
         enters the Holm family; the per-fold values are reported next to it.
  5. Holm over a PRE-SPECIFIED family (`run_family`): a JSON spec lists every comparison
     (id, task, metric, files of method A and B per seed [and per fold]); all of them are
     tested and Holm-adjusted together; the spec's sha256 is recorded in the output. A family
     with any refused result is not run. A spec with a `families` list is a BUNDLE: every
     family in it is run and Holm-adjusted separately (`run_bundle`).
  6. CryptoBench (2026-10-21, camera_ready/cryptobench_audit/DECISIONS.md CB6 / CB7):
     task `cryptobench` (items = apo entries, residue_binary sidecars of
     downstream/cryptobench/; metrics as binding). Its clusters are not derived from a run
     but given in the spec: `clusters` = {path, sha256} of a JSON {entry: cluster} (30 %
     identity clusters of the test entries; sha256 checked). Pre-declared sensitivity sets
     via a comparison's `subset`: `min_positives` (keep entries with >= k positives) and
     `mask` = {path[, sha256]} of a residue mask JSON (schema cryptobench-residue-mask-v1:
     {entries: {entry: {n_res, drop: [residue index]}}}; the dropped residues must be
     labelled 0, i.e. only negatives are masked). Subsets are applied to the seed-averaged
     blocks of both methods identically, before the metric.

Paper
-----
Not in the paper yet (Open, audit D.17: appendix section on significance with the paired
cluster-bootstrap CIs: RMSF by CATH topology, binding by protein group, EC/GO fold-paired
Δ with per-fold values; the EC Welch sentence is dropped; text pass item).

Origin
------
New (2026-10-19, authors' decision A4 (a)). Holm as in significance_tests.holm (copied, so
this module does not import the rebuttal compiler).

Changes 2026-10-21 (CryptoBench port, step 5): task `cryptobench`, spec-given clusters,
`subset`, bundles, `probe` in IDENTITY_KEYS (a comparison never pairs two probes).
Changes 2026-10-22 (CryptoBench port re-audit CB4 F1 / F2 / F4 / F7):
  * a spec (or bundle) key `seeds` (list) makes every method's seed set equal that list,
    else the family is refused (extra / missing seed files were averaged in silently);
  * a spec `split` must equal the --split of the run;
  * a family containing a `cryptobench` comparison is refused if any p is not finite (Holm
    must not shrink the pre-registered family); other tasks unchanged;
  * every comparison records the sha256 of each loaded sidecar (`sidecars_a`, `sidecars_b`),
    so a stale result file can be detected.

Used by the reported model: evaluation only. Not run yet (the probe reruns with sidecars come
first); tests on synthetic data in tests/test_protein_level_stats.py.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   <result>.json + its sidecar(s) listed in the family spec;
          data/external/mdcath_div/_manifest.json, output/phase_b/cached_desc/bioemu/splits.json,
          data/misato_splits_v2.chains.json, data/misato_function_cv_v1.json (clusters only)
  writes  --out <file>.json (results, Holm-adjusted p, spec sha256, cluster sources)

How to run
----------
    python downstream/protein_level_stats.py --family <spec.json> --out <result.json> \\
        [--split test] [--check-only] [--workers N]
`--check-only` loads and verifies every sidecar and cluster map and stops (no statistic).
`--workers N` runs the comparisons of a family in a process pool of N workers (default 1,
serial).  Each comparison reseeds its own generator, so results are bit-identical to serial.
Spec format: see `example_family()` / the docstring of `run_family`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402
from protein_preds import PREDS_KEY, PredsError, load_preds, file_sha256  # noqa: E402
from rmsf_common import PER_PROTEIN_MIN_RESIDUES  # noqa: E402

STATS_VERSION = 'protein-level-stats-v1'
DEFAULT_B = 10000
DEFAULT_SEED = 20261019
CI_LEVEL = 0.95
METRICS = {
    'rmsf': ('spearman_per_protein', 'spearman_pooled'),
    'binding': ('auroc_per_protein', 'auroc_pooled', 'ap_pooled'),
    'affinity': ('spearman', 'r2', 'mse'),
    'EC': ('protein_ap', 'macro_mAP', 'fmax'),
    'GO': ('protein_ap', 'macro_mAP', 'fmax'),
    'cryptobench': ('auroc_per_protein', 'auroc_pooled', 'ap_pooled'),
}
_RESIDUE_TASKS = ('rmsf', 'binding', 'cryptobench')


class RefusedError(PredsError):
    """A result cannot enter the significance step (no / changed sidecar, mismatched runs)."""


# ---------------------------------------------------------------- loading
def _block_of(js: dict, block):
    if block is None:
        return js
    blk = (js.get('results') or {}).get(block)
    if blk is None:
        raise RefusedError(f'no block {block!r} in the result')
    return blk


def load_run(json_path, block=None) -> dict:
    """One probe run: {'path', 'meta', 'classes', 'val', 'test', 'record'} (sidecar verified).

    For a binding / affinity JSON, `block` = '<split>_<task>' (e.g. 'structure_binding').
    Also checks that the sidecar's item hashes are the ones recorded in the JSON."""
    json_path = Path(json_path)
    js = json.loads(json_path.read_text())
    blk = _block_of(js, block)
    rec = blk.get(PREDS_KEY)
    try:
        run = load_preds(rec, json_path)
    except PredsError as e:
        raise RefusedError(str(e)) from None
    meta = run['meta']
    for k in ('items_sha256', 'classes_sha256', 'go_classes_sha256'):
        if k in blk and k in meta and blk[k] != meta[k]:
            raise RefusedError(f'{json_path.name}: sidecar {k} differs from the result JSON')
    for k in ('residue_set_sha256', 'rmsf_label_sha256', 'splits_sha256'):
        if k in js and k in meta and js[k] != meta[k]:
            raise RefusedError(f'{json_path.name}: sidecar {k} differs from the result JSON')
    run.update({'path': str(json_path), 'record': rec})
    return run


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


# A.9 fix: added variant, method, feature, p_inference, checkpoint_sha256 to prevent
# averaging runs of different methods as if they were seeds of one method
IDENTITY_KEYS = ('task', 'kind', 'dataset', 'split', 'rmsf_label_sha256', 'residue_set_sha256',
                 'splits_sha256', 'items_sha256', 'classes_sha256', 'go_classes_sha256',
                 'depth', 'scope', 'cv_mode', 'cv_fold', 'block', 'probe',
                 'variant', 'method', 'feature', 'p_inference', 'checkpoint_sha256',
                 'function_protocol')

# FIXED(numbers, 2026-10-25): method-specific keys must match within one method's
# seeds but not between methods A and B in a cross-method comparison.
_METHOD_SPECIFIC_KEYS = frozenset({
    'variant', 'method', 'feature', 'p_inference', 'checkpoint_sha256',
    'function_protocol',  # label identity already checked via items/classes sha256
})
ITEM_IDENTITY_KEYS = tuple(k for k in IDENTITY_KEYS if k not in _METHOD_SPECIFIC_KEYS)


def check_same_items(runs: list, split: str, what: str = 'runs',
                     keys=None) -> None:
    """RefusedError unless every run has the same items, residues, labels and classes on
    `split` and the same item / label identity (IDENTITY_KEYS of the sidecar meta).
    `keys` overrides which meta keys to compare (default IDENTITY_KEYS)."""
    check_keys = keys if keys is not None else IDENTITY_KEYS
    r0 = runs[0]
    for r in runs[1:]:
        for k in check_keys:
            if r0['meta'].get(k) != r['meta'].get(k):
                raise RefusedError(f'{what}: {k} differs ({Path(r0["path"]).name}: '
                                   f'{r0["meta"].get(k)!r}, {Path(r["path"]).name}: '
                                   f'{r["meta"].get(k)!r})')
        a, b = r0[split], r[split]
        for k in ('ids', 'offsets', 'residue_index', 'label'):
            if (k in a) != (k in b) or (k in a and not _same(a[k], b[k])):
                raise RefusedError(f'{what}: {split} {k} differ between {Path(r0["path"]).name} '
                                   f'and {Path(r["path"]).name}')
        if r0['classes'] != r['classes']:
            raise RefusedError(f'{what}: class lists differ')


def seed_average(runs: list, split: str) -> dict:
    """Seed-averaged block of one method: same items / labels required, distinct seeds.
    Returns the block of the first run with 'pred' = mean over the runs (float64) and
    'n_seeds'."""
    if not runs:
        raise RefusedError('no runs')
    seeds = [r['meta'].get('seed') for r in runs]
    if None in seeds:
        raise RefusedError(f'sidecar without a seed ({[Path(r["path"]).name for r in runs]})')
    if len(set(seeds)) != len(seeds):
        raise RefusedError(f'repeated seeds {seeds} ({[Path(r["path"]).name for r in runs]})')
    check_same_items(runs, split, 'seeds of one method')
    b = dict(runs[0][split])
    b['pred'] = np.mean([np.asarray(r[split]['pred'], dtype=np.float64) for r in runs], axis=0)
    b['n_seeds'] = len(runs); b['seeds'] = seeds
    return b


# ---------------------------------------------------------------- clusters
def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _source(path) -> dict:
    return {'path': _rel(path), 'sha256': file_sha256(path)}


def topology_of(h: str) -> str:
    """C.A.T of a CATH code (an H code or a topology); other strings unchanged."""
    parts = str(h).split('.')
    return '.'.join(parts[:3]) if len(parts) >= 3 and all(parts[:3]) else str(h)


def mdcath_div_clusters(manifest=None) -> tuple[dict, dict]:
    """{domain: CATH topology} from the mdCATH-div manifest, and its source record.

    The source path is stored relative to REPO_ROOT (not absolute)."""
    from ensembits.paths import MDCATH_DIV_SRC
    p = Path(manifest or Path(MDCATH_DIV_SRC) / '_manifest.json').resolve()
    rep = json.loads(p.read_text())['representatives']
    return {pid: v['topology'] for pid, v in rep.items()}, _source(p)


def bioemu_clusters(splits=None) -> tuple[dict, dict]:
    """{system: CATH topology} for BioEmu-CATH (system cath<k>_<domain>; a domain without
    CATH gets its own 'no_cath_<domain>' cluster, as in the split builder)."""
    from rmsf_common import BIOEMU_SPLITS
    p = Path(splits or BIOEMU_SPLITS)
    s = json.loads(p.read_text())
    h = s['h_superfamily']
    out = {}
    for part in ('train', 'val', 'test'):
        for system in s['systems'][part]:
            dom = system.split('_', 1)[1] if '_' in system else system
            code = h.get(dom, f'no_cath_{dom}')
            out[system] = code if code.startswith('no_cath_') else topology_of(code)
    return out, _source(p)


def _misato_chain_table(chains=None):
    from ensembits.paths import MISATO_SPLITS
    p = Path(chains or str(MISATO_SPLITS).replace('.json', '.chains.json'))
    t = json.loads(p.read_text())['complexes']
    clusters = {k.upper(): list(v.get('clusters') or []) for k, v in t.items()}
    cath_h = {k.upper(): list(v.get('cath_h') or []) for k, v in t.items()}
    return clusters, cath_h, p


def _groups_to_map(groups, prefix: str) -> dict:
    return {pid: f'{prefix}{i}' for i, g in enumerate(groups) for pid in g}


def misato_clusters(kind: str, chains=None) -> tuple[dict, dict]:
    """{complex (upper case): cluster} of MISATO.

    kind: 'sequence' (sequence groups of the v2 split), 'structure' (structure groups),
    'topology' (structure groups built on CATH topologies instead of H codes; RMSF)."""
    sys.path.insert(0, str(CODE_ROOT / 'data_prep'))
    from build_misato_splits_v2 import groups_by_shared_keys, structure_groups
    clusters, cath_h, p = _misato_chain_table(chains)
    if kind == 'sequence':
        g = groups_by_shared_keys(clusters)
    elif kind == 'structure':
        g = structure_groups(cath_h, clusters)[0]
    elif kind == 'topology':
        g = structure_groups({k: sorted({topology_of(h) for h in v}) for k, v in cath_h.items()},
                             clusters)[0]
    else:
        raise ValueError(f'unknown MISATO cluster kind {kind!r}')
    return _groups_to_map(g, f'{kind[:3]}'), {**_source(p), 'rule': kind}


def function_cv_clusters(task: str, mode: str, cv_file=None) -> tuple[dict, dict]:
    """{representative: CV group} of an EC / GO CV mode (data/misato_function_cv_v1.json)."""
    from ensembits.paths import MISATO_FUNCTION_CV
    p = Path(cv_file or MISATO_FUNCTION_CV)
    cv = json.loads(p.read_text())
    groups = cv['tasks'][task.lower()]['modes'][mode]['groups']
    return _groups_to_map([[str(x).upper() for x in g] for g in groups], f'{mode[:3]}'), _source(p)


def spec_file(rec, what: str):
    """(parsed JSON, source record) of a file named in a spec ({path[, sha256]}; path relative
    to REPO_ROOT or absolute). RefusedError if missing or the sha256 differs."""
    if not isinstance(rec, dict) or not rec.get('path'):
        raise RefusedError(f'{what}: expected {{"path": ..., "sha256": ...}}, got {rec!r}')
    p = Path(rec['path'])
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        raise RefusedError(f'{what}: {rec["path"]} missing')
    sha = file_sha256(p)
    if rec.get('sha256') and rec['sha256'] != sha:
        raise RefusedError(f'{what}: {rec["path"]} changed (sha256 {sha[:12]} != '
                           f'{rec["sha256"][:12]})')
    return json.loads(p.read_text()), {'path': _rel(p), 'sha256': sha}


def spec_clusters(rec) -> tuple[dict, dict]:
    """{item: cluster} from a spec `clusters` record (CryptoBench)."""
    data, src = spec_file(rec, 'clusters')
    if not isinstance(data, dict) or not data:
        raise RefusedError('clusters: expected a non-empty JSON object {item: cluster}')
    return {str(k): str(v) for k, v in data.items()}, {**src, 'rule': 'given in the spec'}


MASK_SCHEMA = 'cryptobench-residue-mask-v1'


def apply_subset(b: dict, subset: dict | None, mask: dict | None = None) -> dict:
    """A residue block restricted to a pre-declared subset (see the module docstring):
    `min_positives` keeps items with >= k positive residues; `mask` (loaded mask JSON) drops
    the listed residues of every item (only label-0 residues may be dropped)."""
    if not subset:
        return b
    off = np.asarray(b['offsets']); lab = np.asarray(b['label'])
    n, N = len(b['ids']), len(lab)
    keep_item = np.ones(n, dtype=bool); keep_res = np.ones(N, dtype=bool)
    if subset.get('min_positives') is not None:
        k = int(subset['min_positives'])
        keep_item = np.array([lab[off[i]:off[i + 1]].sum() >= k for i in range(n)], dtype=bool)
    if subset.get('mask') is not None:
        if mask is None or mask.get('schema') != MASK_SCHEMA:
            raise RefusedError(f'residue mask missing or not {MASK_SCHEMA}')
        ent = mask['entries']
        for i, it in enumerate(b['ids'].tolist()):
            rec = ent.get(str(it))
            if rec is None:
                raise RefusedError(f'mask has no entry {it}')
            L = int(off[i + 1] - off[i])
            if int(rec['n_res']) != L:
                raise RefusedError(f'mask entry {it}: {rec["n_res"]} residues, run has {L}')
            d = np.asarray(rec['drop'], dtype=np.int64)
            if len(d) and (d.min() < 0 or d.max() >= L or lab[off[i] + d].any()):
                raise RefusedError(f'mask entry {it}: drops a positive or an out-of-range residue')
            keep_res[off[i] + d] = False
    rows = [np.flatnonzero(keep_res[off[i]:off[i + 1]]) + off[i] for i in range(n) if keep_item[i]]
    lens = [len(r) for r in rows]
    sel = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    out = dict(b)
    out['ids'] = np.asarray(b['ids'])[keep_item]
    out['offsets'] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    for k in ('pred', 'label', 'residue_index', 'row'):
        if k in b:
            out[k] = np.asarray(b[k])[sel]
    return out


def cluster_map(meta: dict) -> tuple[dict, dict]:
    """Cluster of every item of a run (from its sidecar meta) and the source record."""
    task = meta.get('task')
    if task == 'cryptobench':
        raise RefusedError('cryptobench comparisons take their clusters from the spec '
                           '(`clusters`: {path, sha256})')
    if task == 'rmsf':
        ds = meta.get('dataset')
        if ds == 'mdcath_div':
            return mdcath_div_clusters()
        if ds == 'bioemu':
            return bioemu_clusters()
        if ds != 'misato':
            raise RefusedError(f'no cluster rule for RMSF dataset {ds!r}')
    if task in ('binding', 'affinity', 'rmsf'):          # rmsf here: MISATO
        sp = meta.get('split')
        if sp not in ('sequence', 'structure', 'random'):
            raise RefusedError(f'no cluster rule for MISATO split {sp!r}')
        return misato_clusters('structure' if sp == 'structure' else 'sequence')
    if task in ('EC', 'GO'):
        if meta.get('cv_mode') not in ('sequence', 'structure'):
            raise RefusedError('EC / GO tests need a CV run (cv_mode, fold)')
        return function_cv_clusters(task, meta['cv_mode'])
    raise RefusedError(f'no cluster rule for task {task!r}')


def item_clusters(ids, cmap: dict) -> np.ndarray:
    """Cluster label of every item; RefusedError if an item has none."""
    miss = [i for i in ids if i not in cmap and str(i).upper() not in cmap]
    if miss:
        raise RefusedError(f'{len(miss)} items without a cluster (e.g. {miss[:5]})')
    return np.asarray([str(cmap.get(i, cmap.get(str(i).upper()))) for i in ids])


# ---------------------------------------------------------------- metrics
def _segments(b):
    off = np.asarray(b['offsets'])
    return [slice(int(off[i]), int(off[i + 1])) for i in range(len(off) - 1)]


def rmsf_per_protein_values(b: dict, min_residues: int = PER_PROTEIN_MIN_RESIDUES) -> np.ndarray:
    """Per-protein Spearman (NaN = not eligible), rule of rmsf_common.per_protein_spearman."""
    pred = np.asarray(b['pred'], dtype=np.float64); y = np.asarray(b['label'], dtype=np.float64)
    if not (np.isfinite(pred).all() and np.isfinite(y).all()):
        raise RefusedError('non-finite RMSF predictions or labels')
    out = np.full(len(b['ids']), np.nan)
    for i, sl in enumerate(_segments(b)):
        yy, pp = y[sl], pred[sl]
        if len(yy) < min_residues or np.all(yy == yy[0]):
            continue
        if np.all(pp == pp[0]):
            out[i] = 0.0; continue
        r = spearmanr(pp, yy).correlation
        out[i] = float(r) if r is not None and np.isfinite(r) else 0.0
    return out


def binding_per_protein_values(b: dict) -> np.ndarray:
    """Per-protein AUROC (NaN when the protein lacks one class)."""
    from sklearn.metrics import roc_auc_score
    out = np.full(len(b['ids']), np.nan)
    for i, sl in enumerate(_segments(b)):
        y = np.asarray(b['label'][sl])
        if len(np.unique(y)) == 2:
            out[i] = roc_auc_score(y, np.asarray(b['pred'][sl], dtype=np.float64))
    return out


def protein_ap_values(b: dict) -> np.ndarray:
    """Per-protein AP of the class scores against the protein's labels (NaN without a
    positive)."""
    from sklearn.metrics import average_precision_score
    S = np.asarray(b['pred'], dtype=np.float64); Y = np.asarray(b['label']) > 0
    out = np.full(len(S), np.nan)
    for i in range(len(S)):
        if Y[i].any():
            out[i] = average_precision_score(Y[i], S[i])
    return out


def macro_mAP(pred, label) -> float:
    """Class macro mean AP: mean over classes of the per-class AP (Q1 metric)."""
    from sklearn.metrics import average_precision_score
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label) > 0
    aps = []
    for c in range(label.shape[1]):
        if label[:, c].any() and not label[:, c].all():
            aps.append(average_precision_score(label[:, c], pred[:, c]))
    return float(np.mean(aps)) if aps else float('nan')


def _expand(b: dict, idx) -> np.ndarray:
    """Residue rows of the items `idx` (with repeats) of a residue block."""
    off = np.asarray(b['offsets'])
    return np.concatenate([np.arange(off[i], off[i + 1]) for i in idx]) if len(idx) else \
        np.zeros(0, dtype=np.int64)


def make_delta_stat(task: str, metric: str, a: dict, b: dict):
    """(stat(idx) -> Delta of the items idx, point Delta, n eligible items, per-item values or
    None). a / b: seed-averaged blocks of methods A and B on the same items."""
    if metric not in METRICS.get(task, ()):
        raise ValueError(f'metric {metric!r} not defined for {task!r} ({METRICS.get(task)})')
    n = len(a['ids'])
    linear = {'spearman_per_protein': rmsf_per_protein_values,
              'auroc_per_protein': binding_per_protein_values,
              'protein_ap': protein_ap_values}
    if metric in linear:
        va, vb = linear[metric](a), linear[metric](b)
        if not _same(np.isnan(va), np.isnan(vb)):
            raise RefusedError('eligible items differ between the methods')
        d = va - vb
        ok = ~np.isnan(d)

        def stat(idx):
            m = ok[idx]
            return float(np.mean(d[idx][m])) if m.any() else float('nan')
        return stat, stat(np.arange(n)), int(ok.sum()), {'a': va, 'b': vb}
    if metric == 'spearman_pooled':
        ya = np.asarray(a['label'], dtype=np.float64)

        def stat(idx):
            r = _expand(a, idx)
            return float(spearmanr(a['pred'][r], ya[r]).correlation
                         - spearmanr(b['pred'][r], ya[r]).correlation)
        return stat, stat(np.arange(n)), n, None
    if metric in ('auroc_pooled', 'ap_pooled'):
        from sklearn.metrics import roc_auc_score, average_precision_score
        f = roc_auc_score if metric == 'auroc_pooled' else average_precision_score
        y = np.asarray(a['label']); off = np.asarray(a['offsets']); L = np.diff(off)

        def stat(idx):
            w = np.repeat(np.bincount(idx, minlength=n), L).astype(np.float64)
            k = w > 0
            if len(np.unique(y[k])) < 2:
                return float('nan')
            return float(f(y[k], a['pred'][k], sample_weight=w[k])
                         - f(y[k], b['pred'][k], sample_weight=w[k]))
        return stat, stat(np.arange(n)), n, None
    if metric in ('spearman', 'r2', 'mse'):
        y = np.asarray(a['label'], dtype=np.float64)
        pa = np.asarray(a['pred'], dtype=np.float64); pb = np.asarray(b['pred'], dtype=np.float64)

        def one(p, yy):
            if metric == 'spearman':
                return spearmanr(p, yy).correlation
            if metric == 'mse':
                return float(np.mean((p - yy) ** 2))
            ss = float(np.sum((yy - yy.mean()) ** 2))
            return 1.0 - float(np.sum((yy - p) ** 2)) / ss if ss > 0 else float('nan')

        def stat(idx):
            return float(one(pa[idx], y[idx]) - one(pb[idx], y[idx]))
        return stat, stat(np.arange(n)), n, None
    if metric == 'fmax':
        from multilabel_metrics import fmax
        Y = np.asarray(a['label'])

        def stat(idx):
            return float(fmax(a['pred'][idx], Y[idx])[0] - fmax(b['pred'][idx], Y[idx])[0])
        return stat, stat(np.arange(n)), n, None
    if metric == 'macro_mAP':
        Y = np.asarray(a['label'])

        def stat(idx):
            return float(macro_mAP(a['pred'][idx], Y[idx])
                         - macro_mAP(b['pred'][idx], Y[idx]))
        return stat, stat(np.arange(n)), n, None
    raise ValueError(metric)


# ---------------------------------------------------------------- bootstrap
def cluster_members(clusters) -> list:
    """Item indices of every cluster (clusters in sorted label order)."""
    clusters = np.asarray(clusters)
    _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.flatnonzero(np.r_[True, np.diff(inv[order]) != 0, True])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def draw_clusters(members: list, rng) -> np.ndarray:
    """One cluster-bootstrap replicate: item indices (with repeats)."""
    pick = rng.integers(0, len(members), len(members))
    return np.concatenate([members[k] for k in pick])


def bootstrap_summary(point: float, reps: np.ndarray, ci: float = CI_LEVEL) -> dict:
    """Point, percentile CI, SE and centred two-sided p of a bootstrap distribution.

    A.9 fix: refuse when B_finite < B (non-finite replicates were silently dropped)."""
    reps = np.asarray(reps, dtype=np.float64)
    B_total = len(reps)
    reps = reps[np.isfinite(reps)]
    B = len(reps)
    if B < B_total:
        raise RefusedError(f'bootstrap: {B_total - B} non-finite replicates out of {B_total}')
    if B == 0 or not np.isfinite(point):
        return {'delta': point, 'ci_low': float('nan'), 'ci_high': float('nan'),
                'se': float('nan'), 'p': float('nan'), 'B_finite': B}
    lo, hi = np.quantile(reps, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    b_count = int(np.sum(np.abs(reps - point) >= abs(point)))
    p = (1 + b_count) / (B + 1)
    p_note = (f'p < {1/(B+1):.2e} (0/{B} replicates reached the observed value)'
              if b_count == 0 else None)
    return {'delta': float(point), 'ci_low': float(lo), 'ci_high': float(hi),
            'se': float(np.std(reps, ddof=1)) if B > 1 else float('nan'), 'p': float(p),
            'p_note': p_note, 'B_finite': B}


def paired_cluster_bootstrap(stat, clusters, B: int = DEFAULT_B, seed: int = DEFAULT_SEED,
                             ci: float = CI_LEVEL) -> dict:
    """Bootstrap of a paired statistic over clusters. stat(idx) -> Delta for the item indices
    idx (with repeats); the point estimate uses every item once."""
    members = cluster_members(clusters)
    rng = np.random.default_rng(seed)
    n = int(sum(len(m) for m in members))
    point = stat(np.arange(n))
    reps = np.array([stat(draw_clusters(members, rng)) for _ in range(B)])
    return {**bootstrap_summary(point, reps, ci), 'B': B, 'seed': seed,
            'n_items': n, 'n_clusters': len(members)}


def fold_paired_bootstrap(stats: list, clusters: list, B: int = DEFAULT_B,
                          seed: int = DEFAULT_SEED, ci: float = CI_LEVEL) -> dict:
    """Per fold f: stats[f](idx) with its own clusters; returns the per-fold bootstraps and
    the fold-paired mean (mean over folds of Delta_f; each replicate draws the clusters of
    every fold independently, from one generator, fold order fixed)."""
    rng = np.random.default_rng(seed)
    members = [cluster_members(c) for c in clusters]
    points = [s(np.arange(int(sum(len(m) for m in mm)))) for s, mm in zip(stats, members)]
    reps = np.array([[s(draw_clusters(mm, rng)) for s, mm in zip(stats, members)]
                     for _ in range(B)])                       # (B, n_folds)
    per_fold = [{**bootstrap_summary(points[f], reps[:, f], ci),
                 'n_items': int(sum(len(m) for m in members[f])),
                 'n_clusters': len(members[f])} for f in range(len(stats))]
    mean_point = float(np.mean(points))
    return {'per_fold': per_fold,
            'fold_mean': {**bootstrap_summary(mean_point, reps.mean(axis=1), ci), 'B': B,
                          'seed': seed, 'n_folds': len(stats)}}


def holm(pvals) -> list:
    """Holm step-down adjusted p-values (input order).

    A.9 fix: a NaN p raises RefusedError (NaN was silently dropped, shrinking m).
    """
    p = np.asarray(pvals, dtype=float)
    nan_idx = np.flatnonzero(~np.isfinite(p))
    if len(nan_idx):
        raise RefusedError(f'NaN p-value at index {nan_idx.tolist()} — Holm family '
                           f'must not shrink silently (count it as p = 1 or fix the source)')
    m = len(p)
    out = np.full(m, np.nan)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind='stable')):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


# ---------------------------------------------------------------- one comparison
def _expand_paths(spec_list) -> list:
    out = []
    for s in spec_list:
        p = Path(s)
        p = p if p.is_absolute() else ROOT / p
        hits = sorted(glob.glob(str(p))) if any(c in str(s) for c in '*?[') else [str(p)]
        if not hits:
            raise RefusedError(f'no result file matches {s!r}')
        out.extend(hits)
    return out


def load_method(paths, block=None) -> list:
    return [load_run(p, block) for p in _expand_paths(paths)]


def compare(task: str, metric: str, runs_a: list, runs_b: list, split: str = 'test',
            B: int = DEFAULT_B, seed: int = DEFAULT_SEED, clusters=None, subset=None,
            mask=None, cluster_source=None) -> dict:
    """Paired cluster bootstrap of one (non-CV) comparison. runs_*: load_run results (seeds).
    clusters: {item: cluster} (default: cluster_map of the runs' meta); subset / mask: a
    pre-declared sensitivity subset (`apply_subset`), applied to both methods."""
    a = seed_average(runs_a, split); b = seed_average(runs_b, split)
    # FIXED(numbers, 2026-10-25): cross-method check uses ITEM_IDENTITY_KEYS
    # (excludes variant/method/checkpoint that legitimately differ between methods)
    check_same_items([runs_a[0], runs_b[0]], split, 'methods A and B',
                     keys=ITEM_IDENTITY_KEYS)
    if subset:
        a = apply_subset(a, subset, mask); b = apply_subset(b, subset, mask)
    if clusters is None:
        clusters, src = cluster_map(runs_a[0]['meta'])
    else:
        src = cluster_source or {'given': True}
    cl = item_clusters(a['ids'].tolist(), clusters)
    stat, point, n_elig, _ = make_delta_stat(task, metric, a, b)
    res = paired_cluster_bootstrap(stat, cl, B=B, seed=seed)
    return {**res, 'n_eligible': n_elig, 'cluster_source': src, 'subset': subset or None,
            'seeds_a': a['seeds'], 'seeds_b': b['seeds']}


def compare_cv(task: str, metric: str, folds_a: dict, folds_b: dict, split: str = 'test',
               B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> dict:
    """EC / GO: {fold: [runs]} of A and B; per-fold Delta + fold-paired mean."""
    if sorted(folds_a) != sorted(folds_b):
        raise RefusedError(f'folds differ: {sorted(folds_a)} vs {sorted(folds_b)}')
    stats, cls, info = [], [], []
    for f in sorted(folds_a, key=int):
        ra, rb = folds_a[f], folds_b[f]
        a = seed_average(ra, split); b = seed_average(rb, split)
        # FIXED(numbers, 2026-10-25): cross-method check uses ITEM_IDENTITY_KEYS
        check_same_items([ra[0], rb[0]], split, f'fold {f}: methods A and B',
                         keys=ITEM_IDENTITY_KEYS)
        if int(ra[0]['meta'].get('cv_fold', -1)) != int(f):
            raise RefusedError(f'fold {f}: runs are of fold {ra[0]["meta"].get("cv_fold")}')
        cmap, src = cluster_map(ra[0]['meta'])
        cls.append(item_clusters(a['ids'].tolist(), cmap))
        st, _, n_elig, _ = make_delta_stat(task, metric, a, b)
        stats.append(st)
        info.append({'fold': int(f), 'n_eligible': n_elig, 'seeds_a': a['seeds'],
                     'seeds_b': b['seeds']})
    res = fold_paired_bootstrap(stats, cls, B=B, seed=seed)
    for d, i in zip(res['per_fold'], info):
        d.update(i)
    res['cluster_source'] = src
    return res


# ---------------------------------------------------------------- family
def example_family() -> dict:
    """Shape of a family spec (paths relative to REPO_ROOT; globs allowed, one file per seed)."""
    return {
        'family': 'A4-main',
        'note': 'pre-specified before any test statistic is computed',
        'B': DEFAULT_B, 'seed': DEFAULT_SEED, 'split': 'test',
        'comparisons': [
            {'id': 'rmsf_mdcath_P1_vs_ESM3struct', 'task': 'rmsf',
             'metric': 'spearman_per_protein',
             'a': ['output/probe_rmsf_div/<variant>_P1inf*__test.json'],
             'b': ['output/probe_rmsf_div/baseline_esm3struct*__test.json']},
            {'id': 'binding_structure_vs_ProToken', 'task': 'binding',
             'block': 'structure_binding', 'metric': 'auroc_per_protein',
             'a': ['output/binding_misato/<variant>_lig_stab*.json'],
             'b': ['output/binding_misato/baseline_protoken_lig_stab*.json']},
            {'id': 'go_sequence_vs_AminoAseed', 'task': 'GO', 'metric': 'protein_ap',
             'folds': {'0': {'a': ['output/go_misato/go_<tag>_cv_sequence_f0_*top50.json'],
                             'b': ['output/go_conv1d_misato/go_aminoaseed_top50_cv_sequence_f0*.json']}}},
        ]}


def check_family(spec: dict) -> list:
    """Problems of a family spec (empty = well formed)."""
    out = []
    comps = spec.get('comparisons') or []
    if not comps:
        out.append('no comparisons')
    ids = [c.get('id') for c in comps]
    if len(set(ids)) != len(ids) or None in ids:
        out.append('comparison ids must be present and unique')
    for c in comps:
        t, m = c.get('task'), c.get('metric')
        if m not in METRICS.get(t, ()):
            out.append(f'{c.get("id")}: metric {m!r} not defined for task {t!r}')
        if t in ('EC', 'GO'):
            if not c.get('folds'):
                out.append(f'{c.get("id")}: EC / GO comparisons need per-fold files (folds)')
        elif not (c.get('a') and c.get('b')):
            out.append(f'{c.get("id")}: files of methods a and b needed')
        if t in ('binding', 'affinity') and not c.get('block'):
            out.append(f'{c.get("id")}: binding / affinity comparisons need the block '
                       f'(<split>_<task>)')
        if t == 'cryptobench' and not (c.get('clusters') or spec.get('clusters')):
            out.append(f'{c.get("id")}: cryptobench comparisons need `clusters` in the spec')
        sub = c.get('subset') or {}
        if set(sub) - {'min_positives', 'mask'}:
            out.append(f'{c.get("id")}: unknown subset keys {sorted(set(sub) - {"min_positives", "mask"})}')
        if sub and t not in _RESIDUE_TASKS:
            out.append(f'{c.get("id")}: subsets apply to residue tasks only')
    return out


def _comparison_inputs(spec: dict, c: dict) -> dict:
    """Spec-given clusters / mask of one comparison (loaded and sha256-checked)."""
    kw = {}
    crec = c.get('clusters') or spec.get('clusters')
    if crec:
        kw['clusters'], kw['cluster_source'] = spec_clusters(crec)
    sub = c.get('subset')
    if sub:
        kw['subset'] = sub
        if sub.get('mask') is not None:
            kw['mask'], msrc = spec_file(sub['mask'], f'{c["id"]} mask')
            kw['subset'] = {**sub, 'mask': msrc}
    return kw


def load_family(spec: dict) -> tuple[dict, list]:
    """Load (and verify) every run of a family: ({id: loaded}, refusals)."""
    loaded, refused = {}, []
    for c in spec['comparisons']:
        try:
            if c['task'] in ('EC', 'GO'):
                loaded[c['id']] = {f: (load_method(v['a']), load_method(v['b']))
                                   for f, v in c['folds'].items()}
            else:
                loaded[c['id']] = (load_method(c['a'], c.get('block')),
                                   load_method(c['b'], c.get('block')))
        except (RefusedError, PredsError, FileNotFoundError) as e:
            refused.append(f'{c["id"]}: {e}')
    return loaded, refused


# FIXED(numbers2, 2026-10-26): parallelise comparisons via --workers N.
# Each comparison reseeds its own RNG (paired_cluster_bootstrap creates
# np.random.default_rng(seed)), so results are bit-identical to the serial
# path regardless of scheduling order.  Holm and output order applied after
# gathering, exactly as in serial.

def _init_pool():
    """Set each pool worker to one thread (prevents thread x worker explosion)."""
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_MAX_THREADS'):
        os.environ[v] = '1'


def _compute_one(args):
    """Process-pool worker: load and compute one comparison independently.

    Each worker re-loads its own data from disk (already validated by load_family
    in the parent) and computes the bootstrap.  The result dict is identical to what
    the serial loop produces for the same comparison."""
    c, spec_ctx, split, B, seed = args
    if c['task'] in ('EC', 'GO'):
        got = {f: (load_method(v['a']), load_method(v['b']))
               for f, v in c['folds'].items()}
        r = compare_cv(c['task'], c['metric'], {f: v[0] for f, v in got.items()},
                       {f: v[1] for f, v in got.items()}, split, B, seed)
        p = r['fold_mean']['p']
    else:
        ra = load_method(c['a'], c.get('block'))
        rb = load_method(c['b'], c.get('block'))
        got = (ra, rb)
        r = compare(c['task'], c['metric'], ra, rb, split, B, seed,
                    **_comparison_inputs(spec_ctx, c))
        p = r['p']
    return {'id': c['id'], 'task': c['task'], 'metric': c['metric'],
            'files_a': sorted({x['path'] for x in _flat(got, 0, c)}),
            'files_b': sorted({x['path'] for x in _flat(got, 1, c)}),
            'sidecars_a': _sidecar_hashes(_flat(got, 0, c)),
            'sidecars_b': _sidecar_hashes(_flat(got, 1, c)),
            'p_raw': p, **r}


def run_family(spec: dict, split: str = 'test', check_only: bool = False,
               workers: int = 1) -> dict:
    """Run every comparison of a pre-specified family and Holm-adjust over all of them.

    spec: {'family', 'B', 'seed', 'comparisons': [{'id', 'task', 'metric', 'a': [files],
    'b': [files], 'block' (binding / affinity)} | {'id', 'task' (EC / GO), 'metric',
    'folds': {fold: {'a': [files], 'b': [files]}}}]}. Raises RefusedError if any result is
    refused (no sidecar, changed sidecar, mismatched items) or the spec is malformed.
    check_only: stop after loading / verifying the runs and the cluster maps.
    workers: number of process-pool workers for the comparisons (1 = serial)."""
    probs = check_family(spec)
    if probs:
        raise RefusedError('family spec: ' + '; '.join(probs))
    if spec.get('split') and spec['split'] != split:
        raise RefusedError(f'family split {spec["split"]!r} differs from --split {split!r}')
    loaded, refused = load_family(spec)
    if refused:
        raise RefusedError('refused results:\n  ' + '\n  '.join(refused))
    if spec.get('seeds') is not None:
        want = sorted(int(x) for x in spec['seeds'])
        for c in spec['comparisons']:
            got = loaded[c['id']]
            pairs = list(got.values()) if c['task'] in ('EC', 'GO') else [got]
            for ra, rb in pairs:
                for runs in (ra, rb):
                    seeds = sorted(r['meta'].get('seed') for r in runs)
                    if seeds != want:
                        raise RefusedError(f'{c["id"]}: seeds {seeds} != the declared {want} '
                                           f'({Path(runs[0]["path"]).name})')
    B = int(spec.get('B', DEFAULT_B)); seed = int(spec.get('seed', DEFAULT_SEED))
    if check_only:
        for c in spec['comparisons']:
            got = loaded[c['id']]
            pairs = list(got.values()) if c['task'] in ('EC', 'GO') else [got]
            for ra, rb in pairs:
                for runs in (ra, rb):
                    check_same_items(runs, split, f'{c["id"]} seeds')
                check_same_items([ra[0], rb[0]], split, f'{c["id"]} methods',
                                 keys=ITEM_IDENTITY_KEYS)
                a = seed_average(ra, split)
                kw = _comparison_inputs(spec, c)
                if kw.get('subset'):
                    a = apply_subset(a, kw['subset'], kw.get('mask'))
                item_clusters(a['ids'].tolist(), kw['clusters'] if 'clusters' in kw
                              else cluster_map(ra[0]['meta'])[0])
        return {'checked': [c['id'] for c in spec['comparisons']], 'split': split}
    if workers > 1:
        spec_ctx = {k: spec[k] for k in ('clusters',) if k in spec}
        tasks = [(c, spec_ctx, split, B, seed) for c in spec['comparisons']]
        w = min(workers, len(tasks))
        print(f'[parallel] {len(tasks)} comparisons, {w} workers', flush=True)
        with ProcessPoolExecutor(max_workers=w, initializer=_init_pool) as pool:
            results = list(pool.map(_compute_one, tasks))
    else:
        results = []
        for c in spec['comparisons']:
            got = loaded[c['id']]
            if c['task'] in ('EC', 'GO'):
                r = compare_cv(c['task'], c['metric'], {f: v[0] for f, v in got.items()},
                               {f: v[1] for f, v in got.items()}, split, B, seed)
                p = r['fold_mean']['p']
            else:
                r = compare(c['task'], c['metric'], got[0], got[1], split, B, seed,
                            **_comparison_inputs(spec, c))
                p = r['p']
            results.append({'id': c['id'], 'task': c['task'], 'metric': c['metric'],
                            'files_a': sorted({x['path'] for x in _flat(got, 0, c)}),
                            'files_b': sorted({x['path'] for x in _flat(got, 1, c)}),
                            'sidecars_a': _sidecar_hashes(_flat(got, 0, c)),
                            'sidecars_b': _sidecar_hashes(_flat(got, 1, c)),
                            'p_raw': p, **r})
    if any(c['task'] == 'cryptobench' for c in spec['comparisons']):
        bad = [r['id'] for r in results if not np.isfinite(r['p_raw'])]
        if bad:
            raise RefusedError(f'non-finite p in a pre-registered cryptobench family: {bad}')
    for r, ph in zip(results, holm([r['p_raw'] for r in results])):
        r['p_holm'] = ph
    return {'stats_version': STATS_VERSION, 'family': spec.get('family'), 'split': split,
            'B': B, 'seed': seed, 'holm_family_size': len(results),
            'p_value_rule': 'centred bootstrap, p = (1 + #{|D*_b - D| >= |D|}) / (B + 1)',
            'ci_rule': f'percentile {CI_LEVEL:.0%}', 'comparisons': results}


def run_bundle(bundle: dict, split: str = 'test', check_only: bool = False,
               workers: int = 1) -> dict:
    """Run every family of a bundle ({'bundle', 'families': [spec, ...]}); each family is
    Holm-adjusted on its own. Shared keys of the bundle (B, seed, clusters) are defaults of
    its families. Any refusal refuses the whole bundle."""
    fams = bundle.get('families') or []
    if not fams:
        raise RefusedError('bundle without families')
    names = [f.get('family') for f in fams]
    if None in names or len(set(names)) != len(names):
        raise RefusedError('family names must be present and unique in a bundle')
    shared = {k: bundle[k] for k in ('B', 'seed', 'clusters', 'seeds', 'split') if k in bundle}
    out = [run_family({**shared, **f}, split, check_only, workers) for f in fams]
    if check_only:
        return {'checked': {n: o['checked'] for n, o in zip(names, out)}, 'split': split}
    return {'stats_version': STATS_VERSION, 'bundle': bundle.get('bundle'), 'split': split,
            'families': out}


def _sidecar_hashes(runs) -> list:
    """[{result, sidecar sha256}] of loaded runs (sorted by result path)."""
    return sorted(({'result': _rel(r['path']), 'sha256': (r.get('record') or {}).get('sha256')}
                   for r in runs), key=lambda d: d['result'])


def _flat(got, k, c):
    if c['task'] in ('EC', 'GO'):
        return [x for v in got.values() for x in v[k]]
    return got[k]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--family', required=True, help='pre-specified family spec (JSON)')
    ap.add_argument('--out', required=False, help='output JSON (required unless --check-only)')
    # A.3 fix: no default (was 'test') — require the user to specify the split explicitly
    ap.add_argument('--split', required=True, choices=('val', 'test'))
    ap.add_argument('--check-only', action='store_true',
                    help='verify every sidecar, item set and cluster map; compute nothing')
    ap.add_argument('--workers', type=int, default=1,
                    help='process-pool workers for the comparisons (default 1 = serial)')
    args = ap.parse_args(argv)
    spec_path = Path(args.family)
    spec = json.loads(spec_path.read_text())
    try:
        if 'families' in spec:
            res = run_bundle(spec, args.split, check_only=args.check_only,
                             workers=args.workers)
        else:
            res = run_family(spec, args.split, check_only=args.check_only,
                             workers=args.workers)
    except RefusedError as e:
        print(f'[refused] {e}', file=sys.stderr); return 1
    res['family_spec'] = {'path': _rel(spec_path), 'sha256': file_sha256(spec_path)}
    if args.check_only:
        n = (sum(len(v) for v in res['checked'].values()) if isinstance(res['checked'], dict)
             else len(res['checked']))
        print(f'[check] {n} comparisons: sidecars, items and clusters OK')
        return 0
    if not args.out:
        print('[error] --out is required', file=sys.stderr); return 2
    from function_common import atomic_write_json
    atomic_write_json(args.out, _jsonable(res))
    n = (sum(len(f['comparisons']) for f in res['families']) if 'families' in res
         else len(res['comparisons']))
    print(f'[done] {args.out} ({n} comparisons; values not printed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
