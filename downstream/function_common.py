"""Shared rules of the MISATO function probes (EC, GO, binding site, affinity): items, inputs, metrics, checks.

What it does
------------
One place for what every EC / GO / binding-site / affinity probe (Ensembits and baselines)
must do the same way, so that every method of a table cell is scored on the same complexes
with the same classes (function_reaudit pre-run fixes, 2026-10-19):

  * items: `misato_function_pids()` (every complex of the split file),
    `require_features()` (a method without a feature for an item of the task is an error,
    not a silent drop), `check_length()` (feature length must equal the label / residue
    count; the old code cut both to the shorter one), `items_record()` (counts and sha256 of
    the sorted train / val / test pid lists actually used), `classes_record()`;
  * inputs: `misato_aa_onehot()` (the `aa` baseline: one-hot of the `seq` of
    data/cached_descriptors/misato_real_bb/<pid>.npz, the residue list of every token cache,
    His / CYX already mapped by the MISATO parser, as in the RMSF probe's `build_aa`),
    `check_coords_ca8()`, `token_cache_record()`, `load_affinity_labels()` /
    `load_ligand_maccs()` (refuse files without their builder's sidecar; records with sha256);
  * metrics on any split (the probes store them for test AND validation):
    `binding_metrics()` (pooled AUROC / AP and the per-protein AUROC), `affinity_metrics()`
    (R², Spearman, MSE);
  * result checks for runners and compilers: `function_result_problems()` (protocol tag,
    verified inputs, affinity label / MACCS sha256), `cell_signature()` (class-list and item
    hashes of one table cell, compared across methods by the compiler).

  * input standardisation (authors' decision 2026-10-19, default for every method of the
    1D-conv function probes; the RMSF MLP probes are not affected): `InputStandardisation`,
    `standardize_lookup_table()`, `train_residue_stats()`; every result JSON records
    `input_standardisation` = {rule, version, stats_sha256, ...}.

FUNCTION_PROTOCOL is written into every result JSON; results without the current tag are
refused by the compilers and never reused by the runners. Older tags: results written before
the pre-run fixes (aa on the old pairs pickle, classes from the complexes with features,
padded Ensembits tokens = codebook[0], no item hashes) and, since 2026-10-19
(misato-function-2026-10-19-instd), every result written before the input standardisation.

Input standardisation (INPUT_STD_VERSION)
-----------------------------------------
  * lookup features (a frozen table indexed by a per-residue id: Ensembits level-1 codebook,
    AminoAseed projected codebook, ESM3struct and ProToken codebooks, the mini3di centroids of
    3Di / Vote_3Di, the aa one-hot = rows of the 21 x 21 identity): each table dimension is
    centred and scaled by its mean / std over the table ROWS (every row once, ddof 0; label
    free, the same for every split); a dimension with std < 1e-8 is only centred;
  * continuous per-residue features (ProtProfileMD profiles, random features, PCA
    eigenvalues, pooled features, any concatenated non-lookup block): per-dimension mean / std
    over the residues of the TRAINING items of that split / CV rotation / EC depth only
    (two passes in float64, ddof 0; std < 1e-8: centred only), applied to train, val and test;
  * the MACCS ligand fingerprint of the affinity head: unchanged (binary 0/1) for every
    method (see LIGAND_RULE);
  * padded positions stay zero vectors (the mean of the standardised feature).
`--no-input-standardisation` (every probe) reproduces the old inputs; such results are
recorded with "enabled": false and refused by the compilers and runners.

Paper
-----
`appendix.tex` \\label{app:downstreams} (EC, GO, binding site, binding affinity probes) and
\\label{app:baselines} (the `aa` baseline). Infrastructure for the comparisons; the rules
(same items and classes for every method) are not stated in the paper yet.

Origin
------
New (2026-10-19, camera_ready/function_reaudit.md pre-run fixes 1-5). Input standardisation
added 2026-10-19 (authors' decision after the EC/GO conditioning check; FIXED markers).

Used by the reported model: yes (every EC / GO / binding / affinity row).

Inputs / outputs
----------------
reads (relative to REPO_ROOT) data/cached_descriptors/misato_real_bb/<pid>.npz (aa),
data/cached_descriptors/coords_ca8_misato.pkl (+ sidecar), data/annotations/
misato_affinity.json and data/cached_descriptors/misato_ligand_maccs.npz (+ sidecars),
the MISATO split file; writes nothing.

How to run
----------
Library module (imported by the probes, runners and compilers); tests in
tests/test_function_prerun_fixes.py and tests/test_function_input_std.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402

# Tag written into every EC / GO / binding / affinity result JSON (compilers refuse others).
# FIXED(function input standardisation, 2026-10-19): bumped from 'misato-function-2026-10-19'
#   (inputs fed raw) so that every EC / GO / binding / affinity result written before the
#   input standardisation is refused by the compilers and runners and must be rerun.
FUNCTION_PROTOCOL = 'misato-function-2026-10-23-labels'
PROTOCOL_NOTE = ('same items and classes for every method (classes from all labelled '
                 'training complexes; a missing feature is an error); aa = misato_real_bb seq; '
                 'Ensembits padding = zero vectors; sorted item order; metrics stored on '
                 'validation (selected epoch) and test; head inputs standardised per '
                 'dimension (lookup tables over their rows, continuous features over the '
                 'training residues; MACCS unchanged); Q7 label fix: EC v3 (present-chain '
                 'restriction + 47 SIFTS-missing), CV v2, affinity v2 (32 rescued, 8 dropped), '
                 'MACCS v2 (aligned), binding v3 (134 off-pocket relabelled/dropped)')
AA_VOCAB = 'ACDEFGHIKLMNPQRSTVWYX'          # 21-D (20 + X), as before (require_fix #22)
AA_IDX = {a: i for i, a in enumerate(AA_VOCAB)}
MIN_RESIDUES = 10                           # binding / affinity: shorter complexes skipped
AFFINITY_SCHEMA = 'misato-affinity-v2'
MACCS_SCHEMA = 'misato-ligand-maccs-v2'
EC_METRICS = ('top1_hit', 'macro_ap', 'micro_ap', 'macro_f1@0.5', 'micro_f1@0.5', 'fmax')
BINDING_METRICS = ('auroc', 'ap', 'auroc_per_protein_mean')
AFFINITY_METRICS = ('r2', 'spearman', 'mse')
SPLITS = ('train', 'val', 'test')


class FunctionInputError(RuntimeError):
    """An item of a function task has no feature, a wrong length, or an unverified input."""


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _file_sha256(path) -> str:
    from ensembits.featurize import file_sha256
    return file_sha256(path)


# ---------------------------------------------------------------- items and classes
def pid_list_sha256(pids) -> str:
    """sha256 over the sorted, upper-cased pids joined by newlines (as go_labels)."""
    return hashlib.sha256('\n'.join(sorted(str(p).upper() for p in pids)).encode()).hexdigest()


def misato_function_pids(path=None) -> list:
    """Every complex of the MISATO split file (union of the train / val / test lists of the
    sequence, structure and random splits), sorted, upper case."""
    from ensembits.data_versions import load_misato_splits
    s = load_misato_splits(path)
    out = set()
    for name in ('sequence', 'structure', 'random'):
        for part in SPLITS:
            out.update(str(p).upper() for p in (s.get(name) or {}).get(part, []))
    return sorted(out)


def require_features(pids, feats, method: str, what: str) -> None:
    """FunctionInputError unless every pid of the task has a feature of this method."""
    missing = sorted(p for p in pids if p not in feats)
    if missing:
        raise FunctionInputError(f'{method}: no feature for {len(missing)} {what} complexes '
                                 f'(e.g. {missing[:5]}); every method must score the same items')


def check_length(pid: str, n_feat: int, n_res: int, method: str, what: str) -> None:
    """FunctionInputError if the feature of `pid` does not have one row per labelled residue.

    FIXED(function_reaudit, 2026-10-19): old = features and labels were cut to the shorter
    of the two lengths from the N-terminus, silently."""
    if int(n_feat) != int(n_res):
        raise FunctionInputError(f'{method} {what} {pid}: feature length {n_feat} != '
                                 f'{n_res} labelled residues')


def items_record(bucket: dict) -> dict:
    """Counts and sha256 of the sorted pid lists actually scored (bucket = {split: [items]})."""
    per = {s: pid_list_sha256([it['pid'] for it in bucket.get(s, [])]) for s in SPLITS}
    return {'n_items': {s: len(bucket.get(s, [])) for s in SPLITS},
            'items_pids_sha256': per,
            'items_sha256': hashlib.sha256('|'.join(f'{s}:{per[s]}' for s in SPLITS)
                                           .encode()).hexdigest()}


def classes_record(classes) -> dict:
    """{'classes': list, 'classes_sha256': sha256 of the class list in its order}."""
    classes = [str(c) for c in classes]
    return {'classes': classes,
            'classes_sha256': hashlib.sha256('\n'.join(classes).encode()).hexdigest()}


def dropped_counts(split_map: dict, kept: dict, reasons: dict) -> dict:
    """{reason: {train, val, test}} counts of complexes dropped for each reason."""
    out = {}
    for r, pids in reasons.items():
        c = {s: 0 for s in SPLITS}
        for p in pids:
            s = split_map.get(p)
            if s in c:
                c[s] += 1
        out[r] = c
    return out


# FIXED(function_reaudit pre-run fixes 2 / 5 / 6, 2026-10-19): one EC depth, shared with the
#   Ensembits probe (tokenize_misato_and_eval.py). Old (both scripts): classes = EC classes
#   of the TRAINING COMPLEXES WITH FEATURES of this method (could differ between methods),
#   complexes without a feature dropped silently, items in set / cache order. New: classes
#   from every labelled training complex of the split (same for every method), a labelled
#   complex without a feature is an error, items in sorted pid order, the class list, item
#   hashes and validation metrics are stored. (Called by both EC scripts.)
def run_ec_depth(depth, labels, split_map, feats, method, train_fn, input_std=None) -> dict:
    """Build the items of one EC depth and train / score one head.

    labels: {pid: set of classes at this depth} (build_label_sets); train_fn(items_tr,
    items_va, items_te, n_classes) -> result dict with a 'val' block. input_std: the method's
    InputStandardisation (FIXED 2026-10-19: applied to the items of this depth, i.e. continuous
    features standardised over the residues of this depth's training items; its record is
    stored as r['input_standardisation']); None = items as built (tests only).
    """
    import time
    labelled = sorted(p for p in labels if p in split_map)
    require_features(labelled, feats, method, f'EC d{depth}')
    train_classes = sorted({c for p in labelled if split_map[p] == 'train' for c in labels[p]})
    cls_to_idx = {c: i for i, c in enumerate(train_classes)}
    n_classes = len(train_classes)
    print(f'[d{depth}] labelled={len(labelled)} n_classes={n_classes}', flush=True)
    bucket = {'train': [], 'val': [], 'test': []}
    for pid in labelled:
        y = np.zeros(n_classes, dtype=np.float32)
        for c in labels[pid]:
            if c in cls_to_idx:
                y[cls_to_idx[c]] = 1.0
        bucket[split_map[pid]].append({'pid': pid, 'x': feats[pid], 'y': y})
    # Val / test complexes without a training class are dropped (counted).
    n_drop = {}
    for s in ('val', 'test'):
        kept = [it for it in bucket[s] if it['y'].sum() > 0]
        n_drop[s] = len(bucket[s]) - len(kept)
        bucket[s] = kept
    if not bucket['train'] or not bucket['test']:
        return {'error': 'empty split', 'depth': depth}
    # Top-1 baseline: predict the most common training class for everyone (test and val).
    counts = np.sum([it['y'] for it in bucket['train']], axis=0)
    maj_class = int(counts.argmax())
    std_rec = None
    if input_std is not None:          # FIXED(function input standardisation, 2026-10-19)
        bucket, std_rec = input_std.apply(bucket)
    t0 = time.time()
    r = train_fn(bucket['train'], bucket['val'], bucket['test'], n_classes)
    if std_rec is not None:
        r['input_standardisation'] = std_rec
    r['majority_top1_hit'] = float(np.mean([it['y'][maj_class] > 0 for it in bucket['test']]))
    r['val']['majority_top1_hit'] = (float(np.mean([it['y'][maj_class] > 0 for it in bucket['val']]))
                                     if bucket['val'] else float('nan'))
    r['runtime_s'] = round(time.time() - t0, 1)
    r['depth'] = depth
    r.update({'ec_classes': classes_record(train_classes)['classes'],
              'classes_sha256': classes_record(train_classes)['classes_sha256'],
              'dropped_no_train_class': n_drop, **items_record(bucket)})
    return r


# FIXED(function_reaudit pre-run fixes 2 / 6, 2026-10-19): GO items, shared by both GO
#   scripts. Old: complexes without a feature of the method were dropped silently and items
#   came in cache order. New: every complex of the split with a selected GO term must have a
#   feature (error otherwise), items in sorted pid order.
def go_items(labels: dict, classes: list, split_map: dict, feats: dict, method: str) -> dict:
    """{split: [{'pid', 'x', 'y'}]} of the GO probe (multi-hot over `classes`)."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    pids = sorted(p for p in labels if p in split_map)
    require_features(pids, feats, method, 'GO')
    bucket = {'train': [], 'val': [], 'test': []}
    for pid in pids:
        y = np.zeros(len(classes), dtype=np.float32)
        for c in labels[pid]:
            if c in cls_to_idx:
                y[cls_to_idx[c]] = 1.0
        bucket[split_map[pid]].append({'pid': pid, 'x': feats[pid], 'y': y})
    for s in ('val', 'test'):                 # (labels holds only pids with a selected term)
        bucket[s] = [it for it in bucket[s] if it['y'].sum() > 0]
    return bucket


def go_val_floor(sel_info: dict, classes: list, bucket: dict):
    """Frequency floor (training frequency as the score, labels only) on the val items."""
    from multilabel_metrics import frequency_floor
    if not bucket.get('val'):
        return None
    freq = np.array([sel_info['train_freq'][c] for c in classes], dtype=np.float64)
    return frequency_floor(freq, np.stack([it['y'] for it in bucket['val']]))


# FIXED(function_reaudit pre-run fixes 2 / 3 / 4, 2026-10-19): binding-site and affinity
#   items, shared by the Ensembits and baseline scripts. Old: features and labels were cut to
#   the shorter length, complexes < 10 residues and (affinity) without MACCS were skipped
#   without a count, complexes without a feature were dropped, and the protein-only affinity
#   head (no --use-ligand-maccs) used more complexes than the ligand-aware one. New: length
#   mismatch and a missing feature are errors, every drop is counted per split, and every
#   affinity run (protein + ligand, protein only, ligand only) scores the same complexes:
#   finite label, MACCS fingerprint, >= 10 residues.
def _split_of(split_sets: dict) -> dict:
    return {p: s for s in SPLITS for p in split_sets.get(s, ())}


def binding_site_items(split_sets: dict, bs_lab: dict, feats: dict, method: str) -> tuple:
    """(bucket, dropped) of the binding-site task; split_sets = {split: set of pids}."""
    where = _split_of(split_sets)
    pids = sorted(where)
    no_label = [p for p in pids if p not in bs_lab]
    short = [p for p in pids if p in bs_lab and len(bs_lab[p]) < MIN_RESIDUES]
    use = [p for p in pids if p in bs_lab and len(bs_lab[p]) >= MIN_RESIDUES]
    require_features(use, feats, method, 'binding-site')
    bucket = {s: [] for s in SPLITS}
    for p in use:
        check_length(p, len(feats[p]), len(bs_lab[p]), method, 'binding site')
        bucket[where[p]].append({'pid': p, 'x': feats[p], 'y': bs_lab[p]})
    dropped = dropped_counts(where, bucket, {'no_binding_label': no_label,
                                             f'fewer_than_{MIN_RESIDUES}_residues': short})
    return bucket, dropped


def affinity_items(split_sets: dict, aff_lab: dict, maccs: dict, n_res: dict, feats,
                   method: str, with_ligand: bool) -> tuple:
    """(bucket, dropped) of the affinity task. `n_res`: residue count per pid (length of the
    binding-site label = the MISATO residue list); `feats` None for the ligand-only control."""
    where = _split_of(split_sets)
    reasons = {'no_affinity_label': [], 'non_finite_label': [], 'no_maccs': [],
               'no_residue_count': [], f'fewer_than_{MIN_RESIDUES}_residues': []}
    use = []
    for p in sorted(where):
        if p not in aff_lab:
            reasons['no_affinity_label'].append(p); continue
        if not np.isfinite(float(aff_lab[p])):
            reasons['non_finite_label'].append(p); continue
        if p not in maccs:
            reasons['no_maccs'].append(p); continue
        if p not in n_res:
            reasons['no_residue_count'].append(p); continue
        if n_res[p] < MIN_RESIDUES:
            reasons[f'fewer_than_{MIN_RESIDUES}_residues'].append(p); continue
        use.append(p)
    if feats is not None:
        require_features(use, feats, method, 'affinity')
    bucket = {s: [] for s in SPLITS}
    for p in use:
        item = {'pid': p, 'y': float(aff_lab[p])}
        if feats is not None:
            check_length(p, len(feats[p]), n_res[p], method, 'affinity')
            item['x'] = feats[p]
        if with_ligand:
            item['lig'] = maccs[p]
        bucket[where[p]].append(item)
    return bucket, dropped_counts(where, bucket, reasons)


# ---------------------------------------------------------------- input standardisation
# FIXED(function input standardisation, authors' decision 2026-10-19): new. Old: every 1D-conv
#   function probe fed its inputs raw (Ensembits codebook per-dimension std ~0.8 vs ESM3struct
#   3.8 / AminoAseed 5.7; the non-paper check `--standardize-codebook` / `aminoaseed_std` /
#   `esm3struct_std` of the EC/GO conditioning check is now the default for every method).
# Open(audit D.6): appendix.tex app:downstreams does not state the input standardisation of
#   the function heads (lookup tables over their rows, continuous features over the training
#   residues, MACCS unchanged). Text pass item.
INPUT_STD_VERSION = 'function-input-std-v1'
STD_EPS = 1e-8
LOOKUP_RULE = ('lookup table: per-dimension (x - mean) / std over the rows of the lookup table '
               '(every row once, ddof 0; std < 1e-8: centred only); fixed, label free')
TRAIN_RULE = ('continuous: per-dimension (x - mean) / std over the residues of the training '
              'items of this split / CV rotation (ddof 0; std < 1e-8: centred only)')
TRAINABLE_RULE = 'none: trainable embedding (no fixed per-residue input)'
NO_INPUT_RULE = 'none: no per-residue input (ligand only)'
DISABLED_RULE = 'none: inputs fed raw (--no-input-standardisation; pre-2026-10-19 protocol)'
# MACCS ligand branch of the affinity head: unchanged for every method. Reason: the 167 MACCS
#   keys are binary (0/1, already on a bounded common scale); the fingerprint and the affinity
#   items are identical for every method, so it cannot change a comparison between methods;
#   z-scoring rare keys (training frequency ~1 %) would turn a present key into |z| ~ 10 and
#   keys never set in training into a constant, i.e. it would add scale problems rather than
#   remove them.
LIGAND_RULE = 'MACCS fingerprint unchanged (binary 0/1), identical for every method'


def _mean_std(X) -> tuple:
    """(mean, std, std with constant dimensions set to 1) per column, float64, ddof 0."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0); sd = X.std(axis=0)
    return mu, sd, np.where(sd < STD_EPS, 1.0, sd)


def stats_sha256(mu, sd) -> str:
    """sha256 of a (mean, std) pair: shape + float64 little-endian bytes of both."""
    mu = np.ascontiguousarray(mu, dtype='<f8'); sd = np.ascontiguousarray(sd, dtype='<f8')
    h = hashlib.sha256(f'{mu.shape}|{sd.shape}|'.encode())
    h.update(mu.tobytes()); h.update(sd.tobytes())
    return h.hexdigest()


def standardize_lookup_table(table, name: str = 'table') -> tuple:
    """(standardised table float32, record) of a frozen lookup table (M rows, d dims).

    Each dimension is centred and scaled by its mean / std over the M rows (every row once,
    ddof 0, not weighted by token frequency, no labels), so the lookup x = T[id] becomes
    (T[id] - mean) / std. A dimension with std < 1e-8 is only centred. Replaces the non-paper
    `standardize_codebook_rows` of the EC/GO conditioning check (same arithmetic).
    """
    mu, sd, sd_safe = _mean_std(table)
    X = np.asarray(table, dtype=np.float64)
    rec = {'name': name, 'kind': 'lookup', 'rule': LOOKUP_RULE, 'n_rows': int(X.shape[0]),
           'dim': int(X.shape[1]), 'n_constant_dims': int((sd < STD_EPS).sum()),
           'mean_abs_mean': float(np.abs(mu).mean()), 'mean_std': float(sd.mean()),
           'stats_sha256': stats_sha256(mu, sd)}
    return ((X - mu) / sd_safe).astype(np.float32), rec


def train_residue_stats(items, a: int = 0, b: int = None) -> tuple:
    """(mean, std, n_residues) per dimension of columns a:b over every residue of `items`
    (each item's 'x' is an (L, D) array). Two passes in float64 (no concatenation), ddof 0."""
    n = 0; s1 = None
    for it in items:
        x = np.asarray(it['x'], dtype=np.float64)[:, a:b]
        s1 = x.sum(axis=0) if s1 is None else s1 + x.sum(axis=0)
        n += x.shape[0]
    if not n:
        raise FunctionInputError('no training residue to standardise on')
    mu = s1 / n
    s2 = None
    for it in items:
        d = np.asarray(it['x'], dtype=np.float64)[:, a:b] - mu
        s2 = (d * d).sum(axis=0) if s2 is None else s2 + (d * d).sum(axis=0)
    return mu, np.sqrt(s2 / n), n


class InputStandardisation:
    """Input rule of one method of a 1D-conv function probe (EC, GO, binding, affinity).

    blocks: [{'name', 'kind', 'dims': [a, b], ...}] covering the per-residue feature columns:
      kind 'lookup'          already standardised over the rows of its table when the feature
                             was built (standardize_lookup_table; its record in 'tables');
      kind 'train_residues'  standardised in `apply` with the mean / std of the training items;
      kind 'trainable'       a trainable embedding (Ensembits --random-init-embedding): no
                             fixed input, nothing to standardise.
    No block = no per-residue input (ligand_only). enabled=False (--no-input-standardisation):
    nothing is standardised (lookup tables must then be raw as well).
    """

    def __init__(self, method: str, blocks: list, enabled: bool = True):
        self.method = str(method)
        self.blocks = [dict(b) for b in blocks]
        self.enabled = bool(enabled)

    # --- constructors
    @classmethod
    def disabled(cls, method: str) -> 'InputStandardisation':
        return cls(method, [], enabled=False)

    @classmethod
    def lookup(cls, method: str, dim: int, table_records: list) -> 'InputStandardisation':
        if not table_records:
            raise FunctionInputError(f'{method}: lookup feature without a standardised table')
        return cls(method, [{'name': method, 'kind': 'lookup', 'dims': [0, int(dim)],
                             'tables': list(table_records)}])

    @classmethod
    def continuous(cls, method: str, dim: int) -> 'InputStandardisation':
        return cls(method, [{'name': method, 'kind': 'train_residues', 'dims': [0, int(dim)]}])

    def concat(self, other: 'InputStandardisation', name: str = None) -> 'InputStandardisation':
        """Rule of the feature [self | other] (columns of `other` shifted after self's)."""
        if self.enabled != other.enabled:
            raise ValueError('cannot concatenate a standardised and a raw input')
        off = max([b['dims'][1] for b in self.blocks] or [0])
        sh = [{**b, 'dims': [b['dims'][0] + off, b['dims'][1] + off]} for b in other.blocks]
        return InputStandardisation(name or f'{self.method}+{other.method}',
                                    self.blocks + sh, self.enabled)

    # --- records
    def _rule(self) -> str:
        if not self.enabled:
            return DISABLED_RULE
        kinds = {b['kind'] for b in self.blocks}
        if not kinds:
            return NO_INPUT_RULE
        if kinds == {'trainable'}:
            return TRAINABLE_RULE
        return ' + '.join(r for k, r in (('lookup', LOOKUP_RULE), ('train_residues', TRAIN_RULE))
                          if k in kinds)

    def _has_train(self) -> bool:
        return self.enabled and any(b['kind'] == 'train_residues' for b in self.blocks)

    def base_record(self) -> dict:
        """Record without training statistics (the whole record for lookup-only methods)."""
        blocks = []
        for b in self.blocks:
            bb = {k: v for k, v in b.items() if k not in ('mean', 'std')}
            blocks.append(bb)
        shas = [t['stats_sha256'] for b in self.blocks for t in b.get('tables', [])]
        return {'enabled': self.enabled, 'version': INPUT_STD_VERSION, 'rule': self._rule(),
                'method': self.method, 'blocks': blocks,
                'stats_sha256': (hashlib.sha256('|'.join(shas).encode()).hexdigest()
                                 if self.enabled else None),
                'ligand_maccs': LIGAND_RULE}

    def run_record(self, block_records=()) -> dict:
        """Top-level record of a result JSON. For methods with training-residue blocks the
        statistics differ per item set (EC depth, split / task): stats_sha256 is then the
        sha256 over the per-block stats_sha256 in the given order (`per_block`: their keys)."""
        rec = self.base_record()
        if self._has_train():
            items = [(str(k), r['stats_sha256']) for k, r in block_records]
            rec['per_block'] = [k for k, _ in items]
            rec['stats_sha256'] = (hashlib.sha256('|'.join(f'{k}:{v}' for k, v in items)
                                                  .encode()).hexdigest() if items else None)
        return rec

    # --- application
    def apply(self, bucket: dict) -> tuple:
        """(bucket with standardised 'x', record) for one item set {split: [items]}.

        Lookup / trainable blocks are left as they are (lookup tables were standardised when
        the feature was built); training-residue blocks get the per-dimension mean / std of
        the residues of bucket['train'] only, applied to every split. The input items and
        their arrays are not modified (new dicts, new arrays)."""
        rec = self.base_record()
        if not self._has_train():
            return bucket, rec
        train = bucket.get('train') or []
        x0 = np.asarray(train[0]['x']) if train else None
        D = x0.shape[1] if x0 is not None and x0.ndim == 2 else None
        cover = max(b['dims'][1] for b in self.blocks)
        if D is not None and D != cover:
            raise FunctionInputError(f'{self.method}: feature width {D} != rule width {cover}')
        mus, sds, blocks, shas = {}, {}, [], []
        for b in self.blocks:
            bb = {k: v for k, v in b.items()}
            if b['kind'] == 'train_residues':
                a, e = b['dims']
                mu, sd, n = train_residue_stats(train, a, e)
                mus[(a, e)] = mu; sds[(a, e)] = np.where(sd < STD_EPS, 1.0, sd)
                bb.update({'n_train_items': len(train), 'n_train_residues': int(n),
                           'n_constant_dims': int((sd < STD_EPS).sum()),
                           'stats_sha256': stats_sha256(mu, sd)})
            blocks.append(bb)
            shas += [t['stats_sha256'] for t in b.get('tables', [])]
            if b['kind'] == 'train_residues':
                shas.append(bb['stats_sha256'])

        def _std(x):
            x = np.array(x, dtype=np.float32, copy=True)
            for (a, e), mu in mus.items():
                x[:, a:e] = ((x[:, a:e].astype(np.float64) - mu) / sds[(a, e)]).astype(np.float32)
            return x
        out = {s: [{**it, 'x': _std(it['x'])} for it in items] for s, items in bucket.items()}
        rec['blocks'] = blocks
        rec['stats_sha256'] = hashlib.sha256('|'.join(shas).encode()).hexdigest()
        return out, rec


def add_input_std_arg(ap) -> None:
    """--no-input-standardisation on a probe's argparse parser (same flag in all six probes)."""
    ap.add_argument('--no-input-standardisation', action='store_true',
                    help='feed the head inputs raw (the protocol before 2026-10-19, to reproduce '
                         'old numbers); the output name gets "rawin" appended to --tag and the '
                         'result is refused by the compilers / runners')


def input_std_tag(args) -> None:
    """Append 'rawin' to args.tag when --no-input-standardisation (no clash with the default)."""
    if getattr(args, 'no_input_standardisation', False):
        args.tag = f'{args.tag}_rawin' if args.tag else 'rawin'


def input_std_problem(js: dict):
    """Why a result's `input_standardisation` record is not the current rule (None = fine)."""
    rec = (js or {}).get('input_standardisation')
    if not isinstance(rec, dict):
        return 'no input_standardisation record (inputs fed raw, before 2026-10-19)'
    if rec.get('version') != INPUT_STD_VERSION:
        return (f'input standardisation {rec.get("version")!r} is not {INPUT_STD_VERSION!r}')
    if not rec.get('enabled', False):
        return 'inputs not standardised (--no-input-standardisation)'
    if rec.get('rule') not in (NO_INPUT_RULE, TRAINABLE_RULE) and not rec.get('stats_sha256'):
        return 'input standardisation without statistics hash'
    return None


# ---------------------------------------------------------------- inputs
def misato_aa_onehot(pids, real_bb_dir=None, verify: bool = True) -> tuple:
    """({pid: (L, 21) float32 one-hot}, record) for the `aa` baseline.

    Sequence = `seq` of misato_real_bb/<pid>.npz (the MISATO residue list of every token cache
    and label; CYX / CYM -> C, HID / HIE / HIP -> H by build_misato_real_bb's THREE_TO_ONE),
    the same source as the RMSF probe's aa (probe_rmsf_misato.build_aa). Every pid must have
    a file from the current MISATO parser (FunctionInputError otherwise, unless verify=False).
    FIXED(function_reaudit Q11, 2026-10-19): old = the sequence of
    aa2ensembits_pairs_misato_v2.pkl (3,903 of the 4,894 complexes, CYX -> X), so the aa rows
    were scored on fewer complexes than every other method.
    """
    from ensembits.data_versions import MISATO_PARSER_VERSION
    d = Path(real_bb_dir) if real_bb_dir is not None else ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'
    feats, bad_parser, missing = {}, [], []
    h = hashlib.sha256()
    for pid in sorted({str(p).upper() for p in pids}):
        f = d / f'{pid}.npz'
        if not f.exists():
            missing.append(pid); continue
        with np.load(f, allow_pickle=False) as z:
            seq = str(z['seq'])
            pv = str(z['parser_version']) if 'parser_version' in z.files else None
        if pv != MISATO_PARSER_VERSION:
            bad_parser.append(pid)
        x = np.zeros((len(seq), len(AA_VOCAB)), dtype=np.float32)
        x[np.arange(len(seq)), [AA_IDX.get(a, AA_IDX['X']) for a in seq]] = 1.0
        feats[pid] = x
        h.update(f'{pid}\t{seq}\n'.encode())
    if verify and (missing or bad_parser):
        raise FunctionInputError(f'aa: {len(missing)} complexes without misato_real_bb file '
                                 f'{missing[:5]}, {len(bad_parser)} from another parser than '
                                 f'{MISATO_PARSER_VERSION} {bad_parser[:5]}')
    rec = {'file': 'misato_real_bb (seq)', 'kind': 'misato_real_bb_seq',
           'real_bb_dir': _rel(d), 'parser_version': MISATO_PARSER_VERSION,
           'n_pids': len(feats), 'sequences_sha256': h.hexdigest(),
           'verified': bool(verify and not missing and not bad_parser)}
    return feats, rec


def check_coords_ca8(path, coords=None) -> dict:
    """Record of coords_ca8_misato.pkl (the `random` / `pca_eig` input): its
    build_coords_ca8_misato.py sidecar must name the current parser only and its coords_sha256
    must equal the content hash of the file (FunctionInputError otherwise; same rule as
    probe_rmsf_misato._use_coords)."""
    import pickle
    from ensembits.data_versions import MISATO_PARSER_VERSION, read_provenance, hash_arrays_by_pid
    path = Path(path)
    prov = read_provenance(path)
    if prov is None or prov.get('parser_version') != [MISATO_PARSER_VERSION]:
        raise FunctionInputError(f'{path.name}: no sidecar or not from parser {MISATO_PARSER_VERSION}')
    if coords is None:
        with open(path, 'rb') as f:
            coords = pickle.load(f)
    if prov.get('coords_sha256') != hash_arrays_by_pid(coords):
        raise FunctionInputError(f'{path.name} does not match its sidecar (coords_sha256)')
    return {'file': path.name, 'path': _rel(path), 'sha256': _file_sha256(path),
            'coords_sha256': prov['coords_sha256'], 'verified': True}


def token_cache_record(path) -> dict:
    """Path and sha256 of an Ensembits token cache (its sidecar is checked by tokenize_misato)."""
    path = Path(path)
    return {'file': path.name, 'path': _rel(path), 'sha256': _file_sha256(path),
            'kind': 'ensembits_token_cache', 'verified': True}


def _label_file_record(path, schema: str, builder: str) -> dict:
    """{path, sha256, schema} of a label / ligand file; SystemExit unless its builder's sidecar
    has `schema` and the file's sha256."""
    from ensembits.data_versions import read_provenance
    path = Path(path)
    if not path.exists():
        raise SystemExit(f'{path} missing: run {builder}')
    prov = read_provenance(path)
    if not prov or prov.get('schema') != schema:
        raise SystemExit(f'{path.name}: no {schema} sidecar; run {builder} --verify-existing '
                         f'(or build it)')
    sha = _file_sha256(path)
    if prov.get('output_sha256') != sha:
        raise SystemExit(f'{path.name}: sha256 {sha[:12]} differs from its sidecar '
                         f'({str(prov.get("output_sha256"))[:12]}); rebuild it')
    return {'path': _rel(path), 'sha256': sha, 'schema': schema}


def load_affinity_labels(path=None) -> tuple:
    """({PID: -log Kd/Ki}, record) of data/annotations/misato_affinity.json (sidecar checked)."""
    from ensembits.paths import MISATO_AFFINITY
    p = Path(path) if path is not None else Path(MISATO_AFFINITY)
    rec = _label_file_record(p, AFFINITY_SCHEMA, 'data_prep/build_misato_affinity_labels.py')
    lab = json.loads(p.read_text())
    return {str(k).upper(): float(v) for k, v in lab.items()}, rec


def load_ligand_maccs(path=None) -> tuple:
    """({PID: (167,) float32}, record) of misato_ligand_maccs.npz (sidecar checked)."""
    from ensembits.paths import MISATO_LIGAND_MACCS
    p = Path(path) if path is not None else Path(MISATO_LIGAND_MACCS)
    rec = _label_file_record(p, MACCS_SCHEMA, 'data_prep/build_misato_ligand_maccs.py')
    with np.load(p, allow_pickle=False) as z:
        return {str(k).upper(): z[k].astype(np.float32) for k in z.files}, rec


def affinity_inputs_record(path_labels=None, path_maccs=None) -> dict:
    """{'affinity_labels': rec, 'ligand_maccs': rec} of the current files (result JSONs)."""
    from ensembits.paths import MISATO_AFFINITY, MISATO_LIGAND_MACCS
    return {'affinity_labels': _label_file_record(path_labels or MISATO_AFFINITY, AFFINITY_SCHEMA,
                                                  'data_prep/build_misato_affinity_labels.py'),
            'ligand_maccs': _label_file_record(path_maccs or MISATO_LIGAND_MACCS, MACCS_SCHEMA,
                                               'data_prep/build_misato_ligand_maccs.py')}


# ---------------------------------------------------------------- metrics
def binding_metrics(probs, labels) -> dict:
    """Per-residue binding-site metrics. probs / labels: lists of per-protein 1-D arrays.

    auroc / ap: pooled over all residues (as before); auroc_per_protein_mean / _median: the
    AUROC within each protein with both classes, averaged over those proteins."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    py = np.concatenate(probs) if probs else np.zeros(0)
    yy = np.concatenate(labels) if labels else np.zeros(0)
    two = len(np.unique(yy)) == 2
    per = [float(roc_auc_score(y, p)) for p, y in zip(probs, labels)
           if len(np.unique(y)) == 2]
    return {'auroc': float(roc_auc_score(yy, py)) if two else float('nan'),
            'ap': float(average_precision_score(yy, py)) if two else float('nan'),
            'auroc_per_protein_mean': float(np.mean(per)) if per else float('nan'),
            'auroc_per_protein_median': float(np.median(per)) if per else float('nan'),
            'n_proteins_auroc': len(per), 'n_proteins': len(labels),
            'n_residues': int(yy.shape[0]), 'frac_pos': float(yy.mean()) if len(yy) else float('nan')}


def affinity_metrics(pred, y) -> dict:
    """R², Spearman and MSE of affinity predictions (as before), plus n."""
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    pred = np.asarray(pred, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return {'r2': float(r2_score(y, pred)), 'spearman': float(spearmanr(pred, y).correlation),
            'mse': float(((pred - y) ** 2).mean()), 'n': int(len(y))}


# ---------------------------------------------------------------- result checks
def atomic_write_json(path, obj, indent: int = 2) -> None:
    """Write `obj` as JSON to `path` through a temporary file + os.replace (no half-written
    result JSON if the job is killed)."""
    import os
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(obj, indent=indent))
    os.replace(tmp, path)


def protocol_record(feature_inputs: list, input_std: dict = None, **extra) -> dict:
    """Top-level fields every function result JSON carries.

    FIXED(function input standardisation, 2026-10-19): new `input_std` (the method's
    InputStandardisation record, written as `input_standardisation`); a result without it is
    refused by function_result_problems."""
    out = {'function_protocol': FUNCTION_PROTOCOL, 'function_protocol_note': PROTOCOL_NOTE,
           'feature_inputs': feature_inputs,
           'cache_verified': all(r.get('verified', False) for r in feature_inputs)
           if feature_inputs else False, **extra}
    if input_std is not None:
        out['input_standardisation'] = input_std
    return out


def function_result_problems(js: dict, task: str = None) -> list:
    """Reasons a function result JSON must not be compiled or reused ([] = fine).

    Checks: the protocol tag; the input standardisation record (current version, enabled,
    statistics hash; 2026-10-19); verified feature inputs (Ensembits token cache / baseline
    cache with a current sidecar, or the misato_real_bb sequences); for affinity, the sha256
    of the current affinity labels and MACCS file."""
    js = js or {}
    out = []
    if js.get('function_protocol') != FUNCTION_PROTOCOL:
        out.append(f'function protocol {js.get("function_protocol")!r} is not '
                   f'{FUNCTION_PROTOCOL!r} (built before the 2026-10-19 pre-run fixes or '
                   f'before the input standardisation; rerun)')
        return out
    why = input_std_problem(js)        # FIXED(function input standardisation, 2026-10-19)
    if why:
        out.append(why)
    if not js.get('feature_inputs'):
        out.append('no feature-input record')
    elif not js.get('cache_verified', False):
        out.append('feature inputs not verified (--allow-unverified-cache)')
    if task == 'affinity':
        try:
            cur = affinity_inputs_record()
        except SystemExit as e:
            return out + [f'current affinity inputs unusable: {e}']
        for k in ('affinity_labels', 'ligand_maccs'):
            got = (js.get(k) or {}).get('sha256')
            if got != cur[k]['sha256']:
                out.append(f'{k} sha256 {str(got)[:12]} is not the current '
                           f'{Path(cur[k]["path"]).name} ({cur[k]["sha256"][:12]})')
    return out


def stale_function_outputs(paths) -> list:
    """Existing EC / GO / binding / affinity result JSONs not built with the current function
    protocol (or on other affinity inputs): [(path, reason)]. The runners refuse to skip
    (= reuse) them."""
    bad = []
    for o in paths:
        if o is None:
            continue
        o = Path(o)
        if not o.exists() or o.suffix != '.json':
            continue
        try:
            js = json.loads(o.read_text())
        except Exception as e:                          # noqa: BLE001
            bad.append((o, f'unreadable ({type(e).__name__})')); continue
        aff = any(str(k).endswith('_affinity') for k in (js.get('results') or {}))
        why = function_result_problems(js, 'affinity' if aff else None)
        if why:
            bad.append((o, why[0]))
    return bad


def cell_signature(js: dict, task: str, split: str = None, depth=None):
    """(classes_sha256 or None, items_sha256) of one table cell of a result JSON, or None.

    task: 'EC' (summary JSON, `depth`), 'GO', 'binding' / 'affinity' (`split`)."""
    js = js or {}
    if task == 'EC':
        c = (js.get('by_depth') or {}).get(str(depth)) or {}
        return (c.get('classes_sha256'), c.get('items_sha256')) if c.get('items_sha256') else None
    if task == 'GO':
        return ((js.get('go_classes_sha256'), js.get('items_sha256'))
                if js.get('items_sha256') else None)
    c = (js.get('results') or {}).get(f'{split}_{task}') or {}
    return (None, c.get('items_sha256')) if c.get('items_sha256') else None
