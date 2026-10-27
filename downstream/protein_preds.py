"""Per-protein prediction sidecars of the downstream probes (write, read, verify).

What it does
------------
Every RMSF, EC, GO, binding-site and affinity probe writes, next to its result JSON, a
compact `.preds.npz` sidecar with the predictions of the SELECTED model (the epoch kept by
early stopping on validation) for every validation and test item, so that protein-level
significance tests (downstream/protein_level_stats.py) can be run without another probe run
(authors' decision A4 (a), camera_ready/final_fix_list.md, 2026-10-19). The result JSON
records the sidecar as `per_protein_preds` = {file, sha256, schema, kind, n_items, ids_sha256,
selected}; the significance step refuses a result without that record or whose file is
missing or changed. The tables do not need the sidecar (nothing in the compilers changes).

Kinds (one per task family):

  * `residue_regression` (RMSF): per protein, per-residue prediction and label; `residue_index`
    = position of the residue in the protein's residue list (label order), and for the flat
    datasets (mdCATH-div, BioEmu-CATH) `row` = row of the label / descriptor arrays;
  * `residue_binary` (binding site): per protein, per-residue probability and 0/1 label,
    `residue_index` as above;
  * `protein_multilabel` (EC per depth, GO): per protein, the sigmoid score of every class and
    the multi-hot label; the class list (`classes`, column order) is stored;
  * `protein_regression` (affinity): per complex, prediction and label.

File layout (np.load(..., allow_pickle=False) reads it; written deterministically: fixed zip
timestamps, sorted members, so the same predictions give the same sha256):

    meta_json                       0-d str: JSON of the meta block (schema, kind, task, ...)
    classes                         (C,) str           protein_multilabel only
    <split>__ids                    (n,) str           item ids (protein / complex), sorted
    residue kinds:  <split>__offsets (n+1,) int64, <split>__pred (N,), <split>__label (N,),
                    <split>__residue_index (N,) int32 [, <split>__row (N,) int64]
    protein_multilabel: <split>__scores (n, C) float32, <split>__labels (n, C) uint8
    protein_regression: <split>__pred (n,), <split>__label (n,)

with <split> in (val, test). Item ids are sorted, residues of one item are contiguous and in
residue order; predictions are stored in the dtype the probe produced (float32 for the
torch heads), labels as given (RMSF float32, binding 0/1 as uint8).

Paper
-----
Not described in the paper yet (infrastructure for the protein-level significance tests of
A4 (a); Open, audit D.17: appendix section on significance; text pass item).

Origin
------
New (2026-10-19, authors' decision A4 (a)).

Used by the reported model: evaluation only (sidecars of every probe run).

Inputs / outputs
----------------
writes <result>.preds.npz (or <result stem>.<block>.preds.npz for the binding / affinity
blocks of one JSON) next to the result JSON; reads them back for the significance step.

How to run
----------
Library module (imported by the probes and by protein_level_stats.py); tests in
tests/test_protein_level_stats.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

PREDS_SCHEMA = 'protein-preds-v1'
PREDS_KEY = 'per_protein_preds'          # key of the record in a result JSON (or block)
KINDS = ('residue_regression', 'residue_binary', 'protein_multilabel', 'protein_regression')
EVAL_SPLITS = ('val', 'test')
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class PredsError(ValueError):
    """A sidecar is missing, changed, malformed or inconsistent with its result JSON."""


# ---------------------------------------------------------------- hashes and paths
def ids_sha256(ids) -> str:
    """sha256 over the sorted ids (case kept) joined by newlines."""
    return hashlib.sha256('\n'.join(sorted(str(i) for i in ids)).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(result_path, block: str | None = None) -> Path:
    """<result stem>[.<block>].preds.npz next to the result JSON."""
    p = Path(result_path)
    stem = p.name[:-5] if p.name.endswith('.json') else p.name
    return p.with_name(f'{stem}{("." + block) if block else ""}.preds.npz')


# ---------------------------------------------------------------- building split blocks
def _contiguous_runs(names: np.ndarray, order_key: np.ndarray):
    """Sort residues by (item id, order_key); return (order, ids, offsets)."""
    order = np.lexsort((order_key, names))
    sn = names[order]
    ids, starts = np.unique(sn, return_index=True)
    offsets = np.append(starts, len(sn)).astype(np.int64)
    return order, ids, offsets


def residue_split_from_residues(names, pred, label, residue_index, row=None) -> dict:
    """One split of a residue kind from per-residue arrays (any order).

    names: item id of every residue; residue_index: position of the residue in its item's
    residue list; row: optional global row (flat RMSF datasets). Residues are regrouped by item
    (ids sorted) and ordered by residue_index inside an item; duplicates are refused.
    """
    names = np.asarray(names).astype(str)
    pred = np.asarray(pred).ravel()
    label = np.asarray(label).ravel()
    ri = np.asarray(residue_index, dtype=np.int64).ravel()
    n = len(names)
    if not (len(pred) == len(label) == len(ri) == n) or (row is not None and len(row) != n):
        raise PredsError(f'per-residue arrays differ in length (names {n}, pred {len(pred)}, '
                         f'label {len(label)}, residue_index {len(ri)})')
    order, ids, offsets = _contiguous_runs(names, ri)
    ri_s = ri[order]
    for i in range(len(ids)):
        seg = ri_s[offsets[i]:offsets[i + 1]]
        if len(np.unique(seg)) != len(seg):
            raise PredsError(f'{ids[i]}: a residue index appears twice')
    out = {'ids': ids, 'offsets': offsets, 'pred': pred[order], 'label': label[order],
           'residue_index': ri_s.astype(np.int32)}
    if row is not None:
        out['row'] = np.asarray(row, dtype=np.int64).ravel()[order]
    return out


def residue_split_from_items(ids, preds, labels, residue_index=None) -> dict:
    """One split of a residue kind from per-item lists (item i: preds[i], labels[i] of equal
    length; residue_index[i] defaults to 0..L-1)."""
    ids = [str(i) for i in ids]
    if not (len(ids) == len(preds) == len(labels)):
        raise PredsError('ids, preds and labels differ in length')
    if len(set(ids)) != len(ids):
        raise PredsError('an item id appears twice')
    names, P, Y, R = [], [], [], []
    for k, (i, p, y) in enumerate(zip(ids, preds, labels)):
        p = np.asarray(p).ravel(); y = np.asarray(y).ravel()
        if len(p) != len(y):
            raise PredsError(f'{i}: {len(p)} predictions, {len(y)} labels')
        r = (np.arange(len(y)) if residue_index is None
             else np.asarray(residue_index[k], dtype=np.int64).ravel())
        names.append(np.full(len(y), i, dtype=object)); P.append(p); Y.append(y); R.append(r)
    if not names:
        return {'ids': np.array([], dtype=str), 'offsets': np.zeros(1, dtype=np.int64),
                'pred': np.zeros(0, np.float32), 'label': np.zeros(0, np.float32),
                'residue_index': np.zeros(0, np.int32)}
    return residue_split_from_residues(np.concatenate(names), np.concatenate(P),
                                       np.concatenate(Y), np.concatenate(R))


def contiguous_residue_index(groups) -> np.ndarray:
    """Position of every residue inside its run of equal group values (groups contiguous, as
    in probe_rmsf_misato.gather_residues); PredsError if a group appears in two runs."""
    g = np.asarray(groups).ravel()
    if len(g) == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    if len(np.unique(g[starts])) != len(starts):
        raise PredsError('a protein is split over two runs of residues')
    run_start = np.repeat(starts, np.diff(np.append(starts, len(g))))
    return np.arange(len(g)) - run_start


def protein_split(ids, pred, label) -> dict:
    """One split of a protein kind: ids (n,), pred (n,) or scores (n, C), label same shape.
    Items are sorted by id."""
    ids = np.asarray([str(i) for i in ids])
    pred = np.asarray(pred); label = np.asarray(label)
    if len(ids) != len(pred) or pred.shape != label.shape:
        raise PredsError(f'ids {len(ids)}, pred {pred.shape}, label {label.shape}')
    if len(set(ids.tolist())) != len(ids):
        raise PredsError('an item id appears twice')
    order = np.argsort(ids, kind='stable')
    return {'ids': ids[order], 'pred': pred[order], 'label': label[order]}


# ---------------------------------------------------------------- write
def _npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            a = io.BytesIO()
            np.lib.format.write_array(a, np.asarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(f'{k}.npy', date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, a.getvalue())
    return buf.getvalue()


def write_preds(path, *, kind: str, splits: dict, meta: dict, classes=None,
                json_dir=None) -> dict:
    """Write one sidecar atomically (temporary file + os.replace) and return its record.

    splits: {'val': block, 'test': block} from the builders above; meta: JSON-serialisable
    facts copied from the result (task, method, seed, item / residue-set hashes, ...).
    json_dir: folder of the result JSON (the record's `file` is relative to it).
    """
    if kind not in KINDS:
        raise PredsError(f'unknown kind {kind!r}')
    arrays, n_items, id_hash = {}, {}, {}
    for s in EVAL_SPLITS:
        if s not in splits:
            raise PredsError(f'split {s!r} missing')
        b = splits[s]
        ids = np.asarray(b['ids']).astype(str)
        arrays[f'{s}__ids'] = ids
        n_items[s] = int(len(ids)); id_hash[s] = ids_sha256(ids.tolist())
        if kind.startswith('residue_'):
            off = np.asarray(b['offsets'], dtype=np.int64)
            if len(off) != len(ids) + 1 or off[-1] != len(b['pred']):
                raise PredsError(f'{s}: offsets do not match the residues')
            arrays[f'{s}__offsets'] = off
            arrays[f'{s}__pred'] = np.asarray(b['pred'])
            lab = np.asarray(b['label'])
            arrays[f'{s}__label'] = lab.astype(np.uint8) if kind == 'residue_binary' else lab
            arrays[f'{s}__residue_index'] = np.asarray(b['residue_index'], dtype=np.int32)
            if 'row' in b:
                arrays[f'{s}__row'] = np.asarray(b['row'], dtype=np.int64)
        elif kind == 'protein_multilabel':
            sc = np.asarray(b['pred'], dtype=np.float32)
            if sc.ndim != 2 or (classes is not None and sc.shape[1] != len(classes)):
                raise PredsError(f'{s}: scores {sc.shape} vs {len(classes or [])} classes')
            arrays[f'{s}__scores'] = sc
            arrays[f'{s}__labels'] = (np.asarray(b['label']) > 0).astype(np.uint8)
        else:
            arrays[f'{s}__pred'] = np.asarray(b['pred']).ravel()
            arrays[f'{s}__label'] = np.asarray(b['label']).ravel()
    if kind == 'protein_multilabel':
        if classes is None:
            raise PredsError('protein_multilabel needs the class list')
        arrays['classes'] = np.asarray([str(c) for c in classes])
    meta = {**meta, 'schema': PREDS_SCHEMA, 'kind': kind, 'n_items': n_items,
            'ids_sha256': id_hash}
    arrays['meta_json'] = np.asarray(json.dumps(meta, sort_keys=True))
    path = Path(path)
    data = _npz_bytes(arrays)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)
    rel = os.path.relpath(path, json_dir) if json_dir is not None else path.name
    return {'file': rel, 'sha256': hashlib.sha256(data).hexdigest(), 'schema': PREDS_SCHEMA,
            'kind': kind, 'n_items': n_items, 'ids_sha256': id_hash, 'selected': True}


def not_selected_record(reason: str) -> dict:
    """Record of a run without a selected model (e.g. RMSF predictions constant at every
    epoch): no sidecar; the significance step refuses it with `reason`."""
    return {'file': None, 'sha256': None, 'schema': PREDS_SCHEMA, 'selected': False,
            'reason': reason}


# ---------------------------------------------------------------- read / verify
def preds_problem(record, json_path) -> str | None:
    """None if `record` (a JSON's per_protein_preds) names an existing sidecar with the
    recorded sha256 and schema; else the reason (for refusals and for the probes' reuse)."""
    if not record:
        return 'no per-protein prediction sidecar (result written before A4 (a); rerun)'
    if record.get('schema') != PREDS_SCHEMA:
        return f'sidecar schema {record.get("schema")!r} is not {PREDS_SCHEMA!r}'
    if not record.get('selected', False) or not record.get('file'):
        return f'no selected model ({record.get("reason", "no sidecar")})'
    p = Path(json_path).parent / record['file']
    if not p.exists():
        return f'sidecar {record["file"]} missing'
    if file_sha256(p) != record.get('sha256'):
        return f'sidecar {record["file"]} changed after the result was written'
    return None


def needs_rerun(record, json_path) -> str | None:
    """Reason an existing result should be recomputed to get a usable sidecar (no record,
    file missing or changed), or None (sidecar fine, or a recorded 'no selected model', which
    a rerun would reproduce). Used by the probes' --redo-missing-preds."""
    if record and record.get('schema') == PREDS_SCHEMA and not record.get('selected', False):
        return None
    return preds_problem(record, json_path)


def load_preds(record, json_path) -> dict:
    """Read and verify one sidecar: {'meta', 'classes', 'val': block, 'test': block}.

    Raises PredsError for any preds_problem, a meta block that disagrees with the record, or
    malformed arrays. Residue blocks get 'ids', 'offsets', 'pred', 'label', 'residue_index'
    (+ 'row'); protein blocks 'ids', 'pred' (scores for multilabel), 'label'.
    """
    why = preds_problem(record, json_path)
    if why:
        raise PredsError(f'{Path(json_path).name}: {why}')
    p = Path(json_path).parent / record['file']
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z['meta_json']))
        kind = meta.get('kind')
        if meta.get('schema') != PREDS_SCHEMA or kind != record.get('kind'):
            raise PredsError(f'{p.name}: meta schema / kind disagree with the result record')
        out = {'meta': meta, 'classes': z['classes'].tolist() if 'classes' in z.files else None}
        for s in EVAL_SPLITS:
            b = {'ids': z[f'{s}__ids'].astype(str)}
            if ids_sha256(b['ids'].tolist()) != (record.get('ids_sha256') or {}).get(s):
                raise PredsError(f'{p.name}: {s} ids differ from the result record')
            if kind.startswith('residue_'):
                for k in ('offsets', 'pred', 'label', 'residue_index', 'row'):
                    if f'{s}__{k}' in z.files:
                        b[k] = z[f'{s}__{k}']
            elif kind == 'protein_multilabel':
                b['pred'] = z[f'{s}__scores']; b['label'] = z[f'{s}__labels']
            else:
                b['pred'] = z[f'{s}__pred']; b['label'] = z[f'{s}__label']
            out[s] = b
    return out


# ---------------------------------------------------------------- function probes
# Train functions of the EC / GO / binding / affinity probes return, with return_preds=True,
# r['_preds'] = {'val': {'ids', 'pred', 'label'}, 'test': {...}} (item order of the bucket;
# binding: per-item lists of per-residue arrays). The probes pop it and call one writer below.
FUNCTION_META_KEYS = ('function_protocol', 'items_sha256', 'items_pids_sha256', 'n_items',
                      'classes_sha256', 'go_classes_sha256', 'depth', 'scope', 'seed',
                      'epochs_run', 'best_val_loss', 'affinity_input', 'ligand_dim')


def items_preds(items, pred, label) -> dict:
    """{'ids', 'pred', 'label'} of one split from bucket items (each with 'pid')."""
    return {'ids': [str(it['pid']) for it in items], 'pred': pred, 'label': label}


def function_meta(r: dict, **extra) -> dict:
    """Facts of a function result block copied into its sidecar (item / class hashes, ...)."""
    m = {k: r[k] for k in FUNCTION_META_KEYS if k in r}
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def _check_items_hash(preds: dict, r: dict) -> None:
    """The sidecar's val / test ids must be the items of the result (items_pids_sha256)."""
    from function_common import pid_list_sha256
    rec = r.get('items_pids_sha256') or {}
    for s in EVAL_SPLITS:
        if s in rec and pid_list_sha256(preds[s]['ids']) != rec[s]:
            raise PredsError(f'{s}: predicted items differ from the recorded items '
                             f'(items_pids_sha256)')


def write_function_preds(out_path, preds, r: dict, *, task: str, block: str | None = None,
                         classes=None, **meta) -> dict:
    """Write the sidecar of one function result (EC depth, GO run, one binding / affinity
    block) and return its record. task: 'EC', 'GO', 'binding' or 'affinity'."""
    if preds is None:
        return not_selected_record('train function returned no predictions')
    _check_items_hash(preds, r)
    if task in ('EC', 'GO'):
        kind = 'protein_multilabel'
        splits = {s: protein_split(preds[s]['ids'], np.asarray(preds[s]['pred']),
                                   np.asarray(preds[s]['label'])) for s in EVAL_SPLITS}
    elif task == 'binding':
        kind = 'residue_binary'
        splits = {s: residue_split_from_items(preds[s]['ids'], preds[s]['pred'],
                                              preds[s]['label']) for s in EVAL_SPLITS}
    elif task == 'affinity':
        kind = 'protein_regression'
        splits = {s: protein_split(preds[s]['ids'], np.asarray(preds[s]['pred']).ravel(),
                                   np.asarray(preds[s]['label']).ravel()) for s in EVAL_SPLITS}
    else:
        raise PredsError(f'unknown function task {task!r}')
    return write_preds(sidecar_path(out_path, block), kind=kind, splits=splits,
                       meta={'task': task, 'block': block, **function_meta(r, **meta)},
                       classes=classes, json_dir=Path(out_path).parent)
