"""MISATO GO molecular-function labels (v2): paths, loader, class selection, result checks.

What it does
------------
Shared by the GO probes (`tokenize_misato_GO_eval.py` for Ensembits,
`baselines/conv1d_GO_misato.py` for the baselines), the label builder
(`data_prep/build_misato_go_labels.py`) and the readers of GO result JSONs
(`_compile_rebuttal_results.py`, `analyses/plot_skill_radar.py`, `seed_rescue.py`, the
two batteries):

  * Paths and pins: the label file `data/annotations/misato_go_mf_v2.json` (built by
    the builder), the in-repo copy of the source annotations and the pinned
    `go-basic.obo` release (URL, data-version, sha256).
  * `load_go_labels`: reads the label file, recomputes its sha256 and checks it against
    the provenance sidecar written by the builder (an edited or foreign file is refused).
  * `select_go_classes` / `build_go_labelset`: the class list of one split (authors'
    decision D7 (a), 2026-10-18, camera_ready/benchmark_audit.md D7), from the TRAINING
    proteins of that split only (every pid the split file puts in train, whatever
    features a method has, so every method of a split gets the same classes):
      - candidates = MF terms (after propagation) annotated in >= MIN_TRAIN_COUNT (10)
        training proteins and in < MAX_TRAIN_FRAC (50 %) of the annotated training
        proteins (training proteins with at least one MF label);
      - ranked by (-count, GO id); `top50` keeps the first 50 (`top200` 200, `all`
        every candidate). The order does not depend on any file's read order.
    Proteins with none of the selected terms are dropped from the split (as before).
  * `go_result_fields`: the GO-specific block of a result JSON (label file + sha256,
    class list with training counts, selection rule, per-split kept / dropped counts,
    sha256 of the test pid list, frequency floor computed from labels only).
  * `go_result_problems`: why a GO result JSON cannot be compiled (not built on the
    current label file); used by the compilers to refuse old results.

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{GO Label Prediction} ("we keep the
50 most frequent terms in the training set"; the label definition and the class band
of D7 are not yet described there); `2_results.tex` \\label{sec:downstreams}.

Origin
------
New (2026-10-18, benchmark_audit D7). Replaces `build_go_labelset` of
`tokenize_misato_GO_eval.py` (old: raw `misato_go.json` MF + BP + CC terms, no
propagation, top-50 by `Counter.most_common` over the training proteins that had
features, ties at rank 50 broken by read order).

Used by the reported model: yes (GO rows of every table).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads  data/annotations/misato_go_mf_v2.json (+ .provenance.json)

How to run
----------
Library module (no CLI); tests in tests/test_go_labels.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT, EXTERNAL   # noqa: E402

# ---------------------------------------------------------------- pins and paths
GO_SCHEMA = 'misato-go-mf-v2'
GO_LABELS_V2 = ROOT / 'data' / 'annotations' / 'misato_go_mf_v2.json'
# The old label file (MF + BP + CC, unpropagated); read only to recognise old results.
GO_LABELS_OLD = ROOT / 'data' / 'annotations' / 'misato_go.json'
# In-repo copy of the source annotations (per PDB entry, SIFTS GO terms, union over the
# entry's chains; builder not in the research repo). The origin path, owner and date of
# the copy are in its sidecar (`<file>.provenance.json`).
GO_SOURCE = ROOT / 'data' / 'annotations' / 'source' / 'misato_go.json'
GO_SOURCE_SHA256 = 'bfeff2bae2411b04af27deff6aefba3f950caf90e342175b5f78615af6d6067c'
# Pinned ontology: go-basic.obo with data-version releases/2026-07-26, as served by the
# GO release archive snapshot 2026-08-05 (identical bytes to the purl / current release on
# 2026-10-18).
GO_OBO = EXTERNAL / 'go' / 'go-basic_2026-07-26.obo'
GO_OBO_URL = 'https://release.geneontology.org/2026-08-05/ontology/go-basic.obo'
GO_OBO_DATA_VERSION = 'releases/2026-07-26'
GO_OBO_SHA256 = 'b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc'

# Class selection (authors' decision D7 (a), 2026-10-18).
MIN_TRAIN_COUNT = 10
MAX_TRAIN_FRAC = 0.5
SCOPE_N = {'top50': 50, 'top200': 200, 'all': None}
SELECTION_RULE = (
    'per split, from the training proteins of the split file (independent of features): '
    f'MF terms (propagated labels) in >= {MIN_TRAIN_COUNT} training proteins and in '
    f'< {MAX_TRAIN_FRAC:.0%} of the training proteins with >= 1 MF label; ranked by '
    '(-count, GO id); top50 = first 50. Proteins without a selected term are dropped.')


def file_sha256(path) -> str:
    """sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pid_list_sha256(pids) -> str:
    """sha256 over the sorted, upper-cased pids joined by newlines."""
    return hashlib.sha256('\n'.join(sorted(str(p).upper() for p in pids)).encode()).hexdigest()


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- loader
def load_go_labels(path=GO_LABELS_V2) -> tuple[dict, dict]:
    """(pid -> list of MF terms, label info) of the v2 label file.

    The file's sha256 must match the one recorded by the builder in its sidecar and the
    sidecar's schema must be GO_SCHEMA; otherwise SystemExit.
    """
    path = Path(path)
    side = Path(str(path) + '.provenance.json')
    if not path.exists() or not side.exists():
        raise SystemExit(f'{path} (or its .provenance.json) is missing; build it with '
                         f'data_prep/build_misato_go_labels.py')
    prov = json.loads(side.read_text())
    sha = file_sha256(path)
    if prov.get('schema') != GO_SCHEMA or prov.get('sha256') != sha:
        raise SystemExit(f'{path}: sha256 {sha[:12]} / schema {prov.get("schema")!r} do not '
                         f'match its sidecar ({str(prov.get("sha256"))[:12]}, {GO_SCHEMA!r}); '
                         f'rebuild with data_prep/build_misato_go_labels.py')
    labels = json.loads(path.read_text())
    info = {'file': _rel(path), 'sha256': sha, 'schema': GO_SCHEMA,
            'ontology': (prov.get('ontology') or {}).get('data_version'),
            'source_sha256': (prov.get('source') or {}).get('sha256')}
    return {str(k).upper(): list(v) for k, v in labels.items()}, info


# ---------------------------------------------------------------- class selection
def select_go_classes(labels: dict, train_pids, scope: str = 'top50',
                      min_count: int = MIN_TRAIN_COUNT, max_frac: float = MAX_TRAIN_FRAC
                      ) -> tuple[list, dict, dict]:
    """(classes in rank order, {class: training count}, selection info).

    `train_pids`: every training pid of the split. Counts are over training proteins;
    the 50 % ceiling is relative to the training proteins with >= 1 MF label.
    Ties are broken by GO id (rank key (-count, GO id)).
    """
    if scope not in SCOPE_N:
        raise ValueError(f'Unknown scope {scope!r}')
    train = sorted({str(p).upper() for p in train_pids})
    annotated = [p for p in train if labels.get(p)]
    counts = Counter(t for p in annotated for t in set(labels[p]))
    n_ann = len(annotated)
    cand = [(t, c) for t, c in counts.items() if c >= min_count and c < max_frac * n_ann]
    cand.sort(key=lambda tc: (-tc[1], tc[0]))
    n = SCOPE_N[scope]
    chosen = cand if n is None else cand[:n]
    info = {'rule': SELECTION_RULE, 'scope': scope, 'min_train_count': min_count,
            'max_train_frac': max_frac, 'n_train_proteins': len(train),
            'n_train_annotated': n_ann, 'n_candidates': len(cand),
            'n_too_frequent': sum(1 for c in counts.values() if c >= max_frac * n_ann),
            'count_at_last_rank': chosen[-1][1] if chosen else None,
            'n_tied_beyond_last_rank': (sum(1 for _, c in cand[len(chosen):]
                                            if c == chosen[-1][1]) if chosen else 0)}
    return [t for t, _ in chosen], {t: c for t, c in chosen}, info


def build_go_labelset(labels: dict, train_pids, scope: str = 'top50'
                      ) -> tuple[dict, list, dict]:
    """(pid -> set of selected terms, classes sorted by GO id, selection info).

    The classes come from `select_go_classes`; every pid (any split) keeps only its
    selected terms, and pids with none are left out (dropped from the probe). The
    selection info carries the classes in rank order with their training counts and the
    training frequency of each class (count / annotated training proteins), the score
    of the frequency-floor predictor.
    """
    ranked, counts, info = select_go_classes(labels, train_pids, scope)
    cls = set(ranked)
    out = {}
    for pid, terms in labels.items():
        s = {t for t in terms if t in cls}
        if s:
            out[pid] = s
    n_ann = max(info['n_train_annotated'], 1)
    info = {**info, 'classes_ranked': ranked, 'train_counts': counts,
            'train_freq': {t: counts[t] / n_ann for t in ranked}}
    return out, sorted(ranked), info


# ---------------------------------------------------------------- result block
def go_result_fields(label_info: dict, sel_info: dict, classes: list, bucket: dict,
                     split_map: dict, feat_pids, labels: dict) -> dict:
    """GO-specific fields of a result JSON (labels and test proteins identified by hash).

    `bucket`: {'train'|'val'|'test': [{'pid', 'y', ...}]} after dropping; `feat_pids`:
    pids with features of this method. The frequency floor scores every test protein
    with the training frequency of each class (label-only predictor).
    """
    from multilabel_metrics import frequency_floor   # same folder
    feat = {str(p).upper() for p in feat_pids}
    counts = {}
    for s in ('train', 'val', 'test'):
        in_split = [p for p, sp in split_map.items() if sp == s and p in feat]
        kept = len(bucket.get(s, []))
        counts[s] = {'with_features': len(in_split), 'kept': kept,
                     'dropped': len(in_split) - kept,
                     'dropped_no_mf_label': sum(1 for p in in_split if not labels.get(p))}
    freq = np.array([sel_info['train_freq'][c] for c in classes], dtype=np.float64)
    yte = (np.stack([it['y'] for it in bucket['test']]) if bucket.get('test')
           else np.zeros((0, len(classes))))
    test_pids = [it['pid'] for it in bucket.get('test', [])]
    return {'go_labels': label_info,
            'go_classes': list(classes),
            'go_class_train_counts': {c: sel_info['train_counts'][c] for c in classes},
            'go_class_selection': {k: v for k, v in sel_info.items()
                                   if k not in ('classes_ranked', 'train_counts', 'train_freq')},
            'go_classes_sha256': pid_list_sha256(classes),
            'go_split_counts': counts,
            'test_pids_sha256': pid_list_sha256(test_pids),
            'frequency_floor': frequency_floor(freq, yte) if len(test_pids) else None}


# ---------------------------------------------------------------- result checks
_CURRENT = {}


def current_label_sha(path=GO_LABELS_V2) -> str | None:
    """sha256 recorded in the sidecar of the current label file (None if not built)."""
    side = Path(str(path) + '.provenance.json')
    if not side.exists():
        return None
    key = (str(side), side.stat().st_mtime_ns)
    if key not in _CURRENT:
        _CURRENT[key] = json.loads(side.read_text()).get('sha256')
    return _CURRENT[key]


def go_result_problems(d: dict, path=GO_LABELS_V2) -> list:
    """Reasons a GO result JSON must not be compiled ([] = fine).

    A result is accepted only if it records the sha256 of the current v2 label file
    (results built on the old misato_go.json carry no `go_labels` block).
    """
    cur = current_label_sha(path)
    rec = (d or {}).get('go_labels')
    if not isinstance(rec, dict) or not rec.get('sha256'):
        return ['built on the old GO labels (no go_labels block; rerun on misato_go_mf_v2)']
    if cur is None:
        return [f'{_rel(path)} has no sidecar (build the GO labels first)']
    if rec['sha256'] != cur:
        return [f'GO label sha256 {rec["sha256"][:12]} != current {cur[:12]}']
    return []


def go_result_is_current(out_path, path=GO_LABELS_V2) -> bool:
    """True if `out_path` exists and was built on the current GO labels."""
    out_path = Path(out_path)
    if not out_path.exists():
        return False
    try:
        return not go_result_problems(json.loads(out_path.read_text()), path)
    except Exception:
        return False
