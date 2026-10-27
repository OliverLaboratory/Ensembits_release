"""MISATO EC / GO cross-validation file: distinct proteins, grouped label-stratified 5-fold CV.

What it does
------------
Authors' decision 2026-10-19 (option A; camera_ready/decision_brief_20261019.md §2 / §4).
Function labels (EC numbers, GO terms) belong to a protein, but MISATO holds one entry per
protein-ligand complex (GO: 4,126 labelled complexes of ~1,230 proteins; carbonic anhydrase
II alone has 284 complexes). On the fixed v2 splits the large families were forced into
train and the val / test class distributions shifted (GO, structure split: peptidase 31 %
of train, 1 % of val). This builder replaces, for the EC and GO probes only, the fixed
splits by cross-validation over distinct proteins. Per task (GO, EC):

  1. Universe: complexes with a label of the task (GO: >= 1 molecular-function term of
     data/annotations/misato_go_mf_v2.json; EC: >= 1 depth-1 class of
     data/annotations/misato_ec_v2.json under the probes' truncation rule).
  2. Distinct proteins. Key of a complex = the sorted set of the SIFTS UniProt accessions
     of its chains of >= MIN_CHAIN_LEN (20) residues (data/misato_splits_v2.chains.json),
     with one refinement: a chain of a component-resolved polyprotein (UniProt entry whose
     EC numbers sit on its mature products; the 73 accessions the EC label builder resolves,
     `polyproteins[].component_resolved` of the EC label sidecar; HIV / HCV / flavivirus
     polyproteins) is keyed by (accession, mature product) as in the split builder's
     `uniprot_keys`, so an HIV protease and an HIV integrase are two proteins (their EC
     labels differ by construction). A complex without any accession is its own key.
     Representative of a key: the complex with the most resolved residues (misato_real_bb
     sequence length), ties by PDB id. The representative's labels are used; how often the
     labels of the complexes of one key differ is reported (label_disagreement in the
     sidecar: GO ~2 % of proteins, EC < 1 %; the union is not used).
  3. Groups (leak-free, as the v2 splits, applied to the representatives;
     build_misato_splits_v2 functions): mode "sequence" = union-find over the graph-rule
     sequence clusters of the chains (groups_by_shared_keys); mode "structure" = CATH H
     union-find with the complexes without CATH attached through their sequence clusters
     (structure_groups).
  4. Always train (coordinator, 2026-10-19): a group larger than one fold's target (N / 5
     representatives) is never cycled into val / test; it is in train in every rotation
     (as the v2 splits' forced-to-train rule; EC structure: one CATH H component of 287 of
     944 proteins). Folds (5 per mode): the other whole groups are assigned by multi-label iterative stratification
     at the group level (Sechidis et al. 2011, adapted to groups): labels are processed
     from the rarest (fewest positive representatives among the unassigned groups); the
     groups carrying that label, largest first (ties by a RandomState(42) permutation),
     each go to the fold with the largest remaining demand for that label, ties to the
     largest remaining size demand, then a seeded fold order; only folds that stay within
     1.1 x N / 5 proteins after taking the group are eligible (SIZE_TOL; if none is, the
     fold with the largest remaining size demand). Without the cap the GO structure folds
     ranged 15-26 % of the proteins; with it 17-22 % (class balance slightly worse; the
     comparison is in the sidecar, `size_tol_comparison`).
     Stratification labels: GO = the propagated MF terms carried by >= MIN_STRAT_COUNT (10)
     representatives (the terms that can become a class); EC = the depth-1 classes.
  5. Rotations r = 0..4: test = fold r, val = fold (r + 1) mod 5, train = the other three
     folds + the always-train groups.
     Classes of each rotation, from its training folds only: GO = go_labels.build_go_labelset
     ('top50': MF terms in >= 10 and < 50 % of the training proteins with an MF label,
     ranked by (-count, GO id)); EC = every class of the labelled training proteins, per
     depth 1-3 (truncate_ec rule). The probes recompute them and must match.
  6. Report (sidecar): representatives and label disagreement, per mode the group sizes,
     per fold n proteins, largest group share, per-class prevalence against the whole set
     (max absolute deviation, stratification labels and the class lists of the probes),
     and the cross-fold leakage (build_misato_splits_v2.leakage with fold r as "test" and
     the other folds as "train": chains >= 30 / 50 / 95 % identical, identical chains,
     shared UniProt, shared CATH H), from the split builder's all-vs-all search.

Paper
-----
`appendix.tex` \\label{app:downstreams} (EC / GO probes), \\label{app:experiments}. The
protocol is not described in the paper yet (Open, audit D.6: EC/GO CV text; text pass item).

Origin
------
New (2026-10-19, authors' decision option A). Reuses the union-find, grouping and leakage
functions of data_prep/build_misato_splits_v2.py and the label / class rules of
downstream/go_labels.py, downstream/ec_labels.py and tokenize_misato_and_eval.build_label_sets.

Used by the reported model: yes (EC and GO rows of every table).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/misato_splits_v2.json (+ .chains.json, + .mmseqs/{chains.fa,allvsall.m8})
          data/cached_descriptors/misato_real_bb/<PID>.npz           (seq: resolved residues)
          data/annotations/misato_go_mf_v2.json, misato_ec_v2.json (+ sidecars)
  writes  data/misato_function_cv_v1.json (+ .provenance.json: rule, inputs with sha256,
          representatives, label disagreement, group / fold statistics, leakage, output
          sha256). Refuses to overwrite without --overwrite; atomic writes.

How to run
----------
    python data_prep/build_misato_function_cv.py [--overwrite] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))
sys.path.insert(0, str(CODE_ROOT / 'data_prep'))
from ensembits.paths import REPO_ROOT, MISATO_SPLITS, MISATO_FUNCTION_CV  # noqa: E402
from ensembits.featurize import file_sha256  # noqa: E402
from ensembits.data_versions import write_provenance, read_provenance  # noqa: E402

SCHEMA = 'misato-function-cv-v1'
SEED = 42
N_FOLDS = 5
TASKS = ('go', 'ec')
MODES = ('sequence', 'structure')
SPLITS = ('train', 'val', 'test')
MIN_CHAIN_LEN = 20            # chains shorter than this (peptides) do not enter the key
MIN_STRAT_COUNT = 10          # GO terms of >= 10 representatives are stratified
SIZE_TOL = 0.10               # a fold may exceed its N / 5 target by <= 10 % (unless no fold
                              # can take a group); 0 = equal sizes, inf = plain stratification
EC_DEPTHS = (1, 2, 3)
GO_SCOPE = 'top50'
RULE = {
    'universe': {'go': 'complexes with >= 1 MF term (misato_go_mf_v2.json)',
                 'ec': 'complexes with >= 1 depth-1 EC class (misato_ec_v2.json, truncate_ec)'},
    'protein_key': ('sorted set of the SIFTS UniProt accessions of the chains with >= '
                    f'{MIN_CHAIN_LEN} residues (misato_splits_v2.chains.json); chains of a '
                    'component-resolved polyprotein (EC label sidecar) keyed by (accession, '
                    'mature product); no accession: the complex itself'),
    'representative': 'most resolved residues (misato_real_bb seq length), ties by PDB id',
    'labels': "the representative's labels (disagreement within a key reported, not merged)",
    'groups': {'sequence': 'union-find over shared graph-rule sequence clusters '
                           '(build_misato_splits_v2.groups_by_shared_keys)',
               'structure': 'union-find over shared CATH H codes, complexes without CATH '
                            'attached through their sequence clusters '
                            '(build_misato_splits_v2.structure_groups)'},
    'always_train': (f'groups larger than N / {N_FOLDS} representatives are in train in '
                     'every rotation (never val / test), as the v2 forced-to-train rule'),
    'folds': (f'{N_FOLDS} folds of the other whole groups; multi-label iterative stratification at the '
              'group level (rarest label first; its groups largest first, ties by a '
              f'RandomState({SEED}) permutation; among the folds that stay within '
              f'{1 + SIZE_TOL:.2f} x N / {N_FOLDS} proteins after taking the group, the fold with '
              'the largest remaining demand for the label, then the largest remaining size '
              'demand, then a seeded fold order; if no fold can take the group, the fold with '
              'the largest remaining size demand)'),
    'stratification_labels': {'go': f'propagated MF terms of >= {MIN_STRAT_COUNT} '
                                    'representatives',
                              'ec': 'depth-1 EC classes'},
    'rotations': ('r = 0..4: test = fold r, val = fold (r+1) mod 5, train = the other three '
                  'folds + the always-train groups'),
    'classes': {'go': 'go_labels.build_go_labelset(scope top50) on the training proteins of '
                      'the rotation (>= 10 and < 50 % of the training proteins with an MF '
                      'label; ranked by (-count, GO id); first 50)',
                'ec': 'every class of the labelled training proteins of the rotation, per '
                      'depth 1-3 (truncate_ec)'},
}


# ---------------------------------------------------------------- proteins
def protein_key(chains: list, polyproteins: set, pid: str,
                min_len: int = MIN_CHAIN_LEN) -> tuple:
    """Key of one complex: sorted accessions (polyprotein chains: 'ACC|product') of its
    chains of >= min_len residues; ('pid:<PID>',) when there is none.

    chains: the complex's chain records of misato_splits_v2.chains.json ('length',
    'uniprot', 'uniprot_keys')."""
    ks = set()
    for c in chains:
        if int(c['length']) < min_len:
            continue
        for acc in c.get('uniprot') or []:
            if acc in polyproteins:
                prods = [k for k in c.get('uniprot_keys') or []
                         if isinstance(k, (list, tuple)) and k and k[0] == acc]
                if prods:
                    ks.update(f'{k[0]}|{k[1]}' for k in prods)
                    continue
            ks.add(acc)
    return tuple(sorted(ks)) if ks else (f'pid:{pid}',)


def choose_representatives(pids, keys: dict, n_res: dict) -> dict:
    """{representative pid: sorted member pids}: per key the complex with the most residues
    (ties: smallest PDB id)."""
    by = defaultdict(list)
    for p in sorted(pids):
        by[keys[p]].append(p)
    out = {}
    for members in by.values():
        rep = sorted(members, key=lambda p: (-int(n_res[p]), p))[0]
        out[rep] = sorted(members)
    return dict(sorted(out.items()))


def label_disagreement(reps: dict, labels: dict) -> dict:
    """How often the complexes of one protein carry different label sets."""
    multi = {r: m for r, m in reps.items() if len(m) > 1}
    differ = {r: m for r, m in multi.items() if len({frozenset(labels[p]) for p in m}) > 1}
    n_nonrep = sum(len(m) - 1 for m in multi.values())
    n_nonrep_diff = sum(frozenset(labels[p]) != frozenset(labels[r])
                        for r, m in multi.items() for p in m if p != r)
    rep_ne_union = [r for r, m in multi.items()
                    if set(labels[r]) != set().union(*(set(labels[p]) for p in m))]
    return {'n_proteins': len(reps), 'n_proteins_with_several_complexes': len(multi),
            'n_proteins_with_differing_labels': len(differ),
            'frac_proteins_with_differing_labels': round(len(differ) / max(len(reps), 1), 4),
            'n_complexes_in_those_proteins': sum(len(m) for m in differ.values()),
            'n_nonrep_complexes': n_nonrep,
            'n_nonrep_complexes_differing_from_rep': n_nonrep_diff,
            'frac_nonrep_complexes_differing_from_rep': round(n_nonrep_diff / max(n_nonrep, 1), 4),
            'n_proteins_rep_labels_not_union': len(rep_ne_union),
            'examples': [{'rep': r, 'n_complexes': len(m),
                          'n_label_sets': len({frozenset(labels[p]) for p in m})}
                         for r, m in sorted(differ.items(), key=lambda kv: -len(kv[1]))[:30]]}


# ---------------------------------------------------------------- stratified folds
def stratified_group_folds(groups: list, pid_labels: dict, strat_labels: list,
                           k: int = N_FOLDS, seed: int = SEED,
                           size_tol: float = SIZE_TOL) -> tuple[list, dict]:
    """Assign whole groups to k folds by multi-label iterative stratification.

    groups: lists of pids; pid_labels: {pid: iterable of labels}; strat_labels: the labels
    to balance. A group goes only to a fold that stays within (1 + size_tol) x N / k
    proteins after taking it (if no fold can, the fold with the largest remaining size
    demand). Returns (folds: k sorted pid lists, info). Deterministic: groups are
    canonicalised (sorted, by first pid) before the seeded permutation.
    """
    groups = sorted((sorted(g) for g in groups), key=lambda g: g[0])
    lab_idx = {l: i for i, l in enumerate(strat_labels)}
    G, L = len(groups), len(strat_labels)
    size = np.array([len(g) for g in groups], dtype=np.float64)
    Y = np.zeros((G, L), dtype=np.float64)
    for gi, g in enumerate(groups):
        for p in g:
            for l in set(pid_labels.get(p, ())):
                if l in lab_idx:
                    Y[gi, lab_idx[l]] += 1
    rng = np.random.RandomState(seed)
    tie = np.empty(G, dtype=np.int64)
    tie[rng.permutation(G)] = np.arange(G)           # random rank of each group
    fold_rank = rng.permutation(k)                    # seeded fold order for the last tie
    need_size = np.full(k, size.sum() / k)
    need_lab = np.tile(Y.sum(0) / k, (k, 1))         # (k, L)
    fold_of = np.full(G, -1, dtype=np.int64)
    order_log = []

    cap = size.sum() / k

    def place(gi, li):
        # Size cap: only folds that stay within (1 + size_tol) x the target size after
        # taking the group; if none can, the fold with the largest remaining size demand.
        fits = [f for f in range(k) if need_size[f] - size[gi] >= -size_tol * cap]
        if not fits or li is None:
            cand = fits or list(range(k))
            f = max(cand, key=lambda j: (need_size[j], -fold_rank[j]))
        else:
            f = max(fits, key=lambda j: (need_lab[j, li], need_size[j], -fold_rank[j]))
        fold_of[gi] = f
        need_size[f] -= size[gi]
        need_lab[f] -= Y[gi]

    while True:
        un = fold_of < 0
        if not un.any():
            break
        rem = Y[un].sum(0)
        if rem.sum() == 0:                            # groups without a stratification label
            for gi in sorted(np.where(un)[0], key=lambda i: (-size[i], tie[i])):
                place(gi, None)
            break
        li = min((i for i in range(L) if rem[i] > 0), key=lambda i: (rem[i], i))
        cand = [gi for gi in np.where(un)[0] if Y[gi, li] > 0]
        for gi in sorted(cand, key=lambda i: (-size[i], tie[i])):
            place(gi, li)
        order_log.append(strat_labels[li])
    folds = [sorted(p for gi in range(G) if fold_of[gi] == f for p in groups[gi])
             for f in range(k)]
    info = {'n_groups': G, 'n_strat_labels': L, 'n_label_steps': len(order_log),
            'first_label_steps': order_log[:10]}
    return folds, info


def check_folds(folds: list, groups: list, universe) -> None:
    """Every pid of the universe in exactly one fold; no group split across folds."""
    where = {}
    for f, ps in enumerate(folds):
        for p in ps:
            if p in where:
                raise ValueError(f'{p} in folds {where[p]} and {f}')
            where[p] = f
    if set(where) != set(universe):
        raise ValueError('folds and universe differ')
    for g in groups:
        if len({where[p] for p in g}) != 1:
            raise ValueError(f'group {g[:5]} spans several folds')


def split_always_train(groups: list, n_total: int, k: int = N_FOLDS) -> tuple[list, list]:
    """(groups larger than one fold's target n_total / k, the other groups); both sorted.

    FIXED(coordinator, 2026-10-19): a group larger than a fold's target is never cycled
    into val / test (it would make a whole fold, or more, one family); it is in train in
    every rotation, as the v2 splits' forced-to-train rule."""
    groups = sorted((sorted(g) for g in groups), key=lambda g: g[0])
    cap = n_total / k
    big = sorted((g for g in groups if len(g) > cap), key=lambda g: (-len(g), g[0]))
    return big, [g for g in groups if len(g) <= cap]


def rotations_of(folds: list, always_train=()) -> list:
    """[{test_fold, val_fold, train_folds, train, val, test}] for r = 0..k-1; the
    always-train proteins are added to train in every rotation."""
    k = len(folds)
    out = []
    for r in range(k):
        v = (r + 1) % k
        tr = [f for f in range(k) if f not in (r, v)]
        out.append({'test_fold': r, 'val_fold': v, 'train_folds': tr,
                    'train': sorted([p for f in tr for p in folds[f]] + list(always_train)),
                    'val': list(folds[v]), 'test': list(folds[r])})
    return out


def group_summary(g: list, table: dict, lab: dict, top: int = 3) -> dict:
    """Size, most common UniProt accessions and CATH H codes, and labels of a group."""
    acc = Counter(u for p in g for c in table[p]['chains'] for u in set(c['uniprot']))
    hh = Counter(h for p in g for h in table[p]['cath_h'])
    lb = Counter(c for p in g for c in lab.get(p, ()))
    return {'size': len(g), 'first_pid': g[0],
            'top_accessions': [[a, n] for a, n in acc.most_common(top)],
            'top_cath_h': [[a, n] for a, n in hh.most_common(top)],
            'top_labels': [[a, n] for a, n in lb.most_common(top)]}


def prevalence_report(folds: list, pid_labels: dict, classes: list) -> dict:
    """Per fold: prevalence of each class vs the whole set; max |deviation| over classes."""
    allp = [p for f in folds for p in f]
    n = len(allp)
    overall = {c: sum(c in pid_labels.get(p, ()) for p in allp) / n for c in classes}
    out = {'overall_prevalence': {c: round(v, 4) for c, v in overall.items()}, 'folds': []}
    for f, ps in enumerate(folds):
        m = max(len(ps), 1)
        prev = {c: sum(c in pid_labels.get(p, ()) for p in ps) / m for c in classes}
        dev = {c: prev[c] - overall[c] for c in classes}
        worst = max(classes, key=lambda c: abs(dev[c])) if classes else None
        out['folds'].append({'fold': f, 'max_abs_deviation': round(abs(dev[worst]), 4)
                             if worst else 0.0, 'worst_class': worst,
                             'worst_class_prevalence': round(prev[worst], 4) if worst else None,
                             'worst_class_overall': round(overall[worst], 4) if worst else None,
                             'mean_abs_deviation': round(float(np.mean([abs(v) for v in dev.values()]))
                                                         if dev else 0.0, 4),
                             'n_classes_absent': sum(prev[c] == 0 for c in classes)})
    out['max_abs_deviation'] = max((r['max_abs_deviation'] for r in out['folds']), default=0.0)
    out['mean_abs_deviation'] = round(float(np.mean([r['mean_abs_deviation']
                                                     for r in out['folds']])), 4)
    return out


def fold_sizes_report(folds: list, groups: list) -> dict:
    gid = {p: i for i, g in enumerate(groups) for p in g}
    n = sum(len(f) for f in folds)
    rows = []
    for f, ps in enumerate(folds):
        gs = Counter(gid[p] for p in ps)
        big = max(gs.values()) if gs else 0
        rows.append({'fold': f, 'n_proteins': len(ps), 'frac': round(len(ps) / max(n, 1), 4),
                     'n_groups': len(gs), 'largest_group': big,
                     'largest_group_share': round(big / max(len(ps), 1), 4)})
    sizes = sorted((len(g) for g in groups), reverse=True)
    return {'n_proteins': n, 'n_groups': len(groups), 'largest_groups': sizes[:10],
            'largest_group_frac_of_all': round(sizes[0] / max(n, 1), 4) if sizes else 0.0,
            'n_singletons': sum(s == 1 for s in sizes), 'folds': rows}


# ---------------------------------------------------------------- inputs
def read_chain_table(path: Path) -> dict:
    """{PID: chains.json record} (upper-case PIDs)."""
    d = json.loads(Path(path).read_text())
    if d.get('schema') != 'misato-splits-v2-chains':
        raise SystemExit(f'{path}: schema {d.get("schema")!r}')
    return {k.upper(): v for k, v in d['complexes'].items()}


def read_sequences(pids, real_bb_dir: Path) -> dict:
    """{PID: misato_real_bb seq}."""
    out = {}
    for p in sorted(pids):
        with np.load(Path(real_bb_dir) / f'{p}.npz', allow_pickle=False) as z:
            out[p] = str(z['seq'])
    return out


def chains_with_seq(table: dict, seqs: dict) -> dict:
    """{PID: [{'seq', ...}]} for build_misato_splits_v2.leakage (chain pieces of the table)."""
    out = {}
    for p, rec in table.items():
        s = seqs[p]
        tot = sum(int(c['length']) for c in rec['chains'])
        if tot != len(s):
            raise ValueError(f'{p}: chain table lengths {tot} != misato_real_bb {len(s)}')
        out[p] = [{'seq': s[int(c['start']):int(c['end'])]} for c in rec['chains']]
    return out


def read_mmseqs(workdir: Path) -> tuple[dict, list]:
    """(seq -> id, hits) of the split builder's all-vs-all search (chains.fa, allvsall.m8)."""
    seq_ids, name = {}, None
    for line in open(Path(workdir) / 'chains.fa'):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]
        elif line:
            seq_ids[line] = name
    hits = []
    for line in open(Path(workdir) / 'allvsall.m8'):
        q, t, fid, aln, ql, tl, qc, tc, _ = line.rstrip('\n').split('\t')
        if q != t:
            hits.append((q, t, float(fid), int(aln), int(ql), int(tl), float(qc), float(tc)))
    return seq_ids, hits


def component_resolved_polyproteins(ec_labels_path: Path) -> list:
    prov = read_provenance(ec_labels_path) or {}
    return sorted({r['accession'] for r in prov.get('polyproteins', [])
                   if r.get('component_resolved')})


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _atomic_json(path: Path, obj, indent=None) -> None:
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=indent))
    os.replace(tmp, path)


# ---------------------------------------------------------------- build
def build(table: dict, n_res: dict, go: dict, ec: dict, polyproteins: set,
          seed: int = SEED, leak=None) -> tuple[dict, dict]:
    """(CV body, report). go: {PID: [MF terms]}; ec: {PID: [EC numbers]}; leak: optional
    callable(folds, mode) -> leakage report."""
    from build_misato_splits_v2 import groups_by_shared_keys, structure_groups
    from go_labels import build_go_labelset
    from tokenize_misato_and_eval import build_label_sets
    keys = {p: protein_key(rec['chains'], polyproteins, p) for p, rec in table.items()}
    ec_d = {d: {p.upper(): set(v) for p, v in build_label_sets(ec, d).items()}
            for d in EC_DEPTHS}
    task_labels = {'go': {p: set(v) for p, v in go.items() if v and p in table},
                   'ec': {p: v for p, v in ec_d[1].items() if p in table}}
    body = {'tasks': {}}
    report = {'tasks': {}}
    for task in TASKS:
        lab = task_labels[task]
        reps = choose_representatives(list(lab), keys, n_res)
        R = sorted(reps)
        dis = label_disagreement(reps, lab)
        if task == 'ec':
            dis['by_depth'] = {str(d): label_disagreement(
                {r: [p for p in m if p in ec_d[d]] for r, m in reps.items() if r in ec_d[d]},
                ec_d[d]) for d in (2, 3)}
            dis['n_rep_without_depth_label_but_member_with'] = {
                str(d): sum(r not in ec_d[d] and any(p in ec_d[d] for p in m)
                            for r, m in reps.items()) for d in (2, 3)}
        if task == 'go':
            cnt = Counter(t for r in R for t in lab[r])
            strat = sorted(t for t, c in cnt.items() if c >= MIN_STRAT_COUNT)
            _, cls_all, _ = build_go_labelset({r: sorted(lab[r]) for r in R}, R, GO_SCOPE)
            eval_classes = {'go_top50_whole_set': cls_all}
        else:
            strat = sorted({c for r in R for c in lab[r]})
            eval_classes = {f'ec_d{d}': sorted({c for r in R for c in ec_d[d].get(r, ())})
                            for d in EC_DEPTHS}
        cc = {r: list(table[r]['clusters']) for r in R}
        hh = {r: list(table[r]['cath_h']) for r in R}
        groups = {'sequence': groups_by_shared_keys(cc)}
        groups['structure'], nocath = structure_groups(hh, cc)
        tb = {'n_complexes_labelled': len(lab), 'n_proteins': len(R),
              'representatives': {r: {'key': list(keys[r]), 'members': reps[r],
                                      'n_residues': int(n_res[r])} for r in R},
              'labels': RULE['labels'], 'stratification_labels': strat, 'modes': {}}
        tr = {'label_disagreement': dis, 'n_stratification_labels': len(strat), 'modes': {}}
        for mode in MODES:
            g_all = groups[mode]
            big, g = split_always_train(g_all, len(R), N_FOLDS)   # FIXED(coordinator, 2026-10-19)
            always = sorted(p for gg in big for p in gg)
            in_folds = sorted(set(R) - set(always))
            folds, sinfo = stratified_group_folds(g, {r: lab[r] for r in in_folds}, strat,
                                                  N_FOLDS, seed)
            check_folds(folds, g, in_folds)
            rots = rotations_of(folds, always)
            for rot in rots:
                if task == 'go':
                    _, cls, sel = build_go_labelset({r: sorted(lab[r]) for r in R},
                                                    rot['train'], GO_SCOPE)
                    rot['classes'] = cls
                    rot['class_selection'] = {k: v for k, v in sel.items()
                                              if k in ('n_train_proteins', 'n_train_annotated',
                                                       'n_candidates', 'count_at_last_rank')}
                    rot['n_with_class'] = {s: sum(bool(set(lab[p]) & set(cls)) for p in rot[s])
                                           for s in SPLITS}
                else:
                    rot['classes'] = {str(d): sorted({c for p in rot['train']
                                                      for c in ec_d[d].get(p, ())})
                                      for d in EC_DEPTHS}
                    rot['n_labelled'] = {str(d): {s: sum(p in ec_d[d] for p in rot[s])
                                                  for s in SPLITS} for d in EC_DEPTHS}
            tb['modes'][mode] = {'groups': g_all, 'always_train': always,
                                 'folds': folds, 'rotations': rots}
            mrep = {'always_train': {
                        'rule': 'groups larger than N / 5 representatives (N / 5 = '
                                f'{len(R) / N_FOLDS:.1f}): train in every rotation',
                        'n_proteins': len(always),
                        'groups': [group_summary(gg, table, lab if task == 'go' else ec_d[1])
                                   for gg in big]},
                    'stratification': sinfo, 'sizes': fold_sizes_report(folds, g),
                    'prevalence_stratification_labels':
                        {k: v for k, v in prevalence_report(folds, lab, strat).items()
                         if k != 'overall_prevalence'},
                    'prevalence': {name: prevalence_report(
                        folds, lab if task == 'go' else ec_d[int(name[-1])], cls)
                        for name, cls in eval_classes.items()}}
            if mode == 'structure':
                mrep['no_cath'] = nocath
            # The same stratification without the size cap (report only).
            free, _ = stratified_group_folds(g, {r: lab[r] for r in in_folds}, strat, N_FOLDS,
                                             seed, size_tol=float('inf'))
            mrep['size_tol_comparison'] = {
                'size_tol': SIZE_TOL,
                'capped': {'fold_sizes': [len(f) for f in folds],
                           **{name: {k2: pr[k2] for k2 in ('max_abs_deviation',
                                                           'mean_abs_deviation')}
                              for name, pr in mrep['prevalence'].items()}},
                'uncapped': {'fold_sizes': [len(f) for f in free],
                             **{name: {k2: v[k2] for k2 in ('max_abs_deviation',
                                                            'mean_abs_deviation')}
                                for name, v in ((name, prevalence_report(
                                    free, lab if task == 'go' else ec_d[int(name[-1])], cls))
                                    for name, cls in eval_classes.items())}}}
            if task == 'go':
                mrep['rotation_classes'] = [{'test_fold': r['test_fold'],
                                             'n_classes': len(r['classes']),
                                             'n_with_class': r['n_with_class']} for r in rots]
                common = set.intersection(*(set(r['classes']) for r in rots))
                mrep['n_classes_common_to_all_rotations'] = len(common)
            else:
                mrep['rotation_classes'] = [{'test_fold': r['test_fold'],
                                             'n_classes': {d: len(c) for d, c in r['classes'].items()},
                                             'classes_missing_from_train': {
                                                 d: sorted(set(eval_classes[f'ec_d{d}']) - set(c))
                                                 for d, c in r['classes'].items()},
                                             'n_labelled': r['n_labelled']} for r in rots]
            if leak is not None:
                mrep['leakage'] = leak(folds, mode, always)
            tr['modes'][mode] = mrep
        body['tasks'][task] = tb
        report['tasks'][task] = tr
    return body, report


def cross_fold_leakage(folds: list, chains: dict, seq_ids: dict, hits: list,
                       complex_h: dict, complex_acc: dict, always_train=()) -> list:
    """Per fold r: build_misato_splits_v2.leakage with fold r as 'test' and the other folds
    plus the always-train proteins as 'train' (val empty)."""
    from build_misato_splits_v2 import leakage
    out = []
    for r, ps in enumerate(folds):
        split = {'train': sorted([p for f, q in enumerate(folds) if f != r for p in q]
                                 + list(always_train)),
                 'val': [], 'test': list(ps)}
        sub = {p: chains[p] for p in split['train'] + split['test']}
        rep = leakage(split, sub, seq_ids, hits, complex_h, complex_acc)['test']
        out.append({'fold': r, **rep})
    return out


def leakage_summary(rows: list) -> dict:
    """Max over folds of the main leakage counts."""
    def mx(f, t):
        return max(r['any_chain'][f][f'ge{t}'] for r in rows)
    return {'max_over_folds': {
        **{f'{f}_ge{t}': mx(f, t) for f in ('nocov_aln50', 'cov05_short', 'cov08_both')
           for t in (30, 50, 95)},
        'identical_chain': max(r['identical_chain_in_train'] for r in rows),
        'shares_uniprot': max(r['shares_uniprot_with_train'] for r in rows),
        'shares_cath_h': max(r['shares_cath_h_with_train'] for r in rows)}}


def main(argv=None) -> int:
    from rmsf_common import MISATO_REAL_BB_DIR
    from go_labels import GO_LABELS_V2, load_go_labels
    from ec_labels import EC_LABELS, load_ec_labels
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=str(MISATO_FUNCTION_CV))
    ap.add_argument('--splits', default=str(MISATO_SPLITS),
                    help='the v2 split file (its .chains.json and .mmseqs/ are read)')
    ap.add_argument('--real-bb-dir', default=str(MISATO_REAL_BB_DIR))
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--no-leakage', action='store_true', help='skip the leakage report (tests)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        print(f'{out} exists; pass --overwrite to rebuild'); return 1
    splits = Path(a.splits)
    chains_p = splits.with_name(splits.stem + '.chains.json')
    work = splits.with_name(splits.stem + '.mmseqs')
    table = read_chain_table(chains_p)
    seqs = read_sequences(table, Path(a.real_bb_dir))
    n_res = {p: len(s) for p, s in seqs.items()}
    go, go_info = load_go_labels(GO_LABELS_V2)
    ec_raw, ec_rec = load_ec_labels(EC_LABELS)
    ec = {str(k).upper(): v for k, v in ec_raw.items()}
    poly = set(component_resolved_polyproteins(Path(EC_LABELS)))
    leak = None                                       # (--no-leakage: no report)
    if not a.no_leakage:
        chains = chains_with_seq(table, seqs)
        seq_ids, hits = read_mmseqs(work)
        complex_h = {p: rec['cath_h'] for p, rec in table.items()}
        complex_acc = {p: sorted({u for c in rec['chains'] for u in c['uniprot']})
                       for p, rec in table.items()}

        def leak_report(folds, mode, always=()):
            rows = cross_fold_leakage(folds, chains, seq_ids, hits, complex_h, complex_acc,
                                      always)
            return {'per_fold': rows, **leakage_summary(rows)}
        leak = leak_report
    body, report = build(table, n_res, go, ec, poly, a.seed, leak)
    import hashlib
    h = hashlib.sha256()
    for p in sorted(n_res):
        h.update(f'{p}\t{n_res[p]}\n'.encode())
    inputs = {'go_labels': {'path': go_info['file'], 'sha256': go_info['sha256']},
              'ec_labels': {'path': ec_rec['path'], 'sha256': ec_rec['sha256']},
              'split_chains': {'path': _rel(chains_p), 'sha256': file_sha256(chains_p)},
              'splits': {'path': _rel(splits), 'sha256': file_sha256(splits)}}
    result = {'schema': SCHEMA, 'seed': a.seed, 'n_folds': N_FOLDS, 'rule': RULE,
              'inputs': inputs, **body}
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(out, result, indent=1)
    sha = file_sha256(out)
    prov = {'builder': 'data_prep/build_misato_function_cv.py', 'schema': SCHEMA,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'seed': a.seed,
            'decision': ("authors' decision 2026-10-19 (option A): EC / GO on distinct "
                         'proteins with grouped, label-stratified 5-fold CV; binding site, '
                         'affinity and RMSF keep the fixed v2 splits'),
            'rule': RULE, 'inputs': {**inputs,
                                     'mmseqs': {f.name: file_sha256(f) for f in
                                                sorted(work.glob('*')) if f.is_file()}
                                     if not a.no_leakage else None,
                                     'misato_real_bb_lengths_sha256': h.hexdigest(),
                                     'n_component_resolved_polyproteins': len(poly)},
            'report': report, 'output_sha256': sha,
            'runtime_s': round(time.time() - t0, 1)}
    tmp = out.with_name(f'.{out.name}.prov.tmp{os.getpid()}')
    write_provenance(tmp, prov)
    os.replace(Path(str(tmp) + '.provenance.json'), Path(str(out) + '.provenance.json'))
    for task in TASKS:
        tr = report['tasks'][task]
        d = tr['label_disagreement']
        print(f'[{task}] {body["tasks"][task]["n_complexes_labelled"]} complexes -> '
              f'{d["n_proteins"]} proteins; differing labels in '
              f'{d["n_proteins_with_differing_labels"]} proteins '
              f'({d["n_nonrep_complexes_differing_from_rep"]}/{d["n_nonrep_complexes"]} '
              f'non-representative complexes)')
        for mode in MODES:
            m = tr['modes'][mode]
            sz = m['sizes']
            at = m['always_train']
            print(f'  {mode}: always train {at["n_proteins"]} proteins in '
                  f'{[x["size"] for x in at["groups"]]}')
            print(f'  {mode}: groups {sz["n_groups"]} largest {sz["largest_groups"][:3]} '
                  f'folds {[r["n_proteins"] for r in sz["folds"]]} '
                  f'largest-group share {[r["largest_group_share"] for r in sz["folds"]]}')
            for name, pr in m['prevalence'].items():
                print(f'    prevalence {name}: max |dev| {pr["max_abs_deviation"]}')
            print(f'    stratification labels: max |dev| '
                  f'{m["prevalence_stratification_labels"]["max_abs_deviation"]}')
            if 'leakage' in m:
                print(f'    leakage (max over folds): {m["leakage"]["max_over_folds"]}')
    print(f'wrote {out} (sha256 {sha[:12]}, + provenance) in {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
