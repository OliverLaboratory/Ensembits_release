"""MISATO probe splits v2: sequence split at 30 % identity, structure split by CATH H.

What it does
------------
Rebuilds the `sequence` and `structure` splits of the MISATO probes (EC, GO, binding
site, affinity, MISATO RMSF) after the leakage audit of the old file
`data/misato_splits.json` (camera_ready/benchmark_audit.md D3: the old sequence split
grouped complexes only by a shared UniProt accession; the old structure split used the
alphabetically first CATH H code of each PDB entry and made every entry without CATH a
free singleton). Authors' decision D3 (2026-10-18). The `random` and `ligand_disjoint`
splits, the complex universe (the 4,894 complexes of the old `random` split) and the
target fractions (80 / 10 / 10 of the complexes) are kept; the old file is read only.

  1. Chains. Every complex is cut into its true deposited chains with the MISATO
     chain-break table (`data/cached_descriptors/misato_chain_breaks.json`, schema v2,
     rule 2.3; `data_prep/baselines/_chain_pieces.py`: cuts at `boundary` / `unresolved`
     breaks, a chain stays whole across a missing loop). The sequence of a chain is the
     simulated MISATO sequence of that piece (`misato_real_bb` `seq`). Its deposited
     author chain id comes from aligning the pieces to the deposited mmCIF in order
     (`build_misato_chain_breaks.assign_segments`: continuation, else the next unused copy
     in file order, so homo-oligomer copies get distinct ids), else from the boundary
     records of the table (`left_chain` / `right_chain`; these name the first matching
     copy of a homomer, e.g. 4KIU, 1JCX).
  2. Sequence clusters at 30 % identity (rule `graph`, 2026-10-19, after the split
     re-audit). The distinct chain sequences of >= MIN_CHAIN_LEN (20) residues are searched
     all against all with MMseqs2 without any coverage or identity prefilter
     (`easy-search --exhaustive-search 1 -s 7.5 -e 1e-3 --alignment-mode 3`). Two chains
     are linked when
       (a) identity >= 30 % over an alignment of >= 30 residues covering >= 50 % of the
           SHORTER chain (a domain against the full-length protein, circular permutants,
           a domain shared inside two multi-domain chains), or
       (b) identity >= 95 % over >= 50 aligned residues (same protein, any coverage), or
       (c) they map to the same UniProt accession (SIFTS `pdb_chain_uniprot`); for a
           polyprotein accession (UniProt entry with >= 2 Chain features or a Contains
           section; `build_misato_ec_labels.entry_products` on the EC agent's UniProt
           entries `data/external/uniprot/misato_sifts_entries.json.gz`) only chains that
           map to the same mature product (largest overlap of the SIFTS UniProt range with
           a product, Jaccard) are linked; a chain outside every product is keyed to the
           precursor region of the accession.
     Clusters = connected components of these links (single linkage). Rule `cov08` (the
     first v2 file of 2026-10-18): (a) with >= 80 % coverage of BOTH chains only, no (b),
     no (c); the re-audit found it hides same-protein leaks (circular permutants, domain
     vs full length, domains inside multi-domain chains); its group sizes and leakage are
     kept in the sidecar (`alternative_cluster_rule`). Shorter pieces are peptides (ligand
     peptides, short inhibitor chains); they are left out of the clustering (a complex
     made only of such pieces would be its own group; counted in the sidecar).
  3. Sequence split. Union-find over complexes: two complexes are in one group when any
     chain of one and any chain of the other share a cluster (a complex with chains in
     clusters A and B joins both).
  4. Structure split. CATH H superfamilies of every chain: the SIFTS
     `pdb_chain_cath_uniprot` table (PDB, author chain -> CATH domain ids) with the domain ->
     C.A.T.H code of the CATH-B daily release (`cath-b-newest-all`), plus the CATH-B domains
     named after that PDB entry + chain directly. Union-find over complexes sharing ANY H
     code (all chains, all domains). Complexes without any CATH code are attached through
     their sequence clusters (step 2): every sequence cluster that holds a complex without
     CATH joins all complexes of that cluster (labelled or not); complexes with CATH are
     not joined to each other through sequence clusters. A complex without CATH whose
     clusters hold no other complex stays a singleton; the counts are in the sidecar.
  5. Assignment (both splits). Groups are ordered by their smallest PDB id and shuffled
     with numpy RandomState(42). A group larger than MAX_EVAL_GROUP_FRAC (10 %) of the
     val / test target goes to train (largest first; such a group would dominate val or
     test); every other group, in shuffled order, goes to the split with the largest
     relative deficit (target - count) / target (ties: train, val, test).
  6. Leakage report (sidecar `report_new`, `report_old`, `report_compare` for --compare
     files): from the same all-vs-all search, per split the val / test complexes with any
     chain >= 30 / 50 / 95 % identical to a chain of a train complex under three coverage
     filters (`nocov_aln50`: any coverage, >= 50 aligned residues; `cov05_short`: >= 50 %
     of the shorter chain; `cov08_both`: >= 80 % of both chains), with an identical chain
     sequence in train, sharing a UniProt accession with train (SIFTS), and sharing a CATH
     H code with train; the class balance of each probe task (EC depth 1, GO top-50 of the
     train split, binding site, affinity; label files only, no probe is run; classes
     present per split) and the largest sequence families per split (named by the most
     common UniProt accession, SIFTS).

Paper
-----
`appendix.tex` \\label{app:experiments} (MISATO sequence / structure / random splits);
`2_results.tex` \\label{sec:downstreams}. The paper text describing the old split rule must
be updated (Open, audit D.6: paper text of the split rule; text pass item).

Origin
------
New (2026-10-18, authors' decision D3; 2026-10-19: rule `graph` after the split re-audit,
chain ids by in-order alignment). Replaces, for the probes, the research script
`scripts/baselines/build_misato_splits.py` (SIFTS UniProt grouping / first CATH H code,
80/10/10 of the clusters). Union-find and the SIFTS / CATH inputs are re-implemented here.

Used by the reported model: yes (every MISATO probe reads `data/misato_splits_v2.json`
through `ensembits.paths.MISATO_SPLITS`).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/misato_splits.json                              (--old-splits; universe,
                                                                random, ligand_disjoint)
          data/cached_descriptors/misato_chain_breaks.json     (chain pieces, v2.3)
          data/cached_descriptors/misato_real_bb/<PID>.npz     (seq)
          data/external/rcsb_mmcif/<PDBID>.cif.gz              (chain ids, in-order alignment)
          data/external/uniprot/misato_sifts_entries.json.gz   (polyprotein products; written
                                                                by build_misato_ec_labels.py)
          data/external/sifts/pdb_chain_cath_uniprot.tsv.gz    (--download fetches it;
          data/external/sifts/pdb_chain_uniprot.tsv.gz          _manifest.json: url, date,
          data/external/cath/cath-b-newest-all.gz               Last-Modified, sha256)
          data/annotations/misato_{ec,go,affinity}.json, data/annotations/
              misato_binding_site_v2.npz (D4 labels)          (class balance only)
  writes  data/misato_splits_v2.json (+ .provenance.json: parameters, input hashes, group
          sizes, giant components, no-CATH handling, chain table, leakage report)
          data/misato_splits_v2.chains.json (per complex: chains, sequence clusters, H codes)
          data/misato_splits_v2.mmseqs/ (chains.fa, allvsall.m8, mmseqs.log;
          sha256 of each in the sidecar)
  uses    the mmseqs executable (ensembits.paths.MMSEQS_BIN) in --workdir

How to run
----------
    python data_prep/build_misato_splits_v2.py [--download] [--threads 32] [--overwrite]
    python data_prep/build_misato_splits_v2.py --overwrite --compare v2prev=<earlier v2 file>
    python data_prep/build_misato_splits_v2.py --report-only   # leakage of an existing file
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))             # rmsf_common
sys.path.insert(0, str(CODE_ROOT / 'data_prep'))              # build_misato_chain_breaks
sys.path.insert(0, str(CODE_ROOT / 'data_prep' / 'baselines'))  # _chain_pieces
from ensembits.paths import (REPO_ROOT, EXTERNAL, MMSEQS_BIN, MISATO_SPLITS,  # noqa: E402
                             MISATO_SPLITS_LEGACY)
from ensembits.data_versions import write_provenance, read_provenance  # noqa: E402
from ensembits.featurize import file_sha256  # noqa: E402

SCHEMA = 'misato-splits-v2'
SEED = 42
SPLITS = ('train', 'val', 'test')
NEW_SPLITS = ('sequence', 'structure')
COPIED_SPLITS = ('random', 'ligand_disjoint')
MIN_CHAIN_LEN = 20            # pieces shorter than this (peptides) are not clustered
MIN_SEQ_ID = 0.30             # (a) identity of a link
MIN_COV_SHORT = 0.50          # (a) alignment covers >= 50 % of the shorter chain
MIN_ALN = 30                  # (a) >= 30 aligned residues
NEAR_ID = 0.95                # (b) same protein: identity >= 95 % ...
NEAR_ALN = 50                 # (b) ... over >= 50 aligned residues
COV08 = 0.80                  # rule cov08 (first v2 file): coverage of both chains
SENSITIVITY = 7.5
EVALUE = 1e-3
POLY_MIN_OVERLAP = 20         # residues of a chain inside a polyprotein product
CLUSTER_RULES = {
    'graph': ('connected components over the exhaustive all-vs-all search (no coverage '
              'prefilter) of links: identity >= 0.30 over >= 30 aligned residues covering '
              '>= 0.50 of the shorter chain; or identity >= 0.95 over >= 50 aligned residues; '
              'or the same UniProt accession (SIFTS; polyproteins: the same mature product)'),
    'cov08': ('first v2 file (2026-10-18): connected components of links with identity '
              '>= 0.30 covering >= 0.80 of both chains (no near-identity or UniProt links)')}
MAX_EVAL_GROUP_FRAC = 0.10    # a group above 10 % of the val / test target goes to train
LEAK_IDS = (0.3, 0.5, 0.95)
LEAK_FILTERS = ('nocov_aln50', 'cov05_short', 'cov08_both')

SIFTS_DIR = EXTERNAL / 'sifts'
CATH_DIR = EXTERNAL / 'cath'
SIFTS_CATH = SIFTS_DIR / 'pdb_chain_cath_uniprot.tsv.gz'
SIFTS_UNIPROT = SIFTS_DIR / 'pdb_chain_uniprot.tsv.gz'
CATH_B = CATH_DIR / 'cath-b-newest-all.gz'
UNIPROT_ENTRIES = EXTERNAL / 'uniprot' / 'misato_sifts_entries.json.gz'   # EC labels builder (D8)
DOWNLOADS = {
    SIFTS_CATH: 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/'
                'pdb_chain_cath_uniprot.tsv.gz',
    SIFTS_UNIPROT: 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/'
                   'pdb_chain_uniprot.tsv.gz',
    CATH_B: 'https://download.cathdb.info/cath/releases/daily-release/newest/'
            'cath-b-newest-all.gz',
}
CHAINS_OUT_SUFFIX = '.chains.json'
ANN = REPO_ROOT / 'data' / 'annotations'
EC_LABELS = ANN / 'misato_ec.json'
GO_LABELS = ANN / 'misato_go.json'
AFF_LABELS = ANN / 'misato_affinity.json'
# FIXED(benchmark_audit D4, 2026-10-19): the class-balance report reads the v2 binding-site
#   labels (old: data/cached_descriptors/misato_binding_site.npz, misaligned). Report only.
from ensembits.misato_binding_labels import MISATO_BINDING_SITE as BS_LABELS  # noqa: E402


# ---------------------------------------------------------------- union-find
class UnionFind:
    """Disjoint sets over hashable items; `groups()` lists the sets of the added items."""

    def __init__(self, items=()):
        self.parent: dict = {}
        for x in items:
            self.add(x)

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:                # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if str(ra) < str(rb):                    # deterministic root
                ra, rb = rb, ra
            self.parent[ra] = rb

    def groups(self, items=None) -> list[list]:
        """Sets of `items` (default every item), each sorted, ordered by their first item."""
        out = defaultdict(list)
        for x in (self.parent if items is None else items):
            out[self.find(x)].append(x)
        return sorted((sorted(v) for v in out.values()), key=lambda g: g[0])


def groups_by_shared_keys(complex_keys: dict) -> list[list[str]]:
    """Complexes sharing any key are in one group (a complex with keys A and B joins both).

    complex_keys: {pid: iterable of keys}; a complex without keys is its own group.
    """
    uf = UnionFind(complex_keys)
    owner: dict = {}
    for pid in sorted(complex_keys):
        for k in sorted(set(complex_keys[pid])):
            if k in owner:
                uf.union(pid, owner[k])
            else:
                owner[k] = pid
    return uf.groups(list(complex_keys))


def structure_groups(complex_h: dict, complex_clusters: dict) -> tuple[list, dict]:
    """Union-find over shared CATH H codes; complexes without H are attached through their
    sequence clusters (every cluster holding an unlabelled complex joins all its complexes).

    Returns (groups, info) with info = counts of how the unlabelled complexes were placed.
    """
    pids = sorted(complex_h)
    uf = UnionFind(pids)
    owner: dict = {}
    for pid in pids:                                  # shared H codes
        for h in sorted(set(complex_h[pid])):
            key = ('H', h)
            if key in owner:
                uf.union(pid, owner[key])
            else:
                owner[key] = pid
    no_cath = [p for p in pids if not complex_h[p]]
    members = defaultdict(set)
    for p in pids:
        for c in complex_clusters.get(p, ()):
            members[c].add(p)
    for p in no_cath:                                 # attach via sequence clusters
        for c in sorted(set(complex_clusters.get(p, ()))):
            for q in sorted(members[c]):
                uf.union(p, q)
    groups = uf.groups(pids)
    gid = {p: i for i, g in enumerate(groups) for p in g}
    has_h = {i: any(complex_h[p] for p in g) for i, g in enumerate(groups)}
    info = {'n_complexes_without_cath': len(no_cath),
            'n_without_cath_in_group_with_cath': sum(has_h[gid[p]] for p in no_cath),
            'n_without_cath_in_group_without_cath_size_gt1':
                sum((not has_h[gid[p]]) and len(groups[gid[p]]) > 1 for p in no_cath),
            'n_without_cath_singleton': sum(len(groups[gid[p]]) == 1 for p in no_cath),
            'n_without_cath_no_clustered_chain':
                sum(not complex_clusters.get(p) for p in no_cath)}
    return groups, info


# ---------------------------------------------------------------- assignment
def assign_groups(groups: list, fractions=(0.8, 0.1, 0.1), seed: int = SEED,
                  max_eval_group_frac: float = MAX_EVAL_GROUP_FRAC) -> tuple[dict, dict]:
    """Groups -> {train, val, test} (sorted pid lists) + info.

    Deterministic: groups are canonicalised (sorted members, ordered by their first member)
    before the RandomState(seed) shuffle, so the input order does not matter. Groups above
    max_eval_group_frac of the smaller val / test target go to train (largest first); every
    other group goes to the split with the largest relative deficit.
    """
    groups = sorted((sorted(g) for g in groups), key=lambda g: g[0])
    n = sum(len(g) for g in groups)
    target = dict(zip(SPLITS, (f * n for f in fractions)))
    cap = max_eval_group_frac * min(target['val'], target['test'])
    count = dict.fromkeys(SPLITS, 0)
    out = {k: [] for k in SPLITS}
    big = sorted((g for g in groups if len(g) > cap), key=lambda g: (-len(g), g[0]))
    for g in big:
        out['train'].extend(g); count['train'] += len(g)
    rest = [g for g in groups if len(g) <= cap]
    order = np.random.RandomState(seed).permutation(len(rest))
    for i in order:
        g = rest[i]
        k = max(SPLITS, key=lambda s: ((target[s] - count[s]) / target[s] if target[s] else
                                       -np.inf, -SPLITS.index(s)))
        out[k].extend(g); count[k] += len(g)
    info = {'n_groups': len(groups), 'cap_eval_group_size': cap,
            'n_groups_forced_train': len(big),
            'n_complexes_forced_train': sum(len(g) for g in big),
            'forced_train_sizes': [len(g) for g in big],
            'largest_group_sizes': sorted((len(g) for g in groups), reverse=True)[:10],
            'giant_component': {'size': max((len(g) for g in groups), default=0),
                                'frac': max((len(g) for g in groups), default=0) / max(n, 1)},
            'n_singletons': sum(len(g) == 1 for g in groups),
            'counts': {k: len(v) for k, v in out.items()},
            'fractions': {k: round(len(v) / max(n, 1), 4) for k, v in out.items()}}
    return {k: sorted(v) for k, v in out.items()}, info


def check_no_straddle(split: dict, groups: list) -> None:
    """Every group lies in exactly one of train / val / test; the split covers each pid once."""
    where = {}
    for k in SPLITS:
        for p in split[k]:
            if p in where:
                raise ValueError(f'{p} in {where[p]} and {k}')
            where[p] = k
    for g in groups:
        ks = {where[p] for p in g}
        if len(ks) != 1:
            raise ValueError(f'group {g[:5]}... straddles {sorted(ks)}')
    if set(where) != {p for g in groups for p in g}:
        raise ValueError('split and groups cover different complexes')


# ---------------------------------------------------------------- chains
def _complex_chains(job):
    """(pid, [{start, end, seq, chain, chain_source, table_chain, aligned_chain}], note) of
    one complex (worker). The chain id comes from the in-order alignment to the deposited
    mmCIF (distinct homomer copies), else from the chain-break table."""
    pid, rb_dir, dep_dir, rec = job
    from _chain_pieces import pieces_from_breaks
    with np.load(Path(rb_dir) / f'{pid}.npz', allow_pickle=False) as z:
        seq = str(z['seq'])
    if int(rec['L']) != len(seq):
        raise ValueError(f'{pid}: chain-break table L={rec["L"]}, misato_real_bb {len(seq)}')
    pieces, _ = pieces_from_breaks(len(seq), rec['breaks'])
    from_breaks = {}
    for b in rec['breaks']:
        if b['kind'] not in ('boundary', 'unresolved'):
            continue
        pos = int(b['pos'])
        i = next(j for j, (s, _) in enumerate(pieces) if s == pos)
        if b.get('left_chain') and isinstance(b['left_chain'], str):
            from_breaks.setdefault(i - 1, b['left_chain'])
        if b.get('right_chain') and isinstance(b['right_chain'], str):
            from_breaks.setdefault(i, b['right_chain'])
    aligned, note = {}, None
    from build_misato_chain_breaks import deposited_file, deposited_chains, assign_segments
    f = deposited_file(Path(dep_dir), pid)
    if f is None:
        note = 'no deposited file'
    else:
        chains = deposited_chains(f)
        _, res, _ = assign_segments(seq, [s for s, _ in pieces[1:]], chains)
        for i, r in enumerate(res):
            if r is not None:
                aligned[i] = r[0]
    out = []
    for i, (s, e) in enumerate(pieces):
        if i in aligned:
            c, src = aligned[i], 'alignment'
        elif i in from_breaks:
            c, src = from_breaks[i], 'chain_break_table'
        else:
            c, src = None, None
        out.append({'start': s, 'end': e, 'seq': seq[s:e], 'chain': c, 'chain_source': src,
                    'table_chain': from_breaks.get(i), 'aligned_chain': aligned.get(i)})
    return pid, out, note


def complex_chains(pids, table, rb_dir, dep_dir, workers: int = 32) -> tuple[dict, dict]:
    """{pid: [chain records]} of every complex + counts (chain-id sources, failures)."""
    jobs = [(p, str(rb_dir), str(dep_dir), table[p]) for p in pids]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_complex_chains, jobs, chunksize=4)
    else:
        res = [_complex_chains(j) for j in jobs]
    out, notes = {}, {}
    for pid, ch, note in res:
        out[pid] = ch
        if note:
            notes[pid] = note
    allc = [c for v in out.values() for c in v]
    both = [c for c in allc if c['table_chain'] and c['aligned_chain']]
    dis = [(p, c['start'], c['table_chain'], c['aligned_chain']) for p, v in out.items()
           for c in v if c['table_chain'] and c['aligned_chain']
           and c['table_chain'] != c['aligned_chain']]
    dup = sorted(p for p, v in out.items()
                 if len([c['chain'] for c in v if c['chain']]) !=
                 len({c['chain'] for c in v if c['chain']}))
    info = {'n_chains': len(allc),
            'chain_id_source': dict(Counter(str(c['chain_source']) for c in allc)),
            'n_chain_table_vs_alignment_checked': len(both),
            'n_chain_table_vs_alignment_disagree': len(dis),
            'chain_table_vs_alignment_disagree_examples': dis[:20],
            'n_complexes_with_repeated_chain_id': len(dup),
            'complexes_with_repeated_chain_id': dup[:50],
            'complex_notes': notes,
            'n_chains_per_complex': dict(sorted(Counter(len(v) for v in out.values()).items()))}
    return out, info


# ---------------------------------------------------------------- MMseqs2
def unique_sequences(chains: dict, min_len: int = MIN_CHAIN_LEN) -> dict:
    """{sequence: 's<k>'} of the distinct chain sequences of >= min_len residues (sorted)."""
    seqs = sorted({c['seq'] for v in chains.values() for c in v if len(c['seq']) >= min_len})
    return {s: f's{i}' for i, s in enumerate(seqs)}


def _write_fasta(path: Path, seq_ids: dict) -> None:
    with open(path, 'w') as f:
        for s, i in sorted(seq_ids.items(), key=lambda kv: int(kv[1][1:])):
            f.write(f'>{i}\n{s}\n')


def _run(cmd, log):
    with open(log, 'a') as fh:
        fh.write('$ ' + ' '.join(map(str, cmd)) + '\n'); fh.flush()
        subprocess.run([str(c) for c in cmd], check=True, stdout=fh, stderr=subprocess.STDOUT)


def mmseqs_version(mmseqs: str = MMSEQS_BIN) -> str:
    return subprocess.run([mmseqs, 'version'], capture_output=True, text=True).stdout.strip()


def search_args(threads) -> list:
    return ['--exhaustive-search', 1, '-s', SENSITIVITY, '-e', EVALUE, '--alignment-mode', 3,
            '--max-seqs', 100000,
            '--format-output', 'query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue',
            '--threads', threads]


def cov_short(h) -> float:
    """Alignment coverage of the shorter chain of a hit (q, t, fid, aln, qlen, tlen, qcov, tcov)."""
    return h[6] if h[4] <= h[5] else h[7]


def link_graph(h) -> bool:
    """Rule `graph` sequence link: (a) >= 30 % over >= 30 residues and >= 50 % of the shorter
    chain, or (b) >= 95 % over >= 50 residues."""
    return ((h[2] >= MIN_SEQ_ID and h[3] >= MIN_ALN and cov_short(h) >= MIN_COV_SHORT)
            or (h[2] >= NEAR_ID and h[3] >= NEAR_ALN))


def link_cov08(h) -> bool:
    """Rule `cov08` link (first v2 file): >= 30 % covering >= 80 % of both chains."""
    return h[2] >= MIN_SEQ_ID and h[6] >= COV08 and h[7] >= COV08


LEAK_FILTER_FN = {'nocov_aln50': lambda h: h[3] >= NEAR_ALN,
                  'cov05_short': lambda h: cov_short(h) >= MIN_COV_SHORT,
                  'cov08_both': lambda h: h[6] >= COV08 and h[7] >= COV08}


def connected_clusters(seq_ids: dict, pairs) -> dict:
    """{seq id: cluster id}: connected components of the links `pairs` [(a, b), ...]
    (single linkage). The cluster id is the smallest member id (by number)."""
    uf = UnionFind(seq_ids.values())
    for q, t in pairs:
        uf.union(q, t)
    comp = defaultdict(list)
    for i in seq_ids.values():
        comp[uf.find(i)].append(i)
    return {i: min(m, key=lambda x: int(x[1:])) for m in comp.values() for i in m}


def mmseqs_all_vs_all(seq_ids: dict, workdir: Path, threads: int, mmseqs=MMSEQS_BIN) -> list:
    """[(query, target, fident, alnlen, qlen, tlen, qcov, tcov)] of the exhaustive all-vs-all
    search (search_args; no coverage or identity prefilter; self hits dropped)."""
    fa = workdir / 'chains.fa'
    if not fa.exists():
        _write_fasta(fa, seq_ids)
    _run([mmseqs, 'easy-search', fa, fa, workdir / 'allvsall.m8', workdir / 'tmp_search',
          *search_args(threads)], workdir / 'mmseqs.log')
    hits = []
    for line in open(workdir / 'allvsall.m8'):
        q, t, fid, aln, ql, tl, qc, tc, _ = line.rstrip('\n').split('\t')
        if q != t:
            hits.append((q, t, float(fid), int(aln), int(ql), int(tl), float(qc), float(tc)))
    return hits


# ---------------------------------------------------------------- SIFTS / CATH
def download_inputs(paths=None, timeout: float = 120.0) -> dict:
    """Fetch the SIFTS / CATH-B files; record url, date, Last-Modified and sha256 in each
    folder's _manifest.json."""
    out = {}
    for path, url in DOWNLOADS.items():
        if paths is not None and path not in paths:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'.{path.name}.tmp')
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
            last_mod = r.headers.get('Last-Modified')
        os.replace(tmp, path)
        man_p = path.parent / '_manifest.json'
        man = json.loads(man_p.read_text()) if man_p.exists() else {}
        man[path.name] = {'url': url, 'downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
                          'last_modified': last_mod, 'sha256': file_sha256(path)}
        tmpm = man_p.with_name('._manifest.json.tmp')
        tmpm.write_text(json.dumps(man, indent=1, sort_keys=True))
        os.replace(tmpm, man_p)
        out[path.name] = man[path.name]
    return out


def input_record(path: Path) -> dict:
    man_p = path.parent / '_manifest.json'
    man = json.loads(man_p.read_text()).get(path.name, {}) if man_p.exists() else {}
    return {'path': _rel(path), 'sha256': file_sha256(path), 'url': man.get('url'),
            'downloaded': man.get('downloaded'), 'last_modified': man.get('last_modified'),
            'manifest_sha256_matches': man.get('sha256') == file_sha256(path) if man else None}


def deposited_record(dep_dir, pids) -> dict:
    """Deposited mmCIF entries used for the chain ids: count and a sha256 over the
    (file, sha256) records of the RCSB manifest (build_misato_chain_breaks.download)."""
    from build_misato_chain_breaks import deposited_file, read_manifest
    man = read_manifest(Path(dep_dir)).get('files', {})
    files = sorted(f.name for p in pids if (f := deposited_file(Path(dep_dir), p)) is not None)
    h = hashlib.sha256()
    for f in files:
        h.update(f'{f}|{(man.get(f) or {}).get("sha256")}\n'.encode())
    return {'dir': _rel(dep_dir), 'n_files': len(files), 'n_missing': len(pids) - len(files),
            'n_without_manifest_sha256': sum(not (man.get(f) or {}).get('sha256')
                                             for f in files),
            'manifest_records_sha256': h.hexdigest()}


def read_cath_b(path: Path) -> dict:
    """{domain id: "C.A.T.H"} of a CATH-B file (domain, version, code, boundaries)."""
    out = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and not line.startswith('#'):
                out[p[0]] = p[2]
    return out


def _read_sifts(path: Path, pdbs: set) -> list[dict]:
    rows = []
    with gzip.open(path, 'rt') as f:
        header = None
        for line in f:
            if line.startswith('#'):
                continue
            p = line.rstrip('\n').split('\t')
            if header is None:
                header = p; continue
            r = dict(zip(header, p))
            if r['PDB'].lower() in pdbs:
                rows.append(r)
    return rows


def chain_cath(pdbs: set, sifts_cath: Path, cath_b: Path) -> tuple[dict, dict]:
    """{(pdb lower, chain): set of H codes} + counts by source.

    H codes of a chain = CATH-B code of every SIFTS domain of that chain, plus every CATH-B
    domain whose id starts with <pdb><chain> (single-character chains)."""
    code = read_cath_b(cath_b)
    out = defaultdict(set)
    n_sifts, n_sifts_missing, n_direct = 0, 0, 0
    for r in _read_sifts(sifts_cath, pdbs):
        n_sifts += 1
        h = code.get(r['CATH_ID'])
        if h is None:
            n_sifts_missing += 1; continue
        out[(r['PDB'].lower(), r['CHAIN'])].add(h)
    for dom, h in code.items():
        pdb, ch = dom[:4].lower(), dom[4]
        if pdb in pdbs:
            if h not in out[(pdb, ch)]:
                n_direct += 1
            out[(pdb, ch)].add(h)
    return dict(out), {'n_sifts_rows': n_sifts, 'n_sifts_domains_not_in_cath_b': n_sifts_missing,
                       'n_codes_added_from_cath_b_direct': n_direct,
                       'n_cath_b_domains': len(code)}


def chain_uniprot(pdbs: set, sifts_uniprot: Path) -> dict:
    """{(pdb lower, chain): set of UniProt accessions} (SIFTS pdb_chain_uniprot)."""
    out = defaultdict(set)
    for r in _read_sifts(sifts_uniprot, pdbs):
        out[(r['PDB'].lower(), r['CHAIN'])].add(r['SP_PRIMARY'].split('-')[0])
    return dict(out)


def chain_uniprot_keys(pdbs: set, sifts_uniprot: Path, entries_path: Path = UNIPROT_ENTRIES
                       ) -> tuple[dict, dict]:
    """{(pdb lower, chain): set of link keys} for rule (c) + counts.

    Key = the accession; for a polyprotein accession (entry_products()['polyprotein'] with
    located products) = (accession, mature product with the largest Jaccard overlap with the
    chain's SIFTS UniProt range, >= POLY_MIN_OVERLAP residues or half the range), else
    (accession, 'precursor region'). Accessions missing from the UniProt file are used as is.
    """
    from build_misato_ec_labels import entry_products
    entries = {}
    if Path(entries_path).exists():
        with gzip.open(entries_path, 'rt') as f:
            entries = json.load(f).get('entries', {})
    pos = defaultdict(lambda: defaultdict(set))
    for r in _read_sifts(sifts_uniprot, pdbs):
        try:
            b, e = int(r['SP_BEG']), int(r['SP_END'])
        except ValueError:
            continue
        pos[(r['PDB'].lower(), r['CHAIN'])][r['SP_PRIMARY'].split('-')[0]].update(range(b, e + 1))
    info_of, out = {}, defaultdict(set)
    n_prod, n_prec, poly, missing = 0, 0, set(), set()
    for key, accs in pos.items():
        for acc, ps in accs.items():
            if acc not in info_of:
                info_of[acc] = entry_products(entries[acc]) if acc in entries else None
                if acc not in entries:
                    missing.add(acc)
            inf = info_of[acc]
            if not inf or not inf['polyprotein'] or not inf['products']:
                out[key].add(acc); continue
            poly.add(acc)
            best, bj = None, 0.0
            for pr in inf['products']:
                span = set(range(pr['start'], pr['end'] + 1))
                ov = len(ps & span)
                if ov >= min(POLY_MIN_OVERLAP, 0.5 * len(ps)):
                    j = ov / len(ps | span)
                    if j > bj:
                        best, bj = pr['name'], j
            if best is None:
                out[key].add((acc, 'precursor region')); n_prec += 1
            else:
                out[key].add((acc, best)); n_prod += 1
    return dict(out), {'uniprot_entries': input_record(Path(entries_path))
                       if Path(entries_path).exists() else None,
                       'n_accessions': len(info_of), 'n_polyprotein_accessions': len(poly),
                       'polyprotein_accessions': sorted(poly),
                       'n_polyprotein_chains_by_product': n_prod,
                       'n_polyprotein_chains_precursor_region': n_prec,
                       'n_accessions_missing_from_uniprot_file': len(missing)}


# ---------------------------------------------------------------- reports
def _owners(chains: dict, seq_ids: dict) -> dict:
    """{seq id: set of pids}."""
    own = defaultdict(set)
    for p, v in chains.items():
        for c in v:
            if c['seq'] in seq_ids:
                own[seq_ids[c['seq']]].add(p)
    return own


def leakage(split: dict, chains: dict, seq_ids: dict, hits: list, complex_h: dict,
            complex_acc: dict | None = None) -> dict:
    """Per val / test: complexes with any chain >= 30 / 50 / 95 % identical to a chain of a
    train complex under each LEAK_FILTERS coverage filter (`any_chain`), with an identical
    chain sequence in train, sharing a UniProt accession with train and sharing a CATH H
    code with train (counts and fractions)."""
    where = {p: k for k in SPLITS for p in split[k]}
    own = _owners(chains, seq_ids)
    in_train = {s: any(where.get(p) == 'train' for p in ps) for s, ps in own.items()}
    best = {f: defaultdict(float) for f in LEAK_FILTERS}  # seq id -> best identity to train
    for h in hits:
        if in_train.get(h[1]):
            for f, fn in LEAK_FILTER_FN.items():
                if fn(h) and h[2] > best[f][h[0]]:
                    best[f][h[0]] = h[2]
    for s in own:                                     # identical sequence in a train complex
        if in_train[s]:
            for f in LEAK_FILTERS:
                best[f][s] = 1.0
    train_h = {h for p in split['train'] for h in complex_h.get(p, ())}
    complex_acc = complex_acc or {}
    train_acc = {a for p in split['train'] for a in complex_acc.get(p, ())}
    out = {}
    for k in ('val', 'test'):
        pids = split[k]
        n = max(len(pids), 1)
        ids = {p: [seq_ids[c['seq']] for c in chains.get(p, []) if c['seq'] in seq_ids]
               for p in pids}
        rec = {'n': len(pids), 'any_chain': {}}
        for f in LEAK_FILTERS:
            r = {}
            for t in LEAK_IDS:
                m = sum(max((best[f].get(i, 0.0) for i in ids[p]), default=0.0) >= t
                        for p in pids)
                r[f'ge{int(t * 100)}'] = m
                r[f'frac_ge{int(t * 100)}'] = round(m / n, 4)
            rec['any_chain'][f] = r
        for name, m in (('identical_chain_in_train',
                         sum(any(in_train.get(i) for i in ids[p]) for p in pids)),
                        ('shares_uniprot_with_train',
                         sum(bool(set(complex_acc.get(p, ())) & train_acc) for p in pids)),
                        ('shares_cath_h_with_train',
                         sum(bool(set(complex_h.get(p, ())) & train_h) for p in pids))):
            rec[name] = m
            rec[f'frac_{name}'] = round(m / n, 4)
        out[k] = rec
    return out


def load_task_labels() -> dict:
    """{task: {PID upper: labels}} of the probe label files (EC, GO, affinity, binding)."""
    out = {}
    ec, src = None, None
    try:                          # the probes' EC labels (v2, benchmark_audit D8) if built
        from ec_labels import load_ec_labels
        ec, rec = load_ec_labels()
        src = Path(rec['path']).name
    except (ImportError, SystemExit):
        if EC_LABELS.exists():
            ec, src = json.loads(EC_LABELS.read_text()), EC_LABELS.name
    if ec is not None:
        out['ec_depth1'] = {p.upper(): sorted({str(e).split('.')[0] for e in v
                                                if str(e).split('.')[0] not in ('', '-')})
                            for p, v in ec.items() if v}
        out['ec_source'] = src
    try:                          # the probes' GO labels (v2, benchmark_audit D7) if built
        from go_labels import GO_LABELS_V2, load_go_labels
        if GO_LABELS_V2.exists():
            out['go_v2'], _ = load_go_labels(GO_LABELS_V2)
    except (ImportError, SystemExit):
        pass
    if 'go_v2' not in out and GO_LABELS.exists():
        out['go_raw'] = {p.upper(): v for p, v in json.loads(GO_LABELS.read_text()).items()
                         if isinstance(v, list)}
    if AFF_LABELS.exists():
        out['affinity'] = {p.upper(): v for p, v in json.loads(AFF_LABELS.read_text()).items()
                           if v is not None}
    if BS_LABELS.exists():        # v2 labels (benchmark_audit D4), sidecar checked
        from ensembits.misato_binding_labels import load_binding_site_labels
        out['binding_site'] = {p.upper(): int(np.asarray(y).sum())
                               for p, y in load_binding_site_labels(BS_LABELS).items()}
    return out


def class_balance(split: dict, labels: dict) -> dict:
    """Complexes with labels per split and task; EC depth-1 class counts; GO top-50 (most
    frequent in train, as tokenize_misato_GO_eval.build_go_labelset) coverage."""
    up = {k: [p.upper() for p in split[k]] for k in SPLITS}
    out = {}
    if 'ec_depth1' in labels:
        ec = labels['ec_depth1']
        out['ec_depth1'] = {k: {'labels': labels.get('ec_source'),
                                'n_labelled': sum(p in ec for p in v),
                                'n_classes_present': len({c for p in v for c in ec.get(p, [])}),
                                'classes': dict(sorted(Counter(c for p in v
                                                               for c in ec.get(p, [])).items()))}
                            for k, v in up.items()}
    if 'go_v2' in labels:         # same class selection as the GO probe (go_labels.py)
        from go_labels import build_go_labelset
        sel, classes, _ = build_go_labelset(labels['go_v2'], up['train'], 'top50')
        out['go_top50'] = {k: {'labels': 'misato_go_mf_v2.json',
                               'n_labelled': sum(p in sel for p in v),
                               'n_classes_present': len({t for p in v for t in sel.get(p, ())})}
                           for k, v in up.items()}
    elif 'go_raw' in labels:
        go = labels['go_raw']
        cnt = Counter(t for p in up['train'] for t in go.get(p, []))
        top = {t for t, _ in cnt.most_common(50)}
        out['go_top50'] = {k: {'labels': 'misato_go.json',
                               'n_labelled': sum(bool(set(go.get(p, [])) & top) for p in v),
                               'n_classes_present': len({t for p in v for t in go.get(p, [])}
                                                        & top)}
                           for k, v in up.items()}
    if 'binding_site' in labels:
        bs = labels['binding_site']
        out['binding_site'] = {k: {'labels': BS_LABELS.name,
                                   'n_with_labels': sum(p in bs for p in v),
                                   'n_with_positive': sum(bs.get(p, 0) > 0 for p in v)}
                               for k, v in up.items()}
    if 'affinity' in labels:
        out['affinity'] = {k: {'n_labelled': sum(p in labels['affinity'] for p in v)}
                           for k, v in up.items()}
    return out


def families(split: dict, complex_clusters: dict, cluster_names: dict, top: int = 5) -> dict:
    """Largest sequence families (MMseqs2 30 % clusters) per split, with their share."""
    out = {}
    for k in SPLITS:
        cnt = Counter(c for p in split[k] for c in set(complex_clusters.get(p, ())))
        n = max(len(split[k]), 1)
        out[k] = [{'cluster': c, 'name': cluster_names.get(c), 'n': m, 'frac': round(m / n, 4)}
                  for c, m in cnt.most_common(top)]
    return out


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _atomic_json(path: Path, obj, indent=None) -> None:
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=indent))
    os.replace(tmp, path)


def splits_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- main
def universe(old: dict) -> list[str]:
    """The complexes of the old `random` split (PIDs as written there)."""
    return sorted({p for k in SPLITS for p in old['random'][k]})


def main(argv=None) -> int:
    from rmsf_common import MISATO_CHAIN_BREAKS, MISATO_REAL_BB_DIR, RCSB_MMCIF_DIR
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--old-splits', default=str(MISATO_SPLITS_LEGACY))
    ap.add_argument('--out', default=str(MISATO_SPLITS))
    ap.add_argument('--chain-breaks', default=str(MISATO_CHAIN_BREAKS))
    ap.add_argument('--real-bb-dir', default=str(MISATO_REAL_BB_DIR))
    ap.add_argument('--deposited-dir', default=str(RCSB_MMCIF_DIR))
    ap.add_argument('--workdir', default=None,
                    help='MMseqs2 files (default: <out stem>.mmseqs/ next to the output; kept: '
                         'chains.fa, allvsall.m8, mmseqs.log)')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--cluster-rule', choices=sorted(CLUSTER_RULES), default='graph')
    ap.add_argument('--compare', action='append', default=[], metavar='NAME=PATH',
                    help='another split file to include in the report (e.g. an earlier v2)')
    ap.add_argument('--download', action='store_true', help='fetch the SIFTS / CATH-B files')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--report-only', action='store_true',
                    help='no new split: print the leakage report of --out and --old-splits')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.resolve() == Path(a.old_splits).resolve():
        print('refusing to write over the old split file'); return 1
    if out.exists() and not (a.overwrite or a.report_only):
        print(f'{out} exists; pass --overwrite to rebuild'); return 1
    compare = {}
    for c in a.compare:
        name, _, path = c.partition('=')
        if not path or not Path(path).exists():
            print(f'--compare {c!r}: expected NAME=PATH of an existing file'); return 1
        compare[name] = Path(path)
    if a.download:
        print('[download]', json.dumps(download_inputs(), indent=1))
    for p in (SIFTS_CATH, SIFTS_UNIPROT, CATH_B):
        if not p.exists():
            print(f'{p} missing: run with --download'); return 1

    old = json.loads(Path(a.old_splits).read_text())
    pids = universe(old)
    up = {p: p.upper() for p in pids}
    if a.download:                                    # deposited entries of every complex
        from build_misato_chain_breaks import download
        failed = download([up[p] for p in pids], Path(a.deposited_dir))
        print(f'[download] mmCIF: {len(failed)} failed {sorted(failed)[:10]}', flush=True)
    n = len(pids)
    fr = {k: len(old['random'][k]) / n for k in SPLITS}
    fractions = (0.8, 0.1, 0.1)
    print(f'[universe] {n} complexes; old random fractions {fr}', flush=True)

    table = json.loads(Path(a.chain_breaks).read_text())
    if table.get('schema') != 'misato-chain-breaks-v2':
        print(f'chain-break table schema {table.get("schema")!r}'); return 1
    tprov = read_provenance(a.chain_breaks) or {}
    chains_up, chain_info = complex_chains([up[p] for p in pids], table['complexes'],
                                           a.real_bb_dir, a.deposited_dir, a.workers)
    chains = {p: chains_up[up[p]] for p in pids}
    print(f'[chains] {chain_info["n_chains"]} chains; ids {chain_info["chain_id_source"]}; '
          f'table vs alignment disagree {chain_info["n_chain_table_vs_alignment_disagree"]}',
          flush=True)

    # ---- SIFTS: CATH H codes, UniProt accessions and link keys per chain
    pdbs = {p.lower() for p in pids}
    ch_h, cath_info = chain_cath(pdbs, SIFTS_CATH, CATH_B)
    ch_up = chain_uniprot(pdbs, SIFTS_UNIPROT)
    ch_key, key_info = chain_uniprot_keys(pdbs, SIFTS_UNIPROT)
    for p in pids:
        for c in chains[p]:
            k = (p.lower(), c['chain'])
            c['cath_h'] = sorted(ch_h.get(k, ())) if c['chain'] else []
            c['uniprot'] = sorted(ch_up.get(k, ())) if c['chain'] else []
            c['uniprot_keys'] = (sorted(ch_key.get(k, ()), key=str) if c['chain'] else [])
    complex_h = {p: sorted({h for c in chains[p] for h in c['cath_h']}) for p in pids}
    complex_acc = {p: sorted({u for c in chains[p] for u in c['uniprot']}) for p in pids}

    # ---- MMseqs2 all-vs-all and the sequence clusters
    seq_ids = unique_sequences(chains)
    work = Path(a.workdir) if a.workdir else out.with_name(out.stem + '.mmseqs')
    if a.report_only:
        work = Path(tempfile.mkdtemp(prefix='misato_splits_report_'))
    elif work.exists():
        if not a.overwrite:
            print(f'{work} exists; pass --overwrite to rebuild'); return 1
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=a.report_only)
    hits = mmseqs_all_vs_all(seq_ids, work, a.threads)
    seq_links = [(h[0], h[1]) for h in hits if link_graph(h)]
    key_owner, acc_links = {}, []                     # rule (c): same accession / product
    for p in pids:
        for c in chains[p]:
            if c['seq'] not in seq_ids:
                continue
            for k in c['uniprot_keys']:
                kk = json.dumps(k)
                if kk in key_owner:
                    acc_links.append((key_owner[kk], seq_ids[c['seq']]))
                else:
                    key_owner[kk] = seq_ids[c['seq']]
    rep_graph = connected_clusters(seq_ids, seq_links + acc_links)
    rep_seqonly = connected_clusters(seq_ids, seq_links)
    rep_cov08 = connected_clusters(seq_ids, [(h[0], h[1]) for h in hits if link_cov08(h)])
    rep = rep_graph if a.cluster_rule == 'graph' else rep_cov08

    def cc_of(r):
        return {p: sorted({r[seq_ids[c['seq']]] for c in chains[p] if c['seq'] in seq_ids})
                for p in pids}
    complex_clusters = cc_of(rep)
    n_acc_merges = len(set(rep_seqonly.values())) - len(set(rep_graph.values()))
    print(f'[mmseqs] {len(seq_ids)} distinct chains >= {MIN_CHAIN_LEN} aa, {len(hits)} hits; '
          f'clusters {len(set(rep.values()))} ({a.cluster_rule}; sequence links only '
          f'{len(set(rep_seqonly.values()))}, cov08 {len(set(rep_cov08.values()))})', flush=True)
    for c in (c for p in pids for c in chains[p]):
        c['cluster'] = rep.get(seq_ids.get(c['seq']))
    acc = defaultdict(Counter)                        # cluster name: most common accession
    for p in pids:
        for c in chains[p]:
            if c['cluster']:
                acc[c['cluster']].update(c['uniprot'] or ['?'])
    cluster_names = {k: v.most_common(1)[0][0] for k, v in acc.items()}

    labels = load_task_labels()

    def report(sp, with_balance=True):
        r = {'sizes': {k: len(sp[k]) for k in SPLITS},
             'leakage': leakage(sp, chains, seq_ids, hits, complex_h, complex_acc),
             'families': families(sp, complex_clusters, cluster_names)}
        if with_balance:
            r['class_balance'] = class_balance(sp, labels)
        return r
    old_report = {name: report(old[name], name != 'random')
                  for name in ('sequence', 'structure', 'random')}
    cmp_report = {}
    for cname, cpath in compare.items():
        cs = json.loads(cpath.read_text())
        cmp_report[cname] = {'path': _rel(cpath), 'sha256': file_sha256(cpath),
                             **{name: report(cs[name]) for name in NEW_SPLITS}}
    if a.report_only:
        print(json.dumps({'old': old_report, 'compare': cmp_report}, indent=1))
        if out.exists():
            new = json.loads(out.read_text())
            for name in NEW_SPLITS:
                print(name, json.dumps(report(new[name])['leakage']))
        return 0

    # ---- groups and assignment
    seq_groups = groups_by_shared_keys(complex_clusters)
    seq_split, seq_info = assign_groups(seq_groups, fractions, a.seed)
    check_no_straddle(seq_split, seq_groups)
    st_groups, nocath_info = structure_groups(complex_h, complex_clusters)
    st_split, st_info = assign_groups(st_groups, fractions, a.seed)
    check_no_straddle(st_split, st_groups)
    alt_rule = 'cov08' if a.cluster_rule == 'graph' else 'graph'
    alt_cc = cc_of(rep_cov08 if alt_rule == 'cov08' else rep_graph)
    alt_seq_split, alt_info = assign_groups(groups_by_shared_keys(alt_cc), fractions, a.seed)
    alt_st_split, alt_st_info = assign_groups(structure_groups(complex_h, alt_cc)[0],
                                              fractions, a.seed)
    keep = ('n_groups', 'giant_component', 'counts', 'n_groups_forced_train',
            'n_complexes_forced_train')
    alternative = {'cluster_rule': alt_rule, 'meaning': CLUSTER_RULES[alt_rule],
                   'sequence': {**{k: alt_info[k] for k in keep},
                                'leakage': leakage(alt_seq_split, chains, seq_ids, hits,
                                                   complex_h, complex_acc)},
                   'structure': {**{k: alt_st_info[k] for k in keep},
                                 'leakage': leakage(alt_st_split, chains, seq_ids, hits,
                                                    complex_h, complex_acc)}}

    result = {'sequence': seq_split, 'structure': st_split,
              'random': old['random'], 'ligand_disjoint': old['ligand_disjoint'],
              'stats': {'schema': SCHEMA, 'cluster_rule': a.cluster_rule, 'n_pdbs': n,
                        'n_seq_clusters': len(set(rep.values())),
                        'n_seq_groups': seq_info['n_groups'],
                        'n_struct_groups': st_info['n_groups'],
                        'n_cath_labelled': sum(bool(v) for v in complex_h.values()),
                        'n_cath_h_codes': len({h for v in complex_h.values() for h in v})}}
    for name in COPIED_SPLITS:                       # copied verbatim
        assert splits_sha(result[name]) == splits_sha(old[name])
    new_report = {'sequence': report(seq_split), 'structure': report(st_split),
                  'random': report(old['random'])}

    # ---- outputs (atomic writes: chains table, split file, then its sidecar)
    out.parent.mkdir(parents=True, exist_ok=True)
    chains_out = out.with_name(out.stem + CHAINS_OUT_SUFFIX)
    ck = ('start', 'end', 'chain', 'chain_source', 'table_chain', 'cluster', 'cath_h', 'uniprot',
          'uniprot_keys')
    _atomic_json(chains_out, {'schema': SCHEMA + '-chains', 'cluster_rule': a.cluster_rule,
                              'complexes': {
        p: {'chains': [{k: c[k] for k in ck} | {'length': len(c['seq'])} for c in chains[p]],
            'clusters': complex_clusters[p], 'cath_h': complex_h[p]} for p in pids}})
    _atomic_json(out, result, indent=1)
    write_provenance(out, {
        'builder': 'data_prep/build_misato_splits_v2.py', 'schema': SCHEMA,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'seed': a.seed,
        'fractions_target': dict(zip(SPLITS, fractions)),
        'old_splits': {'path': _rel(a.old_splits), 'sha256': file_sha256(a.old_splits),
                       'random_fractions': fr},
        'copied_unchanged': list(COPIED_SPLITS),
        'universe': 'complexes of the old random split', 'n_complexes': n,
        'chains': {'source': 'misato_chain_breaks.json pieces (cut at boundary / unresolved)',
                   'chain_id_rule': ('in-order alignment to the deposited mmCIF '
                                     '(assign_segments), else the chain-break table'),
                   'chain_breaks': {'path': _rel(a.chain_breaks),
                                    'sha256': file_sha256(a.chain_breaks),
                                    'rule_version': tprov.get('rule_version')},
                   'min_chain_len_clustered': MIN_CHAIN_LEN,
                   'n_pieces_below_min_len': sum(len(c['seq']) < MIN_CHAIN_LEN
                                                 for v in chains.values() for c in v),
                   'deposited_files': deposited_record(a.deposited_dir,
                                                       [up[p] for p in pids]),
                   'n_complexes_without_clustered_chain':
                       sum(not v for v in complex_clusters.values()),
                   **chain_info},
        'chains_file': _rel(chains_out),
        'clustering': {'cluster_rule': a.cluster_rule,
                       'meaning': CLUSTER_RULES[a.cluster_rule],
                       'thresholds': {'min_seq_id': MIN_SEQ_ID, 'min_cov_shorter': MIN_COV_SHORT,
                                      'min_aln': MIN_ALN, 'near_id': NEAR_ID,
                                      'near_aln': NEAR_ALN, 'cov08': COV08},
                       'n_clusters': len(set(rep.values())),
                       'n_clusters_graph': len(set(rep_graph.values())),
                       'n_clusters_graph_without_uniprot_links': len(set(rep_seqonly.values())),
                       'n_cluster_merges_by_uniprot_links': n_acc_merges,
                       'n_clusters_cov08': len(set(rep_cov08.values())),
                       'n_sequence_links': len(seq_links), 'n_uniprot_links': len(acc_links),
                       'uniprot': key_info},
        'mmseqs': {'binary': str(MMSEQS_BIN), 'version': mmseqs_version(),
                   'search': 'easy-search ' + ' '.join(map(str, search_args('N'))),
                   'n_distinct_sequences': len(seq_ids), 'n_search_hits': len(hits)},
        'cath': {'sifts_pdb_chain_cath_uniprot': input_record(SIFTS_CATH),
                 'cath_b': input_record(CATH_B),
                 'sifts_pdb_chain_uniprot': input_record(SIFTS_UNIPROT), **cath_info,
                 'n_chains_with_h': sum(bool(c['cath_h']) for v in chains.values() for c in v),
                 'n_complexes_with_h': sum(bool(v) for v in complex_h.values())},
        'assignment_rule': ('groups canonicalised (sorted, by first pid), shuffled with '
                            f'RandomState({a.seed}); groups > {MAX_EVAL_GROUP_FRAC} x '
                            'min(val, test target) -> train; others -> split with the '
                            'largest relative deficit (ties train, val, test)'),
        'sequence': {'unit': 'union-find over shared sequence clusters', **seq_info},
        'structure': {'unit': ('union-find over shared CATH H codes (all chains, all domains); '
                               'complexes without CATH joined through their sequence clusters'),
                      **st_info, 'no_cath': nocath_info},
        'alternative_cluster_rule': alternative,
        'leak_filters': {'nocov_aln50': 'any coverage, >= 50 aligned residues',
                         'cov05_short': '>= 50 % of the shorter chain',
                         'cov08_both': '>= 80 % of both chains'},
        'report_new': new_report, 'report_old': old_report, 'report_compare': cmp_report,
        'splits_sha256': {k: splits_sha(result[k]) for k in (*NEW_SPLITS, *COPIED_SPLITS)},
        'file_sha256': file_sha256(out), 'workdir': _rel(work),
        'workdir_files': {f.name: file_sha256(f) for f in sorted(work.glob('*'))
                          if f.is_file()}})
    for d in work.glob('tmp_*'):                      # MMseqs2 scratch folders
        shutil.rmtree(d, ignore_errors=True)
    for name, info in (('sequence', seq_info), ('structure', st_info)):
        print(f'[{name}] {info["counts"]} groups {info["n_groups"]} '
              f'giant {info["giant_component"]} forced-train {info["n_groups_forced_train"]} '
              f'groups / {info["n_complexes_forced_train"]} complexes')
        for k in ('val', 'test'):
            print(f'   new {k}', json.dumps(new_report[name]['leakage'][k]))
    print('[no-CATH]', nocath_info)
    print(f'wrote {out} (+ provenance, {chains_out.name})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
