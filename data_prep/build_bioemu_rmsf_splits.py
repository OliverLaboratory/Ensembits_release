"""CATH-H structure split of the BioEmu-CATH domains for the BioEmu RMSF probe.

What it does
------------
A NEW split for the BioEmu-CATH RMSF probe (user decision 2026-10-16; not the
tokenizer corpus split). All 1,090 systems of `bioemu_cath_real_bb/layout_rows.json`
are included (also the 18 that the tokenizer corpus held out).

  1. H superfamily of each domain (C.A.T.H of `cath-domain-list.txt`, columns 2–5);
     a domain missing from the list gets its own group "no_cath_<domain>".
  2. Groups = H superfamilies; sorted, shuffled with numpy RandomState(42); the first
     int(0.8 n) groups -> train, the next int(0.1 n) -> val, the rest -> test.
  3. Every system follows its domain, so a domain simulated in both deposits
     (cath1_<d>, cath2_<d>) lands in one split; residue rows follow their system
     (rows = rmsf_bioemu_rows.json / layout_rows.json order).

This is `cluster_split` of the research script `scripts/baselines/build_mdcath_splits.py`
(the structure split of mdCATH-div: H superfamilies, seed 42, 80/10/10 of the groups).
Differences: (a) the H superfamily comes from `cath-domain-list.txt` (the CATH list of
the research script `scripts/baselines/build_misato_splits.py`, under
data/external/StructTokenBench), because the mdCATH hierarchy file used for mdCATH-div
(`data/mdcath/mdcath_cath_hierarchy.json`) covers only 200 of the 1,074 BioEmu domains
(it agrees with cath-domain-list.txt on all 200); (b) the split unit is the domain and
systems follow it; (c) the output also lists residue rows.

The sidecar records how many val / test domains of this split are in the TRAINING
split of the tokenizer corpus (`output/<corpus run>/cached_desc/layout.json` +
`splits.json`, read only), so the overlap can be disclosed.

Paper
-----
`2_results.tex` \\label{result:rmsf} (BioEmu-CATH column, phase 2); the split rule
follows the mdCATH-div structure split (`appendix.tex` \\label{app:dataset}).

Origin
------
New (2026-10-16, phase 2). Procedure of `scripts/baselines/build_mdcath_splits.py`
`cluster_split` (re-implemented here, same steps).

Used by the reported model: yes (BioEmu RMSF probe splits for every method).

Inputs / outputs
----------------
  reads   data/cached_descriptors/bioemu_cath_real_bb/layout_rows.json
          data/external/StructTokenBench/struct_token_bench_release_data/data/CATH/
              cath-classification-data/cath-domain-list.txt          (--cath-list)
          output/bioemu_cath_misato_e0/cached_desc/{layout,splits}.json  (--corpus-dir; overlap only)
  writes  output/phase_b/cached_desc/bioemu/splits.json (+ .provenance.json)
          {train, val, test: residue rows; systems: {...}; domains: {...}}

How to run
----------
    python data_prep/build_bioemu_rmsf_splits.py [--corpus-dir output/<run>/cached_desc]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))
from ensembits.paths import REPO_ROOT, EXTERNAL  # noqa: E402
from ensembits.data_versions import write_provenance  # noqa: E402
from ensembits.featurize import file_sha256  # noqa: E402
from rmsf_common import BIOEMU_LAYOUT, BIOEMU_SPLITS  # noqa: E402

CATH_LIST = (EXTERNAL / 'StructTokenBench' / 'struct_token_bench_release_data' / 'data' / 'CATH'
             / 'cath-classification-data' / 'cath-domain-list.txt')
CORPUS_DIR = REPO_ROOT / 'output' / 'bioemu_cath_misato_e0' / 'cached_desc'
SEED, TRAIN, VAL = 42, 0.8, 0.1


def read_cath_h(path, domains) -> dict:
    """{domain: "C.A.T.H"} for the domains found in a cath-domain-list.txt."""
    want, out = set(domains), {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if p[0] in want:
                out[p[0]] = '.'.join(p[1:5])
    return out


def cluster_split(d_to_cluster: dict, seed: int = SEED, train: float = TRAIN,
                  val: float = VAL) -> dict:
    """Same steps as scripts/baselines/build_mdcath_splits.py cluster_split."""
    clusters = sorted(set(d_to_cluster.values()))
    rng = np.random.RandomState(seed)
    rng.shuffle(clusters)
    n = len(clusters)
    n_tr = int(train * n); n_va = int(val * n)
    train_c = set(clusters[:n_tr]); val_c = set(clusters[n_tr:n_tr + n_va])
    out = {'train': [], 'val': [], 'test': []}
    for d, c in d_to_cluster.items():
        out['train' if c in train_c else 'val' if c in val_c else 'test'].append(d)
    return {k: sorted(v) for k, v in out.items()}


def build(layout_rows: list, cath_h: dict, seed: int = SEED) -> dict:
    """-> splits dict (residue rows, systems, domains, H per domain)."""
    domains = sorted({r['pid'] for r in layout_rows})
    d2h = {d: cath_h.get(d, f'no_cath_{d}') for d in domains}
    dsplit = cluster_split(d2h, seed)
    which = {d: k for k, ds in dsplit.items() for d in ds}
    rows = {'train': [], 'val': [], 'test': []}
    systems = {'train': [], 'val': [], 'test': []}
    start = 0
    for r in layout_rows:
        k = which[r['pid']]
        rows[k].extend(range(start, start + int(r['L'])))
        systems[k].append(r['system'])
        start += int(r['L'])
    return {**rows, 'systems': systems, 'domains': dsplit, 'h_superfamily': d2h,
            'n_rows_total': start}


def corpus_overlap(dsplit: dict, corpus_dir: Path) -> dict:
    """Where the val / test domains of this split sit in the tokenizer corpus split."""
    lay = json.loads((corpus_dir / 'layout.json').read_text())
    sp = json.loads((corpus_dir / 'splits.json').read_text())
    sets = {k: set(sp[k]) for k in ('train', 'val', 'test')}
    where = {}
    for r in lay['rows']:
        if r.get('corpus') != 'bioemu_cath':
            continue
        k = next((k for k in ('train', 'val', 'test') if int(r['start']) in sets[k]), 'none')
        where.setdefault(r['pid'], set()).add(k)
    out = {}
    for part in ('val', 'test'):
        cnt = {'tokenizer_train': 0, 'tokenizer_val': 0, 'tokenizer_test': 0,
               'not_in_corpus': 0}
        for d in dsplit[part]:
            ks = where.get(d)
            if not ks:
                cnt['not_in_corpus'] += 1
            else:
                for k in ks:
                    cnt[f'tokenizer_{k}' if k != 'none' else 'not_in_corpus'] += 1
        out[part] = {'n_domains': len(dsplit[part]), **cnt,
                     'domains_in_tokenizer_train': sorted(d for d in dsplit[part]
                                                          if 'train' in where.get(d, ()))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--layout', default=str(BIOEMU_LAYOUT))
    ap.add_argument('--cath-list', default=str(CATH_LIST))
    ap.add_argument('--corpus-dir', default=str(CORPUS_DIR),
                    help='tokenizer corpus cached_desc folder (layout.json, splits.json); '
                         'overlap report only, skipped if absent')
    ap.add_argument('--out', default=str(BIOEMU_SPLITS))
    ap.add_argument('--seed', type=int, default=SEED)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f'{out} exists; move it aside to rebuild'); return 1
    rows = json.loads(Path(a.layout).read_text())
    h = read_cath_h(a.cath_list, {r['pid'] for r in rows})
    sp = build(rows, h, a.seed)
    cdir = Path(a.corpus_dir)
    overlap = (corpus_overlap(sp['domains'], cdir)
               if (cdir / 'layout.json').exists() and (cdir / 'splits.json').exists() else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sp))
    n_dom = {k: len(v) for k, v in sp['domains'].items()}
    write_provenance(out, {
        'builder': 'data_prep/build_bioemu_rmsf_splits.py', 'layout': a.layout,
        # G4 (2026-10-17): the probes check the split was built from the current layout
        'layout_sha256': file_sha256(a.layout),
        'cath_list': a.cath_list, 'cath_list_sha256': file_sha256(a.cath_list),
        'seed': a.seed, 'fractions': [TRAIN, VAL, round(1 - TRAIN - VAL, 3)],
        'unit': 'CATH H superfamily of the domain; systems follow their domain',
        'n_h_groups': len(set(sp['h_superfamily'].values())),
        'n_domains_without_cath': sum(v.startswith('no_cath_') for v in sp['h_superfamily'].values()),
        'n_domains': n_dom, 'n_systems': {k: len(v) for k, v in sp['systems'].items()},
        'n_rows': {k: len(sp[k]) for k in ('train', 'val', 'test')},
        'corpus_dir': str(cdir), 'tokenizer_corpus_overlap': overlap,
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    print(f'wrote {out}: domains {n_dom}, H groups {len(set(sp["h_superfamily"].values()))}')
    if overlap:
        print(f'  test domains in the tokenizer TRAIN split: {overlap["test"]["tokenizer_train"]} '
              f'of {overlap["test"]["n_domains"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
