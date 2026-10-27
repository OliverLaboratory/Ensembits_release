"""MISATO GO molecular-function labels v2: clean, propagate, write with provenance.

What it does
------------
Builds `data/annotations/misato_go_mf_v2.json` ({PID: sorted MF GO ids}) from the source
annotations (authors' decision D7 (a), 2026-10-18, camera_ready/benchmark_audit.md D7:
molecular function only, DeepFRI style):

  1. Source: `data/annotations/source/misato_go.json` = per PDB entry GO terms (SIFTS,
     union over the entry's chains; MF + BP + CC, unpropagated). It is an in-repo copy of
     the file the research repo used through a symbolic link into another user's
     checkout (its builder is not available); `--copy-source <file>` makes the copy once
     and records origin path, owner, mtime and sha256 in its sidecar. The sha256 must be
     GO_SOURCE_SHA256 (go_labels.py).
  2. Ontology: the pinned go-basic.obo (go_labels.GO_OBO; data-version
     releases/2026-07-26; sha256 GO_OBO_SHA256 checked on every run). `--download-obo`
     fetches it from GO_OBO_URL (GO release archive) and records URL, date and sha256.
  3. Per protein, on the direct terms:
       - alt ids are mapped to their primary id (counted);
       - obsolete terms are removed, except an obsolete term with exactly one
         `replaced_by` that is a live MF term, which is mapped to it (listed);
       - only molecular_function terms are kept;
       - excluded: the MF root GO:0003674, GO:0005515 protein binding (excluded by
         CAFA) and every term of the `gocheck_do_not_annotate` subset.
  4. Propagation: each remaining term adds all its ancestors over is_a and part_of
     edges whose target is an MF term; then the root and the excluded terms are removed
     again (ancestors can be do-not-annotate terms).
  5. Proteins left without any MF term are not written (a missing pid = no MF label).
     The label file does not depend on any split.

Writes the label file atomically and its sidecar (`.provenance.json`: ontology
release / URL / sha256, source sha256 and origin, rules, counts before / after, every
removed or mapped term with its annotation count, sha256 of the output). Refuses to
overwrite an existing label file unless --overwrite.

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{GO Label Prediction}. The paper
currently describes the old labels (union of MF / BP / CC, no propagation).

Origin
------
New (2026-10-18, benchmark_audit D7; the GO auditor's scratch checks
`ontology.py` / `labelsets.py` made a builder).

Used by the reported model: yes (GO rows of every table).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/annotations/source/misato_go.json (+ .provenance.json)
          data/external/go/go-basic_2026-07-26.obo (+ .provenance.json)
  writes  data/annotations/misato_go_mf_v2.json (+ .provenance.json)
          (--copy-source) data/annotations/source/misato_go.json (+ .provenance.json)
          (--download-obo) data/external/go/go-basic_2026-07-26.obo (+ .provenance.json)

How to run
----------
    python data_prep/build_misato_go_labels.py --copy-source data/annotations/misato_go.json \\
        --download-obo                      # once: pin the inputs
    python data_prep/build_misato_go_labels.py [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))   # go_labels
from go_labels import (GO_LABELS_V2, GO_OBO, GO_OBO_DATA_VERSION, GO_OBO_SHA256,  # noqa: E402
                       GO_OBO_URL, GO_SCHEMA, GO_SOURCE, GO_SOURCE_SHA256, file_sha256)

MF = 'molecular_function'
MF_ROOT = 'GO:0003674'
PROTEIN_BINDING = 'GO:0005515'
DO_NOT_ANNOTATE = 'gocheck_do_not_annotate'
RULES = {
    'namespace': 'molecular_function only',
    'alt_id': 'mapped to the primary id',
    'obsolete': 'removed; mapped only when the term has exactly one replaced_by that is a '
                'live molecular_function term',
    'excluded': [f'{MF_ROOT} (MF root)', f'{PROTEIN_BINDING} protein binding (CAFA)',
                 f'every term in subset {DO_NOT_ANNOTATE}'],
    'propagation': 'all ancestors over is_a and part_of edges with an MF target; root and '
                   'excluded terms removed again after propagation',
    'empty': 'proteins without an MF term after cleaning are not written',
}


# ---------------------------------------------------------------- ontology
def parse_obo(path) -> tuple[dict, dict, str | None]:
    """(terms, alt_id -> primary id, data-version) of an OBO file ([Term] stanzas only).

    terms[id] = {'name', 'namespace', 'is_a': set, 'part_of': set, 'obsolete': bool,
                 'replaced_by': list, 'subsets': set}
    """
    terms, alt, cur, version = {}, {}, None, None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('data-version: ') and version is None and not terms:
                version = line.split(': ', 1)[1]
            if line.startswith('['):
                cur = ({'is_a': set(), 'part_of': set(), 'obsolete': False, 'replaced_by': [],
                        'subsets': set(), 'name': '', 'namespace': None}
                       if line == '[Term]' else None)
                continue
            if cur is None or not line or line.startswith('!'):
                continue
            key, _, val = line.partition(': ')
            if key == 'id':
                cur['id'] = val
                terms[val] = cur
            elif key == 'name':
                cur['name'] = val
            elif key == 'namespace':
                cur['namespace'] = val
            elif key == 'is_a':
                cur['is_a'].add(val.split()[0])
            elif key == 'relationship':
                rel, tgt = val.split()[:2]
                if rel == 'part_of':
                    cur['part_of'].add(tgt)
            elif key == 'is_obsolete' and val.strip() == 'true':
                cur['obsolete'] = True
            elif key == 'replaced_by':
                cur['replaced_by'].append(val.split()[0])
            elif key == 'alt_id':
                alt[val.split()[0]] = cur.get('id')
            elif key == 'subset':
                cur['subsets'].add(val.split()[0])
    return terms, alt, version


def mf_ancestors(terms: dict, namespace: str = MF):
    """Function term -> set of its ancestors over is_a + part_of within `namespace`."""
    cache: dict = {}

    def anc(t):
        if t in cache:
            return cache[t]
        cache[t] = set()          # guards against cycles (none expected in GO)
        out = set()
        for par in terms.get(t, {}).get('is_a', set()) | terms.get(t, {}).get('part_of', set()):
            if terms.get(par, {}).get('namespace') != namespace:
                continue
            out.add(par)
            out |= anc(par)
        cache[t] = out
        return out
    return anc


def excluded_terms(terms: dict) -> set:
    """Terms never used as labels: MF root, protein binding, do-not-annotate, obsolete."""
    ex = {MF_ROOT, PROTEIN_BINDING}
    ex |= {t for t, T in terms.items() if DO_NOT_ANNOTATE in T['subsets'] or T['obsolete']}
    return ex


def clean_and_propagate(source: dict, terms: dict, alt: dict) -> tuple[dict, dict]:
    """(pid -> sorted MF labels, report) following RULES (pids without labels omitted)."""
    anc = mf_ancestors(terms)
    excl = excluded_terms(terms)
    rep = {'alt_mapped': Counter(), 'obsolete_mapped': {}, 'obsolete_removed': Counter(),
           'not_in_ontology': Counter(), 'non_mf': Counter(), 'excluded_direct': Counter(),
           'excluded_after_propagation': Counter()}
    out = {}
    for pid, lst in source.items():
        direct = set()
        for t in (lst if isinstance(lst, list) else []):
            if t in alt and t not in terms:
                rep['alt_mapped'][f'{t}->{alt[t]}'] += 1
                t = alt[t]
            T = terms.get(t)
            if T is None:
                rep['not_in_ontology'][t] += 1
                continue
            if T['obsolete']:
                r = T['replaced_by']
                if (len(r) == 1 and r[0] in terms and not terms[r[0]]['obsolete']
                        and terms[r[0]]['namespace'] == MF):
                    rep['obsolete_mapped'].setdefault(t, {'to': r[0], 'n': 0})['n'] += 1
                    t, T = r[0], terms[r[0]]
                else:
                    rep['obsolete_removed'][t] += 1
                    continue
            if T['namespace'] != MF:
                rep['non_mf'][T['namespace']] += 1
                continue
            if t in excl:
                rep['excluded_direct'][t] += 1
                continue
            direct.add(t)
        full = set(direct)
        for t in direct:
            full |= anc(t)
        for t in full & excl:
            rep['excluded_after_propagation'][t] += 1
        full -= excl
        if full:
            out[str(pid).upper()] = sorted(full)
    return out, rep


# ---------------------------------------------------------------- pinned inputs
def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def _sidecar(path) -> Path:
    return Path(str(path) + '.provenance.json')


def copy_source(src: Path, overwrite: bool) -> int:
    """Copy the source annotation file into the repo (records origin in the sidecar)."""
    real = Path(src).resolve()
    sha = file_sha256(real)
    if sha != GO_SOURCE_SHA256:
        print(f'[error] {real}: sha256 {sha} != pinned {GO_SOURCE_SHA256}'); return 1
    if GO_SOURCE.exists() and not overwrite:
        if file_sha256(GO_SOURCE) == sha:
            print(f'[source] {GO_SOURCE} already holds this file'); return 0
        print(f'[error] {GO_SOURCE} exists with other content; pass --overwrite'); return 1
    GO_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    tmp = GO_SOURCE.with_name('.' + GO_SOURCE.name + '.tmp')
    shutil.copyfile(real, tmp)
    os.replace(tmp, GO_SOURCE)
    st = real.stat()
    try:
        owner = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        owner = str(st.st_uid)
    _atomic_write(_sidecar(GO_SOURCE), json.dumps({
        'what': 'per PDB entry GO terms (SIFTS; union over the chains of the entry; MF + BP '
                '+ CC, not propagated); builder not available',
        'origin_path': str(real), 'origin_given_as': str(src), 'origin_owner': owner,
        'origin_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
        'sha256': sha, 'copied': time.strftime('%Y-%m-%d %H:%M:%S'),
        'copied_by': 'data_prep/build_misato_go_labels.py --copy-source'}, indent=1))
    print(f'[source] copied {real} -> {GO_SOURCE} (sha256 {sha[:12]})')
    return 0


def download_obo(overwrite: bool) -> int:
    """Fetch the pinned go-basic.obo from the GO release archive (sha256 checked)."""
    if GO_OBO.exists() and not overwrite:
        if file_sha256(GO_OBO) == GO_OBO_SHA256:
            print(f'[obo] {GO_OBO} already present'); return 0
        print(f'[error] {GO_OBO} exists with other content; pass --overwrite'); return 1
    GO_OBO.parent.mkdir(parents=True, exist_ok=True)
    tmp = GO_OBO.with_name('.' + GO_OBO.name + '.part')
    # The archive rejects urllib's default User-Agent (HTTP 403).
    req = urllib.request.Request(GO_OBO_URL, headers={'User-Agent': 'curl/8.0'})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f)
    sha = file_sha256(tmp)
    if sha != GO_OBO_SHA256:
        tmp.unlink()
        print(f'[error] downloaded sha256 {sha} != pinned {GO_OBO_SHA256}'); return 1
    os.replace(tmp, GO_OBO)
    _atomic_write(_sidecar(GO_OBO), json.dumps({
        'url': GO_OBO_URL, 'data_version': GO_OBO_DATA_VERSION, 'sha256': sha,
        'downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'GO release archive snapshot 2026-08-05; the ontology inside is '
                'releases/2026-07-26 (same bytes as the purl on 2026-10-18)'}, indent=1))
    print(f'[obo] downloaded {GO_OBO_URL} -> {GO_OBO}')
    return 0


def _counts(labels: dict) -> dict:
    lists = [v for v in labels.values() if isinstance(v, list)]
    return {'proteins': len(lists), 'proteins_with_terms': sum(1 for v in lists if v),
            'annotations': sum(len(set(v)) for v in lists),
            'unique_terms': len({t for v in lists for t in v})}


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--copy-source', default=None,
                    help='copy this annotation file into data/annotations/source/ first')
    ap.add_argument('--download-obo', action='store_true',
                    help='fetch the pinned go-basic.obo first')
    ap.add_argument('--out', default=str(GO_LABELS_V2))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)
    if args.copy_source and copy_source(Path(args.copy_source), args.overwrite):
        return 1
    if args.download_obo and download_obo(args.overwrite):
        return 1
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f'{out} exists; pass --overwrite to rebuild (every GO result built on it must '
              f'then be rerun)'); return 1
    for p, pin in ((GO_SOURCE, GO_SOURCE_SHA256), (GO_OBO, GO_OBO_SHA256)):
        if not p.exists():
            print(f'[error] {p} missing (use --copy-source / --download-obo)'); return 1
        if file_sha256(p) != pin:
            print(f'[error] {p}: sha256 differs from the pin {pin[:12]}'); return 1
    terms, alt, version = parse_obo(GO_OBO)
    if version != GO_OBO_DATA_VERSION:
        print(f'[error] {GO_OBO}: data-version {version} != {GO_OBO_DATA_VERSION}'); return 1
    source = json.loads(GO_SOURCE.read_text())
    labels, rep = clean_and_propagate(source, terms, alt)

    def named(cnt):
        return {t: {'n': n, 'name': terms[t]['name'] if t in terms else None,
                    'namespace': terms[t]['namespace'] if t in terms else None}
                for t, n in sorted(cnt.items())}
    src_terms = Counter(t for v in source.values() if isinstance(v, list) for t in set(v))
    mf_direct = {p: sorted({t for t in v if terms.get(t, {}).get('namespace') == MF})
                 for p, v in source.items() if isinstance(v, list)}
    text = json.dumps(labels, sort_keys=True, separators=(',', ':'))
    _atomic_write(out, text)
    src_prov = _sidecar(GO_SOURCE)
    obo_prov = _sidecar(GO_OBO)
    prov = {
        'builder': 'data_prep/build_misato_go_labels.py', 'schema': GO_SCHEMA,
        'output': out.name, 'sha256': file_sha256(out),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': {'file': 'data/annotations/source/misato_go.json', 'sha256': GO_SOURCE_SHA256,
                   'provenance': json.loads(src_prov.read_text()) if src_prov.exists() else None},
        'ontology': {'file': f'data/external/go/{GO_OBO.name}', 'url': GO_OBO_URL,
                     'data_version': version, 'sha256': GO_OBO_SHA256,
                     'provenance': json.loads(obo_prov.read_text()) if obo_prov.exists() else None},
        'rules': RULES,
        'counts': {
            'source': {**_counts(source),
                       'unique_terms_by_namespace': dict(Counter(
                           terms[t]['namespace'] if t in terms else 'unknown' for t in src_terms))},
            'source_mf_direct': _counts(mf_direct),
            'output': {**_counts(labels),
                       'source_proteins_without_mf_label': len(source) - len(labels)},
        },
        'alt_ids_mapped': dict(rep['alt_mapped']),
        'obsolete_mapped': {t: {**v, 'name': terms[t]['name'],
                                'to_name': terms[v['to']]['name']}
                            for t, v in sorted(rep['obsolete_mapped'].items())},
        'obsolete_removed': named(rep['obsolete_removed']),
        'not_in_ontology': dict(rep['not_in_ontology']),
        'non_mf_annotations_removed': dict(rep['non_mf']),
        'excluded_direct': named(rep['excluded_direct']),
        'excluded_after_propagation': named(rep['excluded_after_propagation']),
    }
    _atomic_write(_sidecar(out), json.dumps(prov, indent=1, sort_keys=True))
    print(json.dumps(prov['counts'], indent=1))
    print(f'obsolete removed {len(rep["obsolete_removed"])} terms, mapped '
          f'{len(rep["obsolete_mapped"])}; excluded direct {dict(rep["excluded_direct"])}')
    print(f'wrote {out} (+ .provenance.json)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
