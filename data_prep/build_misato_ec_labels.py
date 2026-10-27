"""MISATO EC labels v2: polyprotein chains labelled by the mature products they cover.

What it does
------------
The EC label of a MISATO complex was the union of the SIFTS EC numbers of all chains of
its PDB entry (`data/annotations/misato_ec.json`, built outside this codebase; its builder
is lost). SIFTS gives a chain every EC number of its UniProt entry, so a chain of a viral
polyprotein inherits the EC numbers of the whole precursor: an HIV-1 protease dimer (99
residues per chain) carries the Gag-Pol set (reverse transcriptase 2.7.7.49 / 2.7.7.7,
RNase H 3.1.26.13 / 3.1.13.2, integrase 2.7.7.- / 3.1.-.-, protease 3.4.23.16), an HCV NS5B
polymerase carries the NS3 protease / helicase numbers, and so on. This builder
(authors' decision D8, 2026-10-18):

  1. copies the source file into the repo (`data/annotations/source/misato_ec.json`,
     sidecar: origin path, sha256, size, mtime);
  2. reads SIFTS `pdb_chain_uniprot` (chain -> UniProt residue ranges) and
     `pdb_chain_enzyme` (chain -> EC), and the UniProt entries of every accession that
     SIFTS maps to any MISATO PDB entry (UniProt REST, fields: protein names with the
     "Contains" section and EC numbers, Chain / Peptide features, sequence);
  3. scans ALL MISATO complexes for polyprotein entries (>= 2 Chain features or a
     Contains section). A polyprotein is "component-resolved" when at least one
     Contains component carries an EC number (viral polyproteins, ricin, sucrase-
     isomaltase, ...); only those are relabelled. Polyproteins whose EC numbers sit on the
     entry name only (prothrombin, trypsin, factor X, ...: zymogens cut into chains of one
     enzyme) keep their SIFTS EC numbers, as do all other entries;
  4. for each labelled complex with a chain mapped to a component-resolved entry, finds
     the protein chains actually present in the MISATO complex (pieces between the
     `boundary` breaks of `misato_chain_breaks.json` v2.3, chain id from the break table
     or by alignment to the deposited mmCIF, `build_misato_splits_v2.complex_chains`; a
     piece still without a chain id is aligned to the UniProt segments of the SIFTS
     chains) and gives each such chain the EC numbers of the mature products it covers:

       * mature products = Chain / Peptide features of the entry except the precursor
         itself (a feature named like the entry); the EC numbers of a product are those
         of the Contains component of the same name; a product that matches no
         component is a piece of the entry's own protein (e.g. the two exteins of an
         intein-containing ATPase) and carries the entry-level EC numbers;
       * residues of the precursor that lie in no mature product carry the entry-level
         EC numbers (precursor-only region; e.g. the autoprocessing part of a toxin);
       * a product (or the precursor-only region) counts when it overlaps the chain's
         SIFTS-mapped UniProt residues by >= OVERLAP_MIN_RES (30) residues or by >=
         OVERLAP_MIN_FRAC (50 %) of the chain's mapped residues. Justification: a
         construct often carries a few residues of the neighbouring product (cleavage
         sites, linkers; < 30 residues in every MISATO case), which must not add that
         product's activity; 30 residues are below the smallest enzymatic product in the
         set (HIV-1 protease, 99 residues) yet enough for any real domain; the 50 % arm
         keeps short peptide chains (< 60 residues) that lie inside one product;
       * the complex label = (source label minus the EC numbers that only the relabelled
         chains explained) union (EC numbers of the present relabelled chains). "Only
         explained" = every EC number of the component-resolved entries mapped in the
         complex and the SIFTS EC numbers of their chains, minus the SIFTS EC numbers of
         the other (non-polyprotein) chains of the entry. Complexes without a
         component-resolved chain keep the source label unchanged;
  4b. homology fallback (follow-up to D8, 2026-10-19), same intent: a chain whose UniProt
     entry is not a resolvable polyprotein but whose EC set (entry EC + SIFTS chain EC)
     holds EC numbers of >= 2 distinct mature products of a reviewed (Swiss-Prot)
     polyprotein (TrEMBL pol fragments such as Q90K99 / Q72498 / Q90JJ9 / A0F6C5 / Q90SN9
     giving a protease-only chain RNase H / RT numbers), and every chain of a labelled
     complex with no current SIFTS mapping whose source label holds such numbers (1B6J),
     is aligned (MISATO piece sequence; for a chain not in the MISATO complex its SIFTS
     UniProt segment) to the mature products of the reviewed resolvable polyproteins. A
     product matches with >= HOMOLOGY_MIN_IDENTITY (85 %; 90 % until 2026-10-19) identity over >=
     HOMOLOGY_MIN_COVERAGE (80 %) of the chain (of the product when the chain is longer)
     and >= HOMOLOGY_MIN_ALIGNED (20) aligned residues; the chain is then aligned to the
     whole reference precursor, the overlap rule above gives its EC numbers, and they
     replace the chain's SIFTS EC numbers (recorded with source "homology_fallback", the
     reference accession, identity and coverage). Without a match the chain is unchanged;
  4c. domain overrides (authors' decision, 2026-10-19; table DOMAIN_OVERRIDES, version
     DOMAIN_OVERRIDE_VERSION), applied after steps 4 and 4b: a construct that contains only
     one domain of a multi-domain protein is labelled with the EC numbers of that domain,
     and the EC numbers of the domains it does not contain are removed. SIFTS gives a chain
     every EC number of its UniProt entry (or of the polyprotein component it lies in), so
     an HCV NS3 protease-domain construct carries the helicase EC numbers, a flavivirus NS5
     methyltransferase-domain construct the polymerase EC number, a GART-domain construct of
     the trifunctional purine enzyme GARS and AIRS, and so on. Rule: a domain counts as
     contained when the chain's UniProt residues (SIFTS; the RCSB alignment for 2Q6F, which
     has no SIFTS mapping) cover >= OVERRIDE_CONTAINED_MIN (90 %) of its UniProt Domain /
     Region / Zinc finger feature, and as absent below OVERRIDE_ABSENT_MAX (50 %); every EC
     number removed must belong to an absent domain. A construct whose only domain has no
     catalytic activity (TAF1 bromodomain 2, HDAC6 ZnF-UBP) gets an empty label. Each entry
     is checked against the label of steps 4 / 4b (its `old_ec`); a mismatch stops the build,
     so a stale entry is never applied silently. The table was compiled by hand from the GO +
     EC label re-audit (camera_ready/function_reaudit.md, auditor 5) and the UniProt 2026_03
     features of each accession; the overrides are recorded in the sidecar
     ("domain_overrides", and "domain_override" in each changed complex);
  5. writes `data/annotations/misato_ec_v2.json` ({PID: sorted EC list}, same keys as the
     source; a complex whose label becomes empty keeps an empty list, which the loaders
     skip) and its sidecar: inputs (path, sha256, url, date, UniProt release), rules,
     thresholds, polyproteins found (all MISATO complexes), changed complexes (old / new
     label, per-chain products and overlaps), class counts per EC depth before / after
     (with the wildcard rule of `truncate_ec`), and open issues.

Wildcard EC numbers ("3.1.-.-", "2.7.7.-") are kept in the label file as SIFTS gives them;
the downstream `truncate_ec` (`downstream/tokenize_misato_and_eval.py`,
`downstream/baselines/ec_classify_conv1d_misato.py`) uses one only at depths whose fields
are all specified (`truncate_ec` below is a copy used for the class counts).

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{EC Function Prediction} (EC labels of
MISATO complexes, depths 1-3); `2_results.tex` \\label{sec:downstreams}.

Origin
------
New (2026-10-18, authors' decision D8 of the EC audit). Replaces the lost builder of
`data/annotations/misato_ec.json` for the polyprotein chains; the other labels are the
source labels. FIXED(function_reaudit GO + EC labels, auditor 5, 2026-10-19): step 4c, the
domain override table (30 single-domain constructs); old behaviour = labels of steps 4 / 4b
only (sha256 5092b9ff... of misato_ec_v2.json).

Used by the reported model: yes (EC probes of every method read misato_ec_v2.json).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/annotations/misato_ec.json                 (source; symlink, copied once)
          data/external/sifts/pdb_chain_uniprot.tsv.gz      (EBI SIFTS; --download)
          data/external/sifts/pdb_chain_enzyme.tsv.gz       (EBI SIFTS; --download)
          data/external/sifts/_manifest.json                (url, date, Last-Modified, sha256)
          data/external/uniprot/misato_sifts_entries.json.gz  (UniProt REST; --download)
          data/external/uniprot/_manifest.json
          data/cached_descriptors/misato_chain_breaks.json  (v2.3; pieces, chain ids)
          data/cached_descriptors/misato_real_bb/<PID>.npz  (sequence)
          data/external/rcsb_mmcif/<PID>.cif.gz             (chain ids by alignment)
  writes  data/annotations/source/misato_ec.json (+ .provenance.json)   (first run only)
          data/annotations/misato_ec_v2.json (+ .provenance.json)
Refuses to overwrite (--overwrite replaces the v2 file); atomic writes.

How to run
----------
    python data_prep/build_misato_ec_labels.py --download     # first run: fetch SIFTS / UniProt
    python data_prep/build_misato_ec_labels.py [--workers 16] [--overwrite]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))               # rmsf_common
sys.path.insert(0, str(CODE_ROOT / 'data_prep'))                # build_misato_splits_v2
sys.path.insert(0, str(CODE_ROOT / 'data_prep' / 'baselines'))  # _chain_pieces
from ensembits.paths import (REPO_ROOT, EXTERNAL, MISATO_EC_LABELS,  # noqa: E402
                             MISATO_EC_LABELS_LEGACY)
from ensembits.featurize import file_sha256  # noqa: E402

SCHEMA = 'misato-ec-v3'
OVERLAP_MIN_RES = 30          # residues of a mature product the chain must cover ...
OVERLAP_MIN_FRAC = 0.5        # ... or this fraction of the chain's mapped residues
SOURCE_COPY = REPO_ROOT / 'data' / 'annotations' / 'source' / 'misato_ec.json'
SIFTS_DIR = EXTERNAL / 'sifts'
SIFTS_UNIPROT = SIFTS_DIR / 'pdb_chain_uniprot.tsv.gz'
SIFTS_ENZYME = SIFTS_DIR / 'pdb_chain_enzyme.tsv.gz'
SIFTS_BASE = 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/'
UNIPROT_DIR = EXTERNAL / 'uniprot'
UNIPROT_ENTRIES = UNIPROT_DIR / 'misato_sifts_entries.json.gz'
UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/accessions'
UNIPROT_FIELDS = 'accession,protein_name,ec,ft_chain,ft_peptide,sequence,organism_name,length'
UNIPROT_BATCH = 100
PRODUCT_TYPES = ('Chain', 'Peptide')
MIN_PIECE_IDENTICAL = 10      # UniProt-alignment fallback: identical residues ...
MIN_PIECE_IDENTITY = 0.5      # ... and fraction of the piece
HOMOLOGY_MIN_IDENTITY = 0.85  # homology fallback: identity over the aligned columns ...
HOMOLOGY_MIN_COVERAGE = 0.8   # ... covering this fraction of the chain (of the product if shorter)
HOMOLOGY_MIN_ALIGNED = 20     # ... and at least this many aligned residues (no short peptides)


# ---------------------------------------------------------------- small helpers
def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj, indent=None) -> None:
    _atomic_bytes(path, json.dumps(obj, indent=indent, sort_keys=True, default=str).encode())


def label_sha256(path) -> str:
    """sha256 of a label file (the value recorded in the result JSONs)."""
    return file_sha256(path)


def truncate_ec(ec, depth):
    """Same rule as the downstream truncate_ec (copy for the class counts of the sidecar)."""
    parts = str(ec).strip().split('.')
    if len(parts) < depth:
        return None
    head = parts[:depth]
    if any(p in ('-', '') for p in head):
        return None
    return '.'.join(head)


def truncate_ec_legacy(ec, depth):
    """The old downstream rule ('3.1.-.-' at depth 3 -> class '3.1.-'), for the counts."""
    parts = str(ec).split('.')
    return None if len(parts) < depth else '.'.join(parts[:depth])


def class_counts(labels: dict, trunc) -> dict:
    """{depth: {n_classes, n_complexes, n_wildcard_classes}} over all labelled complexes."""
    out = {}
    for d in (1, 2, 3):
        cls, n = set(), 0
        for v in labels.values():
            s = {t for e in v if (t := trunc(e, d)) is not None}
            cls |= s
            n += bool(s)
        out[str(d)] = {'n_classes': len(cls), 'n_complexes_with_label': n,
                       'n_classes_with_dash': sum('-' in c for c in cls)}
    return out


# ---------------------------------------------------------------- inputs
def copy_source(origin: Path = MISATO_EC_LABELS_LEGACY, dest: Path = SOURCE_COPY) -> dict:
    """Copy the source EC file into the repo once; refuse a differing existing copy."""
    origin = Path(origin)
    data = origin.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if dest.exists():
        if file_sha256(dest) != sha:
            raise SystemExit(f'{_rel(dest)} exists and differs from {origin} (sha256); '
                             f'move it aside to copy again')
        prov = json.loads(Path(str(dest) + '.provenance.json').read_text())
        return prov
    st = origin.stat()
    prov = {'origin_path': str(origin), 'origin_resolved': str(origin.resolve()),
            'sha256': sha, 'size': len(data),
            'origin_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
            'copied': time.strftime('%Y-%m-%d %H:%M:%S'),
            'what': 'per-PDB SIFTS EC numbers, union over all chains of the entry '
                    '(builder not available); MISATO EC labels before 2026-10-18'}
    _atomic_bytes(dest, data)
    _atomic_json(Path(str(dest) + '.provenance.json'), prov, indent=1)
    return prov


def _manifest_update(path: Path, rec: dict) -> None:
    man_p = path.parent / '_manifest.json'
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    man[path.name] = rec
    _atomic_json(man_p, man, indent=1)


def _manifest_get(path: Path) -> dict:
    man_p = path.parent / '_manifest.json'
    return json.loads(man_p.read_text()).get(path.name, {}) if man_p.exists() else {}


def download_sifts(path: Path, timeout: float = 300.0) -> dict:
    """Fetch one SIFTS flat file if absent (never overwrites); record it in _manifest.json."""
    if path.exists():
        return input_record(path)
    url = SIFTS_BASE + path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f)
        last_mod = r.headers.get('Last-Modified')
    os.replace(tmp, path)
    _manifest_update(path, {'url': url, 'downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'last_modified': last_mod, 'sha256': file_sha256(path)})
    return input_record(path)


def sifts_header(path: Path) -> str | None:
    """First comment line of a SIFTS flat file (release date, PDB / UniProt versions)."""
    with gzip.open(path, 'rt') as f:
        line = f.readline()
    return line.strip().lstrip('#').strip() if line.startswith('#') else None


def input_record(path: Path) -> dict:
    man = _manifest_get(path)
    sha = file_sha256(path)
    rec = {'path': _rel(path), 'sha256': sha, 'url': man.get('url'),
           'downloaded': man.get('downloaded'), 'last_modified': man.get('last_modified'),
           'manifest_sha256_matches': (man.get('sha256') == sha) if man else None}
    for k in ('uniprot_release', 'uniprot_release_date', 'fields', 'n_entries', 'n_requested',
              'missing'):
        if k in man:
            rec[k] = man[k]
    if path.name.endswith('.tsv.gz'):
        rec['header'] = sifts_header(path)
    return rec


def read_sifts_rows(path: Path, pdbs: set) -> list[list[str]]:
    """Rows (list of fields after PDB) of a SIFTS flat file for the given PDB ids (upper)."""
    out = []
    with gzip.open(path, 'rt') as f:
        header = None
        for line in f:
            if line.startswith('#'):
                continue
            p = line.rstrip('\n').split('\t')
            if header is None:
                header = p
                continue
            if p[0].upper() in pdbs:
                out.append(p)
    return out


def sifts_chain_uniprot(path: Path, pdbs: set) -> dict:
    """{PID: {chain: [(accession, sp_beg, sp_end), ...]}} (isoform suffix dropped)."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for p in read_sifts_rows(path, pdbs):
        # PDB CHAIN SP_PRIMARY RES_BEG RES_END PDB_BEG PDB_END SP_BEG SP_END
        try:
            beg, end = int(p[7]), int(p[8])
        except ValueError:
            continue
        out[p[0].upper()][p[1]].append((p[2].split('-')[0], beg, end))
    return {k: dict(v) for k, v in out.items()}


def sifts_chain_enzyme(path: Path, pdbs: set) -> dict:
    """{PID: {chain: {accession: set(EC)}}}."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for p in read_sifts_rows(path, pdbs):
        # PDB CHAIN ACCESSION EC_NUMBER
        out[p[0].upper()][p[1]][p[2].split('-')[0]].add(p[3])
    return {k: {c: dict(a) for c, a in v.items()} for k, v in out.items()}


def download_uniprot(accessions, path: Path = UNIPROT_ENTRIES, timeout: float = 120.0) -> dict:
    """Fetch the UniProt entries (REST, UNIPROT_FIELDS) of the accessions if `path` is absent."""
    if path.exists():
        return input_record(path)
    accs = sorted(set(accessions))
    entries, rel, rel_date = {}, None, None
    for i in range(0, len(accs), UNIPROT_BATCH):
        url = (f'{UNIPROT_URL}?accessions={",".join(accs[i:i + UNIPROT_BATCH])}'
               f'&fields={UNIPROT_FIELDS}&format=json&size={UNIPROT_BATCH}')
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    rel = r.headers.get('X-UniProt-Release') or rel
                    rel_date = r.headers.get('X-UniProt-Release-Date') or rel_date
                    d = json.load(r)
                break
            except Exception as e:                     # transient network errors
                if attempt == 4:
                    raise
                print(f'[uniprot] retry {attempt + 1}: {type(e).__name__}: {e}', flush=True)
                time.sleep(5 * (attempt + 1))
        for e in d.get('results', []):
            entries[e['primaryAccession']] = e
        print(f'[uniprot] {min(i + UNIPROT_BATCH, len(accs))}/{len(accs)}', flush=True)
    missing = [a for a in accs if a not in entries]
    obj = {'uniprot_release': rel, 'uniprot_release_date': rel_date, 'fields': UNIPROT_FIELDS,
           'requested': accs, 'missing': missing, 'entries': entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    with gzip.open(tmp, 'wt') as f:
        json.dump(obj, f, sort_keys=True)
    os.replace(tmp, path)
    _manifest_update(path, {'url': UNIPROT_URL + '?accessions=<batch of 100>&fields='
                                   + UNIPROT_FIELDS + '&format=json',
                            'downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'uniprot_release': rel, 'uniprot_release_date': rel_date,
                            'fields': UNIPROT_FIELDS, 'n_requested': len(accs),
                            'n_entries': len(entries), 'missing': missing,
                            'sha256': file_sha256(path)})
    return input_record(path)


def load_uniprot(path: Path = UNIPROT_ENTRIES) -> dict:
    with gzip.open(path, 'rt') as f:
        return json.load(f)


# ---------------------------------------------------------------- UniProt products
def _names(n) -> list[str]:
    if not n:
        return []
    out = [n['fullName']['value']] if 'fullName' in n else []
    return out + [s['value'] for s in n.get('shortNames', [])]


def _ecs(n) -> list[str]:
    return [x['value'] for x in (n or {}).get('ecNumbers', [])]


def _all_names(desc: dict) -> list[str]:
    return _names(desc.get('recommendedName')) + [x for a in desc.get('alternativeNames', [])
                                                  for x in _names(a)]


def _all_ecs(desc: dict) -> list[str]:
    return _ecs(desc.get('recommendedName')) + [x for a in desc.get('alternativeNames', [])
                                                for x in _ecs(a)]


def entry_products(entry: dict) -> dict:
    """Polyprotein status and mature products of one UniProt entry (JSON of the REST API).

    -> {polyprotein, component_resolved, resolvable, entry_ec, name, organism, length,
        products: [{name, type, start, end, ec, component}], precursor: [names],
        unmatched_ec_components: [names]}
    """
    pd = entry.get('proteinDescription', {})
    contains = pd.get('contains', [])
    comps = [([n.lower() for n in _all_names(c)], sorted(set(_all_ecs(c))), _all_names(c)[:1])
             for c in contains]
    entry_ec = sorted(set(_all_ecs(pd)))
    entry_names = [n.lower() for n in _all_names(pd)]
    feats = [f for f in entry.get('features', []) if f.get('type') in PRODUCT_TYPES]
    n_chain = sum(f['type'] == 'Chain' for f in feats)
    products, precursor, matched_comp = [], [], set()
    for f in feats:
        try:
            s = int(f['location']['start']['value'])
            t = int(f['location']['end']['value'])
        except (KeyError, TypeError, ValueError):
            continue                                      # unknown boundary
        d = (f.get('description') or '').strip()
        hit = [i for i, (nm, _, _) in enumerate(comps) if d.lower() in nm]
        if not hit and d.lower() in entry_names:
            precursor.append(d)                           # the precursor itself
            continue
        if hit:
            matched_comp.add(hit[0])
            ec, comp = comps[hit[0]][1], (comps[hit[0]][2] or [d])[0]
        else:
            ec, comp = entry_ec, None                     # a piece of the entry's own protein
        products.append({'name': d, 'type': f['type'], 'start': s, 'end': t,
                         'ec': list(ec), 'component': comp})
    component_resolved = any(ec for _, ec, _ in comps)
    unmatched = [(c[2] or ['?'])[0] for i, c in enumerate(comps) if c[1] and i not in matched_comp]
    return {'polyprotein': n_chain >= 2 or bool(contains),
            'component_resolved': component_resolved,
            # every EC-carrying component must be located by a feature
            'resolvable': component_resolved and not unmatched and bool(products),
            'entry_ec': entry_ec, 'name': (_names(pd.get('recommendedName'))
                                           or _names((pd.get('submissionNames') or [{}])[0])
                                           or [None])[0],
            'organism': (entry.get('organism') or {}).get('scientificName'),
            'entry_type': entry.get('entryType'),
            'length': (entry.get('sequence') or {}).get('length'),
            'n_chain_features': n_chain, 'n_contains': len(contains),
            'products': products, 'precursor': precursor, 'unmatched_ec_components': unmatched,
            'all_ec': sorted(set(entry_ec) | {e for _, ec, _ in comps for e in ec})}


def chain_mature_ec(rows, info_of: dict) -> tuple[set, list]:
    """EC numbers of one PDB chain mapped to component-resolved entries.

    rows: [(accession, sp_beg, sp_end)] of the chain (SIFTS); info_of: accession ->
    entry_products(); only accessions with info_of[acc]['resolvable'] are used here.
    -> (EC set, [{accession, product, overlap, ec}]).
    """
    pos_by_acc: dict = defaultdict(set)
    for acc, b, e in rows:
        pos_by_acc[acc].update(range(int(b), int(e) + 1))
    total = sum(len(v) for v in pos_by_acc.values())
    ec, det = set(), []
    for acc, pos in pos_by_acc.items():
        info = info_of.get(acc)
        if not info or not info['resolvable']:
            continue
        in_products = set()
        for p in info['products']:
            span = set(range(p['start'], p['end'] + 1))
            in_products |= span
            ov = len(pos & span)
            if ov and (ov >= OVERLAP_MIN_RES or ov >= OVERLAP_MIN_FRAC * total):
                ec |= set(p['ec'])
                det.append({'accession': acc, 'product': p['name'], 'overlap': ov,
                            'ec': list(p['ec'])})
        rest = pos - in_products                          # precursor-only residues
        ov = len(rest)
        if ov and info['entry_ec'] and (ov >= OVERLAP_MIN_RES or ov >= OVERLAP_MIN_FRAC * total):
            ec |= set(info['entry_ec'])
            det.append({'accession': acc, 'product': '(precursor-only region)', 'overlap': ov,
                        'ec': list(info['entry_ec'])})
    return ec, det


def relabel_complex(source_ec, sifts_chains: dict, enzyme_chains: dict, present, info_of: dict,
                    fallback: dict | None = None):
    """New label of one complex.

    source_ec: the source label; sifts_chains: {chain: [(acc, beg, end)]} of the PDB entry;
    enzyme_chains: {chain: {acc: set(EC)}} (SIFTS pdb_chain_enzyme); present: chain ids of
    the protein chains in the MISATO complex (None = all SIFTS chains); info_of: accession
    -> entry_products(); fallback: {key: {'sifts_chain': chain id or None, 'present': bool,
    'ec': EC of the matched mature products, 'ref_all_ec': every EC of the reference
    polyprotein, ...}} of the chains labelled by homology (homology_fallback; their SIFTS EC
    numbers are replaced). -> (new sorted list, detail dict) or (None, None) when the complex
    has neither a chain mapped to a resolvable component-resolved entry nor a fallback chain
    (label unchanged).
    """
    fallback = fallback or {}
    cr = lambda acc: bool(info_of.get(acc) and info_of[acc]['resolvable'])
    cr_chains = {c for c, rows in sifts_chains.items() if any(cr(a) for a, _, _ in rows)}
    if not cr_chains and not fallback:
        return None, None
    fb_sifts = {v['sifts_chain'] for v in fallback.values() if v.get('sifts_chain')}
    present = set(sifts_chains) if present is None else set(present)
    explained, other = set(), set()
    for c, rows in sifts_chains.items():
        for acc, ecs in (enzyme_chains.get(c) or {}).items():
            (explained if (cr(acc) or c in fb_sifts) else other).update(ecs)
        for acc in {a for a, _, _ in rows if cr(a)}:
            explained |= set(info_of[acc]['all_ec'])
    for v in fallback.values():
        explained |= set(v['ref_all_ec'])
    only_cr = explained - other
    kept = set(source_ec) - only_cr
    chains = {}
    mature = set()
    for c in sorted(cr_chains):
        rows = sifts_chains[c]
        ec, det = chain_mature_ec(rows, info_of)
        chains[c] = {'present': c in present, 'sifts': [list(r) for r in rows],
                     'ec': sorted(ec), 'products': det}
        if c in present:
            mature |= ec
    for v in fallback.values():
        if v['present']:
            mature |= set(v['ec'])
    new = sorted(kept | mature)
    return new, {'old': sorted(set(source_ec)), 'new': new, 'kept_from_source': sorted(kept),
                 'removed': sorted(set(source_ec) - set(new)),
                 'added': sorted(set(new) - set(source_ec)),
                 'cr_chains': chains,
                 'homology_chains': {k: {**v, 'ec': sorted(v['ec']),
                                         'ref_all_ec': sorted(v['ref_all_ec'])}
                                     for k, v in fallback.items()},
                 'absent_cr_chain_ec': sorted({e for c, v in chains.items() if not v['present']
                                               for e in v['ec']} - set(new))}


# ---------------------------------------------------------------- homology fallback
def _local_aligner():
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = 'local'
    al.match_score, al.mismatch_score = 2.0, -1.0
    al.open_gap_score, al.extend_gap_score = -4.0, -0.5
    return al


def _aln_pairs(al, target: str, query: str):
    """[(target index, query index)] of the aligned (non-gap) columns of the best alignment."""
    try:
        aln = al.align(target, query)[0]
    except (OverflowError, ValueError, IndexError):
        return []
    return [(int(t0 + d), int(q0 + d)) for (t0, t1), (q0, q1) in zip(*aln.aligned)
            for d in range(int(q1 - q0))]


def reference_products(info_of: dict, entries: dict) -> list[dict]:
    """Mature products (with their sequences) of the reviewed resolvable polyproteins."""
    out = []
    for acc in sorted(info_of):
        i = info_of[acc]
        if not (i.get('resolvable') and 'Swiss-Prot' in str(i.get('entry_type'))):
            continue
        seq = _uniprot_seq(entries, acc)
        for p in i['products']:
            s = seq[p['start'] - 1:p['end']]
            if s:
                out.append({'accession': acc, 'product': p['name'], 'start': p['start'],
                            'end': p['end'], 'ec': list(p['ec']), 'seq': s})
    return out


def multi_product_refs(ecs, info_of: dict) -> list[str]:
    """Reviewed resolvable polyproteins with >= 2 distinct EC-carrying mature products whose
    EC numbers occur in `ecs` (the trigger of the homology fallback)."""
    ecs = set(ecs)
    out = []
    for acc, i in info_of.items():
        if not (i.get('resolvable') and 'Swiss-Prot' in str(i.get('entry_type'))):
            continue
        hit = {p['name'] for p in i['products'] if set(p['ec']) & ecs}
        if len(hit) >= 2:
            out.append(acc)
    return sorted(out)


def homology_match(seq: str, refs: list, al=None, top: int = 8) -> dict | None:
    """Best reference product for a chain sequence, or None.

    A product matches with >= HOMOLOGY_MIN_IDENTITY identity (identical / aligned columns)
    over >= HOMOLOGY_MIN_COVERAGE of the chain, or of the product when the chain is longer
    (aligned columns / min(len chain, len product)), and >= HOMOLOGY_MIN_ALIGNED columns.
    The best match has the most identical residues (then identity, then accession order).
    """
    if len(seq) < HOMOLOGY_MIN_ALIGNED or not refs:
        return None
    al = al or _local_aligner()
    uniq = {}
    for r in refs:
        uniq.setdefault(r['seq'], []).append(r)
    scored = sorted(((al.score(s, seq), s) for s in uniq), key=lambda x: -x[0])[:top]
    best = None
    for _, s in scored:
        pairs = _aln_pairs(al, s, seq)
        n_al = len(pairs)
        n_id = sum(s[t] == seq[q] for t, q in pairs)
        ident = n_id / n_al if n_al else 0.0
        cov = n_al / min(len(seq), len(s))
        if n_al < HOMOLOGY_MIN_ALIGNED or ident < HOMOLOGY_MIN_IDENTITY or cov < HOMOLOGY_MIN_COVERAGE:
            continue
        r = uniq[s][0]
        cand = {'accession': r['accession'], 'product': r['product'],
                'identity': round(ident, 4), 'coverage': round(cov, 4), 'n_identical': n_id,
                'n_aligned': n_al, 'chain_length': len(seq), 'product_length': len(s),
                'same_sequence_refs': sorted({x['accession'] for x in uniq[s]})}
        if best is None or (n_id, ident) > (best['n_identical'], best['identity']):
            best = cand
    return best


def homology_chain_ec(seq: str, ref_acc: str, entries: dict, info_of: dict, al=None):
    """EC of a chain matched to reference polyprotein `ref_acc`: the chain is aligned to the
    whole precursor and the overlap rule (chain_mature_ec) is applied to the aligned
    positions. -> (EC set, product details, rows [(acc, beg, end)])."""
    al = al or _local_aligner()
    full = _uniprot_seq(entries, ref_acc)
    pos = sorted(t + 1 for t, _ in _aln_pairs(al, full, seq))
    rows = []
    for p in pos:
        if rows and p == rows[-1][2] + 1:
            rows[-1] = (ref_acc, rows[-1][1], p)
        else:
            rows.append((ref_acc, p, p))
    ec, det = chain_mature_ec(rows, info_of)
    return ec, det, rows


def homology_fallback(seqs, ref_all_ec_of, refs, entries, info_of, al=None) -> dict | None:
    """Fallback record of one chain from its sequence(s) (MISATO pieces or, for a chain not in
    the MISATO complex, its SIFTS UniProt segment), or None when no product matches."""
    al = al or _local_aligner()
    ec, matches, rows_all = set(), [], []
    for s in seqs:
        m = homology_match(s, refs, al)
        if m is None:
            return None                     # every sequence of the chain must match
        e, det, rows = homology_chain_ec(s, m['accession'], entries, info_of, al)
        ec |= e
        matches.append({**m, 'products': det})
        rows_all += [list(r) for r in rows]
    if not matches:
        return None
    refs_used = sorted({m['accession'] for m in matches})
    return {'source': 'homology_fallback', 'ec': ec, 'matches': matches,
            'aligned_rows': rows_all, 'reference_accessions': refs_used,
            'ref_all_ec': set().union(*[ref_all_ec_of[a] for a in refs_used])}


# ---------------------------------------------------------------- MISATO chains
def _uniprot_seq(info_entries: dict, acc: str) -> str:
    e = info_entries.get(acc) or {}
    return (e.get('sequence') or {}).get('value') or ''


def align_piece_to_sifts(seq: str, sifts_chains: dict, entries: dict) -> tuple[list, int]:
    """Chains whose SIFTS UniProt segments best match a piece (local alignment) and the
    number of identical residues; [] when none passes MIN_PIECE_IDENTICAL / _IDENTITY."""
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = 'local'
    al.match_score, al.mismatch_score = 2.0, -1.0
    al.open_gap_score, al.extend_gap_score = -4.0, -0.5
    best, best_n = [], 0
    for c, rows in sifts_chains.items():
        ref = ''.join(_uniprot_seq(entries, a)[b - 1:e] for a, b, e in rows)
        if not ref or not seq:
            continue
        try:
            aln = al.align(ref, seq)[0]
        except (OverflowError, ValueError, IndexError):
            continue
        n = sum(ref[t0 + d] == seq[q0 + d] for (t0, t1), (q0, q1) in zip(*aln.aligned)
                for d in range(q1 - q0))
        if n > best_n:
            best, best_n = [c], n
        elif n == best_n and n > 0:
            best.append(c)
    if best_n < MIN_PIECE_IDENTICAL or best_n < MIN_PIECE_IDENTITY * len(seq):
        return [], best_n
    return best, best_n


def misato_present_chains(pids, sifts: dict, entries: dict, workers: int = 16) -> dict:
    """{pid: {'chains': set of chain ids, 'pieces': [...], 'note': str|None}}."""
    from rmsf_common import MISATO_CHAIN_BREAKS, MISATO_REAL_BB_DIR, RCSB_MMCIF_DIR
    from build_misato_splits_v2 import _complex_chains
    table = json.loads(Path(MISATO_CHAIN_BREAKS).read_text())
    if table.get('schema') != 'misato-chain-breaks-v2':
        raise SystemExit(f'{MISATO_CHAIN_BREAKS}: schema {table.get("schema")!r}')
    comp = table['complexes']
    jobs = [(p, str(MISATO_REAL_BB_DIR), str(RCSB_MMCIF_DIR), comp[p]) for p in pids if p in comp]
    res = []
    if workers > 1 and jobs:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_safe_complex_chains, [(j, _complex_chains) for j in jobs], chunksize=2)
    else:
        res = [_safe_complex_chains((j, _complex_chains)) for j in jobs]
    out = {}
    for pid, pieces, note in res:
        sc = sifts.get(pid, {})
        chains = set()
        for pc in pieces:
            c = pc.get('chain')
            if c is not None and c in sc:
                pc['how'] = pc.get('chain_source')
                chains.add(c)
                continue
            if c is not None:
                pc['note'] = f'chain {c} not in SIFTS'
            hit, n = align_piece_to_sifts(pc['seq'], sc, entries)
            pc['how'] = 'uniprot_alignment' if hit else 'unidentified'
            pc['uniprot_alignment'] = {'chains': hit, 'n_identical': n}
            chains.update(hit)
        out[pid] = {'chains': chains, 'pieces': [{k: v for k, v in pc.items() if k != 'seq'}
                                                 | {'length': len(pc['seq'])} for pc in pieces],
                    'piece_seqs': [(pc.get('chain'), pc['seq']) for pc in pieces],
                    'note': note,
                    'unidentified': sum(pc['how'] == 'unidentified' for pc in pieces)}
    for p in pids:
        if p not in out:
            out[p] = {'chains': None, 'pieces': [], 'note': 'not in chain-break table',
                      'unidentified': 1, 'piece_seqs': []}
    table_meta = {'schema': table.get('schema'), 'rule_version': table.get('rule_version'),
                  'path': _rel(MISATO_CHAIN_BREAKS), 'sha256': file_sha256(MISATO_CHAIN_BREAKS)}
    return out, table_meta


def _safe_complex_chains(arg):
    job, fn = arg
    try:
        return fn(job)
    except Exception as e:                     # recorded; the piece is then unidentified
        pid, rb_dir = job[0], job[1]
        with np.load(Path(rb_dir) / f'{pid}.npz', allow_pickle=False) as z:
            seq = str(z['seq'])
        return pid, [{'start': 0, 'end': len(seq), 'seq': seq, 'chain': None,
                      'chain_source': None, 'aligned_chain': None}], f'{type(e).__name__}: {e}'


# ---------------------------------------------------------------- domain overrides (step 4c)
# FIXED(function_reaudit GO + EC labels, auditor 5, 2026-10-19): new. Authors' decision: a
#   manual override list with a documented rule. A construct that contains only one domain is
#   labelled with the EC numbers of that domain; the EC numbers of the domains it does not
#   contain are removed (SIFTS gives every chain all EC numbers of its UniProt entry or
#   polyprotein component, whatever part of the protein the construct holds). Old behaviour:
#   these 30 complexes kept the labels of steps 4 / 4b (listed as `old_ec` below).
DOMAIN_OVERRIDE_VERSION = 'domain-overrides-v1 (2026-10-19)'
DOMAIN_OVERRIDE_REASON = 'construct contains only this domain; EC numbers of absent domains removed'
DOMAIN_OVERRIDE_RULE = (
    'a construct that contains only one domain of a multi-domain protein is labelled with the EC '
    'numbers of that domain; the EC numbers of the domains it does not contain are removed. A '
    'domain (UniProt Domain / Region / Zinc finger feature) is contained when the chain\'s UniProt '
    'residues cover >= {c:.0%} of it and absent when they cover < {a:.0%}; every removed EC number '
    'belongs to an absent domain; a construct whose only domain has no catalytic activity gets an '
    'empty label (the loaders skip it)')
DOMAIN_FEATURE_SOURCE = ('UniProt REST JSON, release 2026_03 (02-September-2026), features '
                         'Domain / Region / Zinc finger / Chain; fetched 2026-10-19')
OVERRIDE_CONTAINED_MIN = 0.9   # fraction of a domain the construct must cover to contain it
OVERRIDE_ABSENT_MAX = 0.5      # below this fraction a domain is absent from the construct

# UniProt features used by the table: accession -> {key: (feature type, name, begin, end, [EC])}.
# The EC numbers of a feature are those of the activity it carries (UniProt "Includes" /
# "Contains" component, catalytic-activity comments, active-site features); [] = no catalytic
# activity.
_HCV_PROT = ('Domain', 'Peptidase S29 (NS3 serine protease)', 1027, 1208, ['3.4.21.98'])
_HCV_HEL = ['3.6.1.15', '3.6.4.13']
_NS5_MT = ['2.1.1.56', '2.1.1.57']
DOMAIN_FEATURES = {
    # HCV genome polyprotein; NS3 chain 1027-1657 = protease (S29) + helicase
    'P26662': {'prot': _HCV_PROT, 'hel1': ('Domain', 'Helicase ATP-binding', 1217, 1369, _HCV_HEL)},
    'P26664': {'prot': _HCV_PROT, 'hel1': ('Domain', 'Helicase ATP-binding', 1217, 1369, _HCV_HEL)},
    'A8DG50': {'prot': _HCV_PROT, 'hel1': ('Domain', 'Helicase ATP-binding', 1217, 1369, _HCV_HEL),
               'hel2': ('Domain', 'Helicase C-terminal', 1361, 1538, _HCV_HEL)},
    'S4UY05': {'prot': _HCV_PROT, 'hel1': ('Domain', 'Helicase ATP-binding', 1228, 1369, _HCV_HEL),
               'hel2': ('Domain', 'Helicase C-terminal', 1361, 1538, _HCV_HEL)},
    'K4KA16': {'prot': _HCV_PROT, 'hel1': ('Domain', 'Helicase ATP-binding', 1217, 1369, _HCV_HEL),
               'hel2': ('Domain', 'Helicase C-terminal', 1361, 1538, _HCV_HEL)},
    # flavivirus genome polyprotein; NS5 = cap methyltransferase + RdRp
    'P03314': {'mt': ('Domain', 'mRNA cap 0-1 NS5-type MT', 2507, 2771, _NS5_MT),
               'rdrp': ('Domain', 'RdRp catalytic', 3035, 3187, ['2.7.7.48'])},       # YFV 17D
    'P27915': {'mt': ('Domain', 'mRNA cap 0-1 NS5-type MT', 2492, 2753, _NS5_MT),
               'rdrp': ('Domain', 'RdRp catalytic', 3018, 3168, ['2.7.7.48'])},       # DENV-3
    'A0A024B7W1': {'mt': ('Domain', 'mRNA cap 0-1 NS5-type MT', 2521, 2785, _NS5_MT),
                   'rdrp': ('Domain', 'RdRp catalytic', 3049, 3199, ['2.7.7.48'])},   # ZIKV
    # MERS-CoV pp1a; nsp3 component EC 3.2.2.- / 3.4.19.12 / 3.4.22.-; the ADP-ribosylhydrolase
    # active sites (1147, 1155) lie in Macro 1, the PL-PRO active sites (1592, 1759, 1774) in C16
    'K9N638': {'mac1': ('Domain', 'Macro 1', 1110, 1276, ['3.2.2.-']),
               'plpro': ('Domain', 'Peptidase C16 (PLpro)', 1552, 1823, ['3.4.19.12', '3.4.22.-'])},
    # IBV (M41) pp1a; 3C-like proteinase component EC 3.4.22.-
    'P0C6V5': {'mpro': ('Domain', 'Peptidase C30 (3C-like proteinase chain 2782-3088)', 2782, 3088,
                        ['3.4.22.-']),
               'plpro': ('Domain', 'Peptidase C16 (PLpro)', 1238, 1499, ['3.4.19.12', '3.4.22.-'])},
    # TAF1: entry EC 2.3.1.48 (HAT) and 2.7.11.1 (two kinase domains); bromodomains non-catalytic
    'P21675': {'bromo2': ('Domain', 'Bromo 2', 1519, 1628, []),
               'hat': ('Region', 'Histone acetyltransferase (HAT)', 538, 997, ['2.3.1.48']),
               'kin1': ('Domain', 'Protein kinase 1', 1, 435, ['2.7.11.1']),
               'kin2': ('Domain', 'Protein kinase 2', 1446, 1893, ['2.7.11.1'])},
    # HDAC6: deacetylase 3.5.1.- (DAC2, also DAC1); E3 ligase 2.3.2.- in DAC1 (PubMed 24882211:
    # "the DAC1 domain harbors an intrinsic E3 ligase activity"); ZnF-UBP binds ubiquitin
    'Q9UBN7': {'znf': ('Zinc finger', 'UBP-type (ZnF-UBP)', 1111, 1209, []),
               'dac1': ('Region', 'Histone deacetylase 1 (DAC1)', 87, 404, ['2.3.2.-', '3.5.1.-']),
               'dac2': ('Region', 'Histone deacetylase 2 (DAC2)', 482, 800, ['3.5.1.-'])},
    # trifunctional purine biosynthetic protein (Includes: GARS 6.3.4.13, AIRS 6.3.3.1, GART 2.1.2.2)
    'P22102': {'gart': ('Region', 'GART domain', 810, 1010, ['2.1.2.2']),
               'airs': ('Region', 'AIRS domain', 434, 809, ['6.3.3.1']),
               'gars': ('Domain', 'ATP-grasp (GARS)', 111, 318, ['6.3.4.13'])},
    # UMP synthase (Includes: OPRTase 2.4.2.10, OMPdecase 4.1.1.23)
    'P11172': {'ompdc': ('Region', 'OMPdecase', 221, 480, ['4.1.1.23']),
               'oprt': ('Region', 'OPRTase', 2, 214, ['2.4.2.10'])},
}


def _ov(pid, chains, acc, construct, old, new, contained, absent, note=None) -> dict:
    """One override entry. construct: the chain's UniProt residue ranges [(begin, end)];
    contained / absent: keys of DOMAIN_FEATURES[acc]."""
    feat = lambda k: dict(zip(('feature', 'name', 'begin', 'end', 'ec'), DOMAIN_FEATURES[acc][k]))
    return {'pid': pid, 'chains': list(chains), 'accession': acc,
            'construct_uniprot_ranges': [list(r) for r in construct],
            'old_ec': sorted(old), 'new_ec': sorted(new),
            'domain': [feat(k) for k in contained], 'absent_domains': [feat(k) for k in absent],
            'reason': DOMAIN_OVERRIDE_REASON, **({'note': note} if note else {})}


_NS3_OLD = ['3.4.21.98', '3.6.1.15', '3.6.4.13']
_NS3 = dict(old=_NS3_OLD, new=['3.4.21.98'], contained=['prot'])
_NS5 = dict(old=['2.1.1.56', '2.1.1.57', '2.7.7.48'], new=_NS5_MT, contained=['mt'], absent=['rdrp'])
_TAF1 = dict(old=['2.3.1.48', '2.7.11.1'], new=[], contained=['bromo2'],
             absent=['hat', 'kin1', 'kin2'],
             note='the construct (bromodomain 2) covers about 30 % of the UniProt "Protein kinase '
                  '2" domain (1446-1893), which spans both bromodomains; a fragment of a third of '
                  'a kinase domain is not a kinase')
_HDAC6 = dict(old=['2.3.2.-', '3.5.1.-'], new=[], contained=['znf'], absent=['dac1', 'dac2'])
_GART = dict(old=['2.1.2.2', '6.3.3.1', '6.3.4.13'], new=['2.1.2.2'], contained=['gart'],
             absent=['airs', 'gars'])
DOMAIN_OVERRIDES = [
    # HCV NS3 protease-domain constructs (some with the NS4A cofactor peptide, absent from
    # MISATO): protease EC only
    _ov('3P8N', 'A', 'P26662', [(1027, 1206)], absent=['hel1'], **_NS3),
    _ov('3P8O', 'A', 'P26662', [(1027, 1206)], absent=['hel1'], **_NS3),
    _ov('3SU0', 'A', 'A8DG50', [(1013, 1206)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SU1', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SU2', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SU3', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SU4', 'A', 'P26664', [(1013, 1208)], absent=['hel1'], **_NS3),
    _ov('3SU5', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SU6', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SUE', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SUF', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('3SUG', 'A', 'A8DG50', [(1013, 1208)], absent=['hel1', 'hel2'], **_NS3),
    _ov('5VOJ', 'A', 'P26664', [(1013, 1026), (1029, 1208)], absent=['hel1'], **_NS3),
    _ov('5VP9', 'A', 'P26664', [(1013, 1026), (1029, 1208)], absent=['hel1'], **_NS3),
    _ov('6NZV', 'A', 'S4UY05', [(1030, 1208)], absent=['hel1', 'hel2'], **_NS3),
    # HCV NS3 HELICASE construct (RCSB title "NS3 Helicase"): helicase EC only
    _ov('4OKS', 'AB', 'K4KA16', [(1206, 1656)], old=_NS3_OLD, new=_HCV_HEL,
        contained=['hel1', 'hel2'], absent=['prot']),
    # flavivirus NS5 methyltransferase-domain constructs: MTase EC only
    _ov('3EVD', 'A', 'P03314', [(2507, 2772)], **_NS5),
    _ov('3P8Z', 'A', 'P27915', [(2491, 2752)], **_NS5),
    _ov('5ULP', 'A', 'A0A024B7W1', [(2521, 2788)], **_NS5),
    # MERS-CoV nsp3 macrodomain (RCSB title "MERS-CoV macro domain in complex with ADP-ribose")
    _ov('5DUS', 'A', 'K9N638', [(1110, 1273)], old=['3.2.2.-', '3.4.19.12', '3.4.22.-'],
        new=['3.2.2.-'], contained=['mac1'], absent=['plpro']),
    # IBV main protease (no SIFTS mapping; RCSB entity aligned to P0C6V5 2782-3088)
    _ov('2Q6F', 'AB', 'P0C6V5', [(2782, 3088)], old=['3.4.19.12', '3.4.22.-'], new=['3.4.22.-'],
        contained=['mpro'], absent=['plpro'],
        note='no current SIFTS mapping; construct range from the RCSB polymer-entity alignment'),
    # TAF1 bromodomain 2: no catalytic activity
    _ov('5I1Q', 'A', 'P21675', [(1518, 1659)], **_TAF1),
    _ov('5I29', 'A', 'P21675', [(1518, 1659)], **_TAF1),
    _ov('5MG2', 'A', 'P21675', [(1522, 1656)], **_TAF1),
    # HDAC6 ZnF-UBP: no catalytic activity
    _ov('6CE6', 'A', 'Q9UBN7', [(1109, 1213)], **_HDAC6),
    _ov('6CED', 'A', 'Q9UBN7', [(1109, 1213)], **_HDAC6),
    # GART domain of the trifunctional purine biosynthetic protein: 2.1.2.2 only
    _ov('1NJS', 'A', 'P22102', [(808, 1010)], **_GART),
    _ov('4EW2', 'A', 'P22102', [(808, 1010)], **_GART),
    _ov('4EW3', 'A', 'P22102', [(808, 1010)], **_GART),
    # OMP decarboxylase domain of UMP synthase: 4.1.1.23 only
    _ov('3MI2', 'AB', 'P11172', [(223, 480)], old=['2.4.2.10', '4.1.1.23'], new=['4.1.1.23'],
        contained=['ompdc'], absent=['oprt']),
]


def labels_sha256(labels: dict) -> str:
    """sha256 of a label dict serialised as the output file (json, indent 0, sorted keys)."""
    return hashlib.sha256(json.dumps(labels, indent=0, sort_keys=True).encode()).hexdigest()


def domain_coverage(construct, begin: int, end: int) -> float:
    """Fraction of the feature begin-end covered by the construct's UniProt residue ranges."""
    pos = {i for b, e in construct for i in range(int(b), int(e) + 1)}
    return len(pos & set(range(begin, end + 1))) / (end - begin + 1)


def check_override(ov: dict) -> list[str]:
    """Problems of one table entry against the documented rule ([] = consistent)."""
    bad = []
    cons = ov['construct_uniprot_ranges']
    for d in ov['domain']:
        if domain_coverage(cons, d['begin'], d['end']) < OVERRIDE_CONTAINED_MIN:
            bad.append(f'{d["name"]} not contained')
    for d in ov['absent_domains']:
        if domain_coverage(cons, d['begin'], d['end']) >= OVERRIDE_ABSENT_MAX:
            bad.append(f'{d["name"]} not absent')
    if set(ov['new_ec']) != {e for d in ov['domain'] for e in d['ec']}:
        bad.append('new EC != EC of the contained domain(s)')
    if not set(ov['new_ec']) <= set(ov['old_ec']):
        bad.append('new EC not a subset of the old EC')
    removed = set(ov['old_ec']) - set(ov['new_ec'])
    if not removed <= {e for d in ov['absent_domains'] for e in d['ec']}:
        bad.append('a removed EC number belongs to no absent domain')
    return bad


def apply_domain_overrides(labels: dict, changed: dict, source: dict,
                           overrides=None) -> dict:
    """Step 4c: apply the override table to the labels of steps 4 / 4b (in place).

    labels: {pid: sorted EC list}; changed: {pid: detail} of the complexes whose label differs
    from the source; source: the source labels. Each entry's `old_ec` must equal the current
    label (SystemExit otherwise: the table is stale) and the entry must satisfy the rule
    (check_override). -> sidecar record {version, rule, ..., entries (with coverages)}.
    """
    overrides = DOMAIN_OVERRIDES if overrides is None else overrides
    pids = [o['pid'] for o in overrides]
    if len(set(pids)) != len(pids):
        raise SystemExit('domain overrides: duplicate PDB ids')
    entries = []
    for o in overrides:
        pid = o['pid']
        why = check_override(o)
        if why:
            raise SystemExit(f'domain override {pid}: {"; ".join(why)}')
        if pid not in labels:
            raise SystemExit(f'domain override {pid}: not a labelled MISATO complex')
        if labels[pid] != o['old_ec']:
            raise SystemExit(f'domain override {pid}: label {labels[pid]} != table old_ec '
                             f'{o["old_ec"]} (stale entry; re-verify it)')
        cons = o['construct_uniprot_ranges']
        rec = {**o,
               'domain': [{**d, 'coverage': round(domain_coverage(cons, d['begin'], d['end']), 4)}
                          for d in o['domain']],
               'absent_domains': [{**d, 'coverage': round(domain_coverage(cons, d['begin'],
                                                                          d['end']), 4)}
                                  for d in o['absent_domains']]}
        entries.append(rec)
        labels[pid] = list(o['new_ec'])
        src = sorted(set(source[pid]))
        det = changed.get(pid) or {'old': src, 'cr_chains': {}, 'homology_chains': {}}
        det.update({'new': labels[pid], 'domain_override': rec,
                    'removed': sorted(set(src) - set(labels[pid])),
                    'added': sorted(set(labels[pid]) - set(src))})
        if labels[pid] != src:
            changed[pid] = det
        else:
            changed.pop(pid, None)
    return {'version': DOMAIN_OVERRIDE_VERSION,
            'rule': DOMAIN_OVERRIDE_RULE.format(c=OVERRIDE_CONTAINED_MIN, a=OVERRIDE_ABSENT_MAX),
            'reason': DOMAIN_OVERRIDE_REASON, 'feature_source': DOMAIN_FEATURE_SOURCE,
            'thresholds': {'contained_min_coverage': OVERRIDE_CONTAINED_MIN,
                           'absent_max_coverage': OVERRIDE_ABSENT_MAX},
            'n_entries': len(entries), 'entries': entries}


# ---------------------------------------------------------------- build
def build(source: dict, sifts: dict, enzyme: dict, entries: dict, present_of) -> dict:
    """All results of the relabelling (pure; `present_of(pids)` -> {pid: present info})."""
    info_of = {a: entry_products(e) for a, e in entries.items()}
    poly = {a: i for a, i in info_of.items() if i['polyprotein']}
    # Scan of ALL MISATO complexes (keys of `sifts` = every MISATO PDB entry SIFTS maps).
    scan_all, scan_lab = Counter(), Counter()
    for pid, ch in sifts.items():
        for a in {r[0] for rows in ch.values() for r in rows} & set(poly):
            scan_all[a] += 1
            scan_lab[a] += pid in source
    resolvable = {a for a, i in info_of.items() if i['resolvable']}
    cr_todo = {p for p in source
               if any(r[0] in resolvable for rows in sifts.get(p, {}).values() for r in rows)}
    # Homology fallback triggers (coordinator follow-up to D8, 2026-10-19): (a) a SIFTS chain
    # whose accession is not a resolvable polyprotein but whose EC set (UniProt entry EC +
    # SIFTS chain EC) holds numbers of >= 2 distinct mature products of a reviewed
    # polyprotein; (b) a labelled complex without any current SIFTS mapping whose source
    # label holds such numbers.
    fb_trigger: dict = defaultdict(dict)       # pid -> {chain: [reference polyproteins]}
    for p in source:
        for c, rows in sifts.get(p, {}).items():
            if any(r[0] in resolvable for r in rows):
                continue
            for acc in {r[0] for r in rows}:
                ecs = set((info_of.get(acc) or {}).get('all_ec', [])) | \
                    set(((enzyme.get(p) or {}).get(c) or {}).get(acc, ()))
                refs_hit = multi_product_refs(ecs, info_of)
                if refs_hit:
                    fb_trigger[p][c] = sorted(set(fb_trigger[p].get(c, [])) | set(refs_hit))
    no_sifts = sorted(p for p in source if p not in sifts and multi_product_refs(source[p], info_of))
    todo = sorted(cr_todo | set(fb_trigger) | set(no_sifts))
    present = present_of(todo)
    refs = reference_products(info_of, entries)
    ref_all_ec_of = {a: set(info_of[a]['all_ec']) for a in {r['accession'] for r in refs}}
    al = _local_aligner()
    homology_failed = {}
    new_labels, changed, unchanged_cr, unidentified = {}, {}, [], []
    for pid in sorted(source):
        if pid not in todo:
            new_labels[pid] = sorted(set(source[pid]))
            continue
        pres = present[pid]
        if pres['unidentified'] and pid in cr_todo:
            unidentified.append(pid)
        fallback = {}
        if pid in fb_trigger:                                   # (a)
            for c in sorted(fb_trigger[pid]):
                seqs = [s for ch, s in pres['piece_seqs'] if ch == c]
                in_misato = bool(seqs) and c in (pres['chains'] or set())
                if not in_misato:                               # SIFTS UniProt segment instead
                    seqs = [''.join(_uniprot_seq(entries, a)[b - 1:e]
                                    for a, b, e in sifts[pid][c])]
                fb = homology_fallback(seqs, ref_all_ec_of, refs, entries, info_of, al)
                if fb is None:
                    homology_failed.setdefault(pid, []).append(c)
                    continue
                fallback[c] = {**fb, 'sifts_chain': c, 'present': in_misato,
                               'trigger': 'non-polyprotein entry with EC of >= 2 products of '
                                          + ', '.join(fb_trigger[pid][c]),
                               'sifts': [list(r) for r in sifts[pid][c]]}
        if pid in no_sifts:                                     # (b)
            for i, (ch, s) in enumerate(pres['piece_seqs']):
                fb = homology_fallback([s], ref_all_ec_of, refs, entries, info_of, al)
                if fb is None:
                    homology_failed.setdefault(pid, []).append(ch or f'piece{i}')
                    continue
                fallback[f'{ch or "?"}#{i}'] = {**fb, 'sifts_chain': None, 'present': True,
                                                'trigger': 'no current SIFTS mapping; source '
                                                           'label holds EC of >= 2 products'}
        new, det = relabel_complex(source[pid], sifts.get(pid, {}), enzyme.get(pid, {}),
                                   pres['chains'], info_of, fallback)
        if new is None:                                         # nothing applicable
            new_labels[pid] = sorted(set(source[pid]))
            continue
        det['misato_pieces'] = pres['pieces']
        if pres['note']:
            det['misato_note'] = pres['note']
        new_labels[pid] = new
        if new != sorted(set(source[pid])):
            changed[pid] = det
        else:
            unchanged_cr.append(pid)
    # Step 4c (2026-10-19): domain override table, after the polyprotein / homology steps.
    # The sha256 of the labels before it (same serialisation as the output file) is recorded,
    # so reverting the table's entries must give it back (no other complex changed).
    sha_before = labels_sha256(new_labels)
    overrides = apply_domain_overrides(new_labels, changed, source)
    overrides['labels_sha256_before_overrides'] = sha_before
    unchanged_cr = [p for p in unchanged_cr if p not in changed]
    polyproteins = []
    for a in sorted(poly, key=lambda a: (-scan_all[a], a)):
        if not scan_all[a]:
            continue
        i = poly[a]
        polyproteins.append({
            'accession': a, 'name': i['name'], 'organism': i['organism'],
            'entry_type': i['entry_type'], 'n_chain_features': i['n_chain_features'],
            'n_contains': i['n_contains'], 'entry_ec': i['entry_ec'],
            'component_resolved': i['component_resolved'], 'resolvable': i['resolvable'],
            'unmatched_ec_components': i['unmatched_ec_components'],
            'treatment': ('relabelled by mature products' if i['resolvable'] else
                          'kept (EC components not located by features)' if i['component_resolved']
                          else 'kept (EC on the entry name only)'),
            'n_misato_complexes': scan_all[a], 'n_labelled_complexes': scan_lab[a],
            'n_changed_complexes': sum(any(r[0] == a for c in d['cr_chains'].values()
                                           for r in c['sifts']) for d in changed.values()),
            'products_with_ec': [{'name': p['name'], 'start': p['start'], 'end': p['end'],
                                  'ec': p['ec']} for p in i['products'] if p['ec']]})
    transitions = Counter((tuple(d['old']), tuple(d['new'])) for d in changed.values())
    homology_changed = sorted(p for p, d in changed.items() if d.get('homology_chains'))
    return {'labels': new_labels, 'changed': changed, 'unchanged_component_resolved': unchanged_cr,
            'unidentified_pieces': unidentified, 'polyproteins': polyproteins,
            'transitions': [{'old': list(o), 'new': list(n), 'n': k}
                            for (o, n), k in transitions.most_common()],
            'n_relabel_candidates': len(todo),
            'domain_overrides': overrides,
            'homology': {'triggered_sifts_chains': {p: v for p, v in sorted(fb_trigger.items())},
                         'triggered_no_sifts': no_sifts,
                         'no_match': homology_failed,
                         'changed': homology_changed,
                         'n_reference_products': len(refs),
                         'n_reference_entries': len(ref_all_ec_of)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--source', default=str(MISATO_EC_LABELS_LEGACY))
    ap.add_argument('--out', default=str(MISATO_EC_LABELS))
    ap.add_argument('--download', action='store_true',
                    help='fetch missing SIFTS / UniProt inputs (existing files are kept)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--overwrite', action='store_true', help='replace an existing output')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        print(f'{_rel(out)} exists; pass --overwrite to rebuild'); return 1
    src_prov = copy_source(Path(a.source))
    source = json.loads(SOURCE_COPY.read_text())
    from rmsf_common import MISATO_CHAIN_BREAKS
    all_pids = set(json.loads(Path(MISATO_CHAIN_BREAKS).read_text())['complexes'])
    for p in (SIFTS_UNIPROT, SIFTS_ENZYME):
        if not p.exists():
            if not a.download:
                print(f'{_rel(p)} missing; run with --download'); return 1
            download_sifts(p)
    sifts = sifts_chain_uniprot(SIFTS_UNIPROT, all_pids)
    enzyme = sifts_chain_enzyme(SIFTS_ENZYME, all_pids)
    accs = {r[0] for ch in sifts.values() for rows in ch.values() for r in rows}
    if not UNIPROT_ENTRIES.exists():
        if not a.download:
            print(f'{_rel(UNIPROT_ENTRIES)} missing; run with --download'); return 1
        download_uniprot(accs)
    up = load_uniprot()
    missing = sorted(accs - set(up['entries']))
    if missing:
        print(f'[error] {len(missing)} SIFTS accessions not in {_rel(UNIPROT_ENTRIES)} '
              f'(e.g. {missing[:5]}); move the file aside and rerun with --download'); return 1
    entries = up['entries']
    chain_meta = {}

    def present_of(pids):
        res, meta = misato_present_chains(pids, sifts, entries, workers=a.workers)
        chain_meta.update(meta)
        return res

    t0 = time.time()
    R = build(source, sifts, enzyme, entries, present_of)
    labels = R['labels']
    if set(labels) != set(source):
        print('[error] key set differs from the source'); return 1
    counts = {'source_legacy_truncation': class_counts(source, truncate_ec_legacy),
              'source_new_truncation': class_counts(source, truncate_ec),
              'v2_new_truncation': class_counts(labels, truncate_ec)}
    n_empty = sorted(p for p, v in labels.items() if not v)
    wild = Counter(e for v in labels.values() for e in v if '-' in e)
    prov = {
        'schema': SCHEMA, 'builder': 'data_prep/build_misato_ec_labels.py',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'runtime_s': round(time.time() - t0, 1),
        'decision': 'authors decision D8 (2026-10-18, EC audit): polyprotein chains labelled '
                    'by the mature products they cover; wildcard EC numbers only at fully '
                    'specified depths (downstream truncate_ec); domain override table '
                    f'{DOMAIN_OVERRIDE_VERSION} (authors\' decision 2026-10-19, function_reaudit '
                    'GO + EC labels): single-domain constructs carry the EC numbers of that '
                    'domain only',
        'rules': {
            'polyprotein': '>= 2 UniProt Chain features or a Contains section',
            'relabelled': 'component-resolved polyproteins (>= 1 Contains component with an EC '
                          'number) whose EC components are all located by a Chain / Peptide '
                          'feature of the same name; all other entries keep the SIFTS EC',
            'products': 'Chain / Peptide features except the precursor (named like the entry); '
                        'EC of the Contains component of the same name; a feature without a '
                        'component carries the entry-level EC; residues in no product '
                        '(precursor-only) carry the entry-level EC',
            'overlap': f'>= {OVERLAP_MIN_RES} residues or >= {OVERLAP_MIN_FRAC:.0%} of the '
                       f'chain\'s SIFTS-mapped UniProt residues',
            'chains': 'protein chains present in the MISATO complex (pieces of '
                      'misato_chain_breaks.json; chain id from the break table, alignment to '
                      'the deposited mmCIF, else alignment to the SIFTS UniProt segments)',
            'complex_label': 'source label minus the EC numbers only the relabelled chains '
                             'explained, union the EC of the present relabelled chains; '
                             'complexes without such a chain: source label unchanged',
            'homology_fallback': 'triggered for (a) a SIFTS chain whose accession is not a '
                                 'resolvable polyprotein but whose EC set (UniProt entry EC + '
                                 'SIFTS chain EC) holds EC numbers of >= 2 distinct mature '
                                 'products of a reviewed (Swiss-Prot) polyprotein, and (b) a '
                                 'labelled complex without current SIFTS mapping whose source '
                                 'label does; the chain sequence (MISATO piece, else the SIFTS '
                                 'UniProt segment) is aligned to the mature products of the '
                                 'reviewed resolvable polyproteins; a match needs identity >= '
                                 f'{HOMOLOGY_MIN_IDENTITY:.0%} over >= {HOMOLOGY_MIN_COVERAGE:.0%} '
                                 'of the chain (of the product if the chain is longer) and >= '
                                 f'{HOMOLOGY_MIN_ALIGNED} aligned residues; the chain is then '
                                 'aligned to the whole precursor and the overlap rule applied; its '
                                 'SIFTS EC numbers are replaced (source "homology_fallback")',
            'wildcards': 'kept in the file; downstream truncate_ec uses a partial EC number at '
                         'depth d only if its first d fields are specified (no class with "-")',
            'domain_overrides': R['domain_overrides']['rule'] + ' (step 4c, applied after the '
                                'polyprotein and homology steps; table DOMAIN_OVERRIDES)'},
        'thresholds': {'overlap_min_residues': OVERLAP_MIN_RES,
                       'overlap_min_fraction_of_chain': OVERLAP_MIN_FRAC,
                       'piece_alignment_min_identical': MIN_PIECE_IDENTICAL,
                       'piece_alignment_min_identity': MIN_PIECE_IDENTITY,
                       'homology_min_identity': HOMOLOGY_MIN_IDENTITY,
                       'homology_min_coverage': HOMOLOGY_MIN_COVERAGE,
                       'homology_min_aligned': HOMOLOGY_MIN_ALIGNED},
        'inputs': {'source': {'path': _rel(SOURCE_COPY), 'sha256': file_sha256(SOURCE_COPY),
                              **{k: src_prov.get(k) for k in ('origin_path', 'origin_resolved',
                                                              'copied')}},
                   'sifts_pdb_chain_uniprot': input_record(SIFTS_UNIPROT),
                   'sifts_pdb_chain_enzyme': input_record(SIFTS_ENZYME),
                   'uniprot_entries': {**input_record(UNIPROT_ENTRIES),
                                       'uniprot_release': up.get('uniprot_release'),
                                       'uniprot_release_date': up.get('uniprot_release_date')},
                   'misato_chain_breaks': chain_meta},
        'counts': {'n_complexes': len(labels), 'n_misato_complexes_scanned': len(all_pids),
                   'n_misato_complexes_with_sifts': len(sifts),
                   'n_uniprot_entries': len(entries),
                   'n_polyprotein_entries_in_misato': len(R['polyproteins']),
                   'n_component_resolved_entries_in_misato':
                       sum(p['component_resolved'] for p in R['polyproteins']),
                   'n_relabel_candidates': R['n_relabel_candidates'],
                   'n_changed': len(R['changed']),
                   'n_candidates_unchanged': len(R['unchanged_component_resolved']),
                   'n_empty_labels': len(n_empty),
                   'n_with_unidentified_piece': len(R['unidentified_pieces']),
                   'n_changed_by_homology_fallback': len(R['homology']['changed']),
                   'n_domain_overrides': R['domain_overrides']['n_entries'],
                   'wildcard_ec_in_labels': dict(wild.most_common()),
                   'classes_per_depth': counts},
        'empty_labels': n_empty,
        'unidentified_pieces': R['unidentified_pieces'],
        'homology_fallback': R['homology'],
        'domain_overrides': R['domain_overrides'],
        'transitions': R['transitions'],
        'polyproteins': R['polyproteins'],
        'changed': R['changed'],
        'output_sha256': None}
    data = json.dumps(labels, indent=0, sort_keys=True).encode()   # = labels_sha256()
    prov['output_sha256'] = hashlib.sha256(data).hexdigest()
    _atomic_bytes(out, data)
    _atomic_json(Path(str(out) + '.provenance.json'), prov, indent=1)
    c = prov['counts']
    print(f'[done] {_rel(out)}: {c["n_changed"]} of {c["n_complexes"]} complexes changed '
          f'({c["n_relabel_candidates"]} candidates, {c["n_empty_labels"]} empty, '
          f'{c["n_with_unidentified_piece"]} with an unidentified piece); '
          f'sha256 {prov["output_sha256"][:12]}')
    print(f'  domain overrides ({DOMAIN_OVERRIDE_VERSION}): '
          f'{R["domain_overrides"]["n_entries"]} complexes')
    for d in ('1', '2', '3'):
        print(f'  d{d}: classes legacy {counts["source_legacy_truncation"][d]["n_classes"]} -> '
              f'source+new truncation {counts["source_new_truncation"][d]["n_classes"]} -> '
              f'v2 {counts["v2_new_truncation"][d]["n_classes"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
