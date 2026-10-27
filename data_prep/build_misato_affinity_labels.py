"""MISATO binding-affinity labels (-log Kd/Ki per complex) from ProteinShake's MISATO Avro file.

What it does
------------
Reads ProteinShake's `MisatoProteinLigandDataset.residue.avro` (one record per MISATO
complex: `protein` = {ID, sequence, neglog_aff, ligand_id, ...}, `residue` = {residue_number,
binding_site, ...}) in file order and writes `data/annotations/misato_affinity.json`
= {PID: -log10(Kd or Ki)} with exactly the rules of the research-repo builder:

  1. records with fewer than 10 residues (len(residue.residue_number)) are skipped;
  2. records whose `binding_site` list length differs from that residue count are skipped
     (the old ProteinShake binding-site label; the D4 labels replace it, see below);
  3. the affinity is kept when `neglog_aff` is not None and not NaN; the value is written
     as a Python float, the file is `json.dump(..., )` without indentation, keys in Avro
     order (byte-identical to the file the paper's affinity runs read).

The research builder also wrote the old binding-site labels
(`data/cached_descriptors/misato_binding_site.npz`, ProteinShake's misaligned pocket mask).
That part is not ported: the probes read the v2 labels of
`data_prep/build_misato_binding_site_labels.py` (benchmark_audit D4).

The output is written atomically with a provenance sidecar (`.provenance.json`: schema,
Avro path / size / sha256, rules, counts, output sha256). `--verify-existing` rebuilds the
labels in memory, compares them byte for byte with the existing file and, if identical,
writes the sidecar next to it (the file itself is not rewritten); a difference is printed
and the exit code is 1.

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{Binding Affinity} (-log Kd/Ki labels of
the MISATO complexes).

Origin
------
Ported from `submission_exp/src/build_misato_binding_labels.py` (research repo), affinity
part only (2026-10-19, function_reaudit pre-run fixes). Changed: Avro path from
`ensembits.paths.EXTERNAL` (was a hard-coded path in another user's home; override
ENSEMBITS_MISATO_AVRO); atomic write, sidecar, `--verify-existing`, `--out`; the old
binding-site output was dropped (superseded by D4). Filters and value handling unchanged.

Used by the reported model: yes (affinity rows of every table; the probes record the label
file's sha256 in every affinity result).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/external/proteinshake/MisatoProteinLigandDataset.residue.avro
          (symbolic link locally; ENSEMBITS_MISATO_AVRO overrides)
  writes  data/annotations/misato_affinity.json (+ .provenance.json)

How to run
----------
Needs `fastavro` (e.g. the dynamo-legacy environment; Python >= 3.8):
    python data_prep/build_misato_affinity_labels.py --verify-existing   # check + sidecar
    python data_prep/build_misato_affinity_labels.py --out <scratch>/misato_affinity.json   # fresh build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT, EXTERNAL, MISATO_AFFINITY  # noqa: E402

AVRO = Path(os.environ.get('ENSEMBITS_MISATO_AVRO',
                           EXTERNAL / 'proteinshake' / 'MisatoProteinLigandDataset.residue.avro'))
OUT_AFF = MISATO_AFFINITY       # data/annotations/misato_affinity.json
SCHEMA = 'misato-affinity-v1'
MIN_RESIDUES = 10
RULES = ('ProteinShake MisatoProteinLigandDataset.residue.avro in file order; skip records with '
         f'< {MIN_RESIDUES} residues or a binding_site list whose length differs from the '
         'residue count; keep protein.neglog_aff when not None and not NaN (float); '
         'json.dump without indentation, Avro order')


def file_sha256(path) -> str:
    """sha256 of a file's bytes (no torch import; this builder runs in a light environment)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def affinity_from_records(records) -> tuple:
    """({pid: affinity}, counts) from an iterable of Avro records (the research rules)."""
    affinity = {}
    n = {'records': 0, 'short': 0, 'binding_site_length_mismatch': 0, 'no_affinity': 0}
    for rec in records:
        n['records'] += 1
        p = rec['protein']; r = rec['residue']
        L = len(r['residue_number'])
        if L < MIN_RESIDUES:
            n['short'] += 1; continue
        if len(r['binding_site']) != L:
            n['binding_site_length_mismatch'] += 1; continue
        v = p.get('neglog_aff')
        if v is not None and not math.isnan(float(v)):
            affinity[p['ID']] = float(v)
        else:
            n['no_affinity'] += 1
    n['written'] = len(affinity)
    return affinity, n


def encode(affinity: dict) -> bytes:
    """The file bytes, as the research builder wrote them (json.dump, no indentation)."""
    return json.dumps(affinity).encode()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f'.{path.name}.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def write_sidecar(out: Path, counts: dict, mode: str) -> Path:
    st = AVRO.stat()
    info = {'schema': SCHEMA, 'builder': 'data_prep/build_misato_affinity_labels.py',
            'origin': 'submission_exp/src/build_misato_binding_labels.py (affinity part)',
            'mode': mode, 'rules': RULES, 'counts': counts,
            'source': {'path': str(AVRO), 'size': st.st_size, 'sha256': file_sha256(AVRO)},
            'output': _rel(out), 'output_sha256': file_sha256(out),
            'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    side = Path(str(out) + '.provenance.json')
    _atomic_write(side, json.dumps(info, indent=1, sort_keys=True).encode())
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(OUT_AFF))
    ap.add_argument('--overwrite', action='store_true', help='replace an existing --out file')
    ap.add_argument('--verify-existing', action='store_true',
                    help='rebuild in memory, compare byte for byte with --out, write only the '
                         'sidecar if identical (exit 1 otherwise)')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not AVRO.exists():
        print(f'ERROR: {AVRO} missing (set ENSEMBITS_MISATO_AVRO)', file=sys.stderr); return 1
    import fastavro
    t0 = time.time()
    with open(AVRO, 'rb') as f:
        affinity, counts = affinity_from_records(fastavro.reader(f))
    data = encode(affinity)
    print(f'{counts} ({time.time() - t0:.0f}s)', flush=True)
    if a.verify_existing:
        if not out.exists():
            print(f'ERROR: {out} does not exist', file=sys.stderr); return 1
        old = out.read_bytes()
        if old != data:
            olds = json.loads(old)
            diff = sorted(set(olds) ^ set(affinity))
            vals = [k for k in set(olds) & set(affinity) if olds[k] != affinity[k]]
            print(f'DIFFERENT: {len(old)} vs {len(data)} bytes; pids only in one file '
                  f'{len(diff)} {diff[:5]}; values differ {len(vals)} {vals[:5]}; '
                  f'key order equal {list(olds) == list(affinity)}', file=sys.stderr)
            return 1
        side = write_sidecar(out, counts, 'verified existing file: byte-identical rebuild')
        print(f'identical ({len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:12]}); '
              f'wrote {side}', flush=True)
        return 0
    if out.exists() and not a.overwrite:
        print(f'ERROR: {out} exists (pass --overwrite or --verify-existing)', file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out, data)
    side = write_sidecar(out, counts, 'built')
    print(f'wrote {out} and {side.name}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
