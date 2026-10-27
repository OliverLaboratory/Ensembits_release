"""MACCS-167 ligand fingerprints of the MISATO complexes (input of the affinity head).

What it does
------------
For each record of ProteinShake's `MisatoProteinLigandDataset.residue.avro` (file order):

  1. `protein.ligand_id` (PDB chemical-component code, CCD) is read; records without one are
     counted; codes in SKIP_CCDS (water, ions, buffers, cryoprotectants) or longer than 3
     characters are skipped.
  2. CCD -> SMILES from the JSON cache `data/cached_descriptors/ccd_smiles_cache.json`
     ({CCD: SMILES or null}); the research builder filled it from the PDBe compound API
     (canonical SMILES first) with RCSB chem_comp as fallback. By default this builder is
     OFFLINE: a CCD missing from the cache is an error; `--fetch` resolves missing codes
     with the same two APIs and adds them to the cache (atomic write).
  3. RDKit `MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))` -> (167,) uint8 0/1 vector
     (bit 0 unused but kept). CCDs with no SMILES or an unparsable SMILES give no vector.

Writes `data/cached_descriptors/misato_ligand_maccs.npz` ({PID: (167,) uint8}, keys in Avro
order, np.savez) atomically, with a provenance sidecar (schema, Avro and SMILES-cache
sha256, RDKit version, counts, the CCDs without SMILES / unparsable, a content hash of the
fingerprints and the output sha256). `--verify-existing` rebuilds the npz in memory and
compares it byte for byte with the existing file (np.savez writes a fixed 1980 zip time, so
a rebuild of the same content is byte-identical; checked 2026-10-19, sha256 43fc61e9...);
if identical it writes the sidecar next to it, otherwise it reports where the content
differs (keys, order, values) and exits 1.

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{Binding Affinity} ("167-bit MACCS
structural-key fingerprint" of the ligand).

Origin
------
Ported from `submission_exp/src/build_misato_ligand_features.py` (research repo,
2026-10-19, function_reaudit pre-run fixes). Changed: Avro path from
`ensembits.paths.EXTERNAL` (was a hard-coded path in another user's home; override
ENSEMBITS_MISATO_AVRO); offline by default (`--fetch` for the API calls, same URLs and
parsing); atomic writes, sidecar, `--verify-existing`, `--out`. SKIP_CCDS, the SMILES
choice and the fingerprint are unchanged.

Used by the reported model: yes (affinity rows: MACCS input of the head, and the item set of
every affinity run = complexes with a label and a fingerprint).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/external/proteinshake/MisatoProteinLigandDataset.residue.avro
          data/cached_descriptors/ccd_smiles_cache.json
  writes  data/cached_descriptors/misato_ligand_maccs.npz (+ .provenance.json)
          (--fetch) data/cached_descriptors/ccd_smiles_cache.json

How to run
----------
Needs `fastavro` and `rdkit` (the dynamo-legacy environment: RDKit 2022.09.5):
    python data_prep/build_misato_ligand_maccs.py --verify-existing
    python data_prep/build_misato_ligand_maccs.py --out <scratch>/misato_ligand_maccs.npz   # fresh build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT, EXTERNAL, MISATO_LIGAND_MACCS  # noqa: E402
from ensembits.data_versions import hash_arrays_by_pid  # noqa: E402

AVRO = Path(os.environ.get('ENSEMBITS_MISATO_AVRO',
                           EXTERNAL / 'proteinshake' / 'MisatoProteinLigandDataset.residue.avro'))
SMILES_CACHE = ROOT / 'data' / 'cached_descriptors' / 'ccd_smiles_cache.json'
OUT_NPZ = MISATO_LIGAND_MACCS   # data/cached_descriptors/misato_ligand_maccs.npz
SCHEMA = 'misato-ligand-maccs-v1'
N_BITS = 167

# Salts, buffers, cryoprotectants, common crystallization artifacts (unchanged).
SKIP_CCDS = {
    'HOH', 'DOD', 'D2O',
    'SO4', 'PO4', 'NO3', 'CO3', 'NH4', 'CL', 'BR', 'IOD', 'F',
    'NA', 'K', 'LI', 'RB', 'CS', 'MG', 'CA', 'SR', 'BA',
    'ZN', 'MN', 'FE', 'FE2', 'CU', 'CU1', 'NI', 'CO', 'CD', 'HG',
    'GOL', 'PEG', 'PG4', 'PGE', 'PEO', 'EDO', 'EOH', 'MEO',
    'TRS', 'EPE', 'BIS', 'MES', 'CIT',
    'ACT', 'FMT', 'DMS', 'IPA', 'ACN', 'BME',
    'SCN', 'AZI', 'OXY',
}
PDBE_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/{ccd}"
RCSB_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{ccd}"
TIMEOUT = 15
SLEEP_BETWEEN = 0.05


def file_sha256(path) -> str:
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


def ligand_codes(records) -> tuple:
    """({pid: CCD}, counts) in Avro order (the research rules)."""
    pid_to_ccd = {}
    n = {'records': 0, 'no_ligand': 0, 'skipped_artifact': 0}
    for rec in records:
        n['records'] += 1
        pid = rec['protein']['ID']
        ccd = rec['protein'].get('ligand_id')
        if ccd is None:
            n['no_ligand'] += 1; continue
        ccd = ccd.strip().upper()
        if ccd in SKIP_CCDS or len(ccd) > 3:
            n['skipped_artifact'] += 1; continue
        pid_to_ccd[pid] = ccd
    n['pids_with_binder'] = len(pid_to_ccd)
    n['unique_ccds'] = len(set(pid_to_ccd.values()))
    return pid_to_ccd, n


def fetch_smiles(ccd: str):
    """PDBe first (canonical SMILES preferred), RCSB fallback; None if neither resolves.
    Unchanged from the research builder."""
    try:
        req = urllib.request.Request(PDBE_URL.format(ccd=ccd.lower()),
                                     headers={'Accept': 'application/json',
                                              'User-Agent': 'misato-ligand-builder/1.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read())
        entries = data.get(ccd.lower())
        if entries:
            smiles_list = entries[0].get('smiles', [])
            for s in smiles_list:
                if 'canonical' in s.get('name', '').lower():
                    val = s.get('value')
                    if val: return val.strip()
            for s in smiles_list:
                val = s.get('value')
                if val: return val.strip()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            TimeoutError, KeyError, IndexError):
        pass
    try:
        req = urllib.request.Request(RCSB_URL.format(ccd=ccd.upper()),
                                     headers={'Accept': 'application/json',
                                              'User-Agent': 'misato-ligand-builder/1.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read())
        # B.13 fix: correct RCSB API key names (were 'smiles'/'smilesstereo', which don't exist;
        # the fallback path below worked)
        descriptors = data.get('rcsb_chem_comp_descriptor', {})
        for key in ('SMILES_stereo', 'SMILES'):
            val = descriptors.get(key)
            if val: return val.strip()
        for kind in ('SMILES_CANONICAL', 'SMILES'):
            for d in data.get('pdbx_chem_comp_descriptor', []):
                if d.get('type') == kind:
                    val = d.get('descriptor')
                    if val: return val.strip()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            TimeoutError, KeyError):
        pass
    return None


def maccs_167(smiles: str):
    """SMILES -> (167,) uint8 0/1 vector, None if RDKit cannot parse it (unchanged)."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(N_BITS, dtype=np.uint8)
    for i in range(N_BITS):
        if fp.GetBit(i):
            arr[i] = 1
    return arr


def fingerprints(pid_to_ccd: dict, cache: dict) -> tuple:
    """({pid: (167,) uint8} in pid_to_ccd order, counts, no-SMILES CCDs, unparsable CCDs)."""
    out, no_smi, bad = {}, set(), set()
    n = {'no_smiles': 0, 'bad_mol': 0}
    memo = {}
    for pid, ccd in pid_to_ccd.items():
        smi = cache.get(ccd)
        if smi is None:
            n['no_smiles'] += 1; no_smi.add(ccd); continue
        if ccd not in memo:
            memo[ccd] = maccs_167(smi)
        fp = memo[ccd]
        if fp is None:
            n['bad_mol'] += 1; bad.add(ccd); continue
        out[pid] = fp.copy()
    n['pids_with_maccs'] = len(out)
    return out, n, sorted(no_smi), sorted(bad)


def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f'.{path.name}.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def encode_npz(arrays: dict) -> bytes:
    """The npz file bytes np.savez writes for `arrays` (fixed zip times: reproducible)."""
    import io
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def same_content(path: Path, arrays: dict) -> tuple:
    """(identical?, description) of an existing npz against `arrays` (keys, order, values)."""
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        if keys != list(arrays):
            only = sorted(set(keys) ^ set(arrays))
            return False, (f'keys differ: {len(keys)} vs {len(arrays)}; only in one {len(only)} '
                           f'{only[:5]}; order equal {sorted(keys) == sorted(arrays)}')
        diff = [k for k in keys if z[k].dtype != arrays[k].dtype or z[k].shape != arrays[k].shape
                or not np.array_equal(z[k], arrays[k])]
    if diff:
        return False, f'{len(diff)} fingerprints differ, e.g. {diff[:5]}'
    return True, f'{len(keys)} fingerprints identical (keys, order, dtype, values)'


def write_sidecar(out: Path, counts: dict, no_smi, bad, arrays: dict, mode: str) -> Path:
    import rdkit
    st = AVRO.stat()
    info = {'schema': SCHEMA, 'builder': 'data_prep/build_misato_ligand_maccs.py',
            'origin': 'submission_exp/src/build_misato_ligand_features.py',
            'mode': mode, 'rdkit_version': rdkit.__version__, 'n_bits': N_BITS,
            'skip_ccds': sorted(SKIP_CCDS), 'counts': counts,
            'ccds_without_smiles': no_smi, 'ccds_unparsable': bad,
            'source': {'path': str(AVRO), 'size': st.st_size, 'sha256': file_sha256(AVRO)},
            'smiles_cache': {'path': _rel(SMILES_CACHE), 'sha256': file_sha256(SMILES_CACHE)},
            'output': _rel(out), 'output_sha256': file_sha256(out),
            'content_sha256': hash_arrays_by_pid(arrays),
            'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    side = Path(str(out) + '.provenance.json')
    _atomic_bytes(side, json.dumps(info, indent=1, sort_keys=True).encode())
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=str(OUT_NPZ))
    ap.add_argument('--overwrite', action='store_true', help='replace an existing --out file')
    ap.add_argument('--fetch', action='store_true',
                    help='resolve CCDs missing from the SMILES cache with the PDBe / RCSB APIs '
                         '(default: offline, a missing CCD is an error)')
    ap.add_argument('--verify-existing', action='store_true',
                    help='rebuild in memory, compare the content with --out, write only the '
                         'sidecar if identical (exit 1 otherwise)')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not AVRO.exists():
        print(f'ERROR: {AVRO} missing (set ENSEMBITS_MISATO_AVRO)', file=sys.stderr); return 1
    import fastavro
    with open(AVRO, 'rb') as f:
        pid_to_ccd, counts = ligand_codes(fastavro.reader(f))
    cache = json.loads(SMILES_CACHE.read_text()) if SMILES_CACHE.exists() else {}
    todo = sorted(set(pid_to_ccd.values()) - set(cache))
    if todo and not a.fetch:
        print(f'ERROR: {len(todo)} CCDs are not in {SMILES_CACHE.name} (e.g. {todo[:5]}); '
              f'pass --fetch to resolve them online', file=sys.stderr)
        return 1
    for i, ccd in enumerate(todo):
        cache[ccd] = fetch_smiles(ccd)
        time.sleep(SLEEP_BETWEEN)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            _atomic_bytes(SMILES_CACHE, json.dumps(cache, indent=2, sort_keys=True).encode())
    arrays, n2, no_smi, bad = fingerprints(pid_to_ccd, cache)
    counts.update(n2)
    print(counts, flush=True)
    if a.verify_existing:
        if not out.exists():
            print(f'ERROR: {out} does not exist', file=sys.stderr); return 1
        if out.read_bytes() != encode_npz(arrays):
            ok, why = same_content(out, arrays)
            print(f'DIFFERENT bytes; content: {why}', file=sys.stderr); return 1
        side = write_sidecar(out, counts, no_smi, bad, arrays,
                             'verified existing file: byte-identical rebuild')
        print(f'identical ({len(arrays)} fingerprints); wrote {side}', flush=True)
        return 0
    if out.exists() and not a.overwrite:
        print(f'ERROR: {out} exists (pass --overwrite or --verify-existing)', file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(out, encode_npz(arrays))
    side = write_sidecar(out, counts, no_smi, bad, arrays, 'built')
    print(f'wrote {out} and {side.name}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
