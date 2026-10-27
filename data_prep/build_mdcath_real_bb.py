"""Real backbones (N, CA, C, O, Cβ) of the 10 mdCATH-div frames, residues in file order.

What it does
------------
For each mdCATH-div representative (`_manifest.json` of ENSEMBITS_MDCATH_DIV_SRC), reads
the all-atom 320 K trajectories of `data/mdcath/data_full/data/mdcath_dataset_<dom>.h5`
and writes `data/cached_descriptors/mdcath_real_bb/<dom>.npz` with

    bb_10      (10, L, 4, 3) float32   N, CA, C, O of the 10 stored frames (ca_10 order)
    cb_10      (10, L, 3)    float32   Cβ (NaN for Gly / no CB atom)
    seq        one letter per residue (see "Sequence convention")
    match_dist (10,) float32           Frobenius norm |Cα(raw frame) - ca_10[k]|
    ca_idx     (L,) int64              index of each residue's CA among the ATOM records

the same keys, dtypes and conventions as the files written by the research builder.

Residues are identified in FILE ORDER by (chain, resSeq, iCode) of the ATOM records of
`pdbProteinAtoms`, and a residue is the one that owns a CA atom, exactly as the research
mdCATH-div builder (`scripts/build_mdcath_div.py`, `load_320K_frames`) selected the Cα
of ca_5 / ca_10: every ATOM record named "CA", in file order. So L and the residue order
match ca_10, ca_5 and the manifest by construction. Each residue must have N, C and O.

Frames: each of the 10 stored `ca_10` frames is looked up among the raw 320 K frames
(the stride-10 frames the mdCATH-div builder sampled, then every frame), and the first
frame with max |Δ| <= `--match-tol` Å (1e-3) is used; otherwise the domain fails.

FIXED(preretrain_audit_r2 mdCATH re-extraction, 2026-10-17): old (research
`submission_exp/src/build_mdcath_real_bb.py`) = residues keyed by the residue NUMBER only
and sorted by it. In 6 domains residue numbers repeat with insertion codes, so residues
were merged and the count did not match ca_10 (e.g. 1a6tB01 113 vs 117); in 1fwxA01 and
1ggpA02 the numbers are not monotonic in the file, so sorting broke the file order; these
8 domains had no file and were scored from ideal-geometry backbones or excluded. New =
file order by (chain, resSeq, iCode), which reproduces the 2,434 existing files exactly
(`--verify-existing`) and gives the 8 missing ones.

Sequence convention: the existing files map the 20 standard residue names (and HIS) to
one letter and every other name to 'X'; mdCATH uses CHARMM names, so histidines (HSD /
HSE / HSP) are 'X' and no file contains 'H'. This builder keeps that convention (decision
2026-10-17) so the new files are identical in convention to the 2,434 existing ones and
`rmsf_common.normalize_mdcath_real_bb_seq` (X -> H, refuses a sequence that already has
H) stays valid for every consumer; the convention is recorded in the run sidecar.

Paper
-----
`appendix.tex` \\label{app:dataset} \\paragraph{mdCATH-div (evaluation).} (the 2,442
representatives, real backbones of the 10 FPS frames; Conventions: no ideal-geometry
approximation for the datasets of the paper).

Origin
------
Port of `submission_exp/src/build_mdcath_real_bb.py` (research repo) with the residue
identification fixed as above (FIXED marker above), plus: `--verify-existing`,
`--only-missing` / `--pids`, max |Δ| frame check, atomic writes, refusal to overwrite,
a provenance sidecar per run, exit 1 on any failure or mismatch.

Used by the reported model: yes (mdCATH-div descriptor cache, RMSF probes, baselines).

Inputs / outputs
----------------
  reads   data/external/mdcath_div/{_manifest.json,<dom>.npz} (ca_10)   (ENSEMBITS_MDCATH_DIV_SRC)
          data/mdcath/data_full/data/mdcath_dataset_<dom>.h5            (--h5-dir)
          data/cached_descriptors/mdcath_real_bb/<dom>.npz              (--verify-existing)
  writes  <out>/<dom>.npz (refuses an existing file), <out>/_extract_<time>.provenance.json

How to run
----------
    # check: rebuild the 2,434 existing domains elsewhere and compare (exit 1 on a mismatch)
    python data_prep/build_mdcath_real_bb.py --verify-existing --out <scratch dir> --workers 16
    # write the missing domains (the 8 of 2026-10-17) into data/cached_descriptors/mdcath_real_bb
    python data_prep/build_mdcath_real_bb.py --only-missing
    python data_prep/build_mdcath_real_bb.py --pids 1a6tB01,1ggpA02 [--out <dir>]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT, MDCATH_DIV_SRC  # noqa: E402
from ensembits.data_versions import write_provenance  # noqa: E402

H5_DIR = ROOT / 'data' / 'mdcath' / 'data_full' / 'data'
OUT_DIR = ROOT / 'data' / 'cached_descriptors' / 'mdcath_real_bb'
TEMPERATURE = '320'
STRIDE = 10                      # frames sampled by the mdCATH-div builder
MATCH_TOL = 1e-3                 # Å, max |Δ| between a stored ca_10 frame and a raw frame
SEQ_CONVENTION = ('20 standard names + HIS -> one letter, every other name (CHARMM '
                  'HSD/HSE/HSP, ...) -> X; as the existing files')
_RES3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def parse_residues(pdb_atoms: str):
    """ATOM records of `pdbProteinAtoms` -> list of residues in file order.

    A residue is keyed by (chain, resSeq, iCode) (columns 22, 23-26, 27) and kept if it
    owns an atom named CA; atom indices count ATOM records only (= coords dim 1), as in
    load_320K_frames of the mdCATH-div builder. Returns [(key, resname, {atom: index})].
    Raises ValueError on a residue key with two CA atoms or a CA order that differs from
    the order of the CA records.
    """
    res, order, ca_records = {}, [], []
    atom_i = 0
    for line in pdb_atoms.split('\n'):
        if not line.startswith('ATOM'):
            continue
        name = line[12:16].strip()
        key = (line[21:22], line[22:26].strip(), line[26:27].strip())
        if key not in res:
            res[key] = (line[17:21].strip(), {})
            order.append(key)
        atoms = res[key][1]
        if name == 'CA':
            if 'CA' in atoms:
                raise ValueError(f'residue {key} has two CA atoms')
            ca_records.append(atom_i)
        atoms.setdefault(name, atom_i)
        atom_i += 1
    out = [(k, res[k][0], res[k][1]) for k in order if 'CA' in res[k][1]]
    if [a['CA'] for _, _, a in out] != ca_records:
        raise ValueError('residue order (chain, resSeq, iCode) differs from the CA record order')
    return out


def extract_domain(dom: str, src: Path, h5_dir: Path, tol: float = MATCH_TOL):
    """-> (fields dict for the npz, info) or raises ValueError / KeyError."""
    import h5py
    with np.load(Path(src) / f'{dom}.npz', allow_pickle=False) as d:
        target = np.asarray(d['ca_10'], dtype=np.float32)              # (10, L, 3)
    K, L, _ = target.shape
    with h5py.File(Path(h5_dir) / f'mdcath_dataset_{dom}.h5', 'r') as f:
        g = f[dom]
        pa = g['pdbProteinAtoms'][()]
        pa = pa.decode() if isinstance(pa, bytes) else str(pa)
        residues = parse_residues(pa)
        if len(residues) != L:
            raise ValueError(f'{len(residues)} residues with a CA, ca_10 has {L}')
        missing = [k for k, _, a in residues if not all(n in a for n in ('N', 'C', 'O'))]
        if missing:
            raise ValueError(f'residues without N/C/O: {missing[:5]}')
        idx = {n: np.array([a[n] for _, _, a in residues], dtype=np.int64)
               for n in ('N', 'CA', 'C', 'O')}
        cb_idx = np.array([a.get('CB', -1) for _, _, a in residues], dtype=np.int64)
        seq = ''.join(_RES3_TO_1.get(rn, 'X') for _, rn, _ in residues)
        if TEMPERATURE not in g:
            raise KeyError(f'{dom}: no {TEMPERATURE} K data')
        reps = sorted(g[TEMPERATURE].keys())
        strided = {rep: np.asarray(g[TEMPERATURE][rep]['coords'][::STRIDE]) for rep in reps}
        full = {}
        bb = np.empty((K, L, 4, 3), dtype=np.float32)
        cb = np.full((K, L, 3), np.nan, dtype=np.float32)
        match = np.empty(K, dtype=np.float32)
        where = []
        present = cb_idx >= 0

        def first_hit(arr, k):
            dmax = np.abs(arr[:, idx['CA']] - target[k][None]).reshape(len(arr), -1).max(1)
            hit = np.where(dmax <= tol)[0]
            return int(hit[0]) if len(hit) else None

        for k in range(K):
            found = None
            # the stride-10 frames the mdCATH-div builder sampled, replicas in sorted order
            for rep in reps:
                h = first_hit(strided[rep], k)
                if h is not None:
                    found = (rep, h * STRIDE, strided[rep][h]); break
            if found is None:                     # then every frame
                for rep in reps:
                    if rep not in full:
                        full[rep] = np.asarray(g[TEMPERATURE][rep]['coords'][()])
                    h = first_hit(full[rep], k)
                    if h is not None:
                        found = (rep, h, full[rep][h]); break
            if found is None:
                raise ValueError(f'stored ca_10 frame {k} not found among the raw 320 K frames '
                                 f'(tolerance {tol} Å)')
            rep, t, frame = found
            where.append([rep, int(t)])
            for j, n in enumerate(('N', 'CA', 'C', 'O')):
                bb[k, :, j] = frame[idx[n]]
            if present.any():
                cb[k, present] = frame[cb_idx[present]]
            match[k] = float(np.linalg.norm(frame[idx['CA']].astype(np.float32) - target[k]))
    fields = {'bb_10': bb, 'cb_10': cb, 'seq': np.array(seq), 'match_dist': match,
              'ca_idx': idx['CA']}
    info = {'L': L, 'frames': where, 'match_max_abs_A': float(match.max()),
            'n_residues_with_icode': sum(1 for k, _, _ in residues if k[2]),
            'n_chains': len({k[0] for k, _, _ in residues})}
    return fields, info


def compare_with(fields: dict, path: Path) -> list:
    """Differences between freshly extracted fields and a stored file ([] = identical)."""
    out = []
    with np.load(path, allow_pickle=False) as z:
        for k in ('bb_10', 'cb_10', 'ca_idx'):
            a, b = fields[k], z[k]
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=(k != 'ca_idx')):
                out.append(k)
        if str(z['seq']) != str(fields['seq']):
            out.append('seq')
        if not np.allclose(z['match_dist'], fields['match_dist'], atol=1e-4):
            out.append('match_dist')
    return out


def process(dom: str, src: str, h5_dir: str, out_dir: str, existing_dir: str | None,
            tol: float):
    """-> (dom, status, info). status: ok / identical / mismatch: ... / exists / error: ..."""
    outp = Path(out_dir) / f'{dom}.npz'
    if outp.exists():
        return dom, 'exists', None
    try:
        fields, info = extract_domain(dom, Path(src), Path(h5_dir), tol)
    except Exception as e:
        return dom, f'error: {type(e).__name__}: {e}', None
    if existing_dir is not None:
        diff = compare_with(fields, Path(existing_dir) / f'{dom}.npz')
        status = 'identical' if not diff else f'mismatch: {diff}'
    else:
        status = 'ok'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tmp = Path(out_dir) / f'.{dom}.npz.tmp{os.getpid()}'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **fields)
    os.replace(tmp, outp)
    return dom, status, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', default=str(MDCATH_DIV_SRC))
    ap.add_argument('--h5-dir', default=str(H5_DIR))
    ap.add_argument('--existing', default=str(OUT_DIR),
                    help='folder of the existing files (default data/cached_descriptors/mdcath_real_bb)')
    ap.add_argument('--out', default=None, help='output folder (default: --existing)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--verify-existing', action='store_true',
                   help='rebuild every domain that has a file in --existing into --out (must '
                        'differ) and compare; exit 1 on any difference')
    g.add_argument('--only-missing', action='store_true',
                   help='write the domains of the manifest without a file in --existing')
    g.add_argument('--pids', default=None, help='comma-separated domains to write')
    ap.add_argument('--match-tol', type=float, default=MATCH_TOL)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='first N domains only (tests)')
    a = ap.parse_args(argv)
    existing = Path(a.existing)
    out = Path(a.out) if a.out else existing
    pids_all = list(json.loads((Path(a.src) / '_manifest.json').read_text())['representatives'])
    have = {p for p in pids_all if (existing / f'{p}.npz').exists()}
    if a.verify_existing:
        if out.resolve() == existing.resolve():
            print('[error] --verify-existing needs --out different from --existing'); return 1
        pids = [p for p in pids_all if p in have]
    elif a.only_missing:
        pids = [p for p in pids_all if p not in have]
    else:
        pids = [p.strip() for p in a.pids.split(',') if p.strip()]
        unknown = [p for p in pids if p not in pids_all]
        if unknown:
            print(f'[error] not in the manifest: {unknown}'); return 1
    if a.limit:
        pids = pids[:a.limit]
    clash = [p for p in pids if (out / f'{p}.npz').exists()]
    if clash:
        print(f'[error] {len(clash)} output files exist in {out} (e.g. {clash[0]}); never '
              f'overwritten'); return 1
    print(f'[mdcath-real-bb] {len(pids)} domains -> {out}'
          + (' (verify against ' + str(existing) + ')' if a.verify_existing else ''), flush=True)
    args = [(p, a.src, a.h5_dir, str(out), str(existing) if a.verify_existing else None,
             a.match_tol) for p in pids]
    t0 = time.time()
    if a.workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(process, *zip(*args)))
    else:
        res = [process(*x) for x in args]
    status = {d: s for d, s, _ in res}
    infos = {d: i for d, _, i in res if i is not None}
    bad = {d: s for d, s in status.items() if s not in ('ok', 'identical')}
    for d, s in sorted(bad.items())[:50]:
        print(f'  {d}: {s}', flush=True)
    counts = {}
    for s in status.values():
        k = s.split(':')[0]; counts[k] = counts.get(k, 0) + 1
    print(f'[mdcath-real-bb] {counts} in {time.time() - t0:.0f}s', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out / f'_extract_{time.strftime("%Y%m%d_%H%M%S")}', {
        'builder': 'data_prep/build_mdcath_real_bb.py', 'mode': (
            'verify-existing' if a.verify_existing else 'only-missing' if a.only_missing else 'pids'),
        'src': a.src, 'h5_dir': a.h5_dir, 'existing': str(existing), 'out': str(out),
        'match_tol_A': a.match_tol, 'stride_first': STRIDE, 'temperature_K': int(TEMPERATURE),
        'residue_key': '(chain, resSeq, iCode) in file order; residues owning a CA atom',
        'seq_convention': SEQ_CONVENTION, 'counts': counts, 'status': status, 'info': infos,
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
