"""Per-residue Cα RMSF labels for MISATO: all 100 MD frames (default) or the old 8 frames.

What it does
------------
`--label allframes` (default; user decision 2026-10-16, supersedes preretrain_audit_r2
Q1 / item 8): for each complex, the Cα of ALL 100 saved frames of `data/misato/MD.hdf5`,
selected with the same parser as the backbone cache (`build_misato_real_bb.parse_complex`:
protein chains only, CX Cα, same residue order as `misato_real_bb/<PID>.npz`), with the
converged superposition of `build_mdcath_div_rmsf_labels.converged_rmsf` (superpose on
frame 0, then iterate on the mean until it moves < 1e-6 Å). Where a K = 8 backbone file
exists, its `ca_8` must equal the parsed Cα of its `chosen_indices` frames exactly (same
residues, same order, same frames), else the complex fails. Writes
`misato_rmsf_allframes.npz` {PID: (L,) float32} and its sidecar (label_name, parser
version, MD.hdf5 path / size, n_frames, iterations, number of complexes checked against
misato_real_bb, labels sha256). Any failed complex (listed) exits 1 with nothing written,
unless `--max-fail-frac` allows it (default 0; G3 2026-10-17, was 0.5%).

`--label frames8` (the old label, reproduction only): superposes the 8 frames of
`coords_ca8_misato.pkl` on the first (Kabsch) and takes the per-residue Cα RMSF
sqrt(sum over x, y, z of the variance over frames) -> `misato_rmsf.npz` (unchanged).

Paper
-----
`2_results.tex` \\label{result:rmsf} (MISATO RMSF probe); label definition is
require_fix #21.

Origin
------
Copied from `submission_exp/src/build_misato_rmsf_labels.py`. Changed: `ROOT`
from `ensembits.paths.REPO_ROOT`; an existing output is refused instead of
silently kept (code_fix C5: the old labels hold proline Cδ positions); comments
and this docstring. 2026-10-16 (code_fix P28): command-line options (`--coords`,
`--out`, `--allow-no-provenance`); the coordinates must come with the provenance
sidecar written by `build_coords_ca8_misato.py` and match its hash; the labels get
their own sidecar. 2026-10-16 (user decision, all-frames labels): `--label allframes`
(new default, MD.hdf5, converged superposition, no coords_ca8 dependency); the old
computation is `--label frames8`.

Used by the reported model: yes (MISATO RMSF probe labels, `--rmsf-label` of
`downstream/probe_rmsf_misato.py`). Must be built after the MISATO parser fix
(code_fix §4 / T2.2): proline Cα were Cδ.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  allframes  reads   data/misato/MD.hdf5, data/misato_splits_v2.json (complexes, --pids),
                     data/cached_descriptors/misato_real_bb/<PID>.npz (consistency check)
             writes  data/cached_descriptors/misato_rmsf_allframes.npz (+ .provenance.json)
  frames8    reads   data/cached_descriptors/coords_ca8_misato.pkl (+ .provenance.json)
             writes  data/cached_descriptors/misato_rmsf.npz (+ .provenance.json)
  (refuses an existing output; written atomically)

How to run
----------
    python data_prep/build_misato_rmsf_labels.py [--label allframes] [--workers 16]
    python data_prep/build_misato_rmsf_labels.py --label frames8 [--coords PKL] [--out NPZ]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo
from ensembits.data_versions import (hash_arrays_by_pid, read_provenance,  # noqa: E402
                                     write_provenance)

COORDS = ROOT / 'data' / 'cached_descriptors' / 'coords_ca8_misato.pkl'
OUT = ROOT / 'data' / 'cached_descriptors' / 'misato_rmsf.npz'                    # frames8
OUT_ALLFRAMES = ROOT / 'data' / 'cached_descriptors' / 'misato_rmsf_allframes.npz'
# FIXED(benchmark_audit D3, 2026-10-18): default = ensembits.paths.MISATO_SPLITS (data/misato_splits_v2.json;
#   same 4,894 complexes as the old misato_splits.json).
from ensembits.paths import MISATO_SPLITS as SPLITS  # noqa: E402
REAL_BB_DIR = ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'
SPLIT_NAMES = ('sequence', 'structure', 'random')
MAX_FAIL_FRAC = 0.0   # G3 (2026-10-17): default tolerance 0 (was 0.5%); --max-fail-frac
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling data_prep scripts


def allframes_one(pid: str, h5_path: str, real_bb_dir: str, tol: float = 1e-6,
                  max_iter: int = 1000):
    """-> (pid, rmsf (L,) float32 or None, info dict or error string). All 100 frames."""
    import h5py
    from build_misato_real_bb import parse_complex
    from build_mdcath_div_rmsf_labels import converged_rmsf
    try:
        with h5py.File(h5_path, 'r', swmr=True) as f:
            if pid not in f:
                return pid, None, 'not in MD.hdf5'
            g = f[pid]
            parsed = parse_complex(g)
            if parsed is None:
                return pid, None, 'no protein residues'
            ca = g['trajectory_coordinates'][:][:, parsed['ca'], :]   # (100, L, 3)
        checked = False
        rb = Path(real_bb_dir) / f'{pid}.npz'
        if rb.exists():
            with np.load(rb, allow_pickle=False) as z:
                ca8 = np.asarray(z['ca_8']); ch = np.asarray(z['chosen_indices'])
            if ca8.shape[1] != ca.shape[1] or not np.array_equal(
                    ca8, ca[ch].astype(np.float32)):
                return pid, None, 'misato_real_bb ca_8 != parsed Cα of its chosen frames'
            checked = True
        r, n_it, ok = converged_rmsf(ca, tol, max_iter)
        if not ok:
            return pid, None, f'did not converge in {max_iter} rounds'
        return pid, r.astype(np.float32), {'n_frames': int(ca.shape[0]), 'iter': n_it,
                                           'checked_real_bb': checked}
    except Exception as e:
        return pid, None, f'{type(e).__name__}: {e}'


def main_allframes(args) -> int:
    """--label allframes (see the module docstring)."""
    from build_misato_real_bb import H5_PATH, PARSER_VERSION
    h5 = Path(args.h5 or H5_PATH)
    out = Path(args.out or OUT_ALLFRAMES)
    if out.exists():
        print(f'{out} exists; move it aside to rebuild'); return 1
    if args.pids:
        pids = [p.strip().upper() for p in Path(args.pids).read_text().split() if p.strip()]
    else:
        s = json.loads(Path(args.splits).read_text())
        pids = sorted({p.upper() for sp in SPLIT_NAMES for k in ('train', 'val', 'test')
                       for p in s[sp].get(k, [])})
    print(f'[rmsf-misato allframes] {len(pids)} complexes from {h5.name}', flush=True)
    jobs = [(p, str(h5), str(args.real_bb_dir)) for p in pids]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            res = pool.starmap(allframes_one, jobs, chunksize=8)
    else:
        res = [allframes_one(*j) for j in jobs]
    rmsf, infos, failed = {}, {}, {}
    for pid, r, info in res:
        if r is None: failed[pid] = info
        else: rmsf[pid] = r; infos[pid] = info
    print(f'  ok {len(rmsf)}, failed {len(failed)}', flush=True)
    # FIXED(preretrain_audit_r2 G3, 2026-10-17): every failure is listed; the default
    #   tolerance is 0 like every other builder (--max-fail-frac). A failed complex has no
    #   label and is excluded for every method by build_rmsf_residue_sets.py (counted there).
    for pid, why in sorted(failed.items()):
        print(f'  FAILED {pid}: {why}', flush=True)
    if len(failed) > args.max_fail_frac * max(len(pids), 1):
        print(f'[error] {len(failed)} of {len(pids)} complexes failed (> {args.max_fail_frac:.2%}); '
              f'nothing written'); return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f'.{out.name}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **rmsf)
    os.replace(tmp, out)
    st = h5.stat() if h5.exists() else None
    nfr = [i['n_frames'] for i in infos.values()]
    side = write_provenance(out, {
        'builder': 'data_prep/build_misato_rmsf_labels.py', 'output': str(out),
        'label_name': 'allframes',
        'label': 'per-residue Cα RMSF over all MD.hdf5 frames, converged superposition',
        'parser_version': PARSER_VERSION, 'source_h5': str(h5),
        'source_h5_size': st.st_size if st else None,
        'source_h5_mtime_ns': st.st_mtime_ns if st else None,
        'n_pids': len(rmsf), 'n_failed': len(failed), 'failed': failed,
        'max_fail_frac': args.max_fail_frac,     # K4 (2026-10-17): read by the residue-set builder
        'n_frames_min': int(min(nfr)) if nfr else None,
        'n_frames_max': int(max(nfr)) if nfr else None,
        'iterations_max': int(max(i['iter'] for i in infos.values())) if infos else None,
        'n_checked_against_real_bb': int(sum(i['checked_real_bb'] for i in infos.values())),
        'real_bb_dir': str(args.real_bb_dir),
        'labels_sha256': hash_arrays_by_pid(rmsf),
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    print(f'wrote {out.name} and {side.name}')
    return 0


def kabsch_align(P, Q):
    """Superpose P (L, 3) onto Q (L, 3) (optimal rotation + translation)."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return Pc @ R.T + Q.mean(0)


def compute_rmsf(ca):
    """ca: (P, L, 3) -> (L,) RMSF after superposing every frame on frame 0."""
    P, L, _ = ca.shape
    aligned = np.zeros_like(ca)
    aligned[0] = ca[0]
    for p in range(1, P):
        aligned[p] = kabsch_align(ca[p], ca[0])
    var = aligned.var(axis=0)          # (L, 3) variance over frames
    return np.sqrt(var.sum(axis=1))    # (L,)


def main(argv=None):
    # FIXED(code_fix P28, 2026-10-16): command line (old: none; fixed paths). The
    #   defaults are the old fixed paths.
    ap = argparse.ArgumentParser(description='Per-residue Cα RMSF labels for MISATO.')
    # FIXED(user decision 2026-10-16, all-frames labels): old = the 8-frame label only.
    #   New = --label allframes (default) / frames8 (old computation, unchanged).
    ap.add_argument('--label', default='allframes', choices=('allframes', 'frames8'))
    ap.add_argument('--coords', default=str(COORDS),
                    help='frames8: coords pickle {pid: (8, L, 3)} (default: %(default)s)')
    ap.add_argument('--out', default=None,
                    help=f'output npz (default: {OUT_ALLFRAMES.name} / {OUT.name} by --label)')
    ap.add_argument('--h5', default=None, help='allframes: MD.hdf5 (default: data/misato/MD.hdf5)')
    ap.add_argument('--splits', default=str(SPLITS), help='allframes: complexes = union of the splits')
    ap.add_argument('--pids', default=None, help='allframes: file with PIDs (overrides --splits)')
    ap.add_argument('--real-bb-dir', default=str(REAL_BB_DIR),
                    help='allframes: misato_real_bb (K = 8) for the consistency check')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--max-fail-frac', type=float, default=MAX_FAIL_FRAC,
                    help='allframes: allowed fraction of failed complexes (default 0)')
    ap.add_argument('--allow-no-provenance', action='store_true',
                    help='accept a coords pickle without the provenance sidecar of '
                         'build_coords_ca8_misato.py (pickles built before 2026-10-16)')
    args = ap.parse_args(argv)
    if args.label == 'allframes':
        return main_allframes(args)
    coords_path, out = Path(args.coords), Path(args.out or OUT)
    if out.exists():
        # code_fix C5: never keep stale labels silently.
        print(f'{out} exists; move it aside to rebuild (old files hold proline Cδ '
              f'positions, code_fix §4)'); return 1
    coords = pickle.load(open(coords_path, 'rb'))
    # FIXED(code_fix P28, 2026-10-16): the labels record which coordinates they were
    #   computed from (old: nothing). The coords pickle must carry the sidecar of
    #   build_coords_ca8_misato.py, and its recorded hash must match the content.
    coords_hash = hash_arrays_by_pid(coords)
    coords_prov = read_provenance(coords_path)
    if coords_prov is None and not args.allow_no_provenance:
        print(f'{coords_path} has no provenance sidecar (built before 2026-10-16, possibly '
              f'from the old MISATO parser); rebuild it with build_coords_ca8_misato.py, '
              f'or pass --allow-no-provenance'); return 1
    if coords_prov is not None and coords_prov.get('coords_sha256') != coords_hash:
        print(f'{coords_path} does not match its provenance sidecar (coords_sha256 '
              f'{coords_prov.get("coords_sha256")} != {coords_hash}); rebuild it'); return 1
    pids = sorted(coords.keys())
    print(f'[rmsf-misato] {len(pids)} proteins, 8 frames each', flush=True)
    rmsf = {}
    t0 = time.time()
    for i, pid in enumerate(pids):
        ca = coords[pid].astype(np.float64)  # (8, L, 3)
        rmsf[pid] = compute_rmsf(ca).astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(pids)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'wrote {len(rmsf)} pids')
    # written atomically (temporary name, then rename)
    tmp = out.with_name(f'.{out.name}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **rmsf)
    os.replace(tmp, out)
    all_v = np.concatenate(list(rmsf.values()))
    print(f'  RMSF distribution: mean={all_v.mean():.3f} median={np.median(all_v):.3f} '
          f'90%={np.percentile(all_v, 90):.3f} max={all_v.max():.3f} Å')
    print(f'wrote {out.name}')
    # FIXED(code_fix P28, 2026-10-16): provenance sidecar <out>.provenance.json (old:
    #   none); not inside the npz, whose readers iterate its keys as pids.
    cp = coords_prov or {}
    side = write_provenance(out, {
        'builder': 'data_prep/build_misato_rmsf_labels.py', 'output': str(out),
        'coords_file': str(coords_path), 'coords_sha256': coords_hash,
        'coords_provenance_found': coords_prov is not None,
        'parser_version': cp.get('parser_version'),
        'source_dir': cp.get('source_dir'), 'source_fps': cp.get('source_fps'),
        'frame_indices_sha256': cp.get('frame_indices_sha256'),
        'n_pids': len(rmsf), 'n_frames': sorted({int(np.shape(coords[p])[0]) for p in pids}),
        'labels_sha256': hash_arrays_by_pid(rmsf),
        'label': 'per-residue Cα RMSF after superposing every frame on frame 0 (Kabsch)',
        'label_name': 'frames8',
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    print(f'wrote {side.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
