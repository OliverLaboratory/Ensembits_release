"""Per-residue Cα RMSF labels for BioEmu-CATH over all MD frames, converged superposition.

What it does
------------
For every system of `data/cached_descriptors/bioemu_cath_real_bb/layout_rows.json`
(layout order), loads ALL frames of ALL runs (`trajs/run*_protein.cmprsd.xtc`, the
trajectories of the corpus; no subsampling) with the functions of
`build_bioemu_real_bb.py` (same residue filter N/CA/C/O and residue order as `bb_10`),
checks that the Cα count equals the layout L and that the 10 stored `bb_10` Cα frames
are among the loaded frames (max |Δ| <= --match-tol), and computes the per-residue Cα
RMSF with the converged superposition of `build_mdcath_div_rmsf_labels.converged_rmsf`
(superpose on frame 0, then iterate on the mean until it moves < 1e-6 Å).

Writes (in `output/phase_b/cached_desc/bioemu/`):
    rmsf_bioemu_allframes.npy   flat float32 (sum of L,), layout_rows order
    rmsf_bioemu_rows.json       row -> system map: [{system, pid, deposit, row_start, length}]
    rmsf_bioemu_allframes.npy.provenance.json  (n_frames per system, iterations, inputs)
Refuses to overwrite; atomic writes; exit 1 if any system fails.

Paper
-----
`2_results.tex` \\label{result:rmsf} (BioEmu-CATH column, phase 2; label = Cα RMSF over
all MD frames, converged superposition, as for mdCATH-div and MISATO).

Origin
------
New (2026-10-16, phase 2 and the all-frames label decision).

Used by the reported model: yes (BioEmu RMSF probe labels for every method).

Inputs / outputs
----------------
  reads   data/reviewer_3_special_corpus/bioemu_cath/extracted/{ONE_cath1,MSR_cath2}/<system>/
          data/cached_descriptors/bioemu_cath_real_bb/{layout_rows.json,<system>.npz}
  writes  output/phase_b/cached_desc/bioemu/rmsf_bioemu_allframes.npy (+ sidecar),
          output/phase_b/cached_desc/bioemu/rmsf_bioemu_rows.json

How to run
----------
    python data_prep/build_bioemu_rmsf_labels.py [--workers 16]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))
import build_bioemu_real_bb as BB   # noqa: E402
from build_mdcath_div_rmsf_labels import converged_rmsf, stored_frames_found  # noqa: E402
from rmsf_common import (BIOEMU_LAYOUT, BIOEMU_RAW_ROOT, BIOEMU_REAL_BB_DIR,  # noqa: E402
                         BIOEMU_ROWS, RMSF_LABEL_FILES, label_content_sha256)
from ensembits.data_versions import write_provenance  # noqa: E402

OUT = RMSF_LABEL_FILES['bioemu']['allframes']


def all_frames_ca(sysdir, top=None, xyz_runs=None) -> tuple:
    """(Cα of every frame of every run (T, L, 3) float64, frames per run)."""
    sysdir = Path(sysdir)
    if top is None:
        import mdtraj as md
        top = md.load_topology(str(sysdir / 'topology.pdb'))
    _, _, ca_idx, _, _ = BB.backbone_residues(top)
    if xyz_runs is None:
        xyz_runs = BB.load_runs_xyz(BB.system_xtcs(sysdir), top)
    xyz, run_n = xyz_runs
    return np.asarray(xyz[:, ca_idx, :], dtype=np.float64), list(run_n)


def label_system(row: dict, root: str, real_bb: str, tol: float, max_iter: int,
                 match_tol: float):
    """-> (system, rmsf (L,) float32 or None, info dict or error string)."""
    system = row['system']
    try:
        ca, run_n = all_frames_ca(Path(root) / row['deposit'] / system)
        if ca.shape[1] != int(row['L']):
            return system, None, f'{ca.shape[1]} residues, layout {row["L"]}'
        with np.load(Path(real_bb) / f'{system}.npz', allow_pickle=False) as z:
            stored = np.asarray(z['bb_10'])[:, :, 1]
        worst = stored_frames_found(ca, stored, match_tol)
        if worst > match_tol:
            return system, None, f'stored bb_10 frame not among the frames (|Δ| {worst:.3g} Å)'
        r, n_it, ok = converged_rmsf(ca, tol, max_iter)
        if not ok:
            return system, None, f'did not converge in {max_iter} rounds'
        return system, r.astype(np.float32), {'n_frames': int(ca.shape[0]),
                                               'run_n_frames': run_n, 'iter': int(n_it)}
    except Exception as e:
        return system, None, f'{type(e).__name__}: {e}'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=str(BIOEMU_RAW_ROOT))
    ap.add_argument('--real-bb', default=str(BIOEMU_REAL_BB_DIR))
    ap.add_argument('--layout', default=str(BIOEMU_LAYOUT))
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--rows-out', default=str(BIOEMU_ROWS))
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--max-iter', type=int, default=1000)
    ap.add_argument('--match-tol', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args(argv)
    out, rows_out = Path(a.out), Path(a.rows_out)
    for p in (out, rows_out):
        if p.exists():
            print(f'{p} exists; move it aside to rebuild'); return 1
    rows = json.loads(Path(a.layout).read_text())
    args = [(r, a.root, a.real_bb, a.tol, a.max_iter, a.match_tol) for r in rows]
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(label_system, *zip(*args)))
    else:
        res = [label_system(*x) for x in args]
    failed = {s: info for s, r, info in res if r is None}
    if failed:
        print(f'[error] {len(failed)} systems failed: {list(failed.items())[:10]}'); return 1
    parts, rowmap, infos, start = [], [], {}, 0
    for row, (system, r, info) in zip(rows, res):          # layout order
        parts.append(r); infos[system] = info
        rowmap.append({'system': system, 'pid': row['pid'], 'deposit': row['deposit'],
                       'row_start': start, 'length': int(len(r))})
        start += len(r)
    lab = np.concatenate(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f'.{out.name}.tmp')
    with open(tmp, 'wb') as fh:
        np.save(fh, lab)
    os.replace(tmp, out)
    tmp = rows_out.with_name(f'.{rows_out.name}.tmp')
    tmp.write_text(json.dumps(rowmap, indent=1)); os.replace(tmp, rows_out)
    nfr = [i['n_frames'] for i in infos.values()]
    write_provenance(out, {
        'builder': 'data_prep/build_bioemu_rmsf_labels.py', 'label_name': 'allframes',
        'label': 'BioEmu-CATH per-residue C-alpha RMSF over every frame of every run, '
                 'converged superposition',
        'root': a.root, 'trajectories': 'trajs/run*_protein.cmprsd.xtc (all frames)',
        'layout': a.layout, 'rows_file': str(rows_out), 'n_systems': len(rows),
        'n_rows': int(lab.shape[0]), 'tol_A': a.tol, 'max_iter': a.max_iter,
        'match_tol_A': a.match_tol, 'n_frames_per_system': {s: i['n_frames'] for s, i in infos.items()},
        'n_frames_min': int(min(nfr)), 'n_frames_median': float(np.median(nfr)),
        'iterations_per_system': {s: i['iter'] for s, i in infos.items()},
        'labels_sha256': label_content_sha256(lab),      # F5: recorded by the probes
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    print(f'wrote {out} ({lab.shape[0]} rows, {len(rows)} systems, frames median '
          f'{np.median(nfr):.0f}; {time.time() - t0:.0f}s) and {rows_out.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
