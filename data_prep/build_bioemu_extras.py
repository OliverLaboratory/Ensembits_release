"""Re-derive the BioEmu-CATH FPS picks and store their indices, sequence and real Cβ.

What it does
------------
For every system of `data/cached_descriptors/bioemu_cath_real_bb/layout_rows.json`
(1,090 systems), repeats the frame selection of `data_prep/build_bioemu_real_bb.py`
with that script's own functions (same residue filter N/CA/C/O, same trajectories
`trajs/run*_protein.cmprsd.xtc`, same uniform subsampling to <= 2,000 frames, same
`farthest_point_sample` and seed), and checks that the re-derived backbone equals the
stored `bb_10` exactly (max |Δ| = 0; otherwise the system fails). It then writes
`data/cached_descriptors/bioemu_cath_extras/<system>.npz` with

    pick_sub      (10,) index of each picked frame in the subsampled frames (FPS order)
    pick_concat   (10,) index in the concatenated frames of all runs (before subsampling)
    pick_run      (10,) run index (into run_files);  pick_frame (10,) frame within that run
    run_files     run trajectory file names, run_n_frames  frames per run
    res_index     (L,)  topology residue index of each kept residue
    seq           one letter per kept residue (force-field names such as HSD/HSE/HSP,
                  HID/HIE/HIP -> H, CYX -> C; rmsf_common.three_to_one)
    cb_10         (10, L, 3) real Cβ of the picked frames, Å (NaN for Gly / no CB atom)

and the provenance sidecar `bioemu_cath_extras.provenance.json` next to the folder.
An existing file is refused (unless --overwrite); files are written atomically. Exit 1
when any re-derived bb_10 differs from the stored one, or when a system of the BioEmu
RMSF set fails for another reason (`--max-fail-frac`, default 0; F7 2026-10-16). (The 4 extra short `<pid>_<i>.cmprsd.xtc` trajectories of some
ONE_cath1 systems are excluded, as for the corpus.)

Paper
-----
`appendix.tex` \\label{app:dataset} \\paragraph{BioEmu-CATH.} (the frame selection);
input of the BioEmu-CATH RMSF column (`2_results.tex` \\label{result:rmsf}) and of its
baselines (`appendix.tex` \\label{app:baselines}).

Origin
------
New (2026-10-16, phase 2: RMSF probe on BioEmu-CATH). Imports the functions of
`build_bioemu_real_bb.py` (no copy).

Used by the reported model: no (BioEmu RMSF probe inputs and baselines).

Inputs / outputs
----------------
  reads   data/reviewer_3_special_corpus/bioemu_cath/extracted/{ONE_cath1,MSR_cath2}/<system>/
              topology.pdb, trajs/run*_protein.cmprsd.xtc
          data/cached_descriptors/bioemu_cath_real_bb/{layout_rows.json,<system>.npz}
  writes  data/cached_descriptors/bioemu_cath_extras/<system>.npz
          data/cached_descriptors/bioemu_cath_extras.provenance.json

How to run
----------
    python data_prep/build_bioemu_extras.py [--workers 16] [--limit N] [--overwrite]
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
import build_bioemu_real_bb as BB   # noqa: E402  the frame selection of the corpus
from rmsf_common import (BIOEMU_EXTRAS_DIR, BIOEMU_LAYOUT, BIOEMU_RAW_ROOT,  # noqa: E402
                         BIOEMU_REAL_BB_DIR, three_to_one)
from ensembits.data_versions import write_provenance  # noqa: E402



def run_and_frame(concat_idx: np.ndarray, run_n_frames) -> tuple:
    """Concatenated-frame indices -> (run index, frame within the run)."""
    ends = np.cumsum(run_n_frames)
    run = np.searchsorted(ends, concat_idx, side='right')
    starts = np.concatenate([[0], ends[:-1]])
    return run.astype(np.int64), (np.asarray(concat_idx) - starts[run]).astype(np.int64)


def derive_extras(sysdir, k: int, max_frames: int, top=None, xyz_runs=None) -> dict:
    """Re-derive the picks of one system with build_bioemu_real_bb's functions.

    Returns the extras fields plus 'bb_10' (the re-derived backbone, for the check).
    `top` / `xyz_runs` = (xyz, run_n_frames) can be passed in (tests); otherwise they are
    loaded from topology.pdb and the run trajectories exactly as the corpus builder does.
    """
    sysdir = Path(sysdir)
    xtcs = BB.system_xtcs(sysdir)
    if top is None:
        import mdtraj as md
        top = md.load_topology(str(sysdir / 'topology.pdb'))
    res_ids, n_idx, ca_idx, c_idx, o_idx = BB.backbone_residues(top)
    if xyz_runs is None:
        xyz_runs = BB.load_runs_xyz(xtcs, top)
    xyz, run_n = xyz_runs
    T = xyz.shape[0]
    sel = BB.subsample_index(T, max_frames)
    xs = xyz[sel] if sel is not None else xyz
    pick = BB.pick_frames(xs[:, ca_idx, :], k)
    bb = np.stack([xs[pick][:, n_idx, :], xs[pick][:, ca_idx, :],
                   xs[pick][:, c_idx, :], xs[pick][:, o_idx, :]], axis=2).astype(np.float32)
    concat = (sel[pick] if sel is not None else pick).astype(np.int64)
    run, frame = run_and_frame(concat, run_n)
    residues = list(top.residues)
    seq = ''.join(three_to_one(residues[r].name) for r in res_ids)
    cb = np.full((len(pick), len(res_ids), 3), np.nan, dtype=np.float32)
    for j, r in enumerate(res_ids):
        cbs = [a.index for a in residues[r].atoms if a.name == 'CB']
        if cbs:
            cb[:, j] = xs[pick][:, cbs[0], :]
    return {'pick_sub': np.asarray(pick, np.int64), 'pick_concat': concat,
            'pick_run': run, 'pick_frame': frame,
            'run_files': np.array([Path(x).name for x in xtcs]),
            'run_n_frames': np.asarray(run_n, np.int64), 'n_frames_total': np.int64(T),
            'max_frames': np.int64(max_frames), 'k': np.int64(k), 'seed': np.int64(BB.SEED),
            'res_index': np.asarray(res_ids, np.int64), 'seq': np.array(seq),
            'cb_10': cb, 'bb_10': bb}


def process(row: dict, root: str, real_bb: str, out: str, k: int, max_frames: int,
            overwrite: bool):
    """-> (system, status); status 'ok', 'exists', 'mismatch: ...' or 'error: ...'."""
    system = row['system']
    outp = Path(out) / f'{system}.npz'
    if outp.exists() and not overwrite:
        return system, 'exists'
    try:
        ex = derive_extras(Path(root) / row['deposit'] / system, k, max_frames)
        with np.load(Path(real_bb) / f'{system}.npz', allow_pickle=False) as z:
            stored = np.asarray(z['bb_10'])
        bb = ex.pop('bb_10')
        if bb.shape != stored.shape:
            return system, f'mismatch: shape {bb.shape} != stored {stored.shape}'
        d = float(np.abs(bb - stored).max())
        if d != 0.0:
            return system, f'mismatch: max |Δ| {d:.3g} Å vs stored bb_10'
        if len(str(ex['seq'])) != int(row['L']):
            return system, f'mismatch: L {len(str(ex["seq"]))} != layout {row["L"]}'
        Path(out).mkdir(parents=True, exist_ok=True)
        tmp = Path(out) / f'.{system}.npz.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **ex)
        os.replace(tmp, outp)
        return system, 'ok'
    except Exception as e:
        return system, f'error: {type(e).__name__}: {e}'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=str(BIOEMU_RAW_ROOT))
    ap.add_argument('--real-bb', default=str(BIOEMU_REAL_BB_DIR))
    ap.add_argument('--layout', default=str(BIOEMU_LAYOUT))
    ap.add_argument('--out', default=str(BIOEMU_EXTRAS_DIR))
    ap.add_argument('--k', type=int, default=BB.K_DEFAULT)
    ap.add_argument('--max-frames', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    # FIXED(preretrain_audit_r2 F7, 2026-10-16): old = up to 0.5% failed systems tolerated;
    #   new = default 0 (every system is in the BioEmu RMSF set), explicit option.
    ap.add_argument('--max-fail-frac', type=float, default=0.0,
                    help='allowed fraction of systems that fail for a reason other than a '
                         're-derivation mismatch (default 0; a mismatch always fails)')
    a = ap.parse_args(argv)
    rows = json.loads(Path(a.layout).read_text())
    if a.limit:
        rows = rows[:a.limit]
    existing = [r['system'] for r in rows if (Path(a.out) / f"{r['system']}.npz").exists()]
    if existing and not a.overwrite:
        print(f'[error] {len(existing)} extras files exist in {a.out} (e.g. {existing[0]}); '
              f'pass --overwrite'); return 1
    t0 = time.time()
    args = [(r, a.root, a.real_bb, a.out, a.k, a.max_frames, a.overwrite) for r in rows]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(process, *zip(*args)))
    else:
        res = [process(*x) for x in args]
    status = dict(res)
    mism = {s: v for s, v in status.items() if v.startswith('mismatch')}
    err = {s: v for s, v in status.items() if v.startswith('error')}
    ok = sum(v == 'ok' for v in status.values())
    print(f'[bioemu-extras] {ok} ok, {len(mism)} mismatches, {len(err)} errors '
          f'({time.time() - t0:.0f}s)', flush=True)
    for s, v in list(mism.items())[:10] + list(err.items())[:10]:
        print(f'  {s}: {v}', flush=True)
    write_provenance(Path(a.out), {
        'builder': 'data_prep/build_bioemu_extras.py', 'root': a.root, 'real_bb': a.real_bb,
        'layout': a.layout, 'k': a.k, 'max_frames': a.max_frames, 'seed': BB.SEED,
        'trajectories': 'trajs/run*_protein.cmprsd.xtc', 'n_systems': len(rows), 'n_ok': ok,
        'mismatch': mism, 'errors': err,
        'check': 'bb_10 re-derived with build_bioemu_real_bb functions == stored (max |Δ| 0)',
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    from rmsf_common import check_builder_failures
    ok_tol = check_builder_failures(err, [r['system'] for r in rows], 'bioemu', a.max_fail_frac)
    if mism or not ok_tol:
        print('[error] re-derived picks differ from the stored bb_10, or systems of the RMSF '
              'set failed (listed above)')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
