"""Add the 10-frame MISATO backbones (bb_10) that the current checkpoint trained on.

ONLY FOR REPRODUCING THE OLD CHECKPOINT'S DATA. Do not run it for the retrain.
SUPERSEDED (2026-10-15): the retrained model's corpus comes from
`build_misato_real_bb.py --K 10 --fps kabsch` (Kabsch FPS, code_fix T1.6). This
script is kept to document how the current checkpoint's MISATO frames were
chosen; it now uses the fixed atom parser (code_fix §4), so it no longer
reproduces the old proline atoms.
Since 2026-10-16 (code_fix P14) it refuses to run when any file in
`misato_real_bb/` carries a `parser_version` (a file of the fixed parser): it
would add Frobenius-FPS bb_10 frames to the new caches, which readers of bb_10
would then take for the Kabsch corpus.

What it does
------------
For each pid already in `data/cached_descriptors/misato_real_bb/<pid>.npz`
(written by `build_misato_real_bb.py` with K = 8), it re-parses the complex in
MD.hdf5, picks 10 of the 100 saved frames with the same `fps_select`
(greedy FPS on raw CA coordinates, Frobenius distance, no superposition,
seeded at frame 0; independent of the K = 8 selection), extracts the real
N, CA, C, O and CB of those frames, and saves them as extra fields
bb_10 / cb_10 / ca_10 / chosen_indices_10 next to the existing bb_8 / cb_8 / ...
fields. The original K = 8 fields stay intact. Files that already have bb_10
are skipped. ~10 min on 32 CPU workers.

Atom-index parsing reuses `find_residues` / `PROTEIN_N_TYPES` from
`build_misato_real_bb.py`.

Paper
-----
  * `appendix.tex`, \\subsection{Dataset} (\\label{app:dataset}),
    \\paragraph{MISATO.}: "select 10 of the 100 saved frames by FPS on the
    Cα coordinates, starting from the first frame".
  * `2_results.tex`, \\subsection{Datasets}: P_max = 10 frames per system.

Origin
------
Copied from `submission_exp/src/build_misato_real_bb_K10.py`. Changed: the
`ROOT` line now comes from `ensembits.paths.REPO_ROOT` (same directory as
before); the sibling import of `build_misato_real_bb` now finds it in this
folder (`data_prep/`) instead of `submission_exp/src/`; comments and this
docstring. Logic unchanged.

Used by the reported model: yes. The bb_10 field is the MISATO half of the
training corpus (read by `data_prep/build_bioemu_misato_descriptor.py`) and
of the MISATO downstream tasks.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/misato/MD.hdf5, data/cached_descriptors/misato_real_bb/<pid>.npz
  writes  the same <pid>.npz in place, adding
              bb_10 (10, L, 4, 3) [N, CA, C, O], cb_10 (10, L, 3) (NaN for GLY),
              ca_10 (10, L, 3), chosen_indices_10 (10,) frame indices in [0, 100)

How to run
----------
    python data_prep/build_misato_real_bb_K10.py [--workers 32] [--limit N]
"""
from __future__ import annotations
import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in data_prep/
from build_misato_real_bb import (find_residues, fps_select,
                                     PROTEIN_N_TYPES, H5_PATH)

# The real-backbone cache that feeds the reported model (bb_10 written here).
# Superseded for the retrain by misato_real_bb_kabsch (see the header).
OUT_DIR = ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'
K = 10      # Paper: appendix.tex \paragraph{MISATO.} — 10 of the 100 saved frames


def _process_pid(pid: str) -> tuple[str, str | None]:
    """Add the K = 10 fields to one existing <pid>.npz; returns (pid, status)."""
    out_path = OUT_DIR / f'{pid}.npz'
    if not out_path.exists():
        return pid, 'no_existing'
    try:
        existing = dict(np.load(out_path, allow_pickle=False))
    except Exception:
        return pid, 'corrupt'
    # FIXED(code_fix P14, 2026-10-16): never write into a file of the fixed parser
    #   (old: any file was extended). main() checks every file first; this is the
    #   per-file guard.
    if 'parser_version' in existing:
        return pid, 'refused_new_parser_file'
    if 'bb_10' in existing and 'cb_10' in existing:
        return pid, 'skip'

    # Re-parse the protein residues exactly as build_misato_real_bb.py does
    # (protein chains only, all chains concatenated into one residue list).
    try:
        with h5py.File(H5_PATH, 'r', swmr=True) as f:
            if pid not in f:
                return pid, 'no_pid_in_h5'
            g = f[pid]
            an = g['atoms_number'][:]
            at = g['atoms_type'][:]
            mb = list(g['molecules_begin_atom_index'][:])
            chain_ends = mb[1:] + [len(an)]

            residues_per_chain = []
            for mi in range(len(mb)):
                s = mb[mi]; e = chain_ends[mi]
                if an[s] != 7 or int(at[s]) not in PROTEIN_N_TYPES:
                    continue
                res = find_residues(an, at, s, e, g['trajectory_coordinates'][0])
                if not res: continue
                residues_per_chain.append(res)
            all_res = [r for chain in residues_per_chain for r in chain]
            L = len(all_res)
            if L == 0:
                return pid, 'no_residues'

            ca_idx = np.array([r['CA'] for r in all_res], dtype=np.int64)
            n_idx = np.array([r['N'] for r in all_res], dtype=np.int64)
            c_idx = np.array([r['C'] for r in all_res], dtype=np.int64)
            o_idx = np.array([r['O'] for r in all_res], dtype=np.int64)
            cb_present = np.array([r['CB'] is not None for r in all_res])
            cb_idx = np.array([r['CB'] if r['CB'] is not None else 0
                               for r in all_res], dtype=np.int64)

            # Sanity: existing L must match
            if 'bb_8' in existing and existing['bb_8'].shape[1] != L:
                return pid, f'L_mismatch {existing["bb_8"].shape[1]} vs {L}'

            # FPS on the raw CA coordinates of all 100 frames, seeded at frame 0.
            # Paper: appendix.tex \paragraph{MISATO.} — FPS on Cα coordinates,
            # starting from the first frame.
            # (The retrain uses Kabsch-aligned FPS instead: build_misato_real_bb.py --fps kabsch.)
            coords = g['trajectory_coordinates'][:]
            ca_traj = coords[:, ca_idx, :].astype(np.float32)
            chosen = fps_select(ca_traj, K)
            chosen_arr = np.array(chosen, dtype=np.int64)

            # Real backbone (and CB, NaN for glycine) of the chosen frames.
            sel = coords[chosen_arr].astype(np.float32)
            bb_K = np.empty((len(chosen), L, 4, 3), dtype=np.float32)
            bb_K[:, :, 0] = sel[:, n_idx]
            bb_K[:, :, 1] = sel[:, ca_idx]
            bb_K[:, :, 2] = sel[:, c_idx]
            bb_K[:, :, 3] = sel[:, o_idx]

            cb_K = np.full((len(chosen), L, 3), np.nan, dtype=np.float32)
            if cb_present.any():
                cb_pos = sel[:, cb_idx]
                cb_K[:, cb_present, :] = cb_pos[:, cb_present, :]

        # Rewrite the file with the K = 8 fields kept and the K = 10 fields added.
        existing['bb_10'] = bb_K
        existing['cb_10'] = cb_K
        existing['ca_10'] = bb_K[:, :, 1]
        existing['chosen_indices_10'] = chosen_arr
        np.savez_compressed(out_path, **existing)
        return pid, 'ok'
    except Exception as exc:
        return pid, f'err_{type(exc).__name__}_{str(exc)[:50]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    # FIXED(code_fix P14, 2026-10-16): refuse to run on caches of the fixed parser
    #   (old: no check). Every file is checked, not a sample.
    from ensembits.data_versions import misato_parser_version
    new_files = [p.name for p in sorted(OUT_DIR.glob('*.npz'))
                 if misato_parser_version(p) is not None]
    if new_files:
        print(f'{OUT_DIR} holds {len(new_files)} file(s) written by the fixed parser '
              f'(e.g. {new_files[0]}). This script only reproduces the old checkpoint\'s '
              f'data and would write Frobenius-FPS bb_10 into them; use '
              f'build_misato_real_bb.py --K 10 for new caches.', flush=True)
        return 1
    # Every complex already in the K = 8 cache gets the K = 10 fields.
    pids = sorted(p.stem for p in OUT_DIR.glob('*.npz'))
    if args.limit:
        pids = pids[:args.limit]
    print(f'extending {len(pids)} misato_real_bb files with K=10 '
          f'fields (workers={args.workers})', flush=True)

    counts = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_pid, p): p for p in pids}
        for i, fut in enumerate(as_completed(futures)):
            pid, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status not in ('ok', 'skip') and counts.get(status, 0) <= 3:
                print(f'  {pid}: {status}', flush=True)
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(pids) - i - 1) / max(rate, 1e-6)
                print(f'  [{i+1}/{len(pids)}] {time.time()-t0:.0f}s '
                      f'ok={counts.get("ok", 0)} ETA={eta:.0f}s', flush=True)

    print(f'\nDONE in {time.time()-t0:.0f}s — {counts}', flush=True)
    if counts.get('refused_new_parser_file', 0):     # code_fix P14 (see above)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
