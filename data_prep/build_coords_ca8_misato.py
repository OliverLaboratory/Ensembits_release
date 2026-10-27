"""Collect the 8-frame MISATO Cα coordinates into one pickle.

What it does
------------
Reads `ca_8` (the Cα of the 8 FPS frames) from every file of the downstream
MISATO backbone cache and writes one dict {pid: (8, L, 3) float32}. This pickle
feeds the MISATO RMSF labels (`build_misato_rmsf_labels.py`), the CA-only
baselines and the old CA-only tokenization path.

Paper
-----
`appendix.tex` \\label{app:downstreams} (MISATO downstream tasks use 8 MD frames)
and the MISATO RMSF probe (`2_results.tex` \\label{result:rmsf}).

Origin
------
Copied from `submission_exp/src/build_coords_ca8_misato.py`. Changed: `ROOT`
from `ensembits.paths.REPO_ROOT`; refuses input written by the old MISATO
parser (code_fix §4) unless --allow-old-parser; refuses to overwrite an
existing output; comments and this docstring. 2026-10-16 (code_fix P28): writes
a provenance sidecar (parser version, source directory, n_pids, sha256 of the
coordinates and of the frame indices, K / fps of the source files).

Used by the reported model: yes (MISATO RMSF labels and baselines). Must be
rebuilt after the MISATO parser fix (code_fix §4 / T2.2).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/misato_real_bb/<pid>.npz   (ca_8)
  writes  data/cached_descriptors/coords_ca8_misato.pkl
          data/cached_descriptors/coords_ca8_misato.pkl.provenance.json   (code_fix P28)

How to run
----------
    python data_prep/build_coords_ca8_misato.py
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse, pickle, sys, time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in data_prep/
from build_misato_real_bb import PARSER_VERSION  # noqa: E402
from ensembits.data_versions import hash_arrays_by_pid, write_provenance  # noqa: E402

BB_DIR = ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'
OUT = ROOT / 'data' / 'cached_descriptors' / 'coords_ca8_misato.pkl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-old-parser', action='store_true',
                    help='accept input written before the code_fix §4 parser fix')
    args = ap.parse_args()
    if not BB_DIR.exists():
        print(f'missing {BB_DIR} — run build_misato_real_bb.py first', flush=True)
        return 1
    if OUT.exists():
        print(f'{OUT} exists; move it aside to rebuild', flush=True)
        return 1
    files = sorted(BB_DIR.glob('*.npz'))
    print(f'[ca8] reading {len(files)} files from {BB_DIR.name}', flush=True)
    coords: dict[str, np.ndarray] = {}
    frames: dict[str, np.ndarray] = {}            # code_fix P28: frame indices per pid
    versions, src_K, src_fps = set(), set(), set()
    t0 = time.time()
    for i, p in enumerate(files):
        d = np.load(p, allow_pickle=False)
        # code_fix §4: the old parser put proline Cδ in the CA slot.
        v = str(d['parser_version']) if 'parser_version' in d.files else None
        if v != PARSER_VERSION and not args.allow_old_parser:
            print(f'{p.name}: parser {v!r} != {PARSER_VERSION!r}; rebuild misato_real_bb', flush=True)
            return 1
        coords[p.stem] = d['ca_8'].astype(np.float32)
        versions.add(v)
        src_K.add(int(d['K']) if 'K' in d.files else None)
        src_fps.add(str(d['fps']) if 'fps' in d.files else None)
        if 'chosen_indices' in d.files:
            frames[p.stem] = d['chosen_indices']
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(files)} ({time.time() - t0:.0f}s)', flush=True)
    print(f'  total: {len(coords)} pids', flush=True)
    with open(OUT, 'wb') as f:
        pickle.dump(coords, f)
    print(f'wrote {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)', flush=True)
    # FIXED(code_fix P28, 2026-10-16): provenance sidecar (old: none). Kept out of the
    #   pickle, whose readers iterate its keys as pids.
    side = write_provenance(OUT, {
        'builder': 'data_prep/build_coords_ca8_misato.py', 'output': str(OUT),
        'source_dir': str(BB_DIR), 'source_field': 'ca_8',
        'parser_version': sorted(versions, key=str),
        'source_K': sorted(src_K, key=str), 'source_fps': sorted(src_fps, key=str),
        'n_pids': len(coords), 'coords_sha256': hash_arrays_by_pid(coords),
        'frame_indices_sha256': hash_arrays_by_pid(frames) if frames else None,
        'n_pids_with_frame_indices': len(frames),
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    print(f'wrote {side.name}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
