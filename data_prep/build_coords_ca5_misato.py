"""Collect the first 5 of the 8 MISATO C-alpha frames into one pickle (baseline input).

What it does
------------
Reads `ca_8` (the C-alpha of the 8 FPS frames) from every file of the
downstream MISATO backbone cache, keeps the first 5 frames and writes one dict
{pid: (5, L, 3) float32}. The single-frame mini3di builder
(`data_prep/baselines/build_mini3di_misato.py`) uses frame 0 of it. (The
original docstring also listed `build_protprofile_K5_misato.py`, which is not
part of this codebase; the K=8 ProtProfileMD feature is built from the
backbone cache directly.)

Paper
-----
`appendix.tex` \\label{app:baselines} (3Di_tokens: "applied to the single
representative conformation of each protein").

Origin
------
Copied from `submission_exp/src/build_coords_ca5_misato.py`. Changed: `ROOT`
from `ensembits.paths.REPO_ROOT`; refuses input written by the old MISATO
parser (code_fix §4, via `ensembits.data_versions.require_misato_parser`)
unless --allow-old-parser; refuses to overwrite an existing output (the same
treatment as `build_coords_ca8_misato.py`); comments and this docstring. 2026-10-16:
`--overwrite` to replace an existing output (code_fix P15); provenance sidecar
(code_fix P28). The parser version is now checked on every file read (it was
a sample of 50).

Used by the reported model: no (3Di_tokens / mini3di baseline input). Must be
rebuilt after the MISATO parser fix (code_fix §4 / T2.2).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/misato_real_bb/<pid>.npz   (ca_8)
  writes  data/cached_descriptors/coords_ca5_misato.pkl
          data/cached_descriptors/coords_ca5_misato.pkl.provenance.json   (code_fix P28)

How to run
----------
    python data_prep/build_coords_ca5_misato.py [--allow-old-parser] [--overwrite]
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
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.data_versions import require_misato_parser   # noqa: E402
from ensembits.data_versions import (MISATO_PARSER_VERSION, hash_arrays_by_pid,  # noqa: E402
                                     write_provenance)

BB_DIR = ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'
OUT = ROOT / 'data' / 'cached_descriptors' / 'coords_ca5_misato.pkl'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-old-parser', action='store_true',
                    help='accept input written before the code_fix §4 parser fix')
    # FIXED(code_fix P15, 2026-10-16): an existing output is refused unless --overwrite
    #   (old: always refused, no way to replace it from the command line).
    ap.add_argument('--overwrite', action='store_true',
                    help='replace an existing coords_ca5_misato.pkl')
    args = ap.parse_args()
    if not BB_DIR.exists():
        print(f'missing {BB_DIR} — run build_misato_real_bb.py first', flush=True)
        return 1
    if OUT.exists() and not args.overwrite:
        print(f'{OUT} exists; move it aside to rebuild (or pass --overwrite)', flush=True)
        return 1
    # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
    require_misato_parser(BB_DIR, allow_old=args.allow_old_parser)
    files = sorted(BB_DIR.glob('*.npz'))
    print(f'[ca5] reading {len(files)} files from {BB_DIR.name}', flush=True)
    coords: dict[str, np.ndarray] = {}
    versions, src_K, src_fps = set(), set(), set()
    t0 = time.time()
    for i, p in enumerate(files):
        pid = p.stem
        d = np.load(p, allow_pickle=False)
        # FIXED(code_fix P28, 2026-10-16): parser version of every file read (old: the
        #   sample of require_misato_parser only); recorded in the sidecar.
        v = str(d['parser_version']) if 'parser_version' in d.files else None
        if v != MISATO_PARSER_VERSION and not args.allow_old_parser:
            print(f'{p.name}: parser {v!r} != {MISATO_PARSER_VERSION!r}; rebuild misato_real_bb',
                  flush=True)
            return 1
        versions.add(v)
        src_K.add(int(d['K']) if 'K' in d.files else None)
        src_fps.add(str(d['fps']) if 'fps' in d.files else None)
        coords[pid] = d['ca_8'][:5].astype(np.float32)   # first 5 of the 8 FPS frames
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(files)} ({time.time() - t0:.0f}s)', flush=True)
    print(f'  total: {len(coords)} pids', flush=True)
    with open(OUT, 'wb') as f:
        pickle.dump(coords, f)
    print(f'wrote {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)', flush=True)
    # FIXED(code_fix P28, 2026-10-16): provenance sidecar (old: none); not inside the
    #   pickle, whose readers iterate its keys as pids.
    side = write_provenance(OUT, {
        'builder': 'data_prep/build_coords_ca5_misato.py', 'output': str(OUT),
        'source_dir': str(BB_DIR), 'source_field': 'ca_8[:5]',
        'parser_version': sorted(versions, key=str),
        'source_K': sorted(src_K, key=str), 'source_fps': sorted(src_fps, key=str),
        'n_pids': len(coords), 'coords_sha256': hash_arrays_by_pid(coords),
        'created': time.strftime('%Y-%m-%d %H:%M:%S')})
    print(f'wrote {side.name}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
