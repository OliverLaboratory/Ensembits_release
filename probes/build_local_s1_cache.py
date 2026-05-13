"""Build a flat per-residue s_1 cache from a multi-frame Cα corpus.

Reads `--coords` (a pickled `dict[pid -> (P, L, 3) ndarray]`,
the same layout `compute_rmsf.py` consumes) and writes a single
`(N_residues,) float32` `s1.npy` whose rows are concatenated in the
sorted-pid order — i.e. row-by-row aligned with the descriptor
cache that `scripts/encode.py` produces, when both are run on the
same corpus.

Implements the exact recipe documented in
`per_residue_local_pca.py`. Run once per corpus; the result feeds
`token_dynamics.py`.

Usage:
    python -m probes.build_local_s1_cache \\
        --coords data/ca10.pkl \\
        --out    data/s1_per_residue.npy

If you also want the full 6-D feature row per residue (s_1 / s_2 /
s_3 + the principal direction v_1), pass --feats-out as well — same
N rows, 6 columns.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from probes.per_residue_local_pca import local_kabsch_s1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", required=True, type=Path,
                     help="pickle: dict[pid -> (P, L, 3) ndarray] of Cα coords.")
    ap.add_argument("--out", required=True, type=Path,
                     help="(N,) float32 .npy of per-residue s_1.")
    ap.add_argument("--feats-out", type=Path, default=None,
                     help="optional (N, 6) float32 .npy of full PCA features "
                          "(s_1, s_2, s_3, v_1[0..2]).")
    ap.add_argument("--ball-radius", type=float, default=10.0,
                     help="Cα ball radius in Å (default 10.0).")
    ap.add_argument("--ref-frame", type=int, default=0,
                     help="reference frame index (default 0).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        print(f"output exists: {args.out}", file=sys.stderr)
        return 0
    coords = pickle.load(open(args.coords, "rb"))
    pids = sorted(coords.keys())
    P_first, _, _ = next(iter(coords.values())).shape
    print(f"[s1] {len(pids)} proteins, ~{P_first} frames each, "
          f"ball={args.ball_radius} Å, ref_frame={args.ref_frame}",
          flush=True)

    feats = []
    t0 = time.time()
    for i, pid in enumerate(pids):
        ca = np.asarray(coords[pid], dtype=np.float32)  # (P, L, 3)
        feats.append(local_kabsch_s1(ca,
                                      ball_radius=args.ball_radius,
                                      ref_frame=args.ref_frame))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pids)} ({time.time()-t0:.0f}s)", flush=True)
    feats = np.concatenate(feats, axis=0)  # (N, 6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feats[:, 0].astype(np.float32))
    print(f"wrote {args.out}  (N={feats.shape[0]:,};  "
          f"mean s_1={feats[:, 0].mean():.3f},  "
          f"n_zero={(feats[:, 0] == 0).sum():,})")
    if args.feats_out is not None:
        np.save(args.feats_out, feats.astype(np.float32))
        print(f"wrote {args.feats_out}  ({feats.shape[0]:,} × 6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
