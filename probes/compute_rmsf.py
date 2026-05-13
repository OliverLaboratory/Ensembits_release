"""Compute per-residue Cα RMSF labels from a multi-frame ensemble.

Two flavours, switchable by --align:

  --align iterative   Iterative Kabsch alignment to the running mean
                      (default; 3 rounds).  Reference-frame-free.

  --align frame0      Single Kabsch alignment of every frame to frame 0.
                      Cheaper; appropriate when P is small (e.g. 8).

Input layout: a pickled `dict[str, ndarray]` mapping protein id → Cα
coordinate tensor of shape (P, L, 3).  Output: an `.npz` with the same
pid keys, each value a `(L,) float32` of per-residue RMSF in Å.

The math, in either flavour:
    RMSF_r = sqrt( mean_p ‖ X̃^{(p)}_r − X̄_r ‖² )
where `X̃` is the aligned coordinates and `X̄` is their mean over the
P frames.  Implemented as `var(axis=0).sum(axis=1).sqrt()`.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def kabsch_to(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P (L,3) onto Q (L,3) via Kabsch (proper rotation)."""
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (P - Pc) @ R.T + Qc


def compute_rmsf_iterative(ca: np.ndarray, n_rounds: int = 3) -> np.ndarray:
    """Iteratively align all frames to the running mean (Procrustes)."""
    ca = ca.astype(np.float64).copy()
    P = ca.shape[0]
    for _ in range(n_rounds):
        ref = ca.mean(0)
        for p in range(P):
            ca[p] = kabsch_to(ca[p], ref)
    ref = ca.mean(0)
    dev = ca - ref[None]
    return np.sqrt((dev ** 2).sum(-1).mean(0)).astype(np.float32)


def compute_rmsf_frame0(ca: np.ndarray) -> np.ndarray:
    """Align every frame to frame 0; cheaper, fine for small P."""
    ca = ca.astype(np.float64)
    P = ca.shape[0]
    aligned = np.zeros_like(ca)
    aligned[0] = ca[0]
    for p in range(1, P):
        aligned[p] = kabsch_to(ca[p], ca[0])
    var = aligned.var(axis=0)         # (L, 3)
    return np.sqrt(var.sum(axis=1)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coords", required=True, type=Path,
                     help="pickle: dict[pid -> (P, L, 3) ndarray] of Cα coords")
    ap.add_argument("--out", required=True, type=Path,
                     help=".npz output: dict[pid -> (L,) float32 RMSF]")
    ap.add_argument("--align", default="iterative",
                     choices=("iterative", "frame0"),
                     help="iterative Kabsch-to-mean (default) or single frame-0 align")
    ap.add_argument("--n-rounds", type=int, default=3,
                     help="iterations for --align iterative (default 3)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        print(f"output exists: {args.out}", file=sys.stderr); return 0
    coords = pickle.load(open(args.coords, "rb"))
    pids = sorted(coords.keys())
    P_first = next(iter(coords.values())).shape[0]
    print(f"[rmsf] {len(pids)} proteins, ~{P_first} frames each, "
          f"align={args.align}", flush=True)

    rmsf: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, pid in enumerate(pids):
        ca = np.asarray(coords[pid], dtype=np.float64)   # (P, L, 3)
        if args.align == "iterative":
            rmsf[pid] = compute_rmsf_iterative(ca, n_rounds=args.n_rounds)
        else:
            rmsf[pid] = compute_rmsf_frame0(ca)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pids)} ({time.time()-t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **rmsf)
    all_v = np.concatenate(list(rmsf.values()))
    print(f"wrote {args.out}  ({len(rmsf)} pids; "
          f"mean={all_v.mean():.3f}  median={float(np.median(all_v)):.3f}  "
          f"90%={float(np.percentile(all_v, 90)):.3f}  max={all_v.max():.3f} Å)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
