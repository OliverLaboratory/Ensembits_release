"""Per-residue Cα RMSF labels for mdCATH-div, over all 320 K MD frames, converged superposition.

What it does
------------
For each of the 2,442 mdCATH-div representatives (manifest order, the row order of
every mdCATH-div cache), takes the Cα coordinates of a set of frames and
  1. superposes every frame on the first frame (Kabsch),
  2. iterates: mean structure -> superpose every frame on it -> new mean, until the
     mean moves by less than `--tol` Å (default 1e-6) or `--max-iter` rounds,
  3. RMSF_i = sqrt(mean over frames of |x_i - mean_i|^2).
Writes one flat float32 array (308,293 residues) and a provenance sidecar.

Frames (`--label`):
  * `allframes` (default; user decision 2026-10-16, supersedes preretrain_audit_r2 Q1):
    EVERY saved 320 K frame of all replicas (no stride; 5 replicas of ~440–450 frames)
    from the raw mdCATH file `data/mdcath/data_full/data/mdcath_dataset_<domain>.h5`
    (`<domain>/320/<replica>/coords`, Å). Cα atoms are selected exactly as the research
    builder of mdCATH-div (`scripts/build_mdcath_div.py`, `load_320K_frames`): the ATOM
    records of `pdbProteinAtoms` whose atom name is "CA", in file order. Each domain's Cα
    count must equal its `ca_10` length, and each of the stored `ca_10` frames (the
    frames the models read) must be found among the raw frames (max |Δ| ≤ `--match-tol`
    Å), so rows and residues match the existing mdCATH-div rows. -> rmsf_div_allframes.npy
  * `converged5`: the 5 FPS frames `ca_5` of the mdCATH-div source (the label of the
    morning of 2026-10-16; reproduction only). -> rmsf_div_converged.npy

Why converged: the old label (`rmsf_div.npy`, research `compute_rmsf`) started from the
mean of the RAW frames and stopped after 3 rounds. The frames come from different
replicas and sit 60–120 Å apart in raw coordinates, so that start is arbitrary and 3
rounds do not converge (pre-retrain audit 2026-10-16, P3). This label does not depend on
the frames' starting poses (tests/test_rmsf_labels.py).

Paper
-----
`2_results.tex` \\label{result:rmsf} (label definition: Cα RMSF over the 320 K MD
frames of all replicas, superposed on their mean structure).

Origin
------
New (2026-10-16, code_fix P3, user decision). Reworked the same day (user decision:
every RMSF label uses ALL MD frames): `--label allframes` (default, raw h5) added, the
5-frame label kept as `--label converged5`; `converged_rmsf` superposes all frames in one
batched SVD per round (same mathematics as the per-frame loop it replaces). The Cα
selection follows the research script `scripts/build_mdcath_div.py` (read-only).

Used by the reported model: the mdCATH-div RMSF probe labels for every method
(`downstream/probe_rmsf_eval.py`, `downstream/probe_rmsf_baselines.py`, `--rmsf-label`).

Inputs / outputs
----------------
  reads   data/external/mdcath_div/_manifest.json, <pid>.npz (ca_5, ca_10)   (ENSEMBITS_MDCATH_DIV_SRC)
          data/mdcath/data_full/data/mdcath_dataset_<pid>.h5   (allframes; --h5-dir)
  writes  output/phase_b/cached_desc/rmsf_div_allframes.npy (+ .provenance.json)   allframes
          output/phase_b/cached_desc/rmsf_div_converged.npy (+ .provenance.json)   converged5
          refuses to overwrite unless --overwrite; written atomically

How to run
----------
    python data_prep/build_mdcath_div_rmsf_labels.py [--label allframes] [--workers 16] [--check-legacy]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))
from ensembits.paths import REPO_ROOT, MDCATH_DIV_SRC, MDCATH_DIV_RMSF_LEGACY  # noqa: E402
from ensembits.data_versions import write_provenance  # noqa: E402

N_ROWS = 308_293      # residues of the 2,442 representatives (row count of every mdCATH-div cache)
H5_DIR = REPO_ROOT / 'data' / 'mdcath' / 'data_full' / 'data'
TEMPERATURE = '320'


def label_content_sha256(arr) -> str:
    from rmsf_common import label_content_sha256 as _h
    return _h(arr)


def out_path_of(label: str) -> Path:
    from rmsf_common import RMSF_LABEL_FILES
    return Path(RMSF_LABEL_FILES['mdcath_div'][label])


def kabsch_onto(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Superpose P (L, 3) onto Q (L, 3): optimal rotation + translation, no reflection."""
    return kabsch_onto_batch(np.asarray(P)[None], Q)[0]


def kabsch_onto_batch(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Superpose every frame of A (F, L, 3) onto Q (L, 3) (rotation + translation)."""
    A = np.asarray(A, dtype=np.float64); Q = np.asarray(Q, dtype=np.float64)
    Ac = A - A.mean(1, keepdims=True); Qm = Q.mean(0); Qc = Q - Qm
    H = np.einsum('fli,lj->fij', Ac, Qc)
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1)); Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ Ut))
    D = np.zeros((len(A), 3, 3)); D[:, 0, 0] = 1.0; D[:, 1, 1] = 1.0; D[:, 2, 2] = d
    R = V @ D @ Ut
    return np.einsum('fli,fji->flj', Ac, R) + Qm


def converged_rmsf(ca: np.ndarray, tol: float = 1e-6, max_iter: int = 1000):
    """ca (F, L, 3) -> (rmsf (L,), n_iter, converged). Pose-independent by construction:
    the first round superposes everything on frame 0."""
    X = np.asarray(ca, dtype=np.float64)
    A = kabsch_onto_batch(X, X[0])
    mean = A.mean(0)
    for it in range(1, max_iter + 1):
        A = kabsch_onto_batch(A, mean)
        new = A.mean(0)
        shift = float(np.abs(new - mean).max())
        mean = new
        if shift < tol:
            break
    else:
        return np.sqrt(((A - mean) ** 2).sum(-1).mean(0)), max_iter, False
    return np.sqrt(((A - mean) ** 2).sum(-1).mean(0)), it, True


def ca_atom_indices(pdb_protein_atoms: str) -> np.ndarray:
    """Indices (within the ATOM records = coords dim 1) of the atoms named CA.

    Same selection as `load_320K_frames` of the research script scripts/build_mdcath_div.py,
    which built ca_5 / ca_10 of mdCATH-div.
    """
    idx, atom_i = [], 0
    for line in pdb_protein_atoms.split('\n'):
        if line.startswith('ATOM'):
            if line[12:16].strip() == 'CA':
                idx.append(atom_i)
            atom_i += 1
    return np.array(idx, dtype=np.int64)


def load_320K_ca_all_frames(h5_path, domain: str):
    """All 320 K frames of all replicas: (F, L, 3) float64 Cα and {replica: n_frames}."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        g = f[domain]
        if TEMPERATURE not in g:
            raise KeyError(f'{domain} has no {TEMPERATURE} K data')
        pa = g['pdbProteinAtoms'][()]
        pa = pa.decode() if isinstance(pa, bytes) else str(pa)
        ca_idx = ca_atom_indices(pa)
        frames, per_rep = [], {}
        for rep in sorted(g[TEMPERATURE].keys()):
            c = g[TEMPERATURE][rep]['coords'][:]            # (T, N_atoms, 3), Å
            frames.append(c[:, ca_idx].astype(np.float64)); per_rep[rep] = int(c.shape[0])
    return np.concatenate(frames, axis=0), per_rep


def stored_frames_found(all_ca: np.ndarray, stored: np.ndarray, tol: float) -> float:
    """Largest over stored frames of (min over raw frames of max |Δ|); ≤ tol = all found."""
    worst = 0.0
    for fr in np.asarray(stored, dtype=np.float64):
        d = np.abs(all_ca - fr[None]).reshape(len(all_ca), -1).max(1).min()
        worst = max(worst, float(d))
    return worst


def label_one(pid: str, src: str, label: str, h5_dir: str, tol: float, max_iter: int,
              match_tol: float):
    """-> (pid, rmsf float32 (L,) or None, info dict or error string)."""
    try:
        with np.load(os.path.join(src, f'{pid}.npz'), allow_pickle=True) as d:
            ca5 = np.asarray(d['ca_5']); ca10 = np.asarray(d['ca_10'])
        L = ca5.shape[1]
        if label == 'converged5':
            r, n_it, ok = converged_rmsf(ca5, tol, max_iter)
            return pid, r.astype(np.float32), {'n_frames': int(ca5.shape[0]), 'iter': n_it, 'ok': ok}
        h5 = Path(h5_dir) / f'mdcath_dataset_{pid}.h5'
        all_ca, per_rep = load_320K_ca_all_frames(h5, pid)
        if all_ca.shape[1] != L:
            return pid, None, f'{all_ca.shape[1]} CA atoms in the h5, ca_5 has {L}'
        worst = stored_frames_found(all_ca, ca10, match_tol)
        if worst > match_tol:
            return pid, None, f'stored ca_10 frame not found among the raw frames (|Δ| {worst:.3g} Å)'
        r, n_it, ok = converged_rmsf(all_ca, tol, max_iter)
        return pid, r.astype(np.float32), {'n_frames': int(len(all_ca)), 'per_replica': per_rep,
                                           'iter': n_it, 'ok': ok, 'match_max_abs': worst}
    except Exception as e:   # reported per domain; any failure fails the build
        return pid, None, f'{type(e).__name__}: {e}'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="allframes", choices=("allframes", "converged5"))
    ap.add_argument("--src", default=str(MDCATH_DIV_SRC))
    ap.add_argument("--h5-dir", default=str(H5_DIR), help="raw mdCATH h5 files (allframes)")
    ap.add_argument("--out", default=None, help="default by --label (rmsf_common.RMSF_LABEL_FILES)")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--match-tol", type=float, default=1e-3,
                    help="max |Δ| (Å) for finding a stored ca_10 frame among the raw frames")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-rows", type=int, default=N_ROWS, help="expected rows (tests)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-legacy", action="store_true",
                    help="print how the new label differs from the legacy rmsf_div.npy")
    args = ap.parse_args(argv)
    src = Path(args.src)
    out = Path(args.out) if args.out else out_path_of(args.label)
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to rebuild"); return 1
    pids = list(json.loads((src / "_manifest.json").read_text())["representatives"].keys())
    jobs = [(p, str(src), args.label, args.h5_dir, args.tol, args.max_iter, args.match_tol)
            for p in pids]
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            results = pool.starmap(label_one, jobs, chunksize=4)
    else:
        results = [label_one(*j) for j in jobs]
    parts, infos, failed = [], {}, {}
    for pid, r, info in results:            # manifest order (starmap keeps the order)
        if r is None:
            failed[pid] = info; continue
        if not info['ok']:
            failed[pid] = f'did not converge in {args.max_iter} rounds'; continue
        parts.append(r); infos[pid] = info
    if failed:
        print(f"[error] {len(failed)} domains failed: {list(failed.items())[:10]}"); return 1
    rmsf = np.concatenate(parts)
    if rmsf.shape[0] != args.n_rows:
        print(f"[error] {rmsf.shape[0]} rows, expected {args.n_rows}"); return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f'.{out.name}.tmp')
    with open(tmp, 'wb') as fh:
        np.save(fh, rmsf)
    os.replace(tmp, out)
    iters = [i['iter'] for i in infos.values()]
    nfr = [i['n_frames'] for i in infos.values()]
    write_provenance(out, {
        "builder": "data_prep/build_mdcath_div_rmsf_labels.py",
        "label_name": args.label,
        "label": ("mdCATH-div per-residue C-alpha RMSF over every 320 K frame of all replicas, "
                  "converged superposition" if args.label == "allframes" else
                  "mdCATH-div per-residue C-alpha RMSF over the 5 ca_5 frames, converged superposition"),
        "source": str(src), "h5_dir": args.h5_dir if args.label == "allframes" else None,
        "temperature_K": int(TEMPERATURE), "frames": "all (no stride)" if args.label == "allframes" else "ca_5",
        "n_pids": len(pids), "n_rows": int(rmsf.shape[0]), "tol_A": args.tol, "max_iter": args.max_iter,
        "iterations_max": int(max(iters)), "iterations_median": float(np.median(iters)),
        "n_frames_min": int(min(nfr)), "n_frames_median": float(np.median(nfr)),
        "n_frames_max": int(max(nfr)),
        "n_frames_per_pid": {p: i['n_frames'] for p, i in infos.items()},
        "match_tol_A": args.match_tol if args.label == "allframes" else None,
        "row_order": "_manifest.json representatives",
        # FIXED(preretrain_audit_r2 F5, 2026-10-16): content hash, recorded by the probes
        "labels_sha256": label_content_sha256(rmsf),
    })
    print(f"wrote {out} ({rmsf.shape[0]} rows; frames median {np.median(nfr):.0f}; iterations "
          f"median {np.median(iters):.0f}, max {max(iters)})")
    if args.check_legacy and Path(MDCATH_DIV_RMSF_LEGACY).exists():
        old = np.load(MDCATH_DIV_RMSF_LEGACY)
        d = np.abs(old - rmsf)
        print(f"vs legacy: max |diff| {d.max():.3f} A; residues > 0.5 A: {(d > 0.5).mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
