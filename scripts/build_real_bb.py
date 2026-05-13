"""Build real-backbone caches from the raw mdCATH or MISATO h5 sources.

This is the one prerequisite for every structural baseline + the
Ensembits tokenizer: ``data/{mdcath,misato}_real_bb/<pid>.npz`` provides
real N/CA/C/O backbone (and Cβ when present) for every frame in the
FPS-subsampled ensemble. Without this cache the post-fix pipeline falls
back to ideal-geometry reconstruction (less accurate, all baselines'
numbers move by ~0.05 Spearman on the misato sweep).

**Most users should NOT need to run this.** The 142 MB mdCATH cache and
7.8 GB MISATO cache are shipped via Zenodo — see MANIFEST.md. Only run
this if you've cloned without the data drop and want to rebuild from
the raw 3.1 TB mdCATH h5 dir / 124 GB MISATO MD.hdf5.

Usage:

    python -m scripts.build_real_bb --dataset mdcath \\
        --h5-dir <path-to-mdcath_dataset_*.h5-files> \\
        --manifest <mdcath_div/_manifest.json>          \\
        --ca-cache-dir <mdcath_div/>                    \\
        --out data/mdcath_real_bb/                      \\
        --workers 32

    python -m scripts.build_real_bb --dataset misato \\
        --md-hdf5 <path-to-MD.hdf5> \\
        --out data/misato_real_bb/ \\
        --workers 32 --K 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

_RES3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# AMBER ff14SB backbone-N atom types in MISATO's protein chains.
_PROTEIN_N_TYPES = {24, 25, 26, 27, 28}

# Heavy-atom signature → 1-letter AA (MISATO sequence inference).
_HEAVY_SIG_TO_AA = {
    (0, 0, 0, 0): "G", (1, 0, 0, 0): "A", (1, 0, 1, 0): "S",
    (1, 0, 0, 1): "C", (2, 0, 1, 0): "T", (3, 0, 0, 1): "M",
    (2, 1, 1, 0): "N", (2, 0, 2, 0): "D", (4, 1, 0, 0): "K",
    (3, 0, 2, 0): "E", (3, 1, 1, 0): "Q", (4, 2, 0, 0): "H",
    (7, 0, 0, 0): "F", (4, 3, 0, 0): "R", (7, 0, 1, 0): "Y",
    (9, 1, 0, 0): "W",
}


# ============================================================
# mdCATH (per-domain h5 files)
# ============================================================

def _parse_mdcath_backbone(pdb_atoms_str: str) -> dict:
    """mdCATH ``pdbProteinAtoms`` → ``{resid: {N, CA, C, O, CB?, resname}}``."""
    bb = {}
    atom_i = 0
    for line in pdb_atoms_str.split("\n"):
        if not line.startswith("ATOM"):
            continue
        aname = line[12:16].strip()
        try:
            resid = int(line[22:26].strip())
        except ValueError:
            atom_i += 1
            continue
        resname = line[17:21].strip()
        if aname in ("N", "CA", "C", "O", "CB"):
            entry = bb.setdefault(resid, {"resname": resname})
            entry[aname] = atom_i
        atom_i += 1
    # Drop residues without full N/CA/C/O (caps like CAY etc).
    return {r: v for r, v in bb.items()
            if all(k in v for k in ("N", "CA", "C", "O"))}


def _build_one_mdcath(args):
    """Worker: extract bb_K + cb_K for one mdCATH domain."""
    dom, h5_dir, ca_cache_dir, out_dir, K, max_dist = args
    out_path = Path(out_dir) / f"{dom}.npz"
    if out_path.exists():
        return dom, "skip"
    h5_path = Path(h5_dir) / f"mdcath_dataset_{dom}.h5"
    if not h5_path.exists():
        return dom, "no_h5"
    ca_path = Path(ca_cache_dir) / f"{dom}.npz"
    if not ca_path.exists():
        return dom, "no_ca_cache"

    try:
        ca_K = np.load(ca_path, allow_pickle=False)[f"ca_{K}"].astype(np.float32)
        L = ca_K.shape[1]

        with h5py.File(h5_path, "r") as f:
            g = f[dom]
            patoms = g["pdbProteinAtoms"][()]
            if isinstance(patoms, bytes):
                patoms = patoms.decode()
            bb_idx = _parse_mdcath_backbone(patoms)
            if len(bb_idx) != L:
                return dom, f"L_mismatch_{len(bb_idx)}_vs_{L}"

            ordered = sorted(bb_idx.items())
            seq = "".join(_RES3_TO_1.get(v["resname"], "X") for _, v in ordered)
            n_idx = np.array([v["N"] for _, v in ordered], dtype=np.int64)
            ca_idx_arr = np.array([v["CA"] for _, v in ordered], dtype=np.int64)
            c_idx = np.array([v["C"] for _, v in ordered], dtype=np.int64)
            o_idx = np.array([v["O"] for _, v in ordered], dtype=np.int64)
            cb_idx = np.array([v.get("CB", -1) for _, v in ordered],
                              dtype=np.int64)
            cb_present = cb_idx >= 0

            if "320" not in g:
                return dom, "no_320K"

            # Find raw frames in the 320K trajectory matching ca_K.
            frames = []
            for rep in sorted(g["320"].keys()):
                coords = g["320"][rep]["coords"]
                for t in range(0, coords.shape[0], 10):  # stride 10
                    frame = coords[t]
                    frames.append((frame[ca_idx_arr].astype(np.float32), frame))

            bb_K = np.empty((K, L, 4, 3), dtype=np.float32)
            cb_K = np.full((K, L, 3), np.nan, dtype=np.float32)
            match_dist = np.empty(K, dtype=np.float32)
            for k in range(K):
                target = ca_K[k]
                best_d = np.inf; best = None
                for ca_f, frame in frames:
                    d = float(np.linalg.norm(ca_f - target))
                    if d < best_d:
                        best_d = d; best = frame
                        if d < 1e-3:
                            break
                if best is None or best_d > max_dist:
                    return dom, f"frame_{k}_no_match_{best_d:.2f}A"
                match_dist[k] = best_d
                bb_K[k, :, 0] = best[n_idx]
                bb_K[k, :, 1] = best[ca_idx_arr]
                bb_K[k, :, 2] = best[c_idx]
                bb_K[k, :, 3] = best[o_idx]
                if cb_present.any():
                    cb_safe = np.where(cb_present, cb_idx, 0)
                    cb_pos = best[cb_safe].astype(np.float32)
                    cb_pos[~cb_present] = np.nan
                    cb_K[k] = cb_pos

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path,
                             **{f"bb_{K}": bb_K,
                                f"cb_{K}": cb_K,
                                "seq": np.array(seq),
                                "match_dist": match_dist,
                                "ca_idx": ca_idx_arr})
        return dom, "ok"
    except Exception as e:
        return dom, f"err_{type(e).__name__}_{str(e)[:50]}"


# ============================================================
# MISATO (single MD.hdf5 file)
# ============================================================

def _find_misato_residues(an: np.ndarray, at: np.ndarray,
                          s: int, e: int) -> list[dict]:
    """Identify residues within atoms[s:e) by the peptide-bond pattern
    ``C(at=3) → O → N``; returns ``[{N, CA, C, O, CB}, ...]``."""
    starts = [s] if an[s] == 7 else []
    i = s
    while i < e - 1:
        if an[i] == 6 and at[i] == 3:
            j = i + 1
            while j < e and an[j] == 1:
                j += 1
            if j >= e or an[j] != 8:
                i += 1; continue
            k = j + 1
            while k < e and an[k] in (1, 8):
                k += 1
            if k >= e or an[k] != 7:
                i += 1; continue
            m = k + 1
            while m < e and an[m] == 1:
                m += 1
            if m < e and an[m] == 6 and at[m] != 3:
                starts.append(k); i = k; continue
        i += 1
    starts.append(e)

    out = []
    for r in range(len(starts) - 1):
        rs, re_ = starts[r], starts[r + 1]
        ni = rs
        ca_i = ni + 1
        while ca_i < re_ and an[ca_i] != 6:
            ca_i += 1
        if ca_i >= re_:
            return None
        c_cands = [k for k in range(ca_i + 1, re_) if an[k] == 6 and at[k] == 3]
        if not c_cands:
            return None
        ci = c_cands[-1]
        oi = ci + 1
        while oi < re_ and an[oi] == 1:
            oi += 1
        if oi >= re_ or an[oi] != 8:
            return None
        cb = None
        for k in range(ca_i + 1, ci):
            if an[k] == 6:
                cb = k; break
        out.append({"N": ni, "CA": ca_i, "C": ci, "O": oi, "CB": cb})
    return out


def _aa_from_misato_residue(an, frame0, res) -> str:
    """Heavy-atom signature → 1-letter AA (best-effort)."""
    ca, ci, ni = res["CA"], res["C"], res["N"]
    if res["CB"] is None:
        return "G"
    sc = [k for k in range(ca + 1, ci) if an[k] != 1]
    n_C = sum(1 for k in sc if an[k] == 6)
    n_N = sum(1 for k in sc if an[k] == 7)
    n_O = sum(1 for k in sc if an[k] == 8)
    n_S = sum(1 for k in sc if an[k] == 16)
    code = _HEAVY_SIG_TO_AA.get((n_C, n_N, n_O, n_S))
    if code is not None:
        return code
    if (n_C, n_N, n_O, n_S) == (3, 0, 0, 0):
        last_sc = sc[-1] if sc else None
        if last_sc is not None:
            d = float(np.linalg.norm(frame0[last_sc] - frame0[ni]))
            if d < 2.0:
                return "P"
        return "V"
    if (n_C, n_N, n_O, n_S) == (4, 0, 0, 0):
        cb_atom = res["CB"]
        h_count = 0
        k = cb_atom + 1
        while k < ci and an[k] == 1:
            h_count += 1; k += 1
        if h_count >= 2: return "L"
        if h_count == 1: return "I"
        return "X"
    return "X"


def _fps_select(ca_traj: np.ndarray, K: int) -> list[int]:
    """Greedy farthest-point sampling in Cα Frobenius distance, seeded from 0."""
    N = ca_traj.shape[0]
    selected = [0]
    dists = np.linalg.norm(ca_traj - ca_traj[0:1], axis=(1, 2))
    for _ in range(K - 1):
        nxt = int(np.argmax(dists))
        if nxt in selected:
            break
        selected.append(nxt)
        new_d = np.linalg.norm(ca_traj - ca_traj[nxt:nxt + 1], axis=(1, 2))
        dists = np.minimum(dists, new_d)
    return sorted(selected)


def _build_one_misato(args):
    """Worker: extract bb_K + cb_K + seq for one MISATO pid."""
    pid, md_hdf5, out_dir, K = args
    out_path = Path(out_dir) / f"{pid}.npz"
    if out_path.exists():
        return pid, "skip"
    try:
        with h5py.File(md_hdf5, "r", swmr=True) as f:
            if pid not in f:
                return pid, "no_pid"
            g = f[pid]
            an = g["atoms_number"][:]
            at = g["atoms_type"][:]
            mb = list(g["molecules_begin_atom_index"][:])
            chain_ends = mb[1:] + [len(an)]

            residues_per_chain: list[list[dict]] = []
            for mi in range(len(mb)):
                s_ = mb[mi]; e_ = chain_ends[mi]
                if an[s_] != 7 or int(at[s_]) not in _PROTEIN_N_TYPES:
                    continue
                res = _find_misato_residues(an, at, s_, e_)
                if not res:
                    continue
                residues_per_chain.append(res)
            all_res = [r for chain in residues_per_chain for r in chain]
            L = len(all_res)
            if L == 0:
                return pid, "no_residues"

            ca_idx = np.array([r["CA"] for r in all_res], dtype=np.int64)
            n_idx = np.array([r["N"] for r in all_res], dtype=np.int64)
            c_idx = np.array([r["C"] for r in all_res], dtype=np.int64)
            o_idx = np.array([r["O"] for r in all_res], dtype=np.int64)
            cb_present = np.array([r["CB"] is not None for r in all_res])
            cb_idx = np.array([r["CB"] if r["CB"] is not None else 0
                               for r in all_res], dtype=np.int64)

            ar = g["atoms_residue"][:]
            res_type_int = ar[n_idx].astype(np.int32)

            coords = g["trajectory_coordinates"][:]   # (100, n_atoms, 3) f64
            ca_traj = coords[:, ca_idx, :].astype(np.float32)
            chosen = _fps_select(ca_traj, K)
            chosen_arr = np.array(chosen, dtype=np.int64)

            sel_coords = coords[chosen_arr].astype(np.float32)
            bb_K = np.empty((len(chosen), L, 4, 3), dtype=np.float32)
            bb_K[:, :, 0] = sel_coords[:, n_idx]
            bb_K[:, :, 1] = sel_coords[:, ca_idx]
            bb_K[:, :, 2] = sel_coords[:, c_idx]
            bb_K[:, :, 3] = sel_coords[:, o_idx]

            cb_K = np.full((len(chosen), L, 3), np.nan, dtype=np.float32)
            if cb_present.any():
                cb_pos = sel_coords[:, cb_idx]
                cb_K[:, cb_present, :] = cb_pos[:, cb_present, :]

            frame0 = sel_coords[0]
            seq = "".join(_aa_from_misato_residue(an, frame0, r) for r in all_res)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            **{f"bb_{K}": bb_K, f"cb_{K}": cb_K, f"ca_{K}": bb_K[:, :, 1],
               "seq": np.array(seq), "res_type_int": res_type_int,
               "chosen_indices": chosen_arr})
        return pid, "ok"
    except Exception as e:
        return pid, f"err_{type(e).__name__}_{str(e)[:60]}"


# ============================================================
# CLI dispatch
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=("mdcath", "misato"))
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory (per-pid .npz files land here)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--K", type=int, default=10,
                    help="Number of FPS-selected frames per pid "
                         "(mdcath: 10 = ca_10 matched; misato: 8 or 10)")
    ap.add_argument("--limit", type=int, default=0)

    # mdcath-specific
    ap.add_argument("--h5-dir", type=Path,
                    help="(mdcath) directory of mdcath_dataset_<dom>.h5 files")
    ap.add_argument("--manifest", type=Path,
                    help="(mdcath) mdcath_div/_manifest.json")
    ap.add_argument("--ca-cache-dir", type=Path,
                    help="(mdcath) dir of mdcath_div/<dom>.npz with ca_{K}")
    ap.add_argument("--max-dist", type=float, default=2.0,
                    help="(mdcath) max ‖ca_frame − ca_target‖_F for frame match")

    # misato-specific
    ap.add_argument("--md-hdf5", type=Path,
                    help="(misato) path to MD.hdf5 (single file, 16,972 pids)")
    ap.add_argument("--pids-file", type=Path, default=None,
                    help="(misato) text file listing pids (one per line); "
                         "default = all groups in MD.hdf5")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mdcath":
        if not (args.h5_dir and args.manifest and args.ca_cache_dir):
            raise SystemExit("--dataset mdcath requires --h5-dir, --manifest, --ca-cache-dir")
        with open(args.manifest) as f:
            manifest = json.load(f)
        pids = list(manifest["representatives"].keys())
        if args.limit:
            pids = pids[:args.limit]
        worker = _build_one_mdcath
        worker_args = [(p, str(args.h5_dir), str(args.ca_cache_dir),
                        str(args.out), args.K, args.max_dist) for p in pids]
    else:  # misato
        if not args.md_hdf5:
            raise SystemExit("--dataset misato requires --md-hdf5")
        if args.pids_file:
            pids = [l.strip() for l in args.pids_file.read_text().splitlines() if l.strip()]
        else:
            with h5py.File(args.md_hdf5, "r") as f:
                pids = sorted(f.keys())
        if args.limit:
            pids = pids[:args.limit]
        worker = _build_one_misato
        worker_args = [(p, str(args.md_hdf5), str(args.out), args.K) for p in pids]

    print(f"[{args.dataset}] processing {len(pids)} pids on {args.workers} workers")
    counts: dict[str, int] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, a): a[0] for a in worker_args}
        for i, fut in enumerate(as_completed(futures)):
            pid, status = fut.result()
            key = status if status in ("ok", "skip") else status[:12]
            counts[key] = counts.get(key, 0) + 1
            if status not in ("ok", "skip"):
                print(f"  {pid}: {status}", flush=True)
            if (i + 1) % 100 == 0:
                ok = counts.get("ok", 0) + counts.get("skip", 0)
                print(f"  [{i+1}/{len(pids)}] {time.time()-t0:.0f}s "
                      f"ok+skip={ok}", flush=True)

    print(f"\nDONE in {time.time()-t0:.0f}s — {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
