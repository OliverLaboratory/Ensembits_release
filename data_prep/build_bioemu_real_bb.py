"""Pick 10 diverse frames per BioEmu-CATH system and cache their real backbones.

What it does
------------
Reads the BioEmu-CATH MD dataset (Zenodo 10.5281/zenodo.15629740), which ships
as two deposits, `ONE_cath1/` and `MSR_cath2/`. Each system directory
`<root>/{ONE_cath1,MSR_cath2}/cath?_<pid>/` holds

    topology.pdb          atom names and topology
    trajs/*.cmprsd.xtc    GROMACS compressed trajectories, one per run (300 K)

For every system it

  1. keeps the residues that have all four backbone atoms N, CA, C, O;
  2. loads and concatenates the trajectories of all runs (nm -> Angstrom);
  3. subsamples them uniformly to at most `--max-frames` (2,000) frames;
  4. picks K = 10 frames by greedy farthest point sampling (FPS) on CA RMSD
     after optimal (Kabsch) superposition, so global rigid motion does not
     count as diversity (first frame drawn at random with seed 0);
  5. stores the real N/CA/C/O coordinates of those frames.

Systems without trajectories (3 in the current data) or with fewer than 10
complete residues are dropped. The output layout matches
`data/cached_descriptors/mdcath_real_bb`, so the descriptor builders read it
the same way.

Paper
-----
  * `appendix.tex`, \\subsection{Dataset} (\\label{app:dataset}),
    \\paragraph{BioEmu-CATH.}: concatenate all runs, subsample uniformly to at
    most 2,000 frames, select 10 frames by FPS on CA RMSD after optimal
    superposition.
  * `2_results.tex`, \\subsection{Datasets}: P_max = 10 diverse frames by FPS.

Origin
------
Copied from `submission_exp/src/build_bioemu_real_bb.py`. Changed on
2026-10-15 (code_fix §5 / T1.5): files are named by system (`cath1_<pid>`,
`cath2_<pid>`) instead of domain, so the 16 domains simulated in both deposits
keep both simulations; layout rows carry `system`, `pid` (domain) and `deposit`.
2026-10-16: the steps of `_process` (trajectory list, backbone residues, loading,
subsampling, FPS + padding) became functions (`system_xtcs`, `backbone_residues`,
`load_runs_xyz`, `subsample_index`, `pick_frames`) so `build_bioemu_extras.py` re-derives
the picks with the same code; logic unchanged.

Used by the reported model: the current checkpoint used the old, domain-keyed
output `data/cached_descriptors/reviewer_3_special_corpus_real_bb`. The retrained
model uses the system-keyed output `data/cached_descriptors/bioemu_cath_real_bb`
(read by `data_prep/build_bioemu_misato_descriptor.py`).

Inputs / outputs
----------------
Paths relative to REPO_ROOT (as used for the reported model):
  reads   data/reviewer_3_special_corpus/bioemu_cath/extracted/{ONE_cath1,MSR_cath2}/
  writes  data/cached_descriptors/bioemu_cath_real_bb/
              <system>.npz      bb_10 : (K, L, 4, 3) float32, atoms [N, CA, C, O]
                                (system = directory name, e.g. cath2_1o82A00)
              layout_rows.json  [{system, pid, deposit, corpus: "bioemu_cath", L}, ...]

How to run
----------
    python data_prep/build_bioemu_real_bb.py \\
        --root data/reviewer_3_special_corpus/bioemu_cath/extracted \\
        --out  data/cached_descriptors/bioemu_cath_real_bb \\
        --k 10 --workers 16 [--limit N]
(run from REPO_ROOT, or pass absolute paths)
"""
from __future__ import annotations
import os
# Keep numpy/BLAS single-ish threaded: many worker processes run in parallel.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse, glob, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

K_DEFAULT = 10      # frames kept per system (paper: P_max = 10)
SEED = 0            # seed for the random first FPS frame


# ---- Kabsch-aligned Cα RMSD + FPS (copied from scripts/build_mdcath_div.py) ----
def _kabsch_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    """CA RMSD between two frames (L, 3) after centring and optimal rotation."""
    from scipy.spatial.transform import Rotation
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    R, _ = Rotation.align_vectors(Xc, Yc)
    Y_aligned = R.apply(Yc)
    return float(np.sqrt(np.mean(np.sum((Xc - Y_aligned) ** 2, axis=-1))))


def farthest_point_sample(frames: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """k frames maximizing minimum Kabsch-aligned pairwise Cα-RMSD (greedy)."""
    # Paper: appendix.tex \paragraph{BioEmu-CATH.} — "select 10 frames by
    # farthest point sampling (FPS) on Cα RMSD after optimal superposition".
    # Greedy FPS: start from a random frame, then repeatedly add the frame whose
    # distance to the closest already selected frame is largest. Returned in
    # selection order (not time order).
    N = frames.shape[0]
    if N <= k:
        return np.arange(N)
    rng = np.random.RandomState(seed)
    first = int(rng.randint(N))
    selected = [first]
    min_d = np.array([_kabsch_rmsd(frames[first], frames[j]) for j in range(N)])
    min_d[first] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        new_d = np.array([_kabsch_rmsd(frames[nxt], frames[j]) for j in range(N)])
        min_d = np.minimum(min_d, new_d)
        min_d[nxt] = -np.inf
    return np.array(selected)


# ---- Steps of _process, as functions (2026-10-16) ------------------------------
# Split out of _process without any change of logic, so that
# data_prep/build_bioemu_extras.py re-derives the FPS picks with the same code
# (residue filter, frame loading, subsampling, FPS, padding).
def system_xtcs(sysdir) -> list:
    """The run trajectories of a system: trajs/run*_protein.cmprsd.xtc, sorted (the extra
    short <pid>_<i>.cmprsd.xtc files of some ONE_cath1 systems are not used)."""
    return sorted(glob.glob(str(Path(sysdir) / "trajs" / "run*_protein.cmprsd.xtc")))


def backbone_residues(top):
    """Residues with a full N/CA/C/O backbone, in residue order:
    (res_ids, n_idx, ca_idx, c_idx, o_idx) (residues missing any of the four atoms are
    skipped, not reconstructed)."""
    idx = {a: [] for a in ("N", "CA", "C", "O")}
    by_res = {}
    for atom in top.atoms:
        if atom.name in idx:
            by_res.setdefault(atom.residue.index, {})[atom.name] = atom.index
    res_ids = sorted(r for r, d in by_res.items()
                     if all(n in d for n in ("N", "CA", "C", "O")))
    return (res_ids, np.array([by_res[r]["N"] for r in res_ids]),
            np.array([by_res[r]["CA"] for r in res_ids]),
            np.array([by_res[r]["C"] for r in res_ids]),
            np.array([by_res[r]["O"] for r in res_ids]))


def load_runs_xyz(xtcs, top):
    """All runs concatenated, in Å: (xyz (T, n_atoms, 3) float32, [frames per run])."""
    import mdtraj as md
    traj_all = [md.load(xf, top=top) for xf in xtcs]
    traj = traj_all[0] if len(traj_all) == 1 else md.join(traj_all)
    return traj.xyz * 10.0, [int(t.n_frames) for t in traj_all]


def subsample_index(T: int, max_frames: int):
    """Uniform subsample of T frames to at most max_frames (None = keep all)."""
    if T > max_frames:
        return np.linspace(0, T - 1, max_frames).astype(int)
    return None


def pick_frames(ca: np.ndarray, k: int) -> np.ndarray:
    """FPS picks on the (subsampled) Cα frames, padded by repeating the last if T < k."""
    pick = farthest_point_sample(ca, k, seed=SEED)
    if len(pick) < k:
        pick = np.concatenate([pick, np.full(k - len(pick), pick[-1])])
    return pick


def _process(sysdir: str, out: str, k: int, max_frames: int) -> tuple[str, str, str | None, int]:
    """Returns (system, pid, err_or_None, L)."""
    import mdtraj as md
    sysdir = Path(sysdir)
    system = sysdir.name                     # cath1_1o82A00 (unique across deposits)
    pid = system.split("_", 1)[-1]           # cath1_1o82A00 -> 1o82A00 (CATH domain)
    # FIXED(code_fix §5 / T1.5, 2026-10-15): the file is keyed by system, not by
    #   domain. Before, the 16 domains in both deposits shared one <pid>.npz (first
    #   writer won) while layout_rows.json listed them twice, so the corpus held 15
    #   byte-identical duplicate ensembles. Now both simulations are kept; the
    #   corpus builder splits by domain so the two never land in different splits.
    #   Paper text: require_fix #23.
    outp = Path(out) / f"{system}.npz"
    if outp.exists():                         # resumable: reuse an existing file
        try:
            return system, pid, None, int(np.load(outp)["bb_10"].shape[1])
        except Exception:
            pass
    topf = sysdir / "topology.pdb"
    xtcs = system_xtcs(sysdir)
    if not topf.exists() or not xtcs:
        # 3 systems of the Zenodo release (cath2_3bdlA01, cath2_3bpqD00,
        # cath2_3dh3A01) have no trajs/ directory and are dropped here.
        return system, pid, "missing topology/trajs", 0
    try:
        top = md.load_topology(str(topf))
        # residues with a full N/CA/C/O backbone, in residue order
        res_ids, n_idx, ca_idx, c_idx, o_idx = backbone_residues(top)
        if len(res_ids) < 10:
            return system, pid, f"too few residues ({len(res_ids)})", 0
        # load & concat all runs (nm -> Å); cap frames for FPS cost
        # Paper: appendix.tex \paragraph{BioEmu-CATH.} — "concatenate the
        # trajectories of all runs, subsample them uniformly to at most 2,000 frames".
        xyz, _ = load_runs_xyz(xtcs, top)              # (T, n_atoms, 3) Å
        sel = subsample_index(xyz.shape[0], max_frames)
        if sel is not None:                             # subsample uniformly for FPS speed
            xyz = xyz[sel]
        # FPS on the CA trace of the (subsampled) frames.
        ca = xyz[:, ca_idx, :]                       # (T', L, 3)
        pick = pick_frames(ca, k)
        # Store the real backbone of the chosen frames: (k, L, 4, 3) = [N, CA, C, O].
        bb = np.stack([xyz[pick][:, n_idx, :], xyz[pick][:, ca_idx, :],
                       xyz[pick][:, c_idx, :], xyz[pick][:, o_idx, :]], axis=2)  # (k, L, 4, 3)
        Path(out).mkdir(parents=True, exist_ok=True)
        # Q14 (2026-10-16): temporary name, then rename (no truncated file on a kill).
        tmp = Path(out) / f".{system}.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, bb_10=bb.astype(np.float32))
        os.replace(tmp, outp)
        return system, pid, None, len(res_ids)
    except Exception as e:
        return system, pid, f"{type(e).__name__}: {e}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing ONE_cath1/ and/or MSR_cath2/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--max-frames", type=int, default=2000,
                    help="cap frames fed to FPS (uniform subsample) for speed")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    # FIXED(preretrain_audit_r2 Q15, 2026-10-16): old = --root / --out relative to the
    #   working directory, and 0 systems still wrote an empty layout_rows.json. New =
    #   relative paths are taken from REPO_ROOT; no systems, or more than 0.5% failed
    #   systems (apart from the 3 release systems without trajectories), exit 1.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ensembits.paths import REPO_ROOT
    args.root = str(Path(args.root) if Path(args.root).is_absolute() else REPO_ROOT / args.root)
    args.out = str(Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out)

    # Collect the system directories of both deposits (50 in ONE_cath1,
    # 1,043 in MSR_cath2 in the current download; 16 domains are in both).
    sysdirs = []
    for sub in ("ONE_cath1", "MSR_cath2"):
        sysdirs += sorted(glob.glob(str(Path(args.root) / sub / "cath*_*")))
    if args.limit:
        sysdirs = sysdirs[:args.limit]
    print(f"[bioemu-bb] {len(sysdirs)} systems  k={args.k}  workers={args.workers}", flush=True)
    if not sysdirs:
        print(f"[error] no ONE_cath1/ or MSR_cath2/ systems under {args.root}", flush=True)
        return 1

    # Process systems in parallel; one layout row (and one file) per successful
    # system, so a domain simulated in both deposits gets two rows and two files.
    rows, dropped, no_traj, t0 = [], 0, 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process, s, args.out, args.k, args.max_frames): s for s in sysdirs}
        for i, fu in enumerate(as_completed(futs)):
            system, pid, err, L = fu.result()
            if err:
                dropped += 1
                no_traj += err == "missing topology/trajs"
                print(f"  DROP {system}: {err}", flush=True)
            else:
                rows.append({"system": system, "pid": pid,
                             "deposit": Path(futs[fu]).parent.name,
                             "corpus": "bioemu_cath", "L": int(L)})
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(sysdirs)}] {time.time()-t0:.0f}s dropped={dropped}", flush=True)
    # Write the system list sorted by domain, then system (overwrites any previous
    # layout_rows.json).
    rows.sort(key=lambda r: (r["pid"], r["system"]))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "layout_rows.json").write_text(json.dumps(rows, indent=1))
    print(f"[bioemu-bb] wrote {len(rows)} systems ({dropped} dropped, {no_traj} without "
          f"trajectories) -> {args.out}", flush=True)
    if not rows or (dropped - no_traj) > 0.005 * len(sysdirs):
        print(f"[error] {dropped - no_traj} of {len(sysdirs)} systems failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
