"""Cache the C-alpha trajectories of every BioEmu-CATH system (xtc -> npz), one file per system.

What it does
------------
Each BioEmu-CATH system directory holds `topology.pdb` and
`trajs/run*_protein.cmprsd.xtc` (~39 independent runs x 101 frames, 300 K). For
every system of both deposits it

  1. takes `--per-run` frames per run (default 30), evenly spaced
     (`np.linspace(0, n_frames - 1, k)`), C-alpha atoms only (`name CA`), nm -> A;
  2. pools the frames of all runs and records `runlens`, the number of frames each
     run contributed (needed by the disjoint-run reproducibility test of the
     verifier);
  3. writes `<system>.npz` with `ca` (F, L, 3) float32, `runlens`, `system`, `pid`
     (domain), `deposit` and `n_full_bb` (number of residues with all of N, CA, C,
     O; the tokenization step checks it equals L).

Nothing here touches a tokenizer.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Step 1 of `rebuttal/fold_dynamics_mining.md` (Ca cache).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/bioemu_extract_ca.py`. Changed:
  * outputs are keyed by system instead of domain (FIXED code_fix C15 below), and the
    npz carries `system`, `pid`, `deposit`, `n_full_bb`;
  * `--out` and `--limit` flags; paths from `ensembits.paths.REPO_ROOT`
    (`_fp_common.py`); comments and this docstring.
Frame selection, CA selection and the unit conversion are unchanged.

Used by the reported model: no (tokenizer-free preprocessing; feeds the miner, the
verifier and the RMSF of the token comparison).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/reviewer_3_special_corpus/bioemu_cath/extracted/{MSR_cath2,ONE_cath1}/cath*_*/
              topology.pdb, trajs/run*_protein.cmprsd.xtc
  writes  output/fold_pairs/bioemu_ca/<system>.npz

How to run
----------
    python downstream/fold_pairs/bioemu_extract_ca.py --workers 32 --per-run 30 [--limit N]
(needs mdtraj; CPU only)
"""
import os, sys, glob, argparse, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fp_common as C   # noqa: E402  REPO_ROOT-relative paths of this test

OUT = str(C.CA_CACHE)
PER_RUN = 30


def _extract(sysdir):
    import mdtraj as md
    system = os.path.basename(sysdir.rstrip('/'))
    pid = C.pid_of(system)
    deposit = os.path.basename(os.path.dirname(sysdir.rstrip('/')))
    # FIXED(code_fix C15, 2026-10-15): old: the cache was keyed by domain
    #   (`cath2_`/`cath1_` stripped), so for the 16 domains simulated in both deposits
    #   the first worker to finish won and the other simulation was silently skipped
    #   ('cached'). New: keyed by system (both simulations kept, like
    #   data_prep/build_bioemu_real_bb.py); `pid` is stored so the miner can refuse
    #   to pair two systems of the same domain.
    outp = os.path.join(OUT, system + '.npz')
    if os.path.exists(outp):
        return system, 'cached'
    topf = os.path.join(sysdir, 'topology.pdb')
    xtcs = sorted(glob.glob(os.path.join(sysdir, 'trajs', 'run*_protein.cmprsd.xtc')))
    if not os.path.exists(topf) or not xtcs:
        return system, 'no-input'
    try:
        top = md.load_topology(topf); ca = top.select('name CA')
        # Residues with a complete N/CA/C/O backbone (the rule of the bb_10 cache);
        # recorded so the tokenization step can check that the residue sets agree.
        n_full_bb = len(C.backbone_indices(top)[4])
        frames = []; runlens = []
        for xf in xtcs:
            t = md.load(xf, top=top, atom_indices=ca)
            # Evenly spaced PER_RUN frames of this run (all frames if the run is shorter).
            k = min(PER_RUN, t.n_frames)
            idx = np.linspace(0, t.n_frames - 1, k).astype(int)
            frames.append((t.xyz[idx] * 10.0).astype(np.float32))  # nm -> Angstrom
            runlens.append(k)
        allf = np.concatenate(frames, 0)
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(outp, ca=allf, runlens=np.array(runlens),
                            system=np.array(system), pid=np.array(pid),
                            deposit=np.array(deposit), n_full_bb=np.array(n_full_bb))
        return system, allf.shape
    except Exception as e:
        return system, f'ERR {type(e).__name__} {e}'


def main():
    global PER_RUN, OUT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--per-run', type=int, default=30, help='frames kept per run (evenly spaced)')
    ap.add_argument('--out', default=OUT, help='output folder (default REPO_ROOT/output/fold_pairs/bioemu_ca)')
    ap.add_argument('--limit', type=int, default=0, help='only the first N systems (smoke test)')
    args = ap.parse_args(); PER_RUN = args.per_run; OUT = args.out
    sysdirs = C.list_system_dirs()
    if args.limit:
        sysdirs = sysdirs[:args.limit]
    os.makedirs(OUT, exist_ok=True)
    print(f'[bioemu-ca] {len(sysdirs)} systems, {PER_RUN} frames/run, workers={args.workers}', flush=True)
    from multiprocessing import Pool
    t0 = time.time(); done = 0; err = 0
    # Workers are forked, so they see the PER_RUN / OUT set above.
    with Pool(args.workers) as p:
        for i, (system, st) in enumerate(p.imap_unordered(_extract, sysdirs)):
            done += 1
            if isinstance(st, str) and st.startswith('ERR'):
                err += 1; print(f'  [{system}] {st}', flush=True)
            if done % 100 == 0:
                print(f'  {done}/{len(sysdirs)} ({time.time()-t0:.0f}s, {err} err)', flush=True)
    n = len(glob.glob(OUT + '/*.npz'))
    print(f'[bioemu-ca] done: {n} caches in {OUT} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
