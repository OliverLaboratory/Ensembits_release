"""Numeric core of the fold-matched pair miner: TM-align, two-stage alignment, PCA modes, mode cosines.

What it does
------------
Tokenizer-free scoring of one pair of ensembles (A, B), each a (F, L, 3) C-alpha
trajectory held in the module-level dict `DOMS` (filled by the caller):

  1. `tmalign(A[0], B[0])`: TM-align of frame 0 of each ensemble (written as minimal
     CA-only PDB files) -> TM-score normalised by each chain (`tm_min`, `tm_max`,
     `tm_avg`) and the residue correspondence `corr = [(iA, iB), ...]` read from the
     printed alignment (the aligned core). Pairs with < 30 aligned residues are dropped.
  2. Stage 1, `internal_align`: every frame of A's core is Kabsch-superposed onto
     A's own frame 0, every frame of B's core onto B's own frame 0.
  3. Stage 2: exactly ONE rotation `Rba = kabsch_R(Bint[0], Aint[0])`, computed once
     and applied to all B frames (`BinA = Bint @ Rba.T`). One rigid action cannot
     absorb B's internal motion, unlike a per-frame superposition onto A.
  4. `modes`: PCA (SVD of the centred (F, 3N) core) -> top-3 unit modes and the
     squared singular values.
  5. Scores: `cos1..3 = |cos(PC_i^A, PC_i^B)|`, `rmsip` over the top-3 subspaces,
     `fvarA` / `fvarB` = fraction of variance in PC1, `ncommon` = core size.

`process(pair)` wraps `_process` and turns exceptions into an error record. One BLAS
thread per process is pinned before numpy is imported, so many workers do not
oversubscribe the CPU.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Step 2 of `rebuttal/fold_dynamics_mining.md` (sections 2b to 2d).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/fold_dynamics_miner.py`. The numeric
core (`write_ca_pdb`, `tmalign`, `_svd`, `kabsch_R`, `internal_align`, `modes`,
`process`, `_process`) is unchanged line for line, except that the TM-align
executable is `ensembits.paths.TMALIGN_BIN` (was a hard-coded conda-env path; see
FIXED below). Removed, because they only serve the mdCATH variant of the analysis
(not used for the BioEmu result) and hard-coded another user's absolute paths:
`H5DIR`, `HIER`, `CACHE`, `TEMP`, `_ca_indices`, `_extract_one`, `prepare_cache`,
`load_domains` (mdCATH CATH-label loader), `candidate_pairs` (same-topology pairing)
and `main`. The BioEmu loader / pairing / CLI live in `bioemu_fold_dynamics_miner.py`.

Used by the reported model: no (tokenizer-free selection of the pairs).

Inputs / outputs
----------------
None by itself (library); TM-align writes two temporary PDB files per call in the
system temporary folder.

How to run
----------
    import fold_dynamics_miner as M
    M.DOMS[name] = {'topo': ..., 'homo': ..., 'ca': (F, L, 3) float64, 'L': L, 'replens': [...]}
    rec = M.process(('bioemu', a, b))
"""
import os, sys, tempfile, subprocess
from pathlib import Path
# Pin 1 BLAS thread per process BEFORE numpy imports, so (workers x threads) stays <= workers.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import TMALIGN_BIN   # noqa: E402

# FIXED(code_fix C15, 2026-10-15): old: TMALIGN = an absolute path into a personal conda
#   env. New: ensembits.paths.TMALIGN_BIN (env ENSEMBITS_TMALIGN, else TMalign, else
#   USalign on PATH). The output parser below is unchanged; USalign prints the same
#   "TM-score=" lines and alignment block.
TMALIGN = TMALIGN_BIN
DOMS = {}  # global (fork-inherited): name -> {'topo', 'homo', 'ca' (F,L,3), 'L', 'replens'}


def write_ca_pdb(ca):
    fd, path = tempfile.mkstemp(suffix='.pdb'); os.close(fd)
    with open(path, 'w') as f:
        for i, (x, y, z) in enumerate(ca):
            f.write("ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                    % (i + 1, i + 1, x, y, z))
        f.write("TER\nEND\n")
    return path


def tmalign(caA0, caB0):
    """TM-align two CA traces -> (tm_min, tm_max, tm_avg, corr) or None."""
    pa = write_ca_pdb(caA0); pb = write_ca_pdb(caB0)
    try:
        out = subprocess.run([TMALIGN, pa, pb], capture_output=True, text=True, timeout=120).stdout
    finally:
        os.remove(pa); os.remove(pb)
    # Two TM-scores: normalised by the length of A and of B.
    tms = [float(l.split('=')[1].split()[0]) for l in out.splitlines() if l.startswith('TM-score=')]
    if len(tms) < 2:
        return None
    lines = out.splitlines()
    ai = next((i for i, l in enumerate(lines) if l.startswith('(":"')), None)
    if ai is None or ai + 3 >= len(lines):
        return None
    # Walk the two aligned strings; a position where neither has a gap is a core pair.
    seqA, seqB = lines[ai + 1], lines[ai + 3]
    corr = []; iA = iB = 0
    for ca, cb in zip(seqA, seqB):
        ag = (ca == '-'); bg = (cb == '-')
        if not ag and not bg:
            corr.append((iA, iB))
        if not ag: iA += 1
        if not bg: iB += 1
    return (min(tms), max(tms), 0.5 * (tms[0] + tms[1]), corr)


def _svd(M, full_matrices=True):
    """np SVD (gesdd) with a scipy gesvd fallback for non-convergence."""
    try:
        return np.linalg.svd(M, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        import scipy.linalg as sla
        return sla.svd(M, full_matrices=full_matrices, lapack_driver='gesvd')


def kabsch_R(P, Q):  # rotation that best-aligns P onto Q (both (M,3)); translation via centroids
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, _, Vt = _svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def internal_align(core):
    """Stage 1: superpose EVERY frame of one trajectory onto its OWN frame-0 (per-frame,
    within-trajectory) to remove rigid-body tumbling. core (F,N,3) -> (F,N,3)."""
    ref = core[0]; refc = ref.mean(0)
    out = np.empty_like(core)
    for f in range(core.shape[0]):
        R = kabsch_R(core[f], ref)
        out[f] = (core[f] - core[f].mean(0)) @ R.T + refc
    return out


def modes(frames):  # frames (F,N,3) -> top3 unit eigenvectors (3,3N), variances
    X = frames.reshape(frames.shape[0], -1)
    X = X - X.mean(0)
    _, S, Vt = _svd(X, full_matrices=False)
    return Vt[:3], (S ** 2)


def process(pair):
    try:
        return _process(pair)
    except Exception as e:
        return {'pair': list(pair), 'error': f'{type(e).__name__}: {e}'}


def _process(pair):
    t, a, b = pair
    A = DOMS[a]['ca']; B = DOMS[b]['ca']
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        return None
    r = tmalign(A[0], B[0])
    if r is None:
        return None
    tmin, tmax, tavg, corr = r
    if len(corr) < 30:
        return None
    iA = np.array([c[0] for c in corr]); iB = np.array([c[1] for c in corr])
    # Stage 1 — internal alignment (each trajectory to its own frame-0, over the shared core)
    Aint = internal_align(A[:, iA, :])          # A frames in A0 frame
    Bint = internal_align(B[:, iB, :])          # B frames in B0 frame
    # Stage 2 — ONE cross SE3: rotate B0_core onto A0_core, broadcast that single R to ALL B frames
    Rba = kabsch_R(Bint[0], Aint[0])
    BinA = Bint @ Rba.T                          # single rotation applied to every B frame
    pcA, wA = modes(Aint); pcB, wB = modes(BinA)
    # |cos| because the sign of a PCA eigenvector is arbitrary.
    cos = [abs(float(pcA[i] @ pcB[i])) for i in range(3)]
    rmsip = float(np.sqrt(sum((pcA[i] @ pcB[j]) ** 2 for i in range(3) for j in range(3)) / 3.0))
    return {'topo': t, 'A': a, 'B': b, 'tm_min': round(tmin, 3), 'tm_max': round(tmax, 3),
            'tm_avg': round(tavg, 3), 'ncommon': len(corr),
            'homoA': DOMS[a]['homo'], 'homoB': DOMS[b]['homo'],
            'cos1': round(cos[0], 3), 'cos2': round(cos[1], 3), 'cos3': round(cos[2], 3),
            'rmsip': round(rmsip, 3),
            'fvarA': round(float(wA[0] / wA.sum()), 3), 'fvarB': round(float(wB[0] / wB.sum()), 3)}
