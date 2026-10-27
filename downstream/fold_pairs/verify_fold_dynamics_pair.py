"""Verify mined pairs: same fold, unrelated sequences, and reproducibly divergent (or similar) dynamics.

What it does
------------
Tokenizer-free. For each candidate pair (A, B) of systems it redoes the alignment of
the miner (same imported functions) and reports

  SAME FOLD   : TM-score (max / avg), core RMSD between the two mean structures,
                sequence identity over the TM-aligned positions (sequences from each
                system's BioEmu `topology.pdb`), lengths, core size.
  DYNAMICS    : cross-protein PC1/2/3 cosines, RMSIP over the top-10 modes, Hess
                covariance overlap over the top-10 modes (1 = identical, 0 = orthogonal).
  CONVERGENCE : split-half PC1 reproducibility (interleaved frames, reported only),
                disjoint-run PC1 reproducibility (first half of the runs vs the
                second half, via `runlens`; used by the verdict), per-run PC1
                consistency within vs across proteins, frame bootstrap of cos1.

Verdicts (`--mode`):
  divergent (unchanged rule): repro_disjoint_A >= 0.6 and repro_disjoint_B >= 0.6
                and cos1 <= 0.30.
  similar   (control)       : repro_disjoint_A >= 0.6 and repro_disjoint_B >= 0.6
                and cos1 >= 0.70 and cov_overlap10 >= 0.5.
  both modes additionally   : seq_id <= --max-seq-id (default 0.30), so neither set
                contains near-homologs that a sequence-aware tokenizer would match
                trivially.
All thresholds are CLI options with these defaults. The TM >= 0.7 fold gate is applied
upstream by the miner.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Step 3 of `rebuttal/fold_dynamics_mining.md` (verification).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/verify_fold_dynamics_pair.py`. The
measurements (`modes_k`, `rmsip`, `cov_overlap`, `cross_align`, `replica_blocks`,
the body of `verify` up to the verdict, the global bootstrap RNG with seed 0) are
unchanged. Changed (FIXED code_fix C15 markers): argparse CLI with `--mode`, the
similar verdict, `--max-seq-id` in both modes, thresholds as options, pairs read from
the miner's candidates JSON, systems instead of domains (sequence from the system's
own topology), JSON output with the settings; the env variables FD_CACHE / FD_OUTJSON
became `--cache` / `--out`. Removed: the mdCATH `.h5` sequence branch (`H5DIR`,
another user's absolute path); none of the 42 domains of the original BioEmu
verification has an mdCATH `.h5`, so their sequences already came from
`topology.pdb`.

Used by the reported model: no (tokenizer-free selection of the pairs).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/fold_pairs/bioemu_ca/<system>.npz
          output/fold_pairs/candidates_<mode>.json                 (bioemu_fold_dynamics_miner.py)
          data/reviewer_3_special_corpus/bioemu_cath/extracted/<deposit>/<system>/topology.pdb
  writes  output/fold_pairs/verification_<mode>.json               {mode, thresholds, pairs: [...]}

How to run
----------
    python downstream/fold_pairs/verify_fold_dynamics_pair.py --mode divergent
    python downstream/fold_pairs/verify_fold_dynamics_pair.py --mode similar
    python downstream/fold_pairs/verify_fold_dynamics_pair.py --mode divergent cath2_A cath2_B ...   # explicit pairs
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '8'
import sys, glob, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_dynamics_miner as M   # noqa: E402  reuse tmalign, kabsch_R, internal_align, _svd
import _fp_common as C            # noqa: E402

CACHE = str(C.CA_CACHE)
rng = np.random.RandomState(0)
# Caveat: one global RNG is shared by all pairs, so a pair's bootstrap CI
#   (boot_cos1_*) depends on pair order. Reported only, not used by the verdict.
RESULTS = []
CFG = argparse.Namespace(mode='divergent', min_repro=0.6, max_cos1=0.30, min_cos1=0.70,
                         min_cov_overlap=0.5, max_seq_id=0.30)


def load(system):
    d = np.load(os.path.join(CACHE, system + '.npz'))
    ca = d['ca'].astype(np.float64)
    replens = (d['runlens'] if 'runlens' in d.files else d['replens']).tolist()
    return ca, replens


def seq(system):
    # FIXED(code_fix C15, 2026-10-15): old: mdCATH .h5 first, then the BioEmu
    #   topology.pdb of the domain (MSR_cath2 first), 'X' if neither exists. New: the
    #   topology.pdb of this system (deposit-specific), an error if it is missing.
    return C.topology_sequence(system)


def modes_k(frames, k):
    X = frames.reshape(frames.shape[0], -1); X = X - X.mean(0)
    _, S, Vt = M._svd(X, full_matrices=False)
    return Vt[:k], (S[:k] ** 2)


def rmsip(vA, vB, k):
    return float(np.sqrt(sum((vA[i] @ vB[j]) ** 2 for i in range(k) for j in range(k)) / k))


def cov_overlap(vA, wA, vB, wB, k):
    # Hess covariance overlap over top-k modes (1 = identical dynamics, 0 = orthogonal).
    # wA / wB must be covariance eigenvalues (see the call site).
    num = wA[:k].sum() + wB[:k].sum() - 2 * sum(np.sqrt(wA[i]*wB[j]) * (vA[i]@vB[j])**2
                                                 for i in range(k) for j in range(k))
    den = wA[:k].sum() + wB[:k].sum()
    return float(1 - np.sqrt(max(num, 0.0) / den))


def cross_align(A, B, iA, iB):
    Aint = M.internal_align(A[:, iA, :]); Bint = M.internal_align(B[:, iB, :])
    Rba = M.kabsch_R(Bint[0], Aint[0]); BinA = Bint @ Rba.T
    return Aint, BinA


def replica_blocks(frames, replens):
    out = []; s = 0
    for L in replens:
        out.append(frames[s:s+L]); s += L
    return out


def verdict(reproA_dj, reproB_dj, cos1, cov, seqid, cfg=CFG):
    """(valid, parts) for one pair under cfg.mode. Inputs are the unrounded values."""
    repro_ok = bool(reproA_dj >= cfg.min_repro and reproB_dj >= cfg.min_repro)
    # FIXED(code_fix C15, 2026-10-15): old: only the divergent rule
    #   `reproA_dj >= 0.6 and reproB_dj >= 0.6 and cos[0] <= 0.30`, no sequence check.
    #   New: the same divergent rule; a similar (control) rule with cos1 >= 0.70 and
    #   cov_overlap10 >= 0.5; and seq_id <= max_seq_id (0.30) in BOTH modes.
    if cfg.mode == 'divergent':
        dyn_ok = bool(cos1 <= cfg.max_cos1)
    elif cfg.mode == 'similar':
        dyn_ok = bool(cos1 >= cfg.min_cos1 and cov >= cfg.min_cov_overlap)
    else:
        raise ValueError(f'unknown mode {cfg.mode!r}')
    seq_ok = bool(seqid <= cfg.max_seq_id)
    return bool(repro_ok and dyn_ok and seq_ok), {'repro_ok': repro_ok, 'dyn_ok': dyn_ok, 'seq_ok': seq_ok}


def verify(a, b):
    A, rlA = load(a); B, rlB = load(b)
    r = M.tmalign(A[0], B[0])
    tmin, tmax, tavg, corr = r
    iA = np.array([c[0] for c in corr]); iB = np.array([c[1] for c in corr])
    sA, sB = seq(a), seq(b)
    # Identity over the TM-aligned positions only (not a global alignment).
    # Caveat: 'X' == 'X' counts as identical (unknown residue names in topology.pdb).
    seqid = np.mean([sA[i] == sB[j] for i, j in corr]) if corr else float('nan')
    # core RMSD (mean structures, aligned)
    Aint, BinA = cross_align(A, B, iA, iB)
    # core RMSD between the two mean structures (proper: superpose incl. translation)
    mA = Aint.mean(0); mB = BinA.mean(0)
    R = M.kabsch_R(mB, mA); mB_fit = (mB - mB.mean(0)) @ R.T + mA.mean(0)
    rmsd_core = float(np.sqrt(((mA - mB_fit) ** 2).sum(1).mean()))
    K = 10
    vA, wA = modes_k(Aint, K); vB, wB = modes_k(BinA, K)
    cos = [abs(float(vA[i] @ vB[i])) for i in range(3)]
    # FIXED(code_fix C15, 2026-10-15): covariance eigenvalues = S^2 / (F - 1). The
    #   original passed S^2, which scales A and B by their different frame counts
    #   (about 10% apart for 42 vs 38 runs); cov_overlap10 enters the similar verdict.
    rms = rmsip(vA, vB, K)
    cov = cov_overlap(vA, wA / (Aint.shape[0] - 1), vB, wB / (BinA.shape[0] - 1), K)
    # positive control: same-protein two-half reproducibility (interleaved halves)
    def repro(int_frames):
        h1 = int_frames[0::2]; h2 = int_frames[1::2]
        p1, _ = modes_k(h1, 1); p2, _ = modes_k(h2, 1)
        return abs(float(p1[0] @ p2[0]))
    reproA, reproB = repro(Aint), repro(BinA)
    # stricter reproducibility: DISJOINT replica groups (first half of the runs vs the rest)
    def repro_disjoint(frames, replens):
        blks = replica_blocks(frames, replens); h = len(blks) // 2
        g1 = np.concatenate(blks[:h]); g2 = np.concatenate(blks[h:])
        p1, _ = modes_k(g1, 1); p2, _ = modes_k(g2, 1)
        return abs(float(p1[0] @ p2[0]))
    reproA_dj, reproB_dj = repro_disjoint(Aint, rlA), repro_disjoint(BinA, rlB)
    # per-replica PC1 consistency
    blkA = replica_blocks(Aint, rlA); blkB = replica_blocks(BinA, rlB)
    pcA_r = [modes_k(x, 1)[0][0] for x in blkA]; pcB_r = [modes_k(x, 1)[0][0] for x in blkB]
    def within(pcs): return float(np.mean([abs(pcs[i]@pcs[j]) for i in range(len(pcs)) for j in range(i+1,len(pcs))]))
    withinA, withinB = within(pcA_r), within(pcB_r)
    crossrep = float(np.mean([abs(x@y) for x in pcA_r for y in pcB_r]))
    # bootstrap CI on cross cos1 (resample frames w/ replacement)
    boot = []
    for _ in range(30):
        ia = rng.randint(0, Aint.shape[0], Aint.shape[0]); ib = rng.randint(0, BinA.shape[0], BinA.shape[0])
        pa, _ = modes_k(Aint[ia], 1); pb, _ = modes_k(BinA[ib], 1)
        boot.append(abs(float(pa[0] @ pb[0])))
    boot = np.array(boot)
    # verdict: same fold (already TM>=0.7) + robust dynamics criterion of the mode
    #   + unrelated sequences (seq_id <= max_seq_id)
    valid, parts = verdict(reproA_dj, reproB_dj, cos[0], cov, float(seqid))
    rec = {'A': a, 'B': b, 'pidA': C.pid_of(a), 'pidB': C.pid_of(b), 'mode': CFG.mode,
           'tm_max': round(tmax,3), 'tm_avg': round(tavg,3),
           'core_rmsd': round(rmsd_core,2), 'seq_id': round(float(seqid),3),
           'LA': A.shape[1], 'LB': B.shape[1], 'ncore': len(corr),
           'cos1': round(cos[0],3), 'cos2': round(cos[1],3), 'cos3': round(cos[2],3),
           'rmsip10': round(rms,3), 'cov_overlap10': round(cov,3),
           'repro_half_A': round(reproA,3), 'repro_half_B': round(reproB,3),
           'repro_disjoint_A': round(reproA_dj,3), 'repro_disjoint_B': round(reproB_dj,3),
           'within_A': round(withinA,3), 'within_B': round(withinB,3), 'cross_replica': round(crossrep,3),
           'boot_cos1_mean': round(float(boot.mean()),3), 'boot_cos1_lo': round(float(np.percentile(boot,2.5)),3),
           'boot_cos1_hi': round(float(np.percentile(boot,97.5)),3), **parts, 'valid': valid}
    RESULTS.append(rec)
    print(f'\n########## {a}  vs  {b}   [{"VALID" if valid else "REJECTED"}] ({CFG.mode}) ##########')
    print(f'[SAME FOLD]  TM_max={tmax:.3f} TM_avg={tavg:.3f}  core_RMSD={rmsd_core:.2f}A  seq_id={seqid*100:.0f}%'
          f'  L(A/B)={A.shape[1]}/{B.shape[1]}  N_core={len(corr)}  frames(A/B)={A.shape[0]}/{B.shape[0]}')
    print(f'[DYNAMICS ]  cos(PC1/2/3)={cos[0]:.3f}/{cos[1]:.3f}/{cos[2]:.3f}  RMSIP(10)={rms:.3f}  cov_overlap(10)={cov:.3f}')
    print(f'[POS CTRL ]  same-protein half-vs-half PC1 cos:  A={reproA:.3f}  B={reproB:.3f}   (interleaved)')
    print(f'[POS CTRL2]  same-protein DISJOINT replica-groups PC1 cos:  A={reproA_dj:.3f}  B={reproB_dj:.3f}   (HIGH => pooled mode robust to which replicas)')
    print(f'[PER-REPL ]  within-A={withinA:.3f}  within-B={withinB:.3f}   cross A-B={crossrep:.3f}   (within>>cross => robust)')
    print(f'[BOOTSTRAP]  cross PC1 cos = {boot.mean():.3f} +/- {boot.std():.3f}  (95% CI [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}])')
    print(f'[VERDICT  ]  repro_ok={parts["repro_ok"]} dyn_ok={parts["dyn_ok"]} seq_ok={parts["seq_ok"]}')
    return rec


def main():
    global CACHE
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('pairs', nargs='*', help='explicit pairs: A1 B1 [A2 B2 ...] (systems); '
                    'default: the pairs of --pairs-json')
    ap.add_argument('--mode', choices=C.SELECTIONS, required=True)
    ap.add_argument('--pairs-json', default=None,
                    help='candidates JSON of the miner (default output/fold_pairs/candidates_<mode>.json)')
    ap.add_argument('--cache', default=CACHE, help='Ca cache dir (default output/fold_pairs/bioemu_ca)')
    ap.add_argument('--out', default=None, help='default output/fold_pairs/verification_<mode>.json')
    ap.add_argument('--min-repro', type=float, default=0.6, help='disjoint-run PC1 reproducibility, both proteins')
    ap.add_argument('--max-cos1', type=float, default=0.30, help='divergent: cross PC1 cosine at most this')
    ap.add_argument('--min-cos1', type=float, default=0.70, help='similar: cross PC1 cosine at least this')
    ap.add_argument('--min-cov-overlap', type=float, default=0.5, help='similar: covariance overlap (10 modes) at least this')
    ap.add_argument('--max-seq-id', type=float, default=0.30, help='both modes: sequence identity at most this')
    args = ap.parse_args()
    CACHE = args.cache
    for k in ('mode', 'min_repro', 'max_cos1', 'min_cos1', 'min_cov_overlap', 'max_seq_id'):
        setattr(CFG, k, getattr(args, k))

    if args.pairs:
        if len(args.pairs) % 2:
            ap.error('explicit pairs must come in A B couples')
        pairs = [(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]
        src = 'command line'
    else:
        src = args.pairs_json or str(C.candidates_path(args.mode))
        cj = json.load(open(src))
        if cj.get('select') not in (None, args.mode):
            print(f'  WARNING: {src} was selected with --select {cj.get("select")}, '
                  f'verifying with --mode {args.mode}', flush=True)
        pairs = [(r['A'], r['B']) for r in cj['pairs']]
    print(f'[verify] mode={args.mode}  {len(pairs)} pairs from {src}', flush=True)
    for a, b in pairs:
        if C.pid_of(a) == C.pid_of(b):
            print(f'  [skip] {a} / {b}: same domain', flush=True)
            continue
        verify(a, b)
    out = args.out or str(C.verification_path(args.mode))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    thr = {'min_repro_disjoint': CFG.min_repro, 'max_seq_id': CFG.max_seq_id}
    if CFG.mode == 'divergent':
        thr['max_cos1'] = CFG.max_cos1
    else:
        thr.update(min_cos1=CFG.min_cos1, min_cov_overlap10=CFG.min_cov_overlap)
    json.dump({'mode': CFG.mode, 'thresholds': thr, 'source': os.path.basename(src) if src != 'command line' else src,
               'n': len(RESULTS), 'n_valid': sum(r['valid'] for r in RESULTS), 'pairs': RESULTS},
              open(out, 'w'), indent=1)
    print(f'\n[verify] {sum(r["valid"] for r in RESULTS)}/{len(RESULTS)} valid ({args.mode}) -> {out}')


if __name__ == '__main__':
    main()
