"""Mine BioEmu-CATH pairs with the same fold and divergent (or, as a control, similar) dynamics.

What it does
------------
Tokenizer-free. Two stages, both in this script:

  1. Score (once; reused by later calls). Loads every `<system>.npz` of the C-alpha
     cache, sorts the systems by length and emits every pair with
     `(L_b - L_a) / L_b <= --len-tol` (default 0.15), skipping pairs of two systems
     of the same domain (the same protein simulated in both deposits). Each pair is
     scored by the numeric core `fold_dynamics_miner.process` (TM-align of frame 0,
     two-stage alignment, PCA modes, cos1..3, rmsip, fvarA/B). BioEmu-CATH has no
     usable CATH labels, so pairing is length-filtered all-vs-all and TM-align is
     the fold gate. Output: `output/fold_pairs/scored_pairs.json`.
  2. Select (`--select divergent | similar`). Candidates must pass the same filters
     in both selections: `tm_max >= 0.7`, `ncommon >= 50`, `fvarA >= 0.25` and
     `fvarB >= 0.25` (PC1 is a genuinely dominant mode in both proteins).
       divergent : rank by cos1 ascending (most orthogonal dominant motions first),
                   top-N (default 25) go to verification. This is the rule of the
                   original BioEmu result (`rebuttal/fold_dynamics_mining.md` 2e).
       similar   : CONTROL set; identical filters, rank by cos1 descending (most
                   aligned dominant motions first), top-N (default 50, because more
                   of them fail verification, e.g. on sequence identity).
     Ties in cos1 (stored with 3 decimals) are broken by (A, B) so the selection is
     deterministic. Output: `output/fold_pairs/candidates_<select>.json`.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Step 2 of `rebuttal/fold_dynamics_mining.md` (2a candidate pairs, 2e filter).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/bioemu_fold_dynamics_miner.py`.
Changed (FIXED code_fix C15 markers below): systems instead of domains as keys, no
same-domain pairs, the `--select` stage (in the original the filter + top-25 rule was
applied by hand to the scored JSON; the write-up checked that it reproduces the
verified set), the scored JSON is cached and carries its settings, `--smoke`; paths
from `ensembits.paths.REPO_ROOT`. The scoring itself (`fold_dynamics_miner.process`)
and the length-sorted candidate scan are unchanged.
Fixes 2026-10-16 (FIXED markers): scored_pairs.json is not written when no pair
was scored; the first error is reported (code_fix P24).

Used by the reported model: no (tokenizer-free selection of the pairs).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/fold_pairs/bioemu_ca/<system>.npz            (bioemu_extract_ca.py)
  writes  output/fold_pairs/scored_pairs.json                 {meta, records: [...]}
          output/fold_pairs/candidates_{divergent,similar}.json   {select, filters, pairs: [...]}

How to run
----------
    python downstream/fold_pairs/bioemu_fold_dynamics_miner.py --workers 64 --len-tol 0.15 --select divergent
    python downstream/fold_pairs/bioemu_fold_dynamics_miner.py --select similar      # reuses the scores
"""
import os, sys, glob, json, argparse, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import fold_dynamics_miner as M   # noqa: E402  pins 1 BLAS thread BEFORE numpy is imported here
import numpy as np                 # noqa: E402
import _fp_common as C             # noqa: E402

CACHE = str(C.CA_CACHE)

# Candidate filter shared by both selections (rebuttal/fold_dynamics_mining.md 2e).
TM_MIN = 0.7          # same fold, not merely similar
NCOMMON_MIN = 50      # enough core residues for a meaningful mode
FVAR_MIN = 0.25       # PC1 is a dominant mode in BOTH proteins
TOP_N_DEFAULT = {'divergent': 25, 'similar': 50}


def load_domains(cache):
    """Fill M.DOMS from the C-alpha cache, one entry per SYSTEM."""
    for f in sorted(glob.glob(cache + '/*.npz')):
        system = os.path.basename(f)[:-4]
        d = np.load(f)
        ca = d['ca'].astype(np.float64)
        rl = (d['runlens'] if 'runlens' in d.files else d['replens']).tolist()  # bioemu | mdcath
        pid = str(d['pid']) if 'pid' in d.files else C.pid_of(system)
        # FIXED(code_fix C15, 2026-10-15): old: keyed by domain, 'homo' = domain id.
        #   New: keyed by system (both deposits kept); 'homo' and 'pid' = domain, so the
        #   two simulations of one domain are recognisable and never paired.
        M.DOMS[system] = {'topo': 'allvsall', 'homo': pid, 'pid': pid, 'ca': ca,
                          'L': ca.shape[1], 'replens': rl}


def candidate_pairs(len_tol, doms=None):
    """Length-filtered all-vs-all pairs ('bioemu', a, b), never two systems of one domain."""
    doms = M.DOMS if doms is None else doms
    dids = sorted(doms)
    L = {d: doms[d]['L'] for d in dids}
    pid = {d: doms[d].get('pid', C.pid_of(d)) for d in dids}
    order = sorted(dids, key=lambda d: L[d])
    pairs = []
    for i in range(len(order)):
        a = order[i]; La = L[a]
        for j in range(i + 1, len(order)):
            b = order[j]; Lb = L[b]
            if (Lb - La) / Lb > len_tol:   # sorted -> Lb>=La; stop when too far
                break
            # FIXED(code_fix C15, 2026-10-15): old: no such check (and none was needed,
            #   one cache file per domain). New: skip two systems of the same domain;
            #   they are the same protein and would be a trivially "same fold" pair.
            if pid[a] == pid[b]:
                continue
            pairs.append(('bioemu', a, b))
    return pairs


def score_all(cache, len_tol, workers, smoke=0):
    """Score every candidate pair with the numeric core; returns (records, meta)."""
    load_domains(cache)
    pairs = candidate_pairs(len_tol)
    print(f'[bioemu-miner] {len(M.DOMS)} systems, {len(pairs)} length-filtered pairs '
          f'(|dL|/L<={len_tol}, same-domain pairs excluded)', flush=True)
    if smoke:
        pairs = pairs[:smoke]
    from multiprocessing import Pool
    t0 = time.time(); recs = []; errs = 0; n_none = 0; first_err = None
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(M.process, pairs, chunksize=8)):
            if r is None:
                n_none += 1
                continue
            if 'error' in r:
                errs += 1
                if first_err is None:
                    first_err = f'{r["pair"]}: {r["error"]}'
                if errs <= 10:
                    print(f'  [skip] {r["pair"]}: {r["error"]}', flush=True)
                continue
            r['pidA'] = M.DOMS[r['A']]['pid']; r['pidB'] = M.DOMS[r['B']]['pid']
            recs.append(r)
            if (i + 1) % 5000 == 0:
                print(f'  {i+1}/{len(pairs)} ({time.time()-t0:.0f}s, {len(recs)} kept, {errs} err)', flush=True)
    # imap_unordered returns records in completion order; sort for a reproducible file.
    recs.sort(key=lambda r: (r['A'], r['B']))
    meta = {'cache': os.path.relpath(cache, C.ROOT) if cache.startswith(str(C.ROOT)) else cache,
            'len_tol': len_tol, 'n_systems': len(M.DOMS), 'n_pairs_tried': len(pairs),
            'n_scored': len(recs), 'n_errors': errs, 'tmalign': os.path.basename(M.TMALIGN),
            'smoke': smoke,
            # FIXED(code_fix P24, 2026-10-16): new (old: none): why pairs were not scored (None = TM-align output not parsed,
            # non-finite coordinates or < 30 aligned residues).
            'n_unscored_none': n_none, 'first_error': first_err}
    print(f'[bioemu-miner] {len(recs)} scored, {errs} skipped ({time.time()-t0:.0f}s)', flush=True)
    return recs, meta


def select_pairs(recs, select, top_n, tm_min=TM_MIN, ncommon_min=NCOMMON_MIN, fvar_min=FVAR_MIN):
    """Apply the candidate filter and the ranking of one selection; returns (top, n_candidates).

    FIXED(code_fix C15, 2026-10-15): old: only the divergent rule existed, applied by
    hand (filter, sort by cos1 ascending, top 25). New: the same filter for both
    selections; 'divergent' ranks by cos1 ascending, 'similar' (control) by cos1
    descending; ties broken by (A, B); same-domain pairs are dropped defensively.
    """
    if select not in C.SELECTIONS:
        raise ValueError(f'select must be one of {C.SELECTIONS}, got {select!r}')
    cand = [r for r in recs
            if r['tm_max'] >= tm_min and r['ncommon'] >= ncommon_min
            and r['fvarA'] >= fvar_min and r['fvarB'] >= fvar_min
            and r.get('pidA', C.pid_of(r['A'])) != r.get('pidB', C.pid_of(r['B']))]
    if select == 'divergent':
        cand.sort(key=lambda r: (r['cos1'], r['A'], r['B']))
    else:
        cand.sort(key=lambda r: (-r['cos1'], r['A'], r['B']))
    # Caveat: the similar ranking puts near-homologs first (high cos1), but sequence
    #   identity is only checked at verification (--max-seq-id), after the top-N cut.
    # Caveat: pairs may share systems (the original 20 valid pairs span 34 domains);
    #   no per-domain cap is applied in either selection.
    return cand[:top_n], len(cand)


def load_scored(path):
    obj = json.load(open(path))
    if isinstance(obj, list):            # original format: a bare list of records
        return obj, {}
    return obj['records'], obj.get('meta', {})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--len-tol', type=float, default=0.15)
    ap.add_argument('--cache', default=CACHE, help='Ca cache dir (default REPO_ROOT/output/fold_pairs/bioemu_ca)')
    ap.add_argument('--scored', default=str(C.SCORED), help='scored-pairs JSON (written once, then reused)')
    ap.add_argument('--rescore', action='store_true', help='recompute the scores even if --scored exists')
    ap.add_argument('--select', choices=C.SELECTIONS, default=None,
                    help='write the top-N candidates of this selection (omit to only score)')
    ap.add_argument('--top-n', type=int, default=None, help='default: 25 divergent, 50 similar')
    ap.add_argument('--tm-min', type=float, default=TM_MIN)
    ap.add_argument('--ncommon-min', type=int, default=NCOMMON_MIN)
    ap.add_argument('--fvar-min', type=float, default=FVAR_MIN)
    ap.add_argument('--out', default=None, help='default REPO_ROOT/output/fold_pairs/candidates_<select>.json')
    ap.add_argument('--smoke', type=int, default=0, help='score only the first N pairs (writes nothing)')
    args = ap.parse_args()

    if args.smoke:
        recs, meta = score_all(args.cache, args.len_tol, args.workers, smoke=args.smoke)
        for r in recs[:20]:
            print(r, flush=True)
        return
    if os.path.exists(args.scored) and not args.rescore:
        recs, meta = load_scored(args.scored)
        if not recs:   # FIXED(code_fix P24, 2026-10-16): new (old: none): an empty file from an earlier failed run
            print(f'[bioemu-miner] ERROR: {args.scored} holds 0 scored pairs; rerun with '
                  f'--rescore after fixing TM-align', file=sys.stderr, flush=True)
            sys.exit(1)
        print(f'[bioemu-miner] reusing {len(recs)} scored pairs from {args.scored} '
              f'(len_tol={meta.get("len_tol")})', flush=True)
        if meta.get('len_tol') is not None and meta['len_tol'] != args.len_tol:
            print(f'  WARNING: --len-tol {args.len_tol} differs from the scored file '
                  f'({meta["len_tol"]}); pass --rescore to recompute', flush=True)
    else:
        recs, meta = score_all(args.cache, args.len_tol, args.workers)
        # FIXED(code_fix P24, 2026-10-16): old: scored_pairs.json was written even when no
        #   pair was scored (e.g. TM-align missing: every pair failed), and later steps read
        #   the empty file. New: refuse to write it and report the first error.
        if not recs:
            why = meta.get('first_error') or (
                f'no error recorded; all {meta.get("n_unscored_none")} pairs returned no result '
                f'(TM-align output not parsed, non-finite coordinates or < 30 aligned residues)')
            print(f'[bioemu-miner] ERROR: 0 of {meta.get("n_pairs_tried")} pairs scored; '
                  f'{args.scored} NOT written. First error: {why}. Check TM-align '
                  f'(ENSEMBITS_TMALIGN, now {M.TMALIGN!r}).', file=sys.stderr, flush=True)
            sys.exit(1)
        os.makedirs(os.path.dirname(args.scored), exist_ok=True)
        json.dump({'meta': meta, 'records': recs}, open(args.scored, 'w'), indent=1)
        print(f'[bioemu-miner] -> {args.scored}', flush=True)

    if args.select is None:
        return
    top_n = args.top_n if args.top_n is not None else TOP_N_DEFAULT[args.select]
    top, n_cand = select_pairs(recs, args.select, top_n, args.tm_min, args.ncommon_min, args.fvar_min)
    # Report ties at the top-N boundary (cos1 has 3 decimals), so a reader can see
    # whether the cut is sharp.
    n_tied = 0
    if top and n_cand > len(top):
        # FIXED(2026-10-23, fold-pair run crash): old = `_, all_c = select_pairs(...)`
        #   unpacked (top, n_candidates) in the wrong order (TypeError on the int). New =
        #   take the ranked list; only the reported tie count used it, the selection did not.
        all_c, _ = select_pairs(recs, args.select, n_cand, args.tm_min, args.ncommon_min, args.fvar_min)
        n_tied = sum(1 for r in all_c[len(top):] if r['cos1'] == top[-1]['cos1'])
    out = args.out or str(C.candidates_path(args.select))
    res = {'select': args.select,
           'rank': 'cos1 ascending' if args.select == 'divergent' else 'cos1 descending',
           'filters': {'tm_max_min': args.tm_min, 'ncommon_min': args.ncommon_min,
                       'fvarA_min': args.fvar_min, 'fvarB_min': args.fvar_min,
                       'same_domain_pairs': 'excluded'},
           'top_n': top_n, 'n_scored': len(recs), 'n_candidates': n_cand,
           'n_tied_beyond_cut': n_tied, 'scored_meta': meta, 'pairs': top}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)
    print(f'[bioemu-miner] {args.select}: {n_cand} candidates pass the filters, '
          f'top {len(top)} -> {out}' + (f'  ({n_tied} more tied at the cut)' if n_tied else ''), flush=True)


if __name__ == '__main__':
    main()
