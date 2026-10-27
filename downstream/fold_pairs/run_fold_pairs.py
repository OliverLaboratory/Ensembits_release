"""Run the fold-matched pair test end to end, or print its run order (--dry-run).

What it does
------------
Runs the steps of `downstream/fold_pairs/` in order, each as a subprocess with the
current Python interpreter, stopping at the first failure:

  extract          bioemu_extract_ca.py                     C-alpha cache, one file per system
  mine_divergent   bioemu_fold_dynamics_miner.py --select divergent   (scores all pairs once)
  mine_similar     bioemu_fold_dynamics_miner.py --select similar     (reuses the scores)
  verify_divergent verify_fold_dynamics_pair.py --mode divergent
  verify_similar   verify_fold_dynamics_pair.py --mode similar
  tokenize         bioemu_token_tokenize.py --population pairs
  floor            bioemu_token_tokenize.py --population floor
  compare          bioemu_token_compare.py
  (table           build_pair_table.py: markdown / LaTeX table from the comparison;
                   run it after compare, it is not a step here)

Preflight (2026-10-16, code_fix P24): before any step that calls TM-align
(mine_*, verify_*, compare) the TM-align executable (`ensembits.paths.TMALIGN_BIN`:
ENSEMBITS_TMALIGN, else TMalign / USalign on PATH) is resolved and test-executed;
the run refuses to start if it is missing.

Prerequisite (not run here unless --with-real-bb): the standard bb_10 cache
`data/cached_descriptors/bioemu_cath_real_bb/` from `data_prep/build_bioemu_real_bb.py`;
the floor step reads it, and the tokenize step falls back to computing missing
systems with the same code.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.
Infrastructure only.

Origin
------
New file written for this codebase (the original steps were run by hand; commands in
`rebuttal/fold_dynamics_mining.md` and `reproducibility.md` of the research repo).
Fixes 2026-10-16 (FIXED markers): TM-align preflight (code_fix P24); --model-dir /
--overwrite passed to the tokenizer steps (P21).

Used by the reported model: yes (drives the scripts that tokenize with it).

Inputs / outputs
----------------
Those of the individual steps (all outputs under REPO_ROOT/output/fold_pairs/).

How to run
----------
    python downstream/fold_pairs/run_fold_pairs.py --dry-run
    python downstream/fold_pairs/run_fold_pairs.py [--steps extract,mine_divergent,...] \\
        [--workers 64] [--device cuda:0] [--floor-limit N] [--with-real-bb] \\
        [--model-dir <checkpoint>] [--overwrite]
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fp_common as C   # noqa: E402

STEP_ORDER = ('real_bb', 'extract', 'mine_divergent', 'mine_similar', 'verify_divergent',
              'verify_similar', 'tokenize', 'floor', 'compare')
DEFAULT_STEPS = STEP_ORDER[1:]
# Steps that call TM-align (fold_dynamics_miner.tmalign).
TMALIGN_STEPS = ('mine_divergent', 'mine_similar', 'verify_divergent', 'verify_similar', 'compare')


def tmalign_preflight(binary=None):
    """Resolve and test-execute the TM-align executable; -> (ok, message).

    FIXED(code_fix P24, 2026-10-16): new. Without TM-align every pair failed inside the
    miner and an empty result was written. The executable is ensembits.paths.TMALIGN_BIN
    (ENSEMBITS_TMALIGN, else TMalign, else USalign on PATH).
    """
    if binary is None:
        from ensembits.paths import TMALIGN_BIN as binary
    path = binary if os.path.sep in str(binary) else shutil.which(str(binary))
    if not path or not os.path.exists(path):
        return False, (f'TM-align executable not found ({binary!r}). Install TMalign or USalign '
                       f'on PATH, or set ENSEMBITS_TMALIGN=/path/to/TMalign')
    try:
        r = subprocess.run([path, '-h'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, (f'TM-align at {path} could not be executed ({type(e).__name__}: {e}); '
                       f'set ENSEMBITS_TMALIGN to a working TMalign / USalign')
    out = (r.stdout or '') + (r.stderr or '')
    if 'TM' not in out:
        return False, (f'{path} ran but does not look like TM-align / US-align (no "TM" in its '
                       f'help output); set ENSEMBITS_TMALIGN')
    return True, f'TM-align: {path}'


def build_steps(args):
    here = _HERE.relative_to(C.CODE_ROOT)
    dp = Path('data_prep')
    tok_extra = ((['--model-dir', str(args.model_dir)] if args.model_dir else [])
                 + (['--overwrite'] if args.overwrite else []))
    s = {
        'real_bb': [str(dp / 'build_bioemu_real_bb.py'), '--root', str(C.BIOEMU_ROOT),
                    '--out', str(C.REAL_BB), '--k', '10', '--workers', str(min(args.workers, 16))],
        'extract': [str(here / 'bioemu_extract_ca.py'), '--workers', str(min(args.workers, 32)),
                    '--per-run', '30'],
        'mine_divergent': [str(here / 'bioemu_fold_dynamics_miner.py'), '--workers', str(args.workers),
                           '--len-tol', '0.15', '--select', 'divergent'],
        'mine_similar': [str(here / 'bioemu_fold_dynamics_miner.py'), '--workers', str(args.workers),
                         '--len-tol', '0.15', '--select', 'similar'],
        'verify_divergent': [str(here / 'verify_fold_dynamics_pair.py'), '--mode', 'divergent'],
        'verify_similar': [str(here / 'verify_fold_dynamics_pair.py'), '--mode', 'similar'],
        # FIXED(code_fix P21, 2026-10-16): --model-dir and --overwrite are passed through to
        #   the tokenizer (before, the tokenize / floor steps always used the default
        #   checkpoint and kept existing token files).
        'tokenize': [str(here / 'bioemu_token_tokenize.py'), '--population', 'pairs',
                     '--device', args.device] + tok_extra,
        'floor': [str(here / 'bioemu_token_tokenize.py'), '--population', 'floor',
                  '--device', args.device] + (['--floor-limit', str(args.floor_limit)] if args.floor_limit else [])
                 + tok_extra,
        'compare': [str(here / 'bioemu_token_compare.py')],
    }
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='print the run order and commands, run nothing')
    ap.add_argument('--steps', default=','.join(DEFAULT_STEPS),
                    help=f'comma list out of {",".join(STEP_ORDER)} (run in this fixed order)')
    ap.add_argument('--with-real-bb', action='store_true',
                    help='also run data_prep/build_bioemu_real_bb.py first (prerequisite of the floor step)')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--floor-limit', type=int, default=0)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--model-dir', default=None,
                    help='Ensembits checkpoint for the tokenize / floor steps (default: the '
                         "tokenizer's default, ensembits.paths.CKPT_DIR)")
    ap.add_argument('--overwrite', action='store_true',
                    help='tokenize / floor: redo tokenizers whose token file exists')
    args = ap.parse_args()
    want = [x for x in args.steps.split(',') if x]
    bad = [x for x in want if x not in STEP_ORDER]
    if bad:
        ap.error(f'unknown steps {bad}')
    if args.with_real_bb and 'real_bb' not in want:
        want.insert(0, 'real_bb')
    steps = build_steps(args)
    order = [x for x in STEP_ORDER if x in want]

    print('fold-matched pair test: extract -> mine divergent -> mine similar -> verify both '
          '-> tokenize -> floor -> compare')
    print(f'  REPO_ROOT = {C.ROOT}\n  outputs   = {C.OUT}\n  run from  = {C.CODE_ROOT}')
    if 'real_bb' not in order:
        have = C.REAL_BB.is_dir() and any(C.REAL_BB.glob('*.npz'))
        print(f'  prerequisite: {C.REAL_BB} ({"present" if have else "MISSING: run with --with-real-bb or data_prep/build_bioemu_real_bb.py"})')
    for i, name in enumerate(order, 1):
        cmd = [args.python] + steps[name]
        print(f'  [{i}] {name:<17} {" ".join(shlex.quote(c) for c in cmd)}')
    # FIXED(code_fix P24, 2026-10-16): new (old: none): TM-align preflight (refuse to start without it).
    if any(x in order for x in TMALIGN_STEPS):
        ok, msg = tmalign_preflight()
        print(f'  preflight: {msg}')
        if not ok and not args.dry_run:
            print(f'[abort] {msg}', file=sys.stderr)
            return 1
    if args.dry_run:
        return 0
    for name in order:
        cmd = [args.python] + steps[name]
        print(f'\n=== {name} ===', flush=True)
        subprocess.run(cmd, cwd=str(C.CODE_ROOT), check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
