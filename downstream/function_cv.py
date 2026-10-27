"""MISATO EC / GO cross-validation: loader, per-fold split maps, result records and checks.

What it does
------------
Authors' decision 2026-10-19 (option A, camera_ready/decision_brief_20261019.md §2 / §4):
the EC and GO probes are scored on DISTINCT PROTEINS with grouped, label-stratified 5-fold
cross-validation instead of the fixed v2 splits (function labels are per protein, but
MISATO holds one entry per protein-ligand complex, e.g. 284 carbonic anhydrase II
complexes; on the fixed splits the large families were forced into train and the val / test
class distributions shifted). The CV file `data/misato_function_cv_v1.json` is built by
`data_prep/build_misato_function_cv.py`; this module is what the probes, runners and
compilers use to read it:

  * `load_function_cv()`: the CV file and its record {file, sha256, schema}; refuses a file
    whose sidecar does not carry its sha256, and a file built on other GO / EC labels or
    another split chain table than the current ones (the representatives, groups and class
    lists would be stale);
  * `rotation()` / `cv_split_map()`: rotation r of a task and mode: test = fold r,
    val = fold (r + 1) mod 5, train = the other three folds ({PID: 'train'|'val'|'test'},
    representatives only);
  * `cv_result_block()`: the `function_cv` block of a result JSON (file, sha256, task, mode,
    fold, the val fold, rotation sizes, class-list sha256);
  * `cv_result_problem()` / `stale_cv_outputs()`: why a CV result cannot be compiled or
    reused (no block, another CV file, another task);
  * `cv_tag()`: the file-name tag of a (mode, fold) run (`cv_<mode>_f<r>`);
  * `aggregate_cv()`: the table cell of one method: per seed the mean over the 5 folds of
    the per-fold metric, then mean and sample std (ddof = 1) over the seeds; per-fold
    values are returned too.

Paper
-----
`appendix.tex` \\label{app:downstreams} (EC / GO probes) and \\label{app:experiments}; the
cross-validation protocol is not described in the paper yet (Open, audit D.6: paper text
of the EC/GO CV protocol: distinct proteins, grouped stratified 5-fold CV, aggregation;
text pass item).

Origin
------
New (2026-10-19, authors' decision option A).

Used by the reported model: yes (EC and GO rows of every table).

Inputs / outputs
----------------
reads  data/misato_function_cv_v1.json (+ .provenance.json)   (relative to REPO_ROOT;
       ensembits.paths.MISATO_FUNCTION_CV), and for the input check the current GO / EC
       label files and data/misato_splits_v2.chains.json
writes nothing

How to run
----------
Library module (imported by the EC / GO probes, the two runners and the compiler); tests in
tests/test_function_cv.py.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402

CV_SCHEMA = 'misato-function-cv-v2'
CV_TASKS = ('go', 'ec')
CV_MODES = ('sequence', 'structure')
N_FOLDS = 5
SPLITS = ('train', 'val', 'test')
AGGREGATION_RULE = ('per seed: mean over the 5 folds of the per-fold metric (each fold scored '
                    'once as the evaluation split [test or val], by the head trained on the '
                    'three training folds of its rotation); table cell: mean and sample std '
                    '(ddof = 1) over the seeds')


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _file_sha256(path) -> str:
    from ensembits.featurize import file_sha256
    return file_sha256(path)


def cv_path(path=None) -> Path:
    """The CV file (default ensembits.paths.MISATO_FUNCTION_CV)."""
    from ensembits.paths import MISATO_FUNCTION_CV
    return Path(path) if path is not None else Path(MISATO_FUNCTION_CV)


def current_inputs() -> dict:
    """sha256 of the inputs the CV file depends on, as they are now (labels, chain table)."""
    from go_labels import GO_LABELS_V2
    from ec_labels import EC_LABELS
    from ensembits.paths import MISATO_SPLITS
    chains = Path(MISATO_SPLITS).with_name(Path(MISATO_SPLITS).stem + '.chains.json')
    out = {}
    for name, p in (('go_labels', GO_LABELS_V2), ('ec_labels', EC_LABELS),
                    ('split_chains', chains)):
        out[name] = _file_sha256(p) if Path(p).exists() else None
    return out


def load_function_cv(path=None, check_inputs: bool = True) -> tuple[dict, dict]:
    """(CV dict, record {file, sha256, schema}).

    SystemExit when the file or its sidecar is missing, the schema is not CV_SCHEMA, the
    sha256 differs from the sidecar's, or (check_inputs) the GO / EC labels or the split
    chain table it was built on are not the current files.
    """
    p = cv_path(path)
    side = Path(str(p) + '.provenance.json')
    if not p.exists() or not side.exists():
        raise SystemExit(f'{p} (or its .provenance.json) is missing: run '
                         f'data_prep/build_misato_function_cv.py')
    prov = json.loads(side.read_text())
    sha = _file_sha256(p)
    if prov.get('schema') != CV_SCHEMA or prov.get('output_sha256') != sha:
        raise SystemExit(f'{p.name}: sha256 {sha[:12]} / schema {prov.get("schema")!r} do not '
                         f'match its sidecar ({str(prov.get("output_sha256"))[:12]}, '
                         f'{CV_SCHEMA!r}); rebuild it')
    cv = json.loads(p.read_text())
    if cv.get('schema') != CV_SCHEMA:
        raise SystemExit(f'{p.name}: schema {cv.get("schema")!r}, expected {CV_SCHEMA!r}')
    if check_inputs:
        cur = current_inputs()
        for name, rec in (cv.get('inputs') or {}).items():
            if name in cur and rec.get('sha256') != cur[name]:
                raise SystemExit(f'{p.name} was built on {name} sha256 '
                                 f'{str(rec.get("sha256"))[:12]}, the current file is '
                                 f'{str(cur[name])[:12]}: rebuild it with '
                                 f'data_prep/build_misato_function_cv.py --overwrite')
    return cv, {'file': _rel(p), 'sha256': sha, 'schema': CV_SCHEMA}


def rotation(cv: dict, task: str, mode: str, fold: int) -> dict:
    """Rotation `fold` of a task and mode: {'test_fold', 'val_fold', 'train', 'val', 'test',
    'classes', ...} (pid lists sorted, upper case)."""
    if task not in CV_TASKS or mode not in CV_MODES:
        raise ValueError(f'task {task!r} / mode {mode!r}')
    if not 0 <= int(fold) < N_FOLDS:
        raise ValueError(f'fold {fold} not in 0..{N_FOLDS - 1}')
    rot = cv['tasks'][task]['modes'][mode]['rotations'][int(fold)]
    if int(rot['test_fold']) != int(fold):
        raise ValueError(f'rotation {fold} of {task}/{mode} has test fold {rot["test_fold"]}')
    return rot


def cv_split_map(cv: dict, task: str, mode: str, fold: int) -> dict:
    """{PID: 'train'|'val'|'test'} of rotation `fold` (representatives of the task only)."""
    rot = rotation(cv, task, mode, fold)
    out = {}
    for s in SPLITS:
        for p in rot[s]:
            if p in out:
                raise ValueError(f'{p} in {out[p]} and {s} ({task}/{mode}/fold {fold})')
            out[str(p).upper()] = s
    return out


def classes_sha256(classes) -> str:
    """sha256 of a class list in its order (as function_common.classes_record)."""
    return hashlib.sha256('\n'.join(str(c) for c in classes).encode()).hexdigest()


def cv_result_block(rec: dict, cv: dict, task: str, mode: str, fold: int) -> dict:
    """{'function_cv': {...}} for a result JSON of rotation `fold`."""
    rot = rotation(cv, task, mode, fold)
    return {'function_cv': {**rec, 'task': task, 'mode': mode, 'fold': int(fold),
                            'test_fold': int(rot['test_fold']), 'val_fold': int(rot['val_fold']),
                            'n_proteins': {s: len(rot[s]) for s in SPLITS},
                            'aggregation': AGGREGATION_RULE}}


def check_classes(cv: dict, task: str, mode: str, fold: int, classes, depth=None) -> None:
    """ValueError unless the probe's class list equals the one stored in the CV file."""
    rot = rotation(cv, task, mode, fold)
    want = rot['classes'][str(depth)] if task == 'ec' else rot['classes']
    if list(map(str, classes)) != list(map(str, want)):
        raise ValueError(f'{task}/{mode}/fold {fold}' + (f' d{depth}' if depth else '') +
                         f': probe classes ({len(classes)}) differ from the CV file '
                         f'({len(want)}); rebuild the CV file or check the labels')


def cv_tag(mode: str, fold: int) -> str:
    """File-name tag of a CV run: cv_<mode>_f<fold>."""
    return f'cv_{mode}_f{int(fold)}'


def add_cv_args(ap) -> None:
    """--cv-mode / --fold of the EC and GO probes."""
    ap.add_argument('--cv-mode', choices=CV_MODES, default=None,
                    help='EC / GO cross-validation (authors\' decision 2026-10-19): distinct '
                         'proteins, grouped stratified 5-fold CV of data/misato_function_cv_v1'
                         '.json; with --fold. Replaces --split for the paper rows (--split '
                         'is kept for the old fixed-split runs).')
    ap.add_argument('--fold', type=int, default=None, choices=range(N_FOLDS),
                    help='CV rotation r (test = fold r, val = fold r+1 mod 5)')


def check_cv_args(ap, args) -> bool:
    """True for a CV run; argparse error unless --cv-mode and --fold come together."""
    if (args.cv_mode is None) != (args.fold is None):
        ap.error('--cv-mode and --fold go together')
    return args.cv_mode is not None


def cv_result_problem(js: dict, task: str | None = None, path=None,
                      mode: str | None = None, fold: int | None = None) -> str | None:
    """None if result `js` was built on the current CV file (and task/mode/fold), else the reason."""
    blk = (js or {}).get('function_cv')
    if not isinstance(blk, dict) or not blk.get('sha256'):
        return 'no function_cv record (not a cross-validation run)'
    p = cv_path(path)
    if not p.exists():
        return f'current CV file {p.name} missing'
    cur = _file_sha256(p)
    if blk['sha256'] != cur:
        return f'CV file sha256 {blk["sha256"][:12]} is not the current {p.name} ({cur[:12]})'
    if task is not None and blk.get('task') != task:
        return f'CV task {blk.get("task")!r} is not {task!r}'
    if mode is not None and blk.get('mode') != mode:
        return f'CV mode {blk.get("mode")!r} is not {mode!r}'
    if fold is not None and int(blk.get('fold', -1)) != int(fold):
        return f'CV fold {blk.get("fold")!r} is not {fold!r}'
    return None


def stale_cv_outputs(paths, path=None) -> list:
    """Existing CV result JSONs not built on the current CV file: [(path, reason)]."""
    bad = []
    for o in paths:
        if o is None:
            continue
        o = Path(o)
        if not o.exists() or o.suffix != '.json':
            continue
        try:
            why = cv_result_problem(json.loads(o.read_text()), None, path)
        except Exception as e:                          # noqa: BLE001
            why = f'unreadable ({type(e).__name__})'
        if why:
            bad.append((o, why))
    return bad


def aggregate_cv(values: dict, n_folds: int = N_FOLDS, require_all_folds: bool = True):
    """values: {seed: {fold: metric}} -> ((mean, std or None, n_seeds) or None, per-fold).

    Per seed the folds are averaged (a seed with a missing fold is left out when
    require_all_folds); the cell is the mean and sample std over those seed means.
    per-fold = {fold: [values over seeds]}.
    """
    seed_means = []
    dropped_seeds = []
    per_fold = {f: [] for f in range(n_folds)}
    for s in sorted(values):
        fv = {int(f): v for f, v in values[s].items() if v is not None and v == v}
        for f, v in fv.items():
            per_fold.setdefault(f, []).append(v)
        if require_all_folds and len(fv) < n_folds:
            dropped_seeds.append(s)
            continue
        if fv:
            seed_means.append(sum(fv.values()) / len(fv))
    n_seeds = len(seed_means)
    if dropped_seeds:
        print(f'[aggregate_cv] n_seeds={n_seeds} (dropped {len(dropped_seeds)} '
              f'seed(s) with incomplete folds: {dropped_seeds})', flush=True)
    else:
        print(f'[aggregate_cv] n_seeds={n_seeds}', flush=True)
    if not seed_means:
        return None, per_fold
    return ((st.mean(seed_means), st.stdev(seed_means) if len(seed_means) > 1 else None,
             n_seeds), per_fold)
