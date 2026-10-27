"""MISATO EC labels (v2): path, loader, provenance record and result checks.

What it does
------------
One place for the EC label file of the MISATO EC probes (authors' decision D8,
2026-10-18, EC audit):

  * `EC_LABELS` = `ensembits.paths.MISATO_EC_LABELS` (data/annotations/misato_ec_v2.json,
    built by `data_prep/build_misato_ec_labels.py`: polyprotein chains labelled by the
    mature products they cover; since 2026-10-19 a construct that contains only one domain
    of a multi-domain protein is labelled with the EC numbers of that domain, table
    DOMAIN_OVERRIDES);
  * `load_ec_labels()` returns {PID: [EC numbers]} and the label record; it refuses a file
    without the v2 sidecar (e.g. the old per-PDB SIFTS union misato_ec.json), so the old
    labels cannot be read by accident;
  * `ec_labels_record()` is the block written into every EC result JSON
    (`"ec_labels": {path, sha256, schema}`);
  * `ec_result_problems()` / `stale_ec_outputs()` tell the compilers and runners which
    result JSONs were not built on the current label file (results on the old labels carry
    no `ec_labels` block).

The wildcard rule (a partial EC number such as "3.1.-.-" counts only at depths whose
fields are all specified) lives in `truncate_ec` of `tokenize_misato_and_eval.py` and
`baselines/ec_classify_conv1d_misato.py` (identical copies).

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{EC Function Prediction} (EC labels of
the MISATO complexes).

Origin
------
New (2026-10-18, authors' decision D8).

Used by the reported model: yes (EC probe labels of every method).

Inputs / outputs
----------------
  reads   data/annotations/misato_ec_v2.json (+ .provenance.json)   (relative to REPO_ROOT)
  writes  nothing

How to run
----------
Imported by `tokenize_misato_and_eval.py`, `baselines/ec_classify_conv1d_misato.py`,
`_compile_rebuttal_results.py` and the runners; not a script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT, MISATO_EC_LABELS  # noqa: E402
from ensembits.featurize import file_sha256  # noqa: E402

EC_SCHEMA = 'misato-ec-v3'
EC_LABELS = MISATO_EC_LABELS


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sidecar(path) -> dict | None:
    side = Path(str(path) + '.provenance.json')
    return json.loads(side.read_text()) if side.exists() else None


def ec_labels_record(path=None) -> dict:
    """{path, sha256, schema} of the EC label file (the `ec_labels` block of result JSONs)."""
    p = Path(path) if path is not None else Path(EC_LABELS)
    side = _sidecar(p) or {}
    return {'path': _rel(p), 'sha256': file_sha256(p), 'schema': side.get('schema')}


def load_ec_labels(path=None) -> tuple[dict, dict]:
    """({PID: [EC numbers]}, ec_labels_record()) of the v2 label file.

    SystemExit when the file is missing or has no v2 sidecar (old misato_ec.json).
    """
    p = Path(path) if path is not None else Path(EC_LABELS)
    if not p.exists():
        raise SystemExit(f'{p} missing: run data_prep/build_misato_ec_labels.py')
    side = _sidecar(p)
    if not side or side.get('schema') != EC_SCHEMA:
        raise SystemExit(f'{p.name}: no {EC_SCHEMA} sidecar (the old per-PDB SIFTS union? '
                         f'authors\' decision D8); build data/annotations/misato_ec_v2.json '
                         f'with data_prep/build_misato_ec_labels.py')
    labels = json.loads(p.read_text())
    rec = ec_labels_record(p)
    if side.get('output_sha256') and side['output_sha256'] != rec['sha256']:
        raise SystemExit(f'{p.name}: sha256 {rec["sha256"][:12]} differs from its sidecar '
                         f'({side["output_sha256"][:12]}); rebuild it')
    return labels, rec


def ec_result_problems(d: dict, path=None) -> list:
    """Reasons an EC result JSON must not be compiled or reused ([] = fine)."""
    p = Path(path) if path is not None else Path(EC_LABELS)
    rec = (d or {}).get('ec_labels')
    if not isinstance(rec, dict) or not rec.get('sha256'):
        return ['built on the old EC labels (no ec_labels block; rerun on misato_ec_v2)']
    if not p.exists():
        return [f'current EC label file {p.name} missing']
    cur = file_sha256(p)
    if rec['sha256'] != cur:
        return [f'EC label sha256 {rec["sha256"][:12]} != current {cur[:12]}']
    return []


def stale_ec_outputs(paths, path=None) -> list:
    """Existing EC result JSONs not built on the current EC labels: [(path, reason)]."""
    bad = []
    for o in paths:
        if o is None:
            continue
        o = Path(o)
        if not o.exists() or o.suffix != '.json':
            continue
        try:
            why = ec_result_problems(json.loads(o.read_text()), path)
        except Exception as e:                          # noqa: BLE001
            why = [f'unreadable ({type(e).__name__})']
        if why:
            bad.append((o, why[0]))
    return bad
