"""Version tags of derived data, and checks that refuse stale caches.

What it does
------------
`MISATO_PARSER_VERSION` is written into every MISATO backbone file by
`data_prep/build_misato_real_bb.py`. Files without it were written by the old
parser, which put proline Cδ in the CA slot (code_fix §4). The check below is
called by every script that reads those files, so a stale cache cannot be used
by accident.

Since 2026-10-16 (code_fix P28) it also holds the provenance-sidecar helpers
(`provenance_path`, `write_provenance`, `read_provenance`, `hash_arrays_by_pid`):
builders write `<output>.provenance.json` next to a derived cache.

Since 2026-10-18 (benchmark_audit D3) it holds the MISATO split-file helpers
(`load_misato_splits`, `misato_splits_record`, `misato_split_problem`,
`stale_split_outputs`): every MISATO probe reads the v2 split file and records it, and
results built on another split file are refused.

Paper
-----
not described in the paper (infrastructure only)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MISATO_PARSER_VERSION = "cx_2026-10-15"


def misato_parser_version(npz_path) -> str | None:
    """The parser version stored in a MISATO backbone file (None = old parser)."""
    with np.load(npz_path, allow_pickle=False) as z:
        return str(z["parser_version"]) if "parser_version" in z.files else None


def require_misato_parser(directory, n_check: int = 50, allow_old: bool = False) -> None:
    """Raise SystemExit unless the MISATO files in `directory` come from the fixed parser.

    Checks up to `n_check` files spread over the sorted file list. `allow_old`
    skips the check (only to reproduce results of checkpoints trained before the fix).
    """
    if allow_old:
        return
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise SystemExit(f"no MISATO backbone files in {directory}")
    for f in files[:: max(1, len(files) // n_check)]:
        v = misato_parser_version(f)
        if v != MISATO_PARSER_VERSION:
            raise SystemExit(
                f"{f} was written by MISATO parser {v!r}, expected {MISATO_PARSER_VERSION!r} "
                f"(code_fix §4). Rebuild with data_prep/build_misato_real_bb.py, or pass "
                f"--allow-old-parser to reproduce a checkpoint trained before the fix.")


# FIXED(code_fix P28, 2026-10-16): provenance sidecars. Old: derived caches (coords
#   pickles, RMSF labels, descriptor caches) carried no record of what they were built
#   from. New: builders write <output>.provenance.json next to the output (never extra
#   keys inside pid-keyed npz/pkl files, whose loaders iterate the keys as pids), and
#   consumers can compare the recorded hashes.
def provenance_path(output) -> Path:
    """Sidecar of a cache file: `<output>.provenance.json` (e.g. misato_rmsf.npz.provenance.json)."""
    return Path(str(output) + ".provenance.json")


def write_provenance(output, info: dict) -> Path:
    """Write `info` (JSON-serialisable) as the provenance sidecar of `output`."""
    p = provenance_path(output)
    p.write_text(json.dumps(info, indent=1, sort_keys=True, default=str))
    return p


def read_provenance(output) -> dict | None:
    """The provenance sidecar of `output`, or None if it has none (built before 2026-10-16)."""
    p = provenance_path(output)
    return json.loads(p.read_text()) if p.exists() else None


def hash_arrays_by_pid(arrays: dict) -> str:
    """sha256 over the sorted pids and, per pid, the dtype, shape and bytes of its array.

    Same value for the same {pid: array} content, whatever the dict order; used to tie a
    label file to the exact coordinates it was computed from.
    """
    h = hashlib.sha256()
    for pid in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[pid]))
        h.update(str(pid).encode())
        h.update(f"|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


# FIXED(benchmark_audit D3, 2026-10-18): MISATO probe splits v2. Old: the MISATO probes (EC,
#   GO, binding site, affinity) read data/misato_splits.json directly and recorded nothing
#   about it (the RMSF probe recorded its hash). New: one split file for every MISATO probe,
#   ensembits.paths.MISATO_SPLITS (data/misato_splits_v2.json, MMseqs2 30 % sequence
#   clusters / CATH H, data_prep/build_misato_splits_v2.py); `load_misato_splits` refuses a
#   file without the v2 schema, every result JSON records `misato_splits_record()` (path,
#   sha256, schema) and the compilers / runners refuse results whose recorded split file is
#   not the current one (`misato_split_problem`, `stale_split_outputs`).
MISATO_SPLITS_SCHEMA = "misato-splits-v2"


def _misato_splits_path(path=None) -> Path:
    from ensembits.paths import MISATO_SPLITS
    return Path(path) if path is not None else Path(MISATO_SPLITS)


def load_misato_splits(path=None) -> dict:
    """The MISATO split file (default ensembits.paths.MISATO_SPLITS); SystemExit unless it
    carries the v2 schema (stats.schema), so the old leaky file cannot be read by accident."""
    p = _misato_splits_path(path)
    if not p.exists():
        raise SystemExit(f"{p} missing: run data_prep/build_misato_splits_v2.py")
    s = json.loads(p.read_text())
    schema = (s.get("stats") or {}).get("schema")
    if schema != MISATO_SPLITS_SCHEMA:
        raise SystemExit(f"{p.name}: split schema {schema!r}, expected {MISATO_SPLITS_SCHEMA!r} "
                         f"(benchmark_audit D3; build it with data_prep/build_misato_splits_v2.py)")
    return s


def misato_splits_record(path=None) -> dict:
    """{splits_file, splits_sha256, splits_schema} of the MISATO split file, for result JSONs
    (same keys as rmsf_common.splits_record, plus the schema)."""
    from ensembits.featurize import file_sha256
    from ensembits.paths import REPO_ROOT
    p = _misato_splits_path(path)
    try:
        rel = str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(p)
    s = load_misato_splits(p)
    return {"splits_file": rel, "splits_sha256": file_sha256(p),
            "splits_schema": s["stats"]["schema"]}


def misato_split_problem(js: dict, path=None) -> str | None:
    """None if result `js` records the current MISATO split file (sha256), else the reason."""
    from ensembits.featurize import file_sha256
    p = _misato_splits_path(path)
    got = (js or {}).get("splits_sha256")
    if got is None:
        return "no split-file record (splits_sha256): built on the old misato_splits.json"
    if not p.exists():
        return f"current split file {p.name} missing"
    if got != file_sha256(p):
        return (f"split file {js.get('splits_file')!r} sha256 {str(got)[:12]} is not the "
                f"current {p.name} ({file_sha256(p)[:12]})")
    return None


def stale_split_outputs(paths, path=None) -> list:
    """Existing result JSONs (EC / GO / binding) not built on the current MISATO split file:
    [(path, reason)]. The runners refuse to skip (= reuse) them."""
    bad = []
    for o in paths:
        o = Path(o)
        if o is None or not o.exists() or o.suffix != ".json":
            continue
        try:
            js = json.loads(o.read_text())
        except Exception as e:                          # noqa: BLE001
            bad.append((o, f"unreadable ({type(e).__name__})")); continue
        why = misato_split_problem(js, path)
        if why:
            bad.append((o, why))
    return bad
