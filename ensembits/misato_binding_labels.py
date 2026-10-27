"""MISATO binding-site labels v2: path, loader and the record every result JSON carries.

What it does
------------
The per-residue binding-site labels of the MISATO downstream task are the PDBbind v2020
refined-set pocket (`<pid>_pocket.pdb`) mapped onto the MISATO residue list (the residue
order of `misato_real_bb`, which every token cache follows). Version 2 is built by
`data_prep/build_misato_binding_site_labels.py`; version 1 (`data/cached_descriptors/
misato_binding_site.npz`, from the ProteinShake MISATO dataset) matched the pocket's author
residue numbers against MISATO's sequential residue counter and is wrong for most complexes
(benchmark audit, authors' decision D4).

  * `MISATO_BINDING_SITE`: the v2 label file (`data/annotations/misato_binding_site_v2.npz`,
    {pid: int8 array of length L}); ENSEMBITS_MISATO_BINDING_SITE overrides it (tests only).
  * `load_binding_site_labels()`: reads it, refusing a file whose provenance sidecar does not
    carry the v2 schema or whose sha256 differs from the one the builder recorded (so the
    old file, or a hand-edited one, cannot be read by accident).
  * `binding_labels_record()`: {binding_labels_file, binding_labels_sha256,
    binding_labels_schema}, written into every binding result JSON.
  * `binding_label_problem(js)`: None if a result JSON records the current label file, else
    the reason; the probes refuse to merge into, and the compilers refuse to report, binding
    results without it (`stale_binding_label_outputs` for the runners).

Only the binding-site task depends on these labels; affinity results in the same JSON do
not (callers check binding entries only).

Paper
-----
`appendix.tex` \\label{app:downstreams}, \\paragraph{Binding Site} (labels of the binding
site task). The definition (PDBbind pocket residues) is unchanged; only the residue
alignment is fixed.

Origin
------
New (2026-10-18, benchmark audit D4). Mirrors `ensembits.data_versions.misato_splits_record`
/ `misato_split_problem` (D3).

Used by the reported model: yes (labels of the binding-site rows of the function tables).

Inputs / outputs
----------------
reads REPO_ROOT/data/annotations/misato_binding_site_v2.npz and its .provenance.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ensembits.paths import REPO_ROOT

BINDING_LABELS_SCHEMA = "misato-binding-site-v3"
MISATO_BINDING_SITE = Path(os.environ.get(
    "ENSEMBITS_MISATO_BINDING_SITE",
    REPO_ROOT / "data" / "annotations" / "misato_binding_site_v3.npz"))
# v1 labels (ProteinShake author-number match; wrong alignment). Kept only for the builder's
# old-vs-new comparison; no probe may read it.
MISATO_BINDING_SITE_LEGACY = REPO_ROOT / "data" / "cached_descriptors" / "misato_binding_site.npz"


def _path(path=None) -> Path:
    return Path(path) if path is not None else Path(MISATO_BINDING_SITE)


def _check_sidecar(p: Path) -> dict:
    """The v2 sidecar of label file `p`; SystemExit unless schema and sha256 match."""
    from ensembits.data_versions import read_provenance
    from ensembits.featurize import file_sha256
    if not p.exists():
        raise SystemExit(f"{p} missing: run data_prep/build_misato_binding_site_labels.py")
    prov = read_provenance(p)
    if not prov or prov.get("schema") != BINDING_LABELS_SCHEMA:
        raise SystemExit(f"{p.name}: binding-site label schema "
                         f"{(prov or {}).get('schema')!r}, expected {BINDING_LABELS_SCHEMA!r} "
                         f"(benchmark audit D4; build it with "
                         f"data_prep/build_misato_binding_site_labels.py)")
    if prov.get("sha256") != file_sha256(p):
        raise SystemExit(f"{p.name}: sha256 differs from the one its builder recorded "
                         f"(file changed after it was built); rebuild it")
    return prov


def load_binding_site_labels(path=None) -> dict:
    """{pid: float32 array} of the v2 label file (default MISATO_BINDING_SITE).

    Complexes without a usable PDBbind pocket alignment are absent from the file (listed in
    its sidecar / report), so a probe skips them instead of seeing an all-negative protein.
    """
    p = _path(path)
    _check_sidecar(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def binding_labels_record(path=None) -> dict:
    """{binding_labels_file, binding_labels_sha256, binding_labels_schema} for result JSONs."""
    from ensembits.featurize import file_sha256
    p = _path(path)
    prov = _check_sidecar(p)
    try:
        rel = str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(p)
    return {"binding_labels_file": rel, "binding_labels_sha256": file_sha256(p),
            "binding_labels_schema": prov["schema"]}


def binding_label_problem(js: dict, path=None) -> str | None:
    """None if result `js` records the current binding-site label file (sha256), else why."""
    from ensembits.featurize import file_sha256
    p = _path(path)
    got = (js or {}).get("binding_labels_sha256")
    if got is None:
        return ("no binding-label record (binding_labels_sha256): built on the old "
                "misato_binding_site.npz (wrong residue alignment, benchmark audit D4)")
    if not p.exists():
        return f"current binding-site label file {p.name} missing"
    if got != file_sha256(p):
        return (f"binding-site labels {js.get('binding_labels_file')!r} sha256 {str(got)[:12]} "
                f"are not the current {p.name} ({file_sha256(p)[:12]})")
    return None


def has_binding_results(js: dict) -> bool:
    """True if result `js` holds any binding-site entry (`<split>_binding`)."""
    return any(str(k).endswith("_binding") for k in ((js or {}).get("results") or {}))


def stale_binding_label_outputs(paths, path=None) -> list:
    """Existing result JSONs with binding-site entries not built on the current label file:
    [(path, reason)]. The runners refuse to skip (= reuse) them."""
    bad = []
    for o in paths:
        if o is None:
            continue
        o = Path(o)
        if not o.exists() or o.suffix != ".json":
            continue
        try:
            js = json.loads(o.read_text())
        except Exception as e:                          # noqa: BLE001
            bad.append((o, f"unreadable ({type(e).__name__})")); continue
        if not has_binding_results(js):
            continue
        why = binding_label_problem(js, path)
        if why:
            bad.append((o, why))
    return bad
