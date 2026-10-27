"""Shared helpers of the analysis and figure scripts (checkpoint identity, cache checks).

What it does
------------
  * `resolve_model_dir(arg)`: a model directory given on the command line, as an
    absolute path, a path relative to the current directory, to REPO_ROOT or to
    the codebase root.
  * `ckpt_hash(model_dir)`: the first 12 hex digits of SHA-256(best.pt), used to
    name caches so tokens of one checkpoint are never reused for another.
  * `checkpoint_settings(ens)`: the dihedral convention and the resolved chain-break
    cutoff a loaded checkpoint expects (`ensembits.featurize.load_model` reads
    them from config.json; legacy checkpoints -> "legacy_plus180", no cutoff).
  * `cfg_get(cfg, key, default)`: config value with an explicit fallback (old
    configs miss keys such as `descriptor_mode` or `k`; new ones may have them).
  * `find_sidecar(path)` / `verify_sidecar(path, expect)`: look for a JSON sidecar
    next to a descriptor cache and refuse a cache whose recorded convention,
    cutoff or row count disagrees with what the caller expects.

Paper
-----
not described in the paper (infrastructure only)

Origin
------
New file (code_fix P20, 2026-10-16). Not copied from the research repo.

Used by the reported model: yes, by every script in this folder.

Inputs / outputs
----------------
reads `<model_dir>/best.pt` (hash only) and sidecar JSON files next to caches.

How to run
----------
imported by the scripts in `downstream/analyses/`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT  # noqa: E402
from ensembits.descriptor import resolve_chain_break_cutoff  # noqa: E402


def resolve_model_dir(arg) -> Path:
    """Model dir from the CLI: absolute, ./relative, REPO_ROOT-relative or CODE_ROOT-relative."""
    p = Path(arg)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), REPO_ROOT, CODE_ROOT):
        if (base / p / "config.json").exists():
            return (base / p).resolve()
    return (REPO_ROOT / p).resolve()


def ckpt_hash(model_dir, n: int = 12) -> str:
    """First n hex digits of SHA-256 over `<model_dir>/best.pt`."""
    h = hashlib.sha256()
    with open(Path(model_dir) / "best.pt", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def cfg_get(cfg: dict, key: str, default):
    """`cfg[key]` if present and not None, else `default` (never a KeyError)."""
    v = cfg.get(key)
    return default if v is None else v


def checkpoint_settings(ens) -> tuple[str, object]:
    """(dihedral_convention, resolved chain-break cutoff in Å or None) of a loaded model."""
    conv = ens.dihedral_convention
    return conv, resolve_chain_break_cutoff(ens.chain_break_cutoff, conv)


def _same_cutoff(a, b, conv) -> bool:
    ra = resolve_chain_break_cutoff(a, conv)
    rb = resolve_chain_break_cutoff(b, conv)
    if ra is None or rb is None:
        return ra is None and rb is None
    return abs(float(ra) - float(rb)) < 1e-6


SIDECAR_SUFFIXES = (".json", ".meta.json", ".sidecar.json")


def find_sidecar(path) -> Path | None:
    """A JSON sidecar of a cache file: <file>.provenance.json (written by the data
    builders), <file>.json, <stem>.json, <stem>.meta.json or <stem>.sidecar.json
    (whichever exists first), else None."""
    path = Path(path)
    # FIXED(code_fix P20, 2026-10-16): the data builders write `<file>.provenance.json`
    #   (ensembits.data_versions.provenance_path); look there first.
    from ensembits.data_versions import provenance_path
    cands = [Path(provenance_path(path)), path.with_name(path.name + ".json")]
    cands += [path.with_name(path.stem + s) for s in SIDECAR_SUFFIXES]
    for c in cands:
        if c.exists() and c != path:
            return c
    return None


def verify_sidecar(path, *, convention: str, chain_break_cutoff=None,
                   n_rows: int | None = None, require: bool = False) -> dict | None:
    """Check a cache's sidecar against what the caller expects; SystemExit on mismatch.

    Keys compared when present in the sidecar: `dihedral_convention`,
    `chain_break_cutoff` (compared after resolving "auto"), and the row count
    (`n_rows`, `rows` if an int, or `shape[0]`). Returns the sidecar dict, or None
    when there is none (an error if `require`).
    """
    sc = find_sidecar(path)
    if sc is None:
        if require:
            raise SystemExit(f"no sidecar next to {path}; cannot verify its convention")
        return None
    meta = json.loads(Path(sc).read_text())
    if "dihedral_convention" in meta and meta["dihedral_convention"] != convention:
        raise SystemExit(f"{path}: sidecar says dihedral_convention="
                         f"{meta['dihedral_convention']!r}, the checkpoint expects {convention!r}")
    if "chain_break_cutoff" in meta and not _same_cutoff(meta["chain_break_cutoff"],
                                                          chain_break_cutoff, convention):
        raise SystemExit(f"{path}: sidecar says chain_break_cutoff={meta['chain_break_cutoff']!r}, "
                         f"the checkpoint expects {chain_break_cutoff!r}")
    rows = meta.get("n_rows")
    if rows is None and isinstance(meta.get("rows"), int):
        rows = meta["rows"]
    if rows is None and isinstance(meta.get("shape"), list) and meta["shape"]:
        rows = meta["shape"][0]
    if n_rows is not None and rows is not None and int(rows) != int(n_rows):
        raise SystemExit(f"{path}: sidecar says {rows} rows, expected {n_rows}")
    return meta
