"""Shared file layout of the CryptoBench scripts (keeps new runs apart from the rebuttal outputs).

What it does
------------
Every script in this folder reads and writes the same four kinds of files. This
module gives them one place to resolve those locations, so a new run (for the
reported or the retrained checkpoint) never overwrites the files behind the
rebuttal numbers.

  raw benchmark   `--cryptobench-dir`  dataset.json, folds.json, cif-files/
                                       (default REPO_ROOT/data/cryptobench, input only)
  structures      `--structs-root`     <root>/cryptobench_<set>/<apo>_<chain>.npz
  manifests       `--manifest-dir`     <dir>/<set>_manifest.json
  tokens          `--tokens-dir`       <dir>/cryptobench_tokens_<tag>.npz (+ _codebook.npy,
                                       .provenance.json), cryptobench_dense_<tag>.npz
  probe outputs   `--out-dir`          mlp/ and conv/<row>__<readout>[__seed<s>].json +
                                       .preds.npz (since 2026-10-21), stats/, logs/,
                                       completeness.json; the rebuttal-era probe_*.json,
                                       preds_*.npz, diagnose_summary.json, overlap_check.json

Two layouts:

  * run layout (default), one folder per `--run-tag` (default: the name of
    `ensembits.paths.CKPT_DIR`; the run made on 2026-10-15 is `--run-tag camera_ready`):
        structures / manifests / tokens : REPO_ROOT/data/cached_descriptors/cryptobench_runs/<run_tag>/
        outputs                         : REPO_ROOT/output/cryptobench/runs/<run_tag>/
  * `--legacy-layout`: the locations the original rebuttal scripts used
        structures : REPO_ROOT/data/cached_descriptors/cryptobench_<set>/
        manifests  : REPO_ROOT/data/cryptobench/<set>_manifest.json
        tokens     : REPO_ROOT/data/cached_descriptors/cryptobench_tokens_<tag>.npz
        outputs    : REPO_ROOT/output/cryptobench/
    Use it only to READ the existing caches (for example, to probe the existing
    baseline tokens). Writing with it overwrites the rebuttal files.

Any single location can be overridden with its own flag. Relative paths given on
the command line are resolved against the current directory if they exist there,
otherwise against REPO_ROOT.

Paper
-----
Not described in the paper (infrastructure only).

Origin
------
New file written for this codebase (not copied). The original scripts
(`submission_exp/src/rebuttal_analyses/cryptobench_*.py`) hard-coded
ROOT = the research repo root and the legacy locations above.

Used by the reported model: yes, path handling of every CryptoBench script.

Inputs / outputs
----------------
None by itself; creates the output folders a script asks for.

How to run
----------
Imported by the scripts in this folder; not run directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT, CKPT_DIR  # noqa: E402  data/ and output/ stay in the research repo

# FIXED(code_fix P22, 2026-10-16): old: DEFAULT_RUN_TAG = "camera_ready" (one folder shared by
#   every checkpoint, holding the current checkpoint's run). New: the name of
#   ensembits.paths.CKPT_DIR, so a run of the retrained checkpoint (once CKPT_DIR points at
#   it, code_fix E0) never lands in the folder of another checkpoint. The existing run is
#   read with --run-tag camera_ready.
DEFAULT_RUN_TAG = CKPT_DIR.name


def repo_path(p: str | Path) -> Path:
    """Absolute path as is; relative path from the cwd if it exists there, else from REPO_ROOT."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def add_path_args(ap: argparse.ArgumentParser) -> None:
    """Add the shared location flags to a script's parser."""
    g = ap.add_argument_group("locations (see downstream/cryptobench/_cb_paths.py)")
    g.add_argument("--run-tag", default=DEFAULT_RUN_TAG,
                   help="folder name of this run under data/cached_descriptors/cryptobench_runs/ "
                        "and output/cryptobench/runs/ (default: %(default)s)")
    g.add_argument("--legacy-layout", action="store_true",
                   help="use the rebuttal locations (data/cached_descriptors/cryptobench_*, "
                        "data/cryptobench/*_manifest.json, output/cryptobench/). Read-only use: "
                        "writing overwrites the rebuttal files")
    g.add_argument("--cryptobench-dir", default=None,
                   help="raw CryptoBench download (dataset.json, folds.json, cif-files/); "
                        "default REPO_ROOT/data/cryptobench")
    g.add_argument("--structs-root", default=None,
                   help="folder holding cryptobench_<set>/ structure caches")
    g.add_argument("--manifest-dir", default=None, help="folder holding <set>_manifest.json")
    g.add_argument("--tokens-dir", default=None,
                   help="folder holding cryptobench_tokens_<tag>.npz and _codebook.npy")
    g.add_argument("--out-dir", default=None, help="folder for probe / diagnose / overlap outputs")


def resolve_paths(args: argparse.Namespace, create: tuple[str, ...] = ()) -> SimpleNamespace:
    """Resolve the locations from the parsed flags; mkdir the ones named in `create`."""
    if args.legacy_layout:
        d = dict(structs_root=ROOT / "data" / "cached_descriptors",
                 manifest_dir=ROOT / "data" / "cryptobench",
                 tokens_dir=ROOT / "data" / "cached_descriptors",
                 out_dir=ROOT / "output" / "cryptobench")
    else:
        base = ROOT / "data" / "cached_descriptors" / "cryptobench_runs" / args.run_tag
        d = dict(structs_root=base, manifest_dir=base, tokens_dir=base,
                 out_dir=ROOT / "output" / "cryptobench" / "runs" / args.run_tag)
    for k in list(d):
        v = getattr(args, k, None)
        if v:
            d[k] = repo_path(v)
    d["cryptobench_dir"] = (repo_path(args.cryptobench_dir) if args.cryptobench_dir
                            else ROOT / "data" / "cryptobench")
    for k in create:
        d[k].mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(**d)


def manifest_path(paths: SimpleNamespace, set_name: str) -> Path:
    return paths.manifest_dir / f"{set_name}_manifest.json"


def structs_dir(paths: SimpleNamespace, set_name: str) -> Path:
    return paths.structs_root / f"cryptobench_{set_name}"

