"""CryptoBench: 30 %-identity sequence clusters of the test entries (cluster map of the statistics).

What it does
------------
Builds the cluster map that the pre-registered statistics bundle (`cryptobench_families_v1.json`,
key `clusters`) uses for the paired cluster bootstrap: {test apo id: cluster representative}.
Moved into the released code on 2026-10-22 (REAUDIT CB2 F5); before, it existed only as an
audit scratch script.

  1. FASTA of every built apo chain (`<apo>_<chain>`, sequence of the structure cache), as
     `camera_ready/cryptobench_audit/leak_D/build_fasta.py` (`cb_apo.fa`).
  2. All-against-all search: `mmseqs easy-search cb_apo.fa cb_apo.fa apo_vs_apo.m8 tmp -s 7.5
     -e 1e-3 --max-seqs 2000 --format-output query,target,fident,alnlen,qlen,tlen,qcov,tcov,
     evalue,bits` (leak_D/log3.txt).
  3. Single linkage over the TEST apo ids: two entries are joined when a pair of their chains
     (both test) has fident >= 0.30 and max(qcov, tcov) >= 0.5; union-find as
     `leak_D/pdbid.py`, but in sorted order (the audit script iterated Python sets of strings,
     so its representative LABELS depend on the string hash seed; the partition does not).

The pre-registered file `camera_ready/cryptobench_audit/leak_D/test_clusters_30.json` (sha256
9d1f1f84...44cc; 222 entries, 201 clusters) was built from the 2026-10-15 structure cache's
sequences; `clusters_from_m8` reproduces its PARTITION exactly from `leak_D/apo_vs_apo.m8`
(same 201 clusters; representative names may differ, which does not change the bootstrap; test).
The bundle keeps that file (pre-registered); this script documents and reproduces the rule.

Paper
-----
Appendix (significance): "201 sequence clusters (>= 30 % identity, >= 50 % coverage)".

Origin
------
New file (2026-10-22); logic of `camera_ready/cryptobench_audit/leak_D/{build_fasta.py,
pdbid.py}` (audit D, 2026-10-20).

Used by the reported model: evaluation only.

Inputs / outputs
----------------
Read  : `<manifest-dir>/<set>_manifest.json`, `<structs-root>/cryptobench_<set>/*.npz` (seq),
        or an existing hit file (`--m8`) with a chain -> fold map (`--meta`).
Write : `--out` JSON {apo: representative}; with `--workdir` the FASTA and the hit file.

How to run
----------
  $PY downstream/cryptobench/cryptobench_test_clusters.py --run-tag <tag> --workdir <dir> --out <json>
  $PY downstream/cryptobench/cryptobench_test_clusters.py \\
      --m8 camera_ready/cryptobench_audit/leak_D/apo_vs_apo.m8 \\
      --meta camera_ready/cryptobench_audit/leak_D/cb_apo_meta.json --out <json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
import _cb_paths as CBP  # noqa: E402

MIN_FIDENT = 0.30
MIN_COV = 0.5
MMSEQS_ARGS = ["-s", "7.5", "-e", "1e-3", "--max-seqs", "2000", "--format-output",
               "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"]


def write_fasta(manifest: dict, set_dir, out) -> dict:
    """FASTA of every built chain; returns {chain key: fold}."""
    folds = {}
    with open(out, "w") as fa:
        for e in manifest["entries"]:
            key = f"{e['apo']}_{e['chain']}"
            seq = str(np.load(Path(set_dir) / f"{key}.npz", allow_pickle=True)["seq"])
            fa.write(f">{key}\n{seq}\n")
            folds[key] = e["fold"]
    return folds


def clusters_from_m8(m8_lines, fold_of: dict, test_ids) -> dict:
    """{test apo id: representative} by single linkage (leak_D/pdbid.py rule, sorted order)."""
    adj = defaultdict(set)
    for line in m8_lines:
        a = line.rstrip("\n").split("\t")
        if len(a) < 8:
            continue
        qa, ta = a[0], a[1]
        if (fold_of.get(qa) == "test" and fold_of.get(ta) == "test"
                and float(a[2]) >= MIN_FIDENT and max(float(a[6]), float(a[7])) >= MIN_COV):
            adj[qa.split("_")[0]].add(ta.split("_")[0])
    par = {a: a for a in test_ids}

    def fd(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a in sorted(adj):                  # sorted: deterministic representatives
        for b in sorted(adj[a]):
            par[fd(a)] = fd(b)
    return {a: fd(a) for a in test_ids}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="apo")
    ap.add_argument("--m8", default=None, help="existing all-against-all hit file")
    ap.add_argument("--meta", default=None, help="with --m8: JSON {chain key: {fold}}")
    ap.add_argument("--workdir", default=None, help="folder for the FASTA / hits / mmseqs tmp")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    CBP.add_path_args(ap)
    args = ap.parse_args(argv)
    paths = CBP.resolve_paths(args)
    test_ids = json.loads((paths.cryptobench_dir / "folds.json").read_text())["test"]
    if args.m8:
        meta = json.loads(Path(args.meta).read_text())
        fold_of = {k: (v["fold"] if isinstance(v, dict) else v) for k, v in meta.items()}
        lines = open(args.m8).readlines()
    else:
        from ensembits.paths import MMSEQS_BIN
        wd = Path(args.workdir); wd.mkdir(parents=True, exist_ok=True)
        man = json.loads(CBP.manifest_path(paths, args.set).read_text())
        fold_of = write_fasta(man, CBP.structs_dir(paths, args.set), wd / "cb_apo.fa")
        subprocess.run([MMSEQS_BIN, "easy-search", str(wd / "cb_apo.fa"), str(wd / "cb_apo.fa"),
                        str(wd / "apo_vs_apo.m8"), str(wd / "tmp"), *MMSEQS_ARGS,
                        "--threads", str(args.threads)], check=True,
                       stdout=subprocess.DEVNULL)
        lines = open(wd / "apo_vs_apo.m8").readlines()
    cl = clusters_from_m8(lines, fold_of, test_ids)
    Path(args.out).write_text(json.dumps(cl))
    print(f"[clusters] {len(cl)} test entries -> {len(set(cl.values()))} clusters; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
