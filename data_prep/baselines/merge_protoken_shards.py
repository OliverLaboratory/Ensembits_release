"""Merge the ProToken token shards into one npz per dataset.

What it does
------------
`build_protoken_tokens.py --shard k --num-shards n` writes
`protoken_shards/{misato,mdcath}_shard<k>of<n>.npz`. This script reads every
shard of one dataset (sorted by file name), puts all pid -> tokens entries in
one dict (a later shard overwrites an earlier one for the same pid) and writes
`protoken_tokens_<dataset>.npz`, the file the probes read.

Since 2026-10-18 every shard must record chain_split = rmsf_common.chain_split_for(dataset)
("true_chain_v2" for MISATO, made from the current misato_chain_breaks.json, option A;
the old 2.5 Å split is refused) and all shards must agree on it and on the chain-break table.

`--pooled` (added 2026-10-16, code_fix P15) does the same for the shards of
`build_protoken_pooled.py --shard k --num-shards n`
(`protoken_pooled_misato_shard<k>of<n>.npz` in `data/cached_descriptors/`) and
writes `protoken_pooled_misato.npz` (compressed, as the unsharded builder).
Since 2026-10-16 the merge exits 1 when an output exists (unless `--overwrite`)
or when not all n shards are present.

Paper
-----
not described in the paper (infrastructure only; ProToken baseline,
`appendix.tex` \\label{app:baselines}).

Origin
------
`merge_shards` is copied unchanged from `submission_exp/src/_run_protoken_recheck.py`
(the rest of that orchestrator, which backed up old results and re-ran every
task, is not copied). Added: this docstring, the `ROOT` / path setup from
`ensembits.paths.REPO_ROOT`, and a small CLI (`--datasets`, `--num-shards`; the
original used the module constant `NSH = 8`, still the default). 2026-10-16:
`--pooled`, `--overwrite`, shard-count check (code_fix P15). 2026-10-16
(preretrain_audit_r2 F1 / F7): the merged token cache gets a provenance sidecar built
from the shard sidecars; failures of RMSF-set proteins in any shard fail the merge
(`--max-fail-frac`).

Used by the reported model: no (ProToken baseline rows).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/protoken_shards/<dataset>_shard*of<n>.npz
          data/cached_descriptors/protoken_pooled_misato_shard*of<n>.npz      (--pooled)
  writes  data/cached_descriptors/protoken_tokens_<dataset>.npz
          data/cached_descriptors/protoken_pooled_misato.npz                  (--pooled)

How to run
----------
    python data_prep/baselines/merge_protoken_shards.py [--datasets misato,mdcath] [--num-shards 8]
    python data_prep/baselines/merge_protoken_shards.py --pooled [--num-shards 8]
    [--overwrite]
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
sys.path.insert(0, str(CODE_ROOT / "downstream"))   # rmsf_common (sidecars, tolerances)
CACHE = ROOT/"data/cached_descriptors"; SHARD_DIR = CACHE/"protoken_shards"
NSH = 8   # number of shards (set from --num-shards in main)


def merge_shards(prefix, out, shard_dir=None, compressed=False):
    # FIXED(code_fix P15, 2026-10-16): `shard_dir` / `compressed` added for the pooled
    #   shards (defaults = old behaviour); all NSH shards must be present (old: whatever
    #   matched the glob was merged, so a missing shard gave a silently partial file).
    shard_dir = SHARD_DIR if shard_dir is None else shard_dir
    files = sorted(glob.glob(str(shard_dir/f"{prefix}_shard*of{NSH}.npz")))
    if len(files) != NSH:
        print(f"[merge] {prefix}: found {len(files)} of {NSH} shards in {shard_dir}; "
              f"not merging", flush=True)
        raise SystemExit(1)
    feats={}
    for f in files:
        d=np.load(f)
        for k in d.files: feats[k]=d[k]
    (np.savez_compressed if compressed else np.savez)(out, **feats)
    print(f"[merge] {prefix}: {len(feats)} -> {out.name}",flush=True)
    return len(feats)


def merge_sidecars(prefix, out, dataset, max_fail_frac=0.0, shard_dir=None):
    """Merged `<out>.provenance.json` from the shard sidecars (F1 / F7, 2026-10-16).

    FIXED(preretrain_audit_r2 H3, 2026-10-16): every shard must carry a sidecar of the
    current schema; all shards must agree on dataset, inputs (MISATO parser / K / fps),
    frames, chain split, sequence source and codebook sha256; each shard's npz must still
    have the sha256 its sidecar records; no pid may occur in two shards. The failures of
    all shards are checked against the whole RMSF residue set. On any refusal the merged
    cache is removed and the merge exits 1.
    """
    from ensembits.data_versions import read_provenance
    from ensembits.featurize import file_sha256
    from rmsf_common import BASELINE_CACHE_SCHEMA, check_builder_failures, write_cache_provenance
    shard_dir = SHARD_DIR if shard_dir is None else shard_dir
    files = sorted(glob.glob(str(shard_dir/f"{prefix}_shard*of{NSH}.npz")))
    sides = [read_provenance(f) for f in files]

    def refuse(msg):
        print(f"[merge] {prefix}: {msg}; merged cache removed", flush=True)
        if out.exists(): out.unlink()
        raise SystemExit(1)

    if any(sd is None for sd in sides):
        refuse("a shard has no provenance sidecar (rebuild it)")
    for f, sd in zip(files, sides):
        if sd.get("schema") != BASELINE_CACHE_SCHEMA:
            refuse(f"{Path(f).name}: sidecar schema {sd.get('schema')!r}")
        if file_sha256(f) != sd.get("cache_sha256"):
            refuse(f"{Path(f).name} changed after its sidecar was written")
    # 2026-10-18: chain_split replaces chain_break_A; it must be the dataset's value.
    for key in ("dataset", "inputs", "frames", "chain_split", "chain_breaks", "sequence_source",
                "codebook_sha256"):
        vals = {json.dumps(sd.get(key), sort_keys=True) for sd in sides}
        if len(vals) != 1:
            refuse(f"shards disagree on {key}: {sorted(vals)[:3]}")
    seen = {}
    for f in files:
        for k in np.load(f).files:
            if k in seen:
                refuse(f"pid {k} in both {Path(seen[k]).name} and {Path(f).name}")
            seen[k] = f
    failed, attempted = {}, []
    for sd in sides:
        failed.update(sd.get("failed") or {}); attempted += sd.get("attempted") or []
    if not check_builder_failures(failed, attempted, dataset, max_fail_frac):
        refuse("failures inside the RMSF residue set")
    first = sides[0]
    from rmsf_common import split_record
    want = split_record(dataset)
    if first.get("chain_split") != want["chain_split"] or \
            (first.get("chain_breaks") or {}).get("sha256") != (want["chain_breaks"] or {}).get("sha256"):
        have_sha = (first.get("chain_breaks") or {}).get("sha256")
        want_sha = (want["chain_breaks"] or {}).get("sha256")
        refuse(f"the shards record chain_split {first.get('chain_split')!r} with chain-break "
               f"table sha256 {str(have_sha)[:12]}, expected {want['chain_split']!r} with "
               f"{str(want_sha)[:12]} (built before 2026-10-18 or from another "
               f"misato_chain_breaks.json; rebuild them)")
    if first.get("codebook_sha256") != file_sha256(CACHE/"protoken_codebook.npy"):
        refuse("the shards were made with another codebook than protoken_codebook.npy")
    write_cache_provenance(out, dataset=dataset, builder=first.get("builder"),
                           inputs=first.get("inputs"), n_proteins=len(seen),
                           frames=first.get("frames"), chain_split=first.get("chain_split"),
                           chain_breaks=first.get("chain_breaks"),
                           sequence_source=first.get("sequence_source"),
                           sequence_given=sorted({str(sd.get("sequence_given")) for sd in sides}),
                           # G7 (2026-10-17): ideal-geometry frames of all shards
                           n_ideal_frames=int(sum(int(sd.get("n_ideal_frames") or 0) for sd in sides)),
                           ideal_pids=sorted({p for sd in sides for p in (sd.get("ideal_pids") or [])}),
                           failed=failed, codebook=CACHE/"protoken_codebook.npy",
                           merged_from=[Path(f).name for f in files])


def main(argv=None):
    global NSH
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="misato,mdcath")
    ap.add_argument("--num-shards", type=int, default=NSH)
    # FIXED(code_fix P15, 2026-10-16): merge of the build_protoken_pooled.py shards (old:
    #   no merge existed) and refusal of an existing output (old: overwritten).
    ap.add_argument("--pooled", action="store_true",
                    help="merge protoken_pooled_misato_shard*of<n>.npz into "
                         "protoken_pooled_misato.npz instead of the token shards")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing merged output (default: exit 1 if it exists)")
    ap.add_argument("--max-fail-frac", type=float, default=0.0,
                    help="allowed fraction of RMSF-set proteins that failed in the shards")
    a = ap.parse_args(argv)
    NSH = a.num_shards
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs
    if a.pooled:
        out = CACHE/"protoken_pooled_misato.npz"
        refuse_existing_outputs([out], a.overwrite)
        # FIXED(baseline fidelity audit, 2026-10-18): every pooled shard must record the
        #   current MISATO chain split (true_chain_v2 + chain-break table sha256, option A);
        #   the merged file gets a sidecar.
        from ensembits.data_versions import read_provenance, write_provenance
        from rmsf_common import split_record
        want = split_record("misato")
        files = sorted(glob.glob(str(CACHE/f"protoken_pooled_misato_shard*of{NSH}.npz")))
        sides = {Path(f).name: (read_provenance(f) or {}) for f in files}
        bad = [n_ for n_, sd in sides.items()
               if sd.get("chain_split") != want["chain_split"]
               or (sd.get("chain_breaks") or {}).get("sha256") != want["chain_breaks"]["sha256"]]
        if bad:
            print(f"[merge] protoken_pooled_misato: shards without chain_split "
                  f"{want['chain_split']!r} of the current chain-break table (sha256 "
                  f"{want['chain_breaks']['sha256'][:12]}): {bad[:3]} (built before "
                  f"2026-10-18 or from another table; rebuild them)", flush=True)
            raise SystemExit(1)
        n = merge_shards("protoken_pooled_misato", out, shard_dir=CACHE, compressed=True)
        write_provenance(out, {"builder": "merge_protoken_shards.py --pooled", "baseline": "protoken",
                               "dataset": "misato", **want, "n_proteins": n,
                               "merged_from": [Path(f).name for f in files]})
        return 0
    outs = {ds: CACHE/f"protoken_tokens_{ds}.npz" for ds in a.datasets.split(",")}
    refuse_existing_outputs(list(outs.values()), a.overwrite)
    # Same calls as _run_protoken_recheck.py phase 1.
    for ds in a.datasets.split(","):
        merge_shards(ds, outs[ds])
        merge_sidecars(ds, outs[ds], ds, a.max_fail_frac)   # F1 / F7 (2026-10-16)
    return 0


if __name__ == "__main__":
    sys.exit(main())
