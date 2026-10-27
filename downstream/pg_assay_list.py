"""Build and check the ProteinGym assay subset (96 assays, WT <= 200) and write its disclosure.

What it does
------------
1. Reads the first variant row of every ProteinGym v1.2 substitution CSV in
   PROTEINGYM_ZERO_SHOT_DIR and derives the wild-type sequence from it.
2. Selects the assays whose WT length is <= --max-len (200), sorted by id.
3. Checks the selection against the fixed list `downstream/pg_assays_wt_le200.txt`
   (refuses a difference unless --write-list, which rewrites the list atomically)
   and against the ESMFold atom14 cache folders (one folder per selected assay,
   manifest WT == CSV WT).
4. For each selected assay reads every row (float64) and records: UniProt entry,
   Tsuboyama flag, WT length, number of variants, multi-mutant count and fraction,
   max mutations per variant, rows whose sequence disagrees with the mutant string,
   NaN DMS / ESM2 values, variants without a cached structure.
5. Writes the disclosure JSON + markdown (counts the paper must state: 96 assays,
   64 Tsuboyama stability assays, number of proteins, variants, % multi-mutants) with
   the assay list sha256 and the CSV set sha256.

No tokenizer and no score is involved.

Paper
-----
`2_results.tex`, \\paragraph{Mutation Effect Prediction} ("restricted to the 96
deep-mutational-scanning assays whose wild-type sequence length is at most 200");
benchmark_audit D9 / preretrain_audit_r2 Q32: disclose the subset (64/96 Tsuboyama
stability assays, 70% multi-mutant variants).

Origin
------
New (2026-10-19, benchmark_audit D9). The WT rule (WT length <= 200 over all
substitution CSVs) reproduces the research repo's ESMFold cache selection
(`submission_exp/src/pg_esmfold_cache.py`, audit check bench_pg/consist.py:
cache folders == CSVs with WT <= 200).

Used by the reported model: yes (fixes the assay set of every ProteinGym row).

Inputs / outputs
----------------
Reads  data/external/proteingym_zero_shot_scores/*.csv          (PROTEINGYM_ZERO_SHOT_DIR)
       data/proteingym/esmfold_cache_atom14/<assay>/manifest.json + *.npy names
Writes camera_ready/ensembits_code/downstream/pg_assays_wt_le200.txt  (only with --write-list)
       output/proteingym/pg_assay_subset.json                    disclosure + shas
       output/proteingym/pg_assay_subset.md                      the same as a table

How to run
----------
    python downstream/pg_assay_list.py            # check the list, write the disclosure
    python downstream/pg_assay_list.py --write-list   # (re)create the list file
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT                       # noqa: E402
from ensembits.paths import PROTEINGYM_ZERO_SHOT_DIR as PG_DIR      # noqa: E402
import pg_common as C                                               # noqa: E402

CACHE_DIR = ROOT / "data" / "proteingym" / "esmfold_cache_atom14"
OUT_DIR = ROOT / "output" / "proteingym"


def first_row_wt(csv_path: Path) -> str | None:
    """WT sequence implied by the first row of a substitution CSV (None if empty)."""
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            return C.wt_from_row(r["mutant"], r["mutated_sequence"])
    return None


def select_assays(pg_dir: Path, max_len: int) -> tuple[list[str], dict]:
    """(sorted ids with WT length <= max_len, {assay: wt} of the selected ones)."""
    wts = {}
    for p in sorted(Path(pg_dir).glob("*.csv")):
        wt = first_row_wt(p)
        if wt is not None and len(wt) <= max_len:
            wts[p.stem] = wt
    return sorted(wts), wts


def assay_record(assay: str, wt: str, pg_dir: Path, cache_dir: Path) -> dict:
    rows, st = C.load_assay_rows(Path(pg_dir) / f"{assay}.csv", wt)
    nm = np.array([r["n_mut"] for r in rows], dtype=np.int64)
    cdir = Path(cache_dir) / assay
    have = {p.stem for p in cdir.glob("*.npy")} if cdir.is_dir() else set()
    n_no_struct = sum(C.hash_seq(r["mut_seq"]) not in have for r in rows)
    man_wt = None
    if (cdir / "manifest.json").exists():
        man_wt = json.loads((cdir / "manifest.json").read_text()).get("wt")
    return {"assay": assay, "uniprot": C.uniprot_of(assay),
            "tsuboyama": C.is_tsuboyama(assay), "wt_len": len(wt),
            "n_variants": len(rows), "n_multi": int((nm > 1).sum()),
            "frac_multi": float((nm > 1).mean()) if len(nm) else 0.0,
            "max_n_mut": int(nm.max()) if len(nm) else 0,
            "n_inconsistent": st["n_inconsistent"], "n_dup_mutant": st["n_dup_mutant"],
            "n_len_mismatch": st["n_len_mismatch"],
            "n_nan_dms": st["n_nan_dms"], "n_nan_esm2": st["n_nan_esm2"],
            "n_no_structure": int(n_no_struct),
            "cache_folder": cdir.is_dir(), "manifest_wt_matches": man_wt == wt}


def summarize(recs: list[dict]) -> dict:
    nv = sum(r["n_variants"] for r in recs)
    nmulti = sum(r["n_multi"] for r in recs)
    return {"n_assays": len(recs),
            "n_tsuboyama": sum(r["tsuboyama"] for r in recs),
            "n_uniprot": len({r["uniprot"] for r in recs}),
            "n_uniprot_tsuboyama": len({r["uniprot"] for r in recs if r["tsuboyama"]}),
            "n_variants": nv, "n_multi": nmulti,
            "frac_multi": nmulti / nv if nv else 0.0,
            "n_assays_with_multi": sum(r["n_multi"] > 0 for r in recs),
            "n_inconsistent": sum(r["n_inconsistent"] for r in recs),
            "n_dup_mutant": sum(r["n_dup_mutant"] for r in recs),
            "n_nan_dms": sum(r["n_nan_dms"] for r in recs),
            "n_nan_esm2": sum(r["n_nan_esm2"] for r in recs),
            "n_no_structure": sum(r["n_no_structure"] for r in recs),
            "n_missing_cache_folder": sum(not r["cache_folder"] for r in recs),
            "n_manifest_wt_mismatch": sum(not r["manifest_wt_matches"] for r in recs)}


def to_markdown(summary: dict, recs: list[dict], meta: dict) -> str:
    s = summary
    out = ["# ProteinGym assay subset (benchmark_audit D9)", "",
           f"{s['n_assays']} ProteinGym v1.2 substitution assays with WT length <= "
           f"{meta['max_len']} (of {meta['n_csv']} CSVs); {s['n_tsuboyama']} are Tsuboyama 2023 "
           f"stability assays; {s['n_uniprot']} distinct UniProt entries; "
           f"{s['n_variants']:,} variants, {s['n_multi']:,} multi-mutants "
           f"({100 * s['frac_multi']:.1f}%) in {s['n_assays_with_multi']} assays.", "",
           f"assay list sha256 `{meta['assay_list_sha256']}`; CSV set sha256 "
           f"`{meta['csv_set_sha256']}`.", "",
           "| assay | UniProt | Tsuboyama | WT len | variants | multi % | max n_mut |",
           "|---|---|---|---|---|---|---|"]
    for r in recs:
        out.append(f"| {r['assay']} | {r['uniprot']} | {'yes' if r['tsuboyama'] else ''} | "
                   f"{r['wt_len']} | {r['n_variants']} | {100 * r['frac_multi']:.0f} | "
                   f"{r['max_n_mut']} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--list", default=str(C.ASSAY_LIST), help="the fixed assay list file")
    ap.add_argument("--write-list", action="store_true",
                    help="(re)write the list file from the selection")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    n_csv = len(list(Path(PG_DIR).glob("*.csv")))
    assays, wts = select_assays(PG_DIR, args.max_len)
    print(f"{len(assays)} of {n_csv} CSVs have WT length <= {args.max_len}", flush=True)
    lst = Path(args.list)
    if args.write_list:
        hdr = (f"# ProteinGym v1.2 substitution assays with WT length <= {args.max_len} "
               f"({len(assays)} assays); written by downstream/pg_assay_list.py\n")
        C.atomic_write_text(lst, hdr + "\n".join(assays) + "\n")
        print(f"wrote {lst}")
    elif not lst.exists() or C.read_assay_list(lst) != assays:
        print(f"[abort] {lst} differs from the selection (run with --write-list)", file=sys.stderr)
        return 1

    cached = {d.name for d in CACHE_DIR.iterdir() if d.is_dir()} if CACHE_DIR.is_dir() else set()
    if cached != set(assays):
        print(f"[warn] cache folders != selection: {len(cached - set(assays))} extra, "
              f"{len(set(assays) - cached)} missing", flush=True)

    recs = [assay_record(a, wts[a], PG_DIR, CACHE_DIR) for a in assays]
    summary = summarize(recs)
    meta = {"max_len": args.max_len, "n_csv": n_csv,
            "assay_list": str(lst.name), "assay_list_sha256": C.assay_list_sha256(assays),
            "csv_set_sha256": C.structure_set_sha256(
                {a: C.file_sha256(Path(PG_DIR) / f"{a}.csv") for a in assays}),
            "cache_dir": str(CACHE_DIR.relative_to(ROOT)) if CACHE_DIR.is_relative_to(ROOT)
                         else str(CACHE_DIR),
            "cache_folders_equal_selection": cached == set(assays)}
    out = Path(args.out_dir)
    C.atomic_write_json(out / "pg_assay_subset.json",
                        {"meta": meta, "summary": summary, "assays": recs})
    C.atomic_write_text(out / "pg_assay_subset.md", to_markdown(summary, recs, meta))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out / 'pg_assay_subset.json'} and .md")
    bad = summary["n_inconsistent"] + summary["n_dup_mutant"] + summary["n_manifest_wt_mismatch"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
