"""Build the training corpus of the reported model: BioEmu-CATH + MISATO 256-D descriptors and split.

What it does
------------
  1. Lists the systems of both sources (both are real-backbone caches with
     bb_10 = (10, L, 4, 3) N/CA/C/O):
       - BioEmu-CATH : --bioemu-bb, one file per simulated system (built by
         `build_bioemu_real_bb.py`), minus every domain in the mdCATH-div
         evaluation hold-out file;
       - MISATO      : --misato-bb, bb_10 with Kabsch FPS and the fixed atom
         parser (`build_misato_real_bb.py --K 10 --fps kabsch`), no hold-out.
  2. Computes the 256-D descriptor of every frame of every system (the same
     `_descriptor` as everywhere else, so tokens are identical) and writes one
     flat (N_residues, 10, 256) float32 array plus a layout of
     {pid, corpus, L, start} rows.
  3. Splits at random 80/10/10 (seed 42) into train/val/test and writes the
     residue row indices of each split. The unit of the split is the domain
     (BioEmu-CATH; both simulations of a domain go to the same split) or the
     complex (MISATO); `--split-unit system` reproduces the old split by row.

Corpus of the current checkpoint (old settings): 1,072 BioEmu-CATH rows (1,057
distinct domains, 15 duplicated) + 16,972 MISATO complexes = 18,044 systems;
split 14,435 / 1,804 / 1,805 systems, 5,670,326 / 718,011 / 726,643 residues.
The retrain (code_fix E0) uses the defaults below, which change these counts.

To rebuild the current checkpoint's corpus:
    --bioemu-bb data/cached_descriptors/reviewer_3_special_corpus_real_bb
    --misato-bb data/cached_descriptors/misato_real_bb_legacy_proline
    --split-unit system --dihedral-convention legacy_plus180

Paper
-----
  * `appendix.tex`, \\subsection{Dataset} (\\label{app:dataset}):
    \\paragraph{BioEmu-CATH.} (17 mdCATH-div test domains removed, 1,072
    ensembles of 1,057 domains), \\paragraph{MISATO.}, \\paragraph{Combined
    training corpus.} (18,044 ensembles, random 80/10/10 by system,
    14,435 / 1,804 / 1,805 systems, 5.67M / 0.72M / 0.73M residues),
    \\paragraph{mdCATH-div (evaluation).} (the 675 test domains excluded).
  * `2_results.tex`, \\subsection{Datasets}: combined corpus, 80/10/10 split,
    evaluation domains removed from BioEmu-CATH.
  * `appendix.tex`, \\label{app:descriptor-esm3}: the 256-D descriptor.

Origin
------
Copied from `submission_exp/src/build_bioemu_misato_descriptor.py` (built for
the `reviewer_3_special_corpus_MISATO` run, which became the reported model).
Changed: `ROOT` now comes from `ensembits.paths.REPO_ROOT` (same directory as
before); `from build_esm3desc_dihedral import _descriptor` became
`from ensembits.descriptor import _descriptor` (same function); comments and
this docstring. 2026-10-15 fixes: dihedral convention (§1), system-keyed
BioEmu files and split by domain (§5 / T1.5), Kabsch
MISATO cache from the fixed parser (§4 / T1.1, T1.6); all recorded in
layout.json. 2026-10-16 (code_fix P26): every dropped system is listed with its
reason in layout.json ("dropped"); the build aborts above 0.5% dropped systems,
on any MISATO file without bb_10, and on any MISATO file of another parser
version (checked on every file it reads, not a sample).

Used by the reported model: yes. Its output is the reported model's training,
validation and test data.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   <--bioemu-bb>/layout_rows.json, <system>.npz   (default data/cached_descriptors/bioemu_cath_real_bb)
          <--misato-bb>/<pid>.npz                      (default data/cached_descriptors/misato_real_bb_kabsch)
          data/reviewer_3_special_corpus/mdcath_eval_holdout_pids.json
  writes  <--out>/esm3desc_dihedral[_iupac]_K16_P10.npy   (N_residues, 10, 256) float32
          <--out>/layout.json   {"rows": [{system, pid, corpus, L, start}, ...],
                                 "dropped": [{system, pid, corpus, reason}, ...], settings}
          <--out>/splits.json   {"train"|"val"|"test": [residue row indices]}
  The current checkpoint used --out output/reviewer_3_special_corpus_MISATO/cached_desc.

How to run
----------
    python data_prep/build_bioemu_misato_descriptor.py \\
        --out output/bioemu_cath_misato_e0/cached_desc --device cuda:0
(run from REPO_ROOT, or pass an absolute --out)
"""
from __future__ import annotations
import os
# Limit CPU threads used by numpy/torch.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # data/ and output/ stay in the research repo
# FIXED(code_fix §1, 2026-10-15): `_descriptor` takes the dihedral convention
# (--dihedral-convention, default iupac). The cache used by the current checkpoint
# was written with legacy_plus180 (φ/ψ + 180°).
from ensembits.descriptor import _descriptor

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in data_prep/
from ensembits.data_versions import MISATO_PARSER_VERSION, require_misato_parser  # noqa: E402
from ensembits.descriptor import resolve_chain_break_cutoff  # noqa: E402

# Real-backbone caches of the two sources and the mdCATH-div hold-out list.
# FIXED(code_fix §5 / T1.5, 2026-10-15): system-keyed BioEmu-CATH files (the old
#   cache reviewer_3_special_corpus_real_bb was keyed by domain, with duplicates).
BIOEMU_BB = ROOT / "data" / "cached_descriptors" / "bioemu_cath_real_bb"
# FIXED(code_fix §4 / T1.1 and T1.6, 2026-10-15): MISATO frames by Kabsch FPS from
#   the fixed atom parser (the old cache misato_real_bb used raw-coordinate FPS and
#   had proline Cδ in the CA slot).
MISATO_BB = ROOT / "data" / "cached_descriptors" / "misato_real_bb_kabsch"
# 675 pids = union of the mdCATH-div sequence / structure / random test sets.
# Paper: appendix.tex \paragraph{mdCATH-div (evaluation).} — "The 675 domains in
# any mdCATH-div test set ... are excluded from BioEmu-CATH training".
HOLDOUT = ROOT / "data" / "reviewer_3_special_corpus" / "mdcath_eval_holdout_pids.json"
# FIXED(code_fix P26, 2026-10-16): largest tolerated fraction of dropped systems
#   (old: no limit, drops only counted and the first 19 printed).
MAX_DROP_FRAC = 0.005


def _sys_list(bioemu_bb: Path, misato_bb: Path, check_parser: bool):
    """List (system, pid, corpus, npz path) for every system: BioEmu-CATH, then MISATO.

    `pid` is the split unit: the CATH domain for BioEmu-CATH, the PDB id for MISATO.
    """
    # Paper: appendix.tex \paragraph{BioEmu-CATH.} — remove the domains that
    # appear in an mdCATH-div test set.
    if not HOLDOUT.exists():
        raise SystemExit(f"hold-out list missing: {HOLDOUT}")
    hold = set(json.loads(HOLDOUT.read_text()))
    rows = json.loads((bioemu_bb / "layout_rows.json").read_text())
    # Old layout files have no "system" key (file = <pid>.npz).
    bio = [(r.get("system", r["pid"]), r["pid"], "bioemu_cath",
            bioemu_bb / f"{r.get('system', r['pid'])}.npz")
           for r in rows if r["pid"] not in hold]
    # FIXED(preretrain_audit_r2 Q15, 2026-10-16): old = an empty BioEmu list (wrong
    #   --bioemu-bb, empty layout_rows.json) gave a MISATO-only corpus without error.
    if not bio:
        raise SystemExit(f"no BioEmu-CATH systems in {bioemu_bb / 'layout_rows.json'} "
                         f"({len(rows)} rows before the hold-out)")
    # MISATO: every complex in the cache, sorted by pid; no hold-out
    # (authors' decision: the tokenizer never sees downstream labels).
    mis = [(Path(f).stem, Path(f).stem, "misato", Path(f))
           for f in sorted(glob.glob(str(misato_bb / "*.npz")))]
    # Refuse caches written by the old MISATO parser (proline CA wrong, code_fix §4).
    require_misato_parser(misato_bb, allow_old=not check_parser)
    if not mis:
        raise SystemExit(f"no MISATO files in {misato_bb}")
    n_dom = len({p for _, p, _, _ in bio})
    print(f"[build] bioemu={len(bio)} systems / {n_dom} domains (holdout-filtered)  "
          f"misato={len(mis)}", flush=True)
    return bio + mis


def assign_splits(layout: list[dict], split_unit: str = "domain", seed: int = 42):
    """Random 80/10/10 split of layout rows; returns ({row: split}, n_units).

    split_unit "domain": rows with the same (corpus, pid) share a split (both
    simulations of a BioEmu-CATH domain); "system": every row on its own (the
    split of the current checkpoint).
    """
    if split_unit == "domain":
        units = sorted({(r["corpus"], r["pid"]) for r in layout})
        uidx = {u: k for k, u in enumerate(units)}
        unit_of = [uidx[(r["corpus"], r["pid"])] for r in layout]
    else:
        unit_of = list(range(len(layout)))
    rng = np.random.RandomState(seed)
    n = max(unit_of) + 1; perm = rng.permutation(n)
    n_tr = int(0.8 * n); n_va = int(0.1 * n)
    unit_split = {j: ("train" if k < n_tr else "val" if k < n_tr + n_va else "test")
                  for k, j in enumerate(perm)}
    return {j: unit_split[u] for j, u in enumerate(unit_of)}, n


def misato_upstream_failures(misato_bb: Path, present: set[str]) -> list[dict]:
    """Complexes the MISATO backbone build failed on, from its _status.json.

    FIXED(preretrain_audit_r2 Q13, 2026-10-16): old = MISATO systems came from the
    files present only, so complexes the backbone build failed on were invisible to the
    drop guard. New = build_misato_real_bb.py writes <dir>/_status.json (status per
    pid); every pid without a file counts as dropped. 'no_residues' / 'no_pid' (no
    protein to parse) are listed but not counted by the guard.
    """
    sf = misato_bb / "_status.json"
    if not sf.exists():
        raise SystemExit(f"{sf} missing: rebuild {misato_bb} with build_misato_real_bb.py "
                         f"(it records every complex's status)")
    st = json.loads(sf.read_text())["status"]
    return [{"system": pid, "pid": pid, "corpus": "misato", "reason": f"upstream: {v}",
             "counted": v not in ("no_residues", "no_pid")}
            for pid, v in sorted(st.items()) if pid not in present]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    # code_fix §1: dihedral sign convention of the cache. "iupac" (default) for new
    # training runs; "legacy_plus180" reproduces the cache of the reported checkpoint.
    ap.add_argument("--dihedral-convention", default="iupac",
                    choices=("iupac", "legacy_plus180"))
    # code_fix §6: phi/psi across chain breaks (C–N > cutoff) set to (0, 0); "auto" =
    # 2.5 Å for iupac, none for legacy_plus180. Both sources have real backbones.
    ap.add_argument("--chain-break-cutoff", default="auto")
    ap.add_argument("--bioemu-bb", default=str(BIOEMU_BB))
    ap.add_argument("--misato-bb", default=str(MISATO_BB))
    # code_fix §5 / T1.5: split unit. "domain" keeps both simulations of a BioEmu
    # domain in one split; "system" is the old split by layout row.
    ap.add_argument("--split-unit", default="domain", choices=("domain", "system"))
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing corpus in --out")
    args = ap.parse_args()
    cutoff = resolve_chain_break_cutoff(args.chain_break_cutoff, args.dihedral_convention)
    abspath = lambda x: Path(x) if Path(x).is_absolute() else ROOT / x
    # FIXED(preretrain_audit_r2 Q15, 2026-10-16): old = --out relative to the working
    #   directory (a run from the code folder wrote ~70 GB into it) and an existing
    #   corpus was overwritten. New = relative to REPO_ROOT; never inside the code
    #   folder; an existing corpus is refused unless --overwrite.
    out = abspath(args.out).resolve()
    if out == CODE_ROOT or CODE_ROOT in out.parents:
        raise SystemExit(f"--out {out} is inside the code folder {CODE_ROOT}")
    from ensembits.descriptor import with_convention
    desc_name = with_convention("esm3desc_dihedral_K16_P10.npy", args.dihedral_convention)
    if (out / desc_name).exists() and not args.overwrite:
        raise SystemExit(f"{out / desc_name} exists; pass --overwrite to rebuild")
    out.mkdir(parents=True, exist_ok=True)
    systems = _sys_list(abspath(args.bioemu_bb), abspath(args.misato_bb),
                        check_parser=args.dihedral_convention != "legacy_plus180")
    upstream = []
    if args.dihedral_convention != "legacy_plus180":
        upstream = misato_upstream_failures(
            abspath(args.misato_bb), {p for _, p, c, _ in systems if c == "misato"})
        if upstream:
            print(f"[build] MISATO complexes without a backbone file: {len(upstream)} "
                  f"({sum(u['counted'] for u in upstream)} counted as drops)", flush=True)

    # ---- Descriptors: one system at a time, all 10 frames ----
    parts, layout, offset, dropped, t0 = [], [], 0, 0, time.time()
    check_parser = args.dihedral_convention != "legacy_plus180"
    # FIXED(code_fix P26, 2026-10-16): every dropped system with its reason (old: only
    #   a count, and nothing in layout.json).
    dropped_rows = []
    for i, (system, pid, corpus, f) in enumerate(systems):
        if not f.exists():
            dropped += 1
            dropped_rows.append({"system": system, "pid": pid, "corpus": corpus,
                                 "reason": "file missing"})
            continue
        # FIXED(code_fix P26, 2026-10-16): MISATO files are checked one by one (old: a
        #   sample of 50 by require_misato_parser): parser version on EVERY file read,
        #   and bb_10 must be present. Either failure aborts the build (SystemExit is
        #   not caught by the except below).
        if corpus == "misato":
            with np.load(f, allow_pickle=False) as z:
                if "bb_10" not in z.files:
                    raise SystemExit(f"{f} has no bb_10 (fields {z.files}); build the MISATO "
                                     f"cache with build_misato_real_bb.py --K 10")
                v = str(z["parser_version"]) if "parser_version" in z.files else None
                kk = int(z["K"]) if "K" in z.files else None
                ff = str(z["fps"]) if "fps" in z.files else None
            if check_parser and v != MISATO_PARSER_VERSION:
                raise SystemExit(f"{f} was written by MISATO parser {v!r}, expected "
                                 f"{MISATO_PARSER_VERSION!r} (code_fix §4)")
            # FIXED(preretrain_audit_r2 Q19 / R2-C9, 2026-10-16): old = K and the FPS
            #   rule were not checked (a K = 10 raw-coordinate FPS cache was accepted).
            if check_parser and (kk != 10 or ff != "kabsch"):
                raise SystemExit(f"{f} has K={kk}, fps={ff!r}; the corpus needs K = 10 "
                                 f"frames chosen by Kabsch FPS (build_misato_real_bb.py --K 10)")
        try:
            bb10 = np.asarray(np.load(f)["bb_10"], dtype=np.float32)          # (K,L,4,3)
            # N, CA, C only (O is not used by the descriptor).
            ncac = np.stack([bb10[:, :, 0, :], bb10[:, :, 1, :], bb10[:, :, 2, :]], axis=2)
            # Paper: appendix.tex app:descriptor-esm3 — f^p_r in R^256 per residue and frame.
            # FIXED(code_fix §1, 2026-10-15): φ/ψ convention from --dihedral-convention.
            with torch.no_grad():
                desc = _descriptor(torch.from_numpy(ncac), args.device,
                                   convention=args.dihedral_convention,
                                   chain_break_cutoff=cutoff).numpy()  # (K,L,256)
        except Exception as e:
            dropped += 1
            dropped_rows.append({"system": system, "pid": pid, "corpus": corpus,
                                 "reason": f"{type(e).__name__}: {str(e)[:200]}"})
            if dropped < 20:
                print(f"  DROP {system}: {type(e).__name__}: {e}", flush=True)
            continue
        L = desc.shape[1]
        parts.append(desc.transpose(1, 0, 2))                                # (L,K,256)
        # Layout row: where this system's residues start in the flat array.
        layout.append({"system": system, "pid": pid, "corpus": corpus,
                       "L": int(L), "start": int(offset)})
        offset += L
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(systems)}] {time.time()-t0:.0f}s res={offset} dropped={dropped}", flush=True)
    # FIXED(code_fix P26, 2026-10-16): abort above MAX_DROP_FRAC dropped systems (old:
    #   written whatever the count). The full list is printed and nothing is written.
    # Q13 (2026-10-16): complexes the MISATO backbone build failed on count too.
    dropped_rows += upstream
    n_bad = dropped + sum(u["counted"] for u in upstream)
    n_all = len(systems) + sum(u["counted"] for u in upstream)
    if n_bad > MAX_DROP_FRAC * n_all:
        for d in dropped_rows:
            print(f"  DROPPED {d}", flush=True)
        raise SystemExit(f"{n_bad} of {n_all} systems dropped "
                         f"(> {MAX_DROP_FRAC:.1%}); nothing written")
    # FIXED(code_fix E0 / LAM, 2026-10-20): corpus built for the _rv retrain.
    # code_fix §1: the file name carries the convention (iupac caches get "_iupac"),
    # so a legacy cache is never overwritten or mixed with an iupac checkpoint.
    flat = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    del parts
    # Q14 (2026-10-16): written to a temporary name, then renamed (a killed build
    #   leaves no truncated cache under the real name).
    tmp = out / (desc_name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, flat)
    os.replace(tmp, out / desc_name)
    from ensembits.descriptor import PER_NB
    # The trainer reads "dihedral_convention" from layout.json, refuses a mismatch
    # and records the corpus settings in config.json.
    (out / "layout.json").write_text(json.dumps(
        {"rows": layout, "dropped": dropped_rows,     # code_fix P26
         "dihedral_convention": args.dihedral_convention,
         "chain_break_cutoff": cutoff, "split_unit": args.split_unit,
         # Q19 (2026-10-16): neighbour count and frames, read and checked by the trainer.
         "k": int(flat.shape[-1] // PER_NB), "frames": int(flat.shape[1]),
         "bioemu_bb": str(args.bioemu_bb), "misato_bb": str(args.misato_bb),
         "misato_parser_version": MISATO_PARSER_VERSION
         if args.dihedral_convention != "legacy_plus180" else None}, indent=1))
    print(f"[build] flat {flat.shape} ({dropped} dropped) -> {out}", flush=True)

    # ---- Split: random 80/10/10 (seed 42) ----
    # Paper: appendix.tex \paragraph{Combined training corpus.} — random 80/10/10.
    # FIXED(code_fix §5 / T1.5, 2026-10-15): the unit is the domain / complex, so the
    #   two simulations of a BioEmu domain never land in different splits. The old
    #   split (--split-unit system) put 5 duplicated domains in two splits.
    grp, n = assign_splits(layout, args.split_unit, args.seed)
    # Expand system membership to residue row indices into the flat array.
    split = {"train": [], "val": [], "test": []}
    for j, r in enumerate(layout):
        split[grp[j]].extend(range(r["start"], r["start"] + r["L"]))
    (out / "splits.json").write_text(json.dumps(split))
    nb = sum(1 for r in layout if r["corpus"] == "bioemu_cath")
    print(f"[build] systems: bioemu={nb} misato={len(layout)-nb} ({n} split units)  split res "
          f"tr={len(split['train'])} va={len(split['val'])} te={len(split['test'])}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
