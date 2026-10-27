"""CryptoBench: PDB-id overlap between CryptoBench chains and the tokenizer's training corpus.

What it does
------------
Leakage check required before CryptoBench can carry the main text
(`camera_ready/code_fix.md` E3: "check structure overlap between CryptoBench test
chains and the tokenizer's training corpus (MISATO is PDBbind-derived)").

  1. CryptoBench side: apo PDB ids per fold (`folds.json`), and for each apo its
     holo partners (`dataset.json`), with chains. The apo structure is the probe
     input; the holo structures define the labels.
  2. Training-corpus side (reported model, BioEmu-CATH + MISATO):
       * MISATO: PDB ids = file stems of `data/cached_descriptors/misato_real_bb_kabsch/*.npz`
         (the real-backbone MISATO cache the retrain corpus is built from; 2026-10-16,
         code_fix P22; the pre-fix check used misato_real_bb);
       * the corpus layout `output/bioemu_cath_misato_e0/cached_desc/layout.json`
         (one row per system: system, pid, corpus, L, start; pre-fix corpus:
         output/reviewer_3_special_corpus_MISATO/cached_desc) gives the MISATO pids and
         the BioEmu-CATH CATH domain ids (e.g. 1a1wA00 -> PDB 1a1w, chain A)
         that actually entered the corpus;
       * `splits.json` of that corpus indexes RESIDUES of the concatenated layout;
         a protein counts as "in the tokenizer's train split" if any of its
         residues are in `train`.
  3. Reports, per CryptoBench fold, how many apo ids / holo ids / entries share a
     PDB id with MISATO or BioEmu-CATH (and, for BioEmu-CATH, the same chain),
     and the share of test chains and test residues (from the apo manifest)
     whose apo or any holo PDB id is in the corpus.

Limitation: PDB-id identity is a lower bound on protein overlap. The same protein
under another PDB id (other crystal form, homologue) is not detected; a UniProt
or sequence-identity check (CryptoBench gives UniProt ids; MISATO / CATH would
need a SIFTS mapping or the sequences) is the stronger test.
# Open(code_fix E3): extend to UniProt / sequence-identity overlap if PDB-id overlap is non-zero.

Paper
-----
Not in the paper yet: CryptoBench goes into the main text
(`camera_ready/require_fix.md` #24, `camera_ready/code_fix.md` E3).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Corpus: `2_results.tex` Datasets; `appendix.tex` \\label{app:dataset}.

Origin
------
New file written for this codebase (not copied from the research repo).

Used by the reported model: yes (checks the reported model's training corpus).

Inputs / outputs
----------------
Read  : `data/cryptobench/dataset.json`, `folds.json`;
        `<manifest-dir>/apo_manifest.json` (optional, for chain/residue shares;
        falls back to `data/cryptobench/apo_manifest.json`, the rebuttal build);
        `data/cached_descriptors/misato_real_bb_kabsch/*.npz` (file names only);
        `output/bioemu_cath_misato_e0/cached_desc/layout.json`, `splits.json`
        (defaults since 2026-10-16; --corpus-dir / --misato-bb-dir select others).
Write : `<out-dir>/overlap_check.json` (counts + the overlapping ids).

How to run
----------
  $PY downstream/cryptobench/check_training_overlap.py           # CPU, < 1 min
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402  data/ and output/ stay in the research repo
import _cb_paths as CBP  # noqa: E402

# FIXED(code_fix P22, 2026-10-16): old: the current checkpoint's corpus
#   (output/reviewer_3_special_corpus_MISATO/cached_desc, rows keyed by pid) and the
#   pre-fix MISATO cache (misato_real_bb). New: the retrain corpus of
#   data_prep/build_bioemu_misato_descriptor.py (output/<corpus>/cached_desc/layout.json
#   with "system" keys; default corpus bioemu_cath_misato_e0) and misato_real_bb_kabsch.
#   The old check is --corpus-dir output/reviewer_3_special_corpus_MISATO/cached_desc
#   --misato-bb-dir data/cached_descriptors/misato_real_bb.
# FIXED(code_fix E0 / LAM, 2026-10-23): corpus name = bioemu_cath_misato (unchanged).
DEFAULT_CORPUS_DIR = ROOT / "output" / "bioemu_cath_misato_e0" / "cached_desc"
DEFAULT_MISATO_BB = ROOT / "data" / "cached_descriptors" / "misato_real_bb_kabsch"


def corpus_ids(corpus_dir: Path):
    """-> {corpus: {pid: in_train(bool)}} from layout.json + residue-level splits.json."""
    rows = json.load(open(corpus_dir / "layout.json"))["rows"]
    n_res = sum(r["L"] for r in rows)
    split_of = np.full(n_res, -1, dtype=np.int8)
    splits_path = corpus_dir / "splits.json"
    names = ("train", "val", "test")
    if splits_path.exists():
        sp = json.load(open(splits_path))
        for i, nm in enumerate(names):
            split_of[np.asarray(sp.get(nm, []), dtype=np.int64)] = i
    out = defaultdict(dict)
    per_pid_mixed = 0
    for r in rows:
        s = split_of[r["start"]:r["start"] + r["L"]]
        u = set(np.unique(s).tolist())
        per_pid_mixed += len(u - {-1}) > 1
        # FIXED(code_fix P22, 2026-10-16): the new layout has one row per SYSTEM (a
        #   BioEmu-CATH domain can have two); a pid counts as "in train" if any of its
        #   systems has train residues (before: the last row of the pid won).
        out[r["corpus"]][r["pid"]] = out[r["corpus"]].get(r["pid"], False) or bool((s == 0).any())
    return out, per_pid_mixed, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR),
                    help="tokenizer corpus folder with layout.json and splits.json "
                         "(default: the reported model's BioEmu-CATH + MISATO corpus)")
    ap.add_argument("--misato-bb-dir", default=str(DEFAULT_MISATO_BB),
                    help="MISATO real-backbone cache; its *.npz stems are the MISATO PDB ids")
    ap.add_argument("--manifest", default=None,
                    help="apo manifest for chain / residue shares (default: <manifest-dir>/apo_manifest.json, "
                         "else data/cryptobench/apo_manifest.json)")
    CBP.add_path_args(ap)
    args = ap.parse_args()
    for flag, d in (("--corpus-dir", CBP.repo_path(args.corpus_dir) / "layout.json"),
                    ("--misato-bb-dir", CBP.repo_path(args.misato_bb_dir))):
        if not d.exists():   # FIXED(code_fix P22, 2026-10-16): new (old: none): say what is missing instead of a traceback
            ap.error(f"{d} not found; build it first or pass {flag} (the pre-fix corpus is "
                     f"output/reviewer_3_special_corpus_MISATO/cached_desc + "
                     f"data/cached_descriptors/misato_real_bb)")
    paths = CBP.resolve_paths(args, create=("out_dir",))
    cbdir = paths.cryptobench_dir

    # --- CryptoBench ids -------------------------------------------------------
    dataset = json.load(open(cbdir / "dataset.json"))
    folds = json.load(open(cbdir / "folds.json"))
    fold_of = {pid.lower(): f for f, ids in folds.items() for pid in ids}
    holo_of = defaultdict(set)          # apo -> {holo pdb}
    for apo, holos in dataset.items():
        for h in holos:
            holo_of[apo.lower()].add(str(h["holo_pdb_id"]).lower())

    # --- training corpus ids ----------------------------------------------------
    misato_bb = {p.stem.lower() for p in CBP.repo_path(args.misato_bb_dir).glob("*.npz")}
    corp, mixed, n_rows = corpus_ids(CBP.repo_path(args.corpus_dir))
    misato_corpus = {p.lower(): t for p, t in corp.get("misato", {}).items()}
    bioemu = corp.get("bioemu_cath", {})
    bioemu_pdb = defaultdict(bool)       # pdb -> any domain in train
    bioemu_chain = defaultdict(bool)     # (pdb, chain) -> any domain in train
    for dom, t in bioemu.items():
        pdb, ch = dom[:4].lower(), dom[4]
        bioemu_pdb[pdb] |= t
        bioemu_chain[(pdb, ch)] |= t
    print(f"[overlap] CryptoBench: {len(dataset)} apo ids, "
          f"{len(set().union(*holo_of.values()))} holo ids; folds "
          f"{ {k: len(v) for k, v in folds.items()} }")
    print(f"[overlap] corpus rows={n_rows} (MISATO {len(misato_corpus)}, BioEmu-CATH {len(bioemu)} domains / "
          f"{len(bioemu_pdb)} PDB ids); proteins split across >1 of train/val/test: {mixed}; "
          f"MISATO real-bb cache ids={len(misato_bb)}")

    # --- per-fold PDB-id overlap -------------------------------------------------
    res = {"folds": {}}
    groups = dict(folds)
    groups["all"] = [p for ids in folds.values() for p in ids]
    for g, ids in groups.items():
        apo = {p.lower() for p in ids}
        holo = set().union(*[holo_of[a] for a in apo]) if apo else set()
        ent_hit_m = {a for a in apo if a in misato_bb or holo_of[a] & misato_bb}
        ent_hit_mt = {a for a in apo
                      if misato_corpus.get(a) or any(misato_corpus.get(h) for h in holo_of[a])}
        ent_hit_b = {a for a in apo if a in bioemu_pdb or holo_of[a] & set(bioemu_pdb)}
        r = {
            "n_apo": len(apo), "n_holo": len(holo),
            "apo_in_misato": sorted(apo & misato_bb),
            "holo_in_misato": sorted(holo & misato_bb),
            "apo_in_misato_train": sorted(a for a in apo if misato_corpus.get(a)),
            "holo_in_misato_train": sorted(h for h in holo if misato_corpus.get(h)),
            "apo_in_bioemu_pdb": sorted(apo & set(bioemu_pdb)),
            "holo_in_bioemu_pdb": sorted(holo & set(bioemu_pdb)),
            "entries_any_in_misato": len(ent_hit_m),
            "entries_any_in_misato_train": len(ent_hit_mt),
            "entries_any_in_bioemu_pdb": len(ent_hit_b),
        }
        res["folds"][g] = r
        print(f"  {g:<8} apo={len(apo):4d} holo={len(holo):4d} | MISATO: apo {len(r['apo_in_misato']):3d} "
              f"holo {len(r['holo_in_misato']):3d} entries {len(ent_hit_m):3d} "
              f"(train split: apo {len(r['apo_in_misato_train'])}, holo {len(r['holo_in_misato_train'])}, "
              f"entries {len(ent_hit_mt)}) | BioEmu-CATH PDB: apo {len(r['apo_in_bioemu_pdb'])} "
              f"holo {len(r['holo_in_bioemu_pdb'])} entries {len(ent_hit_b)}")

    # --- test chains / residues of the probe input (apo manifest) --------------
    man_path = Path(args.manifest) if args.manifest else CBP.manifest_path(paths, "apo")
    if not man_path.exists():
        man_path = ROOT / "data" / "cryptobench" / "apo_manifest.json"
    if man_path.exists():
        ent = [e for e in json.load(open(man_path))["entries"] if e["fold"] == "test"]
        def hit(e, which):
            a = e["apo"].lower()
            if which == "misato":
                return a in misato_bb or bool(holo_of[a] & misato_bb)
            if which == "bioemu_chain":
                return (a, e["chain"]) in bioemu_chain
            return a in bioemu_pdb or bool(holo_of[a] & set(bioemu_pdb))
        tot_c, tot_r = len(ent), sum(e["L"] for e in ent)
        tc = {}
        for w in ("misato", "bioemu_pdb", "bioemu_chain"):
            h = [e for e in ent if hit(e, w)]
            tc[w] = {"chains": len(h), "residues": int(sum(e["L"] for e in h)),
                     "positives": int(sum(e["n_pos"] for e in h)),
                     "ids": sorted(f"{e['apo']}_{e['chain']}" for e in h)}
            print(f"  test chains hit ({w}): {len(h)}/{tot_c} chains, "
                  f"{tc[w]['residues']}/{tot_r} residues")
        res["test_chains"] = {"manifest": str(man_path.relative_to(ROOT)) if man_path.is_relative_to(ROOT)
                              else man_path.name, "n_chains": tot_c, "n_residues": tot_r, "hits": tc}

    res["meta"] = {"corpus_dir": str(args.corpus_dir), "misato_bb_dir": str(args.misato_bb_dir),
                   "n_misato_bb": len(misato_bb), "n_misato_corpus": len(misato_corpus),
                   "n_bioemu_domains": len(bioemu), "n_bioemu_pdb": len(bioemu_pdb),
                   "proteins_split_across_sets": int(mixed),
                   "note": "PDB-id overlap only (lower bound on protein overlap)"}
    out = paths.out_dir / "overlap_check.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
