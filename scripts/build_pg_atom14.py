"""Cache ESMFold atom14 backbones for ProteinGym variants.

For each ProteinGym DMS CSV (WT + every unique mutated sequence), runs
ESMFold and saves the predicted (L, 14, 3) atom14 trajectory as
``{out_dir}/{assay}/{sha1(seq)[:16]}.npy``. Resumable; ``--gpus 0,1,2``
shards across GPUs.

The OpenFold atom14 layout shipped by ESMFold is, per residue:
``[N=0, CA=1, C=2, O=3, CB=4, CG=5, ...]`` (verified empirically by
C–O bond length). Slot 4 is **junk for GLY**. Downstream consumers
read N/CA/C from slots 0/1/2 and O from slot 3; the structural-tokenizer
path therefore sees ESMFold's predicted backbone directly (not an
ideal-geometry reconstruction from Cα).

Usage:
    python -m scripts.build_pg_atom14 \\
        --pg-csv-dir <path-to-DMS_substitutions/> \\
        --out data/proteingym_esmfold_atom14/ \\
        --gpus 0 --max-len 600
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _hash_seq(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _derive_wt(rows: list[dict]) -> str | None:
    """Recover the WT sequence from a list of variant rows."""
    for r in rows:
        mutant = r.get("mutant", "")
        ms = r.get("mutated_sequence", "")
        if not mutant or not ms or ":" in mutant:
            continue
        try:
            wt = list(ms); pos = int(mutant[1:-1]) - 1
            if 0 <= pos < len(ms):
                wt[pos] = mutant[0]; return "".join(wt)
        except (ValueError, IndexError):
            continue
    # Multi-mutation fallback
    for r in rows:
        mutant = r.get("mutant", ""); ms = r.get("mutated_sequence", "")
        if not mutant or not ms:
            continue
        try:
            wt = list(ms)
            for sub in mutant.split(":"):
                if sub:
                    wt[int(sub[1:-1]) - 1] = sub[0]
            return "".join(wt)
        except (ValueError, IndexError):
            continue
    return None


def _load_assay(csv_path: Path) -> tuple[str | None, list[str]]:
    """(WT seq, list of variant seqs) for one DMS CSV."""
    rows: list[dict] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("mutated_sequence", ""):
                rows.append(row)
    if not rows:
        return None, []
    wt = _derive_wt(rows)
    seqs = list({r["mutated_sequence"] for r in rows})
    return wt, seqs


def _fold_one(model, tokenizer, seq: str):
    """Run ESMFold on one sequence; return atom14 (L, 14, 3) float32."""
    import torch
    with torch.inference_mode():
        toks = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
        toks = {k: v.to(model.device) for k, v in toks.items()}
        out = model(**toks)
        # ESMFold's atom14 lives at out.positions[-1] (last recycle):
        # shape (1, 14, L, 3); transpose → (L, 14, 3).
        a14 = out.positions[-1, 0].permute(1, 0, 2).cpu().numpy().astype(np.float32)
    return a14


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pg-csv-dir", type=Path, required=True,
                    help="Directory of ProteinGym DMS_substitutions/*.csv")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir; one subdir per assay, .npy per seq")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=600,
                    help="Skip variants longer than this (ESMFold OOM guard)")
    ap.add_argument("--assay-limit", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
    except ImportError as e:
        raise SystemExit("transformers + torch required for ESMFold; "
                          "install `pip install transformers`.") from e

    print(f"[1/3] Loading ESMFold (facebook/esmfold_v1) on cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                  torch_dtype=torch.float32).to("cuda:0").eval()
    model.esm = model.esm.half()  # FP16 trunk to save VRAM

    csvs = sorted(args.pg_csv_dir.glob("*.csv"))
    if args.assay_limit:
        csvs = csvs[:args.assay_limit]
    print(f"\n[2/3] {len(csvs)} assays to process")

    t0 = time.time()
    total_seq = 0
    for ai, csv_path in enumerate(csvs):
        assay = csv_path.stem
        assay_out = args.out / assay
        assay_out.mkdir(parents=True, exist_ok=True)

        wt, seqs = _load_assay(csv_path)
        if wt is None:
            print(f"  [skip] {assay}: WT not recoverable"); continue
        seqs_to_fold = [wt] + [s for s in seqs if s != wt]
        seqs_to_fold = [s for s in seqs_to_fold if len(s) <= args.max_len]
        n_new = 0
        for s in seqs_to_fold:
            out_path = assay_out / f"{_hash_seq(s)}.npy"
            if out_path.exists():
                continue
            try:
                a14 = _fold_one(model, tokenizer, s)
            except Exception as exc:
                print(f"    fold fail [{assay}, len={len(s)}]: {exc}"); continue
            np.save(out_path, a14)
            n_new += 1
        total_seq += n_new
        print(f"  [{ai+1}/{len(csvs)}] {assay}: {len(seqs_to_fold)} variants, "
              f"{n_new} new ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[3/3] DONE in {time.time()-t0:.0f}s — {total_seq} new sequences folded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
