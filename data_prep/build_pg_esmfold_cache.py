"""ProteinGym ESMFold structure cache builder.

Provenance
----------
Ported from the research repo pg_esmfold_cache.py (see pg_audit.md)
sha256 c84e8b08094ae7b7ea3a1a9455fc6014d5a38b7a3a2f47235530c2aa1aa870d4.

Changes from the research-repo original:
  * ROOT and PG_DIR resolved through ensembits.paths (REPO_ROOT,
    ENSEMBITS_PROTEINGYM_DIR) instead of hard-coded collaborator paths.
  * Default CACHE_DIR under REPO_ROOT/data/proteingym/esmfold_cache.
  * LOG_DIR under REPO_ROOT/logs/pg_esmfold_cache.
  * Added --assay-list flag (one assay id per line) for consistency with
    pg_score.py.
  * ESMFold is loaded via HuggingFace ``from_pretrained`` (no explicit token
    needed; ``facebook/esmfold_v1`` is public).
  * Provenance docstring added.
  No scoring or metric logic was changed.

What it does
------------
For each ProteinGym DMS assay CSV that meets the length filter:
  1. Reconstructs the wild-type sequence from a single-mutation row.
  2. Collects all unique mutated sequences.
  3. Runs ESMFold (facebook/esmfold_v1, fp32) on each sequence not yet cached.
  4. Saves the predicted coordinates to
     data/proteingym/esmfold_cache/<assay>/<sha1[:16]>.npy
     (Cα only by default, or full atom14 with --save-atom14).
  5. Writes a manifest.json per assay.

Resumable (skips existing .npy files). Multi-GPU with --gpus.

Paper
-----
appendix.tex, Compute Requirements: "ESMFold ≈ 33 GPU-h (partial log)".

Used by the reported model: yes (the caches are required input for
pg_score_dihedral.py and pg_score_baselines.py).

Inputs / outputs
----------------
Reads:
  data/external/proteingym_zero_shot_scores/<assay>.csv  (ENSEMBITS_PROTEINGYM_DIR)
Writes:
  data/proteingym/esmfold_cache/<assay>/{manifest.json, <hash>.npy}
  data/proteingym/esmfold_cache_atom14/<assay>/...  (with --save-atom14 --out-dir ...)

How to run
----------
  PY=python  # the project's Python environment
  $PY data_prep/build_pg_esmfold_cache.py --gpus 0,1,2 --max-len 200
  # atom14 mode (needed by pg_score_baselines for 3Di / ESM3 / AminoAseed):
  $PY data_prep/build_pg_esmfold_cache.py --gpus 0,1,2 --max-len 200 \\
      --save-atom14 --out-dir data/proteingym/esmfold_cache_atom14
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import csv
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402

PG_DIR = Path(os.environ.get(
    "ENSEMBITS_PROTEINGYM_DIR",
    str(ROOT / "data" / "external" / "proteingym_zero_shot_scores")))
CACHE_DIR = ROOT / "data" / "proteingym" / "esmfold_cache"
LOG_DIR = ROOT / "logs" / "pg_esmfold_cache"


def hash_seq(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def derive_wt(rows):
    """Reconstruct WT sequence from any single-mutation variant."""
    for r in rows:
        mutant = r.get("mutant", "")
        ms = r.get("mutated_sequence", "")
        if not mutant or not ms or ":" in mutant:
            continue
        try:
            wt_aa = mutant[0]
            pos = int(mutant[1:-1]) - 1
            if 0 <= pos < len(ms):
                wt = list(ms)
                wt[pos] = wt_aa
                return "".join(wt)
        except (ValueError, IndexError):
            continue
    for r in rows:
        mutant = r.get("mutant", "")
        ms = r.get("mutated_sequence", "")
        if not mutant or not ms:
            continue
        try:
            wt = list(ms)
            for sub in mutant.split(":"):
                if not sub:
                    continue
                wt_aa = sub[0]
                pos = int(sub[1:-1]) - 1
                wt[pos] = wt_aa
            return "".join(wt)
        except (ValueError, IndexError):
            continue
    return None


def load_assay(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("mutated_sequence", ""):
                rows.append(row)
    if not rows:
        return None, None
    wt = derive_wt(rows)
    L = len(rows[0]["mutated_sequence"])
    if wt is None or len(wt) != L:
        return None, None
    return wt, rows


def _save_atom_record(out, batch_idx, L, path, save_atom14: bool):
    """Save (L, 14, 3) atom14 or (L, 3) Cα from an ESMFold output."""
    last = out.positions[-1]
    if save_atom14:
        atom14 = last[batch_idx, :L, :, :].cpu().numpy().astype(np.float32)
        if atom14.shape[0] != L:
            raise ValueError(f"atom14 shape mismatch L={L} got {atom14.shape}")
        np.save(path, atom14)
    else:
        ca = last[batch_idx, :L, 1, :].cpu().numpy().astype(np.float32)
        if ca.shape[0] != L:
            raise ValueError(f"CA shape mismatch L={L} got {ca.shape}")
        np.save(path, ca)


def fold_and_cache_assay(assay, wt, rows, model, tokenizer, device, max_len,
                         batch_size=8, save_atom14: bool = False,
                         out_dir_root: Path | None = None):
    """Fold WT + all unique mutated sequences not yet cached."""
    root = out_dir_root if out_dir_root is not None else CACHE_DIR
    cdir = root / assay
    cdir.mkdir(parents=True, exist_ok=True)

    seqs = {}
    seqs[hash_seq(wt)] = ("WT", wt)
    for r in rows:
        ms = r["mutated_sequence"]
        if len(ms) > max_len:
            continue
        h = hash_seq(ms)
        if h not in seqs:
            seqs[h] = (r.get("mutant", h), ms)

    todo = [(h, mut, s) for h, (mut, s) in seqs.items()
            if not (cdir / f"{h}.npy").exists()]
    if not todo:
        man = cdir / "manifest.json"
        if not man.exists():
            man.write_text(json.dumps({"assay": assay, "wt": wt,
                                       "n_seqs": len(seqs)}))
        return 0

    todo.sort(key=lambda x: len(x[2]))
    print(f"  [{assay}] folding {len(todo)}/{len(seqs)} (rest cached) in "
          f"batches of {batch_size} "
          f'({"atom14" if save_atom14 else "CA"} mode)', flush=True)
    t0 = time.time()
    n_done = 0
    n_fail = 0
    import torch
    for s_idx in range(0, len(todo), batch_size):
        batch = todo[s_idx : s_idx + batch_size]
        seqs_b = [item[2] for item in batch]
        try:
            inp = tokenizer(seqs_b, return_tensors="pt",
                            add_special_tokens=False, padding=True).to(device)
            with torch.no_grad():
                out = model(**inp)
            for i, (h, mut_id, s) in enumerate(batch):
                _save_atom_record(out, i, len(s), cdir / f"{h}.npy",
                                  save_atom14)
                n_done += 1
        except Exception as e:
            print(f"    [{assay}] batch ({s_idx}..{s_idx+len(batch)}) FAIL: "
                  f"{type(e).__name__} {e} — falling back to single-seq",
                  flush=True)
            for h, mut_id, s in batch:
                try:
                    inp = tokenizer([s], return_tensors="pt",
                                    add_special_tokens=False).to(device)
                    with torch.no_grad():
                        out = model(**inp)
                    _save_atom_record(out, 0, len(s), cdir / f"{h}.npy",
                                      save_atom14)
                    n_done += 1
                except Exception as e2:
                    print(f"    [{assay}/{mut_id}] FAIL: "
                          f"{type(e2).__name__} {e2}", flush=True)
                    n_fail += 1

    man_data = {"assay": assay, "wt": wt, "wt_hash": hash_seq(wt),
                "n_seqs": len(seqs),
                "variants": {h: mut for h, (mut, _) in seqs.items()}}
    (cdir / "manifest.json").write_text(json.dumps(man_data))

    el = time.time() - t0
    print(f"  [{assay}] folded {n_done} ({n_fail} failed) in {el:.0f}s "
          f"({el / max(n_done, 1):.2f}s/seq)", flush=True)
    return n_done


_LOAD_LOCK = threading.Lock()


def worker(gpu, assay_q, max_len, batch_size, save_atom14, out_dir_root):
    """One worker per GPU: load ESMFold once, then drain the queue."""
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    device = f"cuda:{gpu}"
    print(f"[GPU {gpu}] loading ESMFold on {device}...", flush=True)
    t0 = time.time()
    with _LOAD_LOCK:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", torch_dtype=torch.float32).eval().to(device)
    print(f"[GPU {gpu}] loaded in {time.time() - t0:.0f}s", flush=True)

    while True:
        try:
            assay = assay_q.get(timeout=2)
        except queue.Empty:
            return
        try:
            csv_path = PG_DIR / f"{assay}.csv"
            wt, rows = load_assay(csv_path)
            if wt is None:
                print(f"[GPU {gpu}] {assay}: failed to derive WT, skip",
                      flush=True)
                assay_q.task_done()
                continue
            if len(wt) > max_len:
                print(f"[GPU {gpu}] {assay}: len={len(wt)} > max_len="
                      f"{max_len}, skip", flush=True)
                assay_q.task_done()
                continue
            log_path = LOG_DIR / f"{assay}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            t0b = time.time()
            print(f"[GPU {gpu}] START {assay} (L={len(wt)}, "
                  f"n_var={len(rows)})", flush=True)
            n = fold_and_cache_assay(assay, wt, rows, model, tokenizer,
                                     device, max_len,
                                     batch_size=batch_size,
                                     save_atom14=save_atom14,
                                     out_dir_root=out_dir_root)
            print(f"[GPU {gpu}] DONE  {assay} ({n} new in "
                  f"{(time.time() - t0b) / 60:.1f}min)", flush=True)
        except Exception as e:
            print(f"[GPU {gpu}] {assay}: ERROR {type(e).__name__} {e}",
                  flush=True)
        finally:
            assay_q.task_done()


def main():
    ap = argparse.ArgumentParser(
        description="Build ESMFold structure cache for ProteinGym assays.")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--max-len", type=int, default=200,
                    help="Skip assays whose WT seq exceeds this length.")
    ap.add_argument("--assays", default=None,
                    help="Comma-separated list of assay names.")
    ap.add_argument("--assay-list", default=None,
                    help="File with one assay id per line (# comments ok).")
    ap.add_argument("--max-assays", type=int, default=None,
                    help="Limit to first N assays (for testing).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-atom14", action="store_true",
                    help="Save full (L, 14, 3) atom14 instead of (L, 3) Cα.")
    ap.add_argument("--out-dir", default=None,
                    help="Override the output cache root.")
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # B.18: resolve --out-dir against REPO_ROOT (not cwd)
    out_dir_root = (ROOT / args.out_dir if args.out_dir else CACHE_DIR)
    out_dir_root.mkdir(parents=True, exist_ok=True)

    import transformers  # noqa: F401
    from transformers import AutoTokenizer, EsmForProteinFolding  # noqa: F401
    print("[main] transformers + ESMFold class pre-imported", flush=True)

    if args.assays:
        assays = args.assays.split(",")
    elif args.assay_list:
        assays = [ln.strip() for ln in open(args.assay_list)
                  if ln.strip() and not ln.startswith("#")]
    else:
        import glob as _g
        assays = [Path(p).stem
                  for p in sorted(_g.glob(str(PG_DIR / "*.csv")))]
    if args.max_assays:
        assays = assays[: args.max_assays]

    eligible = []
    for assay in assays:
        csv_path = PG_DIR / f"{assay}.csv"
        if not csv_path.exists():
            print(f"  skip {assay}: CSV not found", flush=True)
            continue
        wt, _ = load_assay(csv_path)
        if wt is not None and len(wt) <= args.max_len:
            eligible.append((assay, len(wt)))
    eligible.sort(key=lambda x: x[1])
    if not eligible:
        print("No eligible assays.", flush=True)
        return
    print(f"eligible: {len(eligible)} assays (max_len={args.max_len})",
          flush=True)
    print(f"len range: {eligible[0][1]}..{eligible[-1][1]}", flush=True)

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    q: queue.Queue[str] = queue.Queue()
    for assay, _L in eligible:
        q.put(assay)
    threads = [threading.Thread(target=worker,
                                args=(g, q, args.max_len, args.batch_size,
                                      args.save_atom14, out_dir_root),
                                daemon=False) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("All done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
