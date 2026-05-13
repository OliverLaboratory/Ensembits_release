"""Build per-baseline token caches for the downstream probes.

Produces a single ``.npz`` keyed by pid for the chosen baseline +
dataset. Every structural baseline reads real backbone from
``{data_dir}/{dataset}_real_bb/<pid>.npz`` (post-fix path); sequence-
and Cα-direct baselines read the ``seq`` / ``ca_K`` fields from the
same cache.

Examples:

    # mini3di single-frame, mdcath
    python -m scripts.build_baseline_tokens \\
        --baseline 3di_tokens --dataset mdcath \\
        --out data/cached_descriptors/3di_tokens_mdcath.npz

    # AminoAseed on misato (needs StructTokenBench + checkpoint)
    python -m scripts.build_baseline_tokens \\
        --baseline aminoaseed --dataset misato \\
        --aminoaseed-repo $HOME/StructTokenBench/src \\
        --aminoaseed-ckpt $HOME/aminoaseed.ckpt/checkpoint/mp_rank_00_model_states.pt \\
        --out data/cached_descriptors/aminoaseed_tokens_misato.npz

    # AA one-hot baseline (sequence-only)
    python -m scripts.build_baseline_tokens \\
        --baseline aa --dataset misato \\
        --out data/cached_descriptors/aa_tokens_misato.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines._backbone import (                                # noqa: E402
    load_bb4_for_pid, load_cb_for_pid, load_bb_for_pid,
)

STRUCTURAL = (
    "3di_tokens", "vote_3di",
    "aminoaseed", "esm3struct", "protoken",
)
SEQUENCE_ONLY = ("aa",)
CA_ONLY = ("random",)
SUPPORTED = STRUCTURAL + SEQUENCE_ONLY + CA_ONLY


def _load_seq(pid: str, dataset: str, data_dir: Path) -> str | None:
    path = data_dir / f"{dataset}_real_bb" / f"{pid}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    return str(d["seq"]) if "seq" in d.files else None


def _bb_and_cb(pid: str, dataset: str, P: int, data_dir: Path):
    """Return ``(bb4_K, cb_K)`` for the pid where bb4 is (P, L, 4, 3)
    N/CA/C/O and cb is (P, L, 3) Cβ (NaN at GLY). Returns ``(None, None)``
    if the cache is missing or doesn't have the requested K."""
    bb4 = load_bb4_for_pid(pid, dataset=dataset, P=P, data_dir=data_dir)
    cb = load_cb_for_pid(pid, dataset=dataset, P=P, data_dir=data_dir)
    return bb4, cb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, choices=SUPPORTED)
    ap.add_argument("--dataset", required=True, choices=("mdcath", "misato"))
    ap.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--P", type=int, default=10,
                    help="Number of frames per pid (must match the real_bb cache)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")

    # AminoAseed
    ap.add_argument("--aminoaseed-repo", type=Path,
                    help="StructTokenBench src/ directory")
    ap.add_argument("--aminoaseed-ckpt", type=Path,
                    help="AminoAseed checkpoint .pt file")

    # ESM3Struct (mostly auto-downloaded from HF)
    ap.add_argument("--esm3-weights", type=Path, default=None,
                    help="(optional) local ESM3 weights dir")

    # ProToken
    ap.add_argument("--protoken-repo", type=Path,
                    help="ProToken full/ directory")
    ap.add_argument("--protoken-device", default="0")

    # Random baseline
    ap.add_argument("--K", type=int, default=2048,
                    help="(random) codebook size — match the Ensembits primary M for fair comparison")
    ap.add_argument("--seed", type=int, default=0,
                    help="(random) per-pid hash seed")

    # Mini3di frame selection: single-frame (3di_tokens uses frame 0) vs
    # multi-frame (vote_3di uses all P frames; protprofile uses all P).
    ap.add_argument("--device", default=None, help="cuda or cpu for AminoAseed/ESM3")

    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"[skip] {args.out} exists (use --force)")

    real_bb_dir = args.data_dir / f"{args.dataset}_real_bb"
    pids = sorted(p.stem for p in real_bb_dir.glob("*.npz"))
    if args.limit:
        pids = pids[:args.limit]
    print(f"[1/2] {len(pids)} pids in {real_bb_dir}; baseline={args.baseline}")

    # Lazy-load the baseline tokenizer (some need heavy upstream deps)
    if args.baseline == "aminoaseed":
        from baselines.aminoaseed import AminoAseed
        if not (args.aminoaseed_repo and args.aminoaseed_ckpt):
            raise SystemExit("--aminoaseed-repo and --aminoaseed-ckpt required")
        tok = AminoAseed(weights_path=args.aminoaseed_ckpt,
                         repo_path=args.aminoaseed_repo,
                         device=args.device or "cuda")
    elif args.baseline == "esm3struct":
        from baselines.esm3struct import ESM3Struct
        tok = ESM3Struct(weights_path=args.esm3_weights,
                          device=args.device or "cuda")
    elif args.baseline == "protoken":
        from baselines.protoken import ProToken
        if not args.protoken_repo:
            raise SystemExit("--protoken-repo required")
        tok = ProToken(repo_path=args.protoken_repo, device=args.protoken_device)
    elif args.baseline in ("3di_tokens", "vote_3di"):
        from baselines.mini3di import three_di_tokens, vote_3di
        tok = (three_di_tokens, vote_3di)
    elif args.baseline == "aa":
        from baselines.aa import aa_features
        tok = aa_features
    elif args.baseline == "random":
        from baselines.random import random_tokens
        tok = random_tokens
    else:
        raise SystemExit(f"unhandled baseline: {args.baseline}")

    out: dict[str, np.ndarray] = {}
    t0 = time.time(); n_skip = 0
    print(f"\n[2/2] tokenizing …")
    for i, pid in enumerate(pids):
        try:
            if args.baseline == "aa":
                seq = _load_seq(pid, args.dataset, args.data_dir)
                if seq is None:
                    n_skip += 1; continue
                out[pid] = tok(seq).astype(np.float32)
                continue
            if args.baseline == "random":
                # Cα-direct: just need length
                bb4, _ = _bb_and_cb(pid, args.dataset, args.P, args.data_dir)
                if bb4 is None:
                    n_skip += 1; continue
                L = bb4.shape[1]
                out[pid] = tok(pid, L, K=args.K, seed=args.seed)
                continue

            bb4, cb = _bb_and_cb(pid, args.dataset, args.P, args.data_dir)
            if bb4 is None:
                n_skip += 1; continue
            ca_first = bb4[0, :, 1, :]

            if args.baseline == "3di_tokens":
                three_di_tokens, _ = tok
                # Single-frame: use frame 0
                bb_0 = bb4[0]               # (L, 4, 3)
                cb_0 = cb[0] if cb is not None else None
                out[pid] = three_di_tokens(
                    ca_first, bb=bb_0, cb=cb_0, chain_split=True)
            elif args.baseline == "vote_3di":
                _, vote_3di = tok
                out[pid] = vote_3di(bb4[:, :, 1, :], bb_K=bb4, cb_K=cb,
                                     chain_split=True)
            elif args.baseline in ("aminoaseed", "esm3struct"):
                out[pid] = tok.tokenize(ca_first, bb=bb4[0], chain_split=True)
            elif args.baseline == "protoken":
                seq = _load_seq(pid, args.dataset, args.data_dir) or ("X" * bb4.shape[1])
                out[pid] = tok.tokenize(ca_first, seq, bb=bb4[0], chain_split=True)
        except Exception as exc:
            print(f"  [skip] {pid}: {type(exc).__name__}: {exc}", flush=True)
            n_skip += 1
            continue
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pids)}  ok={len(out)}  skip={n_skip}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  done: {len(out)}/{len(pids)} pids in {time.time()-t0:.0f}s")
    if not out:
        raise SystemExit("[abort] no pids tokenized; refusing to write empty cache")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
