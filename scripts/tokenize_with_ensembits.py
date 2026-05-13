"""Tokenize a corpus (mdCATH or MISATO) with a trained Ensembits model.

Reads per-pid real backbone from ``{data_dir}/{dataset}_real_bb/<pid>.npz``,
runs the live ESM3 descriptor + RVQ encoder, and writes a per-pid token
cache that the downstream probes consume.

Includes a **cache-write guard**: if fewer than 50 % of pids tokenize
successfully, the output is *not* written. (This caught a real
regression during the protoken refix: a builder with a stale config
silently produced a near-empty cache that downstream probes accepted
and reported as 0/NaN scores.)

Examples:

    # Tokenize the shipped ours/combined/ESM3 model on misato at P=10
    python -m scripts.tokenize_with_ensembits \\
        --model ckpt/combined_esm3 \\
        --dataset misato \\
        --out data/cached_descriptors/ensembits_combined_esm3_tokens_misato.npz

    # Same model, single-frame inference (P=1) on mdcath
    python -m scripts.tokenize_with_ensembits \\
        --model ckpt/combined_esm3 \\
        --dataset mdcath --p-inference 1 \\
        --out data/cached_descriptors/ensembits_combined_esm3_P1inf_tokens_mdcath.npz

The output is a ``.npz`` keyed by pid; each value is ``(L,) int64``
tokens in ``[0, codebook_sizes[0])``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make ensembits/baselines importable when run as a script (no install).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ensembits import load_model, tokenize_ensemble        # noqa: E402
from baselines._backbone import load_bb_for_pid            # noqa: E402

MIN_OK_FRACTION = 0.5


def _iter_pids(real_bb_dir: Path) -> list[str]:
    return sorted(p.stem for p in real_bb_dir.glob("*.npz"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path,
                    help="Checkpoint dir containing best.pt + config.json + stats.npz")
    ap.add_argument("--dataset", choices=("mdcath", "misato"), required=True)
    ap.add_argument("--data-dir", type=Path,
                    default=_REPO_ROOT / "data",
                    help="Parent dir containing {dataset}_real_bb/<pid>.npz")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .npz path")
    ap.add_argument("--p-inference", type=int, default=None,
                    help="If set, tokenize using only the first P_inf frames "
                         "(otherwise use all frames from the cache)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N pids (0 = all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Re-tokenize even if --out already exists")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"[skip] {args.out} exists (use --force to overwrite)")

    real_bb_dir = args.data_dir / f"{args.dataset}_real_bb"
    if not real_bb_dir.exists():
        raise SystemExit(f"Missing real-bb cache: {real_bb_dir}\n"
                          "  Build it first with scripts/build_real_bb.py")

    print(f"[1/3] Loading model: {args.model}")
    ens = load_model(args.model, device=args.device)
    P_full = int(ens.config["num_prototypes"])
    print(f"  descriptor={ens.config.get('descriptor', 'esm3desc_K16')}  "
          f"P_train={P_full}  M0={ens.config['codebook_sizes'][0]}  "
          f"device={ens.device}")

    pids = _iter_pids(real_bb_dir)
    if args.limit:
        pids = pids[:args.limit]
    print(f"\n[2/3] Tokenizing {len(pids)} pids from {real_bb_dir}")
    if args.p_inference is not None and args.p_inference < P_full:
        print(f"  P_inference={args.p_inference} (using first "
              f"{args.p_inference} of {P_full} frames)")

    tokens: dict[str, np.ndarray] = {}
    n_skip = 0
    t0 = time.time()
    for i, pid in enumerate(pids):
        try:
            bb = load_bb_for_pid(pid, dataset=args.dataset, P=P_full,
                                 data_dir=args.data_dir)
            if bb is None:
                n_skip += 1
                continue
            if args.p_inference is not None and args.p_inference < bb.shape[0]:
                bb = bb[:args.p_inference]
            ca = bb[:, :, 1, :]
            tok = tokenize_ensemble(ens, ca, bb_all=bb)
            tokens[pid] = tok
        except Exception as exc:
            print(f"  [skip] {pid}: {type(exc).__name__}: {exc}", flush=True)
            n_skip += 1
            continue
        if (i + 1) % 500 == 0:
            ok = len(tokens)
            print(f"  {i+1}/{len(pids)}  ok={ok}  skip={n_skip}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ok_frac = len(tokens) / max(len(pids), 1)
    print(f"\n  done: {len(tokens)}/{len(pids)} pids "
          f"({ok_frac:.1%}) in {time.time()-t0:.0f}s")

    # Cache-write guard: refuse if too few pids tokenized successfully.
    if ok_frac < MIN_OK_FRACTION:
        raise SystemExit(
            f"[abort] ok_fraction={ok_frac:.1%} < {MIN_OK_FRACTION:.0%}; "
            f"refusing to write a near-empty cache to {args.out}.")

    print(f"\n[3/3] Writing token cache: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **tokens)
    print(f"  {len(tokens)} pids × ~{int(np.mean([len(t) for t in tokens.values()]))} residues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
