"""Tokenize ProteinGym WT + mutant sequences with a chosen tokenizer.

Reads atom14 frames written by :mod:`scripts.build_pg_atom14`, builds
the (L, 4, 3) N/CA/C/O backbone, and tokenizes each variant. Writes a
single ``.npz`` keyed by ``<assay>__<sha1(seq)[:16]>`` so downstream
PG scoring can look up either WT or mutant tokens by hash.

For ``ours/combined/ESM3`` (the production Ensembits tokenizer):

    python -m scripts.build_pg_token_cache \\
        --tokenizer ensembits --model ckpt/combined_esm3 \\
        --atom14-dir data/proteingym_esmfold_atom14 \\
        --out data/proteingym/pg_tokens.combined_esm3.npz

For baseline tokenizers (aminoaseed / esm3struct / protoken / 3di_tokens /
vote_3di), same CLI with ``--tokenizer`` switched. ProteinGym variants
are single-chain by construction so ``chain_split`` is a no-op.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SUPPORTED = ("ensembits", "aminoaseed", "esm3struct", "protoken",
             "3di_tokens", "vote_3di")


def _hash_seq(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _atom14_to_bb(a14: np.ndarray) -> np.ndarray:
    """ESMFold atom14 ``(L, 14, 3)`` → ``(1, L, 4, 3)`` ``[N, CA, C, O]``.

    OpenFold atom14 slot ordering (verified by C–O bond length on
    ESMFold output): ``[N=0, CA=1, C=2, O=3, CB=4, ...]``. Slot 4 is
    **junk for GLY** (no real Cβ) — irrelevant here since we only pull
    backbone atoms.

    This is the real-backbone path for the ProteinGym task: every
    structural tokenizer downstream sees ESMFold's predicted N/CA/C/O
    directly, not an ideal-geometry reconstruction from Cα.
    """
    return np.stack([a14[:, 0], a14[:, 1], a14[:, 2], a14[:, 3]],
                    axis=1)[None]


def _enum_assays(atom14_dir: Path, limit: int = 0
                  ) -> list[tuple[str, list[Path]]]:
    """Return ``[(assay, [npy_path, ...]), ...]`` over the atom14 dir."""
    assays = sorted(p for p in atom14_dir.iterdir() if p.is_dir())
    if limit:
        assays = assays[:limit]
    return [(a.name, sorted(a.glob("*.npy"))) for a in assays]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True, choices=SUPPORTED)
    ap.add_argument("--atom14-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)

    # Ensembits
    ap.add_argument("--model", type=Path,
                    help="(ensembits) ckpt dir with best.pt/config.json/stats.npz")

    # AminoAseed
    ap.add_argument("--aminoaseed-repo", type=Path)
    ap.add_argument("--aminoaseed-ckpt", type=Path)

    # ESM3Struct
    ap.add_argument("--esm3-weights", type=Path, default=None)

    # ProToken
    ap.add_argument("--protoken-repo", type=Path)
    ap.add_argument("--protoken-device", default="0")

    ap.add_argument("--assay-limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"[skip] {args.out} exists (use --force)")

    # Load tokenizer
    if args.tokenizer == "ensembits":
        if not args.model:
            raise SystemExit("--model required for --tokenizer ensembits")
        from ensembits import load_model, tokenize_ensemble
        ens = load_model(args.model, device=args.device)
        def tokenize(bb_one: np.ndarray) -> np.ndarray:
            # (1, L, 4, 3) → drop O slot for ESM3 descriptor; pass (P=1, L, 3, 3)
            bb_ncac = np.stack([bb_one[0, :, 0], bb_one[0, :, 1], bb_one[0, :, 2]],
                                axis=1)[None]
            ca = bb_one[0, :, 1, :][None]
            return tokenize_ensemble(ens, ca, bb_all=bb_ncac)
    elif args.tokenizer == "aminoaseed":
        from baselines.aminoaseed import AminoAseed
        if not (args.aminoaseed_repo and args.aminoaseed_ckpt):
            raise SystemExit("--aminoaseed-repo + --aminoaseed-ckpt required")
        tok = AminoAseed(weights_path=args.aminoaseed_ckpt,
                          repo_path=args.aminoaseed_repo,
                          device=args.device or "cuda")
        def tokenize(bb_one: np.ndarray) -> np.ndarray:
            return tok.tokenize(bb_one[0, :, 1, :], bb=bb_one[0])
    elif args.tokenizer == "esm3struct":
        from baselines.esm3struct import ESM3Struct
        tok = ESM3Struct(weights_path=args.esm3_weights,
                          device=args.device or "cuda")
        def tokenize(bb_one: np.ndarray) -> np.ndarray:
            return tok.tokenize(bb_one[0, :, 1, :], bb=bb_one[0])
    elif args.tokenizer == "protoken":
        from baselines.protoken import ProToken
        if not args.protoken_repo:
            raise SystemExit("--protoken-repo required")
        tok = ProToken(repo_path=args.protoken_repo, device=args.protoken_device)
        def tokenize(bb_one: np.ndarray) -> np.ndarray:
            L = bb_one.shape[1]
            return tok.tokenize(bb_one[0, :, 1, :], "X" * L, bb=bb_one[0])
    elif args.tokenizer == "3di_tokens":
        from baselines.mini3di import three_di_tokens
        def tokenize(bb_one: np.ndarray) -> np.ndarray:
            return three_di_tokens(bb_one[0, :, 1, :], bb=bb_one[0])
    elif args.tokenizer == "vote_3di":
        from baselines.mini3di import vote_3di
        def tokenize(bb_one: np.ndarray) -> np.ndarray:
            return vote_3di(bb_one[:, :, 1, :], bb_K=bb_one)
    else:
        raise SystemExit(f"unhandled tokenizer: {args.tokenizer}")

    assays = _enum_assays(args.atom14_dir, args.assay_limit)
    n_seqs = sum(len(seqs) for _, seqs in assays)
    print(f"[1/2] {len(assays)} assays, {n_seqs} sequences to tokenize")

    out: dict[str, np.ndarray] = {}
    t0 = time.time(); n_ok = n_skip = 0
    for ai, (assay, seq_paths) in enumerate(assays):
        for sp in seq_paths:
            key = f"{assay}__{sp.stem}"
            try:
                a14 = np.load(sp).astype(np.float32)
                bb_one = _atom14_to_bb(a14)
                out[key] = tokenize(bb_one)
                n_ok += 1
            except Exception as exc:
                print(f"  [skip] {key}: {type(exc).__name__}: {exc}")
                n_skip += 1
        if (ai + 1) % 5 == 0:
            print(f"  [{ai+1}/{len(assays)}] {assay} done; ok={n_ok} skip={n_skip} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[2/2] DONE: ok={n_ok}, skip={n_skip}, {time.time()-t0:.0f}s")
    if not out:
        raise SystemExit("[abort] no sequences tokenized")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"  wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
