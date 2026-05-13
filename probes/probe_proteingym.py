"""Zero-shot mutation-effect scoring on ProteinGym.

For each (wild-type, variant) pair, you bring pre-computed token IDs
+ a codebook, and this script computes the sum-of-distances disruption
score, Spearman of `-disruption` vs DMS, and optional ESM2-blend
metrics.

Inputs expected per assay:

    --wt-tokens    .npz  {assay -> (L,) int64}        WT token sequence
    --mut-tokens   .npz  {assay -> {mut_str: (L,) int64}}  per-variant tokens
    --codebook     .npy  (M, d) float32               primary codebook
    --dms          .json {assay -> {mut_str: float}}  experimental DMS scores
    --esm2         .json {assay -> {mut_str: float}}  optional ESM2 pseudo-LL
                                                       for the z-mix blend

Run once per tokenizer to produce a per-assay summary JSON. The
`scoring` formula (assuming WT and mutant tokens of the same length L):

    disruption(v) = Σ_i ||C[wt_i] - C[mut_i]||_2          # length-L pairs
    score(v)      = -disruption(v)                         # higher = more WT-like
    Spearman_v    = spearmanr(score, dms).correlation       # per assay

The optional `--esm2` input adds within-assay z-score blends at α
∈ {0.1, 0.2, 0.3, 0.4, 0.5} and a parameter-free rank-sum.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(x.std(), 1e-12)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wt-tokens", required=True, type=Path,
                     help=".npz  {assay: (L,) int64}")
    ap.add_argument("--mut-tokens", required=True, type=Path,
                     help=".npz  {assay__mut: (L,) int64}; the inner key may "
                          "be packed into a single dotted string")
    ap.add_argument("--codebook", required=True, type=Path,
                     help="(M, d) .npy primary codebook")
    ap.add_argument("--dms", required=True, type=Path,
                     help=".json {assay: {mut_str: float}}")
    ap.add_argument("--esm2", default=None, type=Path,
                     help=".json {assay: {mut_str: float}} ESM2-650M pseudo-LL")
    ap.add_argument("--out", required=True, type=Path,
                     help="output JSON: per-assay Spearman + aggregate")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cb = np.load(args.codebook).astype(np.float32)               # (M, d)
    wt_z = np.load(args.wt_tokens, allow_pickle=False)
    mut_z = np.load(args.mut_tokens, allow_pickle=False)
    dms = json.loads(args.dms.read_text())
    esm2 = json.loads(args.esm2.read_text()) if args.esm2 else {}

    # mut_tokens is stored flat as `{f"{assay}__{mut}": (L,) int}`. Group by assay.
    mut_by_assay: dict[str, dict[str, np.ndarray]] = {}
    for key in mut_z.files:
        if "__" not in key:
            continue
        assay, mut = key.split("__", 1)
        mut_by_assay.setdefault(assay, {})[mut] = mut_z[key]

    rows = []
    for assay in sorted(set(wt_z.files) & set(mut_by_assay) & set(dms)):
        wt = np.asarray(wt_z[assay], dtype=np.int64)
        wt_emb = cb[wt]                                             # (L, d)
        muts = mut_by_assay[assay]
        names = []
        scores = []
        true = []
        e2 = []
        for mut, tok in muts.items():
            if mut not in dms[assay]:
                continue
            mut_emb = cb[np.asarray(tok, dtype=np.int64)]
            if mut_emb.shape != wt_emb.shape:
                continue
            disrupt = float(np.linalg.norm(mut_emb - wt_emb, axis=-1).sum())
            names.append(mut)
            scores.append(-disrupt)                                 # higher = more WT-like
            true.append(float(dms[assay][mut]))
            if esm2 and mut in esm2.get(assay, {}):
                e2.append(float(esm2[assay][mut]))
            else:
                e2.append(np.nan)
        if len(scores) < 10:
            continue
        scores = np.asarray(scores); true = np.asarray(true)
        e2 = np.asarray(e2)
        row = {
            "assay": assay, "n_variants": int(len(scores)),
            "spearman_ours": float(spearmanr(scores, true).correlation),
        }
        if np.isfinite(e2).all():
            row["spearman_esm2"] = float(spearmanr(e2, true).correlation)
            zo = _zscore(scores); ze = _zscore(e2)
            for a in (0.1, 0.2, 0.3, 0.4, 0.5):
                row[f"zmix_a{int(a*10):02d}"] = float(
                    spearmanr(a * zo + (1 - a) * ze, true).correlation)
            row["ranksum"] = float(spearmanr(rankdata(scores) + rankdata(e2),
                                               true).correlation)
        rows.append(row)

    if not rows:
        raise SystemExit("no assays produced a Spearman row")

    def _mean(key: str) -> float | None:
        vs = [r[key] for r in rows if isinstance(r.get(key), float)]
        return float(np.mean(vs)) if vs else None

    summary = {
        "n_assays": len(rows),
        "mean_spearman_ours": _mean("spearman_ours"),
        "mean_spearman_esm2": _mean("spearman_esm2"),
        "median_spearman_ours": float(np.median(
            [r["spearman_ours"] for r in rows])),
    }
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        summary[f"mean_zmix_a{int(a*10):02d}"] = _mean(f"zmix_a{int(a*10):02d}")
    summary["mean_ranksum"] = _mean("ranksum")
    summary["per_assay"] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[done] {args.out}  mean Spearman (ours) = "
          f"{summary['mean_spearman_ours']}  ({len(rows)} assays)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
