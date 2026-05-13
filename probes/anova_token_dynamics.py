"""Token-Dynamics ANOVA on the per-residue local-Kabsch s_1.

Computes η² = SS_between / SS_total of `s_1` (per-residue local
fluctuation amplitude, see `per_residue_local_pca.py`) grouped by
discrete token labels — a one-way ANOVA effect-size that quantifies
how well a tokenizer's vocabulary explains residue-scale dynamics.

The η² statistic here is descriptive, not inferential: residues
within a protein are not independent, so the F-distribution does not
apply. Treat η² as an effect-size ranking, complementary to the
out-of-sample Spearman the RMSF probe reports.

Filtering:
  --min-count 80 (default, paper setting): for each tokenizer, only
  count residues whose token id has ≥80 assignments in the cache.
  This drops near-empty codes whose mean is dominated by 1-2 outliers.

  Residues that have a zero `s_1` (= rows the local-Kabsch helper
  skipped because the 10 Å Cα ball had <4 atoms) are filtered out
  before computing η². Pass --keep-zero-s1 to disable this.

CLI usage:
    python -m probes.anova_token_dynamics \\
        --s1     data/s1_per_residue.npy   \\
        --tokens ensembits=runs/ensembits/tokens.npy \\
        --tokens 3di_tokens=data/3di_tokens.npy \\
        --tokens vote_3di=data/vote_3di.npy \\
        --out    runs/token_dynamics.json

Each `--tokens NAME=PATH` pair must be a 1-D int64 cache aligned
row-by-row with `--s1`. (For categorical baselines only — the
ProtProfileMD continuous histogram baseline uses an OLS R² rather
than η²; see the README.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def eta_squared(values: np.ndarray, groups: np.ndarray,
                min_count: int = 80) -> Tuple[float, int]:
    """One-way ANOVA effect size: η² = SS_between / SS_total.

    Args:
        values:    (N,) float — the response variable (e.g. s_1).
        groups:    (N,) int   — discrete group labels (token ids).
        min_count: drop groups with fewer than this many members
                   (default 80, matches the paper). Residues whose
                   token is dropped are excluded from both SS_between
                   and SS_total so the ratio is on a common subset.

    Returns:
        (eta2, M) where M is the number of groups passing min_count.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if values.shape != groups.shape:
        raise ValueError(f"shape mismatch: values {values.shape}, "
                         f"groups {groups.shape}")
    counts = np.bincount(groups[groups >= 0])
    keep_codes = np.where(counts >= min_count)[0]
    keep = np.isin(groups, keep_codes)
    v = values[keep]
    g = groups[keep]
    if v.size == 0:
        return float("nan"), 0
    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    if ss_total == 0.0:
        return float("nan"), int(keep_codes.size)
    # SS_between = sum_g n_g * (mean_g - grand)^2
    ss_between = 0.0
    for code in keep_codes:
        m = v[g == code]
        if m.size == 0:
            continue
        ss_between += m.size * (m.mean() - grand) ** 2
    return float(ss_between / ss_total), int(keep_codes.size)


def _parse_tokens_arg(s: str) -> Tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--tokens expects NAME=PATH, got {s!r}")
    name, path = s.split("=", 1)
    return name.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s1", required=True, type=Path,
                     help="(N,) float32 cache of per-residue s_1.")
    ap.add_argument("--tokens", required=True, action="append",
                     type=_parse_tokens_arg, metavar="NAME=PATH",
                     help="Tokenizer name and path to its (N,) int64 cache. "
                          "Repeat for multiple tokenizers.")
    ap.add_argument("--out", required=True, type=Path,
                     help="JSON output: list of {tokenizer, n_residues, n_tokens, eta2}.")
    ap.add_argument("--min-count", type=int, default=80,
                     help="Min residues per token-id to include (default 80).")
    ap.add_argument("--keep-zero-s1", action="store_true",
                     help="By default rows with s_1 == 0 (skipped by the local-Kabsch "
                          "helper because the 10 Å ball had <4 atoms) are filtered out. "
                          "Pass this flag to include them.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    s1 = np.load(args.s1).astype(np.float64)
    print(f"[token_dynamics] s1: {s1.shape[0]:,} residues  "
          f"(mean={s1.mean():.3f}, n_zero={(s1 == 0).sum():,})")
    if not args.keep_zero_s1:
        valid = s1 > 0
        print(f"[token_dynamics] filtering out {(~valid).sum():,} rows with s_1 == 0")
    else:
        valid = np.ones_like(s1, dtype=bool)
    s1_v = s1[valid]

    rows = []
    for name, path in args.tokens:
        toks = np.load(path).astype(np.int64)
        if toks.shape[0] != s1.shape[0]:
            print(f"[token_dynamics] {name}: shape mismatch "
                   f"({toks.shape[0]:,} vs s1 {s1.shape[0]:,}), skipping",
                   file=sys.stderr)
            continue
        eta2, M = eta_squared(s1_v, toks[valid], min_count=args.min_count)
        rows.append({
            "tokenizer":  name,
            "n_residues": int(valid.sum()),
            "n_tokens":   M,
            "eta2":       eta2,
        })
        print(f"  {name:<32}  n={int(valid.sum()):>7,}  M={M:>5d}  η² = {eta2:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "min_count": args.min_count,
        "keep_zero_s1": bool(args.keep_zero_s1),
        "rows": rows,
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
