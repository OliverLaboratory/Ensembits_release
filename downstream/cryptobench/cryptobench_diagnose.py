"""CryptoBench: is a pooled-metric gap between alphabets localization or cross-protein calibration?

What it does
------------
Step 4 (optional) of the CryptoBench pipeline (see `README.md` in this folder).
Reads the seed-averaged per-residue test predictions that `cryptobench_probe.py`
dumps for each method and reports, on the test chains common to all methods:

  pooled           AUROC / AUPRC over all residues (the benchmark's metric)
  per-protein z    the same pooled metrics after z-scoring each chain's scores;
                   this removes every cross-protein offset/scale and keeps only
                   the ranking within a chain. An edge that survives localizes
                   better; an edge that disappears was calibration.
  per-protein AUC  mean of per-chain AUROCs (chains with both classes)
  bootstrap        95% CI of the pooled-metric DIFFERENCE between two methods,
                   1,000 resamples of whole test chains (residues within a
                   chain are not independent)
  Wilcoxon         paired per-protein AUROC test across test chains

Methods and pairs are fixed in METHODS / PAIRS; they must match the `--label`
names given to `cryptobench_probe.py`. Methods without a predictions file are
skipped.

Paper
-----
Not in the paper yet: CryptoBench goes into the main text
(`camera_ready/require_fix.md` #24, `camera_ready/code_fix.md` E3).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Benchmark: Škrhák et al., Bioinformatics 41(1):btae745, 2025.

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/cryptobench_diagnose.py`.
Changes: comments; ROOT from `ensembits.paths.REPO_ROOT`; the output folder
(originally the constant OUT = <research repo>/output/cryptobench)
now comes from `_cb_paths.py` (`--out-dir` / `--run-tag` / `--legacy-layout`),
which adds the only CLI of this script. Analysis logic, METHODS, PAIRS, SET and
NBOOT are unchanged.

Used by the reported model: not yet. Its rebuttal output
(output/cryptobench/diagnose_summary.json) is for the old flagship and the conv1d
probe; it is the basis of the "calibration, not localization" reading in
`rebuttal/cryptobench.md` §4.
# Open(code_fix E3): if the MLP-probe numbers go into the paper, run this diagnostic on
#   that probe's predictions too. The conv1d run found the AUPRC lead to be cross-protein
#   calibration (paired per-protein AUROC vs AminoAseed −0.027, p = 6.7e-4).

Inputs / outputs
----------------
Read  : `<out-dir>/preds_apo_<method>_<features>.npz` (from `cryptobench_probe.py`).
Write : `<out-dir>/diagnose_summary.json`; tables on stdout.

Dependencies left in the research repo: none (numpy, scipy, scikit-learn).

How to run
----------
  $PY downstream/cryptobench/cryptobench_diagnose.py                    # current run tag
  $PY downstream/cryptobench/cryptobench_diagnose.py --legacy-layout    # rebuttal outputs (overwrites diagnose_summary.json there)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402,F401  data/ and output/ stay in the research repo
import _cb_paths as CBP  # noqa: E402

OUT = None   # set in main(); originally ROOT / "output" / "cryptobench"
# (label given to cryptobench_probe.py, its --features)
METHODS = [("ensembits", "tokens"), ("aminoaseed", "tokens"), ("esm3struct", "tokens"),
           ("protoken", "tokens"), ("mini3di", "tokens"), ("aa_onehot", "aa"),
           ("random_tok", "random")]
# Not added (code_fix C13): ("aminoaseed_ourdata", "tokens"), ("esm3style_ourdata", "tokens")
#   once those rows are probed.
# every comparison the rebuttal would need to defend: ours vs each baseline
PAIRS = [("ensembits", "aminoaseed"), ("ensembits", "esm3struct"),
         ("ensembits", "protoken"), ("ensembits", "mini3di"),
         ("ensembits", "aa_onehot"), ("aminoaseed", "esm3struct")]
SET = "apo"
NBOOT = 1000


def load(name, feat):
    """Seed-averaged predictions and labels per test chain of one method."""
    d = np.load(OUT / f"preds_{SET}_{name}_{feat}.npz")
    keys = sorted(k[3:] for k in d.files if k.startswith("p::"))
    return ({k: d[f"p::{k}"] for k in keys}, {k: d[f"y::{k}"] for k in keys})


def z(v):
    """z-score one chain's scores (constant scores -> all zeros)."""
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v * 0.0


def metrics(P, Y, keys):
    y = np.concatenate([Y[k] for k in keys])
    p = np.concatenate([P[k] for k in keys])
    pz = np.concatenate([z(P[k]) for k in keys])
    per = [roc_auc_score(Y[k], P[k]) for k in keys if 0 < Y[k].sum() < len(Y[k])]
    return {"auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "auc_z": roc_auc_score(y, pz), "auprc_z": average_precision_score(y, pz),
            "auc_per_protein": float(np.mean(per))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    CBP.add_path_args(ap)
    args = ap.parse_args()
    global OUT
    OUT = CBP.resolve_paths(args).out_dir

    data = {}
    for n, f in METHODS:
        try:
            data[n] = load(n, f)
        except FileNotFoundError:
            print(f"  (skip {n}: no preds dump)")
    # only chains predicted by every available method
    keys = sorted(set.intersection(*[set(d[0]) for d in data.values()]))
    ref = next(iter(data))
    nres = sum(len(data[ref][1][k]) for k in keys)
    npos = sum(int(data[ref][1][k].sum()) for k in keys)
    print(f"test chains={len(keys)}  residues={nres}  positives={npos} "
          f"({100*npos/nres:.2f}%)\n")

    print(f"{'method':<12}{'AUC':>8}{'AUPRC':>8} | {'AUC(z)':>8}{'AUPRC(z)':>10} | {'perProtAUC':>11}")
    print(f"{'':<12}{'--- pooled ---':>16} | {'-- within-protein only --':>19} | ")
    M = {}
    for n, f in METHODS:
        if n not in data:
            continue
        P, Y = data[n]
        M[n] = metrics(P, Y, keys)
        m = M[n]
        print(f"{n:<12}{m['auc']:>8.4f}{m['auprc']:>8.4f} | "
              f"{m['auc_z']:>8.4f}{m['auprc_z']:>10.4f} | {m['auc_per_protein']:>11.4f}")

    # Chain-level bootstrap of the pooled-metric differences.
    rng = np.random.default_rng(0)
    idx = np.arange(len(keys))
    print("\nbootstrap over test chains (n=%d resamples), pooled-metric differences:" % NBOOT)
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        Pa, Ya = data[a]
        Pb, _ = data[b]
        d_auc, d_ap, d_apz = [], [], []
        for _ in range(NBOOT):
            s = [keys[i] for i in rng.choice(idx, len(idx), replace=True)]
            y = np.concatenate([Ya[k] for k in s])
            if y.sum() == 0 or y.sum() == len(y):
                continue
            pa = np.concatenate([Pa[k] for k in s])
            pb = np.concatenate([Pb[k] for k in s])
            paz = np.concatenate([z(Pa[k]) for k in s])
            pbz = np.concatenate([z(Pb[k]) for k in s])
            d_auc.append(roc_auc_score(y, pa) - roc_auc_score(y, pb))
            d_ap.append(average_precision_score(y, pa) - average_precision_score(y, pb))
            d_apz.append(average_precision_score(y, paz) - average_precision_score(y, pbz))
        def ci(v, label):
            v = np.asarray(v)
            lo, hi = np.percentile(v, [2.5, 97.5])
            sig = "significant" if lo > 0 or hi < 0 else "n.s."
            print(f"   {label:<22} Δ={v.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
        print(f"  {a} - {b}:")
        ci(d_auc, "pooled AUC")
        ci(d_ap, "pooled AUPRC")
        ci(d_apz, "AUPRC, per-protein z")

    # paired within-protein test (localization), the unit reviewers would ask for
    print("\npaired per-protein AUC (localization) across test chains:")
    for a, b in PAIRS:
        if a not in data or b not in data:
            continue
        Pa, Ya = data[a]
        Pb, _ = data[b]
        ks = [k for k in keys if 0 < Ya[k].sum() < len(Ya[k])]
        xa = np.array([roc_auc_score(Ya[k], Pa[k]) for k in ks])
        xb = np.array([roc_auc_score(Ya[k], Pb[k]) for k in ks])
        w = wilcoxon(xa, xb)
        print(f"   {a} vs {b}: n={len(ks)} Δ={xa.mean()-xb.mean():+.4f} "
              f"wins={int((xa>xb).sum())}/{len(ks)} Wilcoxon p={w.pvalue:.3g}")

    json.dump({"set": SET, "n_chains": len(keys), "n_res": nres, "n_pos": npos,
               "metrics": {k: {m: float(v) for m, v in d.items()} for k, d in M.items()}},
              open(OUT / "diagnose_summary.json", "w"), indent=1)
    print(f"\nwrote {OUT}/diagnose_summary.json")


if __name__ == "__main__":
    main()
