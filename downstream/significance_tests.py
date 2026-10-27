"""Significance tests of Ensembits against the static tokenizers (ProteinGym, binding site).

What it does
------------
  * ProteinGym: pairs the per-assay Spearman correlations of Ensembits and of each
    baseline over the shared DMS assays and tests the per-assay differences with a
    paired t-test and a two-sided Wilcoxon signed-rank test, for the score alone and
    for the z-blend with ESM2-650M (alpha = 0.3); also Ensembits + ESM2 vs ESM2 alone.
  * MISATO binding site (one split): compares the per-seed test AUROC of Ensembits
    and of each baseline (10 training seeds each) with a two-sided Welch t-test and a
    two-sided Mann-Whitney U test.
  * Within each table the p-values are also adjusted for multiple comparisons
    (Holm), reported next to the raw values.
Writes a markdown report and a JSON file.

Paper
-----
Appendix section on significance (to be added; the tests were reported in the
rebuttal to Reviewer szcw). Inputs are the results behind `2_results.tex`
Table tab:proteingym_ensemble and the binding-site columns of the function tables.

Origin
------
New (2026-10-15). The rebuttal numbers (`rebuttal/response_significance.md`) were
computed without a script; this reproduces that analysis and adds the Holm column.
Fixes 2026-10-16 (FIXED markers): --variant / --pg-tag required, --baseline-tag
added; the imported compiler module no longer runs at import (code_fix P19).

Used by the reported model: evaluation only.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   output/proteingym/per_assay_pg_<pg-tag>[/ | _shard*of* | .json]   (Ensembits)
          ensembits-repro/runs/proteingym/<baseline>.json                     (released baselines)
          output/binding_misato/<variant>_lig_stab[_seed<s>].json             (Ensembits)
          output/binding_misato/baseline_<F>_lig_stab[_seed<s>].json          (baselines)
  writes  output/significance/<pg-tag>__<variant>.{md,json}

How to run
----------
    python downstream/significance_tests.py --pg-tag <tag> --variant <checkpoint dir name> \\
        [--baseline-tag stab] [--split structure] [--out-dir <dir>]
(--pg-tag and --variant are required since 2026-10-16, code_fix P19.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT as ROOT          # noqa: E402
# FIXED(code_fix P19, 2026-10-16): old: this import ran the whole compile of
#   _compile_rebuttal_results.py (it rewrote a results file). New: that module's body is
#   under main(); only these pure helpers are imported.
from _compile_rebuttal_results import pg_load, tag_of  # noqa: E402  same readers as the tables

PG_BASELINES = {"ESM3-struct": "esm3struct", "AminoAseed": "aminoaseed",
                "ProToken": "protoken", "3Di": "3di_tokens"}       # ensembits-repro run names
BIND_BASELINES = {"ESM3-struct": "esm3struct", "AminoAseed": "aminoaseed",
                  "ProToken": "protoken", "3Di": "3di_tokens"}     # baseline_<F>_lig_stab files
BD = ROOT / "output" / "binding_misato"
TAG = "stab"          # recipe tag of the Ensembits binding files (_run_model_battery.py)
ALPHA_KEY = "a03"                                                  # z-blend alpha = 0.3


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def pg_tests(pg_tag: str) -> list[dict]:
    """Paired tests over the shared ProteinGym assays."""
    # FIXED(benchmark_audit D9, 2026-10-19): camera-ready paired tests (CV α, rank-sum,
    #   −n_mut, Holm) are in pg_aggregate.py. This file is kept for reference only.
    ens = pg_load(pg_tag)
    if not ens:
        raise SystemExit(f"no per-assay ProteinGym results for tag {pg_tag!r}")
    e_alone = {a: r.get("spearman_score_a_allL0") for a, r in ens.items()}
    e_mix = {a: r.get(f"spearman_zmix_a_{ALPHA_KEY}") for a, r in ens.items()}
    esm2 = {a: r.get("spearman_esm2") for a, r in ens.items()}
    rows = []

    def paired(name, x: dict, y: dict, what: str):
        keys = sorted(k for k in x if k in y and x[k] is not None and y[k] is not None
                      and np.isfinite(x[k]) and np.isfinite(y[k]))
        d = np.array([x[k] - y[k] for k in keys])
        t = stats.ttest_rel([x[k] for k in keys], [y[k] for k in keys])
        w = stats.wilcoxon(d, alternative="two-sided")
        rows.append({"comparison": name, "score": what, "n_assays": len(keys),
                     "delta": float(d.mean()), "p_ttest": float(t.pvalue),
                     "p_wilcoxon": float(w.pvalue)})

    paired("Ensembits + ESM2 vs ESM2 alone", e_mix, esm2, "+ESM2")
    for disp, run in PG_BASELINES.items():
        j = json.load(open(ROOT / "ensembits-repro" / "runs" / "proteingym" / f"{run}.json"))
        b_alone = {r["assay"]: r.get("spearman_ours") for r in j["per_assay"]}
        b_mix = {r["assay"]: r.get(f"zmix_{ALPHA_KEY}") for r in j["per_assay"]}
        paired(f"vs {disp}", e_alone, b_alone, "alone")
        paired(f"vs {disp}", e_mix, b_mix, "+ESM2")
    adj = holm([r["p_ttest"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_ttest_holm"] = a
    return rows


def _seed_values(path_of_seed, split: str, task: str, metric: str) -> list[float]:
    # FIXED(benchmark_audit D3, 2026-10-18): a binding JSON not built on the current MISATO split
    #   file (data/misato_splits_v2.json; data_versions.misato_split_problem) is refused.
    from ensembits.data_versions import misato_split_problem
    out = []
    for s in range(10):
        f = path_of_seed(s)
        if f.exists():
            d = json.load(open(f))
            why = misato_split_problem(d)
            if why:
                raise SystemExit(f"{f.name}: {why} (rerun on data/misato_splits_v2.json)")
            # FIXED(benchmark_audit D4, 2026-10-18): binding results not built on the current
            #   binding-site labels (misato_binding_site_v2.npz) are refused.
            if task == "binding":
                from ensembits.misato_binding_labels import binding_label_problem
                why = binding_label_problem(d)
                if why:
                    raise SystemExit(f"{f.name}: {why} (rerun on misato_binding_site_v2.npz)")
            c = d.get("results", {}).get(f"{split}_{task}", {})
            if c.get(metric) is not None:
                out.append(float(c[metric]))
    return out


def bind_tests(variant: str, split: str, baseline_tag: str = TAG) -> list[dict]:
    """Two-sample tests over training seeds (binding-site AUROC).

    baseline_tag: recipe tag of the baseline files baseline_<F>_lig_<tag>[_seed<s>].json.
    """
    sd = lambda s: "" if s == 0 else f"_seed{s}"
    ens = _seed_values(lambda s: BD / f"{tag_of(variant)}_lig_{TAG}{sd(s)}.json",
                       split, "binding", "auroc")
    if len(ens) < 2:
        raise SystemExit(f"fewer than 2 binding seeds for {variant!r}")
    rows = []
    for disp, F in BIND_BASELINES.items():
        b = _seed_values(lambda s: BD / f"baseline_{F}_lig_{baseline_tag}{sd(s)}.json",
                         split, "binding", "auroc")
        if len(b) < 2:
            rows.append({"comparison": f"vs {disp}", "n_seeds": (len(ens), len(b))}); continue
        w = stats.ttest_ind(ens, b, equal_var=False)
        u = stats.mannwhitneyu(ens, b, alternative="two-sided")
        rows.append({"comparison": f"vs {disp}", "n_seeds": (len(ens), len(b)),
                     "ensembits": float(np.mean(ens)), "baseline": float(np.mean(b)),
                     "baseline_std": float(np.std(b, ddof=1)), "delta": float(np.mean(ens) - np.mean(b)),
                     "p_welch": float(w.pvalue), "p_mannwhitney": float(u.pvalue)})
    ok = [r for r in rows if "p_welch" in r]
    for r, a in zip(ok, holm([r["p_welch"] for r in ok])):
        r["p_welch_holm"] = a
    return rows


def to_markdown(pg: list[dict], bind: list[dict], args) -> str:
    L = [f"# Significance tests ({args.pg_tag} / {args.variant}, binding split {args.split})", "",
         "## ProteinGym (paired over assays)", "",
         "| comparison | score | n | Δ | p (paired t) | p (Wilcoxon) | p (t, Holm) |",
         "|---|---|---|---|---|---|---|"]
    for r in pg:
        L.append(f"| {r['comparison']} | {r['score']} | {r['n_assays']} | {r['delta']:+.4f} | "
                 f"{r['p_ttest']:.2g} | {r['p_wilcoxon']:.2g} | {r['p_ttest_holm']:.2g} |")
    L += ["", f"## MISATO binding site, {args.split} split (seeds)", "",
          "| comparison | baseline AUROC | Δ | p (Welch) | p (Mann–Whitney) | p (Welch, Holm) |",
          "|---|---|---|---|---|---|"]
    for r in bind:
        if "p_welch" not in r:
            L.append(f"| {r['comparison']} | missing seeds {r['n_seeds']} | | | | |"); continue
        L.append(f"| {r['comparison']} | {r['baseline']:.4f} ± {r['baseline_std']:.4f} | "
                 f"{r['delta']:+.4f} | {r['p_welch']:.2g} | {r['p_mannwhitney']:.2g} | "
                 f"{r['p_welch_holm']:.2g} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # FIXED(code_fix P19, 2026-10-16): old: defaults = the current (pre-fix) checkpoint's
    #   tags (dih256_fapeH_bioemu_misato / ..._reviewer_3_special_corpus_MISATO). New: both
    #   required, so a run cannot silently test the old model; --baseline-tag added.
    ap.add_argument("--pg-tag", required=True, help="ProteinGym tag of the Ensembits model")
    ap.add_argument("--variant", required=True, help="output/<variant> of the Ensembits model")
    ap.add_argument("--baseline-tag", default=TAG,
                    help="recipe tag of the baseline binding files (default: %(default)s)")
    ap.add_argument("--split", default="structure", choices=("sequence", "structure", "random"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "significance"))
    args = ap.parse_args()
    pg = pg_tests(args.pg_tag)
    bind = bind_tests(args.variant, args.split, baseline_tag=args.baseline_tag)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.pg_tag}__{tag_of(args.variant)}__{args.split}"
    md = to_markdown(pg, bind, args)
    (out / f"{stem}.md").write_text(md)
    (out / f"{stem}.json").write_text(json.dumps({"proteingym": pg, "binding": bind}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
