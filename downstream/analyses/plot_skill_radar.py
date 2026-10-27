"""Skill-score radar of the MISATO downstream tasks across the three splits, for one Ensembits variant.

What it does
------------
For each (split, task, metric) it computes
    skill = (score - random_floor) / (esm3struct_ceiling - random_floor)
where random_floor and esm3struct_ceiling come from the SAME split, so 0 = no
better than random tokens on that split and 1 = matches ESM3struct there.
Scores are the mean over the seeds found (0-9) of the result JSONs written by the
downstream battery, read with the file naming of `downstream/_compile_rebuttal_results.py`
(recipe tag `--tag`, default "stab"): EC (depth 1-3, micro-AP and top-1 hit),
GO top-50 (micro-AP, top-1 hit), binding site (AUROC, AP) and binding affinity
(Spearman). The two Ensembits rows (P = full and P = 1) come from `--variant`.
Draws one polar panel per split and writes PDF, PNG and a CSV of raw and skill
values; prints the number of seeds found per cell.

Paper
-----
  * `2_results.tex`, Figure fig:downstream_radar (`img/downstreams.pdf`):
    "Skill-score radar across MISATO downstream tasks ... skill(m) =
    (score(m) - score(Random)) / (score(ESM3struct) - score(Random)) ... AA and
    Random rows are excluded from the polygons".
  * The numbers are those of the function tables (`appendix.tex`
    \\label{app:downstreams}).

Origin
------
Copied from `submission_exp/src/_plot_skill_radar.py`. Changes, each marked
FIXED(code_fix P20, 2026-10-16):
  * the Ensembits variant comes from `--variant` (model dir or name) instead of the
    hard-coded old tag `combined_mdcath_misato_P10_esm3desc_K16_..._realbb`;
  * the result files are read with the camera-ready naming (`_<tag>` recipe
    suffix, seeds `_seed<s>` / `__seed<s>`, EC `__summary.json` with `by_depth`),
    i.e. the same files `_compile_rebuttal_results.py` compiles into the tables;
    the old naming (no recipe tag, per-depth EC files) is not read any more;
  * output folder from `--out-dir` (default REPO_ROOT/submission_exp/figs) and the
    file stem from `--stem`; the output folder is created in main(), not at import;
  * per-cell seed counts are printed and stored in the CSV, which is written
    with the csv module (labels with commas are quoted).
Skill computation and plotting unchanged.

Used by the reported model: no longer. Superseded (final_fix_list A3, authors' decision
2026-10-19) by `plot_function_dotplot.py`, which reads the CV cells through the compiler's
checked loaders; this script is kept for reference only (it reads no CV file and checks no
protocol).

Inputs / outputs
----------------
  reads   output/ec_conv1d_misato/*__summary.json, output/go_conv1d_misato/go_*_top50*.json,
          output/binding_misato/*_lig_<tag>*.json
  writes  <out-dir>/<stem>.pdf, .png, .csv

How to run
----------
    python downstream/analyses/plot_skill_radar.py --variant output/<retrained_variant>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402

# Superseded(final_fix_list A3, 2026-10-19): this radar was replaced by a dot plot
#   (downstream/analyses/plot_function_dotplot.py). This script is kept for reference only.
EC_DIR   = ROOT / "output" / "ec_conv1d_misato"
GO_DIR   = ROOT / "output" / "go_conv1d_misato"
BIND_DIR = ROOT / "output" / "binding_misato"
OUT_DIR  = ROOT / "submission_exp" / "figs"
N_SEEDS  = 10

SPLITS = ["sequence", "structure", "random"]

ENS_P1, ENS_FULL = "Ensembits, P=1", "Ensembits, P=full"
# display_label, baseline file stem (None = Ensembits rows, filled from --variant)
MODELS = [
    # Random + aa are kept here (Random is the skill-score floor; aa is a useful
    # reference for the per-split CSV) but excluded from PLOT_MODELS so they
    # don't clutter the radar.
    ("Random",            "random"),
    ("aa",                "aa"),
    ("AminoAseed",        "aminoaseed"),
    ("ProToken",          "protoken"),
    ("3Di_tokens",        "3di_tokens"),
    ("ProtProfileMD_K8",  "protprofile_K8"),
    ("Vote_3Di",          "vote_3di"),
    ("ESM3struct",        "esm3struct"),
    (ENS_P1,              None),
    (ENS_FULL,            None),
]
PLOT_MODELS = [
    "AminoAseed", "ProToken", "3Di_tokens", "ProtProfileMD_K8",
    "Vote_3Di", "ESM3struct", ENS_P1, ENS_FULL,
]

AXES = [
    ("EC d1\nμAP",          "ec1", "micro_ap"),
    ("EC d1\ntop-1",        "ec1", "top1_hit"),
    ("EC d2\nμAP",          "ec2", "micro_ap"),
    ("EC d2\ntop-1",        "ec2", "top1_hit"),
    ("EC d3\nμAP",          "ec3", "micro_ap"),
    ("EC d3\ntop-1",        "ec3", "top1_hit"),
    ("GO top-50\nμAP",      "go",  "micro_ap"),
    ("GO top-50\ntop-1",    "go",  "top1_hit"),
    ("Binding-Site\nAUROC", "bs",  "auroc"),
    ("Binding-Site\nAP",    "bs",  "ap"),
    ("Binding-Affinity\nSpearman", "ba", "spearman"),
]


def tag_of(v: str) -> str:
    """Variant name as used in result file names (same as _compile_rebuttal_results.tag_of)."""
    return v.replace("final_model_", "").replace("final_model", "") or v


# ---------------------------------------------------------------- readers
# FIXED(code_fix P20, 2026-10-16): file names as in downstream/_compile_rebuttal_results.py
#   (m_ec / m_go / m_bind for Ensembits, b_ec / b_go / b_bind for baselines), but
#   returning any metric of the JSON, so the radar shows the same runs as the tables.

def _sd_ec(s): return "" if s == 0 else f"_seed{s}"
def _sd_bl(s): return "" if s == 0 else f"__seed{s}"


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def ens_files(kind, V, p1, split, tag):
    pih = "_P1inf" if p1 else ""
    spf = f"_{split}" if split != "sequence" else ""
    if kind.startswith("ec"):
        return [EC_DIR / f"tok_{tag_of(V)}{pih}{spf}_{tag}{_sd_ec(s)}__summary.json" for s in range(N_SEEDS)]
    if kind == "go":
        return [GO_DIR / f"go_{tag_of(V)}{pih}{spf}_{tag}{_sd_ec(s)}_top50.json" for s in range(N_SEEDS)]
    return [BIND_DIR / f"{tag_of(V)}{pih}_lig_{tag}{_sd_ec(s)}.json" for s in range(N_SEEDS)]


def base_files(kind, F, split, tag):
    if kind.startswith("ec"):
        spf = f"__{split}" if split != "sequence" else ""
        return [EC_DIR / f"{F}{spf}__{tag}{_sd_bl(s)}__summary.json" for s in range(N_SEEDS)]
    if kind == "go":
        spf = f"_{split}" if split != "sequence" else ""
        return [GO_DIR / f"go_{F}_top50{spf}_{tag}{_sd_ec(s)}.json" for s in range(N_SEEDS)]
    return [BIND_DIR / f"baseline_{F}_lig_{tag}{_sd_ec(s)}.json" for s in range(N_SEEDS)]


def metric_of(d: dict, kind: str, key: str, split: str):
    if kind.startswith("ec"):
        return d.get("by_depth", {}).get(kind[2], {}).get(key)
    if kind == "go":
        return d.get(key)
    task = "binding" if kind == "bs" else "affinity"
    return d.get("results", {}).get(f"{split}_{task}", {}).get(key)


# FIXED(benchmark_audit D7, 2026-10-18): GO results not built on the current GO labels
#   (misato_go_mf_v2.json; go_labels.go_result_problems) are skipped with a warning.
#   Old: every existing GO JSON was used.
def fetch(files, kind, key, split):
    vals = []
    for f in files:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if kind == "go":
            sys.path.insert(0, str(CODE_ROOT / "downstream"))
            from go_labels import go_result_problems
            why = go_result_problems(d)
            if why:
                print(f"[skip] {f.name}: {why[0]}", file=sys.stderr)
                continue
        # FIXED(benchmark_audit D4, 2026-10-18): binding-site results not built on the current
        #   labels (misato_binding_site_v2.npz) are skipped with a warning.
        if kind == "bs":
            from ensembits.misato_binding_labels import binding_label_problem
            why = binding_label_problem(d)
            if why:
                print(f"[skip] {f.name}: {why}", file=sys.stderr)
                continue
        # FIXED(benchmark_audit D8, 2026-10-18): EC results not built on the current EC labels
        #   (misato_ec_v2.json; ec_labels.ec_result_problems) are skipped with a warning.
        if kind.startswith("ec"):
            sys.path.insert(0, str(CODE_ROOT / "downstream"))
            from ec_labels import ec_result_problems
            why = ec_result_problems(d)
            if why:
                print(f"[skip] {f.name}: {why[0]}", file=sys.stderr)
                continue
        vals.append(metric_of(d, kind, key, split))
    return _mean(vals)


# ---------------------------------------------------------------- driver

def collect(variant: str, tag: str):
    """Returns (out[split][label] = scores per axis, n_seeds[split][label] per axis)."""
    out = {sp: {} for sp in SPLITS}
    nseed = {sp: {} for sp in SPLITS}
    for sp in SPLITS:
        for label, stem in MODELS:
            scores, ns = [], []
            for _, kind, key in AXES:
                if stem is None:
                    files = ens_files(kind, variant, label == ENS_P1, sp, tag)
                else:
                    files = base_files(kind, stem, sp, tag)
                v, n = fetch(files, kind, key, sp)
                scores.append(v); ns.append(n)
            out[sp][label] = scores
            nseed[sp][label] = ns
    return out, nseed


def to_skill(split_scores):
    """Per-split: 0 = Random row, 1 = ESM3struct row."""
    rand = np.array(split_scores["Random"])
    esm  = np.array(split_scores["ESM3struct"])
    denom = esm - rand
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    return {label: (np.array(vals) - rand) / denom for label, vals in split_scores.items()}


# ---------------------------------------------------------------- plotting

def plot_panel(ax, skill, split):
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=10, color="0.4")
    ax.set_ylim(-0.5, 2.2)
    ax.set_rlabel_position(72)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a[0] for a in AXES], fontsize=11)
    ax.tick_params(axis="x", pad=14)
    ax.spines["polar"].set_color("0.6")
    ax.grid(color="0.88", lw=0.5)

    ax.plot(angles, [0.0] * (n + 1), color="0.45", lw=0.9, ls="--", zorder=1.5)
    ax.plot(angles, [1.0] * (n + 1), color="0.45", lw=0.9, ls="--", zorder=1.5)

    style = {
        "Random":             dict(color="0.65",    lw=1.0, ls="-",  alpha=0.85, fill=False),
        "aa":                 dict(color="0.40",    lw=1.0, ls="-",  alpha=0.85, fill=False),
        "AminoAseed":         dict(color="#7570b3", lw=1.5, ls="-",  alpha=0.95, fill=False),
        "ProToken":           dict(color="#a6611a", lw=1.3, ls="-",  alpha=0.9,  fill=False),
        "3Di_tokens":         dict(color="#1f9e89", lw=1.3, ls="-",  alpha=0.9,  fill=False),
        "ProtProfileMD_K8":   dict(color="#1b7837", lw=1.3, ls="-",  alpha=0.9,  fill=False),
        "Vote_3Di":           dict(color="#7fbc41", lw=1.5, ls="-",  alpha=0.95, fill=False),
        "ESM3struct":         dict(color="#444444", lw=1.6, ls="-",  alpha=0.95, fill=False),
        ENS_P1:               dict(color="#fdae61", lw=2.0, ls="-",  alpha=0.95, fill=True, fc="#fdae61", fa=0.10),
        ENS_FULL:             dict(color="#d7191c", lw=2.4, ls="-",  alpha=0.95, fill=True, fc="#d7191c", fa=0.22),
    }

    handles = []
    for label in PLOT_MODELS:
        vals = list(skill[label]) + [skill[label][0]]
        s = style[label]
        h, = ax.plot(angles, vals, color=s["color"], lw=s["lw"], ls=s["ls"], alpha=s["alpha"], label=label, zorder=3)
        if s.get("fill"):
            ax.fill(angles, vals, color=s["fc"], alpha=s["fa"], zorder=2)
        handles.append(h)
    ax.set_title(f"{split} split", fontsize=15, pad=16, weight="bold")
    return handles


def plot_all(skill_per_split, out_dir: Path, stem: str):
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 8.5), subplot_kw={"polar": True})
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.04, right=0.98, wspace=0.40)

    handles = None
    for ax, split in zip(axes, SPLITS):
        h = plot_panel(ax, skill_per_split[split], split)
        if handles is None:
            handles = h

    fig.legend(
        handles, PLOT_MODELS,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=4, fontsize=12, frameon=False, handlelength=2.6, columnspacing=2.2,
    )

    out_pdf = out_dir / f"{stem}.pdf"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_pdf, out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True,
                    help="Ensembits variant: model dir (its folder name is used) or the name "
                         "used in the result files")
    ap.add_argument("--tag", default="stab", help="recipe tag in the result file names")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--stem", default="skill_radar_all_splits")
    args = ap.parse_args()
    variant = Path(args.variant.rstrip("/")).name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, nseed = collect(variant, args.tag)
    skill = {sp: to_skill(raw[sp]) for sp in SPLITS}

    # FIXED(code_fix P20, 2026-10-16): written with the csv module; the labels
    #   "Ensembits, P=1" / "Ensembits, P=full" contain a comma, which shifted every
    #   column of those rows in the original hand-joined CSV.
    import csv
    csv_path = out_dir / f"{args.stem}.csv"
    with csv_path.open("w", newline="") as fp:
        w = csv.writer(fp)
        cols = ["split", "model"] + [a[0].replace("\n", " ") + "_raw" for a in AXES] + \
                                    [a[0].replace("\n", " ") + "_skill" for a in AXES] + \
                                    [a[0].replace("\n", " ") + "_nseeds" for a in AXES]
        w.writerow(cols)
        for sp in SPLITS:
            for label in [m[0] for m in MODELS]:
                v = raw[sp][label]; s = skill[sp][label]; ns = nseed[sp][label]
                w.writerow([sp, label] + [f"{x:.4f}" for x in v] + [f"{x:.4f}" for x in s]
                           + [str(x) for x in ns])

    pdf, png = plot_all(skill, out_dir, args.stem)
    print(f"variant {variant}  tag {args.tag}")
    print(f"Wrote {pdf}\n      {png}\n      {csv_path}")
    print()
    for sp in SPLITS:
        print(f"--- {sp} split ---")
        header = "model".ljust(22) + "  " + "  ".join(a[0].replace("\n", " ").rjust(18) for a in AXES)
        print(header)
        for label in [m[0] for m in MODELS]:
            row = label.ljust(22) + "  " + "  ".join(
                f"{v:>13.3f} ({n:>2d})" for v, n in zip(skill[sp][label], nseed[sp][label]))
            print(row)
        print()
    missing = [(sp, lb) for sp in SPLITS for lb, _ in MODELS if min(nseed[sp][lb]) < N_SEEDS]
    if missing:
        print(f"[warn] cells with < {N_SEEDS} seeds: {missing}")


if __name__ == "__main__":
    main()
