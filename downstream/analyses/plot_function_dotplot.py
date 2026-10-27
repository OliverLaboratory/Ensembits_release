"""Dot (forest) plot of the MISATO function tasks: every method's primary metric with error bars.

What it does
------------
Replaces the skill-score radar (final_fix_list A3, authors' decision (a), 2026-10-19). One
panel row per task, the primary metric of `downstream/_compile_rebuttal_results.py`:
  EC depth 1, 2, 3 (mAP), GO molecular function (mAP), binding site (AUROC);
and two panel columns, the sequence and the structure grouping (EC / GO: grouped 5-fold
cross-validation; binding site: the fixed v2 splits). In each panel every method is one
point (mean over the 10 probe seeds) with an error bar of +- 1 seed sd (sample sd, ddof = 1,
the table cell); for EC and GO the 5 per-fold values (mean over seeds of each test fold)
are drawn as faint marks on the same row. Rows are grouped as in the tables: single-frame
(aa, 3Di, AminoAseed, ProToken, ESM3struct, Ensembits P = 1) above multi-frame (Vote_3Di,
ProtProfileMD, Ensembits P = full), separated by a shaded band. The random-feature row of
the same task and split is drawn as a dashed reference line (not as a point). Both
Ensembits candidates (lambda is not chosen: --variants, default lambda = 1 and
lambda = 0.25) are shown at P = 1 and P = full, each candidate in its own colour, P = 1 as a
circle and P = full as a diamond.

Every number is read through the compiler's checked loaders (`m_ec`, `m_go`, `m_bind`,
`b_ec`, `b_go`, `b_bind`, `_check_items`, `enforce_seeds` of
`downstream/_compile_rebuttal_results.py`), so the same cells are refused as in the tables:
MISATO split file (D3), EC labels (D8), GO labels (D7), binding-site labels (D4), function
protocol incl. input standardisation (FUNCTION_PROTOCOL), function CV file, item / class
signatures, and fewer than 10 seeds. A refused cell is not drawn; the script then exits 1
and lists it (--allow-incomplete draws the remaining cells for a preview and exits 0).

Split metric: by default every value is the VALIDATION field of the result JSONs
(`--split-metric val`; the compiler removes the test fields right after reading each JSON).
`--split-metric test` draws the test values and refuses to run unless `--unseal` is also
given (the test metrics stay sealed until lambda is chosen).

Paper
-----
`2_results.tex` Figure fig:downstream_radar (to be replaced by this figure; the authors
move the function tasks to the appendix, final_fix_list A2 / A3). The numbers are those of
the function tables (`appendix.tex` \\label{app:downstreams}).

Origin
------
New (2026-10-19, final_fix_list A3 option (a)). Replaces `plot_skill_radar.py` for the
camera-ready (that script reads no CV file, checks no protocol and plots secondary
metrics; it is kept for reference only).

Used by the reported model: yes (both lambda candidates until lambda is chosen).

Inputs / outputs
----------------
reads   (through the compiler) output/ec_conv1d_misato/*_cv_*__summary.json,
        output/go_conv1d_misato/go_*_cv_*_top50.json, output/binding_misato/*_lig_stab*.json,
        data/misato_splits_v2.json, data/misato_function_cv_v1.json and the label files
        the compiler checks against
writes  <out-dir>/function_dotplot_<val|test>.pdf, .png and .json (plotted values, refused
        cells, provenance); default <out-dir> = REPO_ROOT/output/figures; atomic writes

How to run
----------
    python downstream/analyses/plot_function_dotplot.py                       # validation
    python downstream/analyses/plot_function_dotplot.py --variants \\
        ensembits_bm_lam10_rv="λ = 10" --out-dir output/figures
    python downstream/analyses/plot_function_dotplot.py --split-metric test --unseal   # sealed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream"))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402

OUT_DIR = ROOT / "output" / "figures"
# FIXED(recon2 A.7, 2026-10-25): default switched to the chosen λ = 10 _rv run.
DEFAULT_VARIANTS = ("ensembits_bm_lam10_rv=λ = 10",)

# Paper: appendix.tex app:downstreams — primary metric per task (the compiler's MAIN_SPECS
# columns "EC d<k> mAP", "GO mAP", "Bind AUROC"). (row title, task, EC depth, metric key).
TASKS = [("EC depth 1 (mAP)", "EC", 1, "macro_ap"),
         ("EC depth 2 (mAP)", "EC", 2, "macro_ap"),
         ("EC depth 3 (mAP)", "EC", 3, "macro_ap"),
         ("GO molecular function (mAP)", "GO", None, "macro_ap"),
         ("Binding site (AUROC)", "binding", None, "auroc")]
SPLITS = [("sequence", "Sequence grouping"), ("structure", "Structure grouping")]
# Baselines in the order of the compiler's blocks (random = the reference line, not a point).
SINGLE_BASE = [("aa", "Amino acid"), ("3di_tokens", "3Di"), ("aminoaseed", "AminoAseed"),
               ("protoken", "ProToken"), ("esm3struct", "ESM3struct")]
# B.7 fix: Vote_3Di uses 8 MISATO frames, not 10; drop "Ensembits" prefix from labels
MULTI_BASE = [("vote_3di", "Vote_3Di (8 frames)"), ("protprofile_K8", "ProtProfileMD (8 frames)")]
RANDOM = "random"
# Colours: baselines neutral ink; one categorical hue per Ensembits candidate (reference
# palette slots 1, 2, 3, 7 in fixed order; never cycled).
BASE_COLOR = "#52514e"
CAND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
FLOOR_COLOR = "#8a8984"
BAND_COLOR = "#efeeea"


class SealedError(SystemExit):
    """--split-metric test without --unseal."""


def check_unseal(split_metric: str, unseal: bool) -> None:
    """Refuse the test metrics unless --unseal is given (raises SealedError, exit code 2)."""
    if split_metric == "test" and not unseal:
        raise SealedError("--split-metric test reads the sealed TEST metrics; refused without "
                          "--unseal (lambda is not chosen; use the default validation fields)")


def parse_variants(items) -> list[tuple[str, str]]:
    """["dir=label", ...] -> [(variant dir name, label)]."""
    out = []
    for it in items:
        if "=" in it:
            v, lab = it.split("=", 1)
        else:
            v, lab = it, it
        out.append((v.strip(), lab.strip()))
    return out


def _cell_dict(t):
    """A compiler cell -> {"mean", "sd", "n", "status"} (status ok / refused / missing)."""
    if t is None:
        return {"mean": None, "sd": None, "n": 0, "status": "missing"}
    if getattr(t, "refused", False):
        return {"mean": None, "sd": None, "n": int(t[2]), "status": "refused"}
    return {"mean": float(t[0]), "sd": None if t[1] is None else float(t[1]), "n": int(t[2]),
            "status": "ok"}


def collect(variants, split_metric: str = "val", allow_incomplete: bool = False) -> dict:
    """All cells of the plot through the compiler's checked loaders.

    Returns {"rows": [row dicts], "floors": {(split, task col): cell}, "refused": [...]}; each
    row dict: {key, label, group, variant, P, color_idx, cells: {split: {task col: cell +
    per_fold}}}. The compiler module state is reset first; the test fields are stripped by
    the compiler in val mode.
    """
    import _compile_rebuttal_results as CR
    CR.set_rmsf_protocol("test")
    CR.set_split_metric(split_metric)
    CR.FN_CV = True                      # EC / GO from the cross-validation runs
    CR.ALLOW_INCOMPLETE = bool(allow_incomplete)
    for lst in (CR._SPLIT_REFUSED, CR._EC_REFUSED, CR._BIND_REFUSED, CR._GO_REFUSED,
                CR._FN_REFUSED, CR._ITEM_REFUSED, CR._CV_REFUSED):
        lst.clear()
    for d in (CR._SIG, CR._REF_SIG, CR._CV_FOLDS):
        d.clear()
    specs = [(t[0], t[1], t[2], t[3]) for t in TASKS]
    entries = []   # (key, label, group, variant, P, color idx, rowid, getter)
    for F, lab in SINGLE_BASE:
        entries.append((f"baseline_{F}", lab, "single-frame", None, None, None, f"baseline_{F}",
                        ("b", F)))
    # B.7 fix: drop "Ensembits" prefix from Ensembits labels (avoids "Ensembits Ensembits …")
    for i, (V, lab) in enumerate(variants):
        entries.append((f"{V}|P1", f"{lab} · P = 1", "single-frame", V, "1", i,
                        CR._rowid(CR.tag_of(V), True), ("m", V, True)))
    for F, lab in MULTI_BASE:
        entries.append((f"baseline_{F}", lab, "multi-frame", None, None, None, f"baseline_{F}",
                        ("b", F)))
    for i, (V, lab) in enumerate(variants):
        entries.append((f"{V}|Pfull", f"{lab} · P = full", "multi-frame", V, "full", i,
                        CR._rowid(CR.tag_of(V), False), ("m", V, False)))
    entries.append(("baseline_random", "random features", "floor", None, None, None,
                    "baseline_random", ("b", RANDOM)))

    def read(getter, sp):
        if getter[0] == "b":
            return CR.brow(getter[1], sp, specs)
        return CR.mrow(getter[1], getter[2], sp, specs)

    raw = {sp: {e[0]: read(e[7], sp) for e in entries} for sp, _ in SPLITS}
    rows, floors, refused = [], {}, []
    for sp, _ in SPLITS:
        # reference item / class signature over every plotted row (the compiler's ranked rows)
        CR._set_reference(sp, [e[6] for e in entries], specs)
    for e in entries:
        key, label, group, V, P, ci, rowid, _ = e
        cells = {}
        for sp, _ in SPLITS:
            vals = CR._check_items(rowid, sp, specs, raw[sp][key])
            vals = CR.enforce_seeds(vals)            # 10 seeds unless --allow-incomplete
            cells[sp] = {}
            for (col, task, depth, metric), v in zip(specs, vals):
                c = _cell_dict(v)
                tc = CR._taskcol(task, depth)
                pf = CR._CV_FOLDS.get(f"{rowid}|{sp}|{tc}|{metric}")
                if pf is not None and c["status"] == "ok":
                    c["per_fold_mean"] = {k: (sum(x) / len(x) if x else None)
                                          for k, x in pf["per_fold"].items()}
                if c["status"] != "ok":
                    refused.append({"row": label, "split": sp, "task": col,
                                    "status": c["status"], "n_seeds": c["n"]})
                cells[sp][col] = c
        if group == "floor":
            for sp, _ in SPLITS:
                for col in cells[sp]:
                    floors[f"{sp}|{col}"] = cells[sp][col]
            continue
        rows.append({"key": key, "label": label, "group": group, "variant": V, "P": P,
                     "color_idx": ci, "cells": cells})
    reasons = {"split file": list(CR._SPLIT_REFUSED), "EC labels": list(CR._EC_REFUSED),
               "GO labels": list(CR._GO_REFUSED), "binding labels": list(CR._BIND_REFUSED),
               "function protocol": list(CR._FN_REFUSED), "CV file": list(CR._CV_REFUSED),
               "items / classes": list(CR._ITEM_REFUSED)}
    return {"rows": rows, "floors": floors, "refused": refused,
            "refusal_reasons": {k: [list(map(str, r)) for r in v][:50] for k, v in reasons.items() if v}}


def draw(data: dict, split_metric: str, out_stem: Path) -> list[Path]:
    """Render the grid (tasks x splits) to <out_stem>.pdf / .png (atomic); returns the paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = data["rows"]
    n = len(rows)
    single = [i for i, r in enumerate(rows) if r["group"] == "single-frame"]
    # y positions: single-frame block, a gap of 0.8, the multi-frame block (top to bottom)
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i > 0 and r["group"] != rows[i - 1]["group"]:
            y += 0.8
        ys.append(y)
        y += 1.0
    multi_y = [ys[i] for i, r in enumerate(rows) if r["group"] == "multi-frame"]
    fig, axes = plt.subplots(len(TASKS), len(SPLITS), figsize=(7.0, 1.55 * len(TASKS) + 0.6),
                             sharey=True, squeeze=False)
    for ti, (col, task, _d, _m) in enumerate(TASKS):
        lo, hi = [], []
        for si, (sp, title) in enumerate(SPLITS):
            ax = axes[ti][si]
            if multi_y:
                ax.axhspan(min(multi_y) - 0.5, max(multi_y) + 0.5, color=BAND_COLOR, zorder=0, lw=0)
            fl = data["floors"].get(f"{sp}|{col}")
            if fl and fl["status"] == "ok":
                ax.axvline(fl["mean"], color=FLOOR_COLOR, ls=(0, (3, 2)), lw=0.9, zorder=1)
                lo.append(fl["mean"]); hi.append(fl["mean"])
            for i, r in enumerate(rows):
                c = r["cells"][sp][col]
                if c["status"] != "ok":
                    ax.text(0.99, ys[i], "refused" if c["status"] == "refused" else "missing",
                            transform=ax.get_yaxis_transform(), ha="right", va="center",
                            fontsize=5.5, color=FLOOR_COLOR)
                    continue
                color = BASE_COLOR if r["color_idx"] is None else CAND_COLORS[r["color_idx"] % len(CAND_COLORS)]
                marker = "o" if r["P"] in (None, "1") else "D"
                size = 3.6 if r["P"] is None else 4.4
                for v in (c.get("per_fold_mean") or {}).values():
                    if v is not None:
                        ax.plot([v], [ys[i]], marker="|", ms=5, mew=0.8, color=color, alpha=0.35,
                                zorder=2, ls="none")
                        lo.append(v); hi.append(v)
                sd = c["sd"] or 0.0
                ax.errorbar(c["mean"], ys[i], xerr=sd, fmt=marker, ms=size, color=color,
                            mec="white", mew=0.5, ecolor=color, elinewidth=1.0, capsize=1.5,
                            zorder=3)
                lo.append(c["mean"] - sd); hi.append(c["mean"] + sd)
            if ti == 0:
                ax.set_title(title, fontsize=8, fontweight="bold")
            # no vertical grid: the faint per-fold marks are vertical ticks themselves
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", length=0)
        if lo:                                       # same x range in both splits of a task
            pad = 0.06 * (max(hi) - min(lo) or 0.01)
            for si in range(len(SPLITS)):
                axes[ti][si].set_xlim(min(lo) - pad, max(hi) + pad)
        axes[ti][0].set_ylabel(col, fontsize=7, fontweight="bold")
    axes[0][0].set_yticks(ys)
    axes[0][0].set_yticklabels([r["label"] for r in rows])
    axes[0][0].set_ylim(max(ys) + 0.7, -0.7)
    for ti in range(len(TASKS)):
        for si in range(len(SPLITS)):
            if si > 0:
                axes[ti][si].tick_params(labelleft=False)
    handles = [Line2D([], [], marker="o", ls="none", color=BASE_COLOR, ms=4, label="baseline"),
               Line2D([], [], marker="|", ls="none", color=BASE_COLOR, alpha=0.5, ms=6,
                      label="per CV fold (EC, GO)"),
               Line2D([], [], color=FLOOR_COLOR, ls=(0, (3, 2)), lw=0.9, label="random features")]
    seen = set()
    for r in rows:
        if r["color_idx"] is not None and r["color_idx"] not in seen:
            seen.add(r["color_idx"])
            handles.append(Line2D([], [], marker="o", ls="none",
                                  color=CAND_COLORS[r["color_idx"] % len(CAND_COLORS)], ms=4,
                                  label=r["label"].split(" · ")[0]))
    handles += [Line2D([], [], marker="o", ls="none", color="#888", ms=4, label="P = 1"),
                Line2D([], [], marker="D", ls="none", color="#888", ms=4, label="P = full")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.0))
    what = "validation" if split_metric == "val" else "test"
    fig.text(0.5, 0.995, f"MISATO function tasks ({what} metrics): mean ± seed sd over 10 probe "
             f"seeds; shaded = multi-frame methods", ha="center", va="top", fontsize=6.5,
             color=BASE_COLOR)
    fig.tight_layout(rect=(0, 0.06, 1, 0.985), h_pad=0.6, w_pad=0.8)
    paths = []
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix("." + ext)
        tmp = p.with_name("." + p.name + ".tmp")
        fig.savefig(tmp, format=ext, dpi=300 if ext == "png" else None)
        os.replace(tmp, p)
        paths.append(p)
    plt.close(fig)
    return paths


def _code_sha256() -> dict:
    out = {}
    for rel in ("downstream/analyses/plot_function_dotplot.py", "downstream/_compile_rebuttal_results.py",
                "downstream/function_common.py", "downstream/function_cv.py"):
        p = CODE_ROOT / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS),
                    help="Ensembits candidates as <output dir name>=<label> (default: both lambdas)")
    ap.add_argument("--split-metric", choices=("val", "test"), default="val",
                    help="val (default): validation fields only; test: sealed, needs --unseal")
    ap.add_argument("--unseal", action="store_true",
                    help="explicitly allow --split-metric test (reads the sealed test metrics)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="preview: draw the cells that pass and exit 0 even if some are refused")
    a = ap.parse_args(argv)
    try:
        check_unseal(a.split_metric, a.unseal)
    except SealedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    variants = parse_variants(a.variants)
    data = collect(variants, a.split_metric, a.allow_incomplete)
    stem = Path(a.out_dir) / f"function_dotplot_{a.split_metric}"
    paths = draw(data, a.split_metric, stem)
    rec = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "split_metric": a.split_metric,
           "variants": variants, "tasks": [t[0] for t in TASKS], "splits": [s for s, _ in SPLITS],
           # B.7 fix: split-aware cell_rule (was "each test fold" even in val mode)
           "cell_rule": f"mean and sample sd (ddof = 1) over 10 seeds; EC / GO per seed = mean "
                        f"over the 5 CV folds (compiler aggregate_cv); per_fold_mean = mean over "
                        f"seeds of each {a.split_metric} fold",
           "rows": data["rows"], "random_floor": data["floors"], "refused": data["refused"],
           "refusal_reasons": data["refusal_reasons"], "allow_incomplete": a.allow_incomplete,
           "code_sha256": _code_sha256(), "figures": [str(p) for p in paths]}
    side = stem.with_suffix(".json")
    tmp = side.with_name("." + side.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(tmp, side)
    for p in paths + [side]:
        print(f"wrote {p}")
    if data["refused"]:
        print(f"{'NOTE' if a.allow_incomplete else 'ERROR'}: {len(data['refused'])} cells refused "
              f"or missing (not drawn):", file=sys.stderr)
        for r in data["refused"][:30]:
            print(f"  {r['row']:<36} {r['split']:<10} {r['task']:<28} {r['status']} "
                  f"({r['n_seeds']} seeds)", file=sys.stderr)
        return 0 if a.allow_incomplete else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
