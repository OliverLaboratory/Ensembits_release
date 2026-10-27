"""Build camera_ready/results_camera_ready.md from rebuttal/results_bf65.md.

Ranking: bold / underline within each block, higher is better except MSE columns (lower
is better; G15 2026-10-17: the header said "higher is better in every column").

Camera-ready model = the reported model named on the command line (--variant, rows
selected by --label; since 2026-10-16, code_fix P16). Before: the fixed
BioEmu-CATH+MISATO checkpoint
  (output/dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO),
whose rows are now labelled "(pre-fix)" and listed in the reference section.

For every results table (EC/GO/binding/affinity per split, RMSF, ProteinGym) keep
only that model's rows + the baselines, strip the old bold/underline marks and
recompute them over the kept rows (bold = best, underline = second best, higher
is better in every column). Previous Ensembits rows (flagship / paper) are listed
separately for reference. Numbers are copied verbatim — nothing is recomputed.

Regenerate results_bf65.md first if new runs landed:
    python downstream/_compile_rebuttal_results.py
then
    python3 camera_ready/compile_camera_ready_tables.py

What it does
------------
Parses the markdown tables of `REPO_ROOT/rebuttal/results_bf65.md` (sections
"Sequence split", "Structure split", "Random split", "RMSF", "ProteinGym"),
sorts each row into the reported model / baseline / older Ensembits reference,
renames the reported model's rows to "Ensembits (BioEmu-CATH+MISATO)", re-ranks
bold / underline over the kept rows (ties share a mark; the fixed "ESM2-650M
alone" row is not ranked), and writes `REPO_ROOT/camera_ready/results_camera_ready.md`
with a list of rebuttal results that still use the flagship and a reference
section with the previous Ensembits rows. Pure text processing; no numbers are
recomputed.

Paper
-----
Infrastructure only. Its output is where the camera-ready numbers for
`2_results.tex` Table tab:rmsf (\\label{result:rmsf}), Table
tab:proteingym_ensemble (\\label{sec:downstreams}) and the function tables of
`appendix.tex` (\\label{app:downstreams}, \\label{app:experiments}) are taken from.

Origin
------
Copied from `camera_ready/compile_camera_ready_tables.py`. Changes: ROOT is now
`ensembits.paths.REPO_ROOT` (the original used `parents[1]` of its location in
`camera_ready/`, i.e. the research repo), so SRC and OUT are the same files as
before; comments and this docstring. Fixes 2026-10-16 (FIXED markers, code_fix P16 /
P9): the reported model's directory and row label come from the CLI (--variant,
--label, default = REPORTED_LABEL of _compile_rebuttal_results.py); the pre-fix
reported rows go to the reference section; the "Ablations" and "Seed
completeness" sections are copied verbatim; the header states sample std.
2026-10-16 (preretrain_audit_r2 Q17 / Q2): the "Per-protein RMSF" section is copied like
the RMSF one; a kept RMSF row with a refused cell (✗, mixed label / residue set /
checkpoint in the source) stops the build (exit 1, nothing written).
Later the same day (preretrain_audit_r2 F3): sections written in blocks by the compiler
("| *— single-frame —* |", "| *— multi-frame —* |", "| *— not ranked —* |") keep the
reported rows and baselines of the two ranked blocks and rank each block on its own
(columns whose header contains "MSE" ascending); the not-ranked block goes to the
reference section. H6: a kept row with a refused or incomplete cell ("✗" / "(k)") or a
section without the reported row stops the build (exit 1) unless --allow-incomplete.
2026-10-18 (constant-prediction rule): a "—†" cell (every seed predicted a constant, every
residue mapped to one token) is copied as is, with the footnote under its table.

2026-10-19 (function_reaudit pre-run fix 5): `--split-metric val` reads the validation
compile (results_bf65_valmetrics.md) and writes results_camera_ready_valmetrics.md (no
ProteinGym); the "Metrics: <task>, <split> split" tables are kept and ranked like the main
tables (GO floor columns not ranked; MSE ascending, as before), the D5 affinity controls are
copied verbatim.

2026-10-19 (function CV, authors' decision option A): the header notes that the EC / GO
cells of the sequence / structure sections are cross-validation cells (the random section has
no EC / GO columns); tables are copied as before.

2026-10-19 (EC metric, authors' decision): the source's EC d1-d3 columns show mAP (headers
"EC d<k> mAP", ranked); the "Metrics: EC d<k>" tables list mAP, Fmax (secondary) and the
other metrics plus "majority top-1 (ref.)" (majority-class top-1, label only), which is not
ranked here either (NOT_RANKED_COL). Both modes (test and --split-metric val).

Used by the reported model: yes (selects its rows by label).

Inputs / outputs
----------------
Reads  REPO_ROOT/rebuttal/results_bf65.md        (written by _compile_rebuttal_results.py)
Writes REPO_ROOT/camera_ready/results_camera_ready.md

Dependencies left in the research repo: none in code.

How to run
----------
    python downstream/_compile_rebuttal_results.py --variant <run> --pgtag <tag> [...]
    python3 downstream/compile_camera_ready_tables.py --variant <run> [--label <label>] \\
        [--rmsf-protocol test|val] [--src <md>] [--out <md>]
(both overwrite their output files in the research repo)
"""
from __future__ import annotations
import re
import sys
from datetime import date
from pathlib import Path

# Codebase root on sys.path; the input and output markdown stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The row label of the reported model (same default as the compiler; importing it has
# no side effects since code_fix P19).
from _compile_rebuttal_results import REPORTED_LABEL, PREFIX  # noqa: E402
from rmsf_common import CONSTANT_MARK, CONSTANT_NOTE  # noqa: E402  constant-prediction cells (2026-10-18)
# code_fix C4 (2026-10-15): read the test-protocol compilation by default
# (written by _compile_rebuttal_results.py); --rmsf-protocol=val reads the old file.
# FIXED(code_fix P16, 2026-10-16): SRC / OUT / labels are chosen in main() from the CLI
#   (before: sys.argv was inspected at import time).
SRC_TEST = ROOT / "camera_ready" / "results_bf65_testprotocol.md"
SRC_VAL = ROOT / "rebuttal" / "results_bf65.md"
OUT = ROOT / "camera_ready" / "results_camera_ready.md"
# FIXED(function_reaudit pre-run fix 5, 2026-10-19): --split-metric val reads the validation
#   compile of _compile_rebuttal_results.py (--split-metric val) and writes its own file; the
#   "Metrics: <task>, <split> split" tables (mAP, μAP, F1, μF1, Fmax, floor, AP, per-protein
#   AUROC) and the D5 affinity controls are copied too (controls verbatim, not ranked).
SRC_VALMETRIC = ROOT / "camera_ready" / "results_bf65_valmetrics.md"
OUT_VALMETRIC = ROOT / "camera_ready" / "results_camera_ready_valmetrics.md"
METRIC_PREFIX = "Metrics: "
# FIXED(function_reaudit EC metric, authors' decision 2026-10-19): "majority" added: the EC
#   majority-class top-1 column of the "Metrics: EC d<k>" tables (label-only reference) is not
#   ranked, like the GO floor. Old: ("floor",).
NOT_RANKED_COL = ("floor", "majority")

# ---- Which rows are the reported model --------------------------------------
# FIXED(code_fix P16, 2026-10-16): old: fixed labels "BioEmu-CATH+MISATO" /
#   "↳ BioEmu+MISATO (dataset abl.)" and FINAL_DIR = the current checkpoint. New: the rows
#   labelled --label (in every table, "<label> · P=..." in the function tables) and
#   FINAL_DIR = output/<--variant>, set by main(). The compiler labels every older row
#   "<label> (pre-fix)", so they cannot match.
# FIXED(require_fix #17, tex pass 2026-10-26): 2_results.tex tables updated with 256-D numbers.
FINAL = "Ensembits (BioEmu-CATH+MISATO)"     # display name of the reported rows (--final-name)
LABEL = REPORTED_LABEL                       # row label in the source markdown (--label)
FINAL_DIR = None                             # output/<--variant>
# Older Ensembits rows, listed only in the reference section at the end (the pre-fix
# reported model included).
REF_PREFIXES = ("256+φ/ψ FAPE-H (flagship)", "ENSEMBITS 192-D (paper)",
                "BioEmu-CATH+MISATO" + PREFIX, "↳ BioEmu+MISATO (dataset abl.)" + PREFIX)
# Sections of results_bf65.md that are copied (matched by title prefix).
# FIXED(preretrain_audit_r2 Q2 / Q17, 2026-10-16): new section "Per-protein RMSF" (the
#   per-protein Spearman table of _compile_rebuttal_results.py, test protocol only; skipped
#   when the source has none).
SECTIONS = ["Sequence split", "Structure split", "Random split", "RMSF", "Per-protein RMSF",
            "ProteinGym"]
# Mark of a refused RMSF cell in the source (_compile_rebuttal_results.REFUSED_MARK).
REFUSED_MARK = "✗"
RMSF_SECTIONS = ("RMSF", "Per-protein RMSF")
SEEDS_RE = re.compile(r"\((\d+)\)$")       # "0.512±0.003 (7)": fewer than 10 seeds
# Sections copied verbatim (no re-ranking): the ablation table and the seed check.
VERBATIM = ["Ablations", "Affinity controls", "Seed completeness"]
NUM = re.compile(r"[-+]?\d+\.\d+")


def clean(c: str) -> str:
    # Drop the bold / underline markup of a cell.
    return re.sub(r"\*\*|</?u>", "", c).strip()


def mean_of(c: str):
    # First number in a cell ("0.553±0.002" -> 0.553), None for "—" or empty.
    m = NUM.search(clean(c))
    return float(m.group()) if m and clean(c) not in ("—", "") else None


def parse(md: str):
    """{section: (intro_lines, header, rows)} for the tables we care about."""
    out, sec, intro = {}, None, []
    for line in md.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            sec = next((s for s in SECTIONS if name.startswith(s)), None)
            if sec is None and name.startswith(METRIC_PREFIX):   # 2026-10-19: one section per table
                sec = name
            intro = []
            if sec:
                out[sec] = {"title": name, "intro": intro, "header": None, "rows": []}
            continue
        if not sec:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if out[sec]["header"] is None:
                out[sec]["header"] = cells
            elif not set("".join(cells)) <= set("-: "):
                out[sec]["rows"].append(cells)
        elif out[sec]["header"] is None and line.strip():
            intro.append(line)
    return out


def classify(sec, rows):
    # Split a section's rows into (reported model, baselines, older Ensembits
    # reference). Baselines: rows after the "— baselines —" separator, the
    # "ESM2-650M alone" row, and in ProteinGym the rows whose first word names a
    # baseline tokenizer. Every other Ensembits row (quant A-F, ablations, ...)
    # is dropped.
    # FIXED(code_fix E1/E2, 2026-10-20): ablation re-runs done; tables from _compile_rebuttal_results.py.
    final, base, ref, in_base = [], [], [], False
    for r in rows:
        lab = clean(r[0])
        if "baselines" in lab:
            in_base = True; continue
        if in_base or lab.startswith("ESM2-650M") or (sec == "ProteinGym" and not lab.startswith("↳")
                                                       and lab.split()[0] in
                                                       ("esm3struct", "aminoaseed", "protoken", "3di",
                                                        "VanillaVQ", "ESM3-style")):
            base.append(r); continue
        if lab.startswith(LABEL + " ·"):
            final.append([FINAL + lab[len(LABEL):]] + r[1:]); continue
        if lab == LABEL:
            final.append([FINAL] + r[1:]); continue
        if lab.startswith(REF_PREFIXES):
            ref.append([lab] + r[1:])
    return final, base, ref


# FIXED(preretrain_audit_r2 F3, 2026-10-16): old = higher is better in every column and one
#   ranking over all kept rows of a table. New = columns whose header contains "MSE" rank
#   ascending, and the block sections (function tables, RMSF) are ranked block by block.
LOWER_BETTER = ("MSE",)
BLOCK_RE = re.compile(r"^\*— (.+) —\*$")
NOT_RANKED = "not ranked"


def rank(rows, skip=lambda r: False, header=None):
    # Re-mark each column: bold = best mean, underline = second best (lowest for a
    # LOWER_BETTER column); no underline when the top value is tied.
    # FIXED(code_fix P16, 2026-10-16): default=0, so a section without any kept row (e.g. no
    #   reported-model row and no baseline yet) gives an empty table instead of a crash.
    ncol = max((len(r) for r in rows), default=0)
    marks = []
    for c in range(1, ncol):                       # ties (equal printed means) share a mark
        vals = {i: mean_of(r[c]) for i, r in enumerate(rows)
                if c < len(r) and not skip(r) and mean_of(r[c]) is not None}
        if header and c < len(header) and any(t in header[c] for t in NOT_RANKED_COL):
            marks.append((set(), set())); continue       # 2026-10-19: GO floor not ranked
        low = bool(header) and c < len(header) and any(t in header[c] for t in LOWER_BETTER)
        levels = sorted(set(vals.values()), reverse=not low)
        top1 = {i for i, v in vals.items() if levels and v == levels[0]}
        top2 = {i for i, v in vals.items() if len(levels) > 1 and v == levels[1]} if len(top1) == 1 else set()
        marks.append((top1, top2))
    out = []
    for i, r in enumerate(rows):
        cells = [r[0]]
        for c in range(1, len(r)):
            v = clean(r[c]); b1, b2 = marks[c - 1]
            cells.append(f"**{v}**" if i in b1 else f"<u>{v}</u>" if i in b2 else v)
        out.append(cells)
    return out


def classify_blocks(rows):
    """Block sections (rows grouped by "| *— name —* |" separator rows) ->
    (ranked blocks [(name, rows)], reference rows). In the ranked blocks the reported rows
    are renamed and the baselines kept; the not-ranked block (older checkpoints, pooled
    baselines) goes to the reference section."""
    blocks, ref, cur = [], [], None
    for r in rows:
        m = BLOCK_RE.match(clean(r[0]))
        if m:
            cur = m.group(1); blocks.append((cur, [])); continue
        lab = clean(r[0])
        if cur is None or cur == NOT_RANKED:
            ref.append([lab] + r[1:]); continue
        if lab.startswith(LABEL + " ·"):
            blocks[-1][1].append([FINAL + lab[len(LABEL):]] + r[1:])
        elif lab == LABEL:
            blocks[-1][1].append([FINAL] + r[1:])
        elif lab.startswith(REF_PREFIXES) or lab.endswith(PREFIX):
            ref.append([lab] + r[1:])
        else:
            blocks[-1][1].append(r)
    return [(n, b) for n, b in blocks if n != NOT_RANKED], ref


def table(header, rows, sep_after=None):
    # Markdown table; a "— baselines —" separator after row index sep_after.
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for i, r in enumerate(rows):
        lines.append("| " + " | ".join(r) + " |")
        if sep_after is not None and i == sep_after:
            lines.append("| *— baselines —* |" + " |" * (len(header) - 1))
    return lines


def verbatim_sections(md_text: str, titles) -> list[str]:
    """Lines of the "## <title>" sections whose title starts with one of `titles`."""
    out, keep = [], False
    for line in md_text.splitlines():
        if line.startswith("## "):
            keep = any(line[3:].strip().startswith(t) for t in titles)
        if keep:
            out.append(line)
    return out


def main(argv=None):
    global FINAL, LABEL, FINAL_DIR
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True,
                    help="output/<variant> of the reported model (code_fix P16)")
    ap.add_argument("--label", default=REPORTED_LABEL,
                    help="row label of the reported model in the source markdown "
                         "(_compile_rebuttal_results.py --label; default %(default)r)")
    ap.add_argument("--final-name", default=FINAL, help="display name of the reported rows")
    ap.add_argument("--rmsf-protocol", choices=("test", "val"), default="test",
                    help="test: camera_ready/results_bf65_testprotocol.md; val: rebuttal/results_bf65.md")
    ap.add_argument("--split-metric", choices=("test", "val"), default="test",
                    help="val: the validation compile (results_bf65_valmetrics.md -> "
                         "results_camera_ready_valmetrics.md; no ProteinGym section)")
    ap.add_argument("--src", default=None, help="source markdown (default by protocol)")
    ap.add_argument("--out", default=None, help=f"default {OUT.name} (val: {OUT_VALMETRIC.name})")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write a preview even if kept rows have refused / incomplete cells or "
                         "a section lacks the reported row (H6)")
    a = ap.parse_args(argv)
    FINAL, LABEL, FINAL_DIR = a.final_name, a.label, f"output/{a.variant}"
    val = a.split_metric == "val"
    src = Path(a.src) if a.src else (SRC_VALMETRIC if val else
                                     SRC_TEST if a.rmsf_protocol == "test" else SRC_VAL)
    out_path = Path(a.out) if a.out else (OUT_VALMETRIC if val else OUT)
    # ---- Header of the output file --------------------------------------------
    src_text = src.read_text()
    T = parse(src_text)
    md = ([f"# VALIDATION METRICS ONLY (--split-metric val; no test field was read)", ""]
          if val else [])
    md += [f"# Camera-ready results — final model: {FINAL}", "",
          f"- **Final model:** `{FINAL_DIR}`, rows labelled `{LABEL}` in the source "
          "(256-D frame+φ/ψ descriptor, Hungarian-FAPE RVQ 2048/128/128; mdCATH-div replaced by "
          "BioEmu-CATH MD, mdCATH eval-test domains held out).",
          f"- **Source:** `{src.name}`, regenerated from the raw result JSONs on "
          f"{date.today().isoformat()} with `downstream/_compile_rebuttal_results.py`. "
          "Numbers are copied verbatim.",
          # G15 (2026-10-17): was "higher is better in every column"
          "- **Rows kept:** the final model (P=full and P=1) + the paper baselines, in the "
          "single-frame and multi-frame blocks. Bold = best, <u>underline</u> = second best, "
          "recomputed within each block (higher is better, except MSE: lower is better).",
          "- EC d1–d3 columns = mAP (macro-averaged average precision over the EC classes of the "
          "depth), the primary EC metric; Fmax (secondary) and the other metrics are in the "
          "\"Metrics: EC\" tables, with the majority-class top-1 as a label-only reference "
          "column (not ranked).",
          # FIXED(coordinator, 2026-10-19): GO main column = mAP.
          "- GO column = mAP over the top-50 MF classes (primary); Fmax, top-1 and the frequency "
          "floor (label only, not ranked) are in the \"Metrics: GO\" tables.",
          "- Cells = mean ± sample std (ddof = 1) over 10 seeds unless marked `(k)` = k seeds; "
          "`—` = not applicable / absent.",
          # FIXED(function CV, 2026-10-19): note on the EC / GO protocol of the source tables.
          "- EC and GO (sequence / structure): distinct proteins, grouped label-stratified "
          "5-fold cross-validation (`data/misato_function_cv_v1.json`); per seed the mean over "
          "the five folds, then mean ± sample std over seeds (per-fold values in the source's "
          "`.cv_folds.json`). Binding site, affinity and the random split: fixed v2 splits.",
          "- Built by `camera_ready/compile_camera_ready_tables.py`.", ""]
    # ---- One table per section: reported model rows, then baselines ---------
    # Paper: 2_results.tex tab:rmsf (RMSF section), tab:proteingym_ensemble
    # (ProteinGym section); appendix.tex app:downstreams / app:experiments (splits).
    refs = {}
    refused = []
    missing = []
    sections = [s for s in SECTIONS if not (val and s == "ProteinGym")] + \
        [s for s in T if s.startswith(METRIC_PREFIX)]
    for sec in sections:
        if sec not in T:
            print(f"WARNING: no section {sec!r} in {src.name}", file=sys.stderr)
            continue
        t = T[sec]
        # A.5 fix: skip sections without a parsed table (e.g. ProteinGym redirected
        # to pg_aggregate.py)
        if t.get("header") is None and not t.get("rows"):
            md += [f"## {t['title']}", ""]
            if t.get("intro"):
                md += t["intro"] + [""]
            continue
        is_blocks = any(BLOCK_RE.match(clean(r[0])) for r in t["rows"])
        if is_blocks:
            blocks, ref = classify_blocks(t["rows"])
            final = [r for _, b in blocks for r in b if clean(r[0]).startswith(FINAL)]
            base = [r for _, b in blocks for r in b if not clean(r[0]).startswith(FINAL)]
        else:
            final, base, ref = classify(sec, t["rows"])
        # FIXED(preretrain_audit_r2 Q17, 2026-10-16): new: a kept RMSF row with a refused
        #   cell (mixed label / residue set / checkpoint in the source) stops the build.
        # FIXED(preretrain_audit_r2 H6, 2026-10-16): extended to every section, and to cells
        #   with fewer than 10 seeds ("(k)", from a source compiled with --allow-incomplete)
        #   and to a missing reported row; --allow-incomplete builds a preview instead.
        refused += [(sec, clean(r[0])) for r in final + base
                    if any(REFUSED_MARK in c or SEEDS_RE.search(clean(c)) for c in r[1:])]
        if not final:   # FIXED(code_fix P16, 2026-10-16): new (old: none): say so instead of a table without the model
            print(f"WARNING: no row labelled {LABEL!r} in section {sec!r} of {src.name}",
                  file=sys.stderr)
            missing.append(sec)
        refs[sec] = (t["header"], ref)
        is_pg = sec == "ProteinGym"
        rows = rank(final + base, skip=(lambda r: clean(r[0]).startswith("ESM2-650M")) if is_pg
                    else (lambda r: False), header=t["header"])
        md += [f"## {t['title']}", ""]
        intro = [l for l in t["intro"] if "Baselines + the paper row" not in l]
        if intro:
            md += intro + [""]
        if is_pg:
            md += ["*The final-model row is the one reported as \"Ensembits (BioEmu+MISATO), P=1\" "
                   "in the rebuttal (`summary_review.md`).*", ""]
        if is_blocks:
            # F3: one ranking per block, block separator rows kept (two halves of the paper tables)
            md += table(t["header"], [])[:2]
            for name, b in blocks:
                if not b: continue
                md.append("| *— " + name + " —* |" + " |" * (len(t["header"]) - 1))
                md += ["| " + " | ".join(r) + " |" for r in rank(b, header=t["header"])]
            md += [""]
        else:
            md += table(t["header"], rows, sep_after=len(final) - 1) + [""]
        # 2026-10-18 (constant-prediction rule): the footnote of the "—†" cells.
        if any(CONSTANT_MARK in c for r in final + base for c in r[1:]):
            md += [CONSTANT_NOTE + ".", ""]

    # ---- Results that still use the flagship model (fixed text) ---------------
    # FIXED(code_fix E1, 2026-10-20): VQ-2048 re-run = output/ensembits_bm_vq2048.
    # FIXED(code_fix E2, 2026-10-20): SFTD λ sweep re-runs = output/ensembits_bm_lam*_rv.
    md += ["## Not yet available for the final model", "",
           "These rebuttal results were produced with the **flagship mdCATH+MISATO model** "
           "(`output/dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT`) or its ablations, not BioEmu+MISATO:", "",
           "| Result | Where in the rebuttal | Model used |",
           "|---|---|---|",
           "| SFTD ablation (consMSE 0.1/0.5, distillation off, SFTD off) | `summary_review.md` (SFTD table), RMSF rows here | flagship + PRODSPLIT ablations |",
           "| VQ-2048 vs RVQ | `summary_review.md` | flagship |",
           "| consMSE 0.5–2.0 sweep, 1-frame-trained models | `results_bf65.md` | flagship corpus |",
           "| CryptoBench (cryptic pockets) | `response_cryptobench.md`, `cryptobench_summary.md` | flagship |",
           "| Significance tests (ProteinGym paired, binding Welch) | `response_significance.md` | flagship (PG 0.382/0.517, binding 0.749) |",
           "| ProteinGym vs baselines retrained on our data | `response_proteingym_ourdata.md` | flagship (0.382/0.517); not in the camera-ready (released checkpoints only) |",
           "| quant A–F / warm-start variants | `results_bf65.md` | BioEmu+MISATO corpus, but different quantizer/training recipes |",
           "",
           "Figures made in this session with the flagship model and needing a re-render if they "
           "should show the final model: `camera_ready/fig_descriptor/` (token IDs), "
           "`camera_ready/fig_codebook/` (picked tokens).", ""]

    # ---- Ablation table and seed check, copied verbatim (code_fix P16) ----------
    # FIXED(code_fix P16, 2026-10-16): new (old: not copied): the compiler's Ablations and
    #   Seed completeness sections.
    md += verbatim_sections(src_text, VERBATIM) + [""]

    # ---- Reference section with the flagship / 192-D / pre-fix rows, then write -
    md += ["## Reference: previous Ensembits rows (not for the camera-ready tables)", ""]
    for sec in sections:
        if sec not in refs:
            continue
        header, ref = refs[sec]
        if ref:
            md += [f"**{T[sec]['title']}**", ""] + table(header, [[clean(c) for c in r] for r in ref]) + [""]
    if (refused or missing) and not a.allow_incomplete:
        print(f"ERROR: {len(refused)} kept rows have refused or incomplete cells "
              f"({REFUSED_MARK} / (k)) and {len(missing)} sections lack the reported row in "
              f"{src.name}; rerun them (see its 'Refused RMSF cells' / 'Seed completeness' "
              f"sections) or pass --allow-incomplete for a preview. Not written:",
              file=sys.stderr)
        for sec, lab in refused: print(f"  {sec}: {lab}", file=sys.stderr)
        for sec in missing: print(f"  {sec}: no row labelled {LABEL!r}", file=sys.stderr)
        return 1
    out_path.write_text("\n".join(md))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
