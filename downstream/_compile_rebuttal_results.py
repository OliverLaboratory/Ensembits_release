"""Compile rebuttal/results.md from the stabilized _stab downstream JSONs + RMSF +
ProteinGym. Lives in the repo (survives /tmp scratchpad wipes). Re-runnable.

What it does
------------
Reads every downstream result JSON of the Ensembits variants and baselines and
writes one markdown file, `REPO_ROOT/rebuttal/results_bf65.md` (the name in the
first line above is outdated; `results.md` is owned by another script):
  * EC depth 1/2/3 (mAP), GO top-50, binding AUROC and affinity Spearman per split
    (sequence / structure / random), each Ensembits variant at P=full and P=1,
    then the baselines;
  * RMSF Spearman on mdCATH-div and MISATO seq / struct, P=full and P=1;
  * ProteinGym (96 DMS, L <= 200): score (a) alone and its z-blend with ESM2 at
    alpha = 0.3, and the gain over ESM2 alone;
  * (2026-10-16) an ablation section (E1 VQ-2048, E2 λ sweep; variants from
    --ablation-variants) and a seed-completeness section listing every paper row
    (reported model, baselines, ablations) with fewer than 10 seeds in a cell.
Cells = mean ± SAMPLE std (ddof = 1) over seeds 0-9; "(k)" = only k < 10 seeds
found; "—" = absent. Per column, bold = best and underline = second best over all
rows. Numbers are only aggregated, never recomputed.

Tables: EC/GO/binding/affinity per split (flagship + paper + baselines incl. pooled),
RMSF (Spearman), ProteinGym. Cells = mean±std over seeds; (k)=k<10 seeds; —=absent.

Paper
-----
Infrastructure only. Its output is the source of the paper tables via
`compile_camera_ready_tables.py`: `2_results.tex` Table tab:rmsf
(\\label{result:rmsf}), Table tab:proteingym_ensemble (\\label{sec:downstreams}),
and `appendix.tex` \\label{app:downstreams} / \\label{app:experiments}.

Origin
------
Copied from `submission_exp/src/_compile_rebuttal_results.py`. Changes: ROOT is
now `ensembits.paths.REPO_ROOT` (inputs and the output file stay in the research
repo); comments and this docstring. Approved fixes 2026-10-16 (FIXED markers):
  * P19: the compile body is in main(); importing the module (significance_tests.py
    imports pg_load / tag_of) has no side effects any more (it used to rewrite the
    results file); CLI behaviour unchanged (`--rmsf-protocol=val` still works).
  * P16: the reported model's rows come from the CLI (--variant, --pgtag, --label);
    every older Ensembits row is relabelled "<label> (pre-fix)" so the camera-ready
    selector cannot pick it; ablation section (--ablation-variants); seed-completeness
    check.
  * P9: sample std (statistics.stdev, ddof = 1) instead of population std.
  * P21: the two pair-test sections (old output/rebuttal_analyses/ JSONs) were removed;
    the pair table is built by downstream/fold_pairs/build_pair_table.py.
  * preretrain_audit_r2 Q17 / Q2 / Q3 (RMSF readers): with the test protocol every RMSF
    result JSON must carry the expected label (--rmsf-label, default allframes), the
    table's common residue set (per dataset; the current residue-set file) and, for
    Ensembits rows, one checkpoint block (= output/<variant>/best.pt when present);
    otherwise the cell is not printed (✗) and listed under "Refused RMSF cells", and
    main() returns 1 if a reported / baseline / ablation cell is refused. A second
    table "Per-protein RMSF" gives the per-protein Spearman (mean over test proteins).
  * phase 2: BioEmu-CATH columns (output/probe_rmsf_bioemu/) in the RMSF tables and the
    ablation table.
  * constant-prediction rule (authors' decision, 2026-10-18): an RMSF cell whose seeds
    are all NaN because the probe predicted a constant with every residue mapped to one
    token is shown as "—†" with the footnote "every residue maps to one token; Spearman
    undefined" (not ranked, its seeds not counted as missing); any other NaN seed refuses
    the cell (test protocol).
  * preretrain_audit_r2 F1 / F3 / F5 (later the same day): the RMSF tables follow tab:rmsf
    (columns = datasets; rows in a single-frame block, baselines + Ensembits P=1, and a
    multi-frame block, ProtProfileMD_K + Vote_3Di + Ensembits P=full); the function tables
    use the same two blocks (pooled baselines and older checkpoints in a block that is not
    ranked) and gain affinity R2 and MSE; bold / underline are computed per block and MSE
    ranks ascending; RMSF cells also need one label content hash and identical feature
    caches across seeds.
  * preretrain_audit_r2 H2 / H6 (final hardening): the input caches recorded in an RMSF
    result must still match the files on disk; a paper-row cell (reported, ablation,
    baseline) with fewer than 10 seeds is refused ("✗ (k)", not ranked) and main() returns
    1; `--allow-incomplete` compiles a preview with "m±s (k)" cells instead.
  * function_reaudit pre-run fixes (2026-10-19): `--split-metric val` compiles every column
    from the validation fields (function JSONs: the "val" block at the selected epoch; RMSF:
    best_val_spearman / val_spearman_per_protein_mean) and strips the test fields from every
    JSON it reads; ProteinGym is left out -> camera_ready/results_bf65_valmetrics.md. New
    "Metrics: <task>, <split> split" tables (EC d1-d3 and GO: top-1, mAP, μAP, F1@0.5,
    μF1@0.5, Fmax; GO frequency floor, not ranked; binding: AUROC, AP, per-protein AUROC) and
    "Affinity controls (D5)" tables (ligand only, protein only; not ranked). Every function
    result must carry the protocol tag of downstream/function_common.py, verified inputs and
    (affinity) the current label / MACCS sha256, and every method of a cell the same class
    list and item hashes; refused cells are listed ("Refused cells (function protocol)",
    "Refused cells (items / classes)") and fail the compile for paper rows.
  * EC metric (authors' decision, 2026-10-19): mAP is the primary EC metric. The EC d1-d3
    columns of the main split tables and the ablation table show mAP (headers "EC d<k> mAP";
    bold / underline rank it); the "Metrics: EC d<k>" tables list mAP, Fmax (secondary),
    top-1, μAP, F1@0.5, μF1@0.5 and the majority-class top-1 ("majority top-1 (ref.)", the
    most frequent training class predicted for every complex; not ranked). Same in the test
    and the --split-metric val compile. Old: the EC columns showed the top-1 hit.
  * benchmark_audit D8 (2026-10-18): an EC cell is refused ("✗", listed under "Refused EC
    cells"; main() returns 1 for paper rows) unless every result JSON records the sha256 of
    the current EC labels (data/annotations/misato_ec_v2.json, `ec_labels.ec_result_problems`).
  * GO metric (coordinator, 2026-10-19): the GO column of the main and ablation tables is mAP
    ("GO mAP", ranked); the "Metrics: GO" tables list mAP, Fmax, top-1, μAP, F1, μF1 and the
    frequency floor (mAP, Fmax, top-1, μAP; not ranked). Old: top-1 hit.
  * ablation tables (final_fix_list A9, 2026-10-19): the old ablation table (RMSF + structure-split
    function columns for every ablation row; only the λ candidates had function results, so it
    could not be built) is replaced by a VQ table (e0 vs VQ-2048: L1 ids / perplexity, recon,
    RMSF of every dataset and token ANOVA at P = full and P = 1) and an SFTD table (SFTD off,
    λ = 0 … 1: best / stop epoch, recon at P = 1 / 2 / 5 / 10, P = 1 L1 ids, then RMSF and ANOVA
    at P = full / P = 1), no function columns; runs from --abl-vq / --abl-sftd (defaults: the
    eight output/ensembits_bm_* runs); inputs output/ablations/{recon_by_p,token_usage_by_p}.json
    and the token ANOVA JSONs, checked against each run's best.pt. --ablation-variants now only
    adds ProteinGym ablation rows (test compile).
  * function CV (authors' decision option A, 2026-10-19): the EC / GO cells of the sequence and
    structure tables (and of the ablation table) come from the cross-validation runs of
    data/misato_function_cv_v1.json (files `*_cv_<mode>_f<r>_*`, downstream/function_cv.py):
    per seed the mean over the 5 folds of the per-fold metric (only seeds with all 5 folds),
    then mean ± sample std over the seeds, in the test and the --split-metric val compile;
    per-fold values in `<out stem>.cv_folds.json`; CV results built on another CV file / task
    are refused ("Refused cells (function CV file)", fail the compile for paper rows); the
    signature check compares one hash per seed over its five folds. The random split has no
    EC / GO columns under CV. `--function-protocol split` compiles the old fixed-split cells.

Used by the reported model: yes. Its rows are labelled with --label (default
REPORTED_LABEL = "Ensembits (reported)") in every table; the current
(pre-fix) checkpoint's rows are "BioEmu-CATH+MISATO (pre-fix)" and
"↳ BioEmu+MISATO (dataset abl.) (pre-fix)".

Inputs / outputs
----------------
Reads (REPO_ROOT):
  output/ec_conv1d_misato/*.json, output/go_conv1d_misato/*.json,
  output/binding_misato/*.json, output/probe_rmsf_div/*.json,
  output/probe_rmsf_misato/*.json, output/probe_rmsf_bioemu/*.json,
  data/cached_descriptors/rmsf_residue_set_<dataset>.json (table residue sets), output/proteingym/per_assay_pg_<tag>*,
  ensembits-repro/runs/proteingym/<name>.json   (preprint reproduction: baselines + 192-D row)
Writes (--out overrides):
  camera_ready/results_bf65_testprotocol.md   (default, --rmsf-protocol test)
  camera_ready/results_bf65_valmetrics.md     (--split-metric val, 2026-10-19)
  rebuttal/results_bf65.md                    (--rmsf-protocol=val)
  <out stem>.cv_folds.json                    (per-fold EC / GO values, function CV)

Dependencies left in the research repo: none in code; the `ensembits-repro/`
ProteinGym JSONs and all result JSONs live under REPO_ROOT.

How to run
----------
    python downstream/_compile_rebuttal_results.py --variant <retrained run> --pgtag <PG tag> \\
        [--label "Ensembits (reported)"] \\
        [--ablation-variants <run>=<label>,...] [--ablation-pgtags <run>=<tag>,...] \\
        [--abl-vq <run>=<label>,...] [--abl-sftd <run>=<label>,...] [--no-ablation-tables] \\
        [--rmsf-protocol test|val] [--split-metric test|val] [--rmsf-label allframes] [--out <md>] \\
        [--function-protocol cv|split]
(original location: submission_exp/src/_compile_rebuttal_results.py in the research repo)
Overwrites the output file. Without --variant only the pre-fix rows are compiled
(with a warning).
"""
import argparse, json, glob, statistics as st, sys
from pathlib import Path
# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
# Result folders and the output file (all in the research repo).
EC = ROOT/"output/ec_conv1d_misato"; GO = ROOT/"output/go_conv1d_misato"; BD = ROOT/"output/binding_misato"
RMDIV = ROOT/"output/probe_rmsf_div"; RMMIS = ROOT/"output/probe_rmsf_misato"; PG = ROOT/"output/proteingym"
TAG = "stab"
# code_fix §7 / C4 (2026-10-15): RMSF protocol. "test" (default) = the corrected probes
# (early stopping on val, test_spearman from the "__test" files) ->
# camera_ready/results_bf65_testprotocol.md. "val" (--rmsf-protocol=val) = the old files
# and metric (best_val_spearman, epoch selected on the scored residues) ->
# rebuttal/results_bf65.md, unchanged.
# FIXED(code_fix P19, 2026-10-16): old: the protocol was read from sys.argv and OUT chosen
#   at import time (and the whole compile ran at import). New: module defaults below
#   ("test"); main() sets them from the CLI via set_rmsf_protocol().
OUT_TEST = ROOT/"camera_ready/results_bf65_testprotocol.md"
OUT_VAL = ROOT/"rebuttal/results_bf65.md"   # separate file; ghost refresher owns results.md
OUT_VALMETRIC = ROOT/"camera_ready/results_bf65_valmetrics.md"   # --split-metric val (2026-10-19)
RMSF_PROTOCOL = "test"; _RM_SUFFIX = "__test"; _RM_KEY = "test_spearman"
def set_rmsf_protocol(protocol):
    """Select the RMSF files / metric ("test" or "val") used by rmsf_div / rmsf_mis."""
    global RMSF_PROTOCOL, _RM_SUFFIX, _RM_KEY
    if protocol not in ("test", "val"): raise ValueError(protocol)
    RMSF_PROTOCOL = protocol
    _RM_SUFFIX = "__test" if protocol == "test" else ""
    _RM_KEY = "test_spearman" if protocol == "test" else "best_val_spearman"

# FIXED(function_reaudit pre-run fix 5, 2026-10-19): new. --split-metric val compiles every
#   column from the validation fields (function JSONs: the "val" block at the selected epoch;
#   RMSF "__test" files: best_val_spearman and val_spearman_per_protein_mean) and strips the
#   test fields from every JSON it reads (_load_result); ProteinGym (no validation split) is
#   left out. Same refusals as the test compile. This is the mode of the first pass.
def set_split_metric(mode):
    """'test' (default) or 'val': which split's fields every table column is compiled from."""
    global SPLIT_METRIC, _RM_KEY, PP_KEY
    if mode not in ("test", "val"): raise ValueError(mode)
    if mode == "val" and RMSF_PROTOCOL != "test":
        raise ValueError("--split-metric val reads the '__test' RMSF files (--rmsf-protocol test)")
    SPLIT_METRIC = mode
    if RMSF_PROTOCOL == "test":
        _RM_KEY = "test_spearman" if mode == "test" else "best_val_spearman"
        PP_KEY = ("test_spearman_per_protein_mean" if mode == "test"
                  else "val_spearman_per_protein_mean")

N_SEEDS = 10
def tag_of(v): return v.replace("final_model_","").replace("final_model","") or v
# Mean, SAMPLE std and count over the seeds found (NaN / None dropped).
# Paper: 2_results.tex result:rmsf — "averaged over 10 random seeds"; tab:rmsf
# reports mean ± std.
# FIXED(code_fix P9, 2026-10-16): old: statistics.pstdev (population std; the significance
#   tests used ddof = 1). New: statistics.stdev (sample std, ddof = 1) everywhere in the
#   tables; with a single seed the std is undefined (None, printed without "±").
def agg(vs):
    vs=[v for v in vs if v is not None and v==v]
    return (st.mean(vs), (st.stdev(vs) if len(vs)>1 else None), len(vs)) if vs else None
def cell(t):
    if t is None: return "—"
    if getattr(t, "refused", False):               # refused cell (provenance or seeds)
        return "✗" if t[2] == 0 else f"✗ ({t[2]})"
    if getattr(t, "constant", False):              # constant prediction (2026-10-18)
        return RC.CONSTANT_MARK + ("" if t[2] >= N_SEEDS else f" ({t[2]})")
    m,s,n=t
    body=f"{m:.3f}±{s:.3f}" if s is not None else f"{m:.3f}"
    return body+("" if n>=N_SEEDS else f" ({n})")

# FIXED(preretrain_audit_r2 F3, 2026-10-16): old = bold / underline over each whole column
#   (all Ensembits checkpoints, all baselines incl. the rebuttal-only *_pooled rows),
#   higher-is-better everywhere. New = ranked within each block of a table ("single-frame":
#   Ensembits P=1 + single-frame baselines; "multi-frame": Ensembits P=full + ProtProfileMD_K
#   + Vote_3Di; the "not ranked" block, i.e. pooled baselines and older checkpoints, gets no
#   marks), and columns whose name contains a LOWER_BETTER tag (MSE) rank ascending.
LOWER_BETTER = ("MSE",)
NOT_RANKED = "not ranked"
def lower_is_better(col): return any(t in col for t in LOWER_BETTER)
def _rank_cols(rows, cols=None):
    """rows: list of value-lists ((m,s,n) tuples, None or Refused). Per column, return
    (top1_row_idx, top2_row_idx) by mean; ascending for lower-is-better columns."""
    marks=[]; ncol=max((len(r) for r in rows), default=0)
    for c in range(ncol):
        # 2026-10-19: label-only reference columns (GO frequency floor) are not ranked
        if cols and c < len(cols) and any(t in cols[c] for t in NOT_RANKED_COL):
            marks.append((None, None)); continue
        low = bool(cols) and c < len(cols) and lower_is_better(cols[c])
        vv=sorted(((rows[i][c][0], i) for i in range(len(rows))
                   if c<len(rows[i]) and rows[i][c] is not None
                   and not getattr(rows[i][c], "refused", False)
                   and not getattr(rows[i][c], "constant", False)), reverse=not low)
        marks.append((vv[0][1] if vv else None, vv[1][1] if len(vv)>1 else None))
    return marks

def cellh(t, i, mk):
    """cell() + bold if row i is col top1, <u>underline</u> if top2."""
    s=cell(t)
    if t is None: return s
    if i==mk[0]: return f"**{s}**"
    if i==mk[1]: return f"<u>{s}</u>"
    return s

def block_table(cols, blocks):
    """Markdown table; blocks = [(name, [(row label, values)])]. Each block starts with a
    "| *— name —* |" row and is ranked on its own (the NOT_RANKED block is not ranked)."""
    L=[f"| model | {' | '.join(cols)} |","|---|"+"---|"*len(cols)]
    for name, rows in blocks:
        if not rows: continue
        L.append(f"| *— {name} —* |"+" |"*len(cols))
        if name != NOT_RANKED:
            rows=[(d, enforce_seeds(r)) for d, r in rows]     # H6: paper rows need every seed
        marks=(_rank_cols([r for _,r in rows], cols) if name != NOT_RANKED
               else [(None, None)]*len(cols))
        for ri,(disp,r) in enumerate(rows):
            L.append(f"| {disp} | "+" | ".join(cellh(r[c],ri,marks[c]) for c in range(len(r)))+" |")
    return "\n".join(L)

# ---- Ensembits variants and baselines in the function tables -------------------
# (display label, output/<variant> directory).
# FIXED(code_fix P16, 2026-10-16): old: the reported model was the fixed row
#   ("BioEmu-CATH+MISATO", dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO)
#   that compile_camera_ready_tables.py selected by label. New: the reported model's row
#   is added by main() from --variant / --pgtag / --label (REPORTED_LABEL); every row
#   below is a pre-fix checkpoint and carries the PREFIX suffix, so the camera-ready
#   selector cannot pick it. The flagship and the 192-D paper rows are kept for
#   reference; quant A-F, warm start and 1-frame rows are side experiments.
# FIXED(code_fix E1, 2026-10-20): VQ-2048 re-run = output/ensembits_bm_vq2048.
# FIXED(code_fix E2, 2026-10-20): SFTD λ sweep re-runs = output/ensembits_bm_lam*_rv.
REPORTED_LABEL = "Ensembits (reported)"
PREFIX = " (pre-fix)"
MODELS=[(d+PREFIX, v) for d, v in [
        ("256+φ/ψ FAPE-H (flagship)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT"),
        ("ENSEMBITS 192-D (paper)","final_model_combined_mdcath_misato_P10_esm3desc_K16_rvq_2048_128_128_varP_consMSE01_distillMax_P10_realbb"),
        ("BioEmu-CATH (dataset abl.)","dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus"),
        ("BioEmu-CATH+MISATO","dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO"),
        ("quant A: grad codebook (ctrl)","dih256_quantA_bioemuMISATO"),
        ("quant B: + rotation trick","dih256_quantB_bioemuMISATO"),
        ("quant C: + NSVQ","dih256_quantC_bioemuMISATO"),
        ("quant D: + rotation + NSVQ","dih256_quantfix_bioemuMISATO"),
        ("quant E: + AminoAseed proj","dih256_quantE_bioemuMISATO"),
        ("quant F: + AA proj L0 only","dih256_quantF_bioemuMISATO"),
        ("warm start: AE pretrain -> RVQ","dih256_warmstart_noSFTD_bioemuMISATO"),
        ("1-frame train (no dynamics)","dih256_fapeH_P1train_mdcathMISATO"),
        ("1-frame single-branch trainer","dih256_singleframe_mdcathMISATO"),
        ("VQ-2048 (single codebook)","dihedral256_fapeH_vq_2048_PRODSPLIT"),
        ("consMSE 0.5 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE05"),
        ("consMSE 1.0 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE10"),
        ("consMSE 1.5 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE15"),
        ("consMSE 2.0 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE20")]]
# Baseline result-file stems for the function tables (appendix.tex app:baselines);
# `*_pooled` = per-residue embedding averaged over the 8 frames (rebuttal).
BASELINES=["esm3struct","aminoaseed","3di_tokens","protoken","vote_3di","protprofile_K8","aa","random",
           "esm3struct_pooled","aminoaseed_pooled","3di_pooled","protoken_pooled"]
# Blocks of the paper tables (appendix.tex app:downstreams; 2_results.tex tab:rmsf), in the
# paper's row order; F3 2026-10-16. Pooled baselines are rebuttal-only (not ranked).
SINGLE_FRAME_BASE=["aa","3di_tokens","random","aminoaseed","protoken","esm3struct"]
MULTI_FRAME_BASE_FN=["protprofile_K8","vote_3di"]
POOLED_BASE=["esm3struct_pooled","aminoaseed_pooled","3di_pooled","protoken_pooled"]
# code_fix C13 (authors' decision 2026-10-15: the camera-ready compares with the released checkpoints only): the rebuttal-only
# "*_ourdata" rows (AminoAseed / VanillaVQ / ESM3-style retrained on the flagship
# corpus) were removed from BASELINES, RM_BASE and PG_ROWS.

# Seed suffixes in file names: variants use "_seed<s>", EC baselines "__seed<s>".
def _sd_ec(s): return "" if s==0 else f"_seed{s}"
def _sd_bl(s): return "" if s==0 else f"__seed{s}"
# FIXED(benchmark_audit D3, 2026-10-18): new. Every EC / GO / binding / affinity result JSON
#   must record the current MISATO split file (ensembits.paths.MISATO_SPLITS =
#   data/misato_splits_v2.json: "splits_sha256", data_versions.misato_split_problem). Results
#   built on the old data/misato_splits.json carry no record and are refused (REFUSED_MARK,
#   listed under "Refused cells (MISATO split file)"; main() returns 1 for paper rows, i.e.
#   reported / ablation / baseline rows). Old: every existing JSON was averaged. The MISATO
#   RMSF cells are checked against the same file through rmsf_common.current_splits_sha256.
_SPLIT_REFUSED = []     # (row stem, task, file name, reason)
def _split_refused(files, row, task):
    """True (reasons recorded) if a result JSON of the cell is not on the current split file."""
    from ensembits.data_versions import misato_split_problem
    bad=[]
    for f in files:
        if not f.exists(): continue
        try: why=misato_split_problem(_load_result(f))
        except Exception as e: why=f"unreadable ({type(e).__name__})"
        if why: bad.append((row, task, f.name, why))
    _SPLIT_REFUSED.extend(bad)
    return bool(bad)
# FIXED(benchmark_audit D4, 2026-10-18): new. A binding-site cell is refused (REFUSED_MARK,
#   listed under "Refused binding-site cells"; main() returns 1 for paper rows) when any of its
#   result JSONs was not built on the current binding-site labels
#   (ensembits.misato_binding_labels.MISATO_BINDING_SITE = data/annotations/
#   misato_binding_site_v2.npz, sha256 recorded as "binding_labels_sha256"). Results on the old
#   data/cached_descriptors/misato_binding_site.npz (pocket residues misaligned) have no record.
#   Affinity cells do not depend on these labels and are not checked here.
_BIND_REFUSED = []      # (row stem, file name, reason)
def _bind_label_refused(files, row):
    """True (reasons recorded) if a result JSON of the cell is not on the current binding labels."""
    from ensembits.misato_binding_labels import binding_label_problem
    bad=[]
    for f in files:
        if not f.exists(): continue
        try: why=binding_label_problem(_load_result(f))
        except Exception as e: why=f"unreadable ({type(e).__name__})"
        if why: bad.append((row, f.name, why))
    _BIND_REFUSED.extend(bad)
    return bool(bad)
# FIXED(function_reaudit pre-run fixes 2 / 5, 2026-10-19): readers rewritten around one cell
#   function (_fn_cell). New: (a) `--split-metric val` compiles every function and RMSF
#   column from the VALIDATION fields of the result JSONs (r["val"] = metrics at the selected
#   epoch; RMSF best_val_spearman / val_spearman_per_protein_mean); the test fields are
#   removed from every JSON right after it is read (_load_result), so they cannot enter the
#   table; (b) every function result must carry the current protocol tag (function_common.
#   FUNCTION_PROTOCOL), verified feature inputs and, for affinity, the current label / MACCS
#   sha256 (refused otherwise, listed under "Refused cells (function protocol)"); (c) every
#   method of a cell must share the class list and the item hashes (cell_signature): the
#   reference is the signature of most ranked rows of the split's main table; deviating rows
#   are refused (listed under "Refused cells (items / classes)"); (d) the readers take the
#   metric as an argument (mAP, μAP, F1, μF1, Fmax, frequency floor, AP, per-protein AUROC
#   columns in the "Metrics:" tables). Old: one metric per task, test fields only, no item /
#   class / protocol checks.
SPLIT_METRIC = "test"          # "test" (default) or "val" (--split-metric val)
_FN_REFUSED = []               # (row stem, task, file name, reason)
_ITEM_REFUSED = []             # (row stem, split, task column, reason)
_SIG = {}                      # (row stem, split, task column) -> set of signatures
_REF_SIG = {}                  # (split, task column) -> reference signature (or None = no majority)
# Keys holding TEST metrics in function / RMSF result JSONs; in val mode they are deleted from
# every loaded JSON (outside the "val" blocks) before anything is read.
TEST_METRIC_KEYS = {"top1_hit","macro_ap","micro_ap","macro_f1@0.5","micro_f1@0.5","fmax",
                    "fmax_threshold","n_classes_in_test","majority_top1_hit","frequency_floor",
                    "auroc","ap","auroc_per_protein_mean","auroc_per_protein_median","r2",
                    "spearman","mse","test_spearman","test_pearson",
                    "test_spearman_per_protein_mean","test_spearman_per_protein_median",
                    "rescue_original"}
def _strip_test(o):
    """Copy of a result JSON without its test-metric keys (the 'val' blocks are kept)."""
    if isinstance(o, dict):
        return {k: (v if k == "val" else _strip_test(v)) for k, v in o.items()
                if k not in TEST_METRIC_KEYS}
    if isinstance(o, list):
        return [_strip_test(v) for v in o]
    return o
def _load_result(f):
    """json.load of a result file; in val mode without its test-metric fields."""
    d = json.load(open(f))
    return _strip_test(d) if SPLIT_METRIC == "val" else d
def _metric_value(c, metric):
    """`metric` of one result container: c itself (test) or c['val'] (val mode).
    'floor:<k>' = key k of the GO frequency floor."""
    if not isinstance(c, dict): return None
    blk = c if SPLIT_METRIC == "test" else c.get("val")
    if not isinstance(blk, dict): return None
    if metric.startswith("floor:"):
        fl = blk.get("frequency_floor")
        return fl.get(metric[6:]) if isinstance(fl, dict) else None
    v = blk.get(metric)
    return v if isinstance(v, (int, float)) else None
def _container(js, task, sp, depth):
    if task == "EC": return (js.get("by_depth") or {}).get(str(depth))
    if task == "GO": return js
    return (js.get("results") or {}).get(f"{sp}_{task}")
def _taskcol(task, depth): return f"EC d{depth}" if task == "EC" else task
def _fn_refused(files, row, task):
    """True (reasons recorded) if a result JSON fails function_common.function_result_problems."""
    from function_common import function_result_problems
    bad=[]
    for f in files:
        if not f.exists(): continue
        try: why=function_result_problems(_load_result(f), "affinity" if task=="affinity" else None)
        except Exception as e: why=[f"unreadable ({type(e).__name__})"]
        if why: bad.append((row, task, f.name, why[0]))
    _FN_REFUSED.extend(bad)
    return bool(bad)
def _fn_cell(files, row, task, sp, depth, metric, signatures=True):
    """One function cell over the seed files: provenance refusals, then the metric (test or
    val block) aggregated over seeds; the cell signatures are kept for the item check.
    signatures=False (CV cells, 2026-10-19): the per-fold signatures are not stored (the CV
    cell stores one signature per seed over its five folds)."""
    from function_common import cell_signature
    label = {"EC": "EC", "GO": "GO"}.get(task, task)
    if _split_refused(files, row, label): return Refused()             # D3
    if task == "EC" and _ec_refused(files, row): return Refused()      # D8
    if task == "binding" and _bind_label_refused(files, row): return Refused()   # D4
    if task == "GO":                                                    # D7
        from go_labels import go_result_problems
        bad=[]
        for f in files:
            if f.exists():
                why=go_result_problems(_load_result(f))
                if why: bad.append((row, f.name, why[0]))
        if bad:
            _GO_REFUSED.extend(bad); return Refused()
    if _fn_refused(files, row, task): return Refused()                  # 2026-10-19
    o=[]; sigs=_SIG.setdefault((row, sp, _taskcol(task, depth)), set()) if signatures else set()
    for f in files:
        if not f.exists(): continue
        js=_load_result(f); c=_container(js, task, sp, depth)
        if c is None: continue
        sigs.add(cell_signature(js, task, sp, depth))
        v=_metric_value(c, metric)
        if v is not None: o.append(v)
    return agg(o)
# ---- model readers ----
# Each reader collects one metric over seeds 0-9 and aggregates it (default argument: EC/GO =
# top-1 hit, binding = AUROC, affinity = Spearman; all with the "stab" recipe tag). The tables
# pass the metric explicitly (MAIN_SPECS: EC = mAP since 2026-10-19).
# FIXED(benchmark_audit D8, 2026-10-18): new. An EC cell is refused (REFUSED_MARK, listed
#   under "Refused EC cells"; main() returns 1 for paper rows) when any of its result JSONs
#   was not built on the current EC labels (data/annotations/misato_ec_v2.json, sha256
#   recorded by the probes in "ec_labels"; results on the old misato_ec.json, whose
#   polyprotein chains carried the whole precursor's EC numbers and whose wildcard EC
#   numbers were classes, have no such block). Old: every existing EC JSON was averaged.
_EC_REFUSED = []        # (row stem, file name, reason)
def _ec_refused(files, row):
    """True (reasons recorded) if a result JSON of the cell is not on the current EC labels."""
    from ec_labels import ec_result_problems   # downstream/ (on sys.path below)
    bad=[]
    for f in files:
        if not f.exists(): continue
        try: why=ec_result_problems(_load_result(f))
        except Exception as e: why=[f"unreadable ({type(e).__name__})"]
        if why: bad.append((row, f.name, why[0]))
    _EC_REFUSED.extend(bad)
    return bool(bad)
# FIXED(benchmark_audit D7, 2026-10-18): new. A GO cell is refused (REFUSED_MARK, listed
#   under "Refused GO cells"; main() returns 1 for paper rows, i.e. reported / ablation /
#   baseline rows, as for RMSF) when any of its result JSONs was not built
#   on the current GO labels (data/annotations/misato_go_mf_v2.json, sha256 recorded by the
#   probes in "go_labels"; results on the old misato_go.json have no such block).
#   Old: every existing GO JSON was averaged.
_GO_REFUSED = []        # (row stem, file name, reason)
def _ec_files(stem_fn): return [EC/stem_fn(s) for s in range(10)]
# FIXED(function CV, authors' decision option A, 2026-10-19): on the sequence / structure
#   "splits" the EC and GO cells are compiled from the cross-validation runs
#   (data/misato_function_cv_v1.json; files *_cv_<mode>_f<r>_*): per seed the mean over the
#   five folds of the per-fold metric (a seed counts only with all five folds), then mean and
#   sample std (ddof = 1) over the seeds (function_cv.aggregate_cv); the per-fold values go to
#   the "<out stem>.cv_folds.json" sidecar. A CV result not built on the current CV file is
#   refused ("Refused cells (function CV file)"). --function-protocol split restores the old
#   fixed-split cells; under cv the random split has no EC / GO columns.
FN_CV = True
_CV_REFUSED = []       # (row stem, task, file name, reason)
_CV_FOLDS = {}         # "row|mode|task column|metric" -> {"per_fold": .., "seed_means": ..}
def _cv_on(sp):
    from function_cv import CV_MODES
    return FN_CV and sp in CV_MODES
def _cv_files(stem_fn):
    """{seed: {fold: path}} of a CV cell; stem_fn(seed, cv_tag) -> path."""
    from function_cv import N_FOLDS
    return {s: {r: stem_fn(s, r) for r in range(N_FOLDS)} for s in range(N_SEEDS)}
def _fn_cell_cv(fs, row, task, sp, depth, metric):
    """A CV cell: refusals on every fold file, then function_cv.aggregate_cv over (seed, fold)."""
    import hashlib
    from function_common import cell_signature
    from function_cv import cv_result_problem, aggregate_cv
    flat=[f for s in fs for f in fs[s].values()]
    bad=[]
    for f in flat:
        if not f.exists(): continue
        try: why=cv_result_problem(_load_result(f), task.lower())
        except Exception as e: why=f"unreadable ({type(e).__name__})"
        if why: bad.append((row, task, f.name, why))
    if bad:
        _CV_REFUSED.extend(bad); return Refused()
    pre=_fn_cell(flat, row, task, sp, depth, metric, signatures=False)
    if getattr(pre, "refused", False): return pre
    vals={}; sigs=_SIG.setdefault((row, sp, _taskcol(task, depth)), set())
    for s, folds in fs.items():
        fsig=[]
        for r, f in folds.items():
            if not f.exists(): continue
            js=_load_result(f); c=_container(js, task, sp, depth)
            if c is None: continue
            fsig.append((r, cell_signature(js, task, sp, depth)))
            v=_metric_value(c, metric)
            if v is not None: vals.setdefault(s, {})[r]=v
        if fsig:
            sigs.add(None if any(g is None for _, g in fsig) else
                     hashlib.sha256(repr(sorted(fsig)).encode()).hexdigest())
    cellv, per_fold=aggregate_cv(vals)
    _CV_FOLDS[f"{row}|{sp}|{_taskcol(task, depth)}|{metric}"]={
        "per_fold": {str(k): v for k, v in per_fold.items()},
        "seed_fold_values": {str(s): {str(r): v for r, v in d.items()} for s, d in vals.items()}}
    return cellv
def m_ec(V,p,sp,d,metric="top1_hit"):
    pih="_P1inf" if p else ""
    if _cv_on(sp):
        fs=_cv_files(lambda s,r: EC/f"tok_{tag_of(V)}{pih}_cv_{sp}_f{r}_{TAG}{_sd_ec(s)}__summary.json")
        return _fn_cell_cv(fs, _rowid(tag_of(V), p), "EC", sp, d, metric)
    spf=f"_{sp}" if sp!="sequence" else ""
    files=[EC/f"tok_{tag_of(V)}{pih}{spf}_{TAG}{_sd_ec(s)}__summary.json" for s in range(10)]
    return _fn_cell(files, _rowid(tag_of(V), p), "EC", sp, d, metric)
def m_go(V,p,sp,metric="top1_hit"):
    pih="_P1inf" if p else ""
    if _cv_on(sp):
        fs=_cv_files(lambda s,r: GO/f"go_{tag_of(V)}{pih}_cv_{sp}_f{r}_{TAG}{_sd_ec(s)}_top50.json")
        return _fn_cell_cv(fs, _rowid(tag_of(V), p), "GO", sp, None, metric)
    spf=f"_{sp}" if sp!="sequence" else ""
    files=[GO/f"go_{tag_of(V)}{pih}{spf}_{TAG}{_sd_ec(s)}_top50.json" for s in range(10)]
    return _fn_cell(files, _rowid(tag_of(V), p), "GO", sp, None, metric)
def m_bind(V,p,sp,task,metric,lig=True):
    """lig=False: the protein-only affinity control (D5; files without "_lig")."""
    pih="_P1inf" if p else ""; lg="_lig" if lig else ""
    files=[BD/f"{tag_of(V)}{pih}{lg}_{TAG}{_sd_ec(s)}.json" for s in range(10)]
    return _fn_cell(files, _rowid(tag_of(V), p, lig), task, sp, None, metric)
# ---- baseline readers ----
def b_ec(F,sp,d,metric="top1_hit"):
    if _cv_on(sp):     # function CV (2026-10-19)
        fs=_cv_files(lambda s,r: EC/f"{F}__cv_{sp}_f{r}__{TAG}{_sd_bl(s)}__summary.json")
        return _fn_cell_cv(fs, f"baseline_{F}", "EC", sp, d, metric)
    spf=f"__{sp}" if sp!="sequence" else ""
    files=[EC/f"{F}{spf}__{TAG}{_sd_bl(s)}__summary.json" for s in range(10)]
    return _fn_cell(files, f"baseline_{F}", "EC", sp, d, metric)
def b_go(F,sp,metric="top1_hit"):
    if _cv_on(sp):     # function CV (2026-10-19)
        fs=_cv_files(lambda s,r: GO/f"go_{F}_top50_cv_{sp}_f{r}_{TAG}{_sd_ec(s)}.json")
        return _fn_cell_cv(fs, f"baseline_{F}", "GO", sp, None, metric)
    spf=f"_{sp}" if sp!="sequence" else ""
    files=[GO/f"go_{F}_top50{spf}_{TAG}{_sd_ec(s)}.json" for s in range(10)]
    return _fn_cell(files, f"baseline_{F}", "GO", sp, None, metric)
def b_bind(F,sp,task,metric,lig=True):
    """lig=False: protein-only control (D5); F == 'ligand_only': the MACCS-only control."""
    lg="_lig" if (lig and F!="ligand_only") else ""
    files=[BD/f"baseline_{F}{lg}_{TAG}{_sd_ec(s)}.json" for s in range(10)]
    return _fn_cell(files, f"baseline_{F}" + ("" if lig else " (control)"), task, sp, None, metric)
def _rowid(stem, p, lig=True):
    return f"{stem}{'_P1inf' if p else ''}" + ("" if lig else " (control)")

# ---- Function table (one per split) -----------------------------------------
# Paper: appendix.tex app:downstreams / app:experiments (EC, GO, binding site,
# binding affinity), sequence / structure / random splits.
# F3 (2026-10-16): affinity R2 and MSE added (appendix tab:affinity_* reports all three;
#   MSE is lower-is-better, ranked ascending).
# Column specs: (header, task, EC depth, metric key).
# FIXED(function_reaudit EC metric, authors' decision 2026-10-19): the EC d1 / d2 / d3 columns
#   of the main tables (and the ablation table) show mAP (macro-averaged average precision over
#   the EC classes of the depth: a ranking metric over every class, not only the top-scored
#   one), with Fmax as the secondary EC metric in the "Metrics: EC d<k>" tables; bold /
#   underline rank the mAP. Old: top-1 hit (headers "EC d1" ...), which a head can score well
#   on by predicting the most frequent class.
EC_PRIMARY_METRIC = "macro_ap"
# FIXED(coordinator, 2026-10-19): the GO column of the main tables (and the ablation table)
#   shows mAP like EC ("GO mAP"); Fmax, top-1 and the frequency floor are in the "Metrics: GO"
#   tables. Old: top-1 hit (header "GO"), which sat at the floor for every method.
GO_PRIMARY_METRIC = "macro_ap"
MAIN_SPECS=[("EC d1 mAP","EC",1,EC_PRIMARY_METRIC),("EC d2 mAP","EC",2,EC_PRIMARY_METRIC),
            ("EC d3 mAP","EC",3,EC_PRIMARY_METRIC),
            ("GO mAP","GO",None,GO_PRIMARY_METRIC),("Bind AUROC","binding",None,"auroc"),
            ("Aff Spr","affinity",None,"spearman"),("Aff R2","affinity",None,"r2"),
            ("Aff MSE","affinity",None,"mse")]
COLS=[c[0] for c in MAIN_SPECS]
# FIXED(function_reaudit pre-run fix 5, 2026-10-19): new metric tables (Q17: no code compiled
#   mAP / μAP / F1 / μF1 / Fmax for EC and GO, nor binding AP); the GO frequency floor
#   (label-only predictor) is shown per row as a check that every row has the same items, and
#   is not ranked.
ML_METRICS=[("top-1","top1_hit"),("mAP","macro_ap"),("μAP","micro_ap"),("F1@0.5","macro_f1@0.5"),
            ("μF1@0.5","micro_f1@0.5"),("Fmax","fmax")]
# EC metric tables (2026-10-19, EC metric decision): mAP first (primary), Fmax second, then the
#   other metrics, and the majority-class top-1 (the most frequent training class predicted for
#   every complex; labels only) as an unranked reference column.
EC_MAJORITY_COL = "majority top-1 (ref.)"
EC_METRICS=[("mAP","macro_ap"),("Fmax","fmax"),("top-1","top1_hit"),("μAP","micro_ap"),
            ("F1@0.5","macro_f1@0.5"),("μF1@0.5","micro_f1@0.5"),
            (EC_MAJORITY_COL,"majority_top1_hit")]
# GO metric tables (2026-10-19): mAP first (primary), Fmax, top-1, then the others.
GO_METRICS=[("mAP","macro_ap"),("Fmax","fmax"),("top-1","top1_hit"),("μAP","micro_ap"),
            ("F1@0.5","macro_f1@0.5"),("μF1@0.5","micro_f1@0.5")]
# Label-only reference columns, never ranked: GO frequency floor, EC majority-class top-1.
NOT_RANKED_COL = ("floor", "majority")
DETAIL_TABLES=([(f"EC d{d}", [(h,"EC",d,m) for h,m in EC_METRICS]) for d in (1,2,3)]
               + [("GO", [(h,"GO",None,m) for h,m in GO_METRICS]
                   + [(f"floor {h}","GO",None,f"floor:{m}") for h,m in
                      (("mAP","macro_ap"),("Fmax","fmax"),("top-1","top1_hit"),("μAP","micro_ap"))])]
               + [("Binding site", [("AUROC","binding",None,"auroc"),("AP","binding",None,"ap"),
                                    ("per-protein AUROC","binding",None,"auroc_per_protein_mean")])])
def mrow(V,p,sp,specs=MAIN_SPECS):
    return [m_ec(V,p,sp,d,m) if t=="EC" else m_go(V,p,sp,m) if t=="GO" else m_bind(V,p,sp,t,m)
            for _,t,d,m in specs]
def brow(F,sp,specs=MAIN_SPECS):
    return [b_ec(F,sp,d,m) if t=="EC" else b_go(F,sp,m) if t=="GO" else b_bind(F,sp,t,m)
            for _,t,d,m in specs]

def _set_reference(sp, rowids, specs):
    """Reference signature per task column of a split: the one most ranked rows carry
    (None when two signatures tie). Set once per split and task column."""
    from collections import Counter
    for _,t,d,_m in specs:
        tc=_taskcol(t,d)
        if (sp,tc) in _REF_SIG: continue
        cnt=Counter()
        for r in rowids:
            sg=_SIG.get((r,sp,tc)) or set()
            if len(sg)==1: cnt[next(iter(sg))]+=1
        if not cnt: continue
        top=cnt.most_common(2)
        _REF_SIG[(sp,tc)] = top[0][0] if len(top)==1 or top[0][1]>top[1][1] else None
def _check_items(rowid, sp, specs, values):
    """Values with every cell refused whose result JSONs do not carry the reference class list
    and item hashes of the split (or whose seeds disagree)."""
    out=[]
    for (_,t,d,_m),v in zip(specs, values):
        tc=_taskcol(t,d); sg=_SIG.get((rowid,sp,tc)) or set()
        if v is None or getattr(v,"refused",False) or not sg:
            out.append(v); continue
        ref=_REF_SIG.get((sp,tc), "unset")
        why=None
        if len(sg)>1: why=f"seeds carry {len(sg)} different item / class signatures"
        elif None in sg: why="no item / class record"
        elif ref is None: why="ranked rows disagree on items / classes (no majority)"
        elif ref!="unset" and next(iter(sg))!=ref: why="items / classes differ from the other methods"
        if why:
            if (rowid,sp,tc,why) not in _ITEM_REFUSED: _ITEM_REFUSED.append((rowid,sp,tc,why))
            out.append(Refused(v[2]))
        else: out.append(v)
    return out

# ---- Seed completeness (code_fix P16) -----------------------------------------
# FIXED(code_fix P16, 2026-10-16): new. Every cell of a paper row (reported model,
#   baselines, ablations) with fewer than N_SEEDS seeds (or none) is collected here and
#   listed in the output and on stdout, instead of only showing "(k)" / "—" in a cell.
def check_seeds(table, row, cols, values, skip_cols=()):
    """-> [(table, row, column, n_seeds)] for every cell with < N_SEEDS seeds."""
    out=[]
    for c,v in zip(cols, values):
        if c in skip_cols: continue
        n = 0 if v is None else v[2]
        if n < N_SEEDS: out.append((table, row, c, n))
    return out

def split_table(sp, models, paper_models=(), incomplete=None, specs=MAIN_SPECS, check=True):
    """Function table of one split, in the paper's two blocks (F3): single-frame (single-frame
    baselines + paper models at P=1) and multi-frame (ProtProfileMD_K8, Vote_3Di + paper
    models at P=full), each ranked on its own; pooled baselines and older checkpoints follow
    in a block that is not ranked. `specs`: the columns (MAIN_SPECS or a metric table);
    check=False skips the seed-completeness list (metric tables repeat the main cells).
    Function CV (2026-10-19): on a split that is not a CV mode (random) the EC / GO columns
    are left out under FN_CV."""
    if FN_CV and not _cv_on(sp):
        specs=[c for c in specs if c[1] not in ("EC","GO")]
        if not specs: return "*(EC / GO: cross-validation in the sequence and structure modes only)*"
    cols=[c[0] for c in specs]
    entries=[]     # (block, display, rowid, values, paper?)
    for F in SINGLE_FRAME_BASE + MULTI_FRAME_BASE_FN:
        entries.append(("single" if F in SINGLE_FRAME_BASE else "multi", F, f"baseline_{F}",
                        brow(F,sp,specs), True))
    for disp,V in models:
        for p,pt in [(True,"P=1"),(False,"P=full")]:
            paper = disp in paper_models
            entries.append((("single" if p else "multi") if paper else "other",
                            f"{disp} · {pt}", _rowid(tag_of(V),p), mrow(V,p,sp,specs), paper))
    for F in POOLED_BASE:
        entries.append(("other", F, f"baseline_{F}", brow(F,sp,specs), False))
    _set_reference(sp, [e[2] for e in entries if e[0] in ("single","multi")], specs)
    single, multi, other = [], [], []
    for blk,disp,rid,r,paper in entries:
        r=_check_items(rid, sp, specs, r)
        if check and incomplete is not None and paper:
            incomplete += check_seeds(f"{sp} split", disp, cols, r)
        if blk=="other":
            if any(v is not None for v in r): other.append((disp, r))
            continue
        if ALLOW_INCOMPLETE and not any(v is not None for v in r): continue
        (single if blk=="single" else multi).append((disp, r))
    return block_table(cols, [("single-frame", single), ("multi-frame", multi), (NOT_RANKED, other)])

# ---- Affinity controls (D5) ----------------------------------------------------
# FIXED(function_reaudit pre-run fix 4, 2026-10-19): new. Non-paper control rows of the
#   affinity task on the same complexes (finite label, MACCS, >= 10 residues), splits, seeds,
#   epochs and early stopping: ligand only (MACCS -> MLP, baseline_ligand_only), protein only
#   (no MACCS: ESM3struct, aa and the reported Ensembits model at P=full / P=1), next to the
#   same methods with MACCS. Not ranked, no seed-completeness entry (not paper rows).
CONTROL_PROTEIN_BASE=["esm3struct","aa"]
AFF_SPECS=[("Aff Spr","affinity",None,"spearman"),("Aff R2","affinity",None,"r2"),
           ("Aff MSE","affinity",None,"mse")]
def control_table(sp, reported):
    rows=[("ligand only (MACCS)", f"baseline_ligand_only (control)",
           [b_bind("ligand_only",sp,"affinity",m,lig=False) for _,_,_,m in AFF_SPECS])]
    for F in CONTROL_PROTEIN_BASE:
        rows.append((f"{F} · protein only", f"baseline_{F} (control)",
                     [b_bind(F,sp,"affinity",m,lig=False) for _,_,_,m in AFF_SPECS]))
        rows.append((f"{F} · protein + MACCS", f"baseline_{F}",
                     [b_bind(F,sp,"affinity",m) for _,_,_,m in AFF_SPECS]))
    if reported:
        for p,pt in [(True,"P=1"),(False,"P=full")]:
            rows.append((f"{reported[0]} · {pt} · protein only", _rowid(tag_of(reported[1]),p,False),
                         [m_bind(reported[1],p,sp,"affinity",m,lig=False) for _,_,_,m in AFF_SPECS]))
            rows.append((f"{reported[0]} · {pt} · protein + MACCS", _rowid(tag_of(reported[1]),p),
                         [m_bind(reported[1],p,sp,"affinity",m) for _,_,_,m in AFF_SPECS]))
    rows=[(d, _check_items(rid, sp, AFF_SPECS, r)) for d,rid,r in rows]
    return block_table([c[0] for c in AFF_SPECS], [(NOT_RANKED, rows)])

# ---- RMSF ----
# Paper: 2_results.tex result:rmsf, Table tab:rmsf (mdCATH-div P=10, MISATO
# seq / struct P=8; P=full and P=1 rows).
# FIXED(require_fix #17, tex pass 2026-10-26): RMSF table updated with 256-D numbers.
# FIXED(code_fix P16, 2026-10-16): pre-fix rows relabelled (PREFIX); the reported row is
#   added by main() from --variant (stem = tag_of(variant)).
# FIXED(code_fix E1, 2026-10-20): VQ-2048 re-run done.
# FIXED(code_fix E2, 2026-10-20): SFTD λ sweep re-runs done.
RM_MODELS=[(d+PREFIX, v) for d, v in [
           ("256+φ/ψ FAPE-H (flagship)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT"),
           ("ENSEMBITS 192-D (paper)","combined_mdcath_misato_P10_esm3desc_K16_rvq_2048_128_128_varP_consMSE01_distillMax_P10_realbb"),
           ("BioEmu-CATH (dataset abl.)","dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus"),
           ("BioEmu-CATH+MISATO","dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO"),
           ("quant A: grad codebook (ctrl)","dih256_quantA_bioemuMISATO"),
           ("quant B: + rotation trick","dih256_quantB_bioemuMISATO"),
           ("quant C: + NSVQ","dih256_quantC_bioemuMISATO"),
           ("quant D: + rotation + NSVQ","dih256_quantfix_bioemuMISATO"),
           ("quant E: + AminoAseed proj","dih256_quantE_bioemuMISATO"),
           ("quant F: + AA proj L0 only","dih256_quantF_bioemuMISATO"),
           ("warm start: AE pretrain -> RVQ","dih256_warmstart_noSFTD_bioemuMISATO"),
           ("1-frame train (no dynamics)","dih256_fapeH_P1train_mdcathMISATO"),
           ("1-frame single-branch trainer","dih256_singleframe_mdcathMISATO"),
           ("noSFTD (distill off)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_noSFTD"),
           ("all-off (no varP/distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_noVarP_noSFTD"),
           ("VQ-2048 (single codebook)","dihedral256_fapeH_vq_2048_PRODSPLIT"),
           ("consMSE 0.5 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE05"),
           ("consMSE 1.0 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE10"),
           ("consMSE 1.5 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE15"),
           ("consMSE 2.0 (↑distill)","dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT_consMSE20")]]
# Baseline stems for the RMSF table (probe_rmsf_baselines.py / probe_rmsf_misato.py).
RM_BASE=["esm3struct","aminoaseed","3di_tokens","protoken","vote_3di","protprofile_K10","aa","random"]
# FIXED(preretrain_audit_r2 F3, 2026-10-16): old = columns dataset x P (8 columns), the
#   single-frame baselines' only value in the "P=full" columns. New = the layout of tab:rmsf:
#   columns = datasets, rows = methods in two blocks, single-frame (baselines + Ensembits
#   P=1) and multi-frame (ProtProfileMD_K, Vote_3Di + Ensembits P=full), each ranked on its
#   own; BioEmu-CATH (phase 2) is the fourth column.
RM_COLS=["mdCATH-div","MISATO-seq","MISATO-struct","BioEmu-CATH"]
RM_SINGLE_BASE=["aa","3di_tokens","random","aminoaseed","protoken","esm3struct"]
RM_MULTI_BASE=["protprofile_K10","vote_3di"]
def rm_base_disp(b): return "protprofile (K10 div / K8 misato / K10 bioemu)" if b=="protprofile_K10" else b
RMBIO = ROOT/"output"/"probe_rmsf_bioemu"
# K = number of input frames per dataset: mdCATH-div and BioEmu-CATH give the models 10
# frames (K10); MISATO gives 8 (K8 matches the 8 input frames; the MISATO RMSF label itself
# uses all 100 MD frames since 2026-10-16, H9). Read div / bioemu = K10, misato = K8.
MIS_STEM={"protprofile_K10":"protprofile_K8"}
def rd(k,f):
    try: return _load_result(f).get(k)
    except Exception: return None

# FIXED(preretrain_audit_r2 Q17 / Q2 / Q3, 2026-10-16): old = the RMSF readers averaged
#   whatever JSONs existed (label, residue set and checkpoint never checked) and reported
#   only the pooled Spearman. New (test protocol): every JSON of a cell must carry the
#   expected RMSF label (--rmsf-label, default allframes), the residue set of the table
#   (the current data_prep/build_rmsf_residue_sets.py file of that dataset; if it has not
#   been built, the hash of the first accepted cell) and, for Ensembits rows, one checkpoint
#   block (equal to output/<variant>/best.pt when that folder exists). A cell that fails is
#   not printed: it shows REFUSED_MARK and the reasons are listed under the table; main()
#   returns 1 when a reported / baseline / ablation cell is refused. The per-protein
#   Spearman (mean over proteins) is aggregated the same way (key PP_KEY).
#   --rmsf-protocol=val reads the old files without these checks (reproduction only).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rmsf_common as RC   # noqa: E402
RMSF_LABEL = RC.RMSF_LABEL_DEFAULT          # expected label (--rmsf-label)
PP_KEY = "test_spearman_per_protein_mean"
REFUSED_MARK = "✗"
class Refused(tuple):
    """A refused cell (prints REFUSED_MARK; never ranked). n = seeds found (0 for a
    provenance refusal; k < N_SEEDS for an incomplete paper cell, printed "✗ (k)")."""
    refused = True
    def __new__(cls, n=0): return super().__new__(cls, (float("nan"), None, n))
# FIXED(constant-prediction rule, 2026-10-18, authors' decision): new. An RMSF cell whose
#   seeds are ALL NaN because the probe predicted a constant with every residue mapped to
#   one token (rmsf_common.constant_prediction_seed: the probes' constant_prediction /
#   n_unique_inputs fields, or for older results best_epoch 0 + NaN + a one-token token
#   cache) is shown as "—†" with the footnote rmsf_common.CONSTANT_NOTE, is not ranked and
#   its seeds are not counted as missing (n = seeds found). Any other NaN seed refuses the
#   cell (old: dropped by the mean, so the cell showed fewer seeds).
class Constant(tuple):
    """An all-constant-prediction cell (prints CONSTANT_MARK; never ranked; n = seeds)."""
    constant = True
    def __new__(cls, n=0): return super().__new__(cls, (float("nan"), None, n))
_RM_CONSTANT = []       # (dataset, stem, column) shown as CONSTANT_MARK
# FIXED(preretrain_audit_r2 H6, 2026-10-16): old = a paper-row cell with fewer than N_SEEDS
#   seeds was printed "m±s (k)", ranked, and only listed under "Seed completeness". New = it
#   is refused like a mismatched cell (✗ (k), not ranked) and main() returns 1;
#   --allow-incomplete restores the old display for previews.
ALLOW_INCOMPLETE = False
def enforce_seeds(values):
    """Paper-row values with every cell of < N_SEEDS seeds (or none) refused (H6)."""
    if ALLOW_INCOMPLETE: return values
    return [v if getattr(v, "refused", False) or (v is not None and v[2] >= N_SEEDS)
            else Refused(0 if v is None else v[2]) for v in values]   # Constant: n = seeds
_RM_REFUSED = []        # (dataset, stem, column, [reasons])
_RM_TABLE_HASH = {}     # dataset -> residue-set hash every row must carry
_RM_TABLE_LABEL = {}    # dataset -> label content hash every row must carry (F5)
_RM_TABLE_SAME = {}     # (dataset, key) -> value every row must share (G4 / G5, 2026-10-17)
def reset_rmsf_checks():
    _RM_REFUSED.clear(); _RM_TABLE_HASH.clear(); _RM_TABLE_LABEL.clear(); _RM_TABLE_SAME.clear()
    _RM_CONSTANT.clear()
def _table_same(dataset, records):
    """FIXED(preretrain_audit_r2 G4 / G5, 2026-10-17): new. The split file (splits_sha256) and
    the probe settings (max_epochs, patience, min_delta, batch_size, selection) must be the
    same in every row of a table.
    FIXED(preretrain_audit_r2 K1 / K2, 2026-10-17): old = the first row read was the
    reference. New = the split hash must be the CURRENT split file's (the first row's only
    if that file does not exist), and the settings must be the probes' defaults
    (rmsf_common.DEFAULT_PROBE_SETTINGS), so the deviant row is the one refused."""
    probs=[]
    want_settings=json.dumps([RC.DEFAULT_PROBE_SETTINGS[k] for k in RC.PROBE_SETTINGS])
    cur=RC.current_splits_sha256(dataset)
    for name, js in records:
        val=js.get("splits_sha256")
        if cur is not None:
            if val != cur: probs.append(f"{name}: splits_sha256 is not the current split file's")
        else:
            ref=_RM_TABLE_SAME.setdefault((dataset, "splits_sha256"), (val, name))
            if ref[0] != val: probs.append(f"{name}: splits_sha256 differs from {ref[1]}")
        got=json.dumps([js.get(k) for k in RC.PROBE_SETTINGS])
        if got != want_settings:
            probs.append(f"{name}: settings {got} differ from the defaults {want_settings}")
    return probs
def _table_label(dataset, records):
    """The label content hash of a table: the current label file's (F5), else the first."""
    if dataset not in _RM_TABLE_LABEL:
        lp = RC.RMSF_LABEL_FILES.get(dataset, {}).get(RMSF_LABEL)
        h = None
        if lp is not None and lp.exists():
            import numpy as np
            lab = np.load(lp)
            h = RC.label_content_sha256({k: lab[k] for k in lab.files} if lp.suffix == ".npz" else lab)
        if h is None:
            hs = [js.get("rmsf_label_sha256") for _, js in records if js.get("rmsf_label_sha256")]
            if not hs: return None
            h = hs[0]
        _RM_TABLE_LABEL[dataset] = h
    return _RM_TABLE_LABEL[dataset]
def _table_hash(dataset, records):
    if dataset not in _RM_TABLE_HASH:
        cur = RC.current_residue_set_sha256(dataset)
        if cur is None:
            hs = [js.get("residue_set_sha256") for _, js in records if js.get("residue_set_sha256")]
            if not hs: return None
            cur = hs[0]
        _RM_TABLE_HASH[dataset] = cur
    return _RM_TABLE_HASH[dataset]
def _rm_agg(dataset, stem, col, files, key):
    """Aggregate `key` over the seed files of one cell after the provenance checks."""
    records=[]
    for f in files:
        try: records.append((f.name, _load_result(f)))
        except Exception: records.append((f.name, {}))
    if not records: return None
    if RMSF_PROTOCOL == "test":
        is_model = not stem.startswith("baseline_")
        ck = None
        if is_model:
            v = (records[0][1].get("checkpoint") or {}).get("variant")
            if v and (ROOT/"output"/v/"best.pt").exists():
                from ensembits.featurize import checkpoint_provenance
                ck = checkpoint_provenance(ROOT/"output"/v)
        probs = RC.check_rmsf_group(records, dataset=dataset, expect_label=RMSF_LABEL,
                                    residue_set_sha256=_table_hash(dataset, records),
                                    checkpoint=ck, is_model=is_model,
                                    label_sha256=_table_label(dataset, records))
        # FIXED(preretrain_audit_r2 H2, 2026-10-16): new: the input caches recorded in each
        #   result must still match the files on disk (rmsf_common.records_changed).
        for name, js in records:
            probs += [f"{name}: {w}" for w in RC.records_changed(js.get("feature_caches"))]
        probs += _table_same(dataset, records)
        if probs:
            if not any(r[:3]==(dataset,stem,col) for r in _RM_REFUSED):
                _RM_REFUSED.append((dataset, stem, col, probs))
            return Refused()
    vals=[js.get(key) for _, js in records]
    if RMSF_PROTOCOL == "val" and dataset == "misato" and key == _RM_KEY:
        vals=[v if v is not None else js.get("spearman") for v, (_, js) in zip(vals, records)]
    if RMSF_PROTOCOL == "test":
        # constant-prediction rule (2026-10-18): all seeds constant -> "—†"; other NaN refuses
        if all(RC.constant_prediction_seed(js, key) for _, js in records):
            if (dataset, stem, col) not in _RM_CONSTANT: _RM_CONSTANT.append((dataset, stem, col))
            return Constant(len(records))
        nan_seeds=[name for (name, _), v in zip(records, vals) if v is None or v != v]
        if nan_seeds:
            if not any(r[:3]==(dataset,stem,col) for r in _RM_REFUSED):
                _RM_REFUSED.append((dataset, stem, col,
                                    [f"{n}: {key} is NaN (not a one-token constant prediction)"
                                     for n in nan_seeds]))
            return Refused(len(records) - len(nan_seeds))
    return agg([v for v in vals if v is not None])
# mdCATH-div RMSF over seeds: P=1 reads "<stem>_P1inf", P=full reads "<stem>"
# (or "<stem>_P10inf" if the plain file is absent).
def _div_files(stem,p1):
    sufs=["_P1inf"] if p1 else ["","_P10inf"]; o=[]
    for s in range(10):
        seed="" if s==0 else f"_seed{s}"
        for suf in sufs:
            f=RMDIV/f"{stem}{suf}{seed}{_RM_SUFFIX}.json"
            if f.exists(): o.append(f); break
    return o
def rmsf_div(stem,p1,key=None):
    return _rm_agg("mdcath_div", stem, "P=1" if p1 else "P=full", _div_files(stem,p1), key or _RM_KEY)
# BioEmu-CATH RMSF over seeds: same file names as mdCATH-div, in output/probe_rmsf_bioemu.
def _bio_files(stem,p1):
    suf="_P1inf" if p1 else ""
    return [f for f in (RMBIO/f"{stem}{suf}{'' if s==0 else f'_seed{s}'}{_RM_SUFFIX}.json"
                        for s in range(10)) if f.exists()]
def rmsf_bio(stem,p1,key=None):
    return _rm_agg("bioemu", stem, "P=1" if p1 else "P=full", _bio_files(stem,p1), key or _RM_KEY)
# MISATO RMSF over seeds: no suffix = sequence split, "_structure" = structure split.
def _mis_files(stem,split,p1):
    pif="_P1inf" if p1 else ""; spf="" if split=="seq" else "_structure"
    return [f for f in (RMMIS/f"{stem}{pif}{spf}{'' if s==0 else f'_seed{s}'}{_RM_SUFFIX}.json"
                        for s in range(10)) if f.exists()]
def rmsf_mis(stem,split,p1,key=None):
    return _rm_agg("misato", stem, f"{split} {'P=1' if p1 else 'P=full'}",
                   _mis_files(stem,split,p1), key or _RM_KEY)
def rmsf_row(stem,p1,key=None):
    """One Ensembits row (P=1 or P=full) over the RM_COLS datasets."""
    return [rmsf_div(stem,p1,key),rmsf_mis(stem,"seq",p1,key),rmsf_mis(stem,"struct",p1,key),
            rmsf_bio(stem,p1,key)]
def rmsf_base_row(b,key=None):
    """One baseline row (baselines have no P; their result files carry no P suffix)."""
    mb=MIS_STEM.get(b,b)
    return [rmsf_div(f"baseline_{b}",False,key),rmsf_mis(f"baseline_{mb}","seq",False,key),
            rmsf_mis(f"baseline_{mb}","struct",False,key),rmsf_bio(f"baseline_{b}",False,key)]
def rmsf_blocks(key, reported_stems, ref_models=(), incomplete=None, label=None):
    """Blocks of the RMSF table (F3): single-frame, multi-frame, not ranked (older rows)."""
    single=[(b, rmsf_base_row(b,key)) for b in RM_SINGLE_BASE]
    multi=[(rm_base_disp(b), rmsf_base_row(b,key)) for b in RM_MULTI_BASE]
    for disp,stem in reported_stems:
        single.append((f"{disp} · P=1", rmsf_row(stem,True,key)))
        multi.append((f"{disp} · P=full", rmsf_row(stem,False,key)))
    if incomplete is not None:
        for d,r in single+multi:
            incomplete += check_seeds(label or "RMSF", d, RM_COLS, r)
    other=[]
    for disp,stem in ref_models:
        for p1,pt in [(True,"P=1"),(False,"P=full")]:
            r=rmsf_row(stem,p1,key)
            if any(v is not None for v in r): other.append((f"{disp} · {pt}", r))
    # H6: paper rows are kept even when empty (refused) unless --allow-incomplete
    keep=lambda rows: ([(d,r) for d,r in rows if any(v is not None for v in r)]
                       if ALLOW_INCOMPLETE else rows)
    return [("single-frame", keep(single)), ("multi-frame", keep(multi)), (NOT_RANKED, other)]
# ---- ProteinGym ----
# (display label, per_assay_pg_<tag>, "alone" key, "+ESM2" key). For Ensembits
# the keys are score (a) = level-1 codeword distance and its z-blend at alpha = 0.3.
# Paper: 2_results.tex sec:downstreams, Mutation Effect Prediction ("score each
# variant from the L_1 codeword only", alpha = 0.3), Table tab:proteingym_ensemble.
# FIXED(code_fix P16, 2026-10-16): the Ensembits rows below are pre-fix (PREFIX); the
#   reported row comes from --pgtag.
# FIXED(code_fix E1, 2026-10-20): VQ-2048 re-run done.
# FIXED(code_fix E2, 2026-10-20): SFTD λ sweep re-runs done.
# FIXED(require_fix #17, tex pass 2026-10-26): PG table updated with 256-D numbers.
PG_A, PG_MIX = "spearman_score_a_allL0", "spearman_zmix_a_a03"
PG_ROWS=[(d+PREFIX, t, PG_A, PG_MIX) for d, t in [
         ("256+φ/ψ FAPE-H (flagship)","dih256_fapeH"),
         ("ENSEMBITS 192-D (paper)","ship"),
         ("↳ BioEmu-CATH (dataset abl.)","dih256_fapeH_bioemu"),
         ("↳ BioEmu+MISATO (dataset abl.)","dih256_fapeH_bioemu_misato"),
         ("↳ noSFTD (distill off)","dih256_fapeH_noSFTD"),
         ("↳ all-off (no varP/distill)","dih256_fapeH_noVarP_noSFTD"),
         ("↳ VQ-2048 (single codebook)","dih256_fapeH_vq2048"),
         ("quant A: grad codebook (ctrl)","dih256_quantA"),
         ("quant B: + rotation trick","dih256_quantB"),
         ("quant C: + NSVQ","dih256_quantC"),
         ("quant D: + rotation + NSVQ","dih256_quantD"),
         ("quant E: + AminoAseed proj","dih256_quantE"),
         ("quant F: + AA proj L0 only","dih256_quantF"),
         ("warm start: AE pretrain -> RVQ","dih256_warmstart"),
         ("1-frame train (no dynamics)","dih256_P1train"),
         ("1-frame single-branch trainer","dih256_SF")]]
# FIXED(benchmark_audit D9, 2026-10-19): baselines rebuilt with correct atoms and CV α.
#   The camera-ready table comes from pg_aggregate.py (D.16 text pass item).
PG_BASE_ROWS=[("esm3struct","esm3struct","spearman_score_a_esm3struct_allpos","spearman_zmix_esm3struct_a03"),
              ("aminoaseed","aminoaseed","spearman_score_a_aminoaseed_allpos","spearman_zmix_aminoaseed_a03"),
              ("protoken","protoken","spearman_score_a_protoken_allpos","spearman_zmix_protoken_a03"),
              ("3di (mini3di)","3di","spearman_score_a_3di_allpos","spearman_zmix_3di_a03")]
PG_N_ASSAYS = 96
# Per-assay results of one PG tag: per-assay folder first, then shard files,
# then the aggregate JSON (earlier sources win for an assay).
# FIXED(benchmark_audit D9, 2026-10-19): pg_load records the provenance of what it read in
#   PG_PROVENANCE[tag] (old: none): the sidecar's assay list sha256, structure set sha256
#   (ESMFold structure contents), CSV set sha256 and pg_version (None for runs before D9),
#   the sha256 of the assay ids actually loaded, and the number of per-assay records
#   without an `inputs` block (pre-D9 schema: float32 ties, NaN blend handling). The
#   returned dict is unchanged. The camera-ready ProteinGym table comes from
#   pg_aggregate.py (alpha by leave-one-UniProt-out CV, -n_mut control).
PG_PROVENANCE = {}
def pg_load(tag):
    d={}
    for f in glob.glob(str(PG/f"per_assay_pg_{tag}/*.json")): d[Path(f).stem]=json.load(open(f))
    for s in glob.glob(str(PG/f"per_assay_pg_{tag}_shard*of*.json")):
        for a,r in json.load(open(s)).items(): d.setdefault(a,r)
    ag=PG/f"per_assay_pg_{tag}.json"
    if ag.exists():
        for a,r in json.load(open(ag)).items(): d.setdefault(a,r)
    side=PG/f"per_assay_pg_{tag}.assays_scored.json"
    sd=json.load(open(side)) if side.exists() else {}
    import hashlib
    PG_PROVENANCE[tag]={k:sd.get(k) for k in ("assay_list_sha256","assays_scored_sha256",
                        "structure_set_sha256","csv_set_sha256","pg_version","cache_dir")}
    PG_PROVENANCE[tag]["loaded_assays_sha256"]=hashlib.sha256("\n".join(sorted(d)).encode()).hexdigest()
    PG_PROVENANCE[tag]["n_loaded"]=len(d)
    PG_PROVENANCE[tag]["n_without_inputs_block"]=sum(1 for r in d.values() if "inputs" not in r)
    return d
# Mean of one key over assays (finite numbers only) and the number of assays.
def pg_mean(d,k):
    v=[r[k] for r in d.values() if isinstance(r.get(k),(int,float)) and r[k]==r[k]]
    return (st.mean(v),len(v)) if v else (None,0)
# Baselines + the paper row from the validated preprint reproduction at alpha=0.3.
# 'ship' (ENSEMBITS 192-D paper) == repro 'combined_esm3' (our shipped variant).
# These rows are read from REPO_ROOT/ensembits-repro/runs/proteingym/<name>.json
# (n_assays, mean_spearman_ours, mean_zmix_a03, mean_spearman_esm2), not from
# output/proteingym.
REPRO_PG={'esm3struct':'esm3struct','aminoaseed':'aminoaseed','protoken':'protoken',
          '3di':'3di_tokens','ship':'combined_esm3'}
# FIXED(preretrain_audit_r2 G10, 2026-10-17): old = a missing ensembits-repro ProteinGym
#   JSON crashed the whole compile (an RMSF-only compile could not run). New = the row is
#   shown as missing ("—") and listed in PG_MISSING; main() refuses only with
#   --require-proteingym.
PG_MISSING = []
def _repro_pg(name):
    f=ROOT/'ensembits-repro/runs/proteingym'/f'{name}.json'
    if not f.exists():
        PG_MISSING.append(str(f.relative_to(ROOT))); return None
    j=json.load(open(f))
    return j['n_assays'], j['mean_spearman_ours'], j['mean_zmix_a03'], j['mean_spearman_esm2']
def pg_row(disp,tag,ak,bk):
    """(disp, n, alone, mix, delta vs ESM2) of one PG row."""
    if tag in REPRO_PG:
        r=_repro_pg(REPRO_PG[tag])
        if r is None: return (disp,"—",None,None,None)
        n,a,mx,e=r; return (disp,n,a,mx,mx-e)
    # Re-run Ensembits dihedral variants: current pipeline, single-level (score_a) / zmix_a, alpha=0.3.
    d=pg_load(tag)
    if not d: return (disp,"—",None,None,None)
    a,n=pg_mean(d,ak); mx,_=pg_mean(d,bk); e,_=pg_mean(d,"spearman_esm2")
    dv=(mx-e) if (mx is not None and e is not None) else None
    return (disp,n,a,mx,dv)
def _rank1(vals):
    vv=sorted(((v,i) for i,v in enumerate(vals) if v is not None),reverse=True)
    return (vv[0][1] if vv else None, vv[1][1] if len(vv)>1 else None)
def _pgf(x,i,mk,plus=False):
    if x is None: return "—"
    s=f"{x:+.3f}" if plus else f"{x:.3f}"
    if i==mk[0]: return f"**{s}**"
    if i==mk[1]: return f"<u>{s}</u>"
    return s

# ---- Ablation tables (final_fix_list A9; E1 VQ-2048, E2 SFTD / λ sweep) ----------------
# FIXED(final_fix_list A9, authors' decision (a), 2026-10-19): old = one table (code_fix
#   P16) with RMSF and structure-split FUNCTION columns for every ablation row (EC d1, GO,
#   binding, affinity), which only the two λ candidates have, so the table could not be
#   built (every other row refused); the function columns are dropped. New = two tables,
#   compiled in the test and the --split-metric val mode:
#   * VQ table (E1): e0 (RVQ 2048-128-128, λ = 0.1) vs VQ-2048 (same λ), one row per run and
#     frame regime (P = full, P = 1): L1 distinct ids / utilisation and perplexity, recon loss
#     (P = 10 / 1; footnote: the RVQ decodes from the sum of its three levels, VQ-2048 from
#     one), RMSF Spearman (mdCATH-div, BioEmu-CATH, MISATO-seq, MISATO-struct; + mdCATH-div
#     per protein), token ANOVA held-out R² of s₁ and of the all-frames RMSF (P = 10 / 1);
#   * SFTD table (E2): SFTD off, λ = 0, 0.05, 0.1 (e0), 0.25, 0.5, 1: (a) best / stop epoch,
#     recon loss at P = 1, 2, 5, 10, L1 distinct ids / perplexity at P = 1; (b) the RMSF and
#     ANOVA columns of the VQ table in a P = full and a P = 1 block. A one-token row (SFTD off
#     at P = 1) shows "—†" in the RMSF cells (constant-prediction rule) and the ANOVA cells.
#   Inputs: output/ablations/recon_by_p.json (downstream/analyses/recon_by_p.py),
#   output/ablations/token_usage_by_p.json (token_usage_by_p.py), the token ANOVA JSONs
#   (ANOVA_FILES; downstream/analyses/token_anova.py). Every recon / usage cell must carry
#   the sha256 of output/<run>/best.pt and every ANOVA row its 12-digit checkpoint hash,
#   else the cell is refused (✗, listed, main() returns 1); a missing number is "—" and
#   counts as incomplete. Recon and token usage are validation-split numbers of the training
#   corpus in both modes; the ANOVA uses all 2,442 mdCATH-div domains (no split; not a
#   selection signal, final_fix_list A1).
# Paper: appendix.tex \label{app:ablation-vq}, \label{app:ablation-sftd}.
ABL_DIR = ROOT/"output"/"ablations"
# FIXED(recon2 A.7, 2026-10-25): defaults switched from pre-retrain to _rv runs
#   (decisions_20261023 R3: λ = 10, ensembits_bm_lam10_rv).
ABL_RECON = ABL_DIR/"recon_by_p_rv.json"
ABL_USAGE = ABL_DIR/"token_usage_by_p_rv.json"
ANOVA_DIR = ROOT/"output"/"token_anova"
ANOVA_FILES = [ANOVA_DIR/"token_anova_rv.json"]
DEFAULT_ABL_VQ = [("ensembits_bm_e0_rv", "RVQ 2048-128-128 (e0; λ = 0.1)"),
                  ("ensembits_bm_vq2048_rv", "VQ-2048 (λ = 0.1)")]
DEFAULT_ABL_SFTD = [("ensembits_bm_nosftd_rv", "SFTD off"), ("ensembits_bm_lam0_rv", "λ = 0"),
                    ("ensembits_bm_lam0p05_rv", "λ = 0.05"), ("ensembits_bm_e0_rv", "λ = 0.1 (e0)"),
                    ("ensembits_bm_lam0p25_rv", "λ = 0.25"), ("ensembits_bm_lam0p5_rv", "λ = 0.5"),
                    ("ensembits_bm_lam1_rv", "λ = 1")]
ABL_RM_COLS = ["mdCATH-div", "BioEmu-CATH", "MISATO-seq", "MISATO-struct", "mdCATH-div per-protein"]
ABL_ANOVA_COLS = ["ANOVA R² s₁", "ANOVA R² RMSF"]
_ABL_REFUSED = []       # (run, what, reason)
_ABL_MISSING = []       # (run, what)
_ABL_CACHE = {}
def _abl_json(path):
    if path not in _ABL_CACHE:
        _ABL_CACHE[path] = json.load(open(path)) if Path(path).exists() else None
    return _ABL_CACHE[path]
def _best_sha(variant):
    f = ROOT/"output"/variant/"best.pt"
    if not f.exists(): return None
    from ensembits.featurize import file_sha256
    return file_sha256(f)
def _run_key(variant): return variant[len("ensembits_bm_"):] if variant.startswith("ensembits_bm_") else variant
def _resolve_result_key(results_dict, variant):
    """Find the key for `variant` in a results dict, handling both short keys (e.g. 'e0')
    and absolute-path keys (e.g. '/…/ensembits_bm_e0_rv') from _rv files."""
    short = _run_key(variant)
    if short in results_dict:
        return short
    # _rv files use absolute paths; try matching by directory name
    for k in results_dict:
        if Path(k).name == variant:
            return k
    return None
def _abl_entry(path, variant, what):
    """Row of a recon / usage JSON for `variant`, after the checkpoint check (None if absent
    or refused; reasons recorded)."""
    d = _abl_json(path)
    results = d.get("results") or {} if d else {}
    key = _resolve_result_key(results, variant)
    if d is None or key is None:
        _ABL_MISSING.append((variant, what)); return None
    # provenance check: try both short and resolved key in the provenance block
    prov_runs = (d.get("provenance") or {}).get("runs") or {}
    prov_rec = prov_runs.get(key, prov_runs.get(_run_key(variant), {}))
    rec = prov_rec.get("best_pt_sha256") if isinstance(prov_rec, dict) else None
    cur = _best_sha(variant)
    if rec is None or cur is None or rec != cur:
        _ABL_REFUSED.append((variant, what, f"{Path(path).name}: best.pt sha256 "
                             f"{str(rec)[:12]} != output/{variant}/best.pt {str(cur)[:12]}"))
        return "refused"
    # A.2 fix: refuse rows where |repro_diff| > 1e-6
    row = results[key]
    if isinstance(row, dict) and abs(row.get("repro_diff", 0.0)) > 1e-6:
        _ABL_REFUSED.append((variant, what, f"{Path(path).name}: |repro_diff| = "
                             f"{abs(row.get('repro_diff', 0.0)):.2e} > 1e-6"))
        return "refused"
    return row
def _anova_row(variant, p):
    """{'s1': (r2, ci), 'rmsf': (r2, ci), 'M': n distinct tokens} of an Ensembits row of the
    token ANOVA at P = p frames (first file of ANOVA_FILES holding it), or None / 'refused'."""
    key = f"ens:{variant}:P{p}"
    for f in ANOVA_FILES:
        d = _abl_json(f)
        if not d: continue
        meth = {m.get("key"): m for m in d.get("methods", [])}
        if key not in meth: continue
        cur = _best_sha(variant)
        rec = (meth[key].get("prov") or {}).get("ckpt_sha256_12")
        if cur is None or rec != cur[:12]:
            _ABL_REFUSED.append((variant, f"ANOVA P={p}", f"{f.name}: checkpoint {rec} != "
                                 f"output/{variant}/best.pt {str(cur)[:12]}"))
            return "refused"
        T = (d.get("analysis") or {}).get("targets") or {}
        out = {}
        for nm, tk in (("s1", "s1"), ("rmsf", "rmsf_allframes")):
            h = ((T.get(tk) or {}).get("heldout_r2") or {}).get(key)
            out[nm] = None if not h else (h.get("r2"), h.get("ci95"))
        an = ((T.get("s1") or {}).get("anova") or {}).get(key) or {}
        out["M"] = an.get("M")
        out["file"] = f.name
        return out
    _ABL_MISSING.append((variant, f"ANOVA P={p}"))
    return None
def _fmt(x, nd=3):
    if x is None: return "—"
    if x == "refused": return REFUSED_MARK
    return f"{x:.{nd}f}"
def _anova_cell(a, nm):
    if a is None: return "—"
    if a == "refused": return REFUSED_MARK
    if a.get("M") is not None and int(a["M"]) <= 1: return RC.CONSTANT_MARK
    v = a.get(nm)
    if not v or v[0] is None or v[0] != v[0]: return "—"
    ci = v[1]
    return f"{v[0]:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
def _usage_cell(u, p, what="ids"):
    if u is None: return "—"
    if u == "refused": return REFUSED_MARK
    x = u.get(f"P{p}")
    if not x: return "—"
    return f"{x['distinct']:,} ({x['util']:.0%})" if what == "ids" else f"{x['ppl']:.1f}"
def _rmsf_cells(variant, p1, incomplete, tname, disp):
    stem = tag_of(variant)
    vals = enforce_seeds([rmsf_div(stem, p1), rmsf_bio(stem, p1), rmsf_mis(stem, "seq", p1),
                          rmsf_mis(stem, "struct", p1), rmsf_div(stem, p1, PP_KEY)])
    incomplete += check_seeds(tname, disp, ABL_RM_COLS, vals)
    return [cell(v) for v in vals]
def _abl_regime_rows(runs, incomplete, tname, recon=True, usage=True):
    """Rows (P = full block, P = 1 block) of the VQ table / the SFTD table (b)."""
    L = []
    for p1, pt, pn in ((False, "P = full", 10), (True, "P = 1", 1)):
        L.append(f"| *— {pt} —* |" + " |" * (len(ABL_RM_COLS) + len(ABL_ANOVA_COLS) + 2 * usage + recon))
        for V, lab in runs:
            disp = f"{lab} · {pt}"
            r = _abl_entry(ABL_RECON, V, "recon") if recon else None
            u = _abl_entry(ABL_USAGE, V, "token usage") if usage else None
            a = _anova_row(V, pn)
            cells = []
            if usage:
                cells += [_usage_cell(u, pn, "ids"), _usage_cell(u, pn, "ppl")]
            if recon:
                cells.append(_fmt(r if r in (None, "refused") else r.get(f"val_P{pn}")))
            cells += _rmsf_cells(V, p1, incomplete, tname, disp)
            cells += [_anova_cell(a, "s1"), _anova_cell(a, "rmsf")]
            L.append(f"| {disp} | " + " | ".join(cells) + " |")
    return L
def ablation_tables(vq_runs, sftd_runs, incomplete):
    """Markdown of the VQ and SFTD ablation tables (final_fix_list A9; see the block comment)."""
    _ABL_REFUSED.clear(); _ABL_MISSING.clear(); _ABL_CACHE.clear()
    val = SPLIT_METRIC == "val"
    rm_what = ("validation Spearman at the selected epoch" if val else
               f"{RMSF_PROTOCOL}-protocol Spearman")
    md = [f"\n## Ablations: VQ (E1)\n",
          f"RVQ (three levels, 2048-128-128) vs a single 2048-entry codebook, same corpus, λ = 0.1 "
          f"and schedule. P = full: 10 input frames (MISATO 8); P = 1: the first frame. L1 ids = "
          f"distinct first-level ids emitted on the validation split of the training corpus "
          f"(utilisation = ids / 2048); ppl = perplexity of the L1 id frequencies. Recon = "
          f"validation reconstruction loss (FAPE + dihedral, lower is better) at P = 10 / 1"
          f"{chr(185)}. RMSF = {rm_what}, mean ± sample std over 10 probe seeds. ANOVA = token "
          f"ANOVA held-out within-protein R² [95% CI, CATH-topology bootstrap] of s₁ and of the "
          f"all-frames RMSF on all 2,442 mdCATH-div domains (no split; not a selection signal) at "
          f"P = 10 / 1.\n",
          "| model | L1 ids (util.) | L1 ppl | recon | " + " | ".join(ABL_RM_COLS) + " | "
          + " | ".join(ABL_ANOVA_COLS) + " |",
          "|---|" + "---|" * (3 + len(ABL_RM_COLS) + len(ABL_ANOVA_COLS))]
    md += _abl_regime_rows(vq_runs, incomplete, "ablation VQ")
    md.append(f"\n{chr(185)} The RVQ reconstructs from the sum of its three quantizer levels, "
              f"VQ-2048 from its single level, so the recon columns are not like-for-like "
              f"(the RVQ decodes from 3 levels).\n")
    md += [f"\n## Ablations: SFTD (E2)\n",
           "SFTD off = trained at P = 10 only (no variable-P exposure, no distillation); λ = the "
           "weight of the distillation (consistency) loss with variable-P training (λ = 0: "
           "variable P without distillation). Best / stop epoch from history.json / config.json. "
           "Recon = validation reconstruction loss at P input (and target) frames. P = 1 L1 ids = "
           "distinct first-level ids emitted at P = 1 on the validation split (ppl = perplexity). "
           "One training seed per configuration.\n",
           "| model | best / stop epoch | recon P = 1 | recon P = 2 | recon P = 5 | recon P = 10 | "
           "P = 1 L1 ids (util.) | P = 1 L1 ppl |",
           "|---|---|---|---|---|---|---|---|"]
    for V, lab in sftd_runs:
        r = _abl_entry(ABL_RECON, V, "recon")
        u = _abl_entry(ABL_USAGE, V, "token usage")
        if r in (None, "refused"):
            ep, rc = _fmt(r), [_fmt(r)] * 4
        else:
            ep = f"{r.get('best_epoch')} / {r.get('total_epochs')}"
            rc = [_fmt(r.get(f"val_P{p}")) for p in (1, 2, 5, 10)]
        md.append(f"| {lab} | {ep} | " + " | ".join(rc) + f" | {_usage_cell(u, 1, 'ids')} | "
                  f"{_usage_cell(u, 1, 'ppl')} |")
    md += ["", "RMSF and token ANOVA of the same runs (columns as in the VQ table):", "",
           "| model | " + " | ".join(ABL_RM_COLS) + " | " + " | ".join(ABL_ANOVA_COLS) + " |",
           "|---|" + "---|" * (len(ABL_RM_COLS) + len(ABL_ANOVA_COLS))]
    md += _abl_regime_rows(sftd_runs, incomplete, "ablation SFTD", recon=False, usage=False)
    # B.6 fix: only print the "one token" footnote when at least one row uses the dagger
    if any(RC.CONSTANT_MARK in line for line in md):
        md.append(f"\n{RC.CONSTANT_NOTE} (`{RC.CONSTANT_MARK}`; ANOVA: one token, no within-token "
                  f"model).\n")
    if _ABL_REFUSED:
        md.append("\n## Refused ablation cells\n")
        md.append(f"Shown as `{REFUSED_MARK}`: the recon / token-usage / ANOVA result was not "
                  f"computed from the current output/<run>/best.pt; rerun it.\n")
        for v, w, why in _ABL_REFUSED: md.append(f"- {v} · {w}: {why}")
    for v, w in sorted(set(_ABL_MISSING)):
        incomplete.append(("ablations", v, w, 0))
    return md
def ablation_pg_section(reported, ablations, ab_pgtags):
    """ProteinGym rows of the ablation variants (test compile only; old pipeline, α = 0.3)."""
    pgr=[]
    if SPLIT_METRIC == "test":     # 2026-10-19: no ProteinGym in the validation compile
        if reported and reported[2]: pgr.append(pg_row(reported[0],reported[2],PG_A,PG_MIX))
        for V,lab in ablations:
            if V in ab_pgtags: pgr.append(pg_row(lab,ab_pgtags[V],PG_A,PG_MIX))
    md=[]
    if pgr:
        md.append(f"\n## Ablations: ProteinGym (P=1, score (a) / z-blend α=0.3 with ESM2-650M)\n")
        md.append("| model | n | alone | +ESM2 (α0.3) | Δ vs ESM2 |"); md.append("|---|---|---|---|---|")
        ta=_rank1([r[2] for r in pgr]); tm=_rank1([r[3] for r in pgr]); td=_rank1([r[4] for r in pgr])
        for i,(disp,n,a,mx,dv) in enumerate(pgr):
            md.append(f"| {disp} | {n} | {_pgf(a,i,ta)} | {_pgf(mx,i,tm)} | {_pgf(dv,i,td,plus=True)} |")
    return md

def parse_pairs(text, what):
    """"a=b,c=d" -> [(a, b), ...] (the value may contain '=' only if quoted away)."""
    out=[]
    for item in [x for x in (text or "").split(",") if x.strip()]:
        if "=" not in item: raise SystemExit(f"{what}: expected name=value, got {item!r}")
        k,v=item.split("=",1); out.append((k.strip(),v.strip()))
    return out

# ---- Build the markdown ------------------------------------------------------
def build_markdown(reported=None, ablations=(), ab_pgtags=None, abl_vq=(), abl_sftd=()):
    """The whole results markdown -> (lines, incomplete cells).

    reported: (label, variant, pgtag) of the reported model, or None (pre-fix rows only).
    ablations: [(variant, label)] (ProteinGym ablation rows); ab_pgtags: {variant: pg tag}.
    abl_vq / abl_sftd: [(variant, label)] of the VQ / SFTD ablation tables (A9).
    """
    ab_pgtags=ab_pgtags or {}
    incomplete=[]
    reset_rmsf_checks(); PG_MISSING.clear(); _GO_REFUSED.clear(); _SPLIT_REFUSED.clear()
    _EC_REFUSED.clear()   # D8
    _BIND_REFUSED.clear()   # D4
    _FN_REFUSED.clear(); _ITEM_REFUSED.clear(); _SIG.clear(); _REF_SIG.clear()   # 2026-10-19
    _CV_REFUSED.clear(); _CV_FOLDS.clear()   # function CV (2026-10-19)
    val = SPLIT_METRIC == "val"
    models=([(reported[0],reported[1])] if reported else [])+MODELS
    paper=({reported[0]} if reported else set())
    md=(["# VALIDATION metrics (--split-metric val): every column is compiled from the "
         "validation fields of the result JSONs (selected epoch); no test field was read; "
         "ProteinGym is omitted (no validation split)\n"] if val else [])
    md+=["# Stabilized downstream results (rebuttal)\n",
        "Downstream heads under the **stabilization recipe** (LR warmup 5 → cosine, min-epochs 20, "
        "patience 12; degenerate seeds re-run to completion for every method with the same rule, "
        f"which reads the stored {'validation' if val else 'test'} metric). mean ± sample std (ddof = 1) over 10 seeds, same "
        "conv1d/MLP head + splits as the paper. Affinity = ligand-aware MACCS-167. `(k)`=k<10 seeds; "
        "`—`=absent. Pooled baselines (Reviewer szcw #4) = per-residue embedding averaged over all 8 frames. "
        "Per column, **bold** = best method, <u>underline</u> = second best. EC d1-d3 and GO "
        "columns = mAP (macro-averaged average precision over the classes of the task); Fmax, "
        "top-1 and the other metrics (GO: frequency floor) in the \"Metrics:\" tables. Rows marked "
        f"`{PREFIX.strip()}` are checkpoints trained before the 2026-10-15 code fixes.\n"]
    if FN_CV:   # function CV (2026-10-19)
        from function_cv import AGGREGATION_RULE, cv_path, N_FOLDS
        md.append("EC and GO (sequence / structure): distinct proteins (one representative "
                  "complex per UniProt accession set), grouped label-stratified 5-fold "
                  "cross-validation (`data/misato_function_cv_v1.json`; groups = sequence "
                  "clusters or CATH H; rotation r: test fold r, val fold r+1, train the other "
                  f"three). Cell = {AGGREGATION_RULE}; per-fold values in the `.cv_folds.json` "
                  "sidecar. Binding site, affinity and the random split use the fixed v2 "
                  "splits (no EC / GO on the random split).\n")
        # Q6: report kept N per CV fold and EC depth / GO (proteins scored vs dropped)
        try:
            _cv = json.loads(cv_path().read_text())
            split_key = 'val' if SPLIT_METRIC == 'val' else 'test'
            for mode in ["sequence", "structure"]:
                md.append(f"\n### CV fold sizes ({mode}, {split_key} proteins)\n")
                hdr = "| fold | EC total | EC d1 classes | EC d2 classes | EC d3 classes | GO total |"
                md.append(hdr)
                md.append("|---|---|---|---|---|---|")
                for r in range(N_FOLDS):
                    ec_rot = _cv['tasks']['ec']['modes'][mode]['rotations'][r]
                    go_rot = _cv['tasks']['go']['modes'][mode]['rotations'][r]
                    ec_n = len(ec_rot[split_key])
                    go_n = len(go_rot[split_key])
                    ec_cls = {d: len(ec_rot['classes'][d]) for d in ec_rot['classes']}
                    md.append(f"| {r} | {ec_n} | {ec_cls.get('1',0)} | {ec_cls.get('2',0)} | {ec_cls.get('3',0)} | {go_n} |")
        except Exception:
            pass
    if reported:
        md.append(f"Reported model: **{reported[0]}** = `output/{reported[1]}`"
                  + (f", ProteinGym tag `{reported[2]}`" if reported[2] else "") + ".\n")
    for sp in ["sequence","structure","random"]:
        md.append(f"\n## {sp.capitalize()} split\n"); md.append(split_table(sp, models, paper, incomplete))
    # function_reaudit pre-run fix 5 (2026-10-19): all metrics per task and split
    for sp in ["sequence","structure","random"]:
        for name, specs in DETAIL_TABLES:
            if FN_CV and not _cv_on(sp) and all(c[1] in ("EC","GO") for c in specs):
                continue                                # function CV: no EC / GO on random
            md.append(f"\n## Metrics: {name}, {sp} split\n")
            if name.startswith("EC"):
                md.append(f"mAP = primary EC metric (the EC columns of the main tables), Fmax = "
                          f"secondary. `{EC_MAJORITY_COL}` = top-1 hit of predicting the most "
                          f"frequent training class for every complex (labels only); equal in "
                          f"every row when the items are equal; not ranked.\n")
            if name == "GO":
                md.append("mAP = primary GO metric (the GO column of the main tables), Fmax and "
                          "top-1 secondary. ")
                md.append("`floor` columns = frequency floor (training frequency of each class as "
                          "the score, labels only); equal in every row when the items are equal; "
                          "not ranked.\n")
            md.append(split_table(sp, models, paper, None, specs=specs, check=False))
    # function_reaudit pre-run fix 4 (2026-10-19): D5 affinity controls (not paper rows)
    for sp in ["sequence","structure","random"]:
        md.append(f"\n## Affinity controls (D5, not paper rows), {sp} split\n")
        md.append("Same complexes (finite label, MACCS fingerprint, >= 10 residues), splits, seeds, "
                  "epochs and early stopping as the affinity rows above; ligand only = MLP on "
                  "MACCS; protein only = the same head without MACCS. Not ranked.\n")
        md.append(control_table(sp, reported))
    md.append("\n## RMSF (Spearman, mean ± sample std)\n")
    # FIXED(preretrain_audit_r2 F11 / F3, 2026-10-16): text updated (all-frames labels, BioEmu,
    #   common residue set, the two blocks of tab:rmsf).
    md.append(f"Flexibility probe, Spearman pooled over the {'validation' if val else 'test'} residues of the common residue set "
              f"(label `{RMSF_LABEL}`: per-residue Cα RMSF over all MD frames, converged "
              "superposition). mdCATH-div and BioEmu-CATH = CATH-H splits, P=10 frames; MISATO "
              "sequence / structure splits, P=8. Rows in the two blocks of tab:rmsf: single-frame "
              "(baselines + Ensembits P=1) and multi-frame (ProtProfileMD_K, K10 on mdCATH-div / "
              "BioEmu-CATH and K8 on MISATO, Vote_3Di + Ensembits P=full); bold / underline within "
              "each block. Older checkpoints follow, not ranked.\n")
    if val:
        md.append(f"*Validation compile: pooled = `{_RM_KEY}` (validation Spearman at the selected "
                  f"epoch); per protein = `{PP_KEY}`.*\n")
    rm_rep=[(reported[0],tag_of(reported[1]))] if reported else []
    md.append(block_table(RM_COLS, rmsf_blocks(_RM_KEY, rm_rep, RM_MODELS, incomplete, "RMSF")))
    # FIXED(preretrain_audit_r2 Q2 / Q17, 2026-10-16): new: the per-protein Spearman (mean
    #   over test proteins, then mean ± sample std over seeds) next to the pooled table, same
    #   rows and blocks; and the provenance line + the refused cells.
    if RMSF_PROTOCOL == "test":
        md.append("\n## Per-protein RMSF (Spearman within each protein, mean over "
                  f"{'validation' if val else 'test'} proteins)\n")
        md.append(f"Same probes, rows and blocks as the pooled RMSF table; per protein = Spearman "
                  f"between prediction and label within each test protein with ≥ "
                  f"{RC.PER_PROTEIN_MIN_RESIDUES} residues and non-constant labels, averaged over "
                  f"proteins (`{PP_KEY}`), then mean ± sample std over seeds.\n")
        md.append(block_table(RM_COLS, rmsf_blocks(PP_KEY, rm_rep, RM_MODELS)))
        md.append(f"\nRMSF provenance: label `{RMSF_LABEL}` (content "
                  + ", ".join(f"{d} `{(h or 'none')[:12]}`" for d,h in sorted(_RM_TABLE_LABEL.items()))
                  + "); residue sets "
                  + ", ".join(f"{d} `{(h or 'none')[:12]}`" for d,h in sorted(_RM_TABLE_HASH.items()))
                  + ".\n")
    # --split-metric val (2026-10-19): ProteinGym has no validation split; left out.
    # A.5 fix: the old-protocol PG table (fixed alpha=0.3, ensembits-repro baselines) is
    # removed in favour of pg_aggregate.py / pg_d9_<name>.md; keeping only a pointer.
    if not val:
        md.append("\n## ProteinGym (96 DMS, L≤200)\n")
        md.append("*Removed: ProteinGym table uses `pg_aggregate.py` with LOUO CV alpha "
                  "(see `output/proteingym/pg_d9_<name>.md`). The old alpha=0.3 table with "
                  "`ensembits-repro` baselines is no longer compiled here.*\n")
    # final_fix_list A9 (2026-10-19): the VQ and SFTD ablation tables (no function columns)
    if abl_vq or abl_sftd:
        md += ablation_tables(list(abl_vq), list(abl_sftd), incomplete)
    if ablations:
        md += ablation_pg_section(reported, list(ablations), ab_pgtags)
    md.append("\n## Notes\n")
    md.append("- **random split** has per-protein label leakage inflating AA/one-hot baselines; sequence & structure are the meaningful splits.")
    md.append("- Residual **std>0.1** cells are weak baselines (3di/aa/protoken/protprofile) on the **structure split**: verified they ran the full 40 epochs and still collapse on some seeds — genuine baseline instability on the hardest split, not fixable early-stop (paper also flags structure-split function variance). Premature collapses are re-run to completion for every method alike (`downstream/seed_rescue.py`; re-run JSONs carry `\"rescued\": true`).")
    md.append("- RMSF pooled/per-token is the correct axis for 'tokens capture dynamics' (matches the ANOVA); ANM is a per-protein baseline answering a different question and isn't a tokenizer.")
    md.append("- **Codebook width differs across methods and the probes are equal-width but not equal-capacity** (worklog §23d): ours 2048×**128**, esm3struct 4096×**128**, aminoaseed/vanillavq 512×**1024**, protoken 512×**32**, 3di 20×**2**. Each is projected to hidden=128 by the probe's first learnable layer, so first-layer parameters scale with the input width (aminoaseed ≈655k vs ours ≈82k vs protoken ≈20k). Read cross-tokenizer *function* rows with that in mind — part of ProToken's structure-split collapse is feature width, and wide-codebook baselines get more projection capacity than our 128-D features.")
    # FIXED(code_fix P21, 2026-10-16): old: two rebuttal sections on same-fold /
    #   different-dynamics pairs read output/rebuttal_analyses/{fold_dynamics_*,
    #   bioemu_fold_dynamics_*, bioemu_token_comparison}.json (old flagship, old pipeline).
    #   New: removed; the pair-test table is built from output/fold_pairs/ by
    #   downstream/fold_pairs/build_pair_table.py.
    md.append("- Fold-matched pair test: see `output/fold_pairs/pair_table.md` "
              "(`downstream/fold_pairs/build_pair_table.py`).")
    if _RM_CONSTANT:
        md.append(f"\n{RC.CONSTANT_NOTE} (`{RC.CONSTANT_MARK}`: the probe predicted a constant "
                  f"for every seed; not ranked, not a missing seed): "
                  + ", ".join(f"{d} {st_} {c}" for d,st_,c in _RM_CONSTANT) + ".\n")
    if _RM_REFUSED:
        md.append("\n## Refused RMSF cells\n")
        md.append(f"Shown as `{REFUSED_MARK}`: the result JSONs of these cells do not carry the "
                  f"expected label, the table's residue set or one checkpoint (rerun them).\n")
        md.append("| dataset | stem | column | reasons |"); md.append("|---|---|---|---|")
        for d,st_,c,why in _RM_REFUSED:
            md.append(f"| {d} | {st_} | {c} | {'; '.join(why[:3])}{' …' if len(why)>3 else ''} |")
    if _GO_REFUSED:   # benchmark_audit D7 (2026-10-18)
        md.append("\n## Refused GO cells\n")
        md.append(f"Shown as `{REFUSED_MARK}`: these GO result JSONs were not built on the current "
                  f"GO labels (misato_go_mf_v2.json); rerun them.\n")
        for _,fn,why in sorted(set(_GO_REFUSED))[:50]: md.append(f"- `{fn}`: {why}")
    if _EC_REFUSED:   # benchmark_audit D8 (2026-10-18)
        md.append("\n## Refused EC cells\n")
        md.append(f"Shown as `{REFUSED_MARK}`: these EC result JSONs were not built on the current "
                  f"EC labels (misato_ec_v2.json); rerun them.\n")
        for _,fn,why in sorted(set(_EC_REFUSED))[:50]: md.append(f"- `{fn}`: {why}")
    if _SPLIT_REFUSED:   # benchmark_audit D3 (2026-10-18)
        md.append("\n## Refused cells (MISATO split file)\n")
        md.append(f"Shown as `{REFUSED_MARK}`: these EC / GO / binding / affinity result JSONs "
                  f"were not built on the current MISATO split file (data/misato_splits_v2.json); "
                  f"rerun them.\n")
        for _,t,fn,why in sorted(set(_SPLIT_REFUSED))[:50]: md.append(f"- {t} `{fn}`: {why}")
    if _BIND_REFUSED:   # benchmark_audit D4 (2026-10-18)
        md.append("\n## Refused binding-site cells\n")
        md.append(f"Shown as `{REFUSED_MARK}`: these binding-site result JSONs were not built on "
                  f"the current binding-site labels (misato_binding_site_v2.npz); rerun them.\n")
        for _,fn,why in sorted(set(_BIND_REFUSED))[:50]: md.append(f"- `{fn}`: {why}")
    if _FN_REFUSED:   # function_reaudit (2026-10-19)
        md.append("\n## Refused cells (function protocol)\n")
        md.append(f"Shown as `{REFUSED_MARK}`: these EC / GO / binding / affinity result JSONs lack the "
                  f"current protocol tag, verified feature inputs or the current affinity label / "
                  f"MACCS sha256; rerun them.\n")
        for r,t,fn,why in sorted(set(_FN_REFUSED))[:50]: md.append(f"- {t} `{fn}`: {why}")
    if _CV_REFUSED:   # function CV (2026-10-19)
        md.append("\n## Refused cells (function CV file)\n")
        md.append(f"Shown as `{REFUSED_MARK}`: these EC / GO cross-validation result JSONs were "
                  f"not built on the current CV file (data/misato_function_cv_v1.json) or "
                  f"task; rerun them.\n")
        for r,t,fn,why in sorted(set(_CV_REFUSED))[:50]: md.append(f"- {t} `{fn}`: {why}")
    if _ITEM_REFUSED:   # function_reaudit (2026-10-19)
        md.append("\n## Refused cells (items / classes)\n")
        md.append(f"Shown as `{REFUSED_MARK}`: every method of a cell must be scored on the same "
                  f"complexes with the same classes (item and class-list sha256 in the result "
                  f"JSONs).\n")
        for r,sp_,tc,why in sorted(set(_ITEM_REFUSED))[:80]: md.append(f"- {r} · {sp_} · {tc}: {why}")
    md.append("\n## Seed completeness\n")
    if not reported:
        md.append("- **No reported-model row** (run with `--variant <run> --pgtag <tag>`).")
    if incomplete:
        md.append(f"{len(incomplete)} paper-row cells have fewer than {N_SEEDS} seeds "
                  f"(ProteinGym: fewer than {PG_N_ASSAYS} assays):\n")
        md.append("| table | row | column | seeds |"); md.append("|---|---|---|---|")
        for t,r,c,n in incomplete: md.append(f"| {t} | {r} | {c} | {n} |")
    else:
        md.append(f"All paper-row cells have {N_SEEDS} seeds.")
    return md, incomplete

def main(argv=None):
    """CLI: compile the results markdown (see the module docstring)."""
    ap=argparse.ArgumentParser(description="Compile the downstream results markdown.")
    ap.add_argument("--rmsf-protocol",choices=("test","val"),default="test",
                    help="test (default): '__test' files, test_spearman -> "
                         "camera_ready/results_bf65_testprotocol.md; val: the old files -> "
                         "rebuttal/results_bf65.md")
    ap.add_argument("--variant",default=None,help="output/<variant> of the reported model (code_fix P16)")
    ap.add_argument("--pgtag",default=None,help="ProteinGym tag of the reported model")
    ap.add_argument("--label",default=REPORTED_LABEL,
                    help=f"row label of the reported model (default {REPORTED_LABEL!r}; "
                         "compile_camera_ready_tables.py selects rows by it)")
    ap.add_argument("--ablation-variants",default="",
                    help="E1 / E2 ablations as <variant>=<label>,... (e.g. "
                         "\"run_vq2048=VQ-2048,run_lam05=λ = 0.5\")")
    ap.add_argument("--ablation-pgtags",default="",help="<variant>=<pg tag>,... of the ablations")
    # final_fix_list A9 (2026-10-19): runs of the VQ / SFTD ablation tables
    ap.add_argument("--abl-vq",default=",".join(f"{v}={l}" for v,l in DEFAULT_ABL_VQ),
                    help="VQ table runs as <variant>=<label>,... (default: e0 and VQ-2048)")
    ap.add_argument("--abl-sftd",default=",".join(f"{v}={l}" for v,l in DEFAULT_ABL_SFTD),
                    help="SFTD table runs as <variant>=<label>,... (default: SFTD off, λ sweep)")
    ap.add_argument("--no-ablation-tables",action="store_true",
                    help="leave out the VQ / SFTD ablation tables")
    # 2026-10-21: override ablation input files (for retrained _rv runs)
    ap.add_argument("--abl-recon-file",default=None,
                    help="override recon_by_p JSON (e.g. output/ablations/recon_by_p_rv.json)")
    ap.add_argument("--abl-usage-file",default=None,
                    help="override token_usage_by_p JSON")
    ap.add_argument("--abl-anova-files",default=None,
                    help="comma-separated ANOVA JSONs (e.g. output/token_anova/token_anova_rv.json)")
    ap.add_argument("--out",default=None,help="output markdown (default by protocol)")
    ap.add_argument("--require-proteingym",action="store_true",
                    help="exit 1 when ProteinGym inputs are missing (G10; default: report only)")
    ap.add_argument("--allow-incomplete",action="store_true",
                    help="preview: show paper cells with fewer than 10 seeds as 'm±s (k)' "
                         "instead of refusing them (H6); the compile then still exits 0")
    ap.add_argument("--split-metric",choices=("test","val"),default="test",
                    help="val: compile every column from the validation fields only (no test "
                         "field is read; ProteinGym left out) -> "
                         "camera_ready/results_bf65_valmetrics.md (2026-10-19)")
    ap.add_argument("--function-protocol",choices=("cv","split"),default="cv",
                    help="EC / GO cells: cv (default, 2026-10-19) = the cross-validation runs "
                         "(mean over folds per seed, mean ± sample std over seeds); split = "
                         "the old fixed-split runs")
    ap.add_argument("--rmsf-label",default=RC.RMSF_LABEL_DEFAULT,
                    help="RMSF label every RMSF result must carry (test protocol; default "
                         "%(default)s); cells with another label are refused")
    a=ap.parse_args(argv)
    # 2026-10-21: override ablation file paths for retrained runs
    global ABL_RECON, ABL_USAGE, ANOVA_FILES
    if a.abl_recon_file:
        ABL_RECON = Path(a.abl_recon_file) if Path(a.abl_recon_file).is_absolute() else ROOT/a.abl_recon_file
    if a.abl_usage_file:
        ABL_USAGE = Path(a.abl_usage_file) if Path(a.abl_usage_file).is_absolute() else ROOT/a.abl_usage_file
    if a.abl_anova_files:
        ANOVA_FILES = [Path(f.strip()) if Path(f.strip()).is_absolute() else ROOT/f.strip()
                       for f in a.abl_anova_files.split(",") if f.strip()]
    global RMSF_LABEL, ALLOW_INCOMPLETE, FN_CV
    RMSF_LABEL=a.rmsf_label
    FN_CV = a.function_protocol == "cv"   # function CV (2026-10-19)
    ALLOW_INCOMPLETE=a.allow_incomplete
    set_rmsf_protocol(a.rmsf_protocol)
    if a.split_metric=="val" and a.rmsf_protocol!="test":
        ap.error("--split-metric val needs --rmsf-protocol test (the '__test' RMSF files)")
    set_split_metric(a.split_metric)
    out=Path(a.out) if a.out else (OUT_VALMETRIC if a.split_metric=="val" else
                                   OUT_TEST if a.rmsf_protocol=="test" else OUT_VAL)
    if a.pgtag and not a.variant: ap.error("--pgtag needs --variant")
    reported=(a.label,a.variant,a.pgtag) if a.variant else None
    if reported is None:
        print("WARNING: no --variant: compiling the pre-fix rows only (no reported-model row)",
              file=sys.stderr, flush=True)
    ablations=[(v,l) for v,l in parse_pairs(a.ablation_variants,"--ablation-variants")]
    ab_pgtags=dict(parse_pairs(a.ablation_pgtags,"--ablation-pgtags"))
    abl_vq=[] if a.no_ablation_tables else parse_pairs(a.abl_vq,"--abl-vq")
    abl_sftd=[] if a.no_ablation_tables else parse_pairs(a.abl_sftd,"--abl-sftd")
    md,incomplete=build_markdown(reported, ablations, ab_pgtags, abl_vq, abl_sftd)
    if incomplete:
        print(f"WARNING: {len(incomplete)} paper-row cells have fewer than {N_SEEDS} seeds:",
              file=sys.stderr)
        for t,r,c,n in incomplete: print(f"  {t:<16} {r:<48} {c:<24} {n}", file=sys.stderr)
    # ---- Write the output markdown ------------------------------------------------
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md)+"\n")
    print(f"wrote {out}")
    if FN_CV:   # function CV (2026-10-19): per-fold values of every CV cell
        from function_cv import AGGREGATION_RULE, cv_path
        from ensembits.featurize import file_sha256
        cvp=cv_path(); side=out.with_name(out.stem+".cv_folds.json")
        tmp=side.with_name("."+side.name+".tmp")
        tmp.write_text(json.dumps({"split_metric": SPLIT_METRIC, "aggregation": AGGREGATION_RULE,
                                   "cv_file": str(cvp.name),
                                   "cv_sha256": file_sha256(cvp) if cvp.exists() else None,
                                   "cells": _CV_FOLDS}, indent=1, sort_keys=True))
        import os; os.replace(tmp, side)
        print(f"wrote {side}")
    # Refused RMSF cells of rows that enter the paper (reported, baselines, ablations) make
    # the compile fail; refused pre-fix reference rows only show the mark.
    paper_stems=({tag_of(a.variant)} if a.variant else set())|{tag_of(v) for v,_ in ablations}
    paper_stems|={tag_of(v) for v,_ in abl_vq+abl_sftd}      # A9 ablation tables
    # 2026-10-19: function row ids carry "_P1inf" for P=1 and " (control)" for the D5 control
    #   rows (not paper rows: their refusals are listed but do not fail the compile)
    def is_paper(row):
        if row.endswith(" (control)"): return False
        stem=row[:-len("_P1inf")] if row.endswith("_P1inf") else row
        return stem in paper_stems or stem.startswith("baseline_")
    bad=[r for r in _RM_REFUSED if r[1] in paper_stems or r[1].startswith("baseline_")]
    rc=0
    if PG_MISSING:
        print(f"{'ERROR' if a.require_proteingym else 'NOTE'}: {len(PG_MISSING)} ProteinGym "
              f"inputs missing (rows shown as —): {sorted(set(PG_MISSING))[:5]}", file=sys.stderr)
        if a.require_proteingym: rc=1
    if incomplete and not ALLOW_INCOMPLETE:
        print(f"ERROR: {len(incomplete)} paper-row cells lack seeds (refused, ✗ (k)); rerun them "
              f"or compile a preview with --allow-incomplete", file=sys.stderr)
        rc=1
    if bad:
        print(f"ERROR: {len(bad)} RMSF cells of paper rows were refused (mixed label / residue "
              f"set / checkpoint / caches); see '## Refused RMSF cells' in {out.name}:", file=sys.stderr)
        for d,st_,c,why in bad[:20]: print(f"  {d:<10} {st_:<40} {c:<16} {why[0]}", file=sys.stderr)
        rc=1
    if _ABL_REFUSED:   # final_fix_list A9 (2026-10-19)
        print(f"ERROR: {len(_ABL_REFUSED)} ablation cells refused (recon / token usage / ANOVA "
              f"not from the current best.pt); see '## Refused ablation cells' in {out.name}",
              file=sys.stderr)
        rc=1
    bad_go={r for r in _GO_REFUSED if is_paper(r[0])}
    if bad_go:   # benchmark_audit D7 (2026-10-18)
        print(f"ERROR: {len(bad_go)} GO result JSONs of paper rows are not on the current GO "
              f"labels (refused); see '## Refused GO cells' in {out.name}", file=sys.stderr)
        rc=1
    bad_ec={r for r in _EC_REFUSED if is_paper(r[0])}
    if bad_ec:   # benchmark_audit D8 (2026-10-18)
        print(f"ERROR: {len(bad_ec)} EC result JSONs of paper rows are not on the current EC "
              f"labels (refused); see '## Refused EC cells' in {out.name}", file=sys.stderr)
        rc=1
    bad_bd={r for r in _BIND_REFUSED if is_paper(r[0])}
    if bad_bd:   # benchmark_audit D4 (2026-10-18)
        print(f"ERROR: {len(bad_bd)} binding-site result JSONs of paper rows are not on the "
              f"current binding-site labels (refused); see '## Refused binding-site cells' in "
              f"{out.name}", file=sys.stderr)
        rc=1
    bad_sp={r for r in _SPLIT_REFUSED if is_paper(r[0])}
    if bad_sp:   # benchmark_audit D3 (2026-10-18)
        print(f"ERROR: {len(bad_sp)} EC / GO / binding result JSONs of paper rows are not on the "
              f"current MISATO split file (refused); see '## Refused cells (MISATO split file)' "
              f"in {out.name}", file=sys.stderr)
        rc=1
    bad_fn={r for r in _FN_REFUSED if is_paper(r[0])} | {r for r in _ITEM_REFUSED if is_paper(r[0])}
    bad_fn|={r for r in _CV_REFUSED if is_paper(r[0])}   # function CV (2026-10-19)
    if bad_fn:   # function_reaudit (2026-10-19)
        print(f"ERROR: {len(bad_fn)} function cells of paper rows were refused (protocol / inputs / "
              f"items / classes); see '## Refused cells (function protocol)' and '## Refused cells "
              f"(items / classes)' in {out.name}", file=sys.stderr)
        rc=1
    return rc

if __name__ == "__main__":
    sys.exit(main())
