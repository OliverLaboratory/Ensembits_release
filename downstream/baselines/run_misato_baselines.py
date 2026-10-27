"""Run the MISATO baseline probes (EC, GO, binding / affinity, RMSF) and the seed rescue.

What it does
------------
For the given baselines, splits and seeds it launches, as subprocesses spread
over the given GPUs (one job per GPU slot, CUDA_VISIBLE_DEVICES set per job,
`--device cuda:0` inside):

  1. Function probes with the stabilised recipe
     (`--tag stab --warmup 5 --min-epochs 20 --patience 12 --cosine`):
       * EC   `downstream/baselines/ec_classify_conv1d_misato.py --depths 1,2,3`
              (one job per baseline x CV mode x fold x seed, `--cv-mode <m> --fold <r>`;
              2026-10-19, see FN_PROTOCOL; old: per split);
       * GO   `downstream/baselines/conv1d_GO_misato.py --scope top50`
              (one job per baseline x CV mode x fold x seed);
       * BIND `downstream/baselines/conv1d_binding_baselines.py --task both
              --use-ligand-maccs --splits <all splits>` (one job per baseline x seed).
  2. RMSF probes (early stopping on val, Spearman on test, outputs tagged `__test`):
       * MISATO: `downstream/probe_rmsf_misato.py --mode <baseline> --split <sp>`
         for the `--rmsf-splits` (default sequence, structure);
       * mdCATH-div: `downstream/probe_rmsf_baselines.py --feature <baseline>`
         (and BioEmu-CATH with `--dataset bioemu`, task rmsf_bioemu, 2026-10-16)
         (no split; `protprofile_K8` is run as `protprofile_K10`, the 10-frame
         mdCATH version, as in `_compile_rebuttal_results.py`).
     Baselines without a mode in a probe (e.g. the `*_pooled` features) are
     skipped for that probe, with a note.
  2'. D5 affinity controls (task `aff_ctrl`, 2026-10-19; not paper rows): `ligand_only`
     and protein only (no MACCS) for esm3struct and aa, `--task affinity`, same splits,
     seeds and recipe (outputs baseline_<F>_stab[_seed<s>].json, no "_lig").
  3. Seed rescue (`downstream/seed_rescue.py`, shared with the Ensembits battery
     `downstream/_run_model_battery.py`; from `_audit_degeneracy_rerun.py`): reads the
     stored test metrics of the EC / GO / BIND result JSONs of step 1; for every cell (task, baseline, split, metric) with
     >= 4 seeds and a seed std > 0.1 it flags seeds below
     median - max(0.10, 3 x 1.4826 x MAD); flagged seeds whose run stopped before
     40 epochs ("premature") are re-run as a complete run
     (`--warmup 5 --min-epochs 40 --patience 999 --cosine`, same output file).
     Seeds that already ran 40 epochs are left as they are (genuine instability).
     Each re-run JSON then gets "rescued": true and the original epoch counts.

A job whose output file already exists is skipped (resumable), except the
rescue re-runs, which overwrite. `--dry-run` only prints the commands (relative
to this codebase's root); the rescue step then lists the seeds it would re-run
from the result files that exist now.

Paper
-----
Infrastructure only; it produces the baseline rows of
  * `appendix.tex` \\label{app:downstreams} (Tables \\label{tab:ec_d1_structure},
    \\label{tab:go_structure}, \\label{tab:binding_structure},
    \\label{tab:affinity_structure}) and \\label{app:experiments} (all three
    splits, mean +- std over 10 seeds);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * `2_results.tex` \\label{result:rmsf}, Table \\label{tab:rmsf} (baseline rows,
    mdCATH-div and MISATO sequence / structure columns);
  * baselines: `appendix.tex` \\label{app:baselines}.

Origin
------
New file, assembled from `submission_exp/src/_audit_degeneracy_rerun.py`
(the baseline part of `jobs()`: command and output-name templates; `read_cells`
and the outlier / premature-seed rule of `main`, unchanged) and from
`downstream/_run_model_battery.py` (`run_pool`, `env_libs`, the RMSF job
templates). The original sweep launcher of the stabilised baseline runs was
lost (code_fix C6); this script rebuilds it from those templates. Not copied:
the audit's Ensembits-model jobs (MODELS / PMODES), its hard-coded interpreter,
CUDA library folder and temporary log folder. Changed with respect to the originals:
`sys.executable` instead of a conda path; scripts are the copies in this
codebase (paths relative to its root, subprocess cwd = codebase root); logs in
REPO_ROOT/logs/misato_baselines_log; the RMSF MISATO jobs do not pass the old
fixed `--epochs 40` (code_fix §7, as in `_run_model_battery.py`).

Input standardisation (2026-10-19): the launched EC / GO / binding / affinity probes
standardise their head inputs by default (function_common.InputStandardisation); the
runner passes nothing new. Existing outputs of the previous FUNCTION_PROTOCOL are
refused by the stale-output check below (abort: move them aside, then rerun).

Used by the reported model: no (baseline rows only).

Inputs / outputs
----------------
Reads the feature caches of each probe (data_prep/baselines/) and, for the
rescue, the result JSONs below. Writes (relative to REPO_ROOT):
  output/ec_conv1d_misato/<baseline>__cv_<mode>_f<r>__stab[__seed<s>]__{d1,d2,d3,summary}.json
  output/go_conv1d_misato/go_<baseline>_top50_cv_<mode>_f<r>_stab[_seed<s>].json
    (function CV, 2026-10-19; --function-protocol split: the old names with [__<split>] /
    [_<split>])
  output/binding_misato/baseline_<baseline>_lig_stab[_seed<s>].json
  output/probe_rmsf_misato/baseline_<baseline>[_<split>][_seed<s>]__test.json
  output/probe_rmsf_div/baseline_<feature>[_seed<s>]__test.json
  output/probe_rmsf_bioemu/baseline_<feature>[_seed<s>]__test.json   (rmsf_bioemu, 2026-10-16)
  logs/misato_baselines_log/<job>.log

How to run
----------
    python downstream/baselines/run_misato_baselines.py --dry-run --baselines aa --seeds 0
    python downstream/baselines/run_misato_baselines.py --gpus 0,1,2,3 \\
        [--baselines esm3struct,aminoaseed,...] [--splits sequence,structure,random] \\
        [--seeds 0-9] [--tasks ec,go,bind,rmsf_misato,rmsf_div,rmsf_bioemu,rescue] \\
        [--function-protocol cv|split] [--folds 0-4]
"""
import os, sys, json, shlex, statistics as st, subprocess, queue, threading, argparse
from pathlib import Path

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
# FIXED(code_fix P2, 2026-10-16): old: RESCUE_RECIPE, MAXEP, read_cells and the outlier rule
#   were defined here (baselines only). New: they live in downstream/seed_rescue.py, which
#   downstream/_run_model_battery.py uses too, so the Ensembits rows get the same rescue
#   and the two callers cannot drift. Rescued result JSONs get "rescued": true.
sys.path.insert(0, str(CODE_ROOT / "downstream"))
from seed_rescue import RESCUE_RECIPE, MAXEP, read_cells, find_rescues, run_rescue_pass  # noqa: E402,F401

PY = sys.executable   # the interpreter running this driver (the original hard-coded a conda path)
# Launched scripts, relative to CODE_ROOT (every subprocess runs with cwd = CODE_ROOT).
SRC = "downstream/baselines"
EC_PY = f"{SRC}/ec_classify_conv1d_misato.py"
GO_PY = f"{SRC}/conv1d_GO_misato.py"
BIND_PY = f"{SRC}/conv1d_binding_baselines.py"
RMMIS_PY = "downstream/probe_rmsf_misato.py"
RMDIV_PY = "downstream/probe_rmsf_baselines.py"

# Output folders (all in the research repo).
EC = ROOT/"output/ec_conv1d_misato"; GO = ROOT/"output/go_conv1d_misato"; BD = ROOT/"output/binding_misato"
RMMIS = ROOT/"output/probe_rmsf_misato"; RMDIV = ROOT/"output/probe_rmsf_div"
RMBIO = ROOT/"output/probe_rmsf_bioemu"   # phase 2 (2026-10-16): BioEmu-CATH RMSF baselines

# Downstream-head training recipe for EC/GO/binding ("stabilised recipe": LR warmup
# 5 epochs -> cosine, at least 20 epochs, early-stopping patience 12), as in
# downstream/_run_model_battery.py.
# FIXED(require_fix #19, 2026-10-15): appendix "Probe training" paragraph lists the recipe.
TAG = "stab"; RECIPE = ["--warmup", "5", "--min-epochs", "20", "--patience", "12", "--cosine"]
# Seed rescue: RESCUE_RECIPE / MAXEP come from downstream/seed_rescue.py (code_fix P2).
# Paper: appendix.tex app:experiments — sequence / structure / random splits, 10 seeds.
SPLITS = ["sequence", "structure", "random"]; SEEDS = list(range(10))
# FIXED(function CV, authors' decision option A, 2026-10-19): the EC / GO jobs run the
#   cross-validation of data/misato_function_cv_v1.json (distinct proteins, grouped
#   label-stratified 5 folds; downstream/function_cv.py): CV modes (sequence, structure,
#   restricted by --splits) x folds 0-4 (--folds for previews) x seeds, with --cv-mode / --fold.
#   Old (--function-protocol split): one EC / GO job per split. BIND / AFFCTRL / RMSF keep the
#   fixed v2 splits.
FN_PROTOCOL = "cv"; CV_MODES = ["sequence", "structure"]; FOLDS = list(range(5))


def ec_go_units(splits, sep):
    """[(probe arguments, file-name part, job split id)] of the EC / GO jobs; sep = the
    separator of the file names ("__" EC, "_" GO)."""
    if FN_PROTOCOL == "split":
        return [(["--split", sp], f"{sep}{sp}" if sp != "sequence" else "", sp) for sp in splits]
    return [(["--cv-mode", m, "--fold", str(r)], f"{sep}cv_{m}_f{r}", f"cv_{m}_f{r}")
            for m in CV_MODES if m in splits for r in FOLDS]
# Paper: 2_results.tex result:rmsf — MISATO sequence and structure columns.
RMSF_SPLITS = ["sequence", "structure"]
# Default baselines: the list of _audit_degeneracy_rerun.py (paper rows + pooled rebuttal rows).
# FIXED(code_fix P16, 2026-10-16): old: no "protoken_pooled" (the compiler had a row for it
#   but nothing ran it). New: added, as in BASELINES of _compile_rebuttal_results.py.
BASELINES = ["esm3struct", "aminoaseed", "3di_tokens", "protoken", "vote_3di", "protprofile_K8", "aa", "random",
             "esm3struct_pooled", "aminoaseed_pooled", "3di_pooled", "protoken_pooled",
             "aminoaseed_retrained", "esm3struct_retrained"]
# Baseline modes of the two RMSF probes (their --mode / --feature choices, minus
# the Ensembits 'variant' mode).
# FIXED(preretrain_audit_r2 G14, 2026-10-17): the side-experiment modes without a cache
#   builder in this codebase (esm2, protprofile_K5, aug_rot6d, ensembits_ifold*, big_*,
#   *_ourdata) were removed from the probes, and from these lists.
RMMIS_MODES = {"aa", "3di_tokens", "protprofile_K8", "vote_3di", "pca_eig", "random",
               "aminoaseed", "protoken", "esm3struct",
               "aminoaseed_retrained", "esm3struct_retrained"}
RMDIV_FEATURES = {"aa", "mini3di", "3di_tokens", "protprofile_K10",
                  "vote_3di", "pca_eig", "random", "aminoaseed", "protoken", "esm3struct",
                  "aminoaseed_retrained", "esm3struct_retrained"}
# mdCATH-div has 10 frames, so the frame-matched ProtProfileMD there is K10
# (MIS_STEM in downstream/_compile_rebuttal_results.py).
DIV_FEATURE = {"protprofile_K8": "protprofile_K10"}


# Seed suffixes of the output names (seed 0 has none): single "_" for GO / BIND /
# RMSF, double "__" for EC.
def sdm(s): return "" if s == 0 else f"_seed{s}"
def sdb(s): return "" if s == 0 else f"__seed{s}"


# FIXED(function_reaudit pre-run fix 4, 2026-10-19): new D5 affinity control rows (not paper
#   rows): `ligand_only` (MACCS -> MLP, batch size and recipe of the other affinity runs) and
#   protein only (no MACCS) for CONTROL_PROTEIN_BASE, with the same splits, seeds, recipe and
#   early stopping, and the same complexes (finite label, MACCS, >= 10 residues; see
#   function_common.affinity_items). Outputs baseline_ligand_only_stab[_seed<s>].json and
#   baseline_<F>_stab[_seed<s>].json (no "_lig"). Task "aff_ctrl" (job type AFFCTRL); the
#   Ensembits protein-only rows are run by _run_model_battery.py (task AFFCTRL).
CONTROL_PROTEIN_BASE = ["esm3struct", "aa"]


def control_jobs(splits, seeds, recipe):
    """AFFCTRL jobs (D5 affinity controls) as (task, key, pmode, split, seed, out, cmd)."""
    J = []
    for F in ["ligand_only"] + CONTROL_PROTEIN_BASE:
        for s in seeds:
            o = BD/f"baseline_{F}_{TAG}{sdm(s)}.json"
            c = [PY, BIND_PY, "--feature", F, "--device", "cuda:0", "--task", "affinity", "--splits", ",".join(splits), "--seed", str(s), "--tag", TAG]+recipe
            J.append(("AFFCTRL", F, None, None, s, o, c))
    return J


def function_jobs(baselines, splits, seeds, recipe, controls=False):
    """EC / GO / BIND jobs as (task, baseline, pmode, split, seed, out, cmd).

    Templates of the baseline loop of _audit_degeneracy_rerun.py `jobs()`; with
    controls=True also the D5 affinity control jobs (control_jobs, 2026-10-19).
    """
    J = control_jobs(splits, seeds, recipe) if controls else []
    for F in baselines:
        # function CV (2026-10-19): ec_go_units() = CV modes x folds (or the old splits)
        for (spa, spf, sp), (_, spf2, _) in zip(ec_go_units(splits, "__"), ec_go_units(splits, "_")):
            for s in seeds:
                o = EC/f"{F}{spf}__{TAG}{sdb(s)}__summary.json"
                c = [PY, EC_PY, "--feature", F, "--device", "cuda:0", *spa, "--seed", str(s), "--depths", "1,2,3", "--tag", TAG]+recipe
                J.append(("EC", F, None, sp, s, o, c))
                o = GO/f"go_{F}_top50{spf2}_{TAG}{sdm(s)}.json"
                c = [PY, GO_PY, "--feature", F, "--device", "cuda:0", "--scope", "top50", *spa, "--seed", str(s), "--tag", TAG]+recipe
                J.append(("GO", F, None, sp, s, o, c))
        for s in seeds:
            o = BD/f"baseline_{F}_lig_{TAG}{sdm(s)}.json"
            c = [PY, BIND_PY, "--feature", F, "--device", "cuda:0", "--task", "both", "--splits", ",".join(splits), "--use-ligand-maccs", "--seed", str(s), "--tag", TAG]+recipe
            J.append(("BIND", F, None, None, s, o, c))
    return J


def rmsf_jobs(baselines, rmsf_splits, seeds, tasks):
    """RMSF jobs as (name, cmd, out); baselines without a probe mode are skipped."""
    J = []
    for F in baselines:
        if "rmsf_misato" in tasks:
            if F not in RMMIS_MODES:
                print(f"[note] {F}: no baseline mode in probe_rmsf_misato.py; skipped", flush=True)
            else:
                for sp in rmsf_splits:
                    spf = "" if sp == "sequence" else f"_{sp}"
                    for s in seeds:
                        o = RMMIS/f"baseline_{F}{spf}{sdm(s)}__test.json"
                        # code_fix §7: no fixed --epochs 40 any more; the probe runs to
                        # convergence (early stopping on val) and reports test.
                        c = [PY, RMMIS_PY, "--mode", F, "--split", sp, "--seed", str(s), "--device", "cuda:0"]
                        J.append((f"RMMIS/{F}/{sp}/s{s}", c, o))
        # FIXED(preretrain_audit_r2 F8, 2026-10-16): new task rmsf_bioemu = the same probe
        #   as rmsf_div with --dataset bioemu (ProtProfileMD K10, 10 frames).
        for task, out_dir, extra, tag in (("rmsf_div", RMDIV, [], "RMDIV"),
                                          ("rmsf_bioemu", RMBIO, ["--dataset", "bioemu"], "RMBIO")):
            if task not in tasks:
                continue
            fd = DIV_FEATURE.get(F, F)
            if fd not in RMDIV_FEATURES:
                print(f"[note] {F}: no feature in probe_rmsf_baselines.py; skipped", flush=True)
            else:
                for s in seeds:
                    o = out_dir/f"baseline_{fd}{sdm(s)}__test.json"
                    c = [PY, RMDIV_PY, "--feature", fd, "--seed", str(s), "--device", "cuda:0"] + extra
                    J.append((f"{tag}/{fd}/s{s}", c, o))
    return J


def rescue_jobs(J):
    """Flag premature low-outlier seeds; -> {out_path: rescue_cmd}.

    FIXED(code_fix P2, 2026-10-16): thin wrapper over seed_rescue.find_rescues (the rule
    is unchanged; it is shared with downstream/_run_model_battery.py). The selection
    uses the stored test metrics of the result JSONs.
    """
    return {pth: ent["cmd"] for pth, ent in find_rescues(J).items()}


def run_pool(jobs, gpus, env0, phase, LOG, per_gpu=2, dry=False):
    # Run (name, command, expected_output) jobs, per_gpu at a time on each GPU;
    # jobs whose output already exists are skipped. Each subprocess runs with
    # cwd = CODE_ROOT and CUDA_VISIBLE_DEVICES set to its GPU.
    # FIXED(benchmark_audit D7, 2026-10-18): a GO output not built on the current GO labels
    #   (go_labels.go_result_is_current) is re-run, not skipped (old: any existing file skipped).
    from go_labels import go_result_is_current
    stale_go = lambda o: Path(o).name.startswith("go_") and not go_result_is_current(o)
    todo = [(n, c, o) for (n, c, o) in jobs if o is None or not Path(o).exists() or stale_go(o)]
    print(f"[{phase}] total={len(jobs)} todo={len(todo)}", flush=True)
    if dry:
        # Print every job; those with an existing output are marked (they would be skipped).
        for n, c, o in jobs:
            mark = "  [exists, skip] " if (o is not None and Path(o).exists()) else "  "
            print(mark + shlex.join(["python"] + [str(x) for x in c[1:]]), flush=True)
        return
    q = queue.Queue()
    for j in todo: q.put(j)
    slots = [g for g in gpus for _ in range(per_gpu)]
    done = [0, len(todo)]; lock = threading.Lock()
    def worker(g):
        while True:
            try: n, c, o = q.get_nowait()
            except queue.Empty: return
            env = dict(env0); env["CUDA_VISIBLE_DEVICES"] = str(g)
            lf = LOG/(n.replace("/", "__").replace(":", "_")+".log")
            rc = subprocess.run(c, cwd=str(CODE_ROOT), env=env, stdout=open(lf, "w"), stderr=subprocess.STDOUT).returncode
            with lock:
                done[0] += 1
                miss = "" if (o is None or Path(o).exists()) else "  !!NO-OUT"
                print(f"[{phase} {done[0]}/{done[1]}] gpu{g} {n} rc={rc}{miss}", flush=True)
            q.task_done()
    ts = [threading.Thread(target=worker, args=(g,), daemon=True) for g in slots]
    for t in ts: t.start()
    for t in ts: t.join()


def env_libs():
    # Subprocess environment (as downstream/_run_model_battery.py): CUDA libraries
    # from this interpreter's pip `nvidia` packages on LD_LIBRARY_PATH, Hugging
    # Face offline, 4 BLAS threads.
    SP = str(Path(sys.prefix)/"lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages"/"nvidia")
    libs = ":".join(str(p) for p in Path(SP).rglob("lib") if p.is_dir())
    e = dict(os.environ); e["LD_LIBRARY_PATH"] = libs+":"+e.get("LD_LIBRARY_PATH", "")
    e["HF_HUB_OFFLINE"] = "1"; e["TRANSFORMERS_OFFLINE"] = "1"
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): e[v] = "4"
    return e


def _seeds(text):
    # "0-9" or "0,1,2"
    if "-" in text and "," not in text:
        a, b = text.split("-"); return list(range(int(a), int(b)+1))
    return [int(x) for x in text.split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baselines", default=",".join(BASELINES),
                    help="comma-separated feature names (default: the audit list)")
    ap.add_argument("--splits", default=",".join(SPLITS),
                    help="splits of the EC / GO / BIND probes")
    ap.add_argument("--rmsf-splits", default=",".join(RMSF_SPLITS),
                    help="splits of the MISATO RMSF probe")
    ap.add_argument("--seeds", default="0-9", help='"0-9" or "0,1,2"')
    # Q4: rescue removed from defaults (supersedes FFL A13; no seed is rerun or excluded)
    ap.add_argument("--tasks", default="ec,go,bind,aff_ctrl,rmsf_misato,rmsf_div,rmsf_bioemu",
                    help="subset of ec,go,bind,aff_ctrl,rmsf_misato,rmsf_div,rmsf_bioemu,rescue "
                         "(aff_ctrl = D5 affinity controls; rescue = opt-in seed rescue)")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per-gpu", type=int, default=2, help="jobs per GPU")
    ap.add_argument("--dry-run", action="store_true", help="only print the commands")
    # FIXED(function CV, 2026-10-19): new --function-protocol / --folds (EC / GO jobs).
    ap.add_argument("--function-protocol", choices=("cv", "split"), default="cv",
                    help="EC / GO jobs: cv (default, paper rows) = --cv-mode x --fold of "
                         "data/misato_function_cv_v1.json; split = the old fixed-split jobs")
    ap.add_argument("--folds", default="0-4",
                    help='CV folds, "0-4" or "0,1" (fewer for previews; the compiler needs all five)')
    a = ap.parse_args()
    global FN_PROTOCOL, FOLDS
    FN_PROTOCOL = a.function_protocol
    FOLDS = _seeds(a.folds)
    if set(FOLDS) - set(range(5)):
        ap.error(f"folds must be in 0-4: {FOLDS}")
    baselines = [x for x in a.baselines.split(",") if x]
    splits = [x for x in a.splits.split(",") if x]
    rmsf_splits = [x for x in a.rmsf_splits.split(",") if x]
    seeds = _seeds(a.seeds)
    tasks = set(a.tasks.split(","))
    gpus = [int(x) for x in a.gpus.split(",")]
    env0 = None if a.dry_run else env_libs()
    LOG = ROOT/"logs"/"misato_baselines_log"
    if not a.dry_run:
        LOG.mkdir(parents=True, exist_ok=True)

    # 1 + 2: function probes (stabilised recipe) and RMSF probes, one GPU pool.
    task_of = {"EC": "ec", "GO": "go", "BIND": "bind", "AFFCTRL": "aff_ctrl"}
    jobs = [(f"{t}/{F}/{sp}/s{s}", c, o)
            for (t, F, p, sp, s, o, c) in function_jobs(baselines, splits, seeds, RECIPE,
                                                        controls="aff_ctrl" in tasks)
            if task_of[t] in tasks]
    rj = rmsf_jobs(baselines, rmsf_splits, seeds, tasks)
    # FIXED(preretrain_audit_r2 F8, 2026-10-16): the pool skips a job whose output exists, so
    #   an existing RMSF result is only reused when it carries the current residue set and
    #   label (rmsf_common.stale_rmsf_outputs); otherwise the run aborts (move it aside).
    sys.path.insert(0, str(CODE_ROOT / "downstream"))
    from rmsf_common import stale_rmsf_outputs
    bad = []
    for key, ds in (("RMDIV/", "mdcath_div"), ("RMMIS/", "misato"), ("RMBIO/", "bioemu")):
        bad += stale_rmsf_outputs([o for n, _, o in rj if n.startswith(key)], ds)
    if bad:
        for o, why in bad[:20]: print(f"[stale] {o.name}: {why}", flush=True)
        if not a.dry_run:
            print("[abort] move the stale RMSF outputs aside", flush=True); sys.exit(1)
    # FIXED(benchmark_audit D3, 2026-10-18): new. Existing EC / GO / BIND outputs not built on
    #   the current MISATO split file (ensembits.paths.MISATO_SPLITS = misato_splits_v2.json;
    #   no record = the old misato_splits.json) are never reused.
    from ensembits.data_versions import stale_split_outputs
    bad_sp = stale_split_outputs([o for _, _, o in jobs if o is not None])
    if bad_sp:
        for o, why in bad_sp[:20]: print(f"[stale] {o.name}: {why}", flush=True)
        if not a.dry_run:
            print("[abort] move the EC / GO / BIND outputs of the old split file aside",
                  flush=True); sys.exit(1)
    # FIXED(benchmark_audit D4, 2026-10-18): new. Existing BIND outputs whose binding results were
    #   not built on the current binding-site labels (misato_binding_site_v2.npz) are never reused.
    from ensembits.misato_binding_labels import stale_binding_label_outputs
    bad_bl = stale_binding_label_outputs([o for _, _, o in jobs if o is not None])
    if bad_bl:
        for o, why in bad_bl[:20]: print(f"[stale] {o.name}: {why}", flush=True)
        if not a.dry_run:
            print("[abort] move the BIND outputs of the old binding-site labels aside",
                  flush=True); sys.exit(1)
    # FIXED(benchmark_audit D8, 2026-10-18): new. Existing EC outputs not built on the current
    #   EC labels (data/annotations/misato_ec_v2.json; no "ec_labels" record = the old
    #   misato_ec.json) are never reused.
    from ec_labels import stale_ec_outputs
    bad_ec = stale_ec_outputs([o for _, _, o in jobs if o is not None and Path(o).parent == EC])
    if bad_ec:
        for o, why in bad_ec[:20]: print(f"[stale] {o.name}: {why}", flush=True)
        if not a.dry_run:
            print("[abort] move the EC outputs of the old EC labels aside",
                  flush=True); sys.exit(1)
    # FIXED(function_reaudit pre-run fixes, 2026-10-19): new. Existing EC / GO / BIND / AFFCTRL
    #   outputs without the current function protocol (function_common.FUNCTION_PROTOCOL:
    #   same items / classes, aa from misato_real_bb, verified inputs, val metrics) or on other
    #   affinity labels / MACCS files are never reused.
    from function_common import stale_function_outputs
    bad_fn = stale_function_outputs([o for _, _, o in jobs if o is not None])
    # FIXED(function CV, 2026-10-19): new. Existing EC / GO CV outputs not built on the current
    #   CV file (function_cv.cv_result_problem: sha256) are never reused.
    if FN_PROTOCOL == "cv":
        from function_cv import stale_cv_outputs
        bad_fn += stale_cv_outputs([o for n, _, o in jobs if o is not None
                                    and n.split("/")[0] in ("EC", "GO")])
    if bad_fn:
        for o, why in bad_fn[:20]: print(f"[stale] {o.name}: {why}", flush=True)
        if not a.dry_run:
            print("[abort] move the function outputs of the old protocol aside", flush=True)
            sys.exit(1)
    jobs += rj
    run_pool(jobs, gpus, env0, "probes", LOG, per_gpu=a.per_gpu, dry=a.dry_run)

    # 3: seed rescue on the EC / GO / BIND outputs (re-runs overwrite).
    if "rescue" in tasks:
        if a.dry_run:
            print("[rescue] dry run: flagged from the result files that exist now", flush=True)
        J = [j for j in function_jobs(baselines, splits, seeds, RESCUE_RECIPE,
                                      controls="aff_ctrl" in tasks) if task_of[j[0]] in tasks]
        # FIXED(code_fix P2, 2026-10-16): old: rescue_jobs + run_pool here. New: the shared
        #   pass (seed_rescue.run_rescue_pass: same rule, same re-run, then "rescued": true
        #   and the original epochs written into each re-run JSON). The selection uses the
        #   stored test metrics.
        run_rescue_pass(J, lambda rj: run_pool(rj, gpus, env0, "rescue", LOG, per_gpu=a.per_gpu,
                                               dry=a.dry_run), dry=a.dry_run)
    print("MISATO BASELINES DONE" if not a.dry_run else "[dry-run] nothing was run", flush=True)


if __name__ == "__main__":
    main()
