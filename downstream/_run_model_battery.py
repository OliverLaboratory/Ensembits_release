"""Generic full-task battery for one trained dihedral tokenizer variant.
Runs ALL tasks incl. MISATO P=1, then regenerates results_bf65.md.

What it does
------------
Launches every downstream evaluation of one checkpoint as subprocesses, spread
over the given GPUs (a job is skipped when its output file already exists, so
the battery can be resumed):

  A: pretokenize misato (P=full + P=1).
     `_pretokenize_misato_dihedral.py` writes the MISATO token caches
     (P = 8 real-backbone frames, and the first frame only).
  B (pool): RMSF-div (10s x {Pfull,P1}), RMSF-misato (seq+struct x10 x {Pfull,P1}),
     RMSF-BioEmu (10s x {Pfull,P1}; phase 2, 2026-10-16),
            function EC/GO/binding+affinity (3 splits x10, stab, Pfull+P1; since
            2026-10-19 EC/GO = 2 CV modes x 5 folds x10, see FN_PROTOCOL).
     `probe_rmsf_eval.py`, `probe_rmsf_misato.py`, `tokenize_misato_and_eval.py`
     (EC depths 1-3), `tokenize_misato_GO_eval.py` (GO top-50),
     `tokenize_misato_binding_eval.py` (binding site + ligand-aware affinity).
  B'': AFFCTRL (2026-10-19): the D5 protein-only affinity control of the variant
      (`tokenize_misato_binding_eval.py --task affinity` without --use-ligand-maccs,
      P=full and P=1, 10 seeds; output <vtag>[_P1inf]_stab[_seed<s>].json).
  B': seed rescue of the EC / GO / binding jobs (`seed_rescue.py`, the same rule
      and re-run recipe as the baselines in `baselines/run_misato_baselines.py`;
      re-run JSONs get "rescued": true). Added 2026-10-16 (code_fix P2).
  C: ProteinGym (pg_score_dihedral).
  D: regenerate results_bf65.md (`_compile_rebuttal_results.py`; pass the
     reported-model rows with --compile-args, see there).

All launched scripts are the copies in this codebase's `downstream/` folder; they
read and write data under REPO_ROOT (each subprocess runs with cwd = REPO_ROOT).

Paper
-----
Infrastructure only; it produces the numbers of
  * `2_results.tex` \\label{result:rmsf} (Table tab:rmsf: 10 seeds, P=full/P=1,
    mdCATH-div + MISATO sequence/structure),
  * `2_results.tex` \\label{sec:downstreams} (ProteinGym, Table
    tab:proteingym_ensemble; EC/GO/binding radar, Fig. fig:downstream_radar),
  * `appendix.tex` \\label{app:downstreams} and \\label{app:experiments}
    (EC/GO/binding/affinity tables, 10 seeds, sequence/structure/random splits),
  * `appendix.tex` \\label{sec:compute} ("Full reproduction budget").

Origin
------
Copied from `submission_exp/src/_run_model_battery.py`. Changes: ROOT is now
`ensembits.paths.REPO_ROOT` (data, outputs and subprocess cwd stay in the
research repo); SRC now points to this codebase's `downstream/` folder, because
every script the battery launches has a copy there; comments and this
docstring. Approved fixes 2026-10-16 (FIXED markers): seed rescue pass for the
variant (P2), `--gpus` default = all visible devices, also passed to
pg_score_dihedral (P11), `--dry-run`, `--allow-unfinished`, `--compile-args`.

Input standardisation (2026-10-19): the launched EC / GO / binding / affinity probes
standardise their head inputs by default (function_common.InputStandardisation); the
runner passes nothing new. Existing outputs of the previous FUNCTION_PROTOCOL are
refused by the stale-output check below (abort: move them aside, then rerun).

Used by the reported model: yes; this is how the reported model's downstream
numbers were produced (variant
`dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO`, PG tag
`dih256_fapeH_bioemu_misato`).

Inputs / outputs
----------------
Reads the checkpoint `output/<variant>/` and the data caches each task reads.
Writes (REPO_ROOT; <vtag> = tag_of(<variant>), the name every script writes; G13):
  data/cached_descriptors/ensembits_<vtag>[_P1inf]_tokens_misato.npz
  output/probe_rmsf_div/<vtag>[_P1inf][_seed<s>]__test.json
  output/probe_rmsf_misato/<vtag>[_P1inf][_structure][_seed<s>]__test.json
  output/probe_rmsf_bioemu/<vtag>[_P1inf][_seed<s>]__test.json
  output/ec_conv1d_misato/tok_<vtag>[_P1inf]_cv_<mode>_f<r>_stab[_seed<s>]__summary.json
  output/go_conv1d_misato/go_<vtag>[_P1inf]_cv_<mode>_f<r>_stab[_seed<s>]_top50.json
    (function CV, 2026-10-19: mode sequence / structure, fold r = 0-4; with
    --function-protocol split the old names [_<split>] instead of _cv_<mode>_f<r>)
  output/binding_misato/<vtag>[_P1inf]_lig_stab[_seed<s>].json
  output/proteingym/per_assay_pg_<pgtag>{/,.json}
  rebuttal/results_bf65.md
Logs: one file per job under REPO_ROOT/logs/battery_<pgtag>_log/ (see LOG).

Dependencies left in the research repo: none in code (all launched scripts are
in `downstream/`); all data and outputs live under REPO_ROOT.

How to run
----------
    python downstream/_run_model_battery.py \\
        --variant dihedral256_fapeH_rvq_2048_128_128_reviewer_3_special_corpus_MISATO \\
        --pgtag dih256_fapeH_bioemu_misato [--gpus 0,1,2,3] [--dry-run] \\
        [--tasks PRETOK,EC,GO,BIND,RMDIV,RMMIS,RMBIO,PG,COMPILE] \\
        [--compile-args "--variant <run> --pgtag <tag>"] \\
        [--function-protocol cv|split] [--folds 0-4]
    # RMSF only (e.g. a snapshot of a running training, downstream/snapshot_checkpoint.py):
    python downstream/_run_model_battery.py --variant snap_<run>_ep<k> \\
        --tasks PRETOK,RMDIV,RMMIS,RMBIO --allow-unfinished
(2026-10-16, F8: --tasks; the compile step runs only with COMPILE and --compile-args;
--pgtag is required only with PG.)
"""
import os, sys, shlex, subprocess, queue, threading, argparse
from pathlib import Path

# Codebase root on sys.path; data/ and output/ stay in the research repo.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo

# Folder of the launched scripts. All eight of them have a copy in this
# codebase's downstream/ folder, so SRC points here (originally
# REPO_ROOT/submission_exp/src):
#   _pretokenize_misato_dihedral.py, probe_rmsf_eval.py, probe_rmsf_misato.py,
#   tokenize_misato_and_eval.py, tokenize_misato_GO_eval.py,
#   tokenize_misato_binding_eval.py, pg_score_dihedral.py,
#   _compile_rebuttal_results.py.
# Nothing else is launched, so no SRC entry still points at the research repo.
SRC = CODE_ROOT / "downstream"
sys.path.insert(0, str(SRC))
# FIXED(code_fix P2, 2026-10-16): old: no rescue for the Ensembits rows (only the baseline
#   runner had one). New: the shared rule of downstream/seed_rescue.py, used by both.
from seed_rescue import RESCUE_RECIPE, run_rescue_pass   # noqa: E402
# Interpreter: the one running this driver (the original hard-coded a conda env path).
PY = sys.executable   # the interpreter running this driver (was a hard-coded conda path)
# Downstream-head training recipe for EC/GO/binding ("stabilization recipe":
# LR warmup 5 epochs -> cosine, at least 20 epochs, early-stopping patience 12).
# FIXED(require_fix #19, 2026-10-15): appendix "Probe training" paragraph lists the recipe.
TAG="stab"; RECIPE=["--warmup","5","--min-epochs","20","--patience","12","--cosine"]
# Paper: appendix.tex app:downstreams — sequence / structure / random splits,
# mean ± std over 10 seeds; P=full and P=1 (None = all frames).
SPLITS=["sequence","structure","random"]; SEEDS=list(range(10)); PMODES=[None,1]
# FIXED(function CV, authors' decision option A, 2026-10-19): the EC / GO jobs run the
#   cross-validation of data/misato_function_cv_v1.json (distinct proteins, grouped
#   label-stratified 5 folds; downstream/function_cv.py): CV modes (sequence, structure,
#   restricted by --splits) x folds 0-4 (--folds for previews) x seeds, with --cv-mode / --fold.
#   Old (--function-protocol split): one EC / GO job per split (sequence, structure, random).
#   Binding site, affinity and RMSF keep the fixed v2 splits.
FN_PROTOCOL="cv"; CV_MODES=["sequence","structure"]; FOLDS=list(range(5))
def ec_go_units():
    """[(probe arguments, file-name part, job split id)] of the EC / GO jobs."""
    if FN_PROTOCOL=="split":
        return [(["--split",sp], f"_{sp}" if sp!="sequence" else "", sp) for sp in SPLITS]
    return [(["--cv-mode",m,"--fold",str(r)], f"_cv_{m}_f{r}", f"cv_{m}_f{r}")
            for m in CV_MODES if m in SPLITS for r in FOLDS]
# Output folders (all in the research repo).
CACHE=ROOT/"data/cached_descriptors"
EC=ROOT/"output/ec_conv1d_misato"; GO=ROOT/"output/go_conv1d_misato"
BD=ROOT/"output/binding_misato"; RMDIV=ROOT/"output/probe_rmsf_div"; RMMIS=ROOT/"output/probe_rmsf_misato"
# Steps of the battery (--tasks; F8 2026-10-16).
# FIXED(function_reaudit pre-run fix 4, 2026-10-19): new task AFFCTRL = the D5 protein-only
#   affinity control of the variant (no MACCS; same complexes, splits, seeds, recipe).
TASKS=("PRETOK","EC","GO","BIND","AFFCTRL","RMDIV","RMMIS","RMBIO","PG","COMPILE")
# FIXED(preretrain_audit_r2 G13, 2026-10-17): old = the expected output names used the raw
#   --variant, the scripts write tag_of(variant) (differs for final_model* names), so the
#   skip / stale checks looked at the wrong files. New = tag_of everywhere.
def tag_of(v): return v.replace("final_model_","").replace("final_model","") or v
# Seed suffix used in output file names (seed 0 has none).
def sd(s): return "" if s==0 else f"_seed{s}"

def run_pool(jobs,gpus,env0,phase,LOG,per_gpu=2,dry=False):
    # Run (name, command, expected_output) jobs, per_gpu at a time on each GPU;
    # jobs whose output already exists are skipped. Each subprocess runs with
    # cwd = REPO_ROOT and CUDA_VISIBLE_DEVICES set to its GPU.
    # FIXED(benchmark_audit D7, 2026-10-18): a GO output not built on the current GO labels
    #   (go_labels.go_result_is_current) is re-run, not skipped (old: any existing file skipped).
    from go_labels import go_result_is_current
    stale_go=lambda o: Path(o).name.startswith("go_") and not go_result_is_current(o)
    todo=[(n,c,o) for (n,c,o) in jobs if o is None or not Path(o).exists() or stale_go(o)]
    print(f"[{phase}] total={len(jobs)} todo={len(todo)}",flush=True)
    if dry:   # --dry-run: list the jobs (an existing output is marked: it would be skipped)
        for n,c,o in jobs:
            mark="  [exists, skip] " if (o is not None and Path(o).exists()) else "  "
            print(mark+shlex.join(["python"]+[str(x) for x in c[1:]]),flush=True)
        return
    q=queue.Queue()
    for j in todo: q.put(j)
    slots=[g for g in gpus for _ in range(per_gpu)]
    done=[0,len(todo)]; lock=threading.Lock()
    def worker(g):
        while True:
            try: n,c,o=q.get_nowait()
            except queue.Empty: return
            env=dict(env0); env["CUDA_VISIBLE_DEVICES"]=str(g)
            lf=LOG/(n.replace("/","__").replace(":","_")+".log")
            rc=subprocess.run(c,cwd=str(ROOT),env=env,stdout=open(lf,"w"),stderr=subprocess.STDOUT).returncode
            with lock:
                done[0]+=1
                miss="" if (o is None or Path(o).exists()) else "  !!NO-OUT"
                print(f"[{phase} {done[0]}/{done[1]}] gpu{g} {n} rc={rc}{miss}",flush=True)
            q.task_done()
    ts=[threading.Thread(target=worker,args=(g,),daemon=True) for g in slots]
    for t in ts: t.start()
    for t in ts: t.join()

def env_libs():
    # Subprocess environment: CUDA libraries from this interpreter's pip
    # `nvidia` packages on LD_LIBRARY_PATH, Hugging Face offline, 4 BLAS threads.
    # NVIDIA libraries shipped with the pip CUDA wheels of this interpreter.
    SP=str(Path(sys.prefix)/"lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages"/"nvidia")
    libs=":".join(str(p) for p in Path(SP).rglob("lib") if p.is_dir())
    e=dict(os.environ); e["LD_LIBRARY_PATH"]=libs+":"+e.get("LD_LIBRARY_PATH","")
    e["HF_HUB_OFFLINE"]="1"; e["TRANSFORMERS_OFFLINE"]="1"
    for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): e[v]="4"
    return e

def function_jobs(V,recipe,extra=()):
    """EC / GO / BIND jobs of the variant as (task, V, pmode, split, seed, out, cmd).

    The templates of the original battery loop, factored out so the rescue pass
    (code_fix P2) re-runs exactly these commands with RESCUE_RECIPE.
    """
    J=[]
    for p in PMODES:
        pih="_P1inf" if p else ""
        pp=["--p-inference",str(p)] if p else []
        # function CV (2026-10-19): ec_go_units() = CV modes x folds (or the old splits)
        for spa,spf,sp in ec_go_units():
            for s in SEEDS:
                o=EC/f"tok_{tag_of(V)}{pih}{spf}_{TAG}{sd(s)}__summary.json"
                c=[PY,str(SRC/"tokenize_misato_and_eval.py"),"--variant",V,"--device","cuda:0",*spa,"--seed",str(s),"--depths","1,2,3","--tag",TAG]+list(recipe)+pp+list(extra)
                J.append(("EC",V,p,sp,s,o,c))
                o=GO/f"go_{tag_of(V)}{pih}{spf}_{TAG}{sd(s)}_top50.json"
                c=[PY,str(SRC/"tokenize_misato_GO_eval.py"),"--variant",V,"--device","cuda:0","--scope","top50",*spa,"--seed",str(s),"--tag",TAG]+list(recipe)+pp+list(extra)
                J.append(("GO",V,p,sp,s,o,c))
        for s in SEEDS:
            o=BD/f"{tag_of(V)}{pih}_lig_{TAG}{sd(s)}.json"
            c=[PY,str(SRC/"tokenize_misato_binding_eval.py"),"--variant",V,"--device","cuda:0","--task","both","--splits","sequence,structure,random","--use-ligand-maccs","--seed",str(s),"--tag",TAG]+list(recipe)+pp+list(extra)
            # FIXED(code_fix P2, 2026-10-16): new (old: none): a rescue re-run must recompute the existing file (the binding
            # script otherwise skips a complete one); the baseline script always recomputes.
            if list(recipe)==list(RESCUE_RECIPE): c+=["--overwrite"]
            J.append(("BIND",V,p,None,s,o,c))
        # D5 protein-only affinity control (2026-10-19; output without "_lig").
        for s in SEEDS:
            o=BD/f"{tag_of(V)}{pih}_{TAG}{sd(s)}.json"
            c=[PY,str(SRC/"tokenize_misato_binding_eval.py"),"--variant",V,"--device","cuda:0","--task","affinity","--splits","sequence,structure,random","--seed",str(s),"--tag",TAG]+list(recipe)+pp+list(extra)
            if list(recipe)==list(RESCUE_RECIPE): c+=["--overwrite"]
            J.append(("AFFCTRL",V,p,None,s,o,c))
    return J

def stale_outputs(paths,prov):
    """Existing outputs of this variant written by another checkpoint: [(path, reason)].

    FIXED(code_fix P17, 2026-10-16): new. The pools skip a job whose output exists; an
      output is only reused if its "checkpoint" block (JSON) or provenance sidecar (token
      cache) names the same checkpoint. Outputs without a record are stale when the
      checkpoint was trained by the fixed trainer (it has a code_sha256); for a legacy
      checkpoint they are accepted.
    """
    import json
    from ensembits.featurize import same_checkpoint, provenance_sidecar
    bad=[]
    for o in paths:
        o=Path(o)
        if not o.exists(): continue
        try:
            if o.suffix==".json": rec=json.loads(o.read_text()).get("checkpoint")
            else:
                sc=provenance_sidecar(o)
                rec=json.loads(sc.read_text()).get("checkpoint") if sc.exists() else None
        except Exception as e:
            bad.append((o,f"unreadable ({type(e).__name__})")); continue
        if rec is None:
            if prov.get("code_sha256") is not None: bad.append((o,"no checkpoint record"))
        elif not same_checkpoint(rec,prov):
            bad.append((o,f"written by {rec.get('variant')} (best.pt {str(rec.get('best_pt_sha256'))[:12]})"))
    return bad

def visible_gpus():
    """All CUDA devices visible to this process, as CUDA_VISIBLE_DEVICES entries.

    FIXED(code_fix P11, 2026-10-16): old: --gpus default "0,1,2,3,4,5,6,7" (jobs on absent
      GPUs failed on a 4-GPU node). New: the default is every visible device:
      torch.cuda.device_count() devices, named by the parent's CUDA_VISIBLE_DEVICES entries
      when it is set (each job's CUDA_VISIBLE_DEVICES must name a physical device).
    """
    try:
        import torch
        n=torch.cuda.device_count()
    except Exception:
        n=0
    env=os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        return [x.strip() for x in env.split(",") if x.strip()][:n]
    return [str(i) for i in range(n)]

def main():
    # ---- CLI -------------------------------------------------------------
    # FIXED(code_fix E0 / LAM, 2026-10-23): reported variant = ensembits_bm_lam10_rv,
    #   PG tag = bm_lam10_rv. Defaults and compiler tables switched to _rv runs.
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant",required=True)
    ap.add_argument("--pgtag",default=None,help="ProteinGym output tag (required with the PG task)")
    # FIXED(preretrain_audit_r2 F8, 2026-10-16): new: --tasks selects the steps (default: all,
    #   as before). The RMSF-only run is --tasks PRETOK,RMDIV,RMMIS,RMBIO.
    ap.add_argument("--tasks",default=",".join(TASKS),
                    help=f"comma-separated subset of {','.join(TASKS)} (default: all)")
    # FIXED(code_fix P11, 2026-10-16): default = all visible devices (was 0..7).
    ap.add_argument("--gpus",default=None,
                    help="comma-separated CUDA device ids (default: all visible devices)")
    ap.add_argument("--dry-run",action="store_true",
                    help="print the jobs and the rescue decisions from the result files that "
                         "exist now; run nothing")
    ap.add_argument("--allow-unfinished",action="store_true",
                    help="pass --allow-unfinished to the scripts that load the checkpoint "
                         "(code_fix P12; for smoke tests of a running training only)")
    ap.add_argument("--force",action="store_true",
                    help="reuse existing outputs even if they were written by another "
                         "checkpoint (code_fix P17; default: refuse)")
    ap.add_argument("--compile-args",default="",
                    help="extra arguments of _compile_rebuttal_results.py in step D, e.g. "
                         "\"--variant <run> --pgtag <tag>\" for the reported model (default: none, "
                         "which compiles the pre-fix rows only)")
    # FIXED(first pass, 2026-10-19): --splits / --seeds restrict the MISATO jobs.
    # Q4: rescue OFF by default (supersedes FFL A13; the reported runs used --no-rescue).
    ap.add_argument("--splits",default=None,
                    help="comma-separated subset of sequence,structure,random (default: all)")
    ap.add_argument("--seeds",default=None,help='"0-9" or "0,1,2" (default: 0-9)')
    ap.add_argument("--rescue",action="store_true",
                    help="run the seed rescue pass (off by default; Q4: no seed is rerun or excluded)")
    # FIXED(function CV, 2026-10-19): new --function-protocol / --folds (EC / GO jobs).
    ap.add_argument("--function-protocol",choices=("cv","split"),default="cv",
                    help="EC / GO jobs: cv (default, paper rows) = --cv-mode x --fold of "
                         "data/misato_function_cv_v1.json; split = the old fixed-split jobs")
    ap.add_argument("--folds",default=None,help='CV folds, "0-4" or "0,1" (default: 0-4; '
                    'fewer folds for previews only, the compiler needs all five)')
    a=ap.parse_args()
    global SPLITS, SEEDS, FN_PROTOCOL, FOLDS
    FN_PROTOCOL=a.function_protocol
    if a.folds:
        FOLDS=(list(range(int(a.folds.split("-")[0]),int(a.folds.split("-")[1])+1))
               if "-" in a.folds else [int(x) for x in a.folds.split(",") if x.strip()])
        if set(FOLDS)-set(range(5)): ap.error(f"folds must be in 0-4: {FOLDS}")
    if a.splits:
        SPLITS=[s.strip() for s in a.splits.split(",") if s.strip()]
        if set(SPLITS)-{"sequence","structure","random"}: ap.error(f"unknown splits {SPLITS}")
    if a.seeds:
        SEEDS=(list(range(int(a.seeds.split("-")[0]),int(a.seeds.split("-")[1])+1))
               if "-" in a.seeds else [int(x) for x in a.seeds.split(",") if x.strip()])
    tasks={t.strip().upper() for t in a.tasks.split(",") if t.strip()}
    if tasks-set(TASKS): ap.error(f"unknown tasks {sorted(tasks-set(TASKS))}; choose from {TASKS}")
    if "PG" in tasks and not a.pgtag: ap.error("--pgtag is required with the PG task")
    V=a.variant; PGTAG=a.pgtag or V
    gpus=[x.strip() for x in a.gpus.split(",") if x.strip()] if a.gpus else visible_gpus()
    if not gpus and not a.dry_run:
        ap.error("no visible CUDA device; pass --gpus")
    dry=a.dry_run
    env0=None if dry else env_libs()
    AU=["--allow-unfinished"] if a.allow_unfinished else []
    # Log folder. The original wrote to a temporary session scratch folder under
    # /tmp; for this codebase it is REPO_ROOT/logs/battery_<tag>_log (path only,
    # no behaviour change: the folder only receives per-job log files).
    LOG=ROOT/"logs"/f"battery_{PGTAG}_log"
    if not dry: LOG.mkdir(parents=True,exist_ok=True)
    print(f"[battery] variant={V} pgtag={PGTAG} gpus={gpus} tasks={sorted(tasks)}",flush=True)
    # FIXED(preretrain_audit_r2 H7, 2026-10-16): new preflight: an unfinished (or config-less)
    #   checkpoint is refused before any job starts unless --allow-unfinished (every job
    #   would otherwise fail on its own).
    if tasks - {"COMPILE"}:
        from ensembits.featurize import check_checkpoint_finished, UnfinishedCheckpointError
        try:
            check_checkpoint_finished(ROOT/"output"/V, allow_unfinished=a.allow_unfinished)
        except UnfinishedCheckpointError as e:
            print(f"[{'dry-run: would abort' if a.dry_run else 'abort'}] {e} -> pass "
                  f"--allow-unfinished, or run the battery on a frozen copy made by "
                  f"downstream/snapshot_checkpoint.py",flush=True)
            if not a.dry_run: sys.exit(1)

    # A: pretokenize
    # MISATO token caches for P=full (8 frames) and P=1; one job per GPU.
    pj=[]
    for p in PMODES:
        tag=tag_of(V)+("_P1inf" if p else "")
        o=CACHE/f"ensembits_{tag}_tokens_misato.npz"
        c=[PY,str(SRC/"_pretokenize_misato_dihedral.py"),"--variant",V,"--device","cuda:0"]+AU
        if p: c+=["--p-inference",str(p)]
        pj.append((f"PRETOK/P{p}",c,o))
    if "PRETOK" not in tasks: pj=[]
    run_pool(pj,gpus,env0,"pretok",LOG,per_gpu=1,dry=dry)

    # B
    # (outputs of A and B are checked for provenance below, before the B pool skips them)
    jobs=[]
    # RMSF on mdCATH-div: 10 seeds x {P=full (10 frames), P=1}.
    # Paper: 2_results.tex result:rmsf, Table tab:rmsf (mdCATH-div column).
    for p in PMODES:
        pih="_P1inf" if p else ""
        for s in SEEDS:
            # code_fix §7 (2026-10-15): the probes now early-stop on val and report test;
            # their outputs carry the "__test" tag (old untagged files are left alone).
            o=RMDIV/f"{tag_of(V)}{pih}{sd(s)}__test.json"
            c=[PY,str(SRC/"probe_rmsf_eval.py"),"--variant",V,"--seed",str(s),"--device","cuda:0"]+AU
            if p: c+=["--p-inference",str(p)]
            jobs.append((f"RMDIV/P{p}/s{s}",c,o))
    # RMSF on MISATO: sequence + structure splits x 10 seeds x {P=full (8), P=1}.
    # Paper: 2_results.tex result:rmsf, Table tab:rmsf (MISATO seq / struct columns).
    for p in PMODES:
        pih="_P1inf" if p else ""
        for sp,spf in [("sequence",""),("structure","_structure")]:
            for s in SEEDS:
                o=RMMIS/f"{tag_of(V)}{pih}{spf}{sd(s)}__test.json"
                # code_fix §7: no fixed --epochs 40 any more; the probe runs to convergence
                # (early stopping on val, --max-epochs 500 --patience 20 defaults).
                c=[PY,str(SRC/"probe_rmsf_misato.py"),"--mode","variant","--variant",V,"--split",sp,"--seed",str(s),"--device","cuda:0"]+AU
                if p: c+=["--p-inference",str(p)]
                jobs.append((f"RMMIS/P{p}/{sp}/s{s}",c,o))
    # RMSF on BioEmu-CATH (phase 2, 2026-10-16): 10 seeds x {P=full (10 frames), P=1}, same
    # probe as mdCATH-div with --dataset bioemu (labels, splits, cache in
    # output/phase_b/cached_desc/bioemu/; results in output/probe_rmsf_bioemu/).
    RMBIO=ROOT/"output/probe_rmsf_bioemu"
    for p in PMODES:
        pih="_P1inf" if p else ""
        for s in SEEDS:
            o=RMBIO/f"{tag_of(V)}{pih}{sd(s)}__test.json"
            c=[PY,str(SRC/"probe_rmsf_eval.py"),"--dataset","bioemu","--variant",V,"--seed",str(s),"--device","cuda:0"]+AU
            if p: c+=["--p-inference",str(p)]
            jobs.append((f"RMBIO/P{p}/s{s}",c,o))
    # Function tasks: EC (depths 1-3) and GO top-50 per split, binding site +
    # ligand-aware affinity (all three splits in one call), 10 seeds each.
    # Paper: appendix.tex app:downstreams / app:experiments.
    # (Job templates in function_jobs(), shared with the rescue pass below.)
    jobs+=[(f"{t}/P{p}/{sp}/s{s}" if t not in ("BIND","AFFCTRL") else f"{t}/P{p}/s{s}",c,o)
           for (t,_,p,sp,s,o,c) in function_jobs(V,RECIPE,AU)]
    jobs=[j for j in jobs if j[0].split("/")[0] in tasks]   # F8: --tasks
    # FIXED(code_fix P17, 2026-10-16): new (old: none): refuse to skip (= reuse) outputs written by another checkpoint.
    from ensembits.featurize import checkpoint_provenance
    prov=checkpoint_provenance(ROOT/"output"/V)
    bad=stale_outputs([o for _,_,o in pj+jobs if o is not None],prov)
    # FIXED(preretrain_audit_r2 Q3, 2026-10-16): new: an existing RMSF result is only reused
    #   if it carries the current common residue set and the default RMSF label (the probe's
    #   own check never runs for a skipped job).
    from rmsf_common import stale_rmsf_outputs
    # FIXED(preretrain_audit_r2 G11, 2026-10-17): stale RMSF outputs (label, residue set,
    #   caches, unverified caches) are never reused, not even with --force.
    bad_rm=[]
    bad_rm+=stale_rmsf_outputs([o for n,_,o in jobs if n.startswith("RMDIV/")],"mdcath_div")
    bad_rm+=stale_rmsf_outputs([o for n,_,o in jobs if n.startswith("RMMIS/")],"misato")
    bad_rm+=stale_rmsf_outputs([o for n,_,o in jobs if n.startswith("RMBIO/")],"bioemu")
    if bad_rm:
        print(f"[stale] {len(bad_rm)} existing RMSF outputs are stale:",flush=True)
        for o,why in bad_rm[:20]: print(f"  {o.name}: {why}",flush=True)
        if not dry:
            print("[abort] move the stale RMSF outputs aside (--force does not apply)",flush=True)
            sys.exit(1)
    # FIXED(benchmark_audit D3, 2026-10-18): new. Existing EC / GO / BIND outputs not built on
    #   the current MISATO split file (ensembits.paths.MISATO_SPLITS = misato_splits_v2.json;
    #   no record = the old misato_splits.json) are never reused, not even with --force.
    from ensembits.data_versions import stale_split_outputs
    bad_sp=stale_split_outputs([o for n,_,o in jobs if o is not None and n.split("/")[0] in ("EC","GO","BIND","AFFCTRL")])
    if bad_sp:
        print(f"[stale] {len(bad_sp)} existing EC / GO / BIND outputs are not on the current "
              f"MISATO split file:",flush=True)
        for o,why in bad_sp[:20]: print(f"  {o.name}: {why}",flush=True)
        if not dry:
            print("[abort] move them aside (--force does not apply)",flush=True)
            sys.exit(1)
    # FIXED(benchmark_audit D4, 2026-10-18): new. Existing BIND outputs whose binding results were
    #   not built on the current binding-site labels (misato_binding_site_v2.npz; no record = the
    #   old misato_binding_site.npz) are never reused, not even with --force.
    from ensembits.misato_binding_labels import stale_binding_label_outputs
    bad_bl=stale_binding_label_outputs([o for n,_,o in jobs if o is not None and n.split("/")[0]=="BIND"])
    if bad_bl:
        print(f"[stale] {len(bad_bl)} existing BIND outputs are not on the current binding-site "
              f"labels:",flush=True)
        for o,why in bad_bl[:20]: print(f"  {o.name}: {why}",flush=True)
        if not dry:
            print("[abort] move them aside (--force does not apply)",flush=True)
            sys.exit(1)
    # FIXED(benchmark_audit D8, 2026-10-18): new. Existing EC outputs not built on the current
    #   EC labels (data/annotations/misato_ec_v2.json; no "ec_labels" record = the old
    #   misato_ec.json) are never reused, not even with --force.
    from ec_labels import stale_ec_outputs
    bad_ec=stale_ec_outputs([o for n,_,o in jobs if o is not None and n.split("/")[0]=="EC"])
    if bad_ec:
        print(f"[stale] {len(bad_ec)} existing EC outputs are not on the current EC labels:",flush=True)
        for o,why in bad_ec[:20]: print(f"  {o.name}: {why}",flush=True)
        if not dry:
            print("[abort] move them aside (--force does not apply)",flush=True)
            sys.exit(1)
    # FIXED(function_reaudit pre-run fixes, 2026-10-19): new. Existing EC / GO / BIND / AFFCTRL
    #   outputs without the current function protocol (function_common.FUNCTION_PROTOCOL) or on
    #   other affinity labels / MACCS files are never reused, not even with --force.
    from function_common import stale_function_outputs
    bad_fn=stale_function_outputs([o for n,_,o in jobs if o is not None and n.split("/")[0] in ("EC","GO","BIND","AFFCTRL")])
    # FIXED(function CV, 2026-10-19): new. Existing EC / GO CV outputs not built on the current
    #   CV file (function_cv.cv_result_problem: sha256) are never reused, not even with --force.
    if FN_PROTOCOL=="cv":
        from function_cv import stale_cv_outputs
        bad_fn+=stale_cv_outputs([o for n,_,o in jobs if o is not None and n.split("/")[0] in ("EC","GO")])
    if bad_fn:
        print(f"[stale] {len(bad_fn)} existing function outputs are not on the current protocol:",flush=True)
        for o,why in bad_fn[:20]: print(f"  {o.name}: {why}",flush=True)
        if not dry:
            print("[abort] move them aside (--force does not apply)",flush=True)
            sys.exit(1)
    if bad:
        print(f"[provenance] {len(bad)} existing outputs do not name checkpoint {V} "
              f"(best.pt {str(prov.get('best_pt_sha256'))[:12]}):",flush=True)
        for o,why in bad[:20]: print(f"  {o.name}: {why}",flush=True)
        if not (a.force or dry):
            print("[abort] move them aside (fresh variant name per run) or pass --force",flush=True)
            sys.exit(1)
    run_pool(jobs,gpus,env0,"downstream",LOG,per_gpu=2,dry=dry)

    # B': seed rescue of the function jobs.
    # FIXED(code_fix P2, 2026-10-16): old: none for the variant (only the baseline runner
    #   re-ran premature outlier seeds). New: the same pass as
    #   downstream/baselines/run_misato_baselines.py (seed_rescue.run_rescue_pass: same
    #   outlier rule, same RESCUE_RECIPE, same output file; "rescued": true and the original
    #   epochs are written into each re-run JSON). NOTE: the selection uses the stored TEST
    #   metrics of the result JSONs (authors' decision 2026-10-16: keep, symmetric, noted
    #   in the paper).
    if not a.rescue:
        print("[rescue] skipped (default: off; pass --rescue to enable)",flush=True)
    else:
      if dry: print("[rescue] dry run: flagged from the result files that exist now",flush=True)
      run_rescue_pass([j for j in function_jobs(V,RESCUE_RECIPE,AU) if j[0] in tasks],
                    lambda rj: run_pool(rj,gpus,env0,"rescue",LOG,per_gpu=2,dry=dry),dry=dry)

    # C: ProteinGym
    # One pg_score_dihedral.py call over all GPUs (P=1 ESMFold structures).
    # Paper: 2_results.tex sec:downstreams, Mutation Effect Prediction.
    # (--model-dir is relative to cwd = REPO_ROOT, i.e. REPO_ROOT/output/<variant>,
    # not the copy in ckpt/.)
    # FIXED(code_fix P11, 2026-10-16): old: --gpus <the battery's list> with the inherited
    #   CUDA_VISIBLE_DEVICES (wrong devices when the parent restricts visibility). New: the
    #   subprocess sees exactly the battery's GPUs (CUDA_VISIBLE_DEVICES) and gets
    #   --gpus 0..n-1, i.e. all of them.
    c=[PY,str(SRC/"pg_score_dihedral.py"),"--model-dir",f"output/{V}","--out-tag",PGTAG,
       "--gpus",",".join(str(i) for i in range(len(gpus)))]+AU
    if "PG" not in tasks:
        print("[pg] skipped (not in --tasks)",flush=True)
    elif dry:
        print(f"[pg] CUDA_VISIBLE_DEVICES={','.join(gpus)} "+shlex.join(["python"]+c[1:]),flush=True)
    else:
        envpg=dict(env0); envpg["CUDA_VISIBLE_DEVICES"]=",".join(gpus)
        pg_rc=subprocess.run(c,cwd=str(ROOT),env=envpg,stdout=open(LOG/"pg.log","w"),stderr=subprocess.STDOUT).returncode
        print(f"[pg] done rc={pg_rc} ({'ok' if (ROOT/'output/proteingym'/f'per_assay_pg_{PGTAG}.json').exists() else 'NO-OUT'})",flush=True)

    # D: aggregate
    # Rebuild REPO_ROOT/rebuttal/results_bf65.md from all result JSONs.
    # code_fix C4: test protocol for the RMSF columns (also the compiler's default).
    # FIXED(code_fix P16, 2026-10-16): --compile-args passes the reported-model rows
    #   (--variant / --pgtag / --ablation-variants ...) to the compiler.
    # FIXED(preretrain_audit_r2 F8 / B7, 2026-10-16): old = the compile always ran (without
    #   --compile-args it rewrote the results file with the pre-fix rows only). New = it runs
    #   only with the COMPILE task AND --compile-args; otherwise it is skipped with a note.
    cc=[PY,str(SRC/"_compile_rebuttal_results.py"),"--rmsf-protocol=test"]+shlex.split(a.compile_args)
    run_compile = "COMPILE" in tasks and bool(a.compile_args.strip())
    if not run_compile:
        print("[compile] skipped (needs the COMPILE task and --compile-args)",flush=True)
    if dry:
        if run_compile: print("[compile] "+shlex.join(["python"]+cc[1:]),flush=True)
        print("[dry-run] nothing was run",flush=True); return
    if run_compile: subprocess.run(cc,cwd=str(ROOT))
    print(f"MODEL_BATTERY_COMPLETE {V}",flush=True)

if __name__=="__main__": main()
