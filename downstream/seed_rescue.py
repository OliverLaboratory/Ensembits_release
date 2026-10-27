"""Seed rescue rule shared by the Ensembits battery and the baseline runner.

What it does
------------
After the EC / GO / binding probes of a method have run for seeds 0-9, some seeds
end far below the others because early stopping fired before the head had learnt
anything (a "premature" run). This module finds those seeds and builds the re-run:

  1. `read_cells(task, out)` reads the stored metric of one result JSON
     (EC: top-1 hit per depth, GO: top-1 hit, binding: AUROC, affinity: Spearman)
     with the number of epochs the run trained.
  2. `find_rescues(jobs)` groups the values by cell (task, method, P mode, split,
     metric). For every cell with >= 4 seeds and a seed std > 0.1 it flags the seeds
     below median - max(0.10, 3 x 1.4826 x MAD). A flagged seed whose run stopped
     before MAXEP = 40 epochs is re-run as a complete run (RESCUE_RECIPE:
     `--warmup 5 --min-epochs 40 --patience 999 --cosine`, same output file);
     flagged seeds that already ran 40 epochs are left as they are (genuine
     instability).
  3. `mark_rescued(out, original)` adds `"rescued": true` and the original epoch
     counts / values to the re-run's result JSON (files without the key were not
     rescued).

The selection reads the STORED TEST metrics of the result JSONs (the probes do not
write a validation metric in a form this rule could use). The authors decided
(2026-10-16) to keep the rescue, apply it to every method alike, and describe the
probes as trained to convergence; the paper text notes that the rule reads test
metrics.

Paper
-----
Not described as a rule in the paper (infrastructure); the probe recipe is
`appendix.tex` \\label{app:downstreams} ("trained to convergence", 10 seeds).

Origin
------
New file (2026-10-16, code_fix P2). `read_cells`, the outlier rule and the
premature-seed test are moved unchanged from `downstream/baselines/run_misato_baselines.py`
(itself from the research repo's `submission_exp/src/_audit_degeneracy_rerun.py`);
the population std (`statistics.pstdev`) of the flagging rule is kept as it was, so
the decisions are unchanged.

Used by the reported model: yes (`downstream/_run_model_battery.py` rescue pass) and
by the baselines (`downstream/baselines/run_misato_baselines.py`).

Inputs / outputs
----------------
Reads the EC / GO / binding result JSONs named by the callers' job lists; the
rescued JSONs are rewritten by the re-run and then annotated by `mark_rescued`.

How to run
----------
Imported by the two runners; not run directly.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

# Complete run, patience disabled (the re-run recipe of _audit_degeneracy_rerun.py).
RESCUE_RECIPE = ["--warmup", "5", "--min-epochs", "40", "--patience", "999", "--cosine"]
MAXEP = 40   # the probes' default --epochs; a run with fewer epochs stopped early
MIN_SEEDS = 4
STD_MIN = 0.1


def read_cells(task, out):
    """-> list of (metric_key, value, epochs).  (unchanged from _audit_degeneracy_rerun.py)

    The values are the stored TEST metrics of the result JSON.
    """
    out = Path(out)
    if not out.exists(): return []
    d = json.load(open(out)); r = []
    if task == "EC":
        for dp, c in d.get("by_depth", {}).items():
            if isinstance(c, dict) and "top1_hit" in c: r.append((f"d{dp}", c["top1_hit"], c.get("epochs_run", 0)))
    elif task == "GO":
        # FIXED(benchmark_audit D7, 2026-10-18): a GO result not built on the current GO
        #   labels (go_labels.go_result_problems) gives no cell (old: always read).
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from go_labels import go_result_problems
        if "top1_hit" in d and not go_result_problems(d):
            r.append(("go", d["top1_hit"], d.get("epochs_run", 0)))
    # FIXED(function_reaudit pre-run fix 4, 2026-10-19): the D5 affinity control jobs
    #   ("AFFCTRL": protein only, ligand only) get the same rescue rule as the BIND jobs.
    elif task in ("BIND", "AFFCTRL"):
        for k, c in d.get("results", {}).items():
            if isinstance(c, dict):
                m = c.get("auroc") if "binding" in k else c.get("spearman")
                if m is not None: r.append((k, m, c.get("epochs_run", 0)))
    return r


def outlier_decision(seeds):
    """The outlier rule on one cell.

    seeds: {seed: (value, epochs)}. Returns (fix, genuine, info): the premature
    low-outlier seeds to re-run, the low outliers that already ran MAXEP epochs, and
    (std, median, min) for the log; ([], [], None) when the cell is not checked.
    """
    xs = [t[0] for t in seeds.values() if t[0] is not None]
    if len(xs) < MIN_SEEDS: return [], [], None
    sd = st.pstdev(xs)
    if sd <= STD_MIN: return [], [], None
    med = st.median(xs); mad = st.median([abs(x-med) for x in xs]) or 1e-6
    thr = med-max(0.10, 3*1.4826*mad)
    deg = [s for s, t in seeds.items() if t[0] is not None and t[0] < thr]
    fix = [s for s in deg if seeds[s][1] < MAXEP]; gen = [s for s in deg if seeds[s][1] >= MAXEP]
    return fix, gen, (sd, med, min(xs))


def find_rescues(J, verbose=True):
    """Flag premature low-outlier seeds.

    J: jobs (task, key, pmode, split, seed, out_path, rescue_cmd), task in EC / GO / BIND,
    the commands built with RESCUE_RECIPE. Returns {str(out_path): {"cmd": rescue_cmd,
    "original": {metric_key: {"epochs_run": ep, "value": v}}}}.
    NOTE: the selection uses the stored TEST metrics (see the module docstring).
    """
    # group by cell = (task, key, pmode, split-or-BINDsplit, metric); track {seed:(val,ep,outpath,cmd)}
    cells = {}
    for task, key, p, sp, s, o, c in J:
        for mk, v, ep in read_cells(task, o):
            # BIND metric key already encodes split; EC/GO use sp+depth
            cid = (task, key, p, sp, mk)
            cells.setdefault(cid, {})[s] = (v, ep, o, c)
    flagged = {}; lines = []
    for cid, seeds in cells.items():
        fix, gen, info = outlier_decision({s: (t[0], t[1]) for s, t in seeds.items()})
        if info is None: continue
        sd, med, mn = info
        if fix or gen: lines.append(f"  std={sd:.3f} {str(cid[1])[:26]} {cid[0]}/{cid[3]}/{cid[4]} med={med:.3f} min={mn:.3f} fix={fix} genuine={gen}")
        for s in fix:
            v, ep, o, c = seeds[s]
            ent = flagged.setdefault(str(o), {"cmd": c, "original": {}})
            ent["original"][cid[4]] = {"epochs_run": ep, "value": v}
    if verbose:
        print(f"=== degenerate cells (std>{STD_MIN} with clear outlier): {len(lines)} ===")
        for l in sorted(lines): print(l)
        print(f"\n=== fixable seed-files (premature, ep<{MAXEP}) to complete-run: {len(flagged)} ===")
        for pth in sorted(flagged): print("  ", Path(pth).name)
    return flagged


def mark_rescued(out_path, original, started_at=None):
    """Annotate a re-run result JSON: "rescued": true + the original epochs / values.

    Returns False (and leaves the file alone) when the file is missing or was not
    rewritten after `started_at` (a failed re-run).
    """
    p = Path(out_path)
    if not p.exists():
        return False
    if started_at is not None and p.stat().st_mtime < started_at:
        return False
    d = json.loads(p.read_text())
    d["rescued"] = True
    d["rescue_recipe"] = " ".join(RESCUE_RECIPE)
    d["rescue_original"] = original          # {metric: {"epochs_run": n, "value": v}}
    d["rescue_selection"] = "stored test metric (seed_rescue.outlier_decision)"
    p.write_text(json.dumps(d, indent=2))
    return True


def run_rescue_pass(J, run_jobs, dry=False):
    """Find the rescues in J, run them with run_jobs(list of (name, cmd, None)), annotate.

    run_jobs is the caller's GPU pool (same job tuple format). Returns the flagged dict.
    """
    import time
    flagged = find_rescues(J)
    rjobs = [("RESCUE/"+Path(pth).stem, ent["cmd"], None) for pth, ent in sorted(flagged.items())]
    t0 = time.time()
    run_jobs(rjobs)
    if not dry:
        for pth, ent in sorted(flagged.items()):
            ok = mark_rescued(pth, ent["original"], started_at=t0)
            if not ok:
                print(f"  [rescue] WARNING: {Path(pth).name} was not rewritten by its re-run "
                      f"(see the log); not marked as rescued", flush=True)
    return flagged
