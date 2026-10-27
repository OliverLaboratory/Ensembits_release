"""CryptoBench: one runner for step 7 (structures, tokens, both probes, statistics, completeness).

What it does
------------
Runs the whole camera-ready CryptoBench experiment (camera_ready/cryptobench_audit/
TRACKER.md step 7) when launched, with the protocol fixed in DECISIONS.md (authors' answers
of 2026-10-20 and 2026-10-22) and PORT.md. Every job is a subprocess of a script of this
folder; the runner can be relaunched after an interruption or a change.

  1. structures + labels  `cryptobench_build_apo.py` (gemmi; `--gemmi-python`), set apo
  2. non-cryptic mask      `cryptobench_common.py noncryptic-mask` (sensitivity analysis)
  3. tokens  Ensembits: all 8 retrained runs output/ensembits_bm_{e0,lam0,lam0p05,lam0p25,
             lam0p5,lam1,nosftd,vq2048}_rv at P = 1 (`cryptobench_tokenize.py`, level-0 token =
             primary readout); z / q / q0 of the two lambda candidates (`cryptobench_featurize.py`);
             baselines AminoAseed, ESM3-struct, ProToken (GPU), 3Di (CPU)
             (`cryptobench_tokenize_baselines.py`, whole chain, label_seq residue index);
             ESM-2 3B embeddings (`cryptobench_esm2.py`, descriptive row, `--esm2-row on`)
  4. probes  MLP (primary, `cryptobench_mlp_probe.py`) and context conv1d (secondary,
             `cryptobench_probe.py`), 5 seeds (exactly 0-4), epoch chosen on validation
             (train-3) only; rows aa, 3Di, AminoAseed, ESM3-struct, ProToken, the 8 Ensembits
             runs (token readout), z / q / q0 of the two lambda candidates, ESM-2 3B (MLP only);
             per-seed prediction sidecars
  5. stats   `downstream/protein_level_stats.py` on the pre-registered bundle
             `cryptobench_families_v1.json` (sha256 FAMILY_SHA256, instantiated with this run's
             folders and the declared seeds 0-4); ESM-2 and the readout rows are in no family
  6. completeness: counts, return codes, seed sets, level-0 cross-checks, finite flags; prints
             no metric value (CryptoBench numbers stay sealed until the lambda choice, CB8)

Skip logic (FIXED REAUDIT CB5 F1 / CB4 F2 / CB2 F10, 2026-10-22): every job decides "done"
through its script's own `is_done(argv)` = the output exists and records the current input
fingerprint (structure manifest sha256, checkpoint best.pt sha256, input token / dense file
sha256, code sha256, all result-affecting arguments); otherwise the script moves the old
output to `_pre_<date>/` and redoes it. A job is also redone when any job it depends on is
redone. The stats job is done only if its results name the current family instance and every
sidecar sha256 it used is still the one on disk. `--redo-all` passes `--redo` to every job.

Options for the authors' TENTATIVE answers of 2026-10-22 (default = the tentative choice;
each goes only to the jobs it affects, so a change reruns only those rows):
  --modified-residues {parent, drop}   structure stage (changes the manifest: every row reruns)
  --esm2-row {on, off}                 the descriptive ESM-2 3B MLP row
  --conv-standardize {train, none}     conv probe input scale (conv rows)
  --selection {val_pooled_auroc, val_per_entry_auroc}   epoch selection (all probe rows)
  --aa-order {collaborator, alphabetical}   aa rows
  --esm2-layer N (TENTATIVE default 33, the coordinator's choice: upstream pLM-NN's layer)
                                       the ESM-2 embeddings and their MLP row

Fix round 2 (2026-10-22, FIXLIST2_20261022.md): the structure job is done only if the manifest
records the current build script, release files (dataset.json, folds.json,
noncryptic-pockets.json) and CCD sha256; the mask job also needs the current mask code;
Ensembits token jobs run with --rerun-on-change (a retrained checkpoint reruns, FX3 F1); a plan
with GPU jobs and no usable GPU is refused at once (`--gpus ""` used to hang, FX3 F2); the stats
results are moved to `_pre_<date>/` before the stats job reruns (FX3 F5); completeness counts
only the outputs of the planned rows (FX3 F7).

GPU jobs run `--jobs-per-gpu` at a time per GPU, each with CUDA_VISIBLE_DEVICES set to its GPU
and XLA_PYTHON_CLIENT_PREALLOCATE=false (ProToken / JAX). A GPU with more than
`--max-used-mib` MiB in use at start is refused, and the launch is refused when nvidia-smi
cannot be queried (fail closed). `--dry-run` prints the GPU state and the job list only.

Paper
-----
Not in the paper yet (CryptoBench text after the lambda choice, DECISIONS.md CB9).

Origin
------
New file (2026-10-21, step 5 of camera_ready/cryptobench_audit/TRACKER.md); reworked
2026-10-22 after the port re-audit (camera_ready/cryptobench_audit/FIXLIST_20261022.md).

Used by the reported model: evaluation only.

Inputs / outputs
----------------
Relative to REPO_ROOT: data/cryptobench/ (release), output/ensembits_bm_*_rv/ (checkpoints),
camera_ready/cryptobench_audit/leak_D/test_clusters_30.json (clusters), the HF cache of
facebook/esm2_t36_3B_UR50D; writes data/cached_descriptors/cryptobench_runs/<run-tag>/
(structures, tokens) and output/cryptobench/runs/<run-tag>/ ({mlp,conv}/ results + sidecars,
stats/, logs/<launch time>/, completeness.json).

How to run
----------
  $PY downstream/cryptobench/run_cryptobench_rv.py --run-tag rv_20261021 \\
      --gemmi-python <python with gemmi> --gpus 0,1,2 [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
for _p in (CODE_ROOT, CODE_ROOT / "downstream", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402
import _cb_paths as CBP  # noqa: E402
import cryptobench_common as CC  # noqa: E402

DEFAULT_RUN_TAG = "rv_20261021"
ENS_RUNS = ("e0", "lam0", "lam0p05", "lam0p25", "lam0p5", "lam1", "nosftd", "vq2048")
LAMBDA_CANDIDATES = ("lam0p25", "lam1")
DENSE_READOUTS = ("z", "q", "q0")
# baseline rows: (row name, tokenizer or None for aa, display name)
BASELINES = (("aa", None, "aa"), ("mini3di_wc", "mini3di", "3Di"),
             ("aminoaseed_wc", "aminoaseed", "AminoAseed"),
             ("esm3struct_wc", "esm3struct", "ESM3-struct"),
             ("protoken_wc", "protoken", "ProToken"))
PROBES = {"mlp": "cryptobench_mlp_probe.py", "conv": "cryptobench_probe.py"}
FAMILY_TEMPLATE = HERE / "cryptobench_families_v1.json"
# FIXED(REAUDIT CB4 F3, 2026-10-22): the pre-registered bundle's sha256, hard-coded; any other
#   file is refused (the template generator below must also still produce it).
FAMILY_SHA256 = "1d84e4844bef163dc7216a3c38a0c8717c00c061769fd27194d95b0362af9683"
SEEDS = 5
ESM2_ROW = "esm2_3b"


def ens_dir(r: str) -> str:
    return f"output/ensembits_bm_{r}_rv"


def ens_tag(r: str) -> str:
    return f"ens_{r}_rv"


class Job:
    def __init__(self, name, argv, deps=(), gpu=False, done=None, python=None, redo_flag=False):
        self.name, self.argv, self.deps, self.gpu = name, list(argv), list(deps), gpu
        self.done = done or (lambda: False)
        self.python = python or sys.executable
        self.redo_flag = redo_flag          # the script accepts --redo
        self.pre = None                     # called just before the job starts
        self.rc = None

    def cmd(self, redo=False):
        return [self.python, *self.argv, *(["--redo"] if redo and self.redo_flag else [])]


def script_done(script: str, argv: list):
    """done() of a job = the script's own is_done(argv) (input fingerprint)."""
    def done():
        mod = importlib.import_module(Path(script).stem)
        return bool(mod.is_done(list(argv)))
    return done


def plan(args, paths) -> list:
    """The job list in dependency order (names unique; deps by name)."""
    D = str(HERE)
    loc = ["--run-tag", args.run_tag]
    for flag, val in (("--structs-root", args.structs_root), ("--manifest-dir", args.manifest_dir),
                      ("--tokens-dir", args.tokens_dir), ("--out-dir", args.out_dir),
                      ("--cryptobench-dir", args.cryptobench_dir)):
        if val:
            loc += [flag, str(val)]
    man = CBP.manifest_path(paths, "apo")
    jobs = []

    def built():
        if not man.exists():
            return False
        meta = json.loads(man.read_text()).get("meta", {})
        return (meta.get("build_version") == CC.STRUCT_BUILD_VERSION
                and meta.get("modified_residues") == args.modified_residues
                and meta.get("build_code_sha256") == CC.file_sha256(HERE / "cryptobench_build_apo.py")
                and meta.get("release_sha256") == release_sha256(paths)
                and meta.get("ccd_sha256") == ccd_sha256(args))

    jobs.append(Job("build", [f"{D}/cryptobench_build_apo.py", "--workers", str(args.workers),
                              "--modified-residues", args.modified_residues, *loc],
                    done=built, python=args.gemmi_python))
    mask = paths.manifest_dir / "apo_noncryptic_mask.json"

    def mask_ok():
        if not (mask.exists() and man.exists()):
            return False
        m = json.loads(mask.read_text())
        return (m.get("manifest_sha256") == CC.file_sha256(man)
                and m.get("mask_code_sha256") == CC.mask_code_sha256())
    jobs.append(Job("mask", [f"{D}/cryptobench_common.py", "noncryptic-mask", *loc],
                    ["build"], done=mask_ok))

    def add(name, script, a, deps, gpu):
        argv = [*a, *loc]
        jobs.append(Job(name, [f"{D}/{script}", *argv], deps, gpu=gpu,
                        done=script_done(script, argv), redo_flag=True))

    for r in ENS_RUNS:
        add(f"tok_{ens_tag(r)}", "cryptobench_tokenize.py",
            ["--model-dir", ens_dir(r), "--tag", ens_tag(r), "--device", "cuda:0",
             "--rerun-on-change"], ["build"], True)
    for r in LAMBDA_CANDIDATES:
        add(f"dense_{ens_tag(r)}", "cryptobench_featurize.py",
            ["--model-dir", ens_dir(r), "--tag", ens_tag(r), "--tokens-tag", ens_tag(r),
             "--device", "cuda:0"], ["build", f"tok_{ens_tag(r)}"], True)
    for row, tok, _ in BASELINES:
        if tok is not None:
            add(f"tok_{row}", "cryptobench_tokenize_baselines.py",
                ["--tokenizer", tok, "--tag", f"{row}_apo"], ["build"], tok != "mini3di")
    if args.esm2_row == "on":
        add(f"tok_{ESM2_ROW}", "cryptobench_esm2.py",
            ["--device", "cuda:0", "--layer", str(args.esm2_layer)], ["build"], True)

    # probe rows: (row, probe args, input job, checkpoint folder, probes)
    rows = [("aa", ["--features", "aa", "--aa-order", args.aa_order], None, None, PROBES)]
    rows += [(row, ["--features", "tokens", "--tokens", f"{row}_apo"], f"tok_{row}", None, PROBES)
             for row, tok, _ in BASELINES if tok]
    rows += [(ens_tag(r), ["--features", "tokens", "--tokens", ens_tag(r)], f"tok_{ens_tag(r)}",
              ens_dir(r), PROBES) for r in ENS_RUNS]
    rows += [(ens_tag(r), ["--features", "dense", "--dense", ens_tag(r), "--readout", ro],
              f"dense_{ens_tag(r)}", ens_dir(r), PROBES)
             for r in LAMBDA_CANDIDATES for ro in DENSE_READOUTS]
    if args.esm2_row == "on":
        rows += [(ESM2_ROW, ["--features", "dense", "--dense", ESM2_ROW, "--readout", "esm2",
                             "--expect-layer", str(args.esm2_layer)],
                  f"tok_{ESM2_ROW}", None, {"mlp": PROBES["mlp"]})]
    probe_jobs = []
    for row, pargs, dep, ck, probes in rows:
        for pk, script in probes.items():
            ro = pargs[pargs.index("--readout") + 1] if "--readout" in pargs else (
                "onehot" if pargs[1] == "aa" else "token")
            extra = ["--seeds", str(args.seeds), "--device", "cuda:0",
                     "--selection", args.selection]
            if pk == "conv":
                extra += ["--standardize", args.conv_standardize]
            if ck:
                extra += ["--expect-checkpoint", ck]
            if args.smoke_epochs:      # smoke only: short training, same code path
                n = str(args.smoke_epochs)
                extra += (["--epochs", n, "--patience", n] if pk == "mlp" else
                          ["--max-epochs", n, "--min-epochs", "1", "--patience", n,
                           "--warmup", "1"])
            jn = f"{pk}_{row}__{ro}"
            probe_jobs.append(jn)
            add(jn, script, ["--row", row, *pargs, *extra], ["build"] + ([dep] if dep else []),
                True)
    fam = paths.out_dir / "stats" / "family_instance.json"
    res = paths.out_dir / "stats" / "cryptobench_families_v1_results.json"
    family_probe_jobs = [j for j in probe_jobs if not j.startswith(f"mlp_{ESM2_ROW}")]
    jobs.append(Job("family", [f"{D}/run_cryptobench_rv.py", "--write-family", *loc_family(args)],
                    family_probe_jobs + ["mask"],
                    done=lambda: fam.exists() and fam.read_bytes() == family_bytes(args, paths)))
    stats_job = Job("stats", [str(CODE_ROOT / "downstream" / "protein_level_stats.py"),
                              "--family", str(fam), "--out", str(res)], ["family"],
                    done=lambda: stats_done(res, args, paths))
    # FIXED(FIXAUDIT FX3 F5, 2026-10-22): keep the previous results (moved aside, not overwritten)
    stats_job.pre = lambda: CC.move_aside(res) if res.exists() else []
    jobs.append(stats_job)
    return jobs


def release_sha256(paths) -> dict:
    """sha256 of the release files the structure stage reads (FIXLIST2 item 4)."""
    import cryptobench_build_apo as BA
    d = paths.cryptobench_dir
    return {f: CC.file_sha256(d / f) for f in BA.RELEASE_FILES if (d / f).exists()}


def ccd_sha256(args):
    """sha256 of the CCD used for the modified-residue parents (None in drop mode)."""
    if args.modified_residues != "parent":
        return None
    import cryptobench_build_apo as BA
    from ensembits.featurize import file_sha256 as cached_sha
    p = BA.default_ccd()
    return cached_sha(p) if p.exists() else None


def loc_family(args) -> list:
    out = ["--run-tag", args.run_tag, "--seeds", str(args.seeds)]
    for flag, val in (("--structs-root", args.structs_root), ("--manifest-dir", args.manifest_dir),
                      ("--tokens-dir", args.tokens_dir), ("--out-dir", args.out_dir)):
        if val:
            out += [flag, str(val)]
    if args.stats_B:
        out += ["--stats-B", str(args.stats_B)]
    if args.smoke_epochs:
        out += ["--smoke-epochs", str(args.smoke_epochs)]
    return out


# ---------------------------------------------------------------- family
def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(p).resolve())


def instantiate_family(template: dict, run_dir: Path, struct_dir: Path, B=None,
                       seeds=None) -> dict:
    """Replace {RUN} (output folder of the run) and {STRUCT} (structure / manifest folder) in
    the pre-specified bundle; paths stay relative to REPO_ROOT where possible. `seeds` (list)
    is added as the declared seed set (enforced by protein_level_stats; FIXED REAUDIT CB4 F1)."""
    txt = json.dumps(template)
    txt = txt.replace("{RUN}", _rel(run_dir)).replace("{STRUCT}", _rel(struct_dir))
    inst = json.loads(txt)
    if B:
        inst["B"] = int(B)
    if seeds is not None:
        inst["seeds"] = [int(s) for s in seeds]
    return inst


def make_family_template() -> dict:
    """The pre-specified statistics bundle (written once to cryptobench_families_v1.json,
    BEFORE any run; tests check that the file equals this function's output).

    Primary family: MLP probe, per-entry AUROC, {lam0p25_rv, lam1_rv} x {aa, 3Di, AminoAseed,
    ESM3-struct, ProToken}, Holm over the 10 tests. Secondary families (each Holm-adjusted on
    its own): conv probe per-entry AUROC; pooled AUROC and pooled AP for both probes;
    sensitivity sets (entries with >= 5 positives; non-cryptic-pocket residues masked) for
    both probes; ablation (lambda candidates vs nosftd_rv and vs lam0_rv) for both probes."""
    def files(probe, row, ro):
        return [f"{{RUN}}/{probe}/{row}__{ro}__seed*.json"]

    def base_comps(probe, metric, subset=None, tag=""):
        out = []
        for c in LAMBDA_CANDIDATES:
            for row, tok, disp in BASELINES:
                ro = "onehot" if tok is None else "token"
                d = {"id": f"{probe}_{metric}{tag}_{ens_tag(c)}_vs_{row}", "task": "cryptobench",
                     "metric": metric, "a": files(probe, ens_tag(c), "token"),
                     "b": files(probe, row, ro), "label": f"Ensembits {c} vs {disp}"}
                if subset:
                    d["subset"] = subset
                out.append(d)
        return out

    mask = {"mask": {"path": "{STRUCT}/apo_noncryptic_mask.json"}}
    fams = [{"family": "primary_mlp_per_entry_auroc", "role": "primary",
             "comparisons": base_comps("mlp", "auroc_per_protein")}]
    fams.append({"family": "secondary_conv_per_entry_auroc", "role": "secondary",
                 "comparisons": base_comps("conv", "auroc_per_protein")})
    for probe in ("mlp", "conv"):
        for metric in ("auroc_pooled", "ap_pooled"):
            fams.append({"family": f"secondary_{probe}_{metric}", "role": "secondary",
                         "comparisons": base_comps(probe, metric)})
    for probe in ("mlp", "conv"):
        fams.append({"family": f"sensitivity_{probe}_min5pos_per_entry_auroc",
                     "role": "sensitivity (pre-declared)",
                     "comparisons": base_comps(probe, "auroc_per_protein",
                                               {"min_positives": 5}, "_min5pos")})
        fams.append({"family": f"sensitivity_{probe}_noncryptic_masked_per_entry_auroc",
                     "role": "sensitivity (pre-declared)",
                     "comparisons": base_comps(probe, "auroc_per_protein", mask, "_ncmask")})
    for probe in ("mlp", "conv"):
        comps = []
        for c in LAMBDA_CANDIDATES:
            for abl in ("nosftd", "lam0"):
                comps.append({"id": f"{probe}_ablation_{ens_tag(c)}_vs_{ens_tag(abl)}",
                              "task": "cryptobench", "metric": "auroc_per_protein",
                              "a": files(probe, ens_tag(c), "token"),
                              "b": files(probe, ens_tag(abl), "token"),
                              "label": f"Ensembits {c} vs {abl} (SFTD ablation)"})
        fams.append({"family": f"ablation_{probe}_per_entry_auroc",
                     "role": "secondary (ablation)", "comparisons": comps})
    return {"bundle": "cryptobench_families_v1",
            "note": ("pre-specified on 2026-10-21 before any CryptoBench run of the _rv "
                     "checkpoints (camera_ready/cryptobench_audit/PORT.md); entry = apo id; "
                     "primary metric per-entry AUROC (auroc_per_protein on entry items); no "
                     "leakage stratum (authors, CB5); z / q / q0 readouts and the other runs "
                     "are descriptive (no test). {RUN} / {STRUCT} are filled by "
                     "run_cryptobench_rv.py."),
            "B": 10000, "seed": 20261019, "split": "test",
            "clusters": {"path": "camera_ready/cryptobench_audit/leak_D/test_clusters_30.json",
                         "sha256": "9d1f1f84871b670a60539e49f2ba5e54c18f0df48a7ca047c427e9402a2144cc"},
            "families": fams}


def template_problem() -> str | None:
    """None if the template file has the pre-registered sha256 and equals the generator."""
    if not FAMILY_TEMPLATE.exists():
        return f"{FAMILY_TEMPLATE.name} missing"
    if CC.file_sha256(FAMILY_TEMPLATE) != FAMILY_SHA256:
        return (f"{FAMILY_TEMPLATE.name} sha256 is not the pre-registered {FAMILY_SHA256[:12]}...; "
                f"the family must not change")
    if json.loads(FAMILY_TEMPLATE.read_text()) != make_family_template():
        return "make_family_template() no longer produces the pre-registered bundle"
    return None


def family_bytes(args, paths) -> bytes:
    """The exact bytes of stats/family_instance.json for this run."""
    tpl = json.loads(FAMILY_TEMPLATE.read_text())
    inst = instantiate_family(tpl, paths.out_dir, paths.manifest_dir, args.stats_B,
                              seeds=list(range(args.seeds)))
    inst["template"] = {"path": _rel(FAMILY_TEMPLATE), "sha256": FAMILY_SHA256}
    return json.dumps(inst, indent=1).encode()


def seed_set_problems(inst: dict) -> list:
    """Methods of the instance whose seed files are not exactly the declared seeds."""
    want = sorted(inst.get("seeds") or [])
    probs = []
    for fam in inst["families"]:
        for c in fam["comparisons"]:
            for side in ("a", "b"):
                for pat in c[side]:
                    p = Path(pat) if Path(pat).is_absolute() else ROOT / pat
                    name = p.name.split("__seed")[0]
                    got = sorted(CC.seed_files(p.parent, name))
                    if got != want:
                        probs.append(f"{c['id']}/{side}: seeds {got}")
    return sorted(set(probs))


def write_family(args, paths) -> int:
    why = template_problem()
    if why:
        print(f"[family] {why}", file=sys.stderr)
        return 2
    data = family_bytes(args, paths)
    probs = seed_set_problems(json.loads(data))
    if probs:
        print("[family] seed sets differ from the declared 0..%d:\n  %s"
              % (args.seeds - 1, "\n  ".join(probs[:20])), file=sys.stderr)
        return 2
    out = paths.out_dir / "stats" / "family_instance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_bytes() != data:
        CC.move_aside(out)
    out.write_bytes(data)
    inst = json.loads(data)
    print(f"[family] wrote {out} ({sum(len(f['comparisons']) for f in inst['families'])} "
          f"comparisons in {len(inst['families'])} families)")
    return 0


def stats_done(res: Path, args, paths) -> bool:
    """The results exist, were computed on the current family instance, and every sidecar
    they used still has the recorded sha256 (FIXED REAUDIT CB4 F2 / CB5 F1). Reads hashes
    only; no value is printed."""
    if not res.exists():
        return False
    try:
        r = json.loads(res.read_text())
    except ValueError:
        return False
    if (r.get("family_spec") or {}).get("sha256") != hashlib.sha256(family_bytes(args, paths)).hexdigest():
        return False
    import protein_preds as PP
    for f in r.get("families", []):
        for c in f["comparisons"]:
            for rec in c.get("sidecars_a", []) + c.get("sidecars_b", []):
                p = Path(rec["result"]); p = p if p.is_absolute() else ROOT / p
                if not p.exists():
                    return False
                cur = (json.loads(p.read_text()).get(PP.PREDS_KEY) or {}).get("sha256")
                if cur != rec["sha256"] or PP.preds_problem(json.loads(p.read_text()).get(PP.PREDS_KEY), p):
                    return False
    return True


# ---------------------------------------------------------------- scheduler
def gpu_used_mib():
    """{gpu index: MiB in use}, or None when nvidia-smi cannot be queried."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=True, timeout=60).stdout
        return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_refusals(gpu_list, used, max_mib) -> list:
    """GPUs to refuse (FIXED REAUDIT CB5 F5: fail closed when the query fails)."""
    if not gpu_list:
        return []
    if used is None:
        return list(gpu_list)
    return [g for g in gpu_list if g not in used or used[g] > max_mib]


def initial_states(jobs, redo_all: bool) -> dict:
    """'skip' for a done job whose dependencies are all skipped, else 'todo' (a job is redone
    when anything it depends on is redone)."""
    state = {}
    for j in jobs:
        ok = (not redo_all) and all(state.get(d) == "skip" for d in j.deps) and j.done()
        state[j.name] = "skip" if ok else "todo"
    return state


def run_jobs(jobs, args, log_dir: Path) -> dict:
    by = {j.name: j for j in jobs}
    for j in jobs:
        for d in j.deps:
            if d not in by:
                raise SystemExit(f"job {j.name}: unknown dependency {d}")
    state = initial_states(jobs, args.redo_all)
    for j in jobs:
        if state[j.name] == "skip":
            j.rc = 0
    slots = {g: [] for g in args.gpu_list}
    cpu_running = []
    running = {}
    log_dir.mkdir(parents=True, exist_ok=True)
    env0 = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env0.setdefault("OMP_NUM_THREADS", "4")
    # FIXED(REAUDIT CB5 F6, 2026-10-22): JAX (ProToken) must not reserve 75 % of a shared GPU.
    env0["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    while True:
        for name, (p, slot, fh) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            by[name].rc = rc
            state[name] = "ok" if rc == 0 else "failed"
            print(f"{dt.datetime.now():%H:%M:%S} [{state[name]}] {name} (rc {rc})", flush=True)
            (cpu_running if slot is None else slots[slot]).remove(name)
            del running[name]
        for j in jobs:
            if state[j.name] == "todo" and any(state[d] in ("failed", "blocked") for d in j.deps):
                state[j.name] = "blocked"
        todo = [j for j in jobs if state[j.name] == "todo"
                and all(state[d] in ("ok", "skip") for d in j.deps)]
        for j in todo:
            if j.gpu and not args.gpu_list:          # defensive: never wait forever
                state[j.name] = "blocked"
                continue
            if j.gpu:
                free = [g for g in args.gpu_list if len(slots[g]) < args.jobs_per_gpu]
                if not free:
                    continue
                slot = min(free, key=lambda g: len(slots[g]))
                env = dict(env0, CUDA_VISIBLE_DEVICES=str(slot))
            else:
                if len(cpu_running) >= args.cpu_jobs:
                    continue
                slot, env = None, dict(env0, CUDA_VISIBLE_DEVICES="")
            cmd = j.cmd(redo=args.redo_all)
            if j.pre is not None:
                j.pre()
            fh = open(log_dir / f"{j.name}.log", "w")
            fh.write(" ".join(cmd) + "\n"); fh.flush()
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 cwd=str(CODE_ROOT))
            running[j.name] = (p, slot, fh)
            state[j.name] = "running"
            (cpu_running if slot is None else slots[slot]).append(j.name)
            print(f"{dt.datetime.now():%H:%M:%S} [start] {j.name}"
                  f"{'' if slot is None else f' (GPU {slot})'}", flush=True)
        if not running and not any(s == "todo" for s in state.values()):
            break
        time.sleep(2)
    return state


# ---------------------------------------------------------------- completeness
def planned_outputs(jobs) -> dict:
    """{probe: set of output names} of the probe jobs in the plan."""
    out = {pk: set() for pk in PROBES}
    for j in jobs:
        pk = j.name.split("_", 1)[0]
        if pk in out:
            out[pk].add(j.name.split("_", 1)[1])
    return out


def completeness(args, paths, state: dict, planned: dict | None = None) -> dict:
    """Counts and finite-value / consistency checks; the returned dict holds no metric value.
    Only outputs of rows in the plan are counted (FIXAUDIT FX3 F7); others are listed."""
    import numpy as np
    import protein_preds as PP
    rep = {"jobs": {s: sum(v == s for v in state.values()) for s in set(state.values())},
           "failed_jobs": sorted(k for k, v in state.items() if v in ("failed", "blocked")),
           "options": {k: getattr(args, k) for k in ("modified_residues", "esm2_row", "esm2_layer",
                                                     "conv_standardize", "selection", "aa_order")},
           "probes": {}}
    for pk in PROBES:
        pdir = paths.out_dir / pk
        aggs = sorted(p for p in pdir.glob("*.json") if "__seed" not in p.name) if pdir.exists() else []
        seeds = sorted(pdir.glob("*__seed*.json")) if pdir.exists() else []
        if planned is not None:
            names = planned.get(pk, set())
            extra = sorted(a.stem for a in aggs if a.stem not in names)
            aggs = [a for a in aggs if a.stem in names]
            seeds = [f for f in seeds if f.name.split("__seed")[0] in names]
        else:
            extra = []
        n_ok = n_finite = 0
        n_test_entries = set()
        bad_seed_sets = [a.stem for a in aggs
                         if sorted(CC.seed_files(pdir, a.stem)) != list(range(args.seeds))]
        for f in seeds:
            js = json.loads(f.read_text())
            if PP.preds_problem(js.get(PP.PREDS_KEY), f) is None:
                n_ok += 1
            m = js.get("metrics", {})
            vals = [m.get(s, {}).get(k) for s in ("val", "test")
                    for k in ("auroc_per_entry", "ap_per_entry", "auroc_pooled", "ap_pooled")]
            if all(v is not None and np.isfinite(v) for v in vals):
                n_finite += 1
            n_test_entries.add(m.get("test", {}).get("n_entries"))
        rep["probes"][pk] = {"aggregates": len(aggs), "seed_results": len(seeds),
                             "seed_results_with_verified_sidecar": n_ok,
                             "seed_results_all_metrics_finite": n_finite,
                             "methods_with_wrong_seed_set": bad_seed_sets,
                             "planned_methods": (len(planned.get(pk, ())) if planned is not None
                                                 else None),
                             "outputs_not_in_plan": extra,
                             "distinct_test_entry_counts": sorted(x for x in n_test_entries
                                                                  if x is not None)}
    dense = {}
    for p in sorted(paths.tokens_dir.glob("cryptobench_dense_*.npz.provenance.json")):
        d = json.loads(p.read_text())
        dense[p.name.split(".npz")[0]] = {k: d.get(k) for k in ("n_failed", "n_token_mismatch",
                                                                  "n_tokenfile_mismatch")}
    rep["dense_cross_checks"] = dense
    rep["dense_cross_checks_all_zero"] = all(not v.get("n_failed") and not v.get("n_token_mismatch")
                                             and not v.get("n_tokenfile_mismatch")
                                             for v in dense.values())
    res = paths.out_dir / "stats" / "cryptobench_families_v1_results.json"
    if res.exists():
        r = json.loads(res.read_text())
        comps = [c for f in r.get("families", []) for c in f["comparisons"]]
        keys = ("delta", "ci_low", "ci_high", "p", "p_holm")
        rep["stats"] = {"families": len(r.get("families", [])), "comparisons": len(comps),
                        "comparisons_all_finite": sum(all(np.isfinite(c.get(k, np.nan)) for k in keys)
                                                      for c in comps),
                        "current": stats_done(res, args, paths)}
    else:
        rep["stats"] = None
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpus", default="0", help="comma-separated GPU indices for GPU jobs")
    ap.add_argument("--jobs-per-gpu", type=int, default=1)
    ap.add_argument("--cpu-jobs", type=int, default=2)
    ap.add_argument("--max-used-mib", type=int, default=2048,
                    help="refuse a GPU with more memory in use at start (another user's GPU)")
    ap.add_argument("--gemmi-python", default=os.environ.get("CB_GEMMI_PYTHON", sys.executable),
                    help="python with gemmi for the structure stage (env CB_GEMMI_PYTHON)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=SEEDS,
                    help="seeds per method (exactly 0..seeds-1); must be 5 unless --smoke-epochs")
    ap.add_argument("--stats-B", type=int, default=None,
                    help="bootstrap replicates override (smoke only; the family says 10,000)")
    ap.add_argument("--smoke-epochs", type=int, default=None,
                    help="smoke only: cap the probes' epochs (never for the real run)")
    # authors' tentative answers of 2026-10-22 (defaults = tentative choice)
    ap.add_argument("--modified-residues", choices=["parent", "drop"], default="parent")
    ap.add_argument("--esm2-row", choices=["on", "off"], default="on")
    ap.add_argument("--conv-standardize", choices=["train", "none"], default="train")
    ap.add_argument("--selection", choices=["val_pooled_auroc", "val_per_entry_auroc"],
                    default="val_pooled_auroc")
    ap.add_argument("--aa-order", choices=sorted(CC.AA_ORDERS), default="collaborator")
    ap.add_argument("--esm2-layer", type=int, default=33,
                    help="ESM-2 3B layer (TENTATIVE default 33 = upstream pLM-NN; 36 = last)")
    ap.add_argument("--redo-all", action="store_true",
                    help="redo every job (each script moves its old outputs to _pre_<date>/)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-family", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--write-family-template", action="store_true",
                    help="write cryptobench_families_v1.json from make_family_template() (once)")
    CBP.add_path_args(ap)
    ap.set_defaults(run_tag=DEFAULT_RUN_TAG)
    args = ap.parse_args(argv)
    paths = CBP.resolve_paths(args)
    if args.seeds != SEEDS and not args.smoke_epochs:
        ap.error(f"--seeds must be {SEEDS} for a real run (FIXED REAUDIT CB4 F1)")
    if args.write_family_template:
        if FAMILY_TEMPLATE.exists():
            print(f"{FAMILY_TEMPLATE.name} exists; not overwritten (pre-specified)")
            return 1
        FAMILY_TEMPLATE.write_text(json.dumps(make_family_template(), indent=1) + "\n")
        print(f"wrote {FAMILY_TEMPLATE.name}")
        return 0
    if args.write_family:
        return write_family(args, paths)
    why = template_problem()
    if why:
        print(f"refusing: {why}", file=sys.stderr)
        return 2
    args.gpu_list = [int(g) for g in args.gpus.split(",") if g != ""]
    used = gpu_used_mib()
    bad = gpu_refusals(args.gpu_list, used, args.max_used_mib)
    jobs = plan(args, paths)
    # FIXED(FIXAUDIT FX3 F2 / FX1 F8, 2026-10-22): a plan with GPU jobs and no usable GPU is
    #   refused at once (an empty --gpus used to leave the GPU jobs waiting forever).
    gpu_jobs = [j.name for j in jobs if j.gpu]
    no_gpu = gpu_jobs and not [g for g in args.gpu_list if g not in bad]
    if args.dry_run:
        print("GPU state: " + ("nvidia-smi query FAILED (a launch would be refused)" if used is None
                               else ", ".join(f"GPU {g}: {used.get(g, '?')} MiB used" for g in args.gpu_list)))
        if bad:
            print(f"a launch would refuse GPUs {bad} (> {args.max_used_mib} MiB or not queryable)")
        state = initial_states(jobs, args.redo_all)
        for j in jobs:
            st = "skip (done)" if state[j.name] == "skip" else "todo"
            print(f"[{st:11s}] {j.name:40s} {'GPU' if j.gpu else 'CPU'}  deps={','.join(j.deps) or '-'}")
            print("              " + " ".join(j.cmd(redo=args.redo_all)))
        print(f"{len(jobs)} jobs; outputs: {paths.out_dir}")
        if no_gpu:
            print(f"a launch would be refused: {len(gpu_jobs)} GPU jobs and no usable GPU "
                  f"(--gpus {args.gpus!r})")
        return 0
    if no_gpu and not bad:
        print(f"refusing: the plan has {len(gpu_jobs)} GPU jobs but --gpus {args.gpus!r} gives "
              f"no GPU", file=sys.stderr)
        return 2
    if bad:
        print(f"refusing GPUs {bad}: more than {args.max_used_mib} MiB in use or nvidia-smi "
              f"failed", file=sys.stderr)
        return 2
    for k in ("structs_root", "manifest_dir", "tokens_dir", "out_dir"):
        getattr(paths, k).mkdir(parents=True, exist_ok=True)
    # FIXED(REAUDIT CB5 F8, 2026-10-22): one log folder per launch.
    log_dir = paths.out_dir / "logs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    state = run_jobs(jobs, args, log_dir)
    rep = completeness(args, paths, state, planned_outputs(jobs))
    rep.update({"run_tag": args.run_tag, "elapsed_s": round(time.time() - t0),
                "log_dir": _rel(log_dir),
                "finished": dt.datetime.now().isoformat(timespec="seconds")})
    (paths.out_dir / "completeness.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if not rep["failed_jobs"] else 1


if __name__ == "__main__":
    sys.exit(main())
