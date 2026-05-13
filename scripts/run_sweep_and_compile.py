"""End-to-end sweep + compile pipeline (misato downstreams).

Runs every available baseline probe inside ``ensembits-repro/``,
captures one JSON per (probe, baseline, split, seed) under ``runs/``,
and writes ``ensembits-repro/all_baselines_summary.md`` in the same
format as the canonical ``submission_exp/mds/all_baselines_summary.md``.

Scope:
  - misato RMSF              (3 splits × 10 seeds)
  - misato EC depth 1/2/3    (3 splits × 10 seeds)
  - misato GO top-50         (3 splits × 10 seeds)
  - misato binding-site      (3 splits × 10 seeds)
  - misato binding affinity  (3 splits × 10 seeds; ligand-aware only —
                              MACCS or MACCS + AA-concat)
  - ProteinGym mutation effects (deterministic, 1 run per tokenizer,
                              ESM2-650M α-blend at α=0.4)

Reads only files under ``data/`` and ``ckpt/`` of this repo, plus the
external ``all_baselines_summary.md`` for cross-checking the output
format. Skips mdCATH RMSF + ProteinGym (not yet wired).

Usage:
    python -m scripts.run_sweep_and_compile --dry-run    # enumerate
    python -m scripts.run_sweep_and_compile              # run all
    python -m scripts.run_sweep_and_compile --seeds 0,1  # 2 seeds
    python -m scripts.run_sweep_and_compile --tasks rmsf,binding  # subset
    python -m scripts.run_sweep_and_compile --compile-only  # skip running
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TOKENS = DATA / "tokens"
CODEBOOKS = DATA / "codebooks"
LABELS = DATA / "labels"
SPLITS_DIR = DATA / "splits"
RUNS = REPO / "runs"

# ── Catalogue of baselines we sweep over ───────────────────────────
# (display_name, token_file_stem, codebook_file_stem_or_None)
BASELINES: list[tuple[str, str, str | None]] = [
    ("3di_tokens",        "3di_tokens",       "3di_tokens"),       # mini3di centroids (20, 2)
    ("vote_3di",          "vote_3di",          "vote_3di"),         # plurality-vote across 8 mini3di frames; uses mini3di centroids
    ("aminoaseed",        "aminoaseed",       "aminoaseed"),
    ("esm3struct",        "esm3struct",       "esm3struct"),
    ("protoken",          "protoken",         "protoken"),
    ("protprofile_K8",    "protprofile_K8",    "protprofile_K8"),  # (L,20) hist + mini3di centroids
    ("protprofile_K5",    "protprofile_K5",    "protprofile_K5"),
    ("mini3di",           "mini3di",           None),               # continuous 10-D
    ("ours/combined/ESM3",      "ours_combined_esm3",       None),  # ours: uses model.codebook
    ("ours/combined/ESM3 (P=1)", "ours_combined_esm3_P1inf", None),
]

SPLITS = ("sequence", "structure", "random")
SEEDS_DEFAULT = list(range(10))


@dataclass
class Job:
    """One probe invocation."""
    name: str                     # human-readable label
    cmd: list[str]                # argv to subprocess
    out_path: Path                # expected output JSON path
    inputs: list[Path]            # files this job reads (for dry-run validation)


# ── Builder helpers ────────────────────────────────────────────────

def _tok(baseline_stem: str, dataset: str) -> Path:
    return TOKENS / f"{baseline_stem}_{dataset}.npz"


def _cb(baseline_stem: str | None) -> Path | None:
    if baseline_stem is None:
        return None
    p = CODEBOOKS / f"{baseline_stem}.npy"
    return p if p.exists() else None


def _ours_codebook_from_ckpt() -> Path:
    """For ours/* baselines, derive an (M, d) codebook by reading the
    primary-level codebook out of best.pt (level 0 = M=2048 alphabet)."""
    cache = CODEBOOKS / "ours_combined_esm3.npy"
    if cache.exists():
        return cache
    import torch
    state = torch.load(REPO / "ckpt" / "combined_esm3" / "best.pt",
                        map_location="cpu", weights_only=True)
    # Find the L0 codebook weight key.
    candidates = [k for k in state if "vq_levels.0.codebook.weight" in k]
    if not candidates:
        raise SystemExit(f"L0 codebook not found in best.pt; keys: {list(state)[:5]}")
    w = state[candidates[0]].numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, w)
    return cache


# ── Job generators ─────────────────────────────────────────────────

def jobs_misato_rmsf(baseline_disp: str, tok_stem: str, cb_stem: str | None,
                      seeds: list[int]) -> list[Job]:
    tok = _tok(tok_stem, "misato")
    if not tok.exists():
        return []
    cb = _cb(cb_stem) if cb_stem else None
    if baseline_disp.startswith("ours/") and cb is None:
        cb = _ours_codebook_from_ckpt()
    jobs = []
    for split in SPLITS:
        for s in seeds:
            out = RUNS / "rmsf_misato" / f"{tok_stem}__{split}__seed{s}.json"
            cmd = [sys.executable, "-m", "probes.probe_rmsf",
                   "--features", str(tok),
                   "--labels",   str(LABELS / "rmsf_misato.npz"),
                   "--splits",   str(SPLITS_DIR / "misato_splits.json"),
                   "--split-name", split,
                   "--out", str(out),
                   "--seed", str(s)]
            if cb is not None:
                cmd += ["--codebook", str(cb)]
            jobs.append(Job(
                name=f"misato_rmsf  {baseline_disp:<24}  split={split:<9}  seed={s}",
                cmd=cmd, out_path=out,
                inputs=[tok, LABELS / "rmsf_misato.npz",
                        SPLITS_DIR / "misato_splits.json"] + ([cb] if cb else []),
            ))
    return jobs


def jobs_ec(baseline_disp: str, tok_stem: str, cb_stem: str | None,
             seeds: list[int], depths: list[int] = (1, 2, 3)) -> list[Job]:
    tok = _tok(tok_stem, "misato")
    if not tok.exists() or not (LABELS / "misato_ec.json").exists():
        return []
    cb = _cb(cb_stem) if cb_stem else None
    if baseline_disp.startswith("ours/") and cb is None:
        cb = _ours_codebook_from_ckpt()
    jobs = []
    for depth in depths:
        for split in SPLITS:
            for s in seeds:
                out = RUNS / "ec_misato" / f"{tok_stem}__d{depth}__{split}__seed{s}.json"
                cmd = [sys.executable, "-m", "probes.probe_ec",
                       "--features", str(tok),
                       "--labels",   str(LABELS / "misato_ec.json"),
                       "--splits",   str(SPLITS_DIR / "misato_splits.json"),
                       "--split-name", split,
                       "--depth", str(depth),
                       "--out", str(out),
                       "--seed", str(s)]
                if cb is not None:
                    cmd += ["--codebook", str(cb)]
                jobs.append(Job(
                    name=f"ec  {baseline_disp:<24}  d={depth}  split={split:<9}  seed={s}",
                    cmd=cmd, out_path=out,
                    inputs=[tok, LABELS / "misato_ec.json",
                            SPLITS_DIR / "misato_splits.json"] + ([cb] if cb else []),
                ))
    return jobs


def jobs_go(baseline_disp: str, tok_stem: str, cb_stem: str | None,
             seeds: list[int]) -> list[Job]:
    tok = _tok(tok_stem, "misato")
    if not tok.exists() or not (LABELS / "misato_go.json").exists():
        return []
    cb = _cb(cb_stem) if cb_stem else None
    if baseline_disp.startswith("ours/") and cb is None:
        cb = _ours_codebook_from_ckpt()
    jobs = []
    for split in SPLITS:
        for s in seeds:
            out = RUNS / "go_misato" / f"{tok_stem}__{split}__seed{s}.json"
            cmd = [sys.executable, "-m", "probes.probe_go",
                   "--features", str(tok),
                   "--labels",   str(LABELS / "misato_go.json"),
                   "--splits",   str(SPLITS_DIR / "misato_splits.json"),
                   "--split-name", split,
                   "--out", str(out),
                   "--seed", str(s)]
            if cb is not None:
                cmd += ["--codebook", str(cb)]
            jobs.append(Job(
                name=f"go  {baseline_disp:<24}  split={split:<9}  seed={s}",
                cmd=cmd, out_path=out,
                inputs=[tok, LABELS / "misato_go.json",
                        SPLITS_DIR / "misato_splits.json"] + ([cb] if cb else []),
            ))
    return jobs


def jobs_binding_site(baseline_disp: str, tok_stem: str, cb_stem: str | None,
                      seeds: list[int]) -> list[Job]:
    tok = _tok(tok_stem, "misato")
    if not tok.exists() or not (LABELS / "misato_binding_site.npz").exists():
        return []
    cb = _cb(cb_stem) if cb_stem else None
    if baseline_disp.startswith("ours/") and cb is None:
        cb = _ours_codebook_from_ckpt()
    jobs = []
    for split in SPLITS:
        for s in seeds:
            out = RUNS / "binding_site" / f"{tok_stem}__{split}__seed{s}.json"
            cmd = [sys.executable, "-m", "probes.probe_binding_site",
                   "--features", str(tok),
                   "--labels",   str(LABELS / "misato_binding_site.npz"),
                   "--splits",   str(SPLITS_DIR / "misato_splits.json"),
                   "--split-name", split,
                   "--out", str(out),
                   "--seed", str(s)]
            if cb is not None:
                cmd += ["--codebook", str(cb)]
            jobs.append(Job(
                name=f"bs  {baseline_disp:<24}  split={split:<9}  seed={s}",
                cmd=cmd, out_path=out,
                inputs=[tok, LABELS / "misato_binding_site.npz",
                        SPLITS_DIR / "misato_splits.json"] + ([cb] if cb else []),
            ))
    return jobs


def jobs_binding_affinity(baseline_disp: str, tok_stem: str, cb_stem: str | None,
                           seeds: list[int]) -> list[Job]:
    """Two variants per (baseline, split, seed):
      - lig:      protein + MACCS ligand
      - lig+aa:   protein + AA-onehot + MACCS ligand

    The "no-ligand" variant was removed: predicting binding affinity from
    a protein alone (no ligand identity) is not a well-defined task — a
    single protein has many binding partners with different affinities,
    so the label is one-to-many and the metric is meaningless.
    """
    tok = _tok(tok_stem, "misato")
    if not tok.exists() or not (LABELS / "misato_affinity.json").exists():
        return []
    cb = _cb(cb_stem) if cb_stem else None
    if baseline_disp.startswith("ours/") and cb is None:
        cb = _ours_codebook_from_ckpt()
    aa_tok = _tok("aa", "misato") if _tok("aa", "misato").exists() else None
    lig = LABELS / "misato_ligand_maccs.npz"
    base_inputs = [tok, LABELS / "misato_affinity.json", SPLITS_DIR / "misato_splits.json"]
    if cb is not None:
        base_inputs.append(cb)

    jobs = []
    for split in SPLITS:
        for s in seeds:
            for variant_tag, extra_args, extra_inputs in (
                ("lig",   ["--ligand-maccs", str(lig)], [lig]),
                ("aa_lig", ["--ligand-maccs", str(lig)] + (
                    ["--concat-aa", str(aa_tok)] if aa_tok else []),
                    [lig] + ([aa_tok] if aa_tok else [])),
            ):
                out = RUNS / "binding_affinity" / f"{tok_stem}__{variant_tag}__{split}__seed{s}.json"
                cmd = [sys.executable, "-m", "probes.probe_binding_affinity",
                       "--features", str(tok),
                       "--labels",   str(LABELS / "misato_affinity.json"),
                       "--splits",   str(SPLITS_DIR / "misato_splits.json"),
                       "--split-name", split,
                       "--out", str(out),
                       "--seed", str(s)] + extra_args
                if cb is not None:
                    cmd += ["--codebook", str(cb)]
                jobs.append(Job(
                    name=f"aff  {baseline_disp:<24}  {variant_tag:<7}  split={split:<9}  seed={s}",
                    cmd=cmd, out_path=out,
                    inputs=base_inputs + extra_inputs,
                ))
    return jobs


# PG tokenizer name → on-disk filename stem under data/pg/{wt,mut}_tokens/<stem>.npz
# and data/pg/codebooks/<stem>.npy. ours/combined/ESM3 stem is `combined_esm3`.
PG_TAG_TO_STEM: dict[str, str] = {
    "3di_tokens":          "3di_tokens",
    "aminoaseed":          "aminoaseed",
    "esm3struct":          "esm3struct",
    "protoken":            "protoken",
    "ours/combined/ESM3":  "combined_esm3",
}


def jobs_proteingym(baseline_disp: str, tok_stem: str, cb_stem: str | None,
                     seeds: list[int]) -> list[Job]:
    """ProteinGym Spearman + ESM2 α-blend. PG is deterministic (no seed —
    one run per tokenizer)."""
    stem = PG_TAG_TO_STEM.get(baseline_disp)
    if stem is None:
        return []
    wt = DATA / "pg" / "wt_tokens" / f"{stem}.npz"
    mt = DATA / "pg" / "mut_tokens" / f"{stem}.npz"
    cb = DATA / "pg" / "codebooks" / f"{stem}.npy"
    dms = DATA / "pg" / "dms_scores.json"
    esm2 = DATA / "pg" / "esm2_scores.json"
    if not all(p.exists() for p in (wt, mt, cb, dms, esm2)):
        return []
    out = RUNS / "proteingym" / f"{stem}.json"
    cmd = [sys.executable, "-m", "probes.probe_proteingym",
           "--wt-tokens", str(wt), "--mut-tokens", str(mt),
           "--codebook", str(cb),
           "--dms", str(dms), "--esm2", str(esm2),
           "--out", str(out)]
    return [Job(name=f"pg  {baseline_disp:<24}  (deterministic, 96 assays)",
                 cmd=cmd, out_path=out,
                 inputs=[wt, mt, cb, dms, esm2])]


TASK_BUILDERS = {
    "rmsf":     jobs_misato_rmsf,
    "ec":       jobs_ec,
    "go":       jobs_go,
    "bs":       jobs_binding_site,
    "aff":      jobs_binding_affinity,
    "pg":       jobs_proteingym,
}


# ── Compile (runs/ → all_baselines_summary.md) ─────────────────────

def _seed_files(probe_dir: Path, tag: str, split: str, depth: str = "") -> list[Path]:
    pattern = f"{tag}__"
    if depth:
        pattern += f"{depth}__"
    pattern += f"{split}__seed*.json"
    return sorted(probe_dir.glob(pattern))


def _agg(paths: list[Path], json_path: list[str]) -> str:
    if not paths:
        return "—"
    vals = []
    for p in paths:
        d = json.loads(p.read_text())
        for k in json_path:
            d = d[k] if isinstance(d, dict) and k in d else None
            if d is None:
                break
        if d is None:
            continue
        vals.append(float(d))
    if not vals:
        return "—"
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f} (n={len(vals)})"


def compile_md() -> Path:
    """Read everything in RUNS/ and write a parallel ``all_baselines_summary.md``."""
    out = REPO / "all_baselines_summary.md"
    lines: list[str] = []
    lines.append("# All-baselines summary — reproduced via "
                 "`scripts/run_sweep_and_compile.py`\n")
    lines.append(f"Compiled: **{time.strftime('%Y-%m-%d %H:%M:%S')}**\n")
    lines.append("Inputs: token caches under `data/tokens/`, labels under "
                 "`data/labels/`, splits under `data/splits/`, shipped "
                 "checkpoint under `ckpt/combined_esm3/`.\n")
    lines.append("Each cell: `mean ± std (n=k)` over seed 0..k-1. "
                 "Cells with no run land as `—`.\n")

    baselines = [(disp, tok_stem) for disp, tok_stem, _ in BASELINES]

    # ── misato RMSF ──
    lines.append("\n## FlexRMSF (misato, MLP head)\n")
    for split in ("structure", "sequence", "random"):
        lines.append(f"\n### misato ({split} split)\n")
        lines.append("| baseline | Spearman |\n|---|---|\n")
        rmsf_dir = RUNS / "rmsf_misato"
        rows = []
        for disp, ts in baselines:
            cell = _agg(_seed_files(rmsf_dir, ts, split),
                        ["best_val_spearman"])
            rows.append((disp, cell, _cell_mean(cell)))
        for disp, cell, _m in sorted(rows, key=lambda r: -r[2]):
            lines.append(f"| {disp} | {cell} |\n")

    # ── EC ──
    for depth in (1, 2, 3):
        lines.append(f"\n## EC depth-{depth} (misato, conv1d head)\n")
        for split in SPLITS:
            lines.append(f"\n### {split} split\n")
            lines.append("| baseline | top-1 | mAP | μAP | F1@0.5 | μF1@0.5 |\n"
                         "|---|---|---|---|---|---|\n")
            ec_dir = RUNS / "ec_misato"
            rows = []
            for disp, ts in baselines:
                seed_paths = _seed_files(ec_dir, ts, split, depth=f"d{depth}")
                c_top1 = _agg(seed_paths, ["top1_hit"])
                c_map = _agg(seed_paths, ["macro_ap"])
                c_uap = _agg(seed_paths, ["micro_ap"])
                c_f1  = _agg(seed_paths, ["macro_f1@0.5"])
                c_uf1 = _agg(seed_paths, ["micro_f1@0.5"])
                rows.append((disp, c_top1, c_map, c_uap, c_f1, c_uf1,
                             _cell_mean(c_top1)))
            for disp, c1, c2, c3, c4, c5, _m in sorted(rows, key=lambda r: -r[6]):
                lines.append(f"| {disp} | {c1} | {c2} | {c3} | {c4} | {c5} |\n")

    # ── GO ──
    lines.append("\n## GO top-50 classification (misato, conv1d head)\n")
    for split in SPLITS:
        lines.append(f"\n### {split} split\n")
        lines.append("| baseline | top-1 | mAP | μAP | F1@0.5 | μF1@0.5 |\n"
                     "|---|---|---|---|---|---|\n")
        go_dir = RUNS / "go_misato"
        rows = []
        for disp, ts in baselines:
            paths = _seed_files(go_dir, ts, split)
            cs = [_agg(paths, [k]) for k in
                  ("top1_hit", "macro_ap", "micro_ap", "macro_f1@0.5", "micro_f1@0.5")]
            rows.append((disp, *cs, _cell_mean(cs[0])))
        for disp, *cs, _m in sorted(rows, key=lambda r: -r[-1]):
            cells = " | ".join(cs)
            lines.append(f"| {disp} | {cells} |\n")

    # ── Binding-site ──
    lines.append("\n## Binding-site (misato, per-residue BCE → AUROC / AP)\n")
    for split in SPLITS:
        lines.append(f"\n### {split} split\n")
        lines.append("| baseline | AUROC | AP |\n|---|---|---|\n")
        bs_dir = RUNS / "binding_site"
        rows = []
        for disp, ts in baselines:
            paths = _seed_files(bs_dir, ts, split)
            c_auroc = _agg(paths, ["auroc"])
            c_ap = _agg(paths, ["ap"])
            rows.append((disp, c_auroc, c_ap, _cell_mean(c_auroc)))
        for disp, c1, c2, _m in sorted(rows, key=lambda r: -r[3]):
            lines.append(f"| {disp} | {c1} | {c2} |\n")

    # ── Binding affinity (ligand-aware only) ──
    for variant_tag, variant_title in (
        ("lig",    "Binding affinity, ligand-aware (MACCS)"),
        ("aa_lig", "Binding affinity, ligand-aware + AA concat"),
    ):
        lines.append(f"\n## {variant_title} (misato, MLP head → R² / Spearman / MSE)\n")
        for split in SPLITS:
            lines.append(f"\n### {split} split\n")
            lines.append("| baseline | R² | Spearman | MSE |\n|---|---|---|---|\n")
            aff_dir = RUNS / "binding_affinity"
            rows = []
            for disp, ts in baselines:
                paths = sorted(aff_dir.glob(
                    f"{ts}__{variant_tag}__{split}__seed*.json"))
                c_r2 = _agg(paths, ["r2"])
                c_sp = _agg(paths, ["spearman"])
                c_mse = _agg(paths, ["mse"])
                rows.append((disp, c_r2, c_sp, c_mse, _cell_mean(c_sp)))
            for disp, c1, c2, c3, _m in sorted(rows, key=lambda r: -r[4]):
                lines.append(f"| {disp} | {c1} | {c2} | {c3} |\n")

    # ── ProteinGym mutation-effects ──
    # PG is deterministic (1 file per tokenizer, no seeds). Cells reported:
    # alone Spearman, +ESM2 α=0.4 blend Spearman, Δ vs ESM2-alone, n_assays.
    lines.append("\n## ProteinGym mutation-effects (96 assays)\n")
    lines.append("| baseline | alone | + ESM2 (α=0.3) | Δ vs ESM2-alone | n_assays |\n"
                 "|---|---|---|---|---|\n")
    pg_dir = RUNS / "proteingym"
    esm2_anchor = None
    pg_rows = []
    for disp, tok_stem, _ in BASELINES:
        stem = PG_TAG_TO_STEM.get(disp)
        if stem is None:
            continue
        p = pg_dir / f"{stem}.json"
        if not p.exists():
            pg_rows.append((disp, "—", "—", "—", "—", -1e9))
            continue
        d = json.loads(p.read_text())
        alone = d.get("mean_spearman_ours")
        # Headline α = 0.3 (grid-best in the canonical sweep). The JSON
        # also carries a01/a02/a04/a05 if you want to display the full
        # α-ladder for a particular table.
        blend = d.get("mean_zmix_a03")
        anchor = d.get("mean_spearman_esm2")
        if esm2_anchor is None and anchor is not None:
            esm2_anchor = float(anchor)
        delta = (float(blend) - float(anchor)) if (blend is not None
                                                    and anchor is not None) else None
        n_assays = d.get("n_assays", "—")
        pg_rows.append((
            disp,
            f"{float(alone):.4f}" if alone is not None else "—",
            f"{float(blend):.4f}" if blend is not None else "—",
            f"{delta:+.4f}" if delta is not None else "—",
            f"{n_assays}",
            float(alone) if alone is not None else -1e9,
        ))
    for disp, a, b, d, n, _ in sorted(pg_rows, key=lambda r: -r[5]):
        lines.append(f"| {disp} | {a} | {b} | {d} | {n} |\n")
    if esm2_anchor is not None:
        lines.append(f"| ESM2-650M alone | {esm2_anchor:.4f} | — | 0 (anchor) | 96 |\n")

    out.write_text("".join(lines))
    return out


_CELL_RE = __import__("re").compile(
    r"(?P<mean>-?\d+\.\d+)\s*[±+]\s*(?P<std>\d+\.\d+)\s*\(n=(?P<n>\d+)\)")


def _cell_mean(cell: str) -> float:
    m = _CELL_RE.search(cell)
    return float(m["mean"]) if m else -1e9


# ── Driver ─────────────────────────────────────────────────────────

def build_all_jobs(seeds: list[int], tasks: set[str],
                    depths: list[int] = (1, 2, 3)) -> list[Job]:
    jobs: list[Job] = []
    for disp, tok_stem, cb_stem in BASELINES:
        for task in ("rmsf", "ec", "go", "bs", "aff", "pg"):
            if task not in tasks:
                continue
            if task == "ec":
                jobs += jobs_ec(disp, tok_stem, cb_stem, seeds, depths=depths)
            else:
                jobs += TASK_BUILDERS[task](disp, tok_stem, cb_stem, seeds)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Enumerate jobs + check inputs without running.")
    ap.add_argument("--compile-only", action="store_true",
                    help="Skip running; only compile runs/ → all_baselines_summary.md.")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS_DEFAULT)),
                    help="Comma-list of seeds (default 0..9)")
    ap.add_argument("--tasks", default="rmsf,ec,go,bs,aff,pg",
                    help="Comma-list from {rmsf,ec,go,bs,aff,pg}")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap the number of jobs run (debug)")
    ap.add_argument("--depths", default="1,2,3",
                    help="EC depths to sweep (default 1,2,3). Pass '1' to skip d2/d3.")
    ap.add_argument("--shard", default=None,
                    help="Round-robin shard 'I/N' (e.g. '0/4'); useful for "
                         "multi-GPU parallel launches sharing the same runs/.")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    tasks = set(args.tasks.split(","))
    depths = [int(x) for x in args.depths.split(",")]
    shard_i, n_shards = (None, 1)
    if args.shard:
        shard_i, n_shards = (int(x) for x in args.shard.split("/"))

    if not args.compile_only:
        jobs = build_all_jobs(seeds, tasks, depths=depths)
        if shard_i is not None:
            jobs = [j for k, j in enumerate(jobs) if k % n_shards == shard_i]
            print(f"[shard] {shard_i}/{n_shards}: {len(jobs)} jobs in this shard")
        if args.limit:
            jobs = jobs[: args.limit]
        print(f"[plan] {len(jobs)} jobs, seeds={seeds}, tasks={sorted(tasks)}")

        if args.dry_run:
            n_missing = 0
            for j in jobs[:5] + jobs[-5:] if len(jobs) > 10 else jobs:
                missing = [p for p in j.inputs if not p.exists()]
                tag = "OK " if not missing else f"MISS({len(missing)})"
                print(f"  [{tag}] {j.name}")
            for j in jobs:
                missing = [p for p in j.inputs if not p.exists()]
                if missing:
                    n_missing += 1
            print(f"\n[dry-run] {n_missing}/{len(jobs)} jobs have missing inputs")
            # show sample command
            if jobs:
                print(f"\n[sample command]\n  {shlex.join(jobs[0].cmd)}")
            return 0

        # Real run
        RUNS.mkdir(parents=True, exist_ok=True)
        t0 = time.time(); n_done = n_skip = n_fail = 0
        for i, j in enumerate(jobs):
            if j.out_path.exists():
                n_skip += 1
                continue
            j.out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{i+1}/{len(jobs)}] {j.name}", flush=True)
            env = {**os.environ, "PYTHONPATH": str(REPO)}
            try:
                subprocess.run(j.cmd, check=True, cwd=str(REPO), env=env,
                               capture_output=True)
                n_done += 1
            except subprocess.CalledProcessError as exc:
                print(f"  FAIL: {exc.stderr.decode()[-300:]}")
                n_fail += 1
        elapsed = time.time() - t0
        print(f"\n[done] new={n_done}  skip={n_skip}  fail={n_fail}  "
              f"({elapsed:.0f}s)")

    out = compile_md()
    print(f"[compile] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
