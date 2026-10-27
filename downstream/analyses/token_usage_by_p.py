"""Level-1 token usage (distinct ids, utilisation, perplexity) of each Ensembits run at P = 1 and P = 10 frames.

What it does
------------
For every run of the ablation tables (default the eight runs of `recon_by_p.py`) it
rebuilds the model as the trainer does (`recon_by_p.build_model`), encodes the
validation split of the run's descriptor cache (the split the trainer reports its
per-level utilisation / perplexity on, config.json `per_level`) with the inputs
truncated to the first P frames, and counts the first-level (L1) token ids:
  distinct ids, utilisation = distinct / M_1, perplexity = exp(entropy of the id
  frequencies) (the formulas of RVQVAETokenizer.utilization / perplexity).
P = 10 is a self-check: it must reproduce config.json `per_level[0]` (`unique`, `ppl`;
the trainer encoded the same split on the GPU; the difference is stored). P = 1 is the
number the SFTD table needs (the retrained SFTD-off run emits 51 L1 ids at P = 1, not
one token). Emitted ids only: duplicated / shadow codebook rows (final_fix_list A6) are
never emitted and are not counted.

`--source anova-cache` instead counts the ids of the token ANOVA caches
(`output/token_anova/token_cache/tokens_mdcath_flat__<variant>__*__P<P>__*.npz`,
mdCATH-div, all 2,442 domains) when they exist; a different residue set (held-out
domains, not the training corpus's validation split), reported separately.

Runs on the CPU by default (`--device cpu`; the GPUs are kept for other jobs).

Paper
-----
`appendix.tex` \\label{app:ablation-vq} (L1 utilisation / perplexity of e0 vs VQ-2048) and
\\label{app:ablation-sftd} (P = 1 token count column; final_fix_list A9).

Origin
------
New (2026-10-19, final_fix_list A9).

Used by the reported model: yes (both lambda candidates and the ablation runs).

Inputs / outputs
----------------
reads   output/ensembits_bm_<run>/{config.json, best.pt, stats.npz}, the descriptor cache
        and splits named in config.json; (--source anova-cache) the token ANOVA caches
writes  output/ablations/token_usage_by_p.json (default; --out) (+ .provenance.json)

How to run
----------
    python downstream/analyses/token_usage_by_p.py                  # validation split, CPU
    python downstream/analyses/token_usage_by_p.py --source anova-cache --out output/ablations/token_usage_mdcath.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT  # noqa: E402
import recon_by_p as RB  # noqa: E402

# FIXED(recon2 A.7, 2026-10-25): default output switched to _rv file.
OUT = REPO_ROOT / "output" / "ablations" / "token_usage_by_p_rv.json"
ANOVA_CACHE = REPO_ROOT / "output" / "token_anova" / "token_cache"
SCHEMA = "token_usage_by_p-2026-10-19"


def usage(ids, M: int) -> dict:
    """{'distinct', 'M', 'util', 'ppl', 'n'} of a 1-D array of token ids (RVQVAE formulas)."""
    ids = np.asarray(ids).astype(np.int64).ravel()
    counts = np.bincount(ids, minlength=M).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    distinct = int((counts > 0).sum())
    return {"distinct": distinct, "M": int(M), "util": distinct / M,
            "ppl": float(np.exp(-(p * np.log(p)).sum())), "n": int(ids.size)}


@torch.no_grad()
def level1_ids(model, data: np.ndarray, p: int, bs: int, device: str) -> np.ndarray:
    """First-level ids of every row of `data` with the inputs truncated to [:, :p]."""
    out = []
    for s in range(0, data.shape[0], bs):
        vb = torch.from_numpy(np.array(data[s:s + bs])).to(device).float()[:, :p, :]
        out.append(model.encode_all(vb)[0].cpu().numpy())
    return np.concatenate(out)


def from_validation(runs, ps, device, threads: int) -> dict:
    torch.set_num_threads(threads)
    cfg0 = json.loads((RB.run_dir(runs[0]) / "config.json").read_text())
    val, _ = RB.load_split_data(cfg0, 0)
    N, P, D = val.shape
    res = {}
    for r in runs:
        t0 = time.time()
        model, cfg = RB.load_run(RB.run_dir(r), P, D, device)
        a = cfg["args"]
        for k in ("desc", "splits"):
            if a[k] != cfg0["args"][k]:
                raise SystemExit(f"{r}: args.{k} differs from the first run's")
        M = int(str(a["codebook_sizes"]).split(",")[0])
        row = {"lambda": a["consistency_mse_weight"], "random_p_range": a["random_p_range"],
               "codebook_sizes": a["codebook_sizes"], "n_val_residues": int(N)}
        for p in ps:
            row[f"P{p}"] = usage(level1_ids(model, val, p, a["batch_size"], device), M)
        pl = (cfg.get("per_level") or [{}])[0]
        if 10 in ps and pl:
            row["config_per_level_L1"] = pl
            row["P10_check"] = {"distinct_diff": row["P10"]["distinct"] - int(pl["unique"]),
                                "ppl_diff": row["P10"]["ppl"] - float(pl["ppl"])}
        res[r] = row
        print(f"{r}: " + ", ".join(f"P{p} {row[f'P{p}']['distinct']} ids ppl "
                                    f"{row[f'P{p}']['ppl']:.1f}" for p in ps)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        del model
    return res


def from_anova_cache(runs, ps) -> dict:
    res = {}
    for r in runs:
        rd = RB.run_dir(r)
        cfg = json.loads((rd / "config.json").read_text())
        M = int(str(cfg["args"]["codebook_sizes"]).split(",")[0])
        sha12 = RB._sha(rd / "best.pt")[:12]
        row = {"lambda": cfg["args"]["consistency_mse_weight"]}
        for p in ps:
            hits = sorted(glob.glob(str(ANOVA_CACHE / f"tokens_mdcath_flat__{rd.name}__{sha12}__*__P{p}__*.npz")))
            if not hits:
                row[f"P{p}"] = None
                continue
            with np.load(hits[-1], allow_pickle=False) as z:
                key = "tokens" if "tokens" in z.files else z.files[0]
                tok = z[key]
            ids = tok[:, 0] if tok.ndim == 2 else tok
            row[f"P{p}"] = {**usage(ids, M), "cache": Path(hits[-1]).name}
        res[r] = row
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=",".join(RB.RUNS))
    ap.add_argument("--ps", default="1,10")
    ap.add_argument("--source", choices=("validation", "anova-cache"), default="validation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    runs = [r for r in args.runs.split(",") if r]
    ps = [int(x) for x in args.ps.split(",")]
    t0 = time.time()
    res = (from_validation(runs, ps, args.device, args.threads) if args.source == "validation"
           else from_anova_cache(runs, ps))
    prov = {"schema": SCHEMA, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": args.source, "ps": ps, "device": args.device,
            "runs": {r: {"best_pt_sha256": RB._sha(RB.run_dir(r) / "best.pt")} for r in runs},
            "code_sha256": {"downstream/analyses/token_usage_by_p.py": RB._sha(Path(__file__)),
                            "downstream/analyses/recon_by_p.py": RB._sha(Path(RB.__file__))},
            "rule": "level-1 ids of best.pt, inputs truncated to the first P frames; distinct "
                    "emitted ids, util = distinct / M_1, ppl = exp(entropy of id frequencies)",
            "runtime_s": round(time.time() - t0, 1)}
    RB.atomic_json(args.out, {"schema": SCHEMA, "results": res, "provenance": prov})
    RB.atomic_json(args.out.with_name(args.out.name + ".provenance.json"), prov)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
