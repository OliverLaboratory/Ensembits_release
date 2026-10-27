"""Validation reconstruction loss of each Ensembits training run at P = 1, 2, 5 and 10 input frames.

What it does
------------
For every run of the ablation tables (default: the eight runs e0, lam0, lam0p05, lam0p25,
lam0p5, lam1, vq2048, nosftd under `REPO_ROOT/output/ensembits_bm_<run>`) it rebuilds the
model exactly as `train/train.py` main() does (RVQ branch, no geometric encoder), loads
`best.pt` and evaluates the validation block of the trainer with the inputs AND the
targets truncated to the first P frames (the trainer's `--eval-p` path, vb[:, :P]):
eval mode, validation split of the run's descriptor cache (fp16, cast to float per
batch), the run's batch size, loss = fape_hungarian_loss(recon, vb, dihedral_weight,
fape_clamp, match_cost = hungarian_cost) (or fape_dihedral_loss without the rectangular
Hungarian matching), averaged over batches like the trainer. The recon is computed with
the ops of RVQVAETokenizer.forward (encoder -> _quantize -> straight-through -> decoder),
i.e. it decodes from the SUM of all quantizer levels (three levels for the RVQ runs, one
for VQ-2048; the VQ table footnote says so). Also: the loss at P = 10 on a fixed random
subset of the training split (`--train-sub`, seed 0; an overfitting check), the best /
stop epoch from history.json and config.json, and a self-check: the recomputed P = 10
value must equal the run's recorded best validation loss (`repro_diff`).

The data is kept on the CPU and moved to the device one batch at a time (GPU memory
< 2 GB; the research version held the whole validation split on the GPU).

`--reuse <json> --verify-run <run>`: recompute only <run>; if every recomputed number
equals the entry of <json> exactly, the other runs are copied from <json> (recorded in
the provenance as "reused_from"); otherwise the script stops and asks for a full run.

Paper
-----
`appendix.tex` \\label{app:ablation-vq} and \\label{app:ablation-sftd} (reconstruction
columns of the VQ and SFTD ablation tables; final_fix_list A9); the P = 1 collapse of
the run without SFTD (\\label{app:ablation-sftd}).

Origin
------
Ported from `scratchpad/recon_by_p/recon_by_p.py` (final audit, 2026-10-19; not in the
research repo). Changes: paths from `ensembits.paths.REPO_ROOT` and the run's
config.json (the research version hard-coded absolute paths), batches moved to the device one
at a time, the reuse / verify mode, provenance, atomic write. Numbers unchanged (checked:
`--verify-run`).

Used by the reported model: yes (both lambda candidates and the ablation runs).

Inputs / outputs
----------------
reads   output/ensembits_bm_<run>/{config.json, best.pt, history.json, stats.npz},
        the descriptor cache and splits named in config.json (args.desc, args.splits)
writes  output/ablations/recon_by_p.json (+ .provenance.json)

How to run
----------
    python downstream/analyses/recon_by_p.py --device cuda:0                       # 8 runs
    python downstream/analyses/recon_by_p.py --device cuda:0 --verify-run lam1 \\
        --reuse <scratchpad>/recon_by_p/results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT  # noqa: E402

# FIXED(recon2 A.7, 2026-10-25): defaults switched to _rv runs and _rv output file.
RUNS = ["e0_rv", "lam0_rv", "lam0p05_rv", "lam0p25_rv", "lam0p5_rv", "lam1_rv", "vq2048_rv", "nosftd_rv"]
PS = [10, 1, 2, 5]
OUT = REPO_ROOT / "output" / "ablations" / "recon_by_p_rv.json"
SCHEMA = "recon_by_p-2026-10-19"


def run_dir(run: str) -> Path:
    """output/ensembits_bm_<run> (or the path itself when it is a directory)."""
    p = Path(run)
    if p.is_dir():
        return p
    return REPO_ROOT / "output" / f"ensembits_bm_{run}"


def _repo_path(p: str) -> Path:
    """A path recorded in config.json; relative paths are taken relative to REPO_ROOT."""
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def build_model(a: dict, P: int, D: int):
    """Same constructor call as train.py main() (RVQ branch, no geometric encoder)."""
    from ensembits import RVQVAETokenizer
    if a["quantizer"] != "rvq" or a["geom_attn_encoder"] or a["target_desc"] is not None:
        raise SystemExit("only RVQ runs without a geometric encoder / target descriptor")
    cbs = [int(x) for x in str(a["codebook_sizes"]).split(",")]
    aap_raw = str(a["aminoaseed_projection"]).strip()
    aap = ([bool(int(x)) for x in aap_raw.split(",")] if "," in aap_raw
           else [bool(int(aap_raw))] * len(cbs))
    return RVQVAETokenizer(
        input_dim=D, output_dim=D, num_prototypes=P, codebook_sizes=cbs,
        hidden_dim=a["hidden_dim"], latent_dim=a["latent_dim"],
        commitment_cost=a["commitment_cost"], encoder_type=a["encoder_type"],
        loss_type=a["loss_type"],
        vq_update=("loss" if a["vq_update"] in ("grad", "loss") else "ema"),
        ema_decay=a["ema_decay"], rotation_trick=a["rotation_trick"],
        nsvq_alpha=a["nsvq_alpha"], nsvq_tau=a["nsvq_tau"],
        n_queries=a["n_queries"], n_heads=a["n_heads"],
        n_encoder_layers=a["n_encoder_layers"], n_decoder_layers=a["n_decoder_layers"],
        aminoaseed_projection=aap, bypass_quantizer=False, residual_mode=a["rvq_residual"])


def load_run(rd: Path, P: int, D: int, device: str):
    """(model on device in eval mode, config dict) of one run; checks identity input stats."""
    cfg = json.loads((rd / "config.json").read_text())
    a = cfg["args"]
    if not a.get("no_standardize"):
        raise SystemExit(f"{rd.name}: standardised inputs are not handled")
    st = np.load(rd / "stats.npz")
    if not (np.all(st["mean"] == 0) and np.all(st["std"] == 1)):
        raise SystemExit(f"{rd.name}: stats.npz is not the identity")
    torch.manual_seed(a["seed"])
    model = build_model(a, P, D)
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), cfg


def load_split_data(cfg0: dict, train_sub: int):
    """(val array, train-subset array) of the descriptor cache named in config.json (CPU)."""
    a = cfg0["args"]
    data = np.load(_repo_path(a["desc"]), mmap_mode="r")
    splits = json.loads(_repo_path(a["splits"]).read_text())
    np_dt = np.float16 if a["desc_dtype"] == "float16" else np.float32

    def gather(index, chunk=100_000):
        index = np.asarray(index)
        out = np.empty((len(index),) + data.shape[1:], dtype=np_dt)
        for s0 in range(0, len(index), chunk):
            out[s0:s0 + chunk] = data[index[s0:s0 + chunk]]
        return out

    val = gather(splits["val"])
    tr = None
    if train_sub:
        rng = np.random.default_rng(0)
        tr = gather(np.sort(rng.choice(np.asarray(splits["train"]), size=train_sub, replace=False)))
    return val, tr


@torch.no_grad()
def recon_loss(model, a: dict, data: np.ndarray, p: int, device: str) -> float:
    """Mean over batches of the trainer's validation recon loss with inputs / targets [:, :p]."""
    from ensembits.losses import fape_hungarian_loss, fape_dihedral_loss
    if a["recon_loss_type"] != "fape":
        raise SystemExit("only fape runs are handled")
    if a["rect_hungarian"]:
        lossf = lambda r, t: fape_hungarian_loss(r, t, a["dihedral_weight"], a["fape_clamp"],
                                                 match_cost=a["hungarian_cost"])
    else:
        lossf = lambda r, t: fape_dihedral_loss(r, t, a["dihedral_weight"], a["fape_clamp"])
    bs = a["batch_size"]
    # A.2 fix: weight by rows (number of residues per batch), matching the trainer's
    # per-residue mean instead of averaging batch means.
    losses, weights = [], []
    for vs in range(0, data.shape[0], bs):
        vb = torch.from_numpy(np.array(data[vs:vs + bs])).to(device).float()
        vb = vb[:, :p, :]
        # Paper: appendix.tex app:model-impl — recon path of RVQVAETokenizer.forward
        z = model.encoder(vb)
        quantized, _, _ = model._quantize(z)
        q_st = z + (quantized - z).detach()
        recon = model.decoder(q_st)
        losses.append(lossf(recon, vb).item())
        weights.append(vb.shape[0])
    w = np.array(weights, dtype=np.float64)
    return float(np.average(losses, weights=w)) if len(losses) else 0.0


def evaluate_run(run: str, val: np.ndarray, tr, ps, device: str, ref_args: dict) -> dict:
    """One row of the output (see the module docstring)."""
    rd = run_dir(run)
    N, P, D = val.shape
    model, cfg = load_run(rd, P, D, device)
    a = cfg["args"]
    for k in ("desc", "splits", "desc_dtype"):
        if a[k] != ref_args[k]:
            raise SystemExit(f"{rd.name}: args.{k} differs from the first run's")
    if cfg.get("n_val_residues") not in (None, N):
        raise SystemExit(f"{rd.name}: n_val_residues {cfg.get('n_val_residues')} != {N}")
    h = json.loads((rd / "history.json").read_text())
    vr, ep = h["val_recon"], h["eval_epochs"]
    # A.1 fix: use the best epoch from config.codebook_health.epoch (the trainer's 1e-5 rule)
    # instead of a plain argmin of history.val_recon.
    cfg_epoch = cfg.get("codebook_health", {}).get("epoch")
    if cfg_epoch is None:
        raise SystemExit(f"{rd.name}: config.codebook_health.epoch missing")
    if cfg_epoch not in ep:
        raise SystemExit(f"{rd.name}: config.codebook_health.epoch {cfg_epoch} not in eval_epochs")
    bi = ep.index(cfg_epoch)
    # Assert the history value at the config-selected epoch equals best_val_recon
    if abs(vr[bi] - cfg["best_val_recon"]) > 1e-12:
        raise SystemExit(f"{rd.name}: history val_recon at epoch {cfg_epoch} ({vr[bi]}) "
                         f"!= config best_val_recon ({cfg['best_val_recon']})")
    row = {"lambda": a["consistency_mse_weight"], "random_p_range": a["random_p_range"],
           "codebook_sizes": a["codebook_sizes"], "best_epoch": ep[bi], "hist_best": vr[bi],
           "config_best_val_recon": cfg["best_val_recon"], "total_epochs": cfg["total_epochs"]}
    for p in ps:
        row[f"val_P{p}"] = recon_loss(model, a, val, p, device)
    if 10 in ps:
        row["repro_diff"] = row["val_P10"] - cfg["best_val_recon"]
    if tr is not None:
        row["train_sub_P10"] = recon_loss(model, a, tr, 10, device)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return row


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def provenance(runs, args, reused_from=None, verified=None) -> dict:
    return {"schema": SCHEMA, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": {r: {"dir": str(run_dir(r).relative_to(REPO_ROOT)) if str(run_dir(r)).startswith(str(REPO_ROOT)) else str(run_dir(r)),
                         "best_pt_sha256": _sha(run_dir(r) / "best.pt")} for r in runs},
            "settings": {"ps": args.ps, "train_sub": args.train_sub, "train_sub_seed": 0,
                         "device": args.device},
            "code_sha256": {"downstream/analyses/recon_by_p.py": _sha(Path(__file__))},
            "reused_from": reused_from, "verified_run": verified,
            "rule": "validation recon loss of best.pt, inputs and targets truncated to the "
                    "first P frames (train.py --eval-p path), mean over batches"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=",".join(RUNS),
                    help="comma-separated run names (output/ensembits_bm_<run>) or run dirs")
    ap.add_argument("--ps", default=",".join(map(str, PS)))
    ap.add_argument("--train-sub", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reuse", type=Path, default=None,
                    help="earlier results JSON (e.g. the scratch run); needs --verify-run")
    ap.add_argument("--verify-run", default=None,
                    help="recompute this run only and copy the others from --reuse when equal")
    args = ap.parse_args(argv)
    args.ps = [int(x) for x in str(args.ps).split(",")]
    runs = [r for r in args.runs.split(",") if r]
    if bool(args.reuse) != bool(args.verify_run):
        ap.error("--reuse and --verify-run go together")
    cfg0 = json.loads((run_dir(runs[0]) / "config.json").read_text())
    t0 = time.time()
    val, tr = load_split_data(cfg0, args.train_sub)
    print(f"loaded val {val.shape} train-sub {None if tr is None else tr.shape} "
          f"({time.time() - t0:.0f}s)", flush=True)
    results = {}
    if args.verify_run:
        old = json.loads(args.reuse.read_text())
        old = old.get("results", old)
        if args.verify_run not in old:
            raise SystemExit(f"{args.verify_run} not in {args.reuse}")
        row = evaluate_run(args.verify_run, val, tr, args.ps, args.device, cfg0["args"])
        diffs = {k: (row[k], old[args.verify_run].get(k)) for k in row
                 if row[k] != old[args.verify_run].get(k)}
        print(f"verify {args.verify_run}: {json.dumps(row)}", flush=True)
        if diffs:
            print(f"NOT identical to {args.reuse}: {diffs}; run without --reuse", file=sys.stderr)
            return 1
        missing = [r for r in runs if r not in old]
        if missing:
            raise SystemExit(f"{missing} not in {args.reuse}; run them without --reuse")
        results = {r: old[r] for r in runs}
        prov = provenance(runs, args, reused_from={"file": str(args.reuse),
                                                   "sha256": _sha(args.reuse)},
                          verified=args.verify_run)
    else:
        for r in runs:
            t1 = time.time()
            results[r] = evaluate_run(r, val, tr, args.ps, args.device, cfg0["args"])
            print(f"{r}: {json.dumps(results[r])} ({time.time() - t1:.0f}s)", flush=True)
        prov = provenance(runs, args)
    atomic_json(args.out, {"schema": SCHEMA, "results": results, "provenance": prov})
    atomic_json(args.out.with_name(args.out.name + ".provenance.json"), prov)
    print(f"wrote {args.out}")
    # A.2 fix: |repro_diff| > 1e-6 is now an error, not a warning
    bad = [r for r, x in results.items() if abs(x.get("repro_diff", 0.0)) > 1e-6]
    if bad:
        print(f"ERROR: P = 10 differs from config best_val_recon for {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
