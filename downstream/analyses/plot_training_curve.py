"""Training curve figure from history.json: recon, consistency, VQ loss and codebook use.

What it does
------------
Reads ``<model-dir>/history.json`` (written by the trainer at each epoch) and
draws a multi-panel figure:
  (a) reconstruction loss (train and validation, plotted at evaluation epochs);
  (b) consistency loss (train);
  (c) VQ commitment loss (train);
  (d) level-0 codebook utilisation and perplexity (at evaluation epochs).
Each panel is one subplot sharing the epoch x-axis. The checkpoint chosen by
early stopping (min val recon) is marked with a vertical line.

Paper
-----
``appendix.tex`` Figure fig:placeholder (``img/TrainingCurveAPP.pdf``).

Origin
------
New (2026-10-25, figs task). Replaces the old ``training_curve.png`` (written
by the trainer with matplotlib during training, old model).

Used by the reported model: yes (fig:placeholder, after the retrain).

Inputs / outputs
----------------
reads   <model-dir>/history.json, config.json
writes  <out-dir>/training_curve.{pdf,png} and .provenance.json

How to run
----------
    python downstream/analyses/plot_training_curve.py \\
        --model-dir output/ensembits_bm_lam10_rv \\
        --out-dir output/figures/lam10
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

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensembits.paths import REPO_ROOT  # noqa: E402
from _common import resolve_model_dir  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a8985"
GREEN = "#1baf7a"
PURPLE = "#6250d6"


def file_sha256(path, n=12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def draw(history: dict, config: dict, model_dir: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    train_recon = np.array(history["train_recon"])
    val_recon = np.array(history["val_recon"])
    eval_epochs = np.array(history["eval_epochs"])
    train_vq = np.array(history["train_vq"])
    train_consistency = np.array(history["train_consistency"])
    util_L0 = np.array(history["util_L0"])
    ppl_L0 = np.array(history["ppl_L0"])

    train_epochs = np.arange(1, len(train_recon) + 1)
    best_epoch = int(eval_epochs[np.argmin(val_recon)])
    lam = config.get("args", {}).get("consistency_mse_weight",
                                     config.get("consistency_mse_weight", "?"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    # (a) Reconstruction loss
    ax = axes[0, 0]
    ax.plot(train_epochs, train_recon, color=BLUE, lw=0.8, alpha=0.7, label="train")
    ax.plot(eval_epochs, val_recon, color=ORANGE, lw=1.2, label="validation")
    ax.axvline(best_epoch, color=GRAY, ls="--", lw=0.8, label=f"best epoch {best_epoch}")
    ax.set_ylabel("reconstruction loss")
    ax.set_title("(a) Reconstruction", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)

    # (b) Consistency loss
    ax = axes[0, 1]
    ax.plot(train_epochs, train_consistency, color=GREEN, lw=0.8)
    ax.axvline(best_epoch, color=GRAY, ls="--", lw=0.8)
    ax.set_ylabel(f"consistency loss ($\\lambda = {lam}$)")
    ax.set_title("(b) Consistency (SFTD)", fontsize=9, loc="left")

    # (c) VQ commitment loss
    ax = axes[1, 0]
    ax.plot(train_epochs, train_vq, color=PURPLE, lw=0.8)
    ax.axvline(best_epoch, color=GRAY, ls="--", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("VQ commitment loss")
    ax.set_title("(c) VQ commitment", fontsize=9, loc="left")

    # (d) Codebook utilisation and perplexity
    ax = axes[1, 1]
    M = int(config.get("codebook_sizes", [2048])[0])
    ax.plot(eval_epochs, util_L0 * 100, color=BLUE, lw=1.0, label="utilisation (%)")
    ax.set_ylabel("utilisation (%)")
    ax.set_ylim(0, 105)
    ax.axvline(best_epoch, color=GRAY, ls="--", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(eval_epochs, ppl_L0, color=ORANGE, lw=1.0, ls="--", label=f"perplexity (M = {M})")
    ax2.set_ylabel("perplexity", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, frameon=False, loc="center right")
    ax.set_xlabel("epoch")
    ax.set_title("(d) Level-0 codebook health", fontsize=9, loc="left")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        if a != axes[1, 1]:
            a.spines["right"].set_visible(False)

    fig.suptitle(f"Training curve: {model_dir.name} ($\\lambda = {lam}$, best epoch {best_epoch})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"training_curve.{ext}"
        tmp = p.with_name("." + p.name + ".tmp")
        fig.savefig(tmp, format=ext, dpi=200 if ext == "png" else None, bbox_inches="tight")
        os.replace(tmp, p)
        paths.append(p)
    plt.close(fig)
    return paths, best_epoch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=str(REPO_ROOT / "output" / "ensembits_bm_lam10_rv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "figures" / "lam10"))
    args = ap.parse_args(argv)

    model_dir = resolve_model_dir(args.model_dir)
    history_path = model_dir / "history.json"
    config_path = model_dir / "config.json"
    if not history_path.exists():
        print(f"ERROR: {history_path} not found", file=sys.stderr)
        return 1
    history = json.loads(history_path.read_text())
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    out_dir = Path(args.out_dir)
    paths, best_epoch = draw(history, config, model_dir, out_dir)

    prov = {
        "model_dir": model_dir.name,
        "history_sha256_12": file_sha256(history_path),
        "config_sha256_12": file_sha256(config_path) if config_path.exists() else None,
        "best_epoch": best_epoch,
        "n_train_epochs": len(history["train_recon"]),
        "n_eval_epochs": len(history["val_recon"]),
        "lambda": config.get("args", {}).get("consistency_mse_weight",
                                             config.get("consistency_mse_weight")),
        "figures": [str(p) for p in paths],
        "script": "downstream/analyses/plot_training_curve.py",
        "script_sha256_12": file_sha256(Path(__file__)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    prov_path = out_dir / "training_curve.provenance.json"
    tmp = prov_path.with_name("." + prov_path.name + ".tmp")
    tmp.write_text(json.dumps(prov, indent=1))
    os.replace(tmp, prov_path)

    for p in paths:
        print(f"wrote {p}")
    print(f"best epoch: {best_epoch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
