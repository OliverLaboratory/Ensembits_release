"""Two small helpers of the trainer: a timestamped run folder and the training-curve plot.

What it does
------------
  * make_run_dir(label): creates <OUTPUT_ROOT>/<YYYY-MM-DD_HH-MM-SS>__<label>/
    where the trainer drops per-run plots.
  * plot_training_curves(history, *paths): 3-panel figure (reconstruction loss,
    code utilisation, perplexity per level) -> `training_curve.png`.

Paper: `appendix.tex` \\label{app:model-impl}, Figure "Training curve"
(`img/TrainingCurveAPP.pdf`) is drawn from the same `history.json`.
# FIXED(require_fix #15 / D.23, tex pass 2026-10-26): training curve for λ=10 in appendix.

Origin
------
Copied from `Ensembits/utils/paths.py` (make_run_dir) and
`Ensembits/utils/plots.py` (plot_training_curves). Logic unchanged. OUTPUT_ROOT
keeps the original location (`<research repo>/Ensembits/output`).

Used by the reported model: yes (run folder + training_curve.png).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable  # used by plot_training_curves

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensembits.paths import REPO_ROOT as _RESEARCH_REPO  # noqa: E402

# Original: REPO_ROOT = Ensembits/ ; OUTPUT_ROOT = Ensembits/output
OUTPUT_ROOT: Path = _RESEARCH_REPO / "Ensembits" / "output"


def make_run_dir(label: str | None = None,
                 *,
                 root: Path = OUTPUT_ROOT,
                 timestamp: str | None = None) -> Path:
    """Return (and create) `<root>/<YYYY-MM-DD_HH-MM-SS>[__<label>]/`."""
    ts = timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{ts}__{label}" if label else ts
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    return p
def plot_training_curves(history: dict, *out_paths: str | Path) -> None:
    """Render the standard 3-panel training curve (recon / utilisation /
    perplexity) to one or more output paths.

    Skips silently if `matplotlib` is not installed or if `history`
    does not yet contain any eval-epoch points.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    eval_eps: Iterable[int] = history.get("eval_epochs", [])
    if not eval_eps:
        return

    n_lev = sum(1 for k in history if k.startswith("util_L"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=110)

    # (a) reconstruction
    ax = axes[0]
    if history.get("train_recon"):
        ax.plot(range(1, 1 + len(history["train_recon"])),
                history["train_recon"],
                label="train", color="C0", alpha=0.7)
    ax.plot(eval_eps, history["val_recon"], "o-",
            label="val", color="C3", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("recon loss")
    ax.set_yscale("log"); ax.legend()
    ax.set_title("(a) reconstruction")

    # (b) per-level utilisation
    ax = axes[1]
    for l in range(n_lev):
        ax.plot(eval_eps, history[f"util_L{l}"], "o-", ms=3, label=f"L{l}")
    ax.set_xlabel("epoch"); ax.set_ylabel("utilisation")
    ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("(b) codebook utilisation")

    # (c) per-level perplexity
    ax = axes[2]
    for l in range(n_lev):
        ax.plot(eval_eps, history[f"ppl_L{l}"], "o-", ms=3, label=f"L{l}")
    ax.set_xlabel("epoch"); ax.set_ylabel("perplexity")
    ax.legend()
    ax.set_title("(c) effective codebook size")

    plt.tight_layout()
    for path in out_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=130)
    plt.close(fig)
