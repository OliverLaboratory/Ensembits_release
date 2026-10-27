"""Train the Ensembits tokenizer (RVQ-VAE + SFTD) on a descriptor cache.

What it does
------------
Reads a cache of per-residue descriptor multisets (N_residues, P = 10, 256),
builds the RVQ-VAE (`ensembits/tokenizer/rvqvae.py`), and trains it. For the
reported model each optimisation step does:

  1. sub-ensemble branch:  pick P' ~ U{1..10} random frames of every residue,
     encode -> z_2, quantize, decode 10 descriptors, Hungarian-matched FAPE loss
     against the P' frames                                    (`_branch`)
  2. full-ensemble branch: same with all 10 frames -> z_1     (`_branch`)
  3. loss = mean(recon_1, recon_2) + mean(vq_1, vq_2)
            + lambda * mean((z_2 - stop_grad(z_1))^2)          (SFTD)
  4. gradient clipping at norm 1, linear LR warmup (1000 steps), AdamW step;
     cosine LR annealing per epoch after the warmup.
Every 5 epochs it evaluates the validation reconstruction loss (all 10 frames),
keeps the best checkpoint, and stops after 40 epochs without improvement.
Also: k-means initialisation of every codebook before training, Gaussian noise
(sigma 0.05) on the encoder output during training, EMA codebook updates inside
the quantizer.

Flags of the reported run (template in `train/train_bioemu_misato.sh` and
`rebuttal/reproducibility.md` §1):
  --codebook-sizes 2048,128,128 --hidden-dim 256 --latent-dim 128
  --n-encoder-layers 4 --n-decoder-layers 3 --n-queries 8 --n-heads 4
  --no-standardize --random-p-range 1,10 --consistency-mse-weight 10
  --consistency-distill-fixed-max --rect-hungarian --recon-loss-type fape
  --kmeans-init --noise-std 0.05 --warmup-steps 1000 --batch-size 4096
  --lr 1e-3 --weight-decay 1e-5 --max-epochs 1000 --patience 40 --eval-every 5
  --seed 42
Everything else (FSQ, geometric pre-encoder, inverse-folding head, geom_dist_dir
loss, AminoAseed projection, rotation trick, NSVQ, warm start, bypass) is an
option for other experiments and NOT used by the reported model.

Paper
-----
  * `1_method.tex`, \subsection{Single-frame-to-token distillation}
    \label{method:SFTD}, Eq. (eq:sftd): the two-branch objective.
  * `1_method.tex`, \paragraph{Decoder and reconstruction loss.}, Eq.
    (eq:hungarian), (eq:fape): reconstruction loss (implemented in
    `ensembits/losses.py`).
  * `appendix.tex`, \subsection{RVQ-VAE} \label{app:RVQVAE}: k-means init,
    encoder-output noise, AdamW + warmup + cosine, early stopping.
  * `appendix.tex`, \subsection{Implementation details} \label{app:model-impl}:
    hyperparameters, "Descriptors are used without standardization"; the reported
    λ = 10 run (ensembits_bm_lam10_rv) stopped at epoch 475, best at epoch 435.

Origin
------
Copied from `Ensembits/scripts/train.py`. Changes: imports/paths (see the path
setup block); the FAPE loss functions were moved verbatim to
`ensembits/losses.py`; `make_run_dir` / `plot_training_curves` come from
`train/train_utils.py` (copied from `Ensembits/utils`). Training logic unchanged.

Used by the reported model: yes.

Inputs / outputs
----------------
Inputs: --desc  REPO_ROOT/output/bioemu_cath_misato_e0/cached_desc/
                esm3desc_dihedral_iupac_K16_P10.npy for the retrain (built by
                data_prep/build_bioemu_misato_descriptor.py; the current checkpoint
                used output/reviewer_3_special_corpus_MISATO/cached_desc/
                esm3desc_dihedral_K16_P10.npy)
        --splits the matching splits.json
Outputs (--out): best.pt, stats.npz, config.json, history.json, training_curve.png
(the reported λ = 10 run's outputs are copied to ckpt/bioemu_cath_misato_lam10/).
# FIXED(code_fix P10/P12/P20/P23/P27, 2026-10-16): --out must not hold a run already
#   (best.pt / config.json) unless --overwrite; config.json is written before the
#   first epoch (status "running", finished_at null) and rewritten at the end
#   (status "finished"); it also records descriptor_mode, k and
#   codebook_update_effective; an iupac corpus without layout.json or without its
#   "chain_break_cutoff" is refused (was: 2.5 Å assumed).
# FIXED(final_fix_list A6, 2026-10-19): codebook dead-code revival. Old = every epoch
#   ended with a partial batch (5,734,822 mod 4096 = 422 rows) that updated the EMA
#   codebooks and reseeded ~400 level-1 codes from 422 rows with replacement, and the
#   checkpoint was saved right after it (150-191 duplicate codes per checkpoint). New =
#   rule "ema-revival-v2" (ensembits/tokenizer/vqvae.py): training batches are full
#   (drop_last: the last N_train mod batch rows of each epoch's shuffled order are
#   skipped, a different set every epoch), dead codes are judged over a window of
#   --revival-window full batches, no revival in the last --revival-cooldown steps of
#   an epoch, and every evaluation checks the codebooks for duplicates (history.json
#   "codebook_health"; a duplicate stops the run, exit code 4, without saving best.pt).
#   config.json records "codebook_update_rule" and the health of best.pt. The code
#   before the fix: camera_ready/ensembits_code_backup_pre_revival_fix_20261019/.

Original description follows.

Train a Residual VQ-VAE tokenizer on a per-residue conformational
ensemble descriptor cache.

Inputs (paths via CLI):
    --desc PATH      .npy of shape (N_residues, P, D) — the per-residue
                       conformation set descriptors (set the value of P
                       and D as you wish; we used P=5, D=266 for the
                       fused k=3 descriptor on Cα ensembles).
    --splits PATH    .json with keys 'train', 'val' mapping to lists of
                       integer indices into the first axis of `desc`.
    --out DIR        Output directory; receives best.pt, stats.npz,
                       config.json, history.json, training_curve.png.

Outputs (in --out):
    best.pt            : RVQVAETokenizer state dict (best val recon).
    stats.npz          : per-feature standardisation {mean, std} of the
                          input descriptor.  Required at inference.
    config.json        : full hyperparameter snapshot + final metrics.
    history.json       : per-epoch trajectory (recon, vq, util, ppl).
    training_curve.png : visualisation of the trajectory.

The argparse defaults match the paper's ARCHITECTURE only (RVQ [2048, 128, 128],
hidden 256, latent 128, 4 enc / 3 dec layers, 8 queries, 4 heads, EMA 0.99,
commitment 0.5, lr 1e-3, wd 1e-5, 1000 epochs). The paper's training RECIPE is set
by train/train_bioemu_misato.sh: SFTD (--random-p-range 1,10
--consistency-mse-weight 10 --consistency-distill-fixed-max), --rect-hungarian
--recon-loss-type fape, --kmeans-init --noise-std 0.05 --warmup-steps 1000,
--batch-size 4096, --patience 40, --no-standardize. The plain defaults (patience
200, batch 512, no noise / warmup / k-means, MSE recon) do NOT reproduce it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cap CPU thread pools BEFORE importing numpy/torch so MKL / OpenBLAS /
# OpenMP / NumExpr / PyTorch all pick up the smaller pool. Without this,
# each trainer process opens `os.cpu_count()` threads per pool — with N
# concurrent trainers on a many-core box this oversubscribes by Nx and
# spikes load average without giving real throughput (the trainers are
# GPU-bound). `setdefault` so callers can still override via env.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Mirror the cap on PyTorch's intra-/inter-op pools (these read the env
# vars at import, but call the API too in case torch was already loaded
# by another import path).
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass  # already set in this process

# Path setup for this codebase: the codebase root (for `import ensembits`) and
# this folder (for `train_utils`). Originally this file lived in
# `Ensembits/scripts/` and imported `utils` from `Ensembits/utils` and the
# rectangular Hungarian loss from the research repo's `src/tokenizer`.
_CODE_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_CODE_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensembits import RVQVAETokenizer  # noqa: E402
from train_utils import make_run_dir, plot_training_curves  # noqa: E402
from ensembits.tokenizer.vqvae import (  # noqa: E402
    VectorQuantizer, CODEBOOK_UPDATE_RULE, CODEBOOK_UPDATE_RULE_DESCRIPTION, codebook_health)
from ensembits.losses import (  # noqa: E402
    rectangular_hungarian_loss, matched_pairs,
    fape_dihedral_loss, fape_hungarian_loss)


# [NOT USED by the reported model: --recon-loss-type geom_dist_dir]
def geom_dist_dir_loss(pred: "torch.Tensor", target: "torch.Tensor",
                        n_atoms: int = 48, clamp_sq: float = 225.0,
                        ) -> "torch.Tensor":
    """AminoAseed-style structural reconstruction loss on K=16 × 3-atom layouts.

    `pred`, `target` shape (..., n_atoms * 3) flattened atom positions.
    Reshaped internally to (..., n_atoms, 3) and the loss is:
        geom_dist  = mean[ clamp((‖pred_i − pred_j‖ − ‖tgt_i − tgt_j‖)², max=225) ]
        geom_dir   = mean[ ‖unit(pred_i − pred_j) − unit(tgt_i − tgt_j)‖² ]

    `clamp_sq=225` corresponds to AminoAseed's 15-Å Cartesian clamp on the
    distance residual (compute_geometric_distance in vqvae_model.py).
    """
    *batch, _ = pred.shape
    pred = pred.reshape(*batch, n_atoms, 3)
    target = target.reshape(*batch, n_atoms, 3)

    # Pairwise distance matrices (..., n_atoms, n_atoms).
    pd = torch.cdist(pred, pred)
    td = torch.cdist(target, target)
    dist_residual = (pd - td) ** 2
    geom_dist = torch.clamp(dist_residual, max=clamp_sq).mean()

    # Pairwise direction unit vectors (..., n_atoms, n_atoms, 3).
    eps = 1e-6
    p_diff = pred.unsqueeze(-2) - pred.unsqueeze(-3)
    t_diff = target.unsqueeze(-2) - target.unsqueeze(-3)
    p_dir = p_diff / p_diff.norm(dim=-1, keepdim=True).clamp_min(eps)
    t_dir = t_diff / t_diff.norm(dim=-1, keepdim=True).clamp_min(eps)
    geom_dir = ((p_dir - t_dir) ** 2).sum(dim=-1).mean()

    return geom_dist + geom_dir


# The FAPE reconstruction loss (`_N_OFFSET`, `_C_OFFSET`, `_gram_schmidt`,
# `fape_dihedral_loss`, `fape_hungarian_loss`) was moved verbatim to
# ensembits/losses.py and is imported above.
# Paper: `1_method.tex`, Decoder and reconstruction loss, Eq. (eq:hungarian),
# Eq. (eq:fape).


def _git_commit() -> str | None:
    """Commit of the code that trained the model (T1.7), or None outside a git checkout."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _code_sha256() -> str:
    """SHA-256 over the source of ensembits/ and train/ (sorted paths + contents)."""
    import hashlib
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in sorted(list((root / "ensembits").rglob("*.py")) + list((root / "train").glob("*.py"))):
        h.update(str(f.relative_to(root)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


def standardize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score over (N_residues × P) samples."""
    flat = data.reshape(-1, data.shape[-1])
    mu = flat.mean(0).astype(np.float32)
    std = flat.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((data - mu) / std).astype(np.float32), mu, std


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desc", required=True, type=Path,
                     help="(N, P, D) descriptor cache (.npy)")
    ap.add_argument("--splits", required=True, type=Path,
                     help="json with 'train' / 'val' index lists")
    ap.add_argument("--out", required=True, type=Path,
                     help="output directory for checkpoints and logs")

    # Model
    # Q13: FSQ removed from the release (was --quantizer fsq)
    ap.add_argument("--quantizer", default="rvq",
                     choices=("rvq", "none"),
                     help="Bottleneck type. 'rvq' = (residual) vector quantization "
                          "with one or more EMA codebooks (set length via "
                          "--codebook-sizes; pass a single value for a plain "
                          "single-level VQ-VAE).")
    ap.add_argument("--codebook-sizes", default="2048,128,128",
                     help="comma-separated per-level codebook sizes (RVQ depth = len). "
                          "Default matches the paper's production tokenizer. Pass a "
                          "single integer (e.g. '2048') for a plain single-level "
                          "VQ-VAE.")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--n-encoder-layers", type=int, default=4)
    ap.add_argument("--n-decoder-layers", type=int, default=3)
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--encoder-type", default="set_transformer",
                     choices=("set_transformer", "sequence"))
    ap.add_argument("--loss-type", default="hungarian",
                     choices=("hungarian", "mse"))
    # NOTE: the quantizer expects 'ema' | 'loss'. "grad" was a CLI alias that would have
    # raised inside VectorQuantizer; kept and mapped below for compatibility.
    ap.add_argument("--vq-update", default="ema", choices=("ema", "grad", "loss"))
    # Rotation trick (arXiv:2410.06424, ICLR 2025 oral): rotation+rescale surrogate for the
    # straight-through estimator. Forward value unchanged (tokens identical); the encoder
    # gradient retains the angle/magnitude relation to the codeword.
    ap.add_argument("--rotation-trick", action="store_true")
    # Materialize the train/val descriptor slices as float16 instead of float32.
    # The trainer holds the whole train split in RAM (67 GB at fp32 for the 256-D
    # dihedral cache); fp16 halves that to 33.6 GB so it fits a 62 GB box.
    # Measured round-trip error on this descriptor: max 0.015 A, 99.9th pct
    # 0.0037 A (values span +-40 A) -- ~2 orders of magnitude below the RMSF
    # signal. Batches are upcast to fp32 on the GPU, so all math is unchanged.
    ap.add_argument("--desc-dtype", default="float32", choices=("float32", "float16"))
    # Keep the whole train split resident in GPU memory instead of host RAM.
    # Default (host RAM) copies a 42 MB batch over PCIe every step -- 58 GB per
    # epoch per run -- which starves the GPU (measured 53-63% util with four
    # concurrent runs). With --desc-dtype float16 the 5.67M-row BioEmu+MISATO
    # split is 29 GB, which fits an 80 GB card alongside model+activations.
    ap.add_argument("--data-on-gpu", action="store_true")
    # Evaluate on the first N frames instead of all P. Needed for frame-count
    # ablations: a model trained at p_eff=1 should be model-selected on 1-frame
    # validation, not on the 10-frame view it never saw. 0 = use all P (default,
    # unchanged behaviour). The decoder still emits P prototypes; rect-Hungarian
    # handles the rectangular match, exactly as it does for varP branch A.
    ap.add_argument("--eval-p", type=int, default=0)
    # Warm-start weights from a previous run's best.pt (plain model state_dict).
    # NOT a full resume: optimizer state, LR schedule and epoch counter restart.
    # Exists so an interrupted long run can continue from its best checkpoint
    # instead of from scratch (see watchdog_quantfix.sh).
    ap.add_argument("--init-from", type=Path, default=None)
    # FIXED(preretrain_audit_r2 training audit, 2026-10-16): old = a killed run could only
    #   restart from scratch (--overwrite deletes best.pt) or warm-start the weights with
    #   --init-from (optimizer, schedule and epoch restart, and --kmeans-init then
    #   re-initialised the codebooks). New = --resume continues from <out>/last.pt
    #   (model incl. EMA buffers, optimizer, LR schedule, step, epoch, best, patience
    #   counter, history, RNG states), skips the k-means init, and refuses a change of
    #   code or settings.
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in --out from its last.pt (same code and settings)")
    ap.add_argument("--debug-stop-after-epoch", type=int, default=0, help=argparse.SUPPRESS)
    # NSVQ non-stationary embedding loss (arXiv:2606.11363): pulls NON-winning codes toward
    # encoder outputs with detached softmax locality weights so unused codes track encoder
    # drift. Codebook-only (zero encoder gradient); requires --vq-update loss.
    ap.add_argument("--nsvq-alpha", type=float, default=0.0)
    ap.add_argument("--nsvq-tau", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--commitment-cost", type=float, default=0.5)
    ap.add_argument("--aminoaseed-projection", default="0",
                     help="AminoAseed-style projection-head reparameterization "
                          "of the codebook (Cao et al. 2025). Pass '1' to enable "
                          "for every RVQ level, '0' to disable, or a comma-"
                          "separated list of 0/1 of length n_levels for per-"
                          "level control (e.g. '1,0,0' to enable only on L0). "
                          "When on for a level, codebook = basis @ projection "
                          "with `basis` fixed orthogonal and `projection` learned "
                          "via the VQ loss; EMA is bypassed for that level.")

    # Training
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                     help="auto-detect cuda/cpu by default")

    # Production training recipe (varP + consistency-distillation):
    ap.add_argument("--random-p-range", default=None,
                     help="MIN,MAX — sample p_eff uniformly per batch and feed only "
                          "p_eff random frames to the encoder. Enables the SFTD "
                          "single-frame-to-token distillation regime when paired "
                          "with --consistency-mse-weight.")
    ap.add_argument("--consistency-mse-weight", type=float, default=0.0,
                     help="lambda of the distillation term λ·MSE(z_a, sg[z_b]). With "
                          "--random-p-range and --consistency-distill-fixed-max the "
                          "two-branch SFTD step runs for every lambda, 0 included "
                          "(lambda = 0: both branches, no distillation term).")
    ap.add_argument("--consistency-distill-fixed-max", action="store_true",
                     help="With --consistency-mse-weight: branch B always uses the "
                          "full P frames as a fixed teacher; branch A's smaller-P "
                          "latent is pulled toward it (stop-gradient on B).")
    ap.add_argument("--warmup-steps", type=int, default=0,
                     help="Linear LR warmup from 0 to --lr over this many optimizer "
                          "steps. 0 = no warmup, just cosine annealing per epoch.")
    ap.add_argument("--freeze-encoder", action="store_true",
                     help="Freeze the encoder path (geom-attn pre-encoder + "
                          "SetTransformer) at random init; train only the "
                          "decoder + quantizer/projections. Isolates whether "
                          "collapse is driven by encoder trainability vs the "
                          "reconstruction target.")
    ap.add_argument("--noise-std", type=float, default=0.0,
                     help="Gaussian noise σ added to the encoder output during "
                          "training (regularises the codebook lookup).")
    ap.add_argument("--kmeans-init", action="store_true",
                     help="Seed each codebook from K-means centroids of encoder "
                          "outputs on a sample batch. Recommended for wider/deeper "
                          "encoders to prevent codebook collapse at init.")
    ap.add_argument("--rect-hungarian", action="store_true",
                     help="In the varP branch, use rectangular Hungarian matching "
                          "between the decoder's full P_max-frame output and the "
                          "p_eff-frame input (no slicing). Removes the implicit "
                          "positional bias of 'first p_eff outputs always penalized'. "
                          "Only affects training when --random-p-range is set.")
    ap.add_argument("--rect-hungarian-unmatched-penalty", type=float, default=0.0,
                     help="Per-element MSE penalty applied to decoder outputs that "
                          "remained unmatched in the rectangular Hungarian step. "
                          "0 = unmatched outputs are free (default option-2 behavior). "
                          ">0 = encourage every decoder slot to predict something "
                          "close to the input set.")
    ap.add_argument("--aa-labels", type=Path, default=None,
                     help="Optional flat (N,) int8 per-residue AA labels "
                          "(0..19 canonical, 20 unknown). When given with "
                          "--ifold-lambda > 0, adds an inverse-folding "
                          "auxiliary CE loss: L += λ·CE(Linear(z), AA). "
                          "Mirrors the recipe in AminoAseed's VQVAEModel.")
    ap.add_argument("--ifold-lambda", type=float, default=0.0,
                     help="Weight on the inverse-folding auxiliary CE loss. "
                          "0 disables the head entirely. Default = 0.5 in "
                          "AminoAseed; set explicitly per ablation.")
    ap.add_argument("--ifold-n-classes", type=int, default=21,
                     help="Number of AA classes including the unknown bucket. "
                          "Unknown (label == this - 1, i.e. 20) is masked out "
                          "of the CE loss but still consumes a logit slot.")
    ap.add_argument("--target-desc", type=Path, default=None,
                     help="Path to an alternate per-residue reconstruction "
                          "target cache, same (N, P, D') layout as --desc but "
                          "with a different last-dim (e.g. 144 = K=16 × 3 "
                          "atoms × 3 xyz for the AminoAseed geom_dist+geom_dir "
                          "recipe). When given, the decoder's output_dim is "
                          "set to D' and the reconstruction loss compares the "
                          "decoder output against this target instead of the "
                          "raw input descriptor.")
    ap.add_argument("--recon-loss-type", default="default",
                     choices=("default", "geom_dist_dir", "fape"),
                     help="default: model's built-in MSE/Hungarian on the "
                          "raw input descriptor (or --target-desc when given). "
                          "geom_dist_dir: AminoAseed-style pairwise-distance + "
                          "pairwise-unit-direction MSE on the K=16 × 3-atom "
                          "structure produced by reshaping the decoder's "
                          "(P, 144) output as (P, 48, 3) atom positions. "
                          "fape: Frame-Aligned Point Error on the reconstructed "
                          "16 neighbor frames (256-D dihedral-augmented ESM3 "
                          "descriptor: per neighbor [R(9)|t(3)|phi/psi sincos(4)]) "
                          "+ a --dihedral-weight-scaled sin/cos term. Autoencode "
                          "(no --target-desc).")
    ap.add_argument("--dihedral-weight", type=float, default=3.0,
                     help="(recon-loss-type fape) weight on the dihedral sin/cos "
                          "MSE term relative to the FAPE (Å) term.")
    ap.add_argument("--fape-clamp", type=float, default=10.0,
                     help="(recon-loss-type fape) Cartesian clamp (Å) on the FAPE.")
    ap.add_argument("--neighbor-frame-only", action="store_true",
                     help="Slice a 256-D dihedral-augmented descriptor down to the "
                          "frame-only 192-D (drop each neighbor's 4-D dihedral): "
                          "reshape (N,P,16,16) -> [...,:12] -> (N,P,192). For the "
                          "matched 192-D baseline on the same cache/splits.")
    ap.add_argument("--geom-attn-encoder", action="store_true",
                     help="Prepend a geometric-attention pre-encoder over the "
                          "K=16 nearest neighbors of each (residue, frame). "
                          "Requires --desc to be the K=16 ESM3 descriptor "
                          "(input_dim must be 192 = 16 × 12). The 192-D is "
                          "reshaped to (16, 12) and treated as 16 relative-"
                          "Affine3D tokens. Replaces our static-concat input "
                          "to the SetTransformer with a learned 2-layer "
                          "geometric-attention block (matching AminoAseed / "
                          "ESM3 structure encoder architecture). Encoder "
                          "input becomes --geom-attn-d-model wide; decoder "
                          "output stays 192-D, so recon target is the "
                          "unchanged original descriptor (no self-reference).")
    ap.add_argument("--geom-attn-d-model", type=int, default=256,
                     help="d_model for the geometric pre-encoder. Default "
                          "256 matches our existing hidden_dim.")
    ap.add_argument("--geom-attn-n-layers", type=int, default=2,
                     help="Number of geometric-attention layers (AminoAseed "
                          "default is 2).")
    ap.add_argument("--geom-attn-n-heads", type=int, default=4)
    ap.add_argument("--geom-attn-v-heads", type=int, default=8,
                     help="v_heads for the per-head rotation/distance attention "
                          "biases. 8 matches AminoAseed.")
    ap.add_argument("--ifold-head-input", default="pre_vq",
                     choices=("pre_vq", "decoder_penult"),
                     help="Where to attach the inverse-folding head. "
                          "'pre_vq': head reads the encoder output z (pre-VQ); "
                          "gradient only constrains the encoder (simplest). "
                          "'decoder_penult': head reads the decoder MLP's "
                          "penultimate hidden vector via a forward-pre-hook on "
                          "its final Linear — matches AminoAseed's recipe "
                          "(StructTokenBench vqvae_model.py:572) and the existing "
                          "submission_exp/src/train_ensembits_ifold.py. Gradient "
                          "flows back through the STE → codebook → encoder, "
                          "directly pressuring the tokens to carry sequence info.")
    ap.add_argument("--no-standardize", action="store_true",
                     help="Skip the per-feature z-score on the input descriptors. "
                          "Use this for SE(3)-structured descriptors like ESM3 "
                          "(192-D = 16 neighbours × 12 Affine3D components) where "
                          "per-dim whitening breaks the rotation-matrix manifold "
                          "and mixes rotation/translation scales artificially. "
                          "Stats.npz is still written (with identity mean/std) so "
                          "downstream inference code that always normalizes by "
                          "stats remains correct.")
    # ---- added 2026-10-15 (camera-ready code fixes) --------------------------------
    # code_fix §1: dihedral sign convention of the descriptor cache. Must match the
    # cache (checked against its layout.json when present) and is stored in config.json
    # so `ensembits.featurize.load_model` computes descriptors the same way later.
    ap.add_argument("--dihedral-convention", default="iupac",
                    choices=("iupac", "legacy_plus180"),
                    help="Dihedral convention of --desc. 'legacy_plus180' = caches built "
                         "before the code_fix §1 fix (all checkpoints before 2026-10-15).")
    # code_fix C1: residual of the RVQ. "standard" = rho_l = rho_{l-1} - sg[q_l] (all
    # levels' commitment terms reach the encoder); "legacy" = every checkpoint before
    # 2026-10-15 (levels 2-3 commitment gives the encoder no gradient).
    ap.add_argument("--rvq-residual", default="standard", choices=("standard", "legacy"))
    # code_fix C9: which SFTD branch updates the codebooks (EMA + dead-code revival).
    # "full" = only the full-ensemble branch (all P_max frames); "both" = both
    # branches, i.e. twice per step (every checkpoint trained before 2026-10-15).
    ap.add_argument("--codebook-update", default="full", choices=("full", "both"))
    # code_fix §2: cost of the Hungarian matching inside the FAPE loss.
    ap.add_argument("--hungarian-cost", default="loss", choices=("loss", "l2"),
                    help="'loss' = match on the per-pair FAPE + dihedral loss (paper, "
                         "Eq. eq:hungarian); 'l2' = squared L2 between descriptors "
                         "(checkpoints trained before the fix).")
    # FIXED(code_fix P10, 2026-10-16): old = an --out that already held a run was
    #   silently overwritten; new = refuse an --out that contains best.pt or
    #   config.json unless --overwrite is given.
    # FIXED(final_fix_list A6, 2026-10-19): parameters of the revival rule
    #   "ema-revival-v2" (not a switch: the old rule is gone).
    # FIXED(final_fix_list A6 training audits, 2026-10-20): window schedule. 100 full
    #   batches (the EMA time constant 1/(1-0.99)) in the first 2 epochs, for a fast
    #   recovery from the warm-up collapse; 400 afterwards (concept audit: a 100-batch
    #   window resets ~1.4-2.2 rare but live level-1 codes per check, 400 ~0.05).
    #   Old fix: 100 throughout.
    ap.add_argument("--revival-window", type=int, default=400,
                    help="dead-code check after this many full training batches (after the "
                         "first --revival-window-early-epochs epochs); a code with no "
                         "assignment over the window is reseeded")
    ap.add_argument("--revival-window-early", type=int, default=100,
                    help="revival window during the first --revival-window-early-epochs epochs")
    ap.add_argument("--revival-window-early-epochs", type=int, default=2,
                    help="number of initial epochs that use --revival-window-early")
    ap.add_argument("--revival-cooldown", type=int, default=50,
                    help="no dead-code revival in the last N training steps of every "
                         "epoch, so evaluated / saved codebooks hold no just-reseeded "
                         "codes (the check is deferred to the next allowed step)")
    # FIXED(final_fix_list A6 training audits, 2026-10-20): a configuration in which the
    #   SFTD sub-ensemble branch updates the codebooks (--codebook-update both, a
    #   two-branch step without --consistency-distill-fixed-max, or a variable-P run with
    #   only the sub-ensemble branch) is refused unless this option is given. Old =
    #   accepted silently (codebook_update_effective "both" in config.json).
    ap.add_argument("--allow-subensemble-codebook-update", action="store_true",
                    help="Accept a run in which sub-ensemble (P' < P_max) latents update the "
                         "codebooks (EMA and revival). Not the paper's recipe, which updates "
                         "them from the full-ensemble branch only.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow an --out that already contains best.pt or config.json "
                         "(the old run's best.pt / history.json / ifold_head.pt / "
                         "geom_encoder.pt are removed before training).")
    return ap.parse_args()


def _set_codebook_update(model, flag: bool) -> None:
    """Turn the codebook bookkeeping (EMA update, dead-code revival) of every
    VectorQuantizer level on or off (code_fix C9)."""
    for m in model.modules():
        if isinstance(m, VectorQuantizer):
            m.update_codebook = flag


def _vq_modules(model) -> list:
    """Every VectorQuantizer of the model (RVQ levels; empty for FSQ)."""
    return [m for m in model.modules() if isinstance(m, VectorQuantizer)]


def _set_revival_allowed(model, flag: bool) -> None:
    """FIXED(final_fix_list A6, 2026-10-19): new. Allow or block the dead-code check
    of every level (blocked in the last --revival-cooldown steps of an epoch)."""
    for m in _vq_modules(model):
        m.revival_allowed = flag


def main() -> int:
    args = parse_args()
    # FIXED(code_fix P10, 2026-10-16): old = any existing --out was reused and its
    #   files overwritten one by one (a new run could mix with an old best.pt);
    #   new = refuse unless --overwrite, which first removes the old run's artifacts.
    _old_run = [f for f in ("best.pt", "config.json") if (args.out / f).exists()]
    _resume_ck = None
    if args.resume:
        if args.overwrite:
            raise SystemExit("--resume and --overwrite exclude each other")
        if not (args.out / "last.pt").exists():
            raise SystemExit(f"--resume: {args.out / 'last.pt'} does not exist")
        _resume_ck = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        _old_run = []                       # continue this run, delete nothing
    if _old_run and not args.overwrite:
        raise SystemExit(f"--out {args.out} already holds a run ({', '.join(_old_run)}); "
                         f"choose a new --out (launcher: a new RUN name) or pass --overwrite")
    _started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    _resumed_at: list[str] = []

    # code_fix §1: refuse to train on a cache whose dihedral convention differs from
    # --dihedral-convention (layout.json written by data_prep/build_bioemu_misato_descriptor.py).
    # The corpus settings in layout.json (convention, chain-break cutoff, split unit,
    # source caches, MISATO parser version) are copied into config.json; the cutoff
    # also as top-level "chain_break_cutoff" so tokenization uses the same descriptor.
    from ensembits.descriptor import resolve_chain_break_cutoff
    _layout = Path(args.desc).parent / "layout.json"
    _corpus_settings = {}
    if _layout.exists():
        _lj = json.loads(_layout.read_text())
        _cache_conv = _lj.get("dihedral_convention", "legacy_plus180")
        if _cache_conv != args.dihedral_convention:
            raise SystemExit(f"--dihedral-convention {args.dihedral_convention} but the cache "
                             f"{args.desc} was built with {_cache_conv} (layout.json)")
        _corpus_settings = {k: v for k, v in _lj.items() if k != "rows"}
        # FIXED(code_fix P27, 2026-10-16): old = an iupac layout without
        #   "chain_break_cutoff" fell back to resolve_chain_break_cutoff("auto") = 2.5 Å;
        #   new = refuse (the cutoff must come from the corpus build). legacy_plus180
        #   layouts without the key keep the old behaviour (None, no break detection).
        if args.dihedral_convention == "iupac" and "chain_break_cutoff" not in _lj:
            raise SystemExit(f"{_layout} has no 'chain_break_cutoff'; an iupac corpus must "
                             f"record it (rebuild with data_prep/build_bioemu_misato_descriptor.py)")
        # Old layouts have no key: legacy (no detection); new ones record the value.
        chain_break_cutoff = _lj.get("chain_break_cutoff",
                                     resolve_chain_break_cutoff("auto", _cache_conv)
                                     if "dihedral_convention" in _lj else None)
    else:
        # FIXED(code_fix P27, 2026-10-16): old = without layout.json an iupac run
        #   assumed the convention and a 2.5 Å cutoff (warning only); new = refuse for
        #   iupac. legacy_plus180 keeps the warning and cutoff None.
        if args.dihedral_convention == "iupac":
            raise SystemExit(f"no layout.json next to {args.desc}: cannot verify the dihedral "
                             f"convention or read the chain-break cutoff of an iupac corpus "
                             f"(build it with data_prep/build_bioemu_misato_descriptor.py)")
        print(f"  [warn] no layout.json next to {args.desc}; cannot verify the dihedral "
              f"convention (assuming {args.dihedral_convention}) or the chain-break cutoff")
        chain_break_cutoff = resolve_chain_break_cutoff("auto", args.dihedral_convention)

    # FIXED(code_fix P10, 2026-10-16): with --overwrite the old run's artifacts are
    #   removed (after the corpus checks above, so a refused start deletes nothing);
    #   config.json is rewritten below. Old = files overwritten one by one.
    if _old_run:
        # Q19 (2026-10-16): config.json too, so the folder never shows the old run's
        #   "finished" config while the new one loads its data.
        for f in ("best.pt", "last.pt", "history.json", "ifold_head.pt", "geom_encoder.pt",
                  "config.json", "training_curve.png"):
            (args.out / f).unlink(missing_ok=True)
        print(f"  --overwrite: replacing the run in {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    # code_fix §2: the FAPE + Hungarian loss with the chosen matching cost.
    def _fape_hungarian(pred, target, dihedral_weight, clamp):
        return fape_hungarian_loss(pred, target, dihedral_weight, clamp,
                                   match_cost=args.hungarian_cost)
    # Per-run plot directory under output/, named after the wall-clock
    # time the run started (plus the --out basename for human readability).
    run_dir = make_run_dir(label=args.out.name)
    plot_paths = (args.out / "training_curve.png", run_dir / "training_curve.png")
    print(f"[run] plots → {run_dir.relative_to(run_dir.parent.parent)}/")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    codebook_sizes = [int(x) for x in args.codebook_sizes.split(",")]

    # Parse --aminoaseed-projection into a per-level bool list.
    aap_raw = args.aminoaseed_projection.strip()
    if "," in aap_raw:
        aap_flags = [bool(int(x)) for x in aap_raw.split(",")]
        if len(aap_flags) != len(codebook_sizes):
            raise SystemExit(
                f"--aminoaseed-projection length {len(aap_flags)} does not "
                f"match n_levels={len(codebook_sizes)}; pass '0', '1', or a "
                f"comma list of length {len(codebook_sizes)}.")
    else:
        aap_flags = [bool(int(aap_raw))] * len(codebook_sizes)

    print(f"[1/4] Loading descriptors {args.desc} and splits {args.splits}")
    # Memory-mapped read: the OS shares page cache across multiple training
    # processes that open the same file, and a process only materializes the
    # train/val slice — saving ~D × N residue floats per process for caches
    # larger than typical fit-in-RAM working sets (e.g. the 299 GB
    # aminoaseed_K16_P10.npy 1024-D cache).
    data = np.load(args.desc, mmap_mode="r")            # (N, P, D)
    splits = json.loads(args.splits.read_text())
    if "train" not in splits or "val" not in splits:
        raise SystemExit(f"--splits must contain 'train' and 'val' keys; "
                          f"got {list(splits)}")

    # Optional separate reconstruction target (e.g. K=16 N/CA/C coords for
    # the AminoAseed geom_dist+geom_dir recipe).
    target_data = None
    if args.target_desc is not None:
        print(f"  loading recon target from {args.target_desc}")
        target_data = np.load(args.target_desc, mmap_mode="r")
        if target_data.shape[0] != data.shape[0] or target_data.shape[1] != data.shape[1]:
            raise SystemExit(
                f"--target-desc shape {target_data.shape} doesn't match "
                f"--desc shape {data.shape} on axes 0,1")
        print(f"  recon target shape={target_data.shape}  dtype={target_data.dtype}")
    if args.recon_loss_type == "geom_dist_dir" and args.target_desc is None:
        raise SystemExit("--recon-loss-type geom_dist_dir requires --target-desc "
                          "to a (N, P, 144) coords cache.")
    # Materialize only the train/val subsets from the (possibly huge) mmap, then
    # standardize using TRAIN statistics. Standardizing the full array before
    # subsetting would materialize every row in RAM — fine for the 50/192-D
    # caches but ~300 GB for the 1024-D AminoAseed feature cache.
    D_ = data.shape[-1]
    # FIXED(code_fix P20, 2026-10-16): old = config.json did not say which descriptor
    #   the model was trained on; new = "descriptor_mode" and "k" (neighbour count)
    #   are recorded. k comes from layout.json ("k"/"K") when the corpus records it,
    #   else from the descriptor width: D = k x 16 numbers per neighbour
    #   ([R(9) | t(3) | phi/psi sin/cos(4)], ensembits/descriptor.py PER_NB).
    #   A width that is not a multiple of 16 is not this descriptor: both are None.
    from ensembits.descriptor import PER_NB as _PER_NB
    _k_layout = None
    if _layout.exists():
        _k_layout = _lj.get("k", _lj.get("K"))
    # FIXED(preretrain_audit_r2 Q19 / R2-C7, 2026-10-16): old = k inferred from the width
    #   when layout.json lacked it; new = an iupac corpus must record k (the corpus
    #   builder writes it since 2026-10-16), and it must agree with the width.
    if args.dihedral_convention == "iupac" and _layout.exists() and _k_layout is None:
        raise SystemExit(f"{_layout} records no neighbour count k; rebuild the corpus with "
                         f"data_prep/build_bioemu_misato_descriptor.py")
    if D_ % _PER_NB == 0:
        desc_k = D_ // _PER_NB
        if _k_layout is not None and int(_k_layout) != desc_k:
            raise SystemExit(f"layout.json records k={_k_layout} but the descriptor width "
                             f"{D_} implies k={desc_k} ({_PER_NB} numbers per neighbour)")
        descriptor_mode = ("esm3desc_dihedral_frame_only" if args.neighbor_frame_only
                           else "esm3desc_dihedral")
    else:
        desc_k, descriptor_mode = None, None
        print(f"  [warn] descriptor width {D_} is not k x {_PER_NB}: descriptor_mode / k "
              f"recorded as null in config.json")
    _desc_np_dtype = np.float16 if args.desc_dtype == "float16" else np.float32

    def _gather_chunked(index, dtype, chunk=100_000):
        """Materialize mmap rows in blocks, casting per block.

        `np.asarray(data[idx], dtype=...)` would build the full float32 result
        first and only then cast, peaking at 67+34 GB for the 256-D cache.
        Blocked reads keep the transient at ~1 GB. Index lists are sorted
        ascending, so this is a forward scan over the file (sequential I/O).
        """
        index = np.asarray(index)
        out = np.empty((len(index),) + data.shape[1:], dtype=dtype)
        for s0 in range(0, len(index), chunk):
            out[s0:s0 + chunk] = data[index[s0:s0 + chunk]]
        return out

    train_data = _gather_chunked(splits["train"], _desc_np_dtype)
    val_data   = _gather_chunked(splits["val"],   _desc_np_dtype)
    if args.desc_dtype == "float16":
        print(f"  --desc-dtype float16: train {train_data.nbytes/1e9:.1f} GB + "
              f"val {val_data.nbytes/1e9:.1f} GB resident "
              f"(fp32 would be {2*(train_data.nbytes+val_data.nbytes)/1e9:.1f} GB)")
    if args.neighbor_frame_only:
        # (N, P, 16*16) -> drop per-neighbor dihedral -> (N, P, 16*12)
        def _frame_only(a):
            n, p, _ = a.shape
            return np.ascontiguousarray(a.reshape(n, p, 16, 16)[..., :12].reshape(n, p, 16 * 12))
        train_data = _frame_only(train_data); val_data = _frame_only(val_data)
        D_ = train_data.shape[-1]
        print(f"  --neighbor-frame-only: sliced to frame-only D={D_}")
    # Paper: `appendix.tex` app:model-impl — "Descriptors are used without
    # standardization" (reported run: --no-standardize -> identity stats.npz).
    if args.no_standardize:
        print("  --no-standardize: feeding raw descriptors (no z-score). "
              "Writing identity stats.npz for downstream consistency.")
        mu = np.zeros(D_, dtype=np.float32)
        std = np.ones(D_, dtype=np.float32)
    else:
        flat = train_data.reshape(-1, D_)
        # float64 accumulation: np.mean/std on a float16 array accumulate in
        # float16 by default, which silently loses the statistic over 65M rows.
        mu = flat.mean(0, dtype=np.float64).astype(np.float32)
        std = flat.std(0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        # in-place, blocked, via fp32 (fp16 arrays cannot be updated in place
        # with fp32 operands under numpy's same-kind casting rule)
        def _standardize_inplace(arr, chunk=100_000):
            for s0 in range(0, len(arr), chunk):
                blk = arr[s0:s0 + chunk].astype(np.float32)
                blk -= mu; blk /= std
                arr[s0:s0 + chunk] = blk
        _standardize_inplace(train_data); _standardize_inplace(val_data)
        print(f"  standardized via train-split statistics (D={D_})")
    np.savez(args.out / "stats.npz", mean=mu, std=std)
    N_tr, P, D = train_data.shape
    N_val = val_data.shape[0]
    print(f"  train={N_tr}  val={N_val}  P={P}  D={D}  device={device}")

    # Optional geometric-attention pre-encoder over K=16 NN tokens.
    # Reshapes input from (B, P, D=192) → (B, P, 16, 12) → (B, P, d_model);
    # encoder/RVQ then operate in d_model-D, decoder reconstructs the
    # original 192-D (so recon target is the unchanged input descriptor).
    geom_encoder = None
    if args.geom_attn_encoder:
        if D != 192:
            raise SystemExit(
                f"--geom-attn-encoder requires --desc with input_dim=192 "
                f"(K=16 × 12 relative-Affine3D), got input_dim={D}")
        from ensembits.tokenizer.geom_attn_encoder import GeometricKNNEncoder
        geom_encoder = GeometricKNNEncoder(
            d_model=args.geom_attn_d_model,
            n_heads=args.geom_attn_n_heads,
            v_heads=args.geom_attn_v_heads,
            n_layers=args.geom_attn_n_layers,
        ).to(device)
        n_geom = sum(p.numel() for p in geom_encoder.parameters())
        print(f"\n[2a/4] geometric pre-encoder: K=16 → "
              f"d_model={args.geom_attn_d_model}  n_layers={args.geom_attn_n_layers}  "
              f"+{n_geom:,} params (uses StructTokenBench geom_attention)")
        rvq_input_dim = args.geom_attn_d_model
        rvq_output_dim = D                       # reconstruct original 192-D
    else:
        rvq_input_dim = D
        rvq_output_dim = D

    # If a separate recon target is given, override output_dim to match it.
    target_train = target_val = None
    if target_data is not None:
        rvq_output_dim = target_data.shape[-1]
        target_train = torch.tensor(target_data[splits["train"]], dtype=torch.float32)
        target_val   = torch.tensor(target_data[splits["val"]],   dtype=torch.float32).to(device)
        print(f"  recon target sliced to train ({target_train.shape}) and "
              f"val ({target_val.shape}); decoder output_dim={rvq_output_dim}")

    # Q13: FSQ branch removed; RVQ only.
    print(f"\n[2/4] Building RVQ model: codebook_sizes={codebook_sizes} "
          f"n_enc={args.n_encoder_layers} n_dec={args.n_decoder_layers} "
          f"Q={args.n_queries} H={args.n_heads}  "
          f"in={rvq_input_dim} out={rvq_output_dim}")
    model = RVQVAETokenizer(
        input_dim=rvq_input_dim, output_dim=rvq_output_dim, num_prototypes=P,
        codebook_sizes=codebook_sizes,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        commitment_cost=args.commitment_cost,
        encoder_type=args.encoder_type, loss_type=args.loss_type,
        vq_update=("loss" if args.vq_update in ("grad", "loss") else "ema"),
        ema_decay=args.ema_decay,
        rotation_trick=args.rotation_trick,
        nsvq_alpha=args.nsvq_alpha, nsvq_tau=args.nsvq_tau,
        n_queries=args.n_queries, n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        aminoaseed_projection=aap_flags,
        bypass_quantizer=(args.quantizer == "none"),
        residual_mode=args.rvq_residual,
    ).to(device)
    if args.quantizer == "none":
        print("  QUANTIZER BYPASSED (--quantizer none): plain continuous "
              "autoencoder, VQ loss == 0, codebooks untrained. Stage-1 "
              "warm-start run; feed best.pt to a real run via --init-from.")
    if any(aap_flags):
        print(f"  AminoAseed projection-head: per-level "
              f"{[int(x) for x in aap_flags]}")
    if args.init_from is not None:
        if not args.init_from.exists():
            raise SystemExit(f"--init-from {args.init_from} does not exist")
        _sd = torch.load(args.init_from, map_location="cpu")
        _missing, _unexpected = model.load_state_dict(_sd, strict=False)
        if _missing or _unexpected:
            print(f"  WARNING --init-from partial load: missing={list(_missing)[:4]} "
                  f"unexpected={list(_unexpected)[:4]}")
        print(f"  warm-started weights from {args.init_from} "
              f"(optimizer/LR/epoch counter restart from zero)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    # FIXED(final_fix_list A6, 2026-10-19): drop_last. Every training batch has
    #   `train_bs` rows; the last N_tr mod train_bs rows of each epoch's shuffled order
    #   are skipped (a fresh permutation every epoch, so no row is excluded for good).
    #   Old = a final partial batch (422 rows for the BioEmu+MISATO corpus) that
    #   updated the codebooks and reseeded ~400 codes from it.
    train_bs = min(args.batch_size, N_tr)
    n_full_batches = N_tr // train_bs
    # FIXED(final_fix_list A6 training audits, 2026-10-20): argument checks; a warning
    #   (also recorded in config.json "revival_warnings") when an epoch is too short for
    #   the cooldown / window, i.e. revival would be rare or never happen.
    if min(args.revival_window, args.revival_window_early) < 1:
        raise SystemExit("--revival-window and --revival-window-early must be >= 1")
    if args.revival_cooldown < 0 or args.revival_window_early_epochs < 0:
        raise SystemExit("--revival-cooldown and --revival-window-early-epochs must be >= 0")
    revival_warnings = []
    if n_full_batches <= args.revival_cooldown:
        revival_warnings.append(f"{n_full_batches} batches per epoch <= cooldown "
                                f"{args.revival_cooldown}: no dead-code revival at all")
    else:
        for _w in sorted({args.revival_window_early, args.revival_window}):
            if n_full_batches < _w + args.revival_cooldown:
                revival_warnings.append(f"{n_full_batches} batches per epoch < window {_w} + "
                                        f"cooldown {args.revival_cooldown}: checks deferred "
                                        f"across epochs")
    for _msg in revival_warnings:
        print(f"  [warn] revival: {_msg}")
    for _vq in _vq_modules(model):
        _vq.full_batch_size = train_bs
    print(f"  codebook update rule {CODEBOOK_UPDATE_RULE}: {n_full_batches} full batches of "
          f"{train_bs} per epoch ({N_tr - n_full_batches * train_bs} rows skipped per epoch, "
          f"reshuffled), revival window {args.revival_window_early} batches in epochs "
          f"1-{args.revival_window_early_epochs}, {args.revival_window} afterwards, no revival "
          f"in the last {args.revival_cooldown} steps of an epoch")

    # Optional inverse-folding auxiliary head.
    # 'pre_vq':         head reads z (encoder output, pre-quantization).
    # 'decoder_penult': head reads the decoder MLP's penultimate hidden
    #                    via a forward-pre-hook on its final Linear — matches
    #                    AminoAseed's recipe (StructTokenBench
    #                    vqvae_model.py:572) and the existing
    #                    train_ensembits_ifold.py. CE backprop flows through
    #                    STE → codebook → encoder, directly pressuring tokens.
    ifold_head = None
    aa_tr = aa_va = None
    ifold_pen_cache: dict[str, torch.Tensor] = {}
    if args.ifold_lambda > 0:
        if args.aa_labels is None:
            raise SystemExit("--ifold-lambda > 0 requires --aa-labels")
        if args.ifold_head_input == "pre_vq":
            head_in = args.latent_dim
        else:  # decoder_penult — find the in_features of the decoder's last Linear
            dec_mlp = model.decoder.mlp
            last_lin_idx = next(i for i in range(len(dec_mlp) - 1, -1, -1)
                                  if isinstance(dec_mlp[i], nn.Linear))
            head_in = dec_mlp[last_lin_idx].in_features

            # The pre-hook captures the input to the final Linear; that input
            # is `(B, hidden_dim)` and is the analogue of AminoAseed's
            # `decoded_states["last_hidden_state"]`.
            def _ifold_pen_hook(_module, args_tuple):
                ifold_pen_cache["h"] = args_tuple[0]
            dec_mlp[last_lin_idx].register_forward_pre_hook(_ifold_pen_hook)
        ifold_head = nn.Linear(head_in, args.ifold_n_classes).to(device)
        n_ifold = sum(p.numel() for p in ifold_head.parameters())
        print(f"  inverse-folding head: Linear({head_in} → "
              f"{args.ifold_n_classes})  +{n_ifold:,} params  "
              f"λ={args.ifold_lambda}  input={args.ifold_head_input}")
        print(f"  loading aa_labels from {args.aa_labels}")
        aa_flat = np.load(args.aa_labels)
        if aa_flat.shape[0] != data.shape[0]:
            raise SystemExit(
                f"aa_labels length {aa_flat.shape[0]} != desc rows "
                f"{data.shape[0]}")
        aa_tr = torch.as_tensor(aa_flat[splits["train"]], dtype=torch.long)
        aa_va = torch.as_tensor(aa_flat[splits["val"]], dtype=torch.long).to(device)
        n_unk = int((aa_tr >= 20).sum())
        print(f"  aa_labels train: known={int((aa_tr < 20).sum()):,} "
              f"unknown={n_unk:,} ({100*n_unk/aa_tr.numel():.2f}%)")
    # Optional encoder freeze: hold the encoder path (geom-attn pre-encoder +
    # SetTransformer model.encoder) at its random init and train only the
    # quantizer/projections + decoder. Isolates whether the collapse is driven
    # by the *trainable* encoder finding a constant-output shortcut: a frozen
    # encoder maps each residue to a fixed (input-dependent) latent that cannot
    # contract to a constant, so if utilization stays high the collapse was
    # encoder-trainability-driven; if it still collapses, the recon target is.
    if args.freeze_encoder:
        n_frozen = 0
        for p in model.encoder.parameters():
            p.requires_grad_(False); n_frozen += p.numel()
        if geom_encoder is not None:
            for p in geom_encoder.parameters():
                p.requires_grad_(False); n_frozen += p.numel()
        print(f"  --freeze-encoder: froze {n_frozen:,} encoder params "
              f"(geom-attn pre-encoder + SetTransformer); training decoder + "
              f"quantizer only")

    opt_params = [p for p in model.parameters() if p.requires_grad]
    if ifold_head is not None:
        opt_params = opt_params + [p for p in ifold_head.parameters() if p.requires_grad]
    if geom_encoder is not None:
        opt_params = opt_params + [p for p in geom_encoder.parameters() if p.requires_grad]
    # Paper: `appendix.tex` app:model-impl / app:RVQVAE — AdamW, lr 1e-3, weight
    # decay 1e-5; cosine annealing towards 1e-6 over a horizon of max_epochs
    # (1000) epochs. Early stopping ended the reported λ = 10 run at epoch 475,
    # when the LR was still ~6.5e-4 (the schedule never reached 1e-6).
    # FIXED(require_fix #17 / D.19, tex pass 2026-10-26): appendix says AdamW.
    opt = torch.optim.AdamW(opt_params, lr=args.lr,
                             weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.max_epochs, 1e-6)

    # Zero-copy bridge: torch.from_numpy shares memory with the numpy array
    # (vs torch.tensor which copies). Saves ~64 GB peak on the 256-D cache
    # — the SLURM mem cgroup includes mmap pages, so the double-copy version
    # tripped OOM even with 512 GB allocated.
    # Respect --desc-dtype here: forcing float32 would allocate a SECOND full copy
    # (65 GB for the 256-D cache) on top of the fp16 arrays and defeat the purpose
    # -- that combination OOM-killed this run at 57 GB anon-rss. The gather already
    # returns C-contiguous arrays of the requested dtype, so with matching dtype
    # np.ascontiguousarray is a no-op and no copy is made. Batches are upcast to
    # fp32 individually at the use sites below (.float() after .to(device)).
    train_data = np.ascontiguousarray(train_data, dtype=_desc_np_dtype)
    val_data   = np.ascontiguousarray(val_data,   dtype=_desc_np_dtype)
    tt = torch.from_numpy(train_data)
    if args.data_on_gpu:
        tt = tt.to(device)          # indices are generated on tt.device below
    vt = torch.from_numpy(val_data).to(device)
    print(f"  resident tensors: train {tuple(tt.shape)} {tt.dtype} on {tt.device} "
          f"({tt.element_size()*tt.nelement()/1e9:.1f} GB), "
          f"val {vt.dtype} ({vt.element_size()*vt.nelement()/1e9:.1f} GB on {device})")

    # Optional encoder-output noise (training-only).
    if args.noise_std > 0:
        _noise_std = float(args.noise_std)

        # Paper: `appendix.tex` app:RVQVAE — "During training we also add Gaussian
        # noise with standard deviation 0.05 to the encoder output z before
        # quantization" (reported run: --noise-std 0.05). Training mode only.
        def _noise_hook(_module, _inp, out_z):
            if _module.training:
                return out_z + _noise_std * torch.randn_like(out_z)
            return out_z

        model.encoder.register_forward_hook(_noise_hook)
        print(f"  encoder-output noise σ = {args.noise_std}")

    # Optional K-means codebook init from encoder outputs on a sample batch.
    if args.kmeans_init and args.quantizer == "none":
        print("  skipping --kmeans-init: quantizer is bypassed (no codebooks in use)")
    elif args.kmeans_init and _resume_ck is not None:
        print("  skipping --kmeans-init: --resume restores the trained codebooks")
    elif args.kmeans_init:
        sample_bs = min(16384, N_tr)
        sample_idx = torch.randperm(N_tr, device=tt.device)[:sample_bs]
        with torch.no_grad():
            x_sample = tt[sample_idx].to(device).float()
            # If a learned geom-attn pre-encoder is active, the model.encoder
            # expects d_model-D features, not the raw D-D descriptor. Apply
            # the pre-encoder first.
            if geom_encoder is not None:
                x_sample_neigh = x_sample.view(*x_sample.shape[:2], 16, 12)
                x_sample = geom_encoder(x_sample_neigh)
            z_sample = model.encoder(x_sample)
        z_residual = z_sample
        for lvl, vq in enumerate(model.vq_levels):
            # Paper: `appendix.tex` app:RVQVAE — k-means on 16,384 sampled training
            # residues, level by level: level 1 on the encoder outputs, each later
            # level on the residuals left by the levels before it.
            print(f"  kmeans_init L{lvl} (M={vq.num_codes}, {sample_bs} samples)")
            vq.kmeans_init(z_residual)
            cb = vq.codebook.weight
            dists = torch.cdist(z_residual.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
            z_residual = z_residual - cb[dists.argmin(-1)]

    # Parse varP range once (None = no varP, just standard training).
    random_p_range: tuple[int, int] | None = None
    # FIXED(preretrain_audit_r2 Q4, 2026-10-16): old = the full-ensemble branch ran only
    #   when lambda > 0, so lambda = 0 trained the P' branch alone and updated the
    #   codebooks from sub-ensemble latents (not Eq. eq:sftd with lambda = 0). New = with
    #   --consistency-distill-fixed-max both branches run for every lambda; lambda = 0
    #   only drops the distillation term (the E2 "distillation off" row).
    _two_branch = bool(args.random_p_range) and (args.consistency_mse_weight > 0
                                                 or args.consistency_distill_fixed_max)
    if args.random_p_range:
        pmin, pmax = (int(x) for x in args.random_p_range.split(","))
        if not (1 <= pmin <= pmax <= P):
            raise SystemExit(f"--random-p-range {pmin}..{pmax} out of bounds for P={P}")
        random_p_range = (pmin, pmax)
        print(f"  variable-P training: p_eff ~ U({pmin},{pmax})  "
              f"(eval at P={args.eval_p or P})")
        if _two_branch:
            mode = "fixed-max" if args.consistency_distill_fixed_max else "symmetric"
            print(f"  consistency-MSE weight = {args.consistency_mse_weight}  "
                  f"({mode} distillation, two branches per step)")

    # FIXED(code_fix P23, 2026-10-16): old = config.json recorded only the requested
    #   --codebook-update; new = also "codebook_update_effective": "full" only when the
    #   full-ensemble-only update path below is active (SFTD two-branch step with
    #   --codebook-update full and --consistency-distill-fixed-max; since Q4 any lambda,
    #   0 included), "both" when two branches update the
    #   codebooks, "single" when each step has one branch (no SFTD).
    if random_p_range is not None and _two_branch:
        codebook_update_effective = ("full" if (args.codebook_update == "full"
                                                and args.consistency_distill_fixed_max)
                                     else "both")
    else:
        codebook_update_effective = "single"
    # FIXED(final_fix_list A6 training audits, 2026-10-20): refuse sub-ensemble codebook
    #   updates unless --allow-subensemble-codebook-update (see the option's comment).
    _sub_updates = (codebook_update_effective == "both"
                    or (random_p_range is not None and not _two_branch))
    if _sub_updates and not args.allow_subensemble_codebook_update:
        raise SystemExit("this configuration updates the codebooks from sub-ensemble latents "
                         "(--codebook-update both, two branches without "
                         "--consistency-distill-fixed-max, or --random-p-range with one "
                         "branch); pass --allow-subensemble-codebook-update to accept it")

    # FIXED(code_fix P12, 2026-10-16): the config dict (previously a literal at the end
    #   of main, same keys and order) is built here so it can be written before the
    #   first epoch too. Fields unknown at the start (best_val_recon, per_level,
    #   total_epochs, training_time_s, finished_at) are null then; "status" is new.
    # FIXED(preretrain_audit_r2 Q19, 2026-10-16): old = git commit and code hash were
    #   recomputed by every _make_cfg call, so the final config recorded the code at the
    #   END of the run; new = computed once here, before training starts.
    _git_at_start, _code_at_start = _git_commit(), _code_sha256()

    def _make_cfg(*, best_val_recon, per_level, total_epochs, training_time_s,
                  finished_at, status, codebook_health_best=None):
        return {
            "codebook_sizes": codebook_sizes,
            "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
            "n_encoder_layers": args.n_encoder_layers,
            "n_decoder_layers": args.n_decoder_layers,
            "n_queries": args.n_queries, "n_heads": args.n_heads,
            "encoder_type": args.encoder_type, "loss_type": args.loss_type,
            "vq_update": args.vq_update, "ema_decay": args.ema_decay,
            "rotation_trick": bool(args.rotation_trick),
            "eval_p": int(args.eval_p),
            "desc_dtype": args.desc_dtype,
            "nsvq_alpha": float(args.nsvq_alpha), "nsvq_tau": float(args.nsvq_tau),
            "commitment_cost": args.commitment_cost,
            "aminoaseed_projection": [bool(x) for x in aap_flags],
            "input_dim": int(D), "num_prototypes": int(P),
            "max_epochs": args.max_epochs, "patience": args.patience,
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "eval_every": args.eval_every,
            "seed": args.seed,
            "tokenizer_class": "RVQVAETokenizer",
            "quantizer": args.quantizer,
            "best_val_recon": best_val_recon,
            "per_level": per_level,
            "total_epochs": total_epochs,
            "training_time_s": training_time_s,
            "n_train_residues": int(N_tr),
            "n_val_residues": int(N_val),
            "n_params": int(n_params),
            "rect_hungarian": bool(args.rect_hungarian),
            "rect_hungarian_unmatched_penalty": float(args.rect_hungarian_unmatched_penalty),
            "no_standardize": bool(args.no_standardize),
            "ifold_lambda": float(args.ifold_lambda),
            "ifold_n_classes": int(args.ifold_n_classes) if args.ifold_lambda > 0 else 0,
            "ifold_head_input": args.ifold_head_input if args.ifold_lambda > 0 else None,
            "aa_labels": str(args.aa_labels) if args.aa_labels else None,
            "geom_attn_encoder": bool(args.geom_attn_encoder),
            "geom_attn_d_model": int(args.geom_attn_d_model) if args.geom_attn_encoder else None,
            "geom_attn_n_layers": int(args.geom_attn_n_layers) if args.geom_attn_encoder else None,
            "geom_attn_n_heads": int(args.geom_attn_n_heads) if args.geom_attn_encoder else None,
            "geom_attn_v_heads": int(args.geom_attn_v_heads) if args.geom_attn_encoder else None,
            "target_desc": str(args.target_desc) if args.target_desc else None,
            "recon_loss_type": args.recon_loss_type,
            "freeze_encoder": bool(args.freeze_encoder),
            # ---- added 2026-10-15 (code_fix §1, §2; T1.7: record every setting) ----
            "dihedral_convention": args.dihedral_convention,
            "corpus_settings": _corpus_settings,
            "chain_break_cutoff": chain_break_cutoff,
            "hungarian_cost": args.hungarian_cost,
            "rvq_residual": args.rvq_residual,
            "codebook_update": args.codebook_update,
            # ---- added 2026-10-16 (code_fix P20, P23) ----
            "codebook_update_effective": codebook_update_effective,
            # ---- added 2026-10-19 (final_fix_list A6) ----
            "codebook_update_rule": CODEBOOK_UPDATE_RULE,
            "codebook_update_rule_description": CODEBOOK_UPDATE_RULE_DESCRIPTION,
            "revival_window": int(args.revival_window),
            "revival_window_early": int(args.revival_window_early),
            # epochs 1..switch use revival_window_early, later epochs revival_window
            "revival_window_switch_epoch": int(args.revival_window_early_epochs),
            "revival_cooldown": int(args.revival_cooldown),
            "revival_warnings": revival_warnings,
            "train_batches_per_epoch": int(n_full_batches),
            "train_rows_skipped_per_epoch": int(N_tr - n_full_batches * train_bs),
            "codebook_health": codebook_health_best,
            "descriptor_mode": descriptor_mode,
            "k": desc_k,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
            "git_commit": _git_at_start,
            # The camera-ready codebase may not be committed; hash the code that trained
            # the model (ensembits/ package + train/), so a checkpoint maps to exact code.
            "code_sha256": _code_at_start,
            "started_at": _started_at,
            "resumed_at": _resumed_at,
            "finished_at": finished_at,
            "status": status,
        }

    if _resume_ck is not None:
        # Same code and settings as the interrupted run, apart from --resume / --device.
        _old_cfg = json.loads((args.out / "config.json").read_text())
        if _old_cfg.get("code_sha256") != _code_at_start:
            raise SystemExit(f"--resume: the code changed since the run started "
                             f"({_old_cfg.get('code_sha256')} -> {_code_at_start})")
        _skip = {"resume", "device", "overwrite", "debug_stop_after_epoch"}
        _new_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        _diff = sorted(k for k in set(_new_args) | set(_old_cfg["args"])
                       if k not in _skip and _new_args.get(k) != _old_cfg["args"].get(k))
        if _diff:
            raise SystemExit(f"--resume: settings differ from the interrupted run: {_diff}")
        _started_at = _old_cfg["started_at"]
        _resumed_at = list(_old_cfg.get("resumed_at") or []) + [time.strftime("%Y-%m-%dT%H:%M:%S")]

    def _atomic_json(obj, path):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)

    def _atomic_save(obj, path):
        # Training audit 2026-10-16: the output folder is on a soft NFS mount; a crash
        #   during torch.save must not leave a truncated best.pt / last.pt.
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    _atomic_json(_make_cfg(best_val_recon=None, per_level=None, total_epochs=None,
                           training_time_s=None, finished_at=None, status="running"),
                 args.out / "config.json")
    print(f"  wrote {args.out / 'config.json'} (status running)")

    print(f"\n[3/4] Training (max {args.max_epochs} epochs, "
          f"patience {args.patience}, batch {args.batch_size})")
    n_lev = len(codebook_sizes)
    history = {
        "train_recon": [], "train_vq": [], "val_recon": [], "eval_epochs": [],
        **{f"util_L{l}": [] for l in range(n_lev)},
        **{f"ppl_L{l}": [] for l in range(n_lev)},
    }
    if ifold_head is not None:
        history["train_ifold"] = []
        history["train_ifold_acc"] = []
        history["val_ifold_acc"] = []
    best_val, best_state, no_imp = float("inf"), None, 0
    codebook_health_best = None
    history["train_consistency"] = []
    history["codebook_health"] = []          # final_fix_list A6: one entry per evaluation
    t0 = time.time()
    step = 0
    start_ep = 0
    if _resume_ck is not None:
        ck = _resume_ck
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        if ifold_head is not None and ck.get("ifold_head") is not None:
            ifold_head.load_state_dict(ck["ifold_head"])
        if geom_encoder is not None and ck.get("geom_encoder") is not None:
            geom_encoder.load_state_dict(ck["geom_encoder"])
        # FIXED(final_fix_list A6, 2026-10-19): the revival bookkeeping (usage window,
        #   counters) is not in the model state dict; restore it from last.pt.
        if ck.get("codebook_update_rule") != CODEBOOK_UPDATE_RULE:
            raise SystemExit(f"--resume: last.pt was written with codebook update rule "
                             f"{ck.get('codebook_update_rule')!r}, this code uses "
                             f"{CODEBOOK_UPDATE_RULE!r}")
        for _vq, _st in zip(_vq_modules(model), ck["vq_revival_state"], strict=True):
            _vq.load_revival_state(_st)
        step, start_ep = int(ck["step"]), int(ck["epoch"])
        best_val, no_imp, history = float(ck["best_val"]), int(ck["no_imp"]), ck["history"]
        codebook_health_best = ck.get("codebook_health_best")
        if (args.out / "best.pt").exists():
            best_state = torch.load(args.out / "best.pt", map_location="cpu")
        t0 = time.time() - float(ck["elapsed_s"])
        torch.set_rng_state(ck["rng_torch"])
        if ck.get("rng_cuda") is not None and torch.device(device).type == "cuda":
            torch.cuda.set_rng_state(ck["rng_cuda"], device)
        np.random.set_state(ck["rng_numpy"])
        print(f"  resumed from {args.out / 'last.pt'} at epoch {start_ep} (step {step}, "
              f"best val {best_val:.4f}, {no_imp} epochs without improvement)")
        del ck, _resume_ck

    # ------------------------------------------------------------------------
    # One SFTD branch: encode a random subset of p_eff frames, quantize, decode
    # P_max descriptors, and score them with the Hungarian-matched FAPE loss.
    # Paper: `1_method.tex` Eq. (eq:sftd) (branch latents z_1, z_2) and
    # Eq. (eq:hungarian)/(eq:fape) (recon loss); `appendix.tex` app:RVQVAE
    # (straight-through estimator on the summed q).
    # ------------------------------------------------------------------------
    def _branch(batch_full: torch.Tensor, p_eff: int,
                  target_full: torch.Tensor | None = None) -> dict:
        """Encode a random p_eff-frame subset of `batch_full`. Returns
        {z, recon_loss, vq_loss}. Mirrors the production trainer's
        per-branch path so consistency-MSE can compare two views.

        When `geom_encoder` is on, `batch_full` is the raw 192-D
        descriptor; we reshape to (B, P, 16, 12), run the geometric
        pre-encoder to get d_model features per (residue, frame), encode
        those, and the decoder still reconstructs `target_full` (defaults
        to the original descriptor when `target_full=None`; otherwise the
        separate per-residue recon target like 144-D N/CA/C coords).
        """
        fidx = torch.randperm(P, device=device)[:p_eff]
        bp_orig = batch_full[:, fidx, :]                     # (B, p_eff, D_in)
        bp_target = bp_orig if target_full is None else target_full[:, fidx, :]
        if geom_encoder is not None:
            bp_neigh = bp_orig.view(*bp_orig.shape[:2], 16, 12)
            bp_enc = geom_encoder(bp_neigh)                   # (B, p_eff, d_model)
        else:
            bp_enc = bp_orig
        z_ = model.encoder(bp_enc)
        quantized_, _, vq_loss_ = model._quantize(z_)
        # VQ codebook lookup is non-differentiable → outer STE wrap
        q_st_ = z_ + (quantized_ - z_).detach()
        if args.rect_hungarian and p_eff < P:
            recon_ = model.decoder(q_st_)                     # (B, P_max, D_out)
            if args.recon_loss_type == "geom_dist_dir":
                # Use first p_eff decoder outputs (sliced like the else
                # branch) — geom_dist_dir doesn't compose with rect-Hungarian
                # in a meaningful way without expensive per-pair matching.
                recon_loss_ = geom_dist_dir_loss(recon_[:, :p_eff, :], bp_target)
            elif args.recon_loss_type == "fape":
                # Reported model: rectangular Hungarian (P_max outputs vs p_eff
                # targets) + FAPE + dihedral term, ensembits/losses.py.
                # FIXED(code_fix §2, 2026-10-15): matching cost set by --hungarian-cost
                #   ("loss" = paper Eq. eq:hungarian; "l2" = old checkpoints).
                # Permutation-invariant: match the full P_max decoder frames to
                # the p_eff targets (no positional slice), then FAPE.
                recon_loss_ = _fape_hungarian(recon_, bp_target,
                                                  args.dihedral_weight, args.fape_clamp)
            else:
                recon_loss_ = rectangular_hungarian_loss(
                    recon_, bp_target,
                    unmatched_penalty=args.rect_hungarian_unmatched_penalty)
        else:
            recon_ = model.decoder(q_st_)[:, :p_eff, :]
            if args.recon_loss_type == "geom_dist_dir":
                recon_loss_ = geom_dist_dir_loss(recon_, bp_target)
            elif args.recon_loss_type == "fape":
                # p_eff == P here (full-P teacher branch) — its target frames are
                # still a random permutation, so match before FAPE when enabled.
                _fape = _fape_hungarian if args.rect_hungarian else fape_dihedral_loss
                recon_loss_ = _fape(recon_, bp_target,
                                    args.dihedral_weight, args.fape_clamp)
            else:
                recon_loss_ = model._recon_loss(recon_, bp_target)
        return {"z": z_, "recon_loss": recon_loss_, "vq_loss": vq_loss_}

    for ep in range(start_ep, args.max_epochs):
        _ep_t0 = time.time()
        model.train()
        if ifold_head is not None:
            ifold_head.train()
        perm = torch.randperm(N_tr, device=tt.device)
        ep_recon = ep_vq = ep_cons = 0.0
        ep_ifold = 0.0
        ep_ifold_correct = ep_ifold_total = 0
        n_bat = 0
        # final_fix_list A6 training audits (2026-10-20): revival window of this epoch.
        _win = (args.revival_window_early if ep < args.revival_window_early_epochs
                else args.revival_window)
        for _vq in _vq_modules(model):
            _vq.revival_window = int(_win)
        # FIXED(final_fix_list A6, 2026-10-19): full batches only (drop_last), and no
        #   dead-code revival in the last --revival-cooldown steps of the epoch. Old =
        #   range(0, N_tr, batch_size) with a partial last batch.
        for bi in range(n_full_batches):
            s = bi * train_bs
            idx = perm[s:s + train_bs]
            batch = tt[idx].to(device).float()
            _set_revival_allowed(model, bi < n_full_batches - args.revival_cooldown)

            if random_p_range is None:
                # Optional explicit target (geom recon)
                tgt_batch = target_train[idx].to(device) if target_train is not None else None
                if geom_encoder is not None:
                    batch_neigh = batch.view(*batch.shape[:2], 16, 12)
                    batch_enc = geom_encoder(batch_neigh)
                    out = model(batch_enc, target=tgt_batch if tgt_batch is not None else batch)
                else:
                    out = model(batch, target=tgt_batch)
                # Override recon loss with custom geom_dist_dir / fape when requested.
                if args.recon_loss_type == "geom_dist_dir":
                    out["recon_loss"] = geom_dist_dir_loss(out["recon"], tgt_batch)
                    out["total_loss"] = out["recon_loss"] + out["vq_loss"]
                elif args.recon_loss_type == "fape":
                    _fape = _fape_hungarian if args.rect_hungarian else fape_dihedral_loss
                    out["recon_loss"] = _fape(out["recon"], batch,
                                              args.dihedral_weight, args.fape_clamp)
                    out["total_loss"] = out["recon_loss"] + out["vq_loss"]
                z_for_ifold = (out["z"] if args.ifold_head_input == "pre_vq"
                                else ifold_pen_cache.get("h"))
                cons_val = 0.0
            else:
                # ---- SFTD step (reported model) ----------------------------------
                # Paper: `1_method.tex` \label{method:SFTD}, Eq. (eq:sftd).
                # Branch a = sub-ensemble P_2 with P' ~ U{1..10} frames  -> z_2
                # Branch b = full ensemble P_1 with all P_max = 10 frames -> z_1
                # (--consistency-distill-fixed-max). One P' per batch.
                p_a = int(torch.randint(random_p_range[0], random_p_range[1] + 1,
                                          (1,)).item())
                tgt_batch = target_train[idx].to(device) if target_train is not None else None
                # FIXED(code_fix C9, 2026-10-15): with --codebook-update full, the
                #   sub-ensemble branch does not update the codebooks; the full-ensemble
                #   branch below does. Before, both branches updated them (twice per step).
                # Q4 (2026-10-16): _two_branch replaces "consistency_mse_weight > 0".
                _full_only = (args.codebook_update == "full" and _two_branch
                              and args.consistency_distill_fixed_max)
                _set_codebook_update(model, not _full_only)
                a = _branch(batch, p_a, target_full=tgt_batch)
                _set_codebook_update(model, True)
                if _two_branch:
                    p_b = P if args.consistency_distill_fixed_max else \
                        int(torch.randint(random_p_range[0], random_p_range[1] + 1,
                                            (1,)).item())
                    b = _branch(batch, p_b, target_full=tgt_batch)
                    if args.consistency_distill_fixed_max:
                        # lambda * ||z_2 - sg[z_1]||^2 / D : `.mean()` averages over the
                        # batch and the D = 128 latent dims (--consistency-mse-weight).
                        # FIXED(require_fix #8, 2026-10-15): Eq. (eq:sftd) now writes λ/D (appendix: λ/D too).
                        cons = ((a["z"] - b["z"].detach()) ** 2).mean()
                    else:
                        cons = ((a["z"] - b["z"]) ** 2).mean()
                    # Paper: "L_recon and L_commit are the mean of the per-branch ...
                    # losses across the two forward passes" (Eq. eq:sftd). `vq`
                    # already contains beta = 0.5 (commitment_cost) per level.
                    recon = 0.5 * (a["recon_loss"] + b["recon_loss"])
                    vq    = 0.5 * (a["vq_loss"]    + b["vq_loss"])
                    total = recon + vq + args.consistency_mse_weight * cons
                    out = {"recon_loss": recon, "vq_loss": vq,
                           "total_loss": total}
                    cons_val = float(cons.detach())
                else:
                    out = {"recon_loss": a["recon_loss"], "vq_loss": a["vq_loss"],
                           "total_loss": a["recon_loss"] + a["vq_loss"]}
                    cons_val = 0.0
                z_for_ifold = (a["z"] if args.ifold_head_input == "pre_vq"
                                else ifold_pen_cache.get("h"))

            # Inverse-folding auxiliary loss
            if ifold_head is not None:
                aa = aa_tr[idx].to(device)
                logits = ifold_head(z_for_ifold)
                mask = (aa < 20)
                if mask.any():
                    if_loss = F.cross_entropy(logits[mask], aa[mask])
                    ep_ifold_correct += int((logits[mask].argmax(-1)
                                              == aa[mask]).sum())
                    ep_ifold_total += int(mask.sum())
                else:
                    if_loss = torch.zeros((), device=device)
                out["total_loss"] = out["total_loss"] + args.ifold_lambda * if_loss
                ep_ifold += float(if_loss.detach())

            opt.zero_grad()
            out["total_loss"].backward()
            # Paper: `appendix.tex` app:model-impl — "gradient clipping at norm 1.0".
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Paper: `appendix.tex` app:RVQVAE — "linear warmup over the first 1000 steps".
            # Linear LR warmup for the first --warmup-steps optimizer steps.
            if args.warmup_steps and step < args.warmup_steps:
                lr_now = args.lr * (step + 1) / args.warmup_steps
                for g in opt.param_groups:
                    g["lr"] = lr_now
            opt.step()
            step += 1
            ep_recon += float(out["recon_loss"].detach())
            ep_vq    += float(out["vq_loss"].detach())
            ep_cons  += cons_val
            n_bat += 1
        # Cosine annealing only kicks in once warmup is past (or always, when no warmup).
        if not args.warmup_steps or step >= args.warmup_steps:
            sch.step()
        history["train_recon"].append(ep_recon / max(n_bat, 1))
        history["train_vq"].append(ep_vq / max(n_bat, 1))
        history["train_consistency"].append(ep_cons / max(n_bat, 1))
        if ifold_head is not None:
            history["train_ifold"].append(ep_ifold / max(n_bat, 1))
            history["train_ifold_acc"].append(
                ep_ifold_correct / max(ep_ifold_total, 1))

        # Training audit 2026-10-16: one line per epoch for the run watcher.
        print(f"  ep {ep+1:>4d} train_recon={history['train_recon'][-1]:.4f} "
              f"train_vq={history['train_vq'][-1]:.4f} "
              f"cons={history['train_consistency'][-1]:.5f} "
              f"lr={opt.param_groups[0]['lr']:.2e} ({time.time() - _ep_t0:.0f}s)", flush=True)
        if (ep + 1) % args.eval_every != 0:
            continue

        model.eval()
        if ifold_head is not None:
            ifold_head.eval()
        if geom_encoder is not None:
            geom_encoder.eval()
        # ---- validation: reconstruction loss on all frames (P = 10), eval mode
        # (no encoder noise, no EMA updates), plus per-level code utilisation /
        # perplexity (paper: `appendix.tex` app:RVQVAE reports 97.5% / 100%).
        with torch.no_grad():
            val_losses = []
            v_if_correct = v_if_total = 0
            for vs in range(0, N_val, args.batch_size):
                vb = vt[vs:vs + args.batch_size].float()
                vb_target = target_val[vs:vs + args.batch_size] if target_val is not None else None
                if args.eval_p:
                    vb = vb[:, :args.eval_p, :]
                    if vb_target is not None:
                        vb_target = vb_target[:, :args.eval_p, :]
                if geom_encoder is not None:
                    vb_enc = geom_encoder(vb.view(*vb.shape[:2], 16, 12))
                    v_out = model(vb_enc, target=vb_target if vb_target is not None else vb)
                else:
                    v_out = model(vb, target=vb_target)
                # Override with geom_dist_dir / fape when requested.
                if args.recon_loss_type == "fape":
                    _fape = _fape_hungarian if args.rect_hungarian else fape_dihedral_loss
                    v_out["recon_loss"] = _fape(v_out["recon"], vb,
                                                args.dihedral_weight, args.fape_clamp)
                elif args.recon_loss_type == "geom_dist_dir":
                    v_out["recon_loss"] = geom_dist_dir_loss(v_out["recon"], vb_target)
                # FIXED(final_fix_list A6 training audits, 2026-10-20): weight each batch
                #   by its row count so val_recon is the exact per-residue mean (old = mean
                #   of batch means; the last, smaller batch was over-weighted).
                val_losses.append((v_out["recon_loss"].item(), int(vb.shape[0])))
                if ifold_head is not None:
                    aa = aa_va[vs:vs + args.batch_size]
                    z_v = (v_out["z"] if args.ifold_head_input == "pre_vq"
                            else ifold_pen_cache.get("h"))
                    v_logits = ifold_head(z_v)
                    mask = (aa < 20)
                    if mask.any():
                        v_if_correct += int((v_logits[mask].argmax(-1) == aa[mask]).sum())
                        v_if_total += int(mask.sum())
            val_recon = (sum(l * n for l, n in val_losses)
                         / max(sum(n for _, n in val_losses), 1))
            # Chunked encode_all for utilization/perplexity stats.
            tok_chunks = [[] for _ in range(n_lev)]
            for vs in range(0, N_val, args.batch_size):
                vb = vt[vs:vs + args.batch_size].float()
                if geom_encoder is not None:
                    vb = geom_encoder(vb.view(*vb.shape[:2], 16, 12))
                ts = model.encode_all(vb)
                for l, t in enumerate(ts):
                    tok_chunks[l].append(t)
            all_tokens = [torch.cat(c, dim=0) for c in tok_chunks]

        # FIXED(final_fix_list A6, 2026-10-19): checkpoint check of the codebooks:
        #   exact / near (distance < 1e-3 x median code norm) duplicate codes per level
        #   (cosine > 0.9999 count logged, not an error) and codes unused on
        #   validation. A duplicate is an error: the checkpoint is not saved as best.pt,
        #   history / last.pt are written, config.json gets status "failed", exit 4.
        _health = (codebook_health(model.vq_levels, all_tokens)
                   if args.quantizer == "rvq" else None)
        history["codebook_health"].append({"epoch": ep + 1, **(_health or {})})
        if _health is not None:
            print("  codebooks: " + " ".join(
                f"L{h['level']} dup={h['exact_duplicates']}/{h['near_duplicates']} "
                f"cos>0.9999={h['cos_near_duplicates']} "
                f"unused_val={h['zero_usage_val']} revived={h['revived_codes']}"
                for h in _health["levels"]), flush=True)
        _health_error = _health["error"] if _health is not None else None
        if _health_error:
            print(f"  ERROR at epoch {ep+1}: {_health_error}; best.pt not updated", flush=True)

        history["eval_epochs"].append(ep + 1)
        history["val_recon"].append(val_recon)
        if ifold_head is not None:
            history["val_ifold_acc"].append(v_if_correct / max(v_if_total, 1))
        for l, t in enumerate(all_tokens):
            history[f"util_L{l}"].append(model.utilization(t, level=l))
            history[f"ppl_L{l}"].append(model.perplexity(t, level=l))

        # Paper: `appendix.tex` app:RVQVAE / app:model-impl — keep the checkpoint
        # with the lowest validation reconstruction loss; stop after `patience`
        # (40) epochs without improvement (counted in epochs: += eval_every).
        # FIXED(recon B.4, 2026-10-24): reported run (λ = 10, ensembits_bm_lam10_rv):
        #   best at epoch 435, stopped at epoch 475.
        if val_recon < best_val - 1e-5 and not _health_error:
            best_val = val_recon
            codebook_health_best = {"epoch": ep + 1, **_health} if _health else None
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Persist best_state to disk on every val improvement so a
            # signal-kill mid-training does not lose the best weights.
            # ~1s overhead per eval; the final post-loop save still runs.
            _atomic_save(best_state, args.out / "best.pt")
            if ifold_head is not None:
                torch.save({k: v.cpu().clone() for k, v in ifold_head.state_dict().items()},
                            args.out / "ifold_head.pt")
            if geom_encoder is not None:
                torch.save({k: v.cpu().clone() for k, v in geom_encoder.state_dict().items()},
                            args.out / "geom_encoder.pt")
            no_imp = 0
        else:
            no_imp += args.eval_every

        # Training audit 2026-10-16: an eval line at every evaluation (old: every 50
        #   epochs), for the run watcher.
        elapsed = time.time() - t0
        _utils = " ".join(f"util_L{l}={history[f'util_L{l}'][-1]:.0%}" for l in range(n_lev))
        print(f"  EVAL ep {ep+1:>4d}: val={val_recon:.4f} best={best_val:.4f} "
              f"no_imp={no_imp} {_utils} ppl_L0={history['ppl_L0'][-1]:.1f} "
              f"({elapsed:.0f}s)", flush=True)

        plot_training_curves(history, *plot_paths)
        _atomic_json(history, args.out / "history.json")
        # --resume state (training audit 2026-10-16), written at every evaluation.
        _atomic_save({
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(), "sch": sch.state_dict(),
            "ifold_head": ifold_head.state_dict() if ifold_head is not None else None,
            "geom_encoder": geom_encoder.state_dict() if geom_encoder is not None else None,
            "step": step, "epoch": ep + 1, "best_val": best_val, "no_imp": no_imp,
            "history": history, "elapsed_s": elapsed,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state(device)
                         if torch.device(device).type == "cuda" else None),
            "rng_numpy": np.random.get_state(),
            # final_fix_list A6 (2026-10-19): revival rule and its bookkeeping.
            "codebook_update_rule": CODEBOOK_UPDATE_RULE,
            "vq_revival_state": [vq.revival_state() for vq in _vq_modules(model)],
            "codebook_health_best": codebook_health_best,
            "codebook_health": _health,
        }, args.out / "last.pt")
        if _health_error:
            _atomic_json(_make_cfg(best_val_recon=float(best_val), per_level=None,
                                   total_epochs=int(ep + 1),
                                   training_time_s=float(time.time() - t0),
                                   finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   status="failed",
                                   codebook_health_best=codebook_health_best)
                         | {"error": _health_error}, args.out / "config.json")
            print(f"  stopping: {_health_error} (codebook update rule "
                  f"{CODEBOOK_UPDATE_RULE})", flush=True)
            return 4
        if args.debug_stop_after_epoch and ep + 1 >= args.debug_stop_after_epoch:
            print(f"  [debug] stopping after epoch {ep+1}", flush=True)
            return 3

        if no_imp >= args.patience:
            print(f"  converged at epoch {ep+1} (patience {args.patience})")
            break

    total_epochs = ep + 1
    if best_state is not None:
        _atomic_save(best_state, args.out / "best.pt")
        model.load_state_dict(best_state)
    model.to(device).eval()
    if geom_encoder is not None:
        geom_encoder.eval()

    with torch.no_grad():
        if geom_encoder is not None:
            # Tokens of the val set with the trained geom-encoder applied.
            tok_chunks = [[] for _ in range(n_lev)]
            for vs in range(0, N_val, args.batch_size):
                vb = vt[vs:vs + args.batch_size].float()
                vb_enc = geom_encoder(vb.view(*vb.shape[:2], 16, 12))
                ts = model.encode_all(vb_enc)
                for l, t in enumerate(ts):
                    tok_chunks[l].append(t)
            all_tokens = [torch.cat(c, dim=0) for c in tok_chunks]
        else:
            # .float(): vt may be fp16 (--desc-dtype float16). Chunked, so the
            # upcast does not double peak GPU memory over the full val split.
            tok_chunks = [[] for _ in range(n_lev)]
            for vs in range(0, N_val, args.batch_size):
                ts = model.encode_all(vt[vs:vs + args.batch_size].float())
                for l, t in enumerate(ts):
                    tok_chunks[l].append(t)
            all_tokens = [torch.cat(c, dim=0) for c in tok_chunks]
    per_level = [
        {"level": l, "M": codebook_sizes[l],
         "util": float(model.utilization(all_tokens[l], level=l)),
         "ppl": float(model.perplexity(all_tokens[l], level=l)),
         "unique": int(len(all_tokens[l].unique()))}
        for l in range(n_lev)
    ]
    # FIXED(final_fix_list A6, 2026-10-19): re-check the codebooks of the saved best.pt
    #   (reloaded above) and record its duplicate counts in config.json; a duplicate here
    #   cannot happen (best.pt is only saved after a clean check) and fails loudly.
    if args.quantizer == "rvq" and best_state is not None:
        _final_health = codebook_health(model.vq_levels, all_tokens)
        if _final_health["error"]:
            raise RuntimeError(f"best.pt has duplicate codes: {_final_health['error']}")
        _ep_best = (codebook_health_best or {}).get("epoch")
        # FIXED(recon2 B.5, 2026-10-25): label the health source. The codebooks are from
        #   best.pt (reloaded above), but zero_usage_val counts come from encoding the
        #   full validation set at end of training (always the same tokens as at best epoch).
        codebook_health_best = {"epoch": _ep_best, "source": "best.pt (reloaded at end of training)",
                                **_final_health}
        for h in _final_health["levels"]:
            print(f"  best.pt codebook L{h['level']}: exact dup {h['exact_duplicates']}, "
                  f"near dup {h['near_duplicates']} (cos>0.9999: {h['cos_near_duplicates']}), "
                  f"unused on val {h['zero_usage_val']}")
    # FIXED(code_fix P12, 2026-10-16): old = config.json written only here, so a
    #   killed run left none; new = the same dict (built by _make_cfg) is written
    #   before the first epoch with status "running" / finished_at null and
    #   rewritten here with the final fields and status "finished".
    cfg = _make_cfg(best_val_recon=float(best_val), per_level=per_level,
                    total_epochs=int(total_epochs),
                    training_time_s=float(time.time() - t0),
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"), status="finished",
                    codebook_health_best=codebook_health_best)
    _atomic_json(cfg, args.out / "config.json")

    plot_training_curves(history, *plot_paths)
    print(f"\n[4/4] Wrote artifacts to {args.out}/  "
          f"(plot also in {run_dir}/)")
    print(f"  best val recon: {best_val:.4f}")
    for pl in per_level:
        print(f"  level {pl['level']}: M={pl['M']}  util={pl['util']:.0%}  "
              f"ppl={pl['ppl']:.1f}  unique={pl['unique']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
