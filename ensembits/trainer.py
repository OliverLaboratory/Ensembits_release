"""Train a Residual VQ-VAE tokenizer on a per-residue ensemble-descriptor cache.

Library entry point — wrapped by ``scripts/train.py``. The defaults
match the paper's production training recipe:

    RVQVAETokenizer(
        codebook_sizes=[2048, 128, 128],
        hidden_dim=256, latent_dim=128,
        n_encoder_layers=4, n_decoder_layers=3,
        n_queries=8, n_heads=4,
        encoder_type='set_transformer', loss_type='hungarian',
        vq_update='ema', commitment_cost=0.5,
    )

…paired with variable-P training + consistency-MSE distillation
(``random_p_range=(1, P)``, ``consistency_mse_weight=0.1``,
``consistency_distill_fixed_max=True``) and a 1000-step linear LR
warmup. K-means init on encoder outputs prevents codebook collapse.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Cap CPU thread pools BEFORE importing numpy/torch so MKL / OpenBLAS /
# OpenMP / NumExpr / PyTorch all pick up the smaller pool. Without this,
# each trainer process opens os.cpu_count() threads per pool and N
# concurrent trainers spike load average without giving real throughput.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass  # already set in this process

from .tokenizer import RVQVAETokenizer


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    """Training configuration. Default values match the production recipe."""

    # Paths
    desc_path: Path                       # .npy of shape (N, P, D)
    splits_path: Path                     # .json {'train': [...], 'val': [...]}
    out_dir: Path                         # outputs land here

    # Model
    codebook_sizes: Sequence[int] = (2048, 128, 128)
    hidden_dim: int = 256
    latent_dim: int = 128
    n_encoder_layers: int = 4
    n_decoder_layers: int = 3
    n_queries: int = 8
    n_heads: int = 4
    encoder_type: str = "set_transformer"
    loss_type: str = "hungarian"
    vq_update: str = "ema"
    ema_decay: float = 0.99
    commitment_cost: float = 0.5
    aminoaseed_projection: Sequence[bool] | bool = False

    # Training
    max_epochs: int = 1000
    patience: int = 200
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    eval_every: int = 5
    seed: int = 42
    device: str | None = None

    # Production recipe extras
    random_p_range: tuple[int, int] | None = None     # e.g. (1, 10)
    consistency_mse_weight: float = 0.0               # e.g. 0.1
    consistency_distill_fixed_max: bool = False
    warmup_steps: int = 0                             # e.g. 1000
    noise_std: float = 0.0
    kmeans_init: bool = False

    # Internal: filled in by the trainer
    _aap_flags: list[bool] = field(default_factory=list)


# ============================================================
# Helpers
# ============================================================

def standardize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score over (N_residues × P) samples.

    Returns ``(standardized, mean, std)``; replaces zeros in ``std`` with 1
    to keep the operation well-defined for constant features.
    """
    flat = data.reshape(-1, data.shape[-1])
    mu = flat.mean(0).astype(np.float32)
    std = flat.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((data - mu) / std).astype(np.float32), mu, std


def _resolve_aap_flags(aap, n_levels: int) -> list[bool]:
    """Per-level AminoAseed-projection flags.

    Accepts a single bool (broadcast to all levels) or a sequence of bools
    of length ``n_levels``.
    """
    if isinstance(aap, bool):
        return [aap] * n_levels
    flags = [bool(x) for x in aap]
    if len(flags) != n_levels:
        raise ValueError(
            f"aminoaseed_projection length {len(flags)} != n_levels={n_levels}")
    return flags


def _plot_training_curves(history: dict, *out_paths: Path):
    """Render a 3-panel training curve PNG (recon / VQ / codebook)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=110)

    train_e = np.arange(1, len(history["train_recon"]) + 1)
    eval_e = history["eval_epochs"]

    axes[0].plot(train_e, history["train_recon"], label="train recon")
    axes[0].plot(eval_e, history["val_recon"], 'o-', label="val recon")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("recon loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("reconstruction")

    axes[1].plot(train_e, history["train_vq"], label="VQ", color="tab:orange")
    if any(x > 0 for x in history.get("train_consistency", [])):
        axes[1].plot(train_e, history["train_consistency"],
                     label="consistency MSE", color="tab:green")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title("VQ / consistency")

    n_levels = sum(1 for k in history if k.startswith("util_L"))
    for level in range(n_levels):
        axes[2].plot(eval_e, history[f"util_L{level}"],
                     label=f"util L{level}", linestyle="-")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("codebook utilization")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_title("codebook")

    plt.tight_layout()
    for p in out_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main training loop
# ============================================================

def train_rvqvae(cfg: TrainConfig) -> dict:
    """Train an RVQ-VAE tokenizer end-to-end.

    Writes ``best.pt``, ``stats.npz``, ``config.json``, ``history.json``,
    and ``training_curve.png`` into ``cfg.out_dir``. Returns the saved
    config dict for caller convenience.
    """
    cfg.out_dir = Path(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    codebook_sizes = list(cfg.codebook_sizes)
    aap_flags = _resolve_aap_flags(cfg.aminoaseed_projection, len(codebook_sizes))

    print(f"[1/4] Loading descriptors {cfg.desc_path} and splits {cfg.splits_path}")
    data = np.load(cfg.desc_path)
    splits = json.loads(Path(cfg.splits_path).read_text())
    if "train" not in splits or "val" not in splits:
        raise SystemExit(f"--splits must contain 'train' and 'val'; got {list(splits)}")
    data_std, mu, std = standardize(data)
    np.savez(cfg.out_dir / "stats.npz", mean=mu, std=std)
    train_data = data_std[splits["train"]]
    val_data = data_std[splits["val"]]
    N_tr, P, D = train_data.shape
    N_val = val_data.shape[0]
    print(f"  train={N_tr}  val={N_val}  P={P}  D={D}  device={device}")

    print(f"\n[2/4] Building RVQ model: codebook_sizes={codebook_sizes} "
          f"n_enc={cfg.n_encoder_layers} n_dec={cfg.n_decoder_layers} "
          f"Q={cfg.n_queries} H={cfg.n_heads}")
    model = RVQVAETokenizer(
        input_dim=D, num_prototypes=P,
        codebook_sizes=codebook_sizes,
        hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim,
        commitment_cost=cfg.commitment_cost,
        encoder_type=cfg.encoder_type, loss_type=cfg.loss_type,
        vq_update=cfg.vq_update, ema_decay=cfg.ema_decay,
        n_queries=cfg.n_queries, n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        aminoaseed_projection=aap_flags,
    ).to(device)
    if any(aap_flags):
        print(f"  AminoAseed projection-head: per-level {[int(x) for x in aap_flags]}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.max_epochs, 1e-6)

    tt = torch.tensor(train_data, dtype=torch.float32)
    vt = torch.tensor(val_data, dtype=torch.float32).to(device)

    if cfg.noise_std > 0:
        _noise_std = float(cfg.noise_std)

        def _noise_hook(_module, _inp, out_z):
            if _module.training:
                return out_z + _noise_std * torch.randn_like(out_z)
            return out_z

        model.encoder.register_forward_hook(_noise_hook)
        print(f"  encoder-output noise σ = {cfg.noise_std}")

    if cfg.kmeans_init:
        sample_bs = min(16384, N_tr)
        sample_idx = torch.randperm(N_tr)[:sample_bs]
        with torch.no_grad():
            x_sample = tt[sample_idx].to(device)
            z_sample = model.encoder(x_sample)
        z_residual = z_sample
        for lvl, vq in enumerate(model.vq_levels):
            print(f"  kmeans_init L{lvl} (M={vq.num_codes}, {sample_bs} samples)")
            vq.kmeans_init(z_residual)
            cb = vq.codebook.weight
            dists = torch.cdist(z_residual.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
            z_residual = z_residual - cb[dists.argmin(-1)]

    if cfg.random_p_range is not None:
        pmin, pmax = cfg.random_p_range
        if not (1 <= pmin <= pmax <= P):
            raise SystemExit(f"random_p_range {pmin}..{pmax} out of bounds for P={P}")
        print(f"  variable-P training: p_eff ~ U({pmin},{pmax})  (eval at full P={P})")
        if cfg.consistency_mse_weight > 0:
            mode = "fixed-max" if cfg.consistency_distill_fixed_max else "symmetric"
            print(f"  consistency-MSE weight = {cfg.consistency_mse_weight}  ({mode})")

    print(f"\n[3/4] Training (max {cfg.max_epochs} epochs, patience {cfg.patience}, "
          f"batch {cfg.batch_size})")
    n_lev = len(codebook_sizes)
    history = {
        "train_recon": [], "train_vq": [], "train_consistency": [],
        "val_recon": [], "eval_epochs": [],
        **{f"util_L{l}": [] for l in range(n_lev)},
        **{f"ppl_L{l}": [] for l in range(n_lev)},
    }
    best_val, best_state, no_imp = float("inf"), None, 0
    t0 = time.time()
    step = 0

    def _branch(batch_full: torch.Tensor, p_eff: int) -> dict:
        """Encode a random p_eff-frame subset (used by varP + distillation)."""
        fidx = torch.randperm(P, device=device)[:p_eff]
        bp = batch_full[:, fidx, :]
        z_ = model.encoder(bp)
        quantized_, _, vq_loss_ = model._quantize(z_)
        q_st_ = z_ + (quantized_ - z_).detach()
        recon_ = model.decoder(q_st_)[:, :p_eff, :]
        recon_loss_ = model._recon_loss(recon_, bp)
        return {"z": z_, "recon_loss": recon_loss_, "vq_loss": vq_loss_}

    for ep in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(N_tr)
        ep_recon = ep_vq = ep_cons = 0.0
        n_bat = 0
        for s in range(0, N_tr, cfg.batch_size):
            batch = tt[perm[s:s + cfg.batch_size]].to(device)

            if cfg.random_p_range is None:
                out = model(batch)
                cons_val = 0.0
            else:
                rp_lo, rp_hi = cfg.random_p_range
                p_a = int(torch.randint(rp_lo, rp_hi + 1, (1,)).item())
                a = _branch(batch, p_a)
                if cfg.consistency_mse_weight > 0:
                    p_b = P if cfg.consistency_distill_fixed_max else \
                        int(torch.randint(rp_lo, rp_hi + 1, (1,)).item())
                    b = _branch(batch, p_b)
                    if cfg.consistency_distill_fixed_max:
                        cons = ((a["z"] - b["z"].detach()) ** 2).mean()
                    else:
                        cons = ((a["z"] - b["z"]) ** 2).mean()
                    recon = 0.5 * (a["recon_loss"] + b["recon_loss"])
                    vq    = 0.5 * (a["vq_loss"]    + b["vq_loss"])
                    total = recon + vq + cfg.consistency_mse_weight * cons
                    out = {"recon_loss": recon, "vq_loss": vq, "total_loss": total}
                    cons_val = float(cons.detach())
                else:
                    out = {"recon_loss": a["recon_loss"], "vq_loss": a["vq_loss"],
                           "total_loss": a["recon_loss"] + a["vq_loss"]}
                    cons_val = 0.0

            opt.zero_grad()
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if cfg.warmup_steps and step < cfg.warmup_steps:
                lr_now = cfg.lr * (step + 1) / cfg.warmup_steps
                for g in opt.param_groups:
                    g["lr"] = lr_now
            opt.step()
            step += 1
            ep_recon += float(out["recon_loss"].detach())
            ep_vq    += float(out["vq_loss"].detach())
            ep_cons  += cons_val
            n_bat += 1

        if not cfg.warmup_steps or step >= cfg.warmup_steps:
            sch.step()
        history["train_recon"].append(ep_recon / max(n_bat, 1))
        history["train_vq"].append(ep_vq / max(n_bat, 1))
        history["train_consistency"].append(ep_cons / max(n_bat, 1))

        if (ep + 1) % cfg.eval_every != 0:
            continue

        model.eval()
        with torch.no_grad():
            val_losses = []
            for vs in range(0, N_val, cfg.batch_size):
                val_losses.append(model(vt[vs:vs + cfg.batch_size])["recon_loss"].item())
            val_recon = sum(val_losses) / max(len(val_losses), 1)
            # Chunked encode_all so large val sets (688k+ residues) don't
            # materialize the full SetTransformer attention activations
            # in one pass.
            tok_chunks = [[] for _ in range(n_lev)]
            for vs in range(0, N_val, cfg.batch_size):
                ts = model.encode_all(vt[vs:vs + cfg.batch_size])
                for l, t in enumerate(ts):
                    tok_chunks[l].append(t)
            all_tokens = [torch.cat(c, dim=0) for c in tok_chunks]

        history["eval_epochs"].append(ep + 1)
        history["val_recon"].append(val_recon)
        for l, t in enumerate(all_tokens):
            history[f"util_L{l}"].append(model.utilization(t, level=l))
            history[f"ppl_L{l}"].append(model.perplexity(t, level=l))

        if val_recon < best_val - 1e-5:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Persist best.pt on every val improvement so a signal-kill
            # mid-training does not lose the best weights.
            torch.save(best_state, cfg.out_dir / "best.pt")
            no_imp = 0
        else:
            no_imp += cfg.eval_every

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            u0 = history["util_L0"][-1]
            p0 = history["ppl_L0"][-1]
            print(f"  ep {ep+1:>4d}: val={val_recon:.4f} best={best_val:.4f} "
                  f"util_L0={u0:.0%} ppl_L0={p0:.1f} ({elapsed:.0f}s)")

        _plot_training_curves(history, cfg.out_dir / "training_curve.png")
        with open(cfg.out_dir / "history.json", "w") as f:
            json.dump(history, f)

        if no_imp >= cfg.patience:
            print(f"  converged at epoch {ep+1} (patience {cfg.patience})")
            break

    total_epochs = ep + 1
    if best_state is not None:
        torch.save(best_state, cfg.out_dir / "best.pt")
        model.load_state_dict(best_state)
    model.to(device).eval()

    with torch.no_grad():
        all_tokens = model.encode_all(vt)
    per_level = [
        {"level": l, "M": codebook_sizes[l],
         "util": float(model.utilization(all_tokens[l], level=l)),
         "ppl": float(model.perplexity(all_tokens[l], level=l)),
         "unique": int(len(all_tokens[l].unique()))}
        for l in range(n_lev)
    ]
    saved_cfg = {
        "codebook_sizes": codebook_sizes,
        "hidden_dim": cfg.hidden_dim, "latent_dim": cfg.latent_dim,
        "n_encoder_layers": cfg.n_encoder_layers,
        "n_decoder_layers": cfg.n_decoder_layers,
        "n_queries": cfg.n_queries, "n_heads": cfg.n_heads,
        "encoder_type": cfg.encoder_type, "loss_type": cfg.loss_type,
        "vq_update": cfg.vq_update, "ema_decay": cfg.ema_decay,
        "commitment_cost": cfg.commitment_cost,
        "aminoaseed_projection": [bool(x) for x in aap_flags],
        "input_dim": int(D), "num_prototypes": int(P),
        "max_epochs": cfg.max_epochs, "patience": cfg.patience,
        "batch_size": cfg.batch_size, "lr": cfg.lr,
        "weight_decay": cfg.weight_decay, "eval_every": cfg.eval_every,
        "seed": cfg.seed,
        "tokenizer_class": "RVQVAETokenizer",
        "best_val_recon": float(best_val),
        "per_level": per_level,
        "total_epochs": int(total_epochs),
        "training_time_s": float(time.time() - t0),
        "n_train_residues": int(N_tr),
        "n_val_residues": int(N_val),
        "n_params": int(n_params),
    }
    with open(cfg.out_dir / "config.json", "w") as f:
        json.dump(saved_cfg, f, indent=2)
    _plot_training_curves(history, cfg.out_dir / "training_curve.png")

    print(f"\n[4/4] Wrote artifacts to {cfg.out_dir}/")
    print(f"  best val recon: {best_val:.4f}")
    for pl in per_level:
        print(f"  level {pl['level']}: M={pl['M']}  util={pl['util']:.0%}  "
              f"ppl={pl['ppl']:.1f}  unique={pl['unique']}")
    return saved_cfg
