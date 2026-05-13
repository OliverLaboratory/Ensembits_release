"""CLI shim for training the Ensembits RVQ-VAE tokenizer.

Wraps ``ensembits.trainer.train_rvqvae`` with an argparse interface that
matches the original ``Ensembits/scripts/train.py``.

To reproduce the paper's shipped tokenizer (``ours/combined/ESM3``):

    python -m scripts.train \\
      --desc <combined_esm3desc_K16_P10.npy> \\
      --splits <splits.json> \\
      --out ckpt/combined_esm3 \\
      --codebook-sizes 2048,128,128 \\
      --hidden-dim 256 --latent-dim 128 \\
      --n-encoder-layers 4 --n-decoder-layers 3 \\
      --n-queries 8 --n-heads 4 \\
      --max-epochs 1000 --patience 40 \\
      --batch-size 4096 --lr 1e-3 --weight-decay 1e-5 --seed 42 \\
      --random-p-range 1,10 \\
      --consistency-mse-weight 0.1 --consistency-distill-fixed-max \\
      --warmup-steps 1000 --kmeans-init
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ensembits.trainer import TrainConfig, train_rvqvae


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--desc", required=True, type=Path,
                    help="(N, P, D) descriptor cache (.npy)")
    ap.add_argument("--splits", required=True, type=Path,
                    help="json with 'train' / 'val' index lists")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory")

    # Model
    ap.add_argument("--codebook-sizes", default="2048,128,128")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--n-encoder-layers", type=int, default=4)
    ap.add_argument("--n-decoder-layers", type=int, default=3)
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--encoder-type", default="set_transformer",
                    choices=("set_transformer", "sequence"))
    ap.add_argument("--loss-type", default="hungarian",
                    choices=("hungarian", "pointwise"))
    ap.add_argument("--vq-update", default="ema", choices=("ema", "loss"))
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--commitment-cost", type=float, default=0.5)
    ap.add_argument("--aminoaseed-projection", default="0",
                    help="'0' (off), '1' (on every level), or comma-list "
                         "of 0/1 per level (e.g. '1,0,0' for L0 only)")

    # Training
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)

    # Production extras
    ap.add_argument("--random-p-range", default=None,
                    help="MIN,MAX — sample p_eff uniformly per batch")
    ap.add_argument("--consistency-mse-weight", type=float, default=0.0)
    ap.add_argument("--consistency-distill-fixed-max", action="store_true")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--kmeans-init", action="store_true")
    return ap.parse_args()


def _parse_aap(raw: str, n_levels: int) -> list[bool]:
    raw = raw.strip()
    if "," in raw:
        flags = [bool(int(x)) for x in raw.split(",")]
        if len(flags) != n_levels:
            raise SystemExit(
                f"--aminoaseed-projection length {len(flags)} != n_levels={n_levels}")
        return flags
    return [bool(int(raw))] * n_levels


def main() -> int:
    args = parse_args()
    codebook_sizes = [int(x) for x in args.codebook_sizes.split(",")]
    aap_flags = _parse_aap(args.aminoaseed_projection, len(codebook_sizes))
    random_p_range = None
    if args.random_p_range:
        pmin, pmax = (int(x) for x in args.random_p_range.split(","))
        random_p_range = (pmin, pmax)

    cfg = TrainConfig(
        desc_path=args.desc,
        splits_path=args.splits,
        out_dir=args.out,
        codebook_sizes=codebook_sizes,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_queries=args.n_queries, n_heads=args.n_heads,
        encoder_type=args.encoder_type, loss_type=args.loss_type,
        vq_update=args.vq_update, ema_decay=args.ema_decay,
        commitment_cost=args.commitment_cost,
        aminoaseed_projection=aap_flags,
        max_epochs=args.max_epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, eval_every=args.eval_every,
        seed=args.seed, device=args.device,
        random_p_range=random_p_range,
        consistency_mse_weight=args.consistency_mse_weight,
        consistency_distill_fixed_max=args.consistency_distill_fixed_max,
        warmup_steps=args.warmup_steps,
        noise_std=args.noise_std,
        kmeans_init=args.kmeans_init,
    )
    train_rvqvae(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
