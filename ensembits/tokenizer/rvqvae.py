"""The Ensembits model: set encoder -> residual vector quantizer -> decoder.

What it does
------------
`RVQVAETokenizer` is the model reported in the paper:

  x (L, P, 256)  descriptor multiset of each residue (P frames, any P >= 1)
    -> encoder  (SetTransformerEncoder)       z (L, 128)
    -> _quantize: N_q = 3 VectorQuantizers     tokens (c_1, c_2, c_3), q = sum of codewords
         level l quantizes the residual rho_{l-1} = z - (codewords of levels < l)
    -> decoder  (SetDecoder)                   P_max = 10 descriptors (L, 10, 256)

`encode()` returns the level-1 token per residue (the alphabet used by every
downstream task); `encode_all()` returns all three levels.

Note on training: the reported run did NOT use `forward()`'s built-in
reconstruction loss (`hungarian_loss`, an MSE). The trainer (`train/train.py`,
`_branch`) calls `model.encoder`, `model._quantize` and `model.decoder`
directly and applies the FAPE + dihedral loss of `ensembits/losses.py`.

Paper
-----
  * `1_method.tex`, \\subsection{Tokenizer}, \\paragraph{Residual quantizer.}:
    q = sum_l C^l_{c_l}, c_l = argmin_i ||rho_{l-1} - C^l_i||, rho_0 = z,
    rho_l = rho_{l-1} - C^l_{c_l}; N_q = 3, M = [2048, 128, 128], D = 128.
  * `appendix.tex`, \\subsection{RVQ-VAE} \\label{app:RVQVAE}: Eq. (eq:rvq-recurrence),
    Algorithm (alg:rvq), straight-through estimator on the summed q,
    commitment term summed over levels.
  * `appendix.tex`, \\label{app:model-impl}: architecture list.

Origin
------
Copied from `Ensembits/ensembits/tokenizer/rvqvae.py` (the trainer's package).
Logic unchanged; this header and comments added. The inference copy in the
research repo (`src/tokenizer/rvqvae.py`) lacks the rotation-trick / NSVQ /
output_dim options; with those off (as in the reported model) both give
identical tokens (checks/check_tokens_match_original.py).

Used by the reported model: yes. Options NOT used by it: `encoder_type =
'sequence'`, `aminoaseed_projection`, `rotation_trick`, `nsvq_*`,
`bypass_quantizer`, `output_dim`, `permute_input`.

Reference: Lee et al. 2022 "Autoregressive Image Generation using Residual
Quantization" (https://arxiv.org/abs/2203.01941).
"""

from __future__ import annotations
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    raise ImportError("PyTorch required for RVQVAE. Install with: pip install torch")

from .vqvae import (
    SetTransformerEncoder, SequenceEncoder, SetDecoder, VectorQuantizer,
    hungarian_loss, _permute_frames,
)


class RVQVAETokenizer(nn.Module):
    """Residual-VQ tokenizer.

    Each residue produces `len(codebook_sizes)` tokens (one per level);
    the primary token is level 0.  Straight-through gradient on the
    summed quantized latent.
    """

    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        codebook_sizes: Sequence[int] = (2048, 128, 128),
        # Reconstruction-target dim. Defaults to input_dim (standard
        # VQ-VAE). Override to e.g. 192 when a pre-encoder (geometric
        # attention over K=16 neighbors) turns a raw 192-D K=16 descriptor
        # into a d_model=256-D feature: encoder.input = 256 (d_model),
        # decoder.output = 192 (original descriptor), giving a fixed
        # non-self-referential recon target.
        output_dim: int | None = None,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        commitment_cost: float = 0.25,  # CLI overrides to 0.5 (reported model)
        encoder_type: str = 'set_transformer',
        loss_type: str = 'hungarian',
        permute_input: bool = False,
        n_queries: int = 4,
        n_heads: int = 4,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        vq_update: str = 'ema',
        # Rotation trick (arXiv:2410.06424) and NSVQ non-stationary embedding loss
        # (arXiv:2606.11363). Orthogonal: the first changes the gradient path
        # through the quantizer, the second adds a codebook-only auxiliary loss, so
        # they compose. Both default off -> the shipped tokenizer is unaffected.
        # NSVQ needs vq_update='loss' (a gradient-updated codebook); it raises otherwise.
        rotation_trick: bool = False,
        nsvq_alpha: float = 0.0,
        nsvq_tau: float = 1.0,
        ema_decay: float = 0.99,
        # AminoAseed-style projection-head reparameterization
        # (Cao et al. 2025, https://arxiv.org/abs/2503.00089).
        # When True for a level, that level's codebook becomes
        #   codebook = basis @ projection
        # with `basis ∈ R^{M×K}` a fixed orthogonal per-code identity
        # buffer and `projection ∈ R^{K×latent_dim}` a learnable shared
        # transform. Gradients flow only through `projection` via the VQ
        # loss; the fixed basis prevents two codes from collapsing onto
        # each other. Pass a single bool to apply uniformly to all
        # levels, or a sequence of bools (one per level) for granular
        # control (typical: enable on L0 only, since L1/L2 of an RVQ
        # already see a near-zero residual and rarely collapse).
        # Note: when enabled, the EMA codebook update is bypassed
        # internally (the codebook is a derived tensor, not a buffer);
        # `vq_update` becomes effectively 'loss' for that level.
        aminoaseed_projection: bool | Sequence[bool] = False,
        # Stage-1 warm-start mode: build the full module (so codebook_sizes,
        # logging and config bookkeeping are unchanged) but SKIP quantization in
        # forward -- the decoder sees the continuous latent z directly and the VQ
        # loss is zero. Used to pretrain encoder+decoder as a plain autoencoder,
        # then warm-start a real RVQ run from the result via train.py --init-from
        # (whose --kmeans-init then seeds the codebooks from the converged latent).
        # The codebooks receive no gradient in this mode and stay at their init.
        bypass_quantizer: bool = False,
        # How the residual passed to the next level is formed (code_fix C1):
        #   "standard": rho_l = rho_{l-1} - sg[Q_l(rho_{l-1})]; every level's
        #               commitment term trains the encoder (paper, app:RVQVAE).
        #   "legacy"  : rho_l = z - sum of straight-through outputs; same values,
        #               but levels >= 2 give the encoder zero gradient. Every
        #               checkpoint trained before 2026-10-15 used this.
        # Inference is identical under both (same residual values, same tokens).
        residual_mode: str = "standard",
        # Accepted-but-ignored kwargs for backward compat.
        sinkhorn_weight: float = 0.0,
        sinkhorn_epsilon: float = 0.1,
    ):
        super().__init__()
        if len(codebook_sizes) < 1:
            raise ValueError("codebook_sizes must be a non-empty sequence.")
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_prototypes = num_prototypes
        self.codebook_sizes = list(codebook_sizes)
        self.n_levels = len(self.codebook_sizes)
        self.loss_type = loss_type
        self.permute_input = permute_input
        self.bypass_quantizer = bool(bypass_quantizer)
        if residual_mode not in ("standard", "legacy"):
            raise ValueError(f"residual_mode must be 'standard' or 'legacy', got {residual_mode!r}")
        self.residual_mode = residual_mode

        # Encoder (same options as VQVAETokenizer)
        if encoder_type == 'set_transformer':
            self.encoder = SetTransformerEncoder(
                input_dim, hidden_dim, latent_dim,
                n_queries=n_queries, n_heads=n_heads,
                n_layers=n_encoder_layers)
        elif encoder_type == 'sequence':
            self.encoder = SequenceEncoder(
                input_dim, num_prototypes, hidden_dim, latent_dim)
        else:
            raise ValueError(
                f"encoder_type must be 'set_transformer' or 'sequence', "
                f"got {encoder_type!r}")

        # Per-level AminoAseed flag. Accepts either a single bool (apply
        # uniformly) or a sequence of length n_levels.
        if isinstance(aminoaseed_projection, bool):
            self.aminoaseed_projection = [aminoaseed_projection] * self.n_levels
        else:
            aap = list(aminoaseed_projection)
            if len(aap) != self.n_levels:
                raise ValueError(
                    f"aminoaseed_projection length {len(aap)} does not match "
                    f"n_levels={self.n_levels}; pass a single bool or one bool "
                    f"per codebook level.")
            self.aminoaseed_projection = aap

        # One VectorQuantizer per level. RVQ's residual structure usually
        # prevents collapse on its own, but the L0 (large) codebook can
        # benefit from the AminoAseed reparameterization on small-data
        # regimes; toggle per level via `aminoaseed_projection`.
        self.vq_levels = nn.ModuleList([
            VectorQuantizer(
                num_codes=M, code_dim=latent_dim,
                commitment_cost=commitment_cost, ema_decay=ema_decay,
                vq_update=vq_update, aminoaseed=bool(use_aa),
                rotation_trick=rotation_trick,
                nsvq_alpha=nsvq_alpha, nsvq_tau=nsvq_tau,
            )
            for M, use_aa in zip(self.codebook_sizes, self.aminoaseed_projection)
        ])

        self.decoder = SetDecoder(
            latent_dim, self.output_dim, num_prototypes, hidden_dim,
            n_layers=n_decoder_layers)

    # ── Residual quantization -------------------------------------------
    def _quantize(self, z: torch.Tensor):
        """Return (quantized_sum, all_tokens list, total VQ-loss)."""
        if self.bypass_quantizer:
            # Identity bottleneck: q == z, so the straight-through estimator in
            # forward() reduces to q_st = z and the decoder trains on the
            # continuous latent. Tokens are placeholder zeros purely so that
            # downstream logging (utilization / perplexity, which read
            # codebook_sizes rather than vq_levels) keeps working.
            zeros = torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device)
            return z, [zeros for _ in self.codebook_sizes], torch.zeros((), device=z.device)
        # Paper: `appendix.tex` Algorithm alg:rvq / Eq. eq:rvq-recurrence.
        quantized = torch.zeros_like(z)        # q   <- 0
        residual = z                           # rho <- z
        all_tokens = []
        total_vq_loss = torch.zeros((), device=z.device)

        for vq in self.vq_levels:              # for l = 1 .. N_q
            q, tokens, vq_loss = vq(residual)  # c_l = argmin ||rho - C^l_i||, q_l = C^l_{c_l}
            quantized = quantized + q          # q <- q + C^l_{c_l}
            # FIXED(code_fix C1, 2026-10-15): rho_l = rho_{l-1} - sg[C^l_{c_l}], so the
            #   residual keeps its gradient to z and the commitment terms of levels 2
            #   and 3 train the encoder, as Eq. eq:rvq-recurrence and L_commit state.
            #   ("legacy": z - sum of straight-through outputs; same values, but the
            #   residual is a constant for autograd.)
            if self.residual_mode == "legacy":
                residual = z - quantized
            else:
                residual = residual - q.detach()
            all_tokens.append(tokens)
            # vq_loss = beta * commitment of this level (EMA mode adds no codebook loss);
            # summed over levels -> L_commit of `appendix.tex` app:RVQVAE.
            total_vq_loss = total_vq_loss + vq_loss

        return quantized, all_tokens, total_vq_loss

    def _recon_loss(self, recon, target):
        if self.loss_type == 'hungarian':
            return hungarian_loss(recon, target)
        elif self.loss_type == 'pointwise':
            return F.mse_loss(recon, target)
        else:
            raise ValueError(
                f"loss_type must be 'hungarian' or 'pointwise', got "
                f"{self.loss_type!r}")

    # ── Public API -------------------------------------------------------
    def forward(self, x: torch.Tensor,
                sinkhorn_weight: float = 0.0,
                sinkhorn_epsilon: float = 0.1,
                target: torch.Tensor | None = None) -> dict:
        """`target` defaults to `x` (standard auto-encoder). Pass a separate
        tensor when an upstream pre-encoder transforms the raw descriptor
        into a learned feature (e.g. GeometricKNNEncoder); the decoder
        still reconstructs the original descriptor, giving a fixed
        non-self-referential recon target."""
        if self.permute_input and self.training:
            x = _permute_frames(x)
            if target is not None:
                # permute target consistently — but in the geom-attn case
                # the pre-encoder already produced features per frame in
                # whatever order x was already permuted at the call site,
                # so target should already be aligned. Be conservative and
                # disallow permute_input + explicit target until we have
                # a use case for it.
                raise NotImplementedError(
                    "permute_input + explicit target not supported")
        if target is None:
            target = x

        z = self.encoder(x)
        quantized, all_tokens, vq_loss = self._quantize(z)

        # Straight-through estimator on the SUM of quantizations
        # Paper: `appendix.tex` app:RVQVAE ("The straight through estimator is
        # applied to the summed embedding q").
        q_st = z + (quantized - z).detach()
        recon = self.decoder(q_st)

        # NOTE: this MSE-based Hungarian loss is NOT what the reported model was
        # trained with; train/train.py replaces it by the FAPE loss of
        # ensembits/losses.py (Eq. eq:fape). forward() is kept for other options.
        recon_loss = self._recon_loss(recon, target)
        total_loss = recon_loss + vq_loss

        return {
            'recon': recon,
            'tokens': all_tokens[0],          # primary token (for probes)
            'all_tokens': all_tokens,          # full tuple
            'z': z,                            # pre-quantization encoder latent
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'sinkhorn_loss': torch.zeros((), device=x.device),
            'total_loss': total_loss,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Primary (level-0) tokens only — same API as VQVAETokenizer.

        Paper: "using only the first level tokens" downstream (`1_method.tex`,
        Residual quantizer). The paper numbers levels from 1 (c_1, M_1 = 2048);
        the code numbers them from 0 (level 0 = the paper's level 1).
        """
        z = self.encoder(x)
        _, all_tokens, _ = self._quantize(z)
        return all_tokens[0]

    def encode_all(self, x: torch.Tensor) -> list[torch.Tensor]:
        z = self.encoder(x)
        _, all_tokens, _ = self._quantize(z)
        return all_tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode from primary tokens only (coarse reconstruction)."""
        q = self.vq_levels[0].codebook(tokens)
        return self.decoder(q)

    def decode_all(self, all_tokens: list[torch.Tensor]) -> torch.Tensor:
        """High-fidelity reconstruction using the full token tuple."""
        q = torch.zeros_like(self.vq_levels[0].codebook(all_tokens[0]))
        for vq, t in zip(self.vq_levels, all_tokens):
            q = q + vq.codebook(t)
        return self.decoder(q)

    # ── Metrics ----------------------------------------------------------
    def utilization(self, tokens: torch.Tensor, level: int = 0) -> float:
        return len(tokens.unique()) / self.codebook_sizes[level]

    def perplexity(self, tokens: torch.Tensor, level: int = 0) -> float:
        M = self.codebook_sizes[level]
        counts = torch.bincount(tokens, minlength=M).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        return entropy.exp().item()

    def effective_bits_per_residue(self) -> float:
        """Sum log2(M_l) over all levels — upper bound on tokens' info content."""
        return float(sum(np.log2(M) for M in self.codebook_sizes))
