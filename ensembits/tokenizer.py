"""Ensembits RVQ-VAE tokenizer (production model class).

This is the only tokenizer class in the package. The production model
shipped with the repo (`ours/combined/ESM3`) is an instance of
``RVQVAETokenizer`` with codebook sizes [2048, 128, 128], hidden 256,
latent 128, and a 4-layer set-transformer encoder.

The file is self-contained: it includes the encoder/decoder modules,
the vector quantizer (with optional AminoAseed projection-head
reparameterization), and the Hungarian set-reconstruction loss.
"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ============================================================
# Encoders
# ============================================================

class SetTransformerEncoder(nn.Module):
    """Permutation-invariant encoder (PerceiverIO-style).

    One cross-attention layer (learnable queries attend to the P
    conformation embeddings) followed by ``n_layers`` self-attention +
    FFN blocks operating on the queries. Output is a (B, latent_dim)
    latent.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_queries: int = 4,
        n_heads: int = 4,
        n_layers: int = 1,
    ):
        super().__init__()
        self.per_element_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=n_heads, batch_first=True),
                norm1=nn.LayerNorm(hidden_dim),
                ffn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                norm2=nn.LayerNorm(hidden_dim),
            ))
            for _ in range(n_layers)
        ])
        self.projection = nn.Linear(n_queries * hidden_dim, latent_dim)
        self.n_queries = n_queries
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.per_element_mlp(x)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        q2, _ = self.cross_attn(q, h, h)
        q = self.cross_norm(q + q2)
        for blk in self.blocks:
            q2, _ = blk['attn'](q, q, q)
            q = blk['norm1'](q + q2)
            q = blk['norm2'](q + blk['ffn'](q))
        flat = q.reshape(B, self.n_queries * q.shape[-1])
        return self.projection(flat)


class SequenceEncoder(nn.Module):
    """Order-aware encoder; pair with ``permute_input=True`` to learn
    frame-order invariance from data. Not used by the shipped model
    (kept for API parity with the original codebase)."""

    def __init__(self, input_dim: int, num_prototypes: int,
                 hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.per_element_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(num_prototypes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.per_element_mlp(x)
        flat = h.reshape(B, -1)
        return self.projection(flat)


# ============================================================
# Decoder
# ============================================================

class SetDecoder(nn.Module):
    """MLP decoder: latent → (B, P, D) reconstructed descriptor vectors."""

    def __init__(self, latent_dim: int, output_dim: int, num_states: int,
                 hidden_dim: int = 128, n_layers: int = 1):
        super().__init__()
        self.num_states = num_states
        self.output_dim = output_dim
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, num_states * output_dim))
        self.mlp = nn.Sequential(*layers)
        self.n_layers = n_layers

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        flat = self.mlp(q)
        return flat.view(-1, self.num_states, self.output_dim)


# ============================================================
# Vector Quantizer
# ============================================================

def _init_orthogonal_basis(num_codes: int, basis_dim: int) -> torch.Tensor:
    """(M, K) matrix of (approximately) orthogonal rows.

    Strict orthonormality via QR when ``num_codes ≤ basis_dim``;
    otherwise unit-normalised Gaussian rows (approximately orthogonal
    in high basis_dim)."""
    if num_codes <= basis_dim:
        A = torch.randn(basis_dim, num_codes)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        return Q.T.contiguous()
    basis = torch.randn(num_codes, basis_dim)
    return basis / basis.norm(dim=1, keepdim=True)


class VectorQuantizer(nn.Module):
    """Single-level vector quantizer with EMA / loss / AminoAseed-projection
    updates.

    See van den Oord 2017 (https://arxiv.org/abs/1711.00937) §3.3 for the
    EMA update and Cao et al. 2025 (https://arxiv.org/abs/2503.00089) for
    the AminoAseed projection-head reparameterization.
    """

    def __init__(
        self, num_codes: int, code_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        vq_update: str = 'ema',
        aminoaseed: bool = False,
        basis_dim: int | None = None,
    ):
        super().__init__()
        if vq_update not in ('ema', 'loss'):
            raise ValueError(f"vq_update must be 'ema' or 'loss', got {vq_update!r}")
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.vq_update = vq_update
        self.aminoaseed = aminoaseed

        if aminoaseed:
            self.basis_dim = basis_dim if basis_dim is not None else code_dim
            self.register_buffer(
                'basis', _init_orthogonal_basis(num_codes, self.basis_dim))
            self.projection = nn.Parameter(
                torch.empty(self.basis_dim, code_dim).uniform_(
                    -1.0 / num_codes, 1.0 / num_codes))
            self.codebook = nn.Embedding(num_codes, code_dim)
            self.codebook.weight.requires_grad_(False)
        else:
            self.basis_dim = None
            self.codebook = nn.Embedding(num_codes, code_dim)
            self.codebook.weight.data.normal_(mean=0.0, std=1.0)
            if vq_update == 'ema':
                self.codebook.weight.requires_grad_(False)

        self.register_buffer('ema_count', torch.zeros(num_codes))
        if vq_update == 'ema' and not aminoaseed:
            self.register_buffer('ema_weight', self.codebook.weight.data.clone())

    def _effective_codebook(self) -> torch.Tensor:
        if self.aminoaseed:
            return self.basis @ self.projection
        return self.codebook.weight

    def forward(self, z: torch.Tensor):
        codebook = self._effective_codebook()
        if self.aminoaseed:
            with torch.no_grad():
                self.codebook.weight.data.copy_(codebook.detach())

        dists = torch.cdist(z.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
        indices = dists.argmin(dim=-1)
        q = codebook[indices]

        commitment_loss = F.mse_loss(z, q.detach())
        if self.vq_update == 'loss' or self.aminoaseed:
            vq_grad_loss = F.mse_loss(q, z.detach())
        else:
            vq_grad_loss = torch.zeros((), device=z.device)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.num_codes).float()
                counts = onehot.sum(0)
                self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)

                if self.vq_update == 'ema' and not self.aminoaseed:
                    sum_z = onehot.T @ z
                    self.ema_weight.mul_(self.ema_decay).add_(sum_z, alpha=1 - self.ema_decay)
                    n = self.ema_count.sum()
                    count_smoothed = (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                    self.codebook.weight.data.copy_(
                        self.ema_weight / count_smoothed.unsqueeze(1))

                # Dead-code revival (only for non-AminoAseed quantizers).
                if not self.aminoaseed:
                    dead = self.ema_count < 1.0
                    if dead.any():
                        n_dead = int(dead.sum().item())
                        rand_idx = torch.randint(0, z.shape[0], (n_dead,), device=z.device)
                        z_targets = z[rand_idx].detach()
                        self.codebook.weight.data[dead] = z_targets
                        if self.vq_update == 'ema':
                            self.ema_weight.data[dead] = z_targets
                        self.ema_count[dead] = 1.0

        q_st = z + (q - z).detach()
        return q_st, indices, self.commitment_cost * commitment_loss + vq_grad_loss

    @torch.no_grad()
    def kmeans_init(self, z_samples: torch.Tensor, n_iter: int = 20, seed: int = 0) -> None:
        """Initialise the codebook by running K-means on encoder outputs.

        Standard VQ-VAE practice: gives near-100 % utilisation out of the
        gate and avoids the 'random init → many dead codes' failure mode.
        """
        B, D = z_samples.shape
        M = self.num_codes
        assert D == self.code_dim, f"Expected dim={self.code_dim}, got {D}"

        g = torch.Generator(device=z_samples.device).manual_seed(seed)
        if B >= M:
            idx = torch.randperm(B, generator=g, device=z_samples.device)[:M]
            centroids = z_samples[idx].clone()
        else:
            pad = z_samples.std(0, keepdim=True) * torch.randn(
                M - B, D, generator=g, device=z_samples.device)
            centroids = torch.cat([z_samples, z_samples.mean(0, keepdim=True) + pad], dim=0)

        for _ in range(n_iter):
            dists = torch.cdist(z_samples, centroids)
            assignments = dists.argmin(dim=1)
            onehot = F.one_hot(assignments, M).float()
            counts = onehot.sum(0)
            sums = onehot.T @ z_samples
            active = counts > 0
            centroids[active] = sums[active] / counts[active].unsqueeze(1)
            if (~active).any():
                n_dead = int((~active).sum().item())
                rand_idx = torch.randint(0, B, (n_dead,),
                                          generator=g, device=z_samples.device)
                centroids[~active] = z_samples[rand_idx]

        if self.aminoaseed:
            sol = torch.linalg.lstsq(self.basis, centroids).solution
            self.projection.data.copy_(sol)
            self.codebook.weight.data.copy_(self.basis @ self.projection)
        else:
            self.codebook.weight.data.copy_(centroids)
            if self.vq_update == 'ema':
                self.ema_weight.data.copy_(centroids)
        self.ema_count.data.fill_(1.0)


# ============================================================
# Losses
# ============================================================

_PERM_CACHE: dict[int, torch.Tensor] = {}


def _build_perms(P: int) -> torch.Tensor:
    return torch.tensor(list(permutations(range(P))), dtype=torch.long)


def hungarian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batched permutation-optimal matching loss for set reconstruction.

    For small P (≤6) brute-forces all P! permutations on GPU; for larger
    P falls back to scipy's Hungarian algorithm on CPU.
    """
    B, P, D = pred.shape
    if P <= 6:
        if P not in _PERM_CACHE or _PERM_CACHE[P].device != pred.device:
            _PERM_CACHE[P] = _build_perms(P).to(pred.device)
        perms = _PERM_CACHE[P]
        target_permed = target[:, perms]
        pred_exp = pred.unsqueeze(1).expand_as(target_permed)
        costs = (pred_exp - target_permed).pow(2).mean(dim=(-1, -2))
        best_perm_idx = costs.argmin(dim=1)
        best_perms = perms[best_perm_idx]
        batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand_as(best_perms)
        matched_target = target[batch_idx, best_perms]
        return F.mse_loss(pred, matched_target)
    # P > 6: scipy fallback, batched
    costs = torch.cdist(pred, target).pow(2)
    costs_np = costs.detach().cpu().numpy()
    perms_np = np.empty((B, P), dtype=np.int64)
    for b in range(B):
        _, col_idx = linear_sum_assignment(costs_np[b])
        perms_np[b] = col_idx
    perms = torch.from_numpy(perms_np).to(pred.device)
    batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand_as(perms)
    matched_target = target[batch_idx, perms]
    return F.mse_loss(pred, matched_target)


def _permute_frames(x: torch.Tensor) -> torch.Tensor:
    """Random permutation along the P frame axis, independent per sample."""
    B, P, _ = x.shape
    perms = torch.stack(
        [torch.randperm(P, device=x.device) for _ in range(B)])
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(perms)
    return x[batch_idx, perms]


# ============================================================
# Production model
# ============================================================

class RVQVAETokenizer(nn.Module):
    """Residual VQ-VAE tokenizer (Lee et al. 2022).

    Stacks ``len(codebook_sizes)`` quantizers. At each level the running
    residual ``r_l = z - Σ_{k<l} q_k`` is quantized; the final quantized
    latent is the sum across levels. The primary token (level 0) is the
    one consumed by downstream probes.

    The production model uses ``codebook_sizes = [2048, 128, 128]`` —
    a large primary alphabet plus two smaller refinement codebooks.
    """

    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        codebook_sizes: Sequence[int] = (2048, 128, 128),
        hidden_dim: int = 128,
        latent_dim: int = 64,
        commitment_cost: float = 0.25,
        encoder_type: str = 'set_transformer',
        loss_type: str = 'hungarian',
        permute_input: bool = False,
        n_queries: int = 4,
        n_heads: int = 4,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        vq_update: str = 'ema',
        ema_decay: float = 0.99,
        aminoaseed_projection: bool | Sequence[bool] = False,
        # Accepted-but-ignored for back-compat.
        sinkhorn_weight: float = 0.0,
        sinkhorn_epsilon: float = 0.1,
    ):
        super().__init__()
        if len(codebook_sizes) < 1:
            raise ValueError("codebook_sizes must be non-empty.")
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.codebook_sizes = list(codebook_sizes)
        self.n_levels = len(self.codebook_sizes)
        self.loss_type = loss_type
        self.permute_input = permute_input

        if encoder_type == 'set_transformer':
            self.encoder = SetTransformerEncoder(
                input_dim, hidden_dim, latent_dim,
                n_queries=n_queries, n_heads=n_heads, n_layers=n_encoder_layers)
        elif encoder_type == 'sequence':
            self.encoder = SequenceEncoder(
                input_dim, num_prototypes, hidden_dim, latent_dim)
        else:
            raise ValueError(
                f"encoder_type must be 'set_transformer' or 'sequence', got {encoder_type!r}")

        # Per-level AminoAseed flag (single bool → broadcast; or sequence).
        if isinstance(aminoaseed_projection, bool):
            self.aminoaseed_projection = [aminoaseed_projection] * self.n_levels
        else:
            aap = list(aminoaseed_projection)
            if len(aap) != self.n_levels:
                raise ValueError(
                    f"aminoaseed_projection length {len(aap)} != n_levels={self.n_levels}")
            self.aminoaseed_projection = aap

        self.vq_levels = nn.ModuleList([
            VectorQuantizer(
                num_codes=M, code_dim=latent_dim,
                commitment_cost=commitment_cost, ema_decay=ema_decay,
                vq_update=vq_update, aminoaseed=bool(use_aa),
            )
            for M, use_aa in zip(self.codebook_sizes, self.aminoaseed_projection)
        ])

        self.decoder = SetDecoder(
            latent_dim, input_dim, num_prototypes, hidden_dim,
            n_layers=n_decoder_layers)

    def _quantize(self, z: torch.Tensor):
        quantized = torch.zeros_like(z)
        residual = z
        all_tokens = []
        total_vq_loss = torch.zeros((), device=z.device)
        for vq in self.vq_levels:
            q, tokens, vq_loss = vq(residual)
            quantized = quantized + q
            residual = z - quantized
            all_tokens.append(tokens)
            total_vq_loss = total_vq_loss + vq_loss
        return quantized, all_tokens, total_vq_loss

    def _recon_loss(self, recon, target):
        if self.loss_type == 'hungarian':
            return hungarian_loss(recon, target)
        if self.loss_type == 'pointwise':
            return F.mse_loss(recon, target)
        raise ValueError(f"loss_type must be 'hungarian' or 'pointwise', got {self.loss_type!r}")

    # ── Public API ─────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                sinkhorn_weight: float = 0.0,
                sinkhorn_epsilon: float = 0.1) -> dict:
        if self.permute_input and self.training:
            x = _permute_frames(x)
        z = self.encoder(x)
        quantized, all_tokens, vq_loss = self._quantize(z)
        q_st = z + (quantized - z).detach()
        recon = self.decoder(q_st)
        recon_loss = self._recon_loss(recon, x)
        total_loss = recon_loss + vq_loss
        return {
            'recon': recon,
            'tokens': all_tokens[0],
            'all_tokens': all_tokens,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'sinkhorn_loss': torch.zeros((), device=x.device),
            'total_loss': total_loss,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
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
        """High-fidelity reconstruction via the full token tuple."""
        q = torch.zeros_like(self.vq_levels[0].codebook(all_tokens[0]))
        for vq, t in zip(self.vq_levels, all_tokens):
            q = q + vq.codebook(t)
        return self.decoder(q)

    # ── Metrics ─────────────────────────────────────────────────
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
        return float(sum(np.log2(M) for M in self.codebook_sizes))
