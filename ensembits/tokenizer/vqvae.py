"""Building blocks of the Ensembits model: set encoder, decoder, quantizer.

What it does
------------
  * SetTransformerEncoder  — turns the P descriptors of a residue into one
    latent z (permutation invariant). USED by the reported model.
  * SetDecoder             — maps a quantized latent q to P_max = 10 descriptors.
    USED by the reported model.
  * VectorQuantizer        — one codebook: nearest-code assignment, commitment
    loss, EMA codebook update, dead-code revival, k-means initialisation.
    USED (three of them, stacked by `rvqvae.RVQVAETokenizer`).
  * Everything else (SequenceEncoder, VQVAETokenizer, ResidualVQVAETokenizer,
    hungarian_loss, train_vqvae, _rotate_to, _init_orthogonal_basis) belongs to
    earlier models or alternative options and is NOT used by the reported model.

Paper
-----
  * `1_method.tex`, \paragraph{Set encoder.} (SetTransformerEncoder)
  * `1_method.tex`, \paragraph{Residual quantizer.}; `appendix.tex`
    \subsection{RVQ-VAE} \label{app:RVQVAE}: Eq. (eq:ema) EMA update with
    Laplace smoothed counts, k-means initialisation, dead-code revival
    (VectorQuantizer)
  * `appendix.tex`, \label{app:model-impl}: hidden size 256, 8 queries,
    4 self-attention blocks, FFN 512, 4 heads, latent 128, decoder with three
    hidden layers of 256 (SetTransformerEncoder, SetDecoder)

Origin
------
Copied from `Ensembits/ensembits/tokenizer/vqvae.py` (the trainer's package).
Logic unchanged; this header and comments added, except:
FIXED(final_fix_list A6, 2026-10-19): dead-code revival rewritten (rule
"ema-revival-v2", `CODEBOOK_UPDATE_RULE`); see `VectorQuantizer` and
`codebook_health`. The code before the fix is kept verbatim in
`camera_ready/ensembits_code_backup_pre_revival_fix_20261019/`.

Used by the reported model: yes (the classes marked USED above).

Original module notes follow.

Two encoder paths (toggle via encoder_type):
  - 'set_transformer': permutation-invariant encoder (architectural invariance).
  - 'sequence': per-frame MLP + concat + MLP. Pair with permute_input=True
    during training so the model learns frame-order invariance from data.

Decoder: MLP outputting P reconstructed states.
Loss: Hungarian matching by default (set-equality reconstruction). Pointwise
MSE is selectable but only valid when the input order is meaningful — not
recommended in conjunction with permute_input=True.

Rg-sorting is used at descriptor *construction* time to pick reference
segments; the P frames presented to the encoder are an unordered ensemble.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("PyTorch and scipy required. Install with: pip install torch scipy")


# FIXED(final_fix_list A6, 2026-10-19): version string of the codebook update rule
#   (EMA update + dead-code revival). Recorded in config.json / last.pt by the trainer
#   so runs trained with the corrected rule are identifiable. Old rule (no string):
#   every training batch reseeded each code with EMA count < 1 from that batch, with
#   replacement, partial batches included.
CODEBOOK_UPDATE_RULE = "ema-revival-v2"
# FIXED(final_fix_list A6 training audits, 2026-10-20): one-line description of the rule,
#   recorded next to the name in config.json ("codebook_update_rule_description").
CODEBOOK_UPDATE_RULE_DESCRIPTION = (
    "EMA codebook update (decay 0.99, Laplace-smoothed counts) on full training batches only "
    "(drop_last; partial batches do no EMA update, usage count or revival); a code with no "
    "assignment over a window of full batches (trainer schedule: 100 in the first 2 epochs, "
    "400 afterwards) is dead; at most min(dead, batch) dead codes per check are reseeded from "
    "distinct rows of the level input (randperm) plus Gaussian noise of per-dimension sd "
    "max(0.05 x batch std, 1e-3 x median code norm); the window restarts after every check; "
    "no revival in the last 50 steps of an epoch; codebooks updated from the full-ensemble "
    "SFTD branch only; duplicate check (exact / distance < 1e-3 x median norm) at every "
    "evaluation")


# ======================================================================
# Set Transformer Encoder (permutation-invariant)
# ======================================================================

class SetTransformerEncoder(nn.Module):
    """Permutation-invariant encoder, PerceiverIO-style.  [USED]

    Paper: `1_method.tex` \paragraph{Set encoder.} — shared per-element MLP,
    learnable queries cross-attend to the P embeddings, self-attention + FFN
    blocks among the queries, queries concatenated and projected to z.
    Reported model: input 256, hidden 256, n_queries 8, n_heads 4, n_layers 4,
    latent 128.

    One cross-attention layer (learnable queries attend to the P conformation
    embeddings) followed by `n_layers` self-attention + FFN blocks operating
    only on the queries.  Depth scales by increasing `n_layers`.

    n_layers=1 (default) is the "light" baseline used through Phases A–D.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64,
                 n_queries: int = 4, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.per_element_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        # Initial cross-attention: queries ← conformation embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Stack of N self-attention + FFN blocks on the queries
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

        # Project flattened queries to latent
        self.projection = nn.Linear(n_queries * hidden_dim, latent_dim)
        self.n_queries = n_queries
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, D) set of descriptors.

        Returns:
            z: (B, d_z) latent vectors.
        """
        B = x.shape[0]
        h = self.per_element_mlp(x)  # (B, P, d_h)

        # Expand queries for batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_q, d_h)

        # Cross-attention: queries read from the P conformation embeddings
        q2, _ = self.cross_attn(q, h, h)
        q = self.cross_norm(q + q2)

        # Stacked self-attention + FFN blocks on queries
        for blk in self.blocks:
            q2, _ = blk['attn'](q, q, q)
            q = blk['norm1'](q + q2)
            q = blk['norm2'](q + blk['ffn'](q))

        flat = q.reshape(B, self.n_queries * q.shape[-1])
        return self.projection(flat)


# ======================================================================
# Sequence Encoder (order-aware, pair with permute_input for invariance)
# ======================================================================

class SequenceEncoder(nn.Module):
    """Order-aware encoder: per-frame MLP + concat + MLP.  [NOT USED by the reported model]

    Not permutation-invariant by construction — to use this for ensembles
    where frame order is meaningless, pair it with permute_input=True so the
    model learns invariance from random per-step shuffling of the P frames.
    """

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
        """
        Args:
            x: (B, P, D) descriptor tensor.

        Returns:
            z: (B, d_z) latent vectors.
        """
        B = x.shape[0]
        h = self.per_element_mlp(x)  # (B, P, d_h)
        flat = h.reshape(B, -1)  # (B, P * d_h)
        return self.projection(flat)  # (B, d_z)


# ======================================================================
# Decoder
# ======================================================================

class SetDecoder(nn.Module):
    """MLP decoder: latent → P reconstructed descriptor vectors.  [USED]

    Paper: `1_method.tex` \paragraph{Decoder and reconstruction loss.} ("The
    decoder maps q to a fixed set of P_max = 10 descriptors"); `appendix.tex`
    app:model-impl ("an MLP with three hidden layers of size 256 and GELU").
    Reported model: num_states = P_max = 10, output_dim = 256, n_layers = 3.
    The output count is fixed; the trainer matches it to however many frames
    were encoded (ensembits/losses.py).

    Depth is parameterised by `n_layers` = number of (Linear + GELU) hidden
    blocks between the input projection and the output projection.

    n_layers=1 (default) produces `Linear → GELU → Linear`, i.e. one hidden
    block, matching the "light" baseline.  Larger n_layers adds capacity.
    """

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
        """
        Args:
            q: (B, d_z) quantized latent vectors.

        Returns:
            out: (B, P, D) reconstructed states.
        """
        flat = self.mlp(q)
        return flat.view(-1, self.num_states, self.output_dim)


# ======================================================================
# Vector Quantizer
# ======================================================================

def _init_orthogonal_basis(num_codes: int, basis_dim: int) -> 'torch.Tensor':
    """Build an (M, K) matrix with (approximately) orthogonal rows.
    [NOT USED by the reported model: AminoAseed reparameterization option;
    paper mentions it as tried and not effective, `appendix.tex` app:RVQVAE.]

    If num_codes ≤ basis_dim, the rows can be strictly orthonormal; built via
    QR decomposition of a random Gaussian.  If num_codes > basis_dim, strict
    orthogonality is impossible (pigeonhole), so we fall back to unit-normalized
    random Gaussian rows (approximately orthogonal in high basis_dim).
    """
    if num_codes <= basis_dim:
        A = torch.randn(basis_dim, num_codes)
        Q, _ = torch.linalg.qr(A, mode='reduced')  # Q: (basis_dim, num_codes), orthonormal columns
        return Q.T.contiguous()                    # (num_codes, basis_dim), orthonormal rows
    else:
        basis = torch.randn(num_codes, basis_dim)
        return basis / basis.norm(dim=1, keepdim=True)


def _rotate_to(src: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation-trick surrogate for the straight-through estimator.
    [NOT USED by the reported model: quantizer ablation "quant B/D" only.]

    Fifty et al., "Restructuring Vector Quantization with the Rotation Trick"
    (ICLR 2025 oral, arXiv:2410.06424), §4.2. Ported from the reference
    implementation in lucidrains/vector-quantize-pytorch
    (`efficient_rotation_trick_transform` / `rotate_to`).

    Returns a tensor whose VALUE equals `tgt` (so quantization is unchanged) but
    whose gradient w.r.t. `src` is the constant linear map λR — a rotation R
    taking src/‖src‖ to tgt/‖tgt‖ composed with rescaling λ=‖tgt‖/‖src‖ — rather
    than the identity that the straight-through estimator uses. R is applied via
    its Householder form, never materialised as a d×d matrix:

        R x = x − 2(x·w)w + 2(x·u)q,     w = normalize(u + q),
        u = src/‖src‖,  q = tgt/‖tgt‖,   with u, q, w all detached.

    `src` (2-D, [N, d]) and `tgt` must have the same shape.
    """
    norm_src = src.norm(dim=-1, keepdim=True)
    norm_tgt = tgt.norm(dim=-1, keepdim=True)
    u = src / norm_src.clamp_min(eps)
    q = tgt / norm_tgt.clamp_min(eps)

    w = F.normalize(u + q, dim=-1, eps=eps).detach()
    u_d, q_d = u.detach(), q.detach()

    # rotate src itself (not the unit vector), then rescale to ‖tgt‖
    rotated = (src
               - 2.0 * (src * w).sum(-1, keepdim=True) * w
               + 2.0 * (src * u_d).sum(-1, keepdim=True) * q_d)
    return rotated * (norm_tgt / norm_src.clamp_min(eps)).detach()


class VectorQuantizer(nn.Module):
    """Vector quantization with configurable codebook updates.  [USED, x3]

    One level of the residual quantizer. The reported model uses
    vq_update='ema', ema_decay=0.99, commitment_cost=0.5, aminoaseed=False,
    rotation_trick=False, nsvq_alpha=0 (the last three are ablation options).

    Paper: `appendix.tex` \subsection{RVQ-VAE} \label{app:RVQVAE}:
      * nearest-code assignment  Q_l(rho) = argmin_e ||rho - e||
      * commitment term (1/D)||rho - sg[Q_l(rho)]||^2, weighted by beta = 0.5
      * EMA update, Eq. (eq:ema), with Laplace smoothed counts (eps = 1e-5)
      * dead-code revival: codes with EMA count < 1 reseeded to a random input
        of this level from the current batch, count reset to 1
        FIXED(final_fix_list A6, 2026-10-19): now rule "ema-revival-v2" (codes
        with no assignment over a configurable window of full batches, default
        100 in epochs 0-1 and 400 afterwards; distinct rows plus noise; never
        on a partial batch; trainer blocks it before checkpoints); see `__init__`.
      * k-means initialisation (`kmeans_init`, 20 Lloyd iterations)

    Args:
        vq_update: 'ema' (van den Oord 2017, https://arxiv.org/abs/1711.00937 §3.3)
            or 'loss' (gradient descent on ||sg[z] - e||²).
        aminoaseed: if True, reparameterize the codebook as
              codebook = basis @ projection
            where:
              - basis ∈ R^{M × K} is FIXED (orthogonal per-code identity)
              - projection ∈ R^{K × d} is LEARNABLE (shared linear transformation)
            This is the "Q = Linear(C)" recipe in Cao et al. 2025
            (https://arxiv.org/abs/2503.00089).  Gradients flow only through the
            projection via the VQ loss; the fixed orthogonal basis prevents
            two codes' identities from collapsing.  `vq_update` is ignored when
            AminoAseed is on (the EMA path only makes sense when the codebook
            itself is a non-parameter buffer).
        basis_dim: K for AminoAseed.  Default = code_dim (strict orthogonality
            requires K ≥ num_codes, but K = code_dim gives approximately-orthogonal
            rows in high basis_dim, which is what the paper proposes).
    """

    # Note: the Python default commitment_cost=0.25 is overridden to 0.5 by the CLI
    # (train.py --commitment-cost 0.5). The reported model uses 0.5.
    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25,
                 ema_decay: float = 0.99, vq_update: str = 'ema',
                 aminoaseed: bool = False, basis_dim: int | None = None,
                 rotation_trick: bool = False,
                 nsvq_alpha: float = 0.0, nsvq_tau: float = 1.0):
        super().__init__()
        if vq_update not in ('ema', 'loss'):
            raise ValueError(f"vq_update must be 'ema' or 'loss', got {vq_update!r}")
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        # code_fix C9 (2026-10-15): when False, a training-mode forward skips the
        # codebook bookkeeping (EMA update and dead-code revival). The trainer sets it
        # False for the SFTD sub-ensemble branch, so codebooks follow full-ensemble
        # latents only.
        self.update_codebook = True

        # FIXED(final_fix_list A6, 2026-10-19): dead-code revival rule "ema-revival-v2".
        #   What was wrong: every training forward reseeded each code whose EMA count
        #   was < 1 (a code reseeded with count 1.0 fell below 1 again after one batch
        #   without a hit, so rarely used codes were reseeded at nearly every step),
        #   the replacement rows were drawn WITH replacement (`torch.randint`) from the
        #   current batch, partial batches included. The trainer's last batch of each
        #   epoch had 422 rows, so ~400 level-1 codes were reseeded from 422 rows
        #   (150-191 exact duplicate codes per checkpoint, ~20% raw-sample codes), and
        #   best.pt was saved right after it.
        #   Rule now (all levels, RVQ and single-level VQ; not AminoAseed):
        #   * usage window: hard assignment counts are summed in `usage_window` over
        #     the codebook-updating forwards on FULL batches; `window_steps` counts them;
        #   * dead code: fewer than `revival_min_usage` (1, i.e. no assignment at all)
        #     assignments over the window; the check runs once the window holds
        #     `revival_window` (100 = the EMA time constant 1/(1-0.99)) full batches
        #     and revival is allowed (`revival_allowed`, switched off by the trainer
        #     for the last steps of every epoch, so no checkpoint holds codes reseeded
        #     just before it); a deferred check runs at the next allowed step, so the
        #     window can only grow;
        #   * after a check (with or without dead codes) the window restarts for all
        #     codes, so a revived code gets a full window before it can be judged dead;
        #   * replacements: at most min(n_dead, batch) codes per check; rows of this
        #     level's input drawn WITHOUT replacement (`torch.randperm`) plus Gaussian
        #     noise of per-dimension sd max(`revival_noise` (0.05) x batch std,
        #     `revival_noise_floor` (1e-3) x median code norm of this level), so revived
        #     codes are distinct from each other and from live codes even when the
        #     batch rows are identical (zero std); when more codes are dead than rows
        #     exist, a random subset is revived and the rest wait;
        #   * partial batch: a forward whose batch is smaller than `full_batch_size`
        #     (set by the trainer; None = largest batch seen so far) does no codebook
        #     bookkeeping at all (no EMA update, no usage count, no revival).
        self.revival_window = 100
        self.revival_min_usage = 1.0
        self.revival_noise = 5e-2
        # FIXED(final_fix_list A6 training audits, 2026-10-20): absolute noise floor
        #   (old fix: 0.05 x batch std only, zero for identical rows).
        self.revival_noise_floor = 1e-3
        self.revival_allowed = True
        self.full_batch_size: int | None = None
        self._max_batch_seen = 0
        # Non-persistent: best.pt keeps the same keys as before (strict loading by
        # ensembits.featurize.load_model); the trainer saves them in last.pt.
        self.register_buffer('usage_window', torch.zeros(num_codes), persistent=False)
        self.register_buffer('window_steps', torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer('n_revival_checks', torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer('n_revived_codes', torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer('n_partial_skipped', torch.zeros((), dtype=torch.long), persistent=False)
        self.vq_update = vq_update
        self.aminoaseed = aminoaseed

        # --- Rotation trick (Fifty et al., ICLR 2025 oral, arXiv:2410.06424) ---
        # Replaces the straight-through estimator with a rotation+rescaling that is
        # treated as a constant in the backward pass, so the *angle and magnitude*
        # between z and its codeword survive into the encoder gradient instead of
        # being discarded. Purely a gradient-path change: the forward value is still
        # exactly the codeword (asserted in the unit test), so tokens are unaffected.
        self.rotation_trick = rotation_trick

        # --- NSVQ non-stationary embedding loss (arXiv:2606.11363) ---
        #   L_NS = (1/Nd) Σ_i Σ_{j≠q_i} p̃_ij ‖sg[z_i] − c_j‖²,
        #   p̃_ij = sg[softmax_j(−d_ij / τ)]
        # An auxiliary loss that pulls every NON-winning code toward the current
        # encoder outputs with detached locality weights, so codes that never win
        # still track encoder drift (the paper's diagnosis of codebook collapse).
        # z is stop-gradiented, so this term reaches the CODEBOOK ONLY and cannot
        # move the encoder. It therefore requires a gradient-updated codebook
        # (vq_update='loss' or aminoaseed); under pure EMA the codebook is a buffer
        # and the term would be silently inert -- we raise instead.
        # τ is dimensionless: distances are divided by their batch mean before the
        # softmax (see forward), so the paper's default transfers across latent
        # widths. Paper defaults (arXiv:2606.11363, verified against ar5iv):
        #   alpha = 0.1, tau = 0.35, beta = 0.25.
        # Sanity numbers measured at code_dim=128 with RAW distances (d^2 ~ 256):
        # raw tau=1 -> 11% of mass on non-winners (inert); raw tau=10 -> 88%.
        # With the batch-mean normalisation those pathologies do not arise.
        self.nsvq_alpha = nsvq_alpha
        self.nsvq_tau = nsvq_tau
        if nsvq_alpha > 0.0 and vq_update != 'loss' and not aminoaseed:
            raise ValueError(
                "nsvq_alpha>0 needs a gradient-updated codebook: pass vq_update='loss' "
                "(or aminoaseed=True). Under vq_update='ema' the codebook is a buffer, "
                "so the NSVQ term would have no effect.")

        # Codebook initialization scale matches the expected encoder-output scale
        # (std ≈ 1 after LayerNorm'd encoder on standardized input).  This matches
        # the Sonnet EMA reference (https://github.com/deepmind/sonnet, VectorQuantizerEMA
        # uses random_normal_initializer).  Much larger than the previous
        # uniform(±1/M) default, which started codes near zero (~500× below z scale).
        if aminoaseed:
            self.basis_dim = basis_dim if basis_dim is not None else code_dim
            # basis ∈ R^{M × K}: FIXED orthogonal per-code identity.
            self.register_buffer(
                'basis', _init_orthogonal_basis(num_codes, self.basis_dim))
            # projection ∈ R^{K × d}: LEARNABLE shared transformation.
            # Deliberately SMALL init: derived codebook (basis @ projection) must
            # start near zero so every code is approximately equidistant from the
            # encoder's initial z. AminoAseed has no dead-code revival (fixed
            # orthogonal basis), so any initial asymmetry in code positions → a
            # single code captures all assignments → permanent collapse of the
            # other codes. Empirically, N(0, 1) init breaks AminoAseed; small
            # uniform init works.
            self.projection = nn.Parameter(
                torch.empty(self.basis_dim, code_dim).uniform_(
                    -1.0 / num_codes, 1.0 / num_codes))
            # Embedding buffer for API compatibility (.encode() / .decode()).
            self.codebook = nn.Embedding(num_codes, code_dim)
            self.codebook.weight.requires_grad_(False)
        else:
            self.basis_dim = None
            self.codebook = nn.Embedding(num_codes, code_dim)
            self.codebook.weight.data.normal_(mean=0.0, std=1.0)
            if vq_update == 'loss':
                # Codebook learned via gradient; leave requires_grad=True (default).
                pass
            else:
                # EMA: codebook is updated in-place, not via optimizer.
                self.codebook.weight.requires_grad_(False)

        # EMA count tracks code usage; used for EMA update + dead-code detection.
        self.register_buffer('ema_count', torch.zeros(num_codes))
        if vq_update == 'ema' and not aminoaseed:
            # Only pure-EMA mode uses the encoder-output EMA buffer.
            self.register_buffer('ema_weight', self.codebook.weight.data.clone())

    def _is_full_batch(self, n_rows: int) -> bool:
        """True when a training forward of `n_rows` rows may update the codebook.

        FIXED(final_fix_list A6, 2026-10-19): new. A batch smaller than
        `full_batch_size` (or, when that is None, than the largest batch seen so far)
        is partial: it does no EMA update, no usage count and no revival.
        """
        self._max_batch_seen = max(self._max_batch_seen, int(n_rows))
        full = self.full_batch_size if self.full_batch_size is not None else self._max_batch_seen
        if n_rows < full:
            self.n_partial_skipped.add_(1)
            return False
        return True

    @torch.no_grad()
    def _revive_dead_codes(self, z: torch.Tensor) -> int:
        """Reseed the codes unused over the current window; return how many.

        FIXED(final_fix_list A6, 2026-10-19): new (rule in `__init__`). Dead = fewer
        than `revival_min_usage` assignments in `usage_window`; at most
        min(n_dead, batch) codes are reseeded, from distinct rows of `z` (randperm)
        plus noise; the usage window restarts for every code.
        """
        dead_idx = (self.usage_window < self.revival_min_usage).nonzero(as_tuple=True)[0]
        self.usage_window.zero_()
        self.window_steps.zero_()
        self.n_revival_checks.add_(1)
        n_dead, B = int(dead_idx.numel()), int(z.shape[0])
        if n_dead == 0 or B == 0:
            return 0
        if n_dead > B:
            dead_idx = dead_idx[torch.randperm(n_dead, device=dead_idx.device)[:B]]
            n_dead = B
        rows = torch.randperm(B, device=z.device)[:n_dead]
        std = z.std(0, keepdim=True) if B > 1 else torch.zeros_like(z[:1])
        floor = self.revival_noise_floor * float(
            self._effective_codebook().detach().norm(dim=1).median().clamp_min(1e-6))
        sd = torch.clamp(self.revival_noise * torch.nan_to_num(std), min=floor)
        targets = z[rows] + sd * torch.randn(
            n_dead, z.shape[1], device=z.device, dtype=z.dtype)
        self.codebook.weight.data[dead_idx] = targets
        if self.vq_update == 'ema':
            self.ema_weight.data[dead_idx] = targets
        self.ema_count[dead_idx] = 1.0
        self.n_revived_codes.add_(n_dead)
        return n_dead

    def revival_state(self) -> dict:
        """Non-persistent revival bookkeeping, for the trainer's last.pt (--resume)."""
        return {"usage_window": self.usage_window.detach().cpu().clone(),
                "window_steps": int(self.window_steps),
                "n_revival_checks": int(self.n_revival_checks),
                "n_revived_codes": int(self.n_revived_codes),
                "n_partial_skipped": int(self.n_partial_skipped),
                "max_batch_seen": int(self._max_batch_seen)}

    def load_revival_state(self, state: dict) -> None:
        """Inverse of `revival_state`."""
        self.usage_window.copy_(state["usage_window"].to(self.usage_window.device))
        self.window_steps.fill_(int(state["window_steps"]))
        self.n_revival_checks.fill_(int(state["n_revival_checks"]))
        self.n_revived_codes.fill_(int(state["n_revived_codes"]))
        self.n_partial_skipped.fill_(int(state["n_partial_skipped"]))
        self._max_batch_seen = int(state["max_batch_seen"])

    def _effective_codebook(self) -> torch.Tensor:
        """The M × d codebook actually used for quantization."""
        if self.aminoaseed:
            return self.basis @ self.projection
        return self.codebook.weight

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codebook = self._effective_codebook()
        if self.aminoaseed:
            # Keep the Embedding buffer in sync for downstream `.encode()` / `.decode()`.
            with torch.no_grad():
                self.codebook.weight.data.copy_(codebook.detach())

        # Paper: Q_l(rho) = argmin_e ||rho - e|| (`appendix.tex` app:RVQVAE).
        # Here `z` is this level's input: the encoder output at level 1, the
        # residual rho_{l-1} at later levels.
        dists = torch.cdist(z.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
        indices = dists.argmin(dim=-1)
        q = codebook[indices]

        # Commitment loss (encoder → codes)
        # Paper: (1/D)||rho_{l-1} - sg[Q_l(rho_{l-1})]||^2 (MSE = mean over the D dims),
        # multiplied by commitment_cost = beta = 0.5 at the return statement.
        commitment_loss = F.mse_loss(z, q.detach())

        # VQ loss: gives gradient to codebook (loss-mode) or to projection (AminoAseed).
        # Not used in pure EMA mode.
        if self.vq_update == 'loss' or self.aminoaseed:
            vq_grad_loss = F.mse_loss(q, z.detach())
        else:
            vq_grad_loss = torch.zeros((), device=z.device)

        # Codebook bookkeeping (skipped when update_codebook is False, code_fix C9).
        # FIXED(final_fix_list A6, 2026-10-19): also skipped on a partial batch (old:
        #   any batch size updated the EMA and triggered revival).
        if self.training and self.update_codebook and self._is_full_batch(z.shape[0]):
            with torch.no_grad():
                onehot = F.one_hot(indices, self.num_codes).float()
                counts = onehot.sum(0)
                self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)

                # Paper: Eq. (eq:ema) — N_i <- gamma N_i + (1-gamma) n_i  (line above),
                # m_i <- gamma m_i + (1-gamma) sum of assigned inputs, e_i = m_i / N~_i
                # with Laplace smoothed N~_i = (N_i + eps) sum_j N_j / (sum_j N_j + M eps).
                # Pure-EMA codebook update (only for non-AminoAseed EMA mode).
                if self.vq_update == 'ema' and not self.aminoaseed:
                    sum_z = onehot.T @ z
                    self.ema_weight.mul_(self.ema_decay).add_(sum_z, alpha=1 - self.ema_decay)
                    n = self.ema_count.sum()
                    count_smoothed = (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                    self.codebook.weight.data.copy_(self.ema_weight / count_smoothed.unsqueeze(1))

                # Paper: dead-code revival (`appendix.tex` app:RVQVAE).
                # FIXED(final_fix_list A6(b), 2026-10-20): revival rule is now ema-revival-v2
                #   (no assignment over a window of full batches; rows drawn without
                #   replacement plus noise; none in the last steps of an epoch).
                #   Paper text D.1: appendix describes the rule. All 8 _rv runs used it.
                # Dead-code revival: EMA and loss modes only.
                # AminoAseed's fixed orthogonal basis prevents collapse structurally,
                # and per-code identity cannot be modified at runtime — so no revival.
                if not self.aminoaseed:
                    self.usage_window.add_(counts)
                    self.window_steps.add_(1)
                    if self.revival_allowed and int(self.window_steps) >= self.revival_window:
                        self._revive_dead_codes(z.detach())

        # NSVQ non-stationary embedding loss: pull non-winning codes toward the
        # encoder outputs with detached locality weights (arXiv:2606.11363).
        nsvq_loss = torch.zeros((), device=z.device)
        if self.nsvq_alpha > 0.0:
            # cdist on detached z: gradient reaches ONLY the codebook, and this
            # avoids materialising the [N, K, D] broadcast (4.3 GB at N=4096,
            # K=2048, D=128) that the naive expansion allocates every step.
            d_cb = torch.cdist(z.detach().unsqueeze(0),
                               codebook.unsqueeze(0)).squeeze(0).pow(2)
            with torch.no_grad():
                # τ is applied to distances normalised by their batch mean, so it is
                # dimensionless and transfers across latent widths. The paper's
                # τ=0.35 is defined on ITS distance scale; at our code_dim=128 the
                # raw squared distances are ≈256, where a literal 0.35 would put
                # ~all softmax mass on the winner and make L_NS inert.
                scale = d_cb.mean().clamp_min(1e-8)
                p = F.softmax(-(d_cb / scale) / self.nsvq_tau, dim=-1)  # p̃_ij
                p = p.scatter(1, indices.unsqueeze(1), 0.0)             # j ≠ q_i
            nsvq_loss = (p * d_cb).sum() / (z.shape[0] * self.code_dim)

        if self.rotation_trick:
            q_st = _rotate_to(z, q)          # forward value == q, richer backward
        else:
            q_st = z + (q - z).detach()      # straight-through (default)
        return (q_st, indices,
                self.commitment_cost * commitment_loss + vq_grad_loss
                + self.nsvq_alpha * nsvq_loss)

    @torch.no_grad()
    def kmeans_init(self, z_samples: torch.Tensor, n_iter: int = 20,
                    seed: int = 0) -> None:
        """Initialize the codebook by running K-means on encoder outputs.

        Paper: `appendix.tex` app:RVQVAE — "every codebook is initialised by
        k-means (20 Lloyd iterations) on 16,384 sampled training residues, level
        by level". The level-by-level residual loop is in train/train.py.

        Standard VQ-VAE practice: instead of uniform / Gaussian codebook init,
        compute K-means centroids on a sample batch of encoder outputs and
        seed the codebook with them.  Typically gives near-100% utilization
        out of the gate and avoids the "random init → many dead codes" failure.

        Args:
            z_samples: (B, d) batch of encoder outputs to cluster.
            n_iter: Lloyd's iterations.
            seed: random seed for sample-based centroid init.
        """
        B, D = z_samples.shape
        M = self.num_codes
        assert D == self.code_dim, f"Expected z_samples dim={self.code_dim}, got {D}"

        g = torch.Generator(device=z_samples.device).manual_seed(seed)
        # Initial centroids: random (unique) samples; pad with Gaussian noise if B < M.
        if B >= M:
            idx = torch.randperm(B, generator=g, device=z_samples.device)[:M]
            centroids = z_samples[idx].clone()
        else:
            pad = z_samples.std(0, keepdim=True) * torch.randn(
                M - B, D, generator=g, device=z_samples.device)
            centroids = torch.cat([z_samples, z_samples.mean(0, keepdim=True) + pad], dim=0)

        # Lloyd's iterations
        for _ in range(n_iter):
            dists = torch.cdist(z_samples, centroids)
            assignments = dists.argmin(dim=1)
            onehot = F.one_hot(assignments, M).float()
            counts = onehot.sum(0)
            sums = onehot.T @ z_samples
            active = counts > 0
            centroids[active] = sums[active] / counts[active].unsqueeze(1)
            # Revive any dead centroid to a random sample
            # FIXED(final_fix_list A6, 2026-10-19): samples drawn without replacement
            #   (randperm) when there are enough; old = randint (with replacement),
            #   which could give two centroids the same sample.
            if (~active).any():
                n_dead = int((~active).sum().item())
                if n_dead <= B:
                    rand_idx = torch.randperm(B, generator=g,
                                              device=z_samples.device)[:n_dead]
                else:
                    rand_idx = torch.randint(0, B, (n_dead,),
                                             generator=g, device=z_samples.device)
                centroids[~active] = z_samples[rand_idx]

        # Install centroids
        if self.aminoaseed:
            # Solve  basis @ projection = centroids  for projection (least squares).
            # cuSOLVER's `lstsq` is flaky on some GPU/driver combos (e.g.
            # H200 + CUDA 12.x); fall back to CPU + back-to-device, which
            # is fast (the basis is (M, basis_dim) with M ≤ 8K — milliseconds).
            try:
                sol = torch.linalg.lstsq(self.basis, centroids).solution
            except RuntimeError as exc:
                if "cusolver" in str(exc).lower() or "magma" in str(exc).lower():
                    sol = torch.linalg.lstsq(
                        self.basis.detach().cpu(),
                        centroids.detach().cpu(),
                    ).solution.to(centroids.device)
                else:
                    raise
            self.projection.data.copy_(sol)
            self.codebook.weight.data.copy_(self.basis @ self.projection)
        else:
            self.codebook.weight.data.copy_(centroids)
            if self.vq_update == 'ema':
                self.ema_weight.data.copy_(centroids)
        # Give every code one unit of count so dead-code revival doesn't fire
        # immediately on the first batch.
        self.ema_count.data.fill_(1.0)
        # FIXED(final_fix_list A6, 2026-10-19): the usage window starts with the
        # k-means codebook.
        self.usage_window.zero_()
        self.window_steps.zero_()


@torch.no_grad()
def codebook_duplicate_stats(codebook: torch.Tensor, rel_tol: float = 1e-3,
                             cos_threshold: float = 0.9999) -> dict:
    """Exact and near duplicate rows of one codebook.

    FIXED(final_fix_list A6, 2026-10-19): new; used by the trainer's checkpoint check.
    exact_duplicates = M - number of distinct rows (redundant rows);
    near_duplicates  = number of codes j with an earlier code i < j at Euclidean
                       distance < `rel_tol` (1e-3) x the median code norm (exact
                       duplicates included). This is the error criterion;
    cos_near_duplicates = the same count with cosine similarity > `cos_threshold`
                       (0.9999). Reported only: at code norms of 10-15 a cosine of
                       0.9999 still allows a distance of ~0.2, the spacing of genuinely
                       distinct, used codes (smoke run 2026-10-19: two live codes at
                       cosine 0.99991, distance 0.53, with the closest pair of the
                       codebook at 0.39);
    min_dist_rel     = smallest pairwise distance / median code norm.
    """
    cb = codebook.detach().float().cpu()
    M = cb.shape[0]
    exact = M - int(torch.unique(cb, dim=0).shape[0])
    med = float(cb.norm(dim=1).median().clamp_min(1e-12))
    # exact Euclidean distances (the matrix-product shortcut gives ~1e-2 for equal rows)
    d = torch.cdist(cb, cb, compute_mode='donot_use_mm_for_euclid_dist')
    upper = torch.ones(M, M, dtype=torch.bool).triu(diagonal=1)
    near = int(((d < rel_tol * med) & upper).any(dim=0).sum())
    normed = F.normalize(cb, dim=1)
    cos_near = int((((normed @ normed.T) > cos_threshold) & upper).any(dim=0).sum())
    min_rel = float(d[upper].min()) / med if M > 1 else float("inf")
    return {"n_codes": M, "exact_duplicates": exact, "near_duplicates": near,
            "cos_near_duplicates": cos_near, "min_dist_rel": min_rel}


@torch.no_grad()
def codebook_health(vq_levels, tokens_per_level=None, rel_tol: float = 1e-3) -> dict:
    """Per-level codebook check for a checkpoint.

    FIXED(final_fix_list A6, 2026-10-19): new. For every `VectorQuantizer` level:
    exact / near duplicate codes (`codebook_duplicate_stats`: near = distance below
    1e-3 x median code norm; the cosine > 0.9999 count is reported too), the number of
    codes with zero
    usage in `tokens_per_level` (validation tokens; None = not computed), and the
    revival counters since the start of training. Returns
    {"levels": [...], "error": None | str}; "error" is set when any level has a
    duplicate code (the trainer then refuses to save the checkpoint).
    """
    levels, bad = [], []
    for l, vq in enumerate(vq_levels):
        st = codebook_duplicate_stats(vq._effective_codebook(), rel_tol)
        st["level"] = l
        if tokens_per_level is not None:
            used = torch.bincount(tokens_per_level[l].reshape(-1).cpu(), minlength=vq.num_codes)
            st["zero_usage_val"] = int((used == 0).sum())
        else:
            st["zero_usage_val"] = None
        st["revival_checks"] = int(vq.n_revival_checks)
        st["revived_codes"] = int(vq.n_revived_codes)
        st["partial_batches_skipped"] = int(vq.n_partial_skipped)
        levels.append(st)
        if st["exact_duplicates"] or st["near_duplicates"]:
            bad.append(f"L{l}: {st['exact_duplicates']} exact / {st['near_duplicates']} near")
    return {"levels": levels,
            "error": ("duplicate codes: " + ", ".join(bad)) if bad else None}


# ======================================================================
# Losses
# ======================================================================

def _build_perms(P: int) -> torch.Tensor:
    """Precompute all P! permutations as a (P!, P) index tensor."""
    from itertools import permutations
    return torch.tensor(list(permutations(range(P))), dtype=torch.long)


# Cache permutations for common P values
_PERM_CACHE: dict[int, torch.Tensor] = {}


def hungarian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batched permutation-optimal matching loss for set reconstruction.
    [NOT USED by the reported model: this is the square, MSE version. The
    reported model uses the rectangular Hungarian + FAPE loss of
    ensembits/losses.py, Eq. eq:hungarian / eq:fape.]

    For small P (≤8), brute-forces all P! permutations in parallel on GPU.
    Falls back to scipy linear_sum_assignment for P>8.
    """
    B, P, D = pred.shape

    # Brute-force is O(B * P! * P * D) memory. P=8 ⇒ 40k perms — at large
    # batch sizes (e.g. 4096 with D=50) this needs ~260 GB. Threshold lowered
    # to P=6 (720 perms) to stay safely within GPU memory at the configs we
    # actually train; P=7..10 fall through to scipy.
    if P <= 6:
        # Brute-force: try all P! permutations, pick the best per sample
        if P not in _PERM_CACHE or _PERM_CACHE[P].device != pred.device:
            _PERM_CACHE[P] = _build_perms(P).to(pred.device)
        perms = _PERM_CACHE[P]  # (n_perms, P)

        # Expand: target_permed[b, perm_idx, p, :] = target[b, perms[perm_idx, p], :]
        target_permed = target[:, perms]  # (B, n_perms, P, D)
        pred_exp = pred.unsqueeze(1).expand_as(target_permed)  # (B, n_perms, P, D)

        # MSE per permutation per sample
        costs = (pred_exp - target_permed).pow(2).mean(dim=(-1, -2))  # (B, n_perms)
        best_perm_idx = costs.argmin(dim=1)  # (B,)
        best_perms = perms[best_perm_idx]  # (B, P)

        # Gather best-matched targets
        batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand_as(best_perms)
        matched_target = target[batch_idx, best_perms]  # (B, P, D)
        return F.mse_loss(pred, matched_target)
    else:
        # Fallback for large P (P > 8).  We can't brute-force P! permutations
        # on GPU, so we use scipy's Hungarian algorithm on the CPU.  The
        # original implementation did one .cpu() sync per batch item, which
        # costs ~50× more than P=5 because every iteration blocks on the GPU.
        # Fix: compute ALL (B, P, P) cost matrices at once on GPU, transfer
        # them in a single sync, then loop scipy in a tight Python loop (no
        # GPU sync inside), and finally apply permutations back on GPU.
        costs = torch.cdist(pred, target).pow(2)          # (B, P, P)
        costs_np = costs.detach().cpu().numpy()

        perms_np = np.empty((B, P), dtype=np.int64)
        for b in range(B):
            _, col_idx = linear_sum_assignment(costs_np[b])
            perms_np[b] = col_idx

        perms = torch.from_numpy(perms_np).to(pred.device)
        batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand_as(perms)
        matched_target = target[batch_idx, perms]          # (B, P, D)
        return F.mse_loss(pred, matched_target)


def _permute_frames(x: torch.Tensor) -> torch.Tensor:
    """Apply an independent random permutation of the P frame axis per sample."""
    B, P, _ = x.shape
    perms = torch.stack(
        [torch.randperm(P, device=x.device) for _ in range(B)]
    )  # (B, P)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(perms)
    return x[batch_idx, perms]


# ======================================================================
# Main tokenizer
# ======================================================================

class VQVAETokenizer(nn.Module):
    """Ensembits VQ-VAE tokenizer.  [NOT USED by the reported model: single
    codebook variant; the planned VQ-2048 ablation (code_fix E1) uses RVQ with
    `--codebook-sizes 2048` instead.]

    Encoder: SetTransformerEncoder (default) or SequenceEncoder.
    Bottleneck: Vector quantization with EMA updates.
    Decoder: MLP outputting P reconstructed states.
    Loss: Hungarian (default) or pointwise MSE.

    Frame-order invariance options:
      - encoder_type='set_transformer': architectural invariance.
      - encoder_type='sequence' + permute_input=True: learned invariance via
        per-step random permutation of the P frames.

    With permute_input=True, loss_type must be 'hungarian' (pointwise MSE
    against shuffled input is incoherent).
    """

    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        codebook_size: int = 256,
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
        aminoaseed: bool = False,
        basis_dim: int | None = None,
        # Accepted-but-ignored kwargs for backward compat (Sinkhorn was the
        # old cross-protein contrastive head; not currently in use).
        sinkhorn_weight: float = 0.0,
        sinkhorn_epsilon: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.codebook_size = codebook_size
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

        self.vq = VectorQuantizer(
            codebook_size, latent_dim, commitment_cost,
            vq_update=vq_update, aminoaseed=aminoaseed, basis_dim=basis_dim)
        self.decoder = SetDecoder(latent_dim, input_dim, num_prototypes, hidden_dim,
                                   n_layers=n_decoder_layers)

    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'hungarian':
            return hungarian_loss(recon, target)
        elif self.loss_type == 'pointwise':
            return F.mse_loss(recon, target)
        else:
            raise ValueError(
                f"loss_type must be 'hungarian' or 'pointwise', got {self.loss_type!r}")

    def forward(self, x: torch.Tensor, sinkhorn_weight: float = 0.0,
                sinkhorn_epsilon: float = 0.1) -> dict:
        """
        Args:
            x: (B, P, D) input conformational sets.
            sinkhorn_weight, sinkhorn_epsilon: accepted for backward compat; ignored.

        Returns:
            dict with keys: recon, tokens, recon_loss, vq_loss, sinkhorn_loss, total_loss.
        """
        if self.permute_input and self.training:
            x = _permute_frames(x)

        z = self.encoder(x)
        q, tokens, vq_loss = self.vq(z)
        recon = self.decoder(q)

        recon_loss = self._recon_loss(recon, x)
        total_loss = recon_loss + vq_loss

        # sinkhorn_loss kept (as 0) for back-compat with training history dicts.
        sk_loss = torch.tensor(0.0, device=x.device)

        return {
            'recon': recon,
            'tokens': tokens,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'sinkhorn_loss': sk_loss,
            'total_loss': total_loss,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, tokens, _ = self.vq(z)
        return tokens

    def encode_all(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Duck-typed to match RVQVAETokenizer: single-level returns [tokens]."""
        return [self.encode(x)]

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.vq.codebook(tokens)
        return self.decoder(q)

    def utilization(self, tokens: torch.Tensor, level: int = 0) -> float:
        return len(tokens.unique()) / self.codebook_size

    def perplexity(self, tokens: torch.Tensor, level: int = 0) -> float:
        counts = torch.bincount(tokens, minlength=self.codebook_size).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        return entropy.exp().item()


# ======================================================================
# Residual VQ-VAE
# ======================================================================

class ResidualVQVAETokenizer(nn.Module):
    """Ensembits tokenizer with Residual Vector Quantization.
    [NOT USED: an older RVQ class; the reported model is rvqvae.RVQVAETokenizer.]

    Same encoder/loss/permutation knobs as VQVAETokenizer. The primary token
    is used for search; the full tuple gives higher-fidelity reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        num_prototypes: int,
        primary_codebook_size: int = 512,
        refinement_codebook_size: int = 128,
        n_refinement: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        commitment_cost: float = 0.25,
        encoder_type: str = 'set_transformer',
        loss_type: str = 'hungarian',
        permute_input: bool = False,
        n_queries: int = 4,
        n_heads: int = 4,
        sinkhorn_weight: float = 0.0,
        sinkhorn_epsilon: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.primary_codebook_size = primary_codebook_size
        self.refinement_codebook_size = refinement_codebook_size
        self.n_levels = 1 + n_refinement
        self.loss_type = loss_type
        self.permute_input = permute_input

        if encoder_type == 'set_transformer':
            self.encoder = SetTransformerEncoder(
                input_dim, hidden_dim, latent_dim,
                n_queries=n_queries, n_heads=n_heads)
        elif encoder_type == 'sequence':
            self.encoder = SequenceEncoder(
                input_dim, num_prototypes, hidden_dim, latent_dim)
        else:
            raise ValueError(
                f"encoder_type must be 'set_transformer' or 'sequence', got {encoder_type!r}")

        # Level 0: primary (large)
        self.vq_levels = nn.ModuleList([
            VectorQuantizer(primary_codebook_size, latent_dim, commitment_cost)
        ])
        # Levels 1+: refinement (smaller)
        for _ in range(n_refinement):
            self.vq_levels.append(
                VectorQuantizer(refinement_codebook_size, latent_dim, commitment_cost))

        self.decoder = SetDecoder(latent_dim, input_dim, num_prototypes, hidden_dim)

    def _quantize(self, z: torch.Tensor) -> tuple:
        """Run residual quantization. Returns (quantized_sum, token_list, vq_loss)."""
        quantized = torch.zeros_like(z)
        residual = z
        all_tokens = []
        total_vq_loss = torch.tensor(0.0, device=z.device)

        for vq in self.vq_levels:
            q, tokens, vq_loss = vq(residual)
            quantized = quantized + q
            residual = z - quantized  # always relative to original z
            all_tokens.append(tokens)
            total_vq_loss = total_vq_loss + vq_loss

        return quantized, all_tokens, total_vq_loss

    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'hungarian':
            return hungarian_loss(recon, target)
        elif self.loss_type == 'pointwise':
            return F.mse_loss(recon, target)
        else:
            raise ValueError(
                f"loss_type must be 'hungarian' or 'pointwise', got {self.loss_type!r}")

    def forward(self, x: torch.Tensor, sinkhorn_weight: float = 0.0,
                sinkhorn_epsilon: float = 0.1) -> dict:
        if self.permute_input and self.training:
            x = _permute_frames(x)

        z = self.encoder(x)
        quantized, all_tokens, vq_loss = self._quantize(z)

        # Straight-through estimator on the sum
        q_st = z + (quantized - z).detach()
        recon = self.decoder(q_st)

        recon_loss = self._recon_loss(recon, x)
        total_loss = recon_loss + vq_loss

        sk_loss = torch.tensor(0.0, device=x.device)

        return {
            'recon': recon,
            'tokens': all_tokens[0],        # primary token
            'all_tokens': all_tokens,        # full tuple
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'sinkhorn_loss': sk_loss,
            'total_loss': total_loss,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to primary tokens only."""
        z = self.encoder(x)
        _, all_tokens, _ = self._quantize(z)
        return all_tokens[0]

    def encode_all(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode to full token tuple (primary + refinements)."""
        z = self.encoder(x)
        _, all_tokens, _ = self._quantize(z)
        return all_tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode from primary tokens only (coarse)."""
        q = self.vq_levels[0].codebook(tokens)
        return self.decoder(q)

    def decode_all(self, all_tokens: list[torch.Tensor]) -> torch.Tensor:
        """Decode from full token tuple (high fidelity)."""
        q = torch.zeros_like(self.vq_levels[0].codebook(all_tokens[0]))
        for vq, tokens in zip(self.vq_levels, all_tokens):
            q = q + vq.codebook(tokens)
        return self.decoder(q)

    def utilization(self, tokens: torch.Tensor, level: int = 0) -> float:
        M = self.primary_codebook_size if level == 0 else self.refinement_codebook_size
        return len(tokens.unique()) / M

    def perplexity(self, tokens: torch.Tensor, level: int = 0) -> float:
        M = self.primary_codebook_size if level == 0 else self.refinement_codebook_size
        counts = torch.bincount(tokens, minlength=M).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        return entropy.exp().item()


# ======================================================================
# Training
# ======================================================================

# [NOT USED] Legacy single-file training loop for early experiments; the
# reported model was trained with train/train.py.
def train_vqvae(
    model: VQVAETokenizer,
    data: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    verbose: bool = True,
    # Accepted-but-ignored kwargs for backward compat.
    sinkhorn_weight: float = 0.0,
    sinkhorn_start_epoch: int = 50,
    sinkhorn_epsilon: float = 0.1,
) -> dict:
    """Train the VQ-VAE tokenizer.

    Args:
        model: VQVAETokenizer instance.
        data: (N, P, D) numpy array of conformational sets.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: torch device.
        verbose: Print progress.
        sinkhorn_*: accepted for backward compat; ignored.

    Returns:
        History dict with loss curves.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.tensor(data, dtype=torch.float32)
    N = dataset.shape[0]

    history = {'recon_loss': [], 'vq_loss': [], 'sinkhorn_loss': [],
               'total_loss': [], 'utilization': [], 'perplexity': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_recon = 0.0
        epoch_vq = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            batch = dataset[idx].to(device)

            out = model(batch)
            optimizer.zero_grad()
            out['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_recon += out['recon_loss'].item()
            epoch_vq += out['vq_loss'].item()
            n_batches += 1

        # Eval metrics
        model.eval()
        with torch.no_grad():
            all_tokens = model.encode(dataset.to(device))
            util = model.utilization(all_tokens)
            ppl = model.perplexity(all_tokens)

        avg_recon = epoch_recon / n_batches
        avg_vq = epoch_vq / n_batches

        history['recon_loss'].append(avg_recon)
        history['vq_loss'].append(avg_vq)
        history['sinkhorn_loss'].append(0.0)
        history['total_loss'].append(avg_recon + avg_vq)
        history['utilization'].append(util)
        history['perplexity'].append(ppl)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"recon={avg_recon:.6f}, vq={avg_vq:.6f}, "
                  f"util={util:.2%}, ppl={ppl:.1f}")

    return history
