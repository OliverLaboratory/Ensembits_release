"""EMA vector quantizer matching ESM3's structure-tokenizer codebook.

What it does
------------
Implements the van den Oord EMA codebook update rule with ESM3's
hyperparameters (ema_decay=0.99, eps=1e-5, no_random_restart=True).
The released `esm` package ships only the inference path of EMACodebook
(the training branch asserts "Not implemented"), so the update had to
be reconstructed from the surviving buffers (N, z_avg).

Conforms to StructTokenBench's BaseQuantizer interface so VQVAEModel
can use it unmodified: forward(z) -> (quantized_z, indices, loss, metrics).

Paper
-----
`appendix.tex` static retrain appendix: "ESM3-style retrained" uses this
quantizer with 4096 codes x 128 dims.

Origin
------
Copied from submission_exp/src/rebuttal_analyses/sametrain_baselines/ema_quantizer.py
(sha256 of source: see provenance). Changed: docstring, import path guards,
no functional changes.

How to run
----------
Not run standalone; imported by train_static.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from vqvae.quantizer_module import BaseQuantizer, get_codebook_utility


class EMAQuantizer(BaseQuantizer):
    """van den Oord EMA codebook, ESM3 hyperparameters."""

    def __init__(self, ema_decay: float = 0.99, eps: float = 1e-5,
                 no_random_restart: bool = True, restart_thres: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.ema_decay = ema_decay
        self.eps = eps
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        self.codebook.weight.requires_grad = False
        self.register_buffer("N", torch.zeros(self.codebook_size))
        self.register_buffer("z_avg", self.codebook.weight.data.clone())

        if self.use_linear_project:
            raise ValueError("EMAQuantizer is the ESM3-style quantizer; "
                             "use_linear_project belongs to AminoAseed")

    def _tile(self, x):
        d, ew = x.shape
        if d < self.codebook_size:
            n_repeats = (self.codebook_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        self._need_init = False
        flat_inputs = z.reshape(-1, self.codebook_embed_size)
        y = self._tile(flat_inputs)
        _k_rand = y[torch.randperm(y.shape[0])][: self.codebook_size]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.codebook.weight.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.codebook_size, device=_k_rand.device))

    @torch.no_grad()
    def _ema_update(self, flat_inputs, flat_indices):
        """Standard EMA update with Laplace-smoothed cluster sizes."""
        onehot = F.one_hot(flat_indices, self.codebook_size).type(flat_inputs.dtype)
        counts = onehot.sum(dim=0)
        z_sum = onehot.t() @ flat_inputs

        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(z_sum, op=dist.ReduceOp.SUM)

        self.N.data.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.z_avg.data.mul_(self.ema_decay).add_(z_sum, alpha=1 - self.ema_decay)

        n = self.N.sum()
        weights = (self.N + self.eps) / (n + self.codebook_size * self.eps) * n
        self.codebook.weight.data.copy_(self.z_avg / weights.unsqueeze(1))

        if not self.no_random_restart:
            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.codebook_size]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)
            usage = (self.N.view(self.codebook_size, 1) >= self.restart_thres).type(
                flat_inputs.dtype)
            self.codebook.weight.data.mul_(usage).add_(_k_rand * (1 - usage))

        return counts

    def forward(self, z: torch.Tensor):
        if self._need_init and self.training:
            self._init_embeddings(z)

        quantized_indices = self.embedding2indices(z)
        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        flat_indices = rearrange(quantized_indices, "b l -> (b l)")

        quantized_z = F.embedding(flat_indices, self.codebook.weight)

        metrics = {}
        commitment_loss = F.mse_loss(quantized_z.detach(), flat_z)
        loss = self.loss_weight["commitment_loss_weight"] * commitment_loss
        metrics["commitment_loss"] = commitment_loss

        if self.training:
            self._ema_update(flat_z.detach(), flat_indices)

        quantized_z = flat_z + (quantized_z - flat_z).detach()
        quantized_z = rearrange(quantized_z, "(b l) h -> b l h",
                                b=batch_size, l=seq_length, h=dim_size)

        metrics.update(get_codebook_utility(flat_indices, self.codebook.weight))
        return quantized_z, quantized_indices, loss, metrics
