"""Ensembits: a residual VQ-VAE tokenizer for protein conformational ensembles.

Package map (paper section → module):
  descriptor.py      256-D per-residue, per-frame descriptor
                     (`appendix.tex` app:descriptor-esm3; `1_method.tex` Dynamical mode)
  ideal_backbone.py  ideal-geometry N/C from C-alpha (`appendix.tex` Conventions)
  tokenizer/         set encoder, residual quantizer, decoder
                     (`1_method.tex` Tokenizer; `appendix.tex` app:RVQVAE, app:model-impl)
  losses.py          Hungarian matching + FAPE + dihedral term
                     (`1_method.tex` Eq. eq:hungarian, eq:fape)
  featurize.py       load the checkpoint, tokenize ensembles
  paths.py           where code, data and the checkpoint live

The SFTD objective (`1_method.tex` Eq. eq:sftd) lives in the training loop,
`train/train.py`.
"""
from .tokenizer.vqvae import VQVAETokenizer
from .tokenizer.rvqvae import RVQVAETokenizer
from . import tokenizer

__all__ = ["RVQVAETokenizer", "VQVAETokenizer", "tokenizer"]
