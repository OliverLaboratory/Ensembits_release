"""Neural network parts of Ensembits.

  vqvae.py   SetTransformerEncoder (set encoder), SetDecoder, VectorQuantizer
             (one codebook with EMA updates, dead-code revival, k-means init)
  rvqvae.py  RVQVAETokenizer: encoder -> 3 stacked VectorQuantizers -> decoder
             (the model reported in the paper)
  geom_attn_encoder.py: alternative encoder option of the trainer;
             NOT used by the reported model.

Origin: copied from `Ensembits/ensembits/tokenizer/` (the package the trainer
imports). Logic unchanged; comments added.
"""
from .vqvae import (
    VQVAETokenizer,
    VectorQuantizer,
    SetTransformerEncoder,
    SequenceEncoder,
    SetDecoder,
    hungarian_loss,
)
from .rvqvae import RVQVAETokenizer

__all__ = [
    "RVQVAETokenizer", "VQVAETokenizer",
    "VectorQuantizer", "SetTransformerEncoder", "SequenceEncoder", "SetDecoder",
    "hungarian_loss",
]
