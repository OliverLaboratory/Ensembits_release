"""Ensembits — structural tokenizer for protein conformational ensembles."""

from .descriptors import compute_esm3_descriptor
from .featurize import (
    Ensembits,
    codebook_size,
    load_bb_for_pid,
    load_model,
    onehot_fulltoken,
    onehot_tokens,
    tokenize_ensemble,
    tokenize_ensemble_all,
)
from .tokenizer import (
    RVQVAETokenizer,
    SetDecoder,
    SetTransformerEncoder,
    VectorQuantizer,
    hungarian_loss,
)
from .trainer import TrainConfig, standardize, train_rvqvae

__version__ = "0.1.0"

__all__ = [
    # Model
    "RVQVAETokenizer",
    "SetDecoder",
    "SetTransformerEncoder",
    "VectorQuantizer",
    "hungarian_loss",
    # Descriptor
    "compute_esm3_descriptor",
    # Loader + inference
    "Ensembits",
    "load_model",
    "tokenize_ensemble",
    "tokenize_ensemble_all",
    "load_bb_for_pid",
    "codebook_size",
    "onehot_tokens",
    "onehot_fulltoken",
    # Training
    "TrainConfig",
    "train_rvqvae",
    "standardize",
]
