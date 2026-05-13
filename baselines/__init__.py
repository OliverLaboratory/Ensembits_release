"""Baseline tokenizer implementations (post-fix).

Public modules:

- :mod:`baselines._backbone` — ideal-geometry reconstruction, real-bb
  cache lookup, peptide-bond chain split, tetrahedral Cβ helper.
- :mod:`baselines.aa` — amino-acid one-hot (sequence-only).
- :mod:`baselines.random` — deterministic random tokens per pid + seed.
- :mod:`baselines.mini3di` — Foldseek 3Di tokens, plurality vote, and
  histogram (60-D ProtProfile target — bit-equivalent foldseek+mmseqs
  builder is :mod:`scripts.build_protprofile_foldseek`).
- :mod:`baselines.aminoaseed` — StructTokenBench VQ-VAE wrapper.
- :mod:`baselines.esm3struct` — ESM3 structure-track encoder wrapper.
- :mod:`baselines.protoken` — ProToken-1.0 TF-SavedModel wrapper.
- :mod:`baselines.esm2` — ESM2-650M pseudo-LL scorer (PG α-blend only).

All structural baselines accept an optional ``bb=(L, 4, 3)`` real
backbone argument and a ``chain_split=True`` keyword to enable
per-chain tokenization on multi-chain proteins (a hard requirement
for MISATO, where 52 % of pids are multi-chain).
"""

from . import _backbone, aa, esm2, esm3struct, mini3di, protoken
from . import random as random_baseline

__all__ = [
    "_backbone",
    "aa",
    "random_baseline",
    "mini3di",
    "aminoaseed",
    "esm3struct",
    "protoken",
    "esm2",
]
