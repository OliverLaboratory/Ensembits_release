"""Reference implementations and wrappers of the baseline tokenizers.

What it does
------------
Collects the per-baseline helpers so scripts can write
`from ensembits.baselines import ProToken` (the ProToken token builders do).
The baseline taxonomy of the paper:

    Single-frame    aa, 3di_tokens, random, aminoaseed, protoken, esm3struct
    Multi-frame     protprofile_K, vote_3di, Ensembits P=full

Modules:
    aa                    `aa.py`                AA one-hot
    random                `random_tokens.py`     deterministic random tokens
    3di_tokens            `mini3di_baselines.py` single-frame mini3di
    vote_3di              `mini3di_baselines.py` multi-frame mini3di plurality
    protprofile_K         `mini3di_baselines.py` multi-frame mini3di histogram
    aminoaseed            `aminoaseed.py`        StructTokenBench checkpoint wrapper
    protoken              `protoken.py`          ProToken-1.0 wrapper (used by data_prep/baselines)
    esm3struct            `esm3struct.py`        ESM3 structure encoder wrapper
    ideal backbone        `_backbone.py`         N/C/O from C-alpha (see its header)
    real-or-ideal loader  `_real_or_ideal_bb.py` real MISATO / mdCATH backbone per frame

Every module imports only numpy at import time; the heavy third-party packages
(mini3di, torch, omegaconf, esm, tensorflow, jax) are imported inside the
functions / constructors that need them.

Paper
-----
`appendix.tex` \\section{Baselines} \\label{app:baselines} (single-frame and
multi-frame methods).

Origin
------
Copied from `Ensembits/utils/baselines/__init__.py`. Imports unchanged (they are
relative); docstring rewritten. The scripts in this codebase use only
`ProToken` (through this package) and `_backbone.reconstruct_backbone_ideal`;
`aa`, `random_tokens`, `mini3di_baselines`, `aminoaseed` and `esm3struct` are
copied because this `__init__` imports them.

Used by the reported model: no (baselines only).

Inputs / outputs
----------------
none (library)

How to run
----------
    from ensembits.baselines import ProToken
"""
from .aa import aa_features, aa_features_per_pid
from .random_tokens import random_tokens, random_tokens_per_pid, random_codebook
from .mini3di_baselines import (
    have_mini3di,
    three_di_tokens,
    vote_3di,
    protprofile_k,
    mini3di_centroids,
)
# External-checkpoint wrappers — imports are kept lazy so they don't
# break the package for users who don't need the upstream deps.
from .aminoaseed import AminoAseed
from .protoken import ProToken
from .esm3struct import ESM3Struct
from .retrained import RetrainedTokenizer

__all__ = [
    "aa_features", "aa_features_per_pid",
    "random_tokens", "random_tokens_per_pid", "random_codebook",
    "have_mini3di",
    "three_di_tokens", "vote_3di", "protprofile_k",
    "mini3di_centroids",
    "AminoAseed", "ProToken", "ESM3Struct", "RetrainedTokenizer",
]
