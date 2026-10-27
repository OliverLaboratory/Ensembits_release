"""Random-token baseline (single-frame, deterministic per pid), as a library function.

What it does
------------
For each (pid, residue) emits one integer in [0, K), a deterministic function
of (pid, residue, seed) via a 32-bit FNV-1a hash; `random_codebook` gives a
(K, d) standard-normal codebook to embed them.

Paper
-----
`appendix.tex` \\label{app:baselines}, *Random* ("random integer in
{0, ..., K-1}, with K matched to the codebook size"). The MISATO probes use a
per-residue 2-D Gaussian vector instead (require_fix #22).

Origin
------
Copied from `Ensembits/utils/baselines/random_tokens.py`. Logic unchanged;
docstring rewritten. Not called by the scripts of this codebase; copied because
`ensembits/baselines/__init__.py` imports it.

Used by the reported model: no.

Inputs / outputs
----------------
none (library)

How to run
----------
    from ensembits.baselines import random_tokens, random_codebook
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _seed_for_pid(pid: str, base_seed: int) -> int:
    """Stable 32-bit FNV-1a hash of `(base_seed, pid)`."""
    h = 2166136261 ^ (base_seed & 0xFFFFFFFF)
    h = (h * 16777619) & 0xFFFFFFFF
    for ch in pid:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def random_tokens(pid: str, length: int, K: int = 2048,
                   seed: int = 0) -> np.ndarray:
    """`(length,) int64` of deterministic random tokens in `[0, K)`."""
    h = _seed_for_pid(pid, seed)
    rng = np.random.default_rng(h)
    return rng.integers(low=0, high=K, size=length, dtype=np.int64)


def random_tokens_per_pid(lengths: Mapping[str, int],
                            K: int = 2048,
                            seed: int = 0) -> dict[str, np.ndarray]:
    """Apply `random_tokens` to every `pid` in the `lengths` mapping."""
    return {pid: random_tokens(pid, L, K=K, seed=seed)
            for pid, L in lengths.items()}


def random_codebook(K: int, d: int = 128, seed: int = 0) -> np.ndarray:
    """A `(K, d) float32` codebook of standard-normal entries.
    Pair with `random_tokens` to feed `probe_rmsf.py --codebook ...`."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((K, d)).astype(np.float32)
