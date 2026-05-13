"""Random-token baseline — deterministic per ``(pid, residue, seed)``.

For each ``(pid, residue)`` we emit a single integer in ``[0, K)``.
The integer is a deterministic function of ``(pid, residue, seed)``,
so reruns of the probe see the same "random" assignment and the
``seed`` parameter is what shuffles between repeats.

``K`` should match the codebook size of the model you're comparing
against — the manuscript's random baseline isolates the contribution
of the *learned* vocabulary from the downstream classifier capacity,
which only makes sense at matched K.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _seed_for_pid(pid: str, base_seed: int) -> int:
    """Stable 32-bit FNV-1a hash of ``(base_seed, pid)``."""
    h = 2166136261 ^ (base_seed & 0xFFFFFFFF)
    h = (h * 16777619) & 0xFFFFFFFF
    for ch in pid:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def random_tokens(pid: str, length: int, K: int = 2048,
                   seed: int = 0) -> np.ndarray:
    """``(length,) int64`` of deterministic random tokens in ``[0, K)``."""
    h = _seed_for_pid(pid, seed)
    rng = np.random.default_rng(h)
    return rng.integers(low=0, high=K, size=length, dtype=np.int64)


def random_tokens_per_pid(lengths: Mapping[str, int],
                           K: int = 2048, seed: int = 0
                           ) -> dict[str, np.ndarray]:
    """Apply :func:`random_tokens` to every pid in the mapping."""
    return {pid: random_tokens(pid, L, K=K, seed=seed)
            for pid, L in lengths.items()}


def random_codebook(K: int, d: int = 128, seed: int = 0) -> np.ndarray:
    """``(K, d)`` float32 codebook with standard-normal entries.

    Pair with :func:`random_tokens` to feed
    ``probes/rmsf.py --codebook ...``.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((K, d)).astype(np.float32)
