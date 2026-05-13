"""Amino-acid one-hot baseline (sequence-only, no structure).

For a sequence of length L, returns an (L, 21) float32 matrix — 20
canonical amino acids plus a single "unknown" channel for any
non-canonical letter.

Unaffected by the structural-pipeline bug-fix campaign (it doesn't
touch backbone coords at all).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_UNKNOWN_INDEX = 20
_AA_TO_INDEX = {a: i for i, a in enumerate(ALPHABET)}


def aa_features(sequence: str) -> np.ndarray:
    """``sequence`` → ``(L, 21)`` float32 one-hot."""
    L = len(sequence)
    out = np.zeros((L, 21), dtype=np.float32)
    for i, ch in enumerate(sequence):
        out[i, _AA_TO_INDEX.get(ch.upper(), _UNKNOWN_INDEX)] = 1.0
    return out


def aa_features_per_pid(seq_by_pid: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Batch wrapper: ``{pid: seq}`` → ``{pid: (L, 21) one-hot}``."""
    return {pid: aa_features(seq) for pid, seq in seq_by_pid.items()}
