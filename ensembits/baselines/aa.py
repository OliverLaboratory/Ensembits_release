"""Amino-acid one-hot baseline (single-frame), as a library function.

What it does
------------
For a sequence of length L, returns an (L, 21) float32 matrix: 20 standard
amino acids plus one "unknown" channel for any other letter.

Paper
-----
`appendix.tex` \\label{app:baselines}, *aa* ("20-dimensional one-hot").
The 21st (unknown) channel is the discrepancy logged as require_fix #22.

Origin
------
Copied from `Ensembits/utils/baselines/aa.py`. Logic unchanged; docstring
rewritten. Not called by the scripts of this codebase (the probes build the aa
feature inline); copied because `ensembits/baselines/__init__.py` imports it.

Used by the reported model: no.

Inputs / outputs
----------------
none (library)

How to run
----------
    from ensembits.baselines import aa_features
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_UNKNOWN_INDEX = 20
_AA_TO_INDEX = {a: i for i, a in enumerate(ALPHABET)}


def aa_features(sequence: str) -> np.ndarray:
    """`sequence` (str of length L) → `(L, 21) float32` one-hot."""
    L = len(sequence)
    out = np.zeros((L, 21), dtype=np.float32)
    for i, ch in enumerate(sequence):
        idx = _AA_TO_INDEX.get(ch.upper(), _UNKNOWN_INDEX)
        out[i, idx] = 1.0
    return out


def aa_features_per_pid(seq_by_pid: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Apply `aa_features` to every entry of `{pid: sequence}`. Useful
    for the per-pid input shape that probe_rmsf.py accepts."""
    return {pid: aa_features(seq) for pid, seq in seq_by_pid.items()}
