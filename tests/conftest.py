"""Shared pytest fixtures for the Ensembits codebase.

What it does
------------
Puts the codebase root on sys.path and loads the small test ensemble in
`tests/fixtures/` (mdCATH domain 1ctfA00, 10 frames, 68 residues, N/CA/C only;
provenance in `tests/fixtures/README.json`).

Paper
-----
not described in the paper (infrastructure only)

How to run
----------
  python -m pytest tests            # from the codebase root
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def ensemble():
    """(P=10, L=68, 3, 3) float32 N/CA/C coordinates of 1ctfA00."""
    return torch.from_numpy(np.load(FIXTURES / "1ctfA00_P10_ncac.npz")["ncac"])


@pytest.fixture(scope="session")
def reference_tokens():
    """(L=68, N_q=3) tokens of the checkpoint in ckpt/ at P = 10."""
    return np.load(FIXTURES / "1ctfA00_P10_ncac.npz")["tokens_P10_reported_ckpt"]


def random_rotation(seed: int = 0) -> torch.Tensor:
    """A random proper rotation matrix (det = +1)."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.float()
