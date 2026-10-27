"""ESM3-free geometry (`ensembits/geometry.py`) against ESM3's own utilities.

What it does
------------
Computes the descriptor with backend="native" and backend="esm" and requires
bit-identical output. Skipped when the `esm` package (or the ESM3-open weights
that its kNN helper loads) is not available. The broader check on 136 cached
samples and edge cases is `checks/check_geometry_vs_esm.py`.

Paper
-----
`appendix.tex` \\label{app:descriptor-esm3} (ESM3 frames and kNN).

How to run
----------
  python -m pytest tests/test_geometry.py
"""
import pytest
import torch

from ensembits.descriptor import _descriptor


def test_native_equals_esm(ensemble):
    pytest.importorskip("esm")
    try:
        ref = _descriptor(ensemble, "cpu", backend="esm")
    except Exception as e:                       # weights not downloadable, no login, ...
        pytest.skip(f"ESM3 backend unavailable: {e}")
    nat = _descriptor(ensemble, "cpu", backend="native")
    assert torch.equal(nat, ref)
