"""mdCATH-div RMSF label with a converged superposition (code_fix P3).

What it does
------------
  * the label does not depend on the starting rigid pose of each frame;
  * it equals the RMSF about the converged mean (no change after more rounds);
  * on frames that only differ by rigid motions the label is zero.

Paper
-----
`2_results.tex` \\label{result:rmsf} (label definition).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
from build_mdcath_div_rmsf_labels import converged_rmsf  # noqa: E402


def _rot(seed):
    g = np.random.default_rng(seed)
    q, r = np.linalg.qr(g.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _ensemble(seed=0, F=5, L=60):
    g = np.random.default_rng(seed)
    base = np.cumsum(g.normal(size=(L, 3)) * 2.0, axis=0)
    return np.stack([base + g.normal(size=(L, 3)) * (0.3 + 0.2 * f) for f in range(F)])


def test_pose_independent():
    X = _ensemble()
    r0, _, ok = converged_rmsf(X)
    assert ok
    Y = np.stack([X[f] @ _rot(10 + f).T + 50.0 * np.random.default_rng(f).normal(size=3)
                  for f in range(X.shape[0])])
    r1, _, _ = converged_rmsf(Y)
    np.testing.assert_allclose(r0, r1, atol=1e-6)


def test_rigid_copies_have_zero_rmsf():
    X = _ensemble(F=1)[0]
    Y = np.stack([X @ _rot(f).T + f for f in range(5)])
    r, _, ok = converged_rmsf(Y)
    assert ok and r.max() < 1e-6


def test_fixed_point():
    X = _ensemble(seed=3)
    r_a, _, _ = converged_rmsf(X, tol=1e-9)
    r_b, _, _ = converged_rmsf(X, tol=1e-12, max_iter=5000)
    np.testing.assert_allclose(r_a, r_b, atol=1e-6)
