"""Tests of the tokenizer checkpoint in `ckpt/` (`ensembits/featurize.py`).

What it does
------------
  * tokens of the fixture ensemble at P = 10 equal the stored reference;
  * the set encoder is invariant to the order of the frames;
  * single-frame tokenization (P = 1) runs and returns valid token ids.

FIXED(recon, 2026-10-24): reference tokens regenerated for the λ = 10 checkpoint
  (ensembits_bm_lam10_rv, iupac convention).

Paper
-----
`1_method.tex` (set encoder, residual quantizer, SFTD);
`appendix.tex` \\label{app:model-impl}.

How to run
----------
  python -m pytest tests/test_model.py
"""
import numpy as np
import pytest

from ensembits.featurize import load_model, tokenize_ensemble, tokenize_ensemble_all
from ensembits.paths import CKPT_DIR


@pytest.fixture(scope="module")
def ens():
    if not (CKPT_DIR / "best.pt").exists():
        pytest.skip("checkpoint not present in ckpt/")
    return load_model(CKPT_DIR, device="cpu")


def test_reference_tokens(ens, ensemble, reference_tokens):
    toks = tokenize_ensemble_all(ens, None, bb_all=ensemble.numpy())
    np.testing.assert_array_equal(toks, reference_tokens)


def test_frame_order_invariance(ens, ensemble):
    bb = ensemble.numpy()
    perm = np.random.default_rng(0).permutation(bb.shape[0])
    a = tokenize_ensemble_all(ens, None, bb_all=bb)
    b = tokenize_ensemble_all(ens, None, bb_all=bb[perm])
    # floating point summation order can flip a near tie; require >= 97% equal
    assert (a == b).all(axis=1).mean() >= 0.97


def test_single_frame(ens, ensemble):
    toks = tokenize_ensemble(ens, None, bb_all=ensemble.numpy()[:1])
    assert toks.shape == (ensemble.shape[1],)
    assert toks.min() >= 0 and toks.max() < ens.config["codebook_sizes"][0]
