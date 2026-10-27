"""Tests of the reconstruction loss and the Hungarian matching (`ensembits/losses.py`).

What it does
------------
  * the loss is zero for identical inputs and invariant to frame permutations;
  * the pairwise cost matrix used for matching equals the loss of each pair;
  * matching on the loss (code_fix §2) never gives a larger loss than matching on
    squared L2 (the matching used before the fix);
  * rectangular matching: P_max = 10 outputs, fewer targets.

Paper
-----
`1_method.tex`, Decoder and reconstruction loss, Eq. (eq:fape) and
Eq. (eq:hungarian).

How to run
----------
  python -m pytest tests/test_losses.py
"""
import pytest
import torch

from ensembits.descriptor import _descriptor
from ensembits.losses import (fape_dihedral_loss, fape_hungarian_loss,
                              matched_pairs, pairwise_recon_cost)


@pytest.fixture(scope="module")
def target(ensemble):
    """(B = 68 residues, P = 10 frames, 256) descriptors."""
    return _descriptor(ensemble, "cpu").transpose(0, 1).contiguous()


def _noisy(x, scale=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return x + scale * torch.randn(x.shape, generator=g)


def test_zero_for_identical(target):
    assert fape_hungarian_loss(target, target).item() == pytest.approx(0.0, abs=1e-5)


def test_permutation_invariance(target):
    pred = _noisy(target)
    perm = torch.randperm(target.shape[1], generator=torch.Generator().manual_seed(3))
    a = fape_hungarian_loss(pred, target)
    b = fape_hungarian_loss(pred[:, perm], target)
    c = fape_hungarian_loss(pred, target[:, perm])
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a, c, rtol=1e-5, atol=1e-5)


def test_pairwise_cost_is_the_loss(target):
    pred = _noisy(target[:8, :4])
    tgt = target[:8, :3]
    cost = pairwise_recon_cost(pred, tgt)                         # (8, 4, 3)
    for i in range(4):
        for j in range(3):
            direct = torch.stack([fape_dihedral_loss(pred[b, i], tgt[b, j]) for b in range(8)])
            torch.testing.assert_close(cost[:, i, j], direct, rtol=1e-4, atol=1e-5)


def test_loss_matching_not_worse_than_l2(target):
    pred = _noisy(target, scale=0.5, seed=1)
    loss_match = fape_hungarian_loss(pred, target, match_cost="loss")
    l2_match = fape_hungarian_loss(pred, target, match_cost="l2")
    assert loss_match.item() <= l2_match.item() + 1e-6


def test_rectangular(target):
    pred = _noisy(target)                     # 10 decoder outputs
    tgt = target[:, :4]                       # P' = 4 targets
    mp, mt = matched_pairs(pred, tgt, cost="loss")
    assert mp.shape == mt.shape == (target.shape[0], 4, 256)
    # every target frame is used exactly once
    torch.testing.assert_close(mt.sort(dim=1).values, tgt.sort(dim=1).values)
