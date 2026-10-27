"""Residual quantizer: gradient path of the commitment terms (code_fix C1).

What it does
------------
  * "standard" and "legacy" residual modes give identical tokens and residual
    values (so old checkpoints tokenize the same);
  * in "legacy" mode only the level-1 commitment term reaches the encoder output;
    in "standard" mode the level-2 and level-3 terms do too.

Paper
-----
`appendix.tex` app:RVQVAE, Eq. eq:rvq-recurrence and L_commit.

How to run
----------
  python -m pytest tests/test_rvq.py
"""
import copy

import torch

from ensembits.tokenizer.rvqvae import RVQVAETokenizer


def _model(mode):
    torch.manual_seed(0)
    return RVQVAETokenizer(input_dim=256, num_prototypes=10, codebook_sizes=[64, 16, 16],
                           hidden_dim=64, latent_dim=32, vq_update="ema",
                           residual_mode=mode).eval()


def _grad_of_vq_loss(model, z):
    z = z.clone().requires_grad_(True)
    _, tokens, loss = model._quantize(z)
    (g,) = torch.autograd.grad(loss, z)
    return g, tokens


def test_same_tokens_both_modes():
    std = _model("standard")
    leg = copy.deepcopy(std); leg.residual_mode = "legacy"
    x = torch.randn(32, 10, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a = std.encode_all(x); b = leg.encode_all(x)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_commitment_reaches_encoder_at_all_levels():
    std = _model("standard")
    leg = copy.deepcopy(std); leg.residual_mode = "legacy"
    z = torch.randn(64, 32, generator=torch.Generator().manual_seed(2))
    g_std, _ = _grad_of_vq_loss(std, z)
    g_leg, _ = _grad_of_vq_loss(leg, z)
    # legacy: gradient = level-1 term only = beta * 2/(N*D) * (z - q1)
    z1 = z.clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(leg.vq_levels[0](z1)[2], z1)
    torch.testing.assert_close(g_leg, g1)
    assert not torch.allclose(g_std, g_leg)          # levels 2-3 contribute in standard mode


def test_codebook_update_switch():
    """update_codebook=False: a training-mode forward leaves codebook and EMA state
    unchanged (the SFTD sub-ensemble branch, code_fix C9)."""
    m = _model("standard").train()
    vq = m.vq_levels[0]
    z = torch.randn(256, 32, generator=torch.Generator().manual_seed(3))
    before = (vq.codebook.weight.clone(), vq.ema_count.clone(), vq.ema_weight.clone())
    vq.update_codebook = False
    vq(z)
    assert all(torch.equal(a, b) for a, b in
               zip(before, (vq.codebook.weight, vq.ema_count, vq.ema_weight)))
    vq.update_codebook = True
    vq(z)
    assert not torch.equal(before[1], vq.ema_count)
