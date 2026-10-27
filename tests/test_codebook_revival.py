"""Dead-code revival rule "ema-revival-v2" of the VectorQuantizer (final_fix_list A6).

What it does
------------
  * the code before the fix (verbatim backup in
    `camera_ready/ensembits_code_backup_pre_revival_fix_20261019/`) creates exact
    duplicate codes when a small last batch reseeds many dead codes (the bug);
  * the fixed quantizer does no codebook bookkeeping on a partial batch;
  * dead codes (no assignment over the window) are reseeded from distinct batch rows
    plus noise: no exact or near duplicates, even when the batch rows repeat; at most
    one code per batch row; live codes untouched;
  * a code with one assignment in the window is not dead; the check is deferred while
    revival is blocked (trainer: last steps of an epoch) and runs at the next allowed
    step; the SFTD sub-ensemble branch (update_codebook False) does not count usage;
  * the checkpoint check (`codebook_health`) reports exact / near (distance) duplicates,
    the cosine > 0.9999 count and codes unused on validation, and flags an error on
    exact / near duplicates only;
  * all levels of an RVQ and a single-level VQ follow the rule.

Paper
-----
`appendix.tex` \\subsection{RVQ-VAE} \\label{app:RVQVAE} (dead-code revival; the text
still states the old rule).

Origin
------
New (final_fix_list A6, 2026-10-19).

How to run
----------
  python -m pytest tests/test_codebook_revival.py -q     # from the codebase root
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from ensembits.tokenizer.rvqvae import RVQVAETokenizer
from ensembits.tokenizer.vqvae import (
    CODEBOOK_UPDATE_RULE, VectorQuantizer, codebook_duplicate_stats, codebook_health)

BACKUP = (Path(__file__).resolve().parents[2]
          / "ensembits_code_backup_pre_revival_fix_20261019" / "ensembits" / "tokenizer"
          / "vqvae.py")


def _old_vqvae():
    if not BACKUP.exists():
        pytest.skip(f"backup of the pre-fix code not found: {BACKUP}")
    spec = importlib.util.spec_from_file_location("_vqvae_pre_revival_fix", BACKUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vq(cls, M=64, D=8, seed=0):
    """A training-mode EMA quantizer with unit counts (the state after k-means init)."""
    torch.manual_seed(seed)
    vq = cls(num_codes=M, code_dim=D, vq_update="ema").train()
    with torch.no_grad():
        vq.ema_count.fill_(1.0)
    return vq


def _batch_near(vq, live, n, g, sd=0.05):
    """n rows cycling over the live codes `live` plus small noise (each live code wins)."""
    cb = vq.codebook.weight.detach()[live]
    return cb[torch.arange(n) % len(live)] + sd * torch.randn(n, vq.code_dim, generator=g)


def _far_codes(vq, idx, scale=1e3):
    """Move codes `idx` far away so they never win (forced dead codes)."""
    with torch.no_grad():
        g = torch.Generator().manual_seed(123)
        d = torch.nn.functional.normalize(torch.randn(len(idx), vq.code_dim, generator=g), dim=1)
        far = scale * (d + 2.0 * torch.ones(vq.code_dim) / vq.code_dim ** 0.5)  # distinct, far
        vq.codebook.weight.data[idx] = far
        vq.ema_weight.data[idx] = far


def test_rule_name():
    assert CODEBOOK_UPDATE_RULE == "ema-revival-v2"


def test_old_rule_duplicates_on_partial_batch():
    """Pre-fix code: a small batch reseeds many dead codes with replacement."""
    old = _old_vqvae()
    vq = _vq(old.VectorQuantizer)
    with torch.no_grad():
        vq.ema_count.fill_(5.0)
        vq.ema_count[:40] = 0.5                 # 40 dead codes under the old rule
    z = torch.randn(8, 8, generator=torch.Generator().manual_seed(1))   # partial batch
    vq(z)
    st = codebook_duplicate_stats(vq.codebook.weight)
    assert st["exact_duplicates"] >= 32         # 40 codes from 8 rows
    # and the reseeded codes are raw batch rows
    raw = (vq.codebook.weight[:40].unsqueeze(1) == z.unsqueeze(0)).all(-1).any(1)
    assert bool(raw.all())


def test_partial_batch_does_nothing():
    vq = _vq(VectorQuantizer)
    vq.full_batch_size = 32
    _far_codes(vq, list(range(40)))
    vq.revival_window = 1
    before = [t.clone() for t in (vq.codebook.weight, vq.ema_count, vq.ema_weight,
                                  vq.usage_window, vq.window_steps)]
    vq(torch.randn(8, 8))
    after = (vq.codebook.weight, vq.ema_count, vq.ema_weight, vq.usage_window,
             vq.window_steps)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert int(vq.n_partial_skipped) == 1 and int(vq.n_revived_codes) == 0


def test_partial_batch_inferred_from_largest_batch():
    vq = _vq(VectorQuantizer)                    # full_batch_size None
    vq(torch.randn(32, 8))
    ema = vq.ema_count.clone()
    vq(torch.randn(10, 8))
    assert torch.equal(ema, vq.ema_count) and int(vq.n_partial_skipped) == 1


@pytest.mark.parametrize("repeat_rows", [False, True])
def test_revival_without_duplicates(repeat_rows):
    vq = _vq(VectorQuantizer, M=48)
    vq.full_batch_size = 32
    vq.revival_window = 5
    dead, live = list(range(40)), list(range(40, 48))
    _far_codes(vq, dead)
    g = torch.Generator().manual_seed(2)
    live_before = vq.codebook.weight[40:].clone()
    for step in range(5):
        z = _batch_near(vq, live, 32, g)
        if repeat_rows:                          # only 8 distinct rows in the batch
            z = z[:8].repeat(4, 1)
        vq(z)
        if step < 4:
            assert int(vq.n_revived_codes) == 0  # window not complete yet
    # 40 dead codes, 32 rows: 32 reseeded now (one per row), 8 wait
    assert int(vq.n_revived_codes) == 32 and int(vq.window_steps) == 0
    cb = vq.codebook.weight
    st = codebook_duplicate_stats(cb)
    assert st["exact_duplicates"] == 0 and st["near_duplicates"] == 0
    # revived codes are not raw batch rows
    assert not bool((cb.unsqueeze(1) == z.unsqueeze(0)).all(-1).any())
    still_far = (cb[:40].norm(dim=1) > 100).sum()
    assert int(still_far) == 8
    # live codes (40..47) moved only by the EMA update, not reseeded
    assert torch.allclose(cb[40:], live_before, atol=0.5)
    # the next window revives the rest (the codes revived above are judged afresh)
    for _ in range(5):
        vq(_batch_near(vq, live, 32, g))
    assert int((vq.codebook.weight[:40].norm(dim=1) > 100).sum()) == 0
    st = codebook_duplicate_stats(vq.codebook.weight)
    assert st["exact_duplicates"] == 0 and st["near_duplicates"] == 0


def test_used_code_is_not_dead_and_window_resets():
    vq = _vq(VectorQuantizer, M=4, D=2)
    vq.full_batch_size = 8
    vq.revival_window = 3
    with torch.no_grad():
        cb = torch.tensor([[0., 0.], [10., 0.], [0., 10.], [1e3, 1e3]])
        vq.codebook.weight.data.copy_(cb); vq.ema_weight.data.copy_(cb)
        vq.ema_count.fill_(1.0)
    near0 = torch.zeros(8, 2) + 0.01 * torch.randn(8, 2)
    z_hit = near0.clone(); z_hit[0] = torch.tensor([10., 0.]); z_hit[1] = torch.tensor([0., 10.])
    vq(z_hit)                 # codes 1, 2 hit once; code 3 never
    vq(near0); vq(near0)      # window complete -> only code 3 is dead
    assert int(vq.n_revived_codes) == 1
    assert float(vq.codebook.weight[3].norm()) < 100
    assert int(vq.window_steps) == 0 and float(vq.usage_window.sum()) == 0


def test_revival_deferred_while_blocked():
    vq = _vq(VectorQuantizer, M=8)
    vq.full_batch_size = 32
    vq.revival_window = 2
    _far_codes(vq, [0, 1, 2])
    g = torch.Generator().manual_seed(4)
    vq.revival_allowed = False
    for _ in range(4):
        vq(_batch_near(vq, [3, 4, 5, 6, 7], 32, g))
    assert int(vq.n_revived_codes) == 0 and int(vq.window_steps) == 4
    vq.revival_allowed = True
    vq(_batch_near(vq, [3, 4, 5, 6, 7], 32, g))
    assert int(vq.n_revived_codes) == 3 and int(vq.window_steps) == 0


def test_sub_ensemble_branch_does_not_count():
    vq = _vq(VectorQuantizer)
    vq.update_codebook = False
    vq(torch.randn(32, 8))
    assert int(vq.window_steps) == 0 and float(vq.usage_window.sum()) == 0
    assert int(vq.n_partial_skipped) == 0


def test_eval_mode_does_nothing():
    vq = _vq(VectorQuantizer).eval()
    cb = vq.codebook.weight.clone()
    vq(torch.randn(32, 8))
    assert torch.equal(cb, vq.codebook.weight) and int(vq.window_steps) == 0


def test_codebook_health_flags_duplicates():
    m = RVQVAETokenizer(input_dim=16, num_prototypes=2, codebook_sizes=[16, 4],
                        hidden_dim=16, latent_dim=8)
    toks = [torch.tensor([0, 1, 2, 2]), torch.tensor([0, 0, 1, 3])]
    h = codebook_health(m.vq_levels, toks)
    assert h["error"] is None
    assert [x["zero_usage_val"] for x in h["levels"]] == [13, 1]
    with torch.no_grad():
        m.vq_levels[0].codebook.weight[5] = m.vq_levels[0].codebook.weight[3]
        m.vq_levels[1].codebook.weight[2] = 1.000001 * m.vq_levels[1].codebook.weight[0]
    h = codebook_health(m.vq_levels, toks)
    assert h["levels"][0]["exact_duplicates"] == 1 and h["levels"][0]["near_duplicates"] == 1
    assert h["levels"][1]["exact_duplicates"] == 0 and h["levels"][1]["near_duplicates"] == 1
    assert h["levels"][1]["cos_near_duplicates"] == 1
    assert h["error"] and "L0" in h["error"] and "L1" in h["error"]


def test_parallel_but_distinct_codes_are_not_duplicates():
    """Cosine > 0.9999 alone (same direction, different norm) is reported, not an error."""
    cb = torch.randn(16, 8, generator=torch.Generator().manual_seed(5))
    cb[7] = 1.2 * cb[3]
    st = codebook_duplicate_stats(cb)
    assert st["cos_near_duplicates"] == 1 and st["near_duplicates"] == 0
    assert st["exact_duplicates"] == 0


@pytest.mark.parametrize("sizes", [[64, 16, 16], [64]])
def test_rvq_and_single_level_follow_the_rule(sizes):
    """Every level (RVQ) and the single-level VQ revive from full batches only."""
    torch.manual_seed(0)
    m = RVQVAETokenizer(input_dim=16, num_prototypes=2, codebook_sizes=sizes,
                        hidden_dim=16, latent_dim=8).train()
    for vq in m.vq_levels:
        vq.full_batch_size = 32
        vq.revival_window = 2
        vq.ema_count.fill_(1.0)
        _far_codes(vq, list(range(vq.num_codes // 2)))
    m._quantize(torch.randn(8, 8))               # partial: nothing
    assert all(int(vq.window_steps) == 0 for vq in m.vq_levels)
    for _ in range(2):
        m._quantize(torch.randn(32, 8))
    for vq in m.vq_levels:
        # the far half is dead (plus any live code without a hit), capped at 32 rows
        assert min(vq.num_codes // 2, 32) <= int(vq.n_revived_codes) <= 32
    h = codebook_health(m.vq_levels)
    assert h["error"] is None, h


def test_revival_state_roundtrip():
    a = _vq(VectorQuantizer); a.full_batch_size = 32
    a(torch.randn(32, 8)); a(torch.randn(32, 8))
    b = _vq(VectorQuantizer)
    b.load_revival_state(a.revival_state())
    assert torch.equal(a.usage_window, b.usage_window)
    assert int(b.window_steps) == 2 and b._max_batch_seen == 32
    # not in the state dict: best.pt keeps its old keys
    assert "usage_window" not in a.state_dict()


def test_identical_rows_get_distinct_codes():
    """Training audits 2026-10-20: the absolute noise floor (1e-3 x median code norm)
    keeps revived codes apart when every batch row is the same (batch std 0)."""
    vq = _vq(VectorQuantizer, M=48)
    vq.full_batch_size = 32
    vq.revival_window = 1
    _far_codes(vq, list(range(40)))
    z = vq.codebook.weight.detach()[40].clone().expand(32, -1).clone()
    vq(z)
    assert int(vq.n_revived_codes) >= 32
    st = codebook_duplicate_stats(vq.codebook.weight)
    assert st["exact_duplicates"] == 0 and st["near_duplicates"] == 0
