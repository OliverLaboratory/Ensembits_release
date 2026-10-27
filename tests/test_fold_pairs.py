"""Unit tests of the fold-matched pair test (downstream/fold_pairs/) on synthetic data.

What it does
------------
  * the chance-adjusted agreement (obs - p) / (1 - p) and the collision probability
    p = sum_t f(t)^2;
  * the paired bootstrap of Delta = adj(similar) - adj(divergent): for random tokens
    the CI of Delta contains 0, and for two identical tokenizers the difference of
    their Deltas is exactly 0 (CI contains 0);
  * the miner never pairs two systems of the same domain (two deposits of one pid),
    neither at candidate generation (from a synthetic C-alpha cache) nor at selection;
  * the verifier's verdict for both modes, including the sequence-identity cap.
No TM-align, trajectory or tokenizer is needed.

Paper
-----
appendix, fold-matched pair test (section to be written); rebuttal to Reviewer szcw.

How to run
----------
  python -m pytest tests/test_fold_pairs.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest

FOLD_PAIRS = Path(__file__).resolve().parents[1] / "downstream" / "fold_pairs"
sys.path.insert(0, str(FOLD_PAIRS))

# The fold-pair modules pin BLAS thread counts in os.environ at import; restore them so
# the other tests are not affected.
_saved_env = dict(os.environ)
import bioemu_fold_dynamics_miner as BM      # noqa: E402
import bioemu_token_compare as CMP           # noqa: E402
import fold_dynamics_miner as M              # noqa: E402
import verify_fold_dynamics_pair as VER      # noqa: E402
os.environ.clear()
os.environ.update(_saved_env)


# ---------------------------------------------------------------- (i) adjusted agreement
def test_adjusted_agreement_formula():
    assert CMP.adjusted_agreement(0.5, 0.5) == pytest.approx(0.0)      # chance level
    assert CMP.adjusted_agreement(1.0, 0.1) == pytest.approx(1.0)      # perfect agreement
    assert CMP.adjusted_agreement(0.6, 0.2) == pytest.approx(0.5)      # (0.6-0.2)/(1-0.2)
    assert CMP.adjusted_agreement(0.0, 0.2) == pytest.approx(-0.25)    # below chance
    # collision probability: two equally frequent tokens -> 0.5; uniform over K -> 1/K
    assert CMP.chance_collision([np.array([0, 0, 1, 1])]) == pytest.approx(0.5)
    K = 16
    assert CMP.chance_collision([np.arange(K), np.arange(K)]) == pytest.approx(1.0 / K)
    # pooled over several arrays, as for the floor population
    assert CMP.chance_collision([np.array([0, 0]), np.array([1, 2])]) == pytest.approx(0.25 + 2 * 0.0625)


# ---------------------------------------------------------------- (ii) bootstrap of Delta
def _random_pair_adj(rng, n_pairs, K, L=80):
    """Per-pair adjusted agreement of a random K-letter tokenizer (floor from its own tokens)."""
    toks = [(rng.integers(0, K, L), rng.integers(0, K, L)) for _ in range(n_pairs)]
    floor = [rng.integers(0, K, L) for _ in range(200)]
    p = CMP.chance_collision(floor)
    return np.array([CMP.adjusted_agreement(float((a == b).mean()), p) for a, b in toks])


def test_bootstrap_delta_random_tokenizers_contains_zero():
    rng = np.random.default_rng(123)
    r1_sim, r1_div = _random_pair_adj(rng, 40, 20), _random_pair_adj(rng, 30, 20)
    r2_sim, r2_div = _random_pair_adj(rng, 40, 20), _random_pair_adj(rng, 30, 20)
    adj_sim = {'R1': r1_sim, 'R1 copy': r1_sim.copy(), 'R2': r2_sim}
    adj_div = {'R1': r1_div, 'R1 copy': r1_div.copy(), 'R2': r2_div}
    names = ['R1', 'R1 copy', 'R2']
    res = CMP.bootstrap_contrast(adj_sim, adj_div, names, n_boot=2000, seed=0, ens_names=('R1',))
    for n in names:
        lo, hi = res['per_tokenizer'][n]['ci_delta']
        assert lo <= 0.0 <= hi, (n, lo, hi)
    # identical tokenizers: the paired difference of Deltas is exactly 0 in every draw
    same = res['ensembits_minus_baseline']['R1 - R1 copy']
    assert same['diff'] == 0.0 and same['ci'] == [0.0, 0.0]
    # two independent random tokenizers: the paired difference CI contains 0
    lo, hi = res['ensembits_minus_baseline']['R1 - R2']['ci']
    assert lo <= 0.0 <= hi
    # the bootstrap is reproducible with a fixed seed
    res2 = CMP.bootstrap_contrast(adj_sim, adj_div, names, n_boot=2000, seed=0, ens_names=('R1',))
    assert res2['per_tokenizer']['R2']['ci_delta'] == res['per_tokenizer']['R2']['ci_delta']


def test_bootstrap_delta_detects_dynamics_signal():
    """A tokenizer that agrees more on similar pairs gets a positive Delta with CI > 0."""
    rng = np.random.default_rng(1)
    sim = 0.4 + 0.05 * rng.standard_normal(30)
    div = 0.1 + 0.05 * rng.standard_normal(25)
    res = CMP.bootstrap_contrast({'E': sim}, {'E': div}, ['E'], n_boot=2000, seed=0, ens_names=())
    lo, hi = res['per_tokenizer']['E']['ci_delta']
    assert lo > 0.2 and hi < 0.4


# ---------------------------------------------------------------- (iii) no same-domain pairs
@pytest.fixture
def synthetic_cache(tmp_path):
    """C-alpha cache with two deposits of the same domains plus unrelated domains."""
    rng = np.random.default_rng(0)
    systems = {'cath1_1abcA00': 60, 'cath2_1abcA00': 60,     # same domain, both deposits
               'cath1_2xyzB01': 62, 'cath2_2xyzB01': 62,     # same domain, both deposits
               'cath2_3defA02': 61, 'cath2_4ghiA00': 58, 'cath2_5jklA01': 90}
    for s, L in systems.items():
        np.savez(tmp_path / f'{s}.npz', ca=rng.normal(size=(6, L, 3)).astype(np.float32),
                 runlens=np.array([3, 3]), system=np.array(s), pid=np.array(s.split('_', 1)[1]))
    yield tmp_path
    M.DOMS.clear()


def test_miner_never_pairs_same_domain(synthetic_cache):
    M.DOMS.clear()
    BM.load_domains(str(synthetic_cache))
    assert set(M.DOMS) == {p.stem for p in synthetic_cache.glob('*.npz')}   # keyed by system
    assert M.DOMS['cath2_1abcA00']['pid'] == '1abcA00'
    pairs = BM.candidate_pairs(0.15)
    assert pairs, 'expected some candidate pairs'
    for _, a, b in pairs:
        assert M.DOMS[a]['pid'] != M.DOMS[b]['pid'], (a, b)
    # the length filter is still applied (L=90 is too far from the others at 0.15)
    assert not any('cath2_5jklA01' in (a, b) for _, a, b in pairs)
    # cross-domain pairs of the duplicated domains are still emitted
    assert ('bioemu', 'cath1_1abcA00', 'cath2_2xyzB01') in pairs


def test_selection_drops_same_domain_and_ranks_both_ways():
    base = dict(tm_max=0.8, ncommon=60, fvarA=0.3, fvarB=0.4)
    recs = [dict(base, A='cath1_1abcA00', B='cath2_1abcA00', pidA='1abcA00', pidB='1abcA00', cos1=0.99),
            dict(base, A='cath1_1abcA00', B='cath2_3defA02', pidA='1abcA00', pidB='3defA02', cos1=0.90),
            dict(base, A='cath2_3defA02', B='cath2_4ghiA00', pidA='3defA02', pidB='4ghiA00', cos1=0.05),
            dict(base, A='cath2_4ghiA00', B='cath2_2xyzB01', pidA='4ghiA00', pidB='2xyzB01', cos1=0.50),
            dict(base, A='cath2_4ghiA00', B='cath2_5jklA01', pidA='4ghiA00', pidB='5jklA01', cos1=0.01,
                 fvarB=0.1)]                                            # fails the fvar filter
    for select in ('divergent', 'similar'):
        top, n_cand = BM.select_pairs(recs, select, top_n=10)
        assert n_cand == 3
        assert all(r['pidA'] != r['pidB'] for r in top)
    top_div, _ = BM.select_pairs(recs, 'divergent', top_n=2)
    top_sim, _ = BM.select_pairs(recs, 'similar', top_n=2)
    assert [r['cos1'] for r in top_div] == [0.05, 0.50]
    assert [r['cos1'] for r in top_sim] == [0.90, 0.50]


def test_main_selection_reports_ties_beyond_cut(tmp_path, monkeypatch):
    # FIXED(2026-10-23): main() crashed in the tie report when the candidates outnumber
    # --top-n (wrong tuple unpacking of select_pairs). The selection itself is unchanged.
    import json
    base = dict(tm_max=0.8, ncommon=60, fvarA=0.3, fvarB=0.4)
    recs = [dict(base, A=f'cath2_{i}abcA00', B=f'cath2_{i}xyzB01', pidA=f'{i}abcA00',
                 pidB=f'{i}xyzB01', cos1=c) for i, c in enumerate([0.01, 0.02, 0.02, 0.02, 0.40])]
    scored = tmp_path / 'scored.json'
    scored.write_text(json.dumps({'meta': {}, 'records': recs}))
    out = tmp_path / 'cand.json'
    monkeypatch.setattr(sys, 'argv', ['miner', '--scored', str(scored), '--select', 'divergent',
                                      '--top-n', '2', '--out', str(out)])
    BM.main()
    res = json.loads(out.read_text())
    assert [r['cos1'] for r in res['pairs']] == [0.01, 0.02]
    assert res['n_candidates'] == 5 and res['n_tied_beyond_cut'] == 2


# ---------------------------------------------------------------- verdict of the verifier
def test_verdict_both_modes_and_seq_cap():
    cfg = argparse.Namespace(mode='divergent', min_repro=0.6, max_cos1=0.30, min_cos1=0.70,
                             min_cov_overlap=0.5, max_seq_id=0.30)
    assert VER.verdict(0.9, 0.9, 0.02, 0.1, 0.20, cfg)[0] is True
    assert VER.verdict(0.5, 0.9, 0.02, 0.1, 0.20, cfg)[0] is False     # A under-converged
    assert VER.verdict(0.9, 0.9, 0.40, 0.1, 0.20, cfg)[0] is False     # not divergent
    assert VER.verdict(0.9, 0.9, 0.02, 0.1, 0.35, cfg)[0] is False     # near-homolog
    cfg.mode = 'similar'
    assert VER.verdict(0.9, 0.9, 0.85, 0.6, 0.20, cfg)[0] is True
    assert VER.verdict(0.9, 0.9, 0.85, 0.4, 0.20, cfg)[0] is False     # low covariance overlap
    assert VER.verdict(0.9, 0.9, 0.60, 0.6, 0.20, cfg)[0] is False     # cos1 too low
    assert VER.verdict(0.9, 0.9, 0.85, 0.6, 0.31, cfg)[0] is False     # near-homolog
