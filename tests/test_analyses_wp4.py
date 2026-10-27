"""Tests of the ported ANOVA (downstream/analyses/token_anova.py) and its helpers.

What it does
------------
  * The ANOVA core (eta^2, F, p) on synthetic data equals a direct numpy / scipy
    computation (scipy.stats.f_oneway for F and p, SS_between / SS_total for eta^2);
    the vectorised permutation helper gives the same eta^2; the no-intercept OLS R^2
    on one-hot groups equals eta^2.
  * Sidecar check of descriptor caches: a convention / row-count mismatch is refused.
  * Token cache keyed by the descriptor cache's sidecar (and, since 2026-10-18, by P).
  (The tests of the residue-set modes, the count filter and the rare-token pooling were
  removed with those features on 2026-10-18, anova_audit A1; the new statistics are
  tested in tests/test_token_anova_stats.py.)

Paper
-----
`appendix.tex` \\label{app:token_anova} (Eq. eq:eta2, the F-statistic) and
Table tab:token_anova_baselines (same residues for every tokenizer, code_fix P20 / P6).

How to run
----------
  python -m pytest tests/test_analyses_wp4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream" / "analyses"))

import token_anova as TA  # noqa: E402
import _common as C  # noqa: E402


def _synthetic(seed=0, n_groups=7, n=600):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, size=n)
    means = rng.normal(0, 2, size=n_groups)
    values = means[groups] + rng.gamma(2.0, 1.0, size=n)
    return values.astype(np.float64), groups


def test_eta2_and_F_match_scipy():
    values, groups = _synthetic()
    eta2, F, dfb, dfw, p, M = TA.eta_squared(values, groups)
    uniq = np.unique(groups)
    samples = [values[groups == g] for g in uniq]
    F_ref, p_ref = stats.f_oneway(*samples)
    grand = values.mean()
    ss_b = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_t = ((values - grand) ** 2).sum()
    assert M == len(uniq) and dfb == M - 1 and dfw == len(values) - M
    assert eta2 == pytest.approx(ss_b / ss_t, rel=1e-9)
    assert F == pytest.approx(F_ref, rel=1e-9)
    assert p == pytest.approx(p_ref, rel=1e-6, abs=1e-300)
    # eta^2 = R^2 of the one-hot regression (paper: "Equivalently, eta^2 is the R^2 ...")
    onehot = np.eye(len(uniq))[np.searchsorted(uniq, groups)]
    assert TA.ols_r2_no_intercept(onehot, values) == pytest.approx(eta2, rel=1e-8)
    assert TA.eta_squared_fast(values, groups) == pytest.approx(eta2, rel=1e-9)


def test_eta2_extremes():
    g = np.repeat(np.arange(4), 10)
    const_within = np.repeat([1.0, 2.0, 5.0, 9.0], 10)
    assert TA.eta_squared(const_within, g)[0] == pytest.approx(1.0, abs=1e-9)
    same_means = np.tile(np.array([0.0, 1.0] * 5), 4)
    assert TA.eta_squared(same_means, g)[0] == pytest.approx(0.0, abs=1e-12)


def test_sidecar_checks(tmp_path):
    f = tmp_path / "esm3desc_dihedral_iupac_k16_P10.npy"
    np.save(f, np.zeros((3, 2, 4), np.float32))
    assert C.verify_sidecar(f, convention="iupac", chain_break_cutoff=2.5, n_rows=3) is None
    with pytest.raises(SystemExit):
        C.verify_sidecar(f, convention="iupac", require=True)
    (tmp_path / (f.name + ".json")).write_text(json.dumps(
        {"dihedral_convention": "iupac", "chain_break_cutoff": "auto", "n_rows": 3}))
    assert C.verify_sidecar(f, convention="iupac", chain_break_cutoff=2.5, n_rows=3)
    with pytest.raises(SystemExit):
        C.verify_sidecar(f, convention="legacy_plus180", n_rows=3)
    with pytest.raises(SystemExit):
        C.verify_sidecar(f, convention="iupac", chain_break_cutoff=2.5, n_rows=4)
    with pytest.raises(SystemExit):
        C.verify_sidecar(f, convention="iupac", chain_break_cutoff=None, n_rows=3)



def test_token_cache_keyed_by_descriptor_sidecar(tmp_path, monkeypatch):
    # N2 / N3 (2026-10-17): a descriptor cache without a sidecar is refused unless
    # --allow-legacy-cache (recorded); the token cache is keyed by and records the
    # descriptor sidecar sha256 + IDEAL_PIDS, and a token file of another descriptor
    # cache is refused.
    import argparse
    import shutil
    from ensembits.data_versions import provenance_path
    from ensembits.featurize import file_sha256, load_model
    v = CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10"   # reported λ = 10 checkpoint
    conv, _ = C.checkpoint_settings(load_model(str(v), device="cpu"))
    L_by_pid = {"d0": (0, 6), "d1": (6, 5), "d2": (11, 7)}
    total = 18
    cd = tmp_path / "phase_b"; cd.mkdir()
    monkeypatch.setattr(TA, "PHASE_B", cd)
    desc = cd / TA.with_convention("esm3desc_dihedral_k16_P10.npy", conv)
    np.save(desc, np.random.default_rng(0).normal(size=(total, 10, 256)).astype(np.float32))
    tc = tmp_path / "token_cache"

    def run(**kw):
        a = argparse.Namespace(device="cpu", descriptor_source="cache", token_cache_dir=str(tc),
                               no_token_cache=False, allow_ideal=False, allow_legacy_cache=False)
        for key, val in kw.items():
            setattr(a, key, val)
        return TA.load_variant_tokens(v, L_by_pid, list(L_by_pid), a)

    with pytest.raises(SystemExit, match="allow-legacy-cache"):        # N3
        run()
    tok, prov = run(allow_legacy_cache=True)
    assert prov["allow_legacy_cache"] is True and prov["descriptor_sidecar_sha256"] is None
    assert "desclegacy" in Path(prov["token_cache"]).name and (tok >= 0).all()

    sc = Path(provenance_path(desc))
    sc.write_text(json.dumps({"dataset": "mdcath", "n_rows": total, "IDEAL_PIDS": ["d1"]}))
    # anova_audit fix round 2 (2026-10-18): ideal-geometry domains are an error unless
    # --allow-ideal (was: excluded with a warning)
    with pytest.raises(SystemExit, match="ideal geometry"):
        run()
    tok1, prov1 = run(allow_ideal=True)
    assert prov1["descriptor_sidecar_sha256"] == file_sha256(sc)
    assert prov1["descriptor_ideal_pids"] == ["d1"] and prov1["ideal_pids_used"] == ["d1"]
    assert (tok1 >= 0).all()
    assert f"desc{file_sha256(sc)[:12]}" in Path(prov1["token_cache"]).name
    assert Path(prov1["token_cache"]).name.endswith("__ideal.npz")
    z = np.load(prov1["token_cache"])
    assert str(z["descriptor_sidecar_sha256"]) == file_sha256(sc)
    assert json.loads(str(z["ideal_pids"])) == ["d1"]
    tok2, prov2 = run(allow_ideal=True)                                  # reused
    assert prov2["token_cache"] == prov1["token_cache"] and (tok2 == tok1).all()

    # another descriptor cache (new sidecar): a token file under its name made from the
    # old cache is refused (N2)
    sc.write_text(json.dumps({"dataset": "mdcath", "n_rows": total, "IDEAL_PIDS": []}))
    other = tc / Path(prov1["token_cache"]).name.replace(
        f"desc{prov1['descriptor_sidecar_sha256'][:12]}", f"desc{file_sha256(sc)[:12]}")
    assert other.name != Path(prov1["token_cache"]).name
    shutil.copy(prov1["token_cache"], other)
    with pytest.raises(SystemExit, match="does not match"):
        run(allow_ideal=True)
    other.unlink()
    tok3, prov3 = run(allow_ideal=True)                                  # rebuilt
    assert prov3["token_cache"] == str(other) and (tok3 >= 0).all()
