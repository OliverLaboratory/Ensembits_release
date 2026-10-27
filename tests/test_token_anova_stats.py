"""Tests of the token-ANOVA statistics (anova_stats.py) and of token_anova.py end to end.

What it does
------------
  * omega^2 on a toy case computed by hand; eta^2 counts singleton groups.
  * The circular-shift null keeps each protein's token multiset (and only rotates within
    proteins); p = (b + 1) / (n + 1).
  * Held-out R^2: unseen tokens predict 0; cross-fitted token means use training folds only;
    CATH-grouped folds keep a label in one fold.
  * The protein bootstrap from per-protein sums equals the R^2 of the resampled residues.
  * Vote_3Di uses the paper rule (plurality >= 2, else nearest centroid, lowest id on ties).
  * Size-balanced CATH folds, topology-cluster bootstrap, "p < 10^-3 (0/1,000)" summary.
  * End to end on a synthetic mdCATH-div (6 domains built from the 1ctfA00 fixture, the
    legacy checkpoint in ckpt/, synthetic baseline caches with provenance sidecars): no
    pooling by default (M = every token used), raw token ids in `<stem>.residues.npz` and
    the export (a token used once keeps its id), `--min-count` only adds a diagnostic
    block, a missing baseline cache exits non-zero; chance row per Ensembits row; Delta per
    frame regime; folds stored; aa read for the confounds even when not a row; unknown
    --pids and a protein missing inside a cache exit non-zero.

Paper
-----
`appendix.tex` \\label{app:token_anova}; decisions A1-A5 of `camera_ready/anova_audit.md`.

How to run
----------
  python -m pytest tests/test_token_anova_stats.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream" / "analyses"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"

import anova_stats as S  # noqa: E402


def test_omega2_toy_case():
    # groups {1,2,3}, {4,5,6}, {7,8,9}: grand mean 5; SS_b = 3*(9+0+9) = 54;
    # SS_w = 3 * 2 = 6; SS_t = 60; MS_w = 6/6 = 1; omega^2 = (54 - 2*1) / (60 + 1)
    y = np.arange(1, 10, dtype=float)
    g = np.repeat([10, 20, 30], 3)
    a = S.oneway_anova(y, g)
    assert a["M"] == 3 and a["df_between"] == 2 and a["df_within"] == 6
    assert a["eta2"] == pytest.approx(54 / 60)
    assert a["omega2"] == pytest.approx(52 / 61)
    assert a["F"] == pytest.approx((54 / 2) / 1.0)
    assert "p_parametric_invalid_under_dependence" in a and "p" not in a


def test_eta2_counts_singletons():
    y = np.array([0.0, 0.0, 1.0, 1.0, 10.0])
    g = np.array([0, 0, 1, 1, 2])                      # group 2 has one residue
    a = S.oneway_anova(y, g)
    gm = y.mean()
    ss_b = 2 * (0 - gm) ** 2 + 2 * (1 - gm) ** 2 + (10 - gm) ** 2
    assert a["M"] == 3 and a["n_singleton_groups"] == 1
    assert a["eta2"] == pytest.approx(ss_b / ((y - gm) ** 2).sum())
    assert a["eta2"] == pytest.approx(1.0)             # all variance is between groups


def test_circular_shift_keeps_protein_multiset():
    rng = np.random.default_rng(0)
    lens = [30, 12, 50, 8]                              # the 8-residue protein stays put
    prot = np.repeat(np.arange(len(lens)), lens)
    g = rng.integers(0, 15, size=len(prot))
    blocks = S.protein_blocks(prot)
    moved = 0
    for _ in range(20):
        h = S.circular_shift(g, blocks, rng, min_shift=5)
        for p, (s, L) in enumerate(zip(*blocks)):
            a, b = g[s:s + L], h[s:s + L]
            assert sorted(a) == sorted(b)                                     # multiset kept
            assert any(np.array_equal(np.roll(a, k), b) for k in range(L))    # a rotation
            if L < 11:
                assert np.array_equal(a, b)
        moved += int((h != g).any())
    assert moved == 20
    with pytest.raises(ValueError):
        S.protein_blocks(np.array([0, 1, 0]))


def test_permutation_null_and_p():
    rng = np.random.default_rng(1)
    prot = np.repeat(np.arange(20), 40)
    g = rng.integers(0, 6, size=len(prot))
    y = g * 1.0 + rng.normal(0, 0.5, len(prot))       # tokens explain y
    blocks = S.protein_blocks(prot)
    obs = S.oneway_anova(y, g)["eta2"]
    for kind in ("circular", "shuffle"):
        null = S.permutation_null(y, g=g, blocks=blocks, n=50, seed=0, kind=kind)
        assert len(null) == 50 and null.max() < obs
        assert S.perm_p(null, obs) == pytest.approx(1 / 51)
    assert S.perm_p(np.array([0.1, 0.5, 0.9]), 0.5) == pytest.approx(3 / 4)
    # continuous: OLS R^2 under the null equals a direct lstsq on the shifted rows
    X = np.eye(6)[g]
    nullc = S.permutation_null(y, X=X, blocks=blocks, n=3, seed=2, kind="circular")
    r = np.random.default_rng(2)
    for v in nullc:
        src = S.circular_shift(np.arange(len(y)), blocks, r, 5)
        assert v == pytest.approx(S.ols_r2_no_intercept(X, y[src]), abs=1e-9)


def test_heldout_unseen_token_predicts_zero():
    yc = np.array([1.0, 1.0, -1.0, -1.0, 5.0, 3.0])
    g = np.array([7, 7, 8, 8, 9, 7])
    folds = np.array([0, 0, 0, 0, 1, 1])
    yhat = S.heldout_token_means(yc, g, folds)
    # fold 1 is predicted from fold 0: token 9 unseen -> 0, token 7 -> mean(1, 1) = 1
    assert yhat[4] == 0.0 and yhat[5] == pytest.approx(1.0)
    # fold 0 predicted from fold 1: token 7 -> 3, token 8 unseen -> 0
    assert yhat[:4].tolist() == [3.0, 3.0, 0.0, 0.0]
    prot = np.array([0, 0, 0, 0, 1, 1])
    hc = S.center_by_protein(yhat, prot)
    assert S.within_r2(yc, yhat, prot) == pytest.approx(1 - ((yc - hc) ** 2).sum() / (yc ** 2).sum())


def test_group_folds_and_bootstrap():
    labels = np.array(["1.10.10", "1.10.10", "2.40.50", "3.30.70", "3.30.70", "4.10.220"])
    f = S.group_folds(labels, k=3, seed=0)
    for lab in np.unique(labels):
        assert len(set(f[labels == lab])) == 1
    rng = np.random.default_rng(3)
    prot = np.repeat(np.arange(5), 10)
    yc = S.center_by_protein(rng.normal(size=50), prot)
    yhat = yc + rng.normal(0, 0.5, 50)
    W = S.bootstrap_weights(5, 4, seed=0)
    boot = S.bootstrap_r2(W, *S.per_protein_ss(yc, yhat, prot))
    for w, b in zip(W, boot):
        rows = np.repeat(np.arange(50), w[prot].astype(int))      # resampled residues
        hc = S.center_by_protein(yhat, prot)
        assert b == pytest.approx(1 - ((yc[rows] - hc[rows]) ** 2).sum() / (yc[rows] ** 2).sum())


def test_gain_over_confounds_recovers_token_signal():
    rng = np.random.default_rng(4)
    prot = np.repeat(np.arange(40), 50)
    conf = rng.integers(0, 4, len(prot)); tok = rng.integers(0, 8, len(prot))
    y = 1.0 * conf + 0.7 * tok + rng.normal(0, 0.3, len(prot))
    yc = S.center_by_protein(y, prot)
    folds = S.group_folds(np.arange(40), k=5, seed=0)[prot]
    y0, yh = S.heldout_gain_over_confounds(yc, [conf], folds,
                                           {"tok": ("tokens", tok),
                                            "noise": ("tokens", rng.integers(0, 8, len(prot)))})
    r0 = S.within_r2(yc, y0, prot)
    assert S.within_r2(yc, yh["tok"], prot) - r0 > 0.2
    assert abs(S.within_r2(yc, yh["noise"], prot) - r0) < 0.02
    # residual mode: same confound model, the token term fitted on its training residual
    y0r, yhr = S.heldout_gain_over_confounds(yc, [conf], folds, {"tok": ("tokens", tok)},
                                             mode="residual")
    assert np.allclose(y0r, y0)
    assert S.within_r2(yc, yhr["tok"], prot) - r0 > 0.2
    # a continuous method (ridge component) works in both modes
    X = np.eye(8)[tok]
    for mode in ("joint", "residual"):
        _, yx = S.heldout_gain_over_confounds(yc, [conf], folds, {"x": ("ridge", X)}, mode=mode)
        assert S.within_r2(yc, yx["x"], prot) - r0 > 0.2


def test_balanced_folds_and_cluster_bootstrap():
    labels = np.array(["a", "a", "b", "c", "d", "e", "e", "f"])
    sizes = np.array([100, 50, 140, 30, 60, 20, 20, 90])
    f = S.balanced_group_folds(labels, sizes, k=3)
    assert np.array_equal(f, S.balanced_group_folds(labels, sizes, k=3))       # deterministic
    for lab in np.unique(labels):
        assert len(set(f[labels == lab])) == 1                                   # label kept whole
    tot = np.bincount(f, weights=sizes, minlength=3)
    # greedy by size: a=150, b=140, f=90, d=60, e=40, c=30 -> folds 150 | 140+40 | 90+60+30
    assert sorted(tot.tolist()) == [150.0, 180.0, 180.0]
    W = S.bootstrap_weights(len(labels), 50, seed=0, clusters=labels)
    for lab in np.unique(labels):                    # proteins of a topology share the weight
        assert (W[:, labels == lab] == W[:, labels == lab][:, :1]).all()
    nc = len(np.unique(labels))
    _, first = np.unique(labels, return_index=True)
    assert np.allclose(W[:, first].sum(1), nc)       # nc topology draws per replicate


def test_perm_summary_text():
    s0 = S.perm_summary(np.full(1000, 0.03), 0.3)
    assert s0["b"] == 0 and s0["n"] == 1000 and s0["p"] == pytest.approx(1 / 1001)
    assert s0["text"] == "p < 10^-3 (0/1,000)" and s0["null_mean"] == pytest.approx(0.03)
    s1 = S.perm_summary(np.array([0.1, 0.5, 0.9]), 0.5)
    assert s1["b"] == 2 and s1["text"].startswith("p = 0.75")


def test_checkpoint_lambda():
    import token_anova as TA
    assert TA.checkpoint_lambda({"args": {"consistency_mse_weight": 0.25}}) == 0.25
    assert TA.checkpoint_lambda({"consistency_mse_weight": 0.0, "args": {}}) == 0.0
    assert TA.checkpoint_lambda({}) is None
    cfg = CODE_ROOT.parents[1] / "output" / "ensembits_bm_e0" / "config.json"
    if cfg.exists():
        assert TA.checkpoint_lambda(json.loads(cfg.read_text())) == 0.1


def test_ss_proxy_helix():
    # ideal alpha helix C-alpha trace: rise 1.5 A, 100 deg per residue, radius 2.3 A
    t = np.radians(100) * np.arange(20)
    ca = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(20)], 1)
    ss = S.ss_proxy(ca)
    assert (ss[2:-3] == 1).all() and ss[0] == 0


def test_vote_rule():
    from ensembits.baselines.mini3di_baselines import vote_from_frame_tokens
    cent = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    tk = np.array([[1, 1, 2, 3],           # plurality 1 (count 2, unique)
                   [0, 1, 2, 3],           # no plurality: mean (0.5, 0.5) -> nearest (first) 0
                   [3, 3, 2, 2]])          # tie: mean (0.5, 1) -> 2 (0,1) or 3 (1,1): first = 2
    assert vote_from_frame_tokens(tk, cent).tolist() == [1, 0, 2]
    # near-exact tie (ids 1 and 3 both 0.05 from the mean x = 0.15) -> lowest id, whatever the
    # float32 rounding of the centroids (fix round 2)
    cent2 = np.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.0], [0.2, 0.0]], dtype=np.float32)
    assert vote_from_frame_tokens(np.array([[0, 1, 2, 3]]), cent2).tolist() == [1]
    cent4 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 3.0], [0.0, -3.0]])
    tk4 = np.array([[0, 1, 2, 3]])                  # mean (0, 0): ids 0 and 1 tie at 1.0 -> 0
    assert vote_from_frame_tokens(tk4, cent4).tolist() == [0]
    assert vote_from_frame_tokens(tk4, cent4[[1, 0, 2, 3]]).tolist() == [0]


# ---------------------------------------------------------------- end to end

def _synthetic_repo(root: Path):
    """6 mdCATH-div-like domains from the 1ctfA00 fixture + baseline caches with sidecars."""
    sys.path.insert(0, str(CODE_ROOT / "downstream"))
    import rmsf_common as RC
    from ensembits.data_versions import write_provenance
    ncac = np.load(FIXTURES / "1ctfA00_P10_ncac.npz")["ncac"].astype(np.float32)   # (10, 68, 3, 3)
    src = root / "data" / "external" / "mdcath_div"; src.mkdir(parents=True)
    cd = root / "data" / "cached_descriptors"; rb = cd / "mdcath_real_bb"; rb.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pids = [f"d{i}xA00" for i in range(6)]
    lens = [68, 64, 60, 56, 52, 48]
    topo = ["1.10.10", "1.10.20", "2.40.50", "3.30.70", "3.40.50", "4.10.220"]
    man = {"representatives": {p: {"topology": t} for p, t in zip(pids, topo)}}
    (src / "_manifest.json").write_text(json.dumps(man))
    bbs = {}
    for p, L in zip(pids, lens):
        bb = ncac[:, :L] + rng.normal(0, 0.05, (10, L, 3, 3)).astype(np.float32)
        bb4 = np.concatenate([bb, bb[:, :, 2:3]], axis=2)                    # N, CA, C, O
        np.savez(rb / f"{p}.npz", bb_10=bb4, seq=np.array("".join(rng.choice(list("ACDEFGIKLX"), L))))
        np.savez(src / f"{p}.npz", ca_10=bb[:, :, 1], ca_5=bb[:5, :, 1])
        bbs[p] = L
    total = sum(lens)
    out = root / "output" / "phase_b" / "cached_desc"; out.mkdir(parents=True)
    rmsf = rng.gamma(2.0, 0.5, total).astype(np.float32)
    np.save(out / "rmsf_div_allframes.npy", rmsf)
    write_provenance(out / "rmsf_div_allframes.npy", {"label_name": "allframes", "n_rows": total,
                                                      "labels_sha256": RC.label_content_sha256(rmsf),
                                                      "row_order": "_manifest.json representatives",
                                                      "n_pids": len(pids)})

    def cache(name, arrays, frames, uses_seq=False, codebook=None):
        np.savez(cd / name, **arrays)
        RC.write_cache_provenance(cd / name, dataset="mdcath_div", builder="test", inputs={},
                                  n_proteins=len(arrays), frames=frames,
                                  chain_split=RC.CHAIN_SPLITS["mdcath_div"],
                                  sequence_source=(RC.SEQUENCE_SOURCES["mdcath_div"] if uses_seq
                                                   else RC.SEQ_NOT_USED), codebook=codebook)
    cache("mini3di_tokens_mdcath.npz", {p: rng.integers(0, 20, L) for p, L in bbs.items()}, "frame 0")
    cache("mini3di_tokens_K10_mdcath.npz", {p: rng.integers(0, 20, (L, 10)) for p, L in bbs.items()},
          "all 10")
    hist = {}
    for p, L in bbs.items():
        h = rng.random((L, 20)).astype(np.float32); hist[p] = h / h.sum(1, keepdims=True)
    cache("protprofile_K10_mdcath.npz", hist, "first 10 of bb_10", uses_seq=True)
    for name, K in (("aminoaseed", 50), ("protoken", 40), ("esm3struct", 1000)):
        np.save(cd / f"{name}_codebook.npy", np.zeros((K, 4), np.float32))
        toks = {p: rng.integers(0, min(K, 30), L) for p, L in bbs.items()}
        if name == "esm3struct":
            toks[pids[0]][3] = 999                                     # a token used once
        cache(f"{name}_tokens_mdcath.npz", toks, "frame 0", codebook=cd / f"{name}_codebook.npy")
    return src


def _run(root, src, out, *extra):
    env = dict(os.environ, ENSEMBITS_REPO_ROOT=str(root), ENSEMBITS_MDCATH_DIV_SRC=str(src),
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, str(CODE_ROOT / "downstream" / "analyses" / "token_anova.py"),
           "--variants", str(CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10"), "--device", "cpu",
           "--descriptor-source", "compute", "--workers", "2", "--n-perm", "20", "--n-boot", "20",
           "--n-folds", "3", "--out", str(out), "--fig-dir", str(root / "figs"), *extra]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)


@pytest.mark.skipif(not (CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10" / "best.pt").exists(),
                    reason="needs the checkpoint in ckpt/")
def test_token_anova_end_to_end(tmp_path):
    src = _synthetic_repo(tmp_path)
    out = tmp_path / "anova.json"
    exp = tmp_path / "export_{variant}.npz"
    r = _run(tmp_path, src, out, "--export-token-cache", str(exp))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    js = json.loads(out.read_text())
    keys = [m["key"] for m in js["methods"]]
    assert keys[:2] == ["ens:bioemu_cath_misato_lam10:P10", "ens:bioemu_cath_misato_lam10:P1"]
    assert set(keys[2:]) == {"vote_3di", "protprofile_K10", "3di", "aminoaseed", "esm3struct",
                             "protoken", "aa", "chance:bioemu_cath_misato_lam10:P10",
                             "chance:bioemu_cath_misato_lam10:P1"}
    R = np.load(str(out.with_suffix("")) + ".residues.npz")
    n = js["residue_set"]["n_residues"]
    assert n == len(R["ress"]) == js["analysis"]["targets"]["s1"]["anova"]["3di"]["N"]
    # no pooling by default: M = number of distinct raw ids; raw ids exported
    for k in keys:
        a = js["analysis"]["targets"]["s1"]["anova"][k]
        assert "diagnostic_pooled" not in a
        if f"tokens__{k}" in R.files:
            t = R[f"tokens__{k}"]
            assert (t >= 0).all() and a["M"] == len(np.unique(t))
    esm = R["tokens__esm3struct"]
    assert (esm == 999).sum() == 1                                     # the singleton keeps its id
    E = np.load(str(exp).format(variant="bioemu_cath_misato_lam10"), allow_pickle=True)
    assert np.array_equal(E["tokens"], R["tokens__ens:bioemu_cath_misato_lam10:P10"])
    assert np.array_equal(E["tokens_P1"], R["tokens__ens:bioemu_cath_misato_lam10:P1"])
    a = js["analysis"]["targets"]["s1"]["anova"]["ens:bioemu_cath_misato_lam10:P10"]
    assert a["perm_circular"]["n"] == 20 and 1 / 21 <= a["perm_circular"]["p"] <= 1
    assert "rmsf_allframes" in js["analysis"]["targets"]
    d = js["analysis"]["targets"]["s1"]["delta_vs_best_baseline_by_regime"]
    assert set(d) == {"ens:bioemu_cath_misato_lam10:P10", "ens:bioemu_cath_misato_lam10:P1"}
    assert d["ens:bioemu_cath_misato_lam10:P1"]["best_baseline"] in {"3di", "aminoaseed", "esm3struct", "protoken"}
    assert d["ens:bioemu_cath_misato_lam10:P10"]["best_baseline"] in {"vote_3di", "protprofile_K10"}
    # chance rows: the Ensembits row's tokens permuted (same multiset), not identical
    for P in (10, 1):
        e, c = R[f"tokens__ens:bioemu_cath_misato_lam10:P{P}"], R[f"tokens__chance:bioemu_cath_misato_lam10:P{P}"]
        assert sorted(e) == sorted(c) and not np.array_equal(e, c)
    # folds stored, label-disjoint; lambda and code recorded
    assert len(R["fold"]) == n and len(set(R["protein_fold"])) == 3
    for t in set(R["protein_topology"]):
        assert len(set(R["protein_fold"][R["protein_topology"] == t])) == 1
    assert "consistency_mse_weight" in js and js["code"]["files"]
    assert js["analysis"]["bootstrap"]["unit"] == "CATH topology"
    md = (Path(str(out.with_suffix(""))) .with_suffix(".table.md")).read_text()
    assert "p shuffle" not in md and "3 folds" in md and "Foldseek's invalid state 2" in md
    assert "residual" not in md.split("Notes:")[0]
    assert (tmp_path / "figs" / "bioemu_cath_misato_lam10" / "ANOVAtest.pdf").exists()
    assert Path(str(out) + ".provenance.json").exists()

    # --min-count adds a diagnostic block only
    out2 = tmp_path / "anova_mc.json"
    r = _run(tmp_path, src, out2, "--min-count", "5", "--no-figures", "--n-perm", "0")
    assert r.returncode == 0, r.stderr[-3000:]
    js2 = json.loads(out2.read_text())
    a2 = js2["analysis"]["targets"]["s1"]["anova"]["esm3struct"]
    assert a2["diagnostic_pooled"]["n_pooled_tokens"] > 0
    assert a2["eta2"] == pytest.approx(js["analysis"]["targets"]["s1"]["anova"]["esm3struct"]["eta2"])
    R2 = np.load(str(out2.with_suffix("")) + ".residues.npz")
    assert (R2["tokens__esm3struct"] == 999).sum() == 1

    # amino acid not requested as a row: still read for the confound model
    out3 = tmp_path / "anova_noaa.json"
    r = _run(tmp_path, src, out3, "--baselines", "3di", "vote_3di", "--no-figures", "--n-perm", "0")
    assert r.returncode == 0, r.stderr[-3000:]
    js3 = json.loads(out3.read_text())
    assert "aa" not in js3["analysis"]["targets"]["s1"]["anova"]
    assert js3["analysis"]["targets"]["s1"]["gain_over_confounds"]["confounds_only"]["r2"] == pytest.approx(
        js["analysis"]["targets"]["s1"]["gain_over_confounds"]["confounds_only"]["r2"])
    # unknown --pids is an error
    r = _run(tmp_path, src, tmp_path / "x1.json", "--pids", "d0xA00", "nopeA00", "--no-figures")
    assert r.returncode != 0 and "not mdCATH-div representatives" in (r.stdout + r.stderr)
    # a protein missing inside a baseline cache is an error
    import shutil
    cd = tmp_path / "data" / "cached_descriptors"
    f3 = cd / "mini3di_tokens_mdcath.npz"
    shutil.copy2(f3, tmp_path / "keep.npz")
    z = dict(np.load(f3)); del z["d2xA00"]
    np.savez(f3, **z)
    r = _run(tmp_path, src, tmp_path / "x2.json", "--no-figures", "--allow-unverified-baselines")
    assert r.returncode != 0 and "missing from the cache" in (r.stdout + r.stderr)
    shutil.copy2(tmp_path / "keep.npz", f3)
    # a missing baseline cache is an error, not a skipped row
    (tmp_path / "data" / "cached_descriptors" / "protoken_tokens_mdcath.npz").unlink()
    r = _run(tmp_path, src, tmp_path / "anova_missing.json", "--no-figures")
    assert r.returncode != 0 and "protoken_tokens_mdcath.npz missing" in (r.stdout + r.stderr)
    # a missing checkpoint is an error (exit non-zero), not an err: row
    env = dict(os.environ, ENSEMBITS_REPO_ROOT=str(tmp_path), ENSEMBITS_MDCATH_DIV_SRC=str(src))
    r = subprocess.run([sys.executable, str(CODE_ROOT / "downstream" / "analyses" / "token_anova.py"),
                        "--variants", str(tmp_path / "nockpt"), "--device", "cpu", "--out",
                        str(tmp_path / "x.json")], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "no checkpoint config" in (r.stdout + r.stderr)
