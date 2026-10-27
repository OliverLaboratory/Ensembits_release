"""Tests of the input standardisation of the 1D-conv function probes (authors' decision 2026-10-19).

What it does
------------
  1. lookup tables: per-dimension mean / std over the table rows (ddof 0), constant dimensions
     only centred, record + statistics hash (`standardize_lookup_table`);
  2. continuous features: mean / std over the residues of the TRAINING items only, applied to
     every split, input items untouched, record with the training-residue count
     (`InputStandardisation.apply`, `train_residue_stats`); per EC depth in `run_ec_depth`;
     concatenated blocks keep their own rule;
  3. protocol refusal: results of the old protocol tag, without an `input_standardisation`
     record, of another version or with --no-input-standardisation are refused
     (`function_result_problems`);
  4. identical treatment across methods: the Ensembits codebook and the baseline lookup
     tables (ESM3struct / AminoAseed / ProToken / AminoAseed-PCA128 codebooks, mini3di
     centroids of 3Di and Vote_3Di, the aa identity) go through the same function and give the
     same features and head outputs; every baseline of the runners has a rule; the removed
     `*_std` features are refused; all six probes expose --no-input-standardisation; the RMSF
     probes are not touched.

Paper
-----
not described in the paper (head-input protocol of the function probes)

How to run
----------
  python -m pytest tests/test_function_input_std.py -q      # from the codebase root (CPU)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = CODE_ROOT / "downstream"
for _p in (CODE_ROOT, DOWNSTREAM, DOWNSTREAM / "baselines"):
    sys.path.insert(0, str(_p))

import function_common as FC  # noqa: E402

PROBES = ("tokenize_misato_and_eval.py", "tokenize_misato_GO_eval.py",
          "tokenize_misato_binding_eval.py", "baselines/ec_classify_conv1d_misato.py",
          "baselines/conv1d_GO_misato.py", "baselines/conv1d_binding_baselines.py")


# ---------------------------------------------------------------- 1. lookup tables
def test_lookup_table_stats_over_rows():
    rng = np.random.default_rng(0)
    T = rng.normal(3.0, 0.5, size=(50, 4)).astype(np.float32)
    T[:, 2] = 7.0                                              # constant dimension
    Z, rec = FC.standardize_lookup_table(T, "cb")
    assert Z.dtype == np.float32 and Z.shape == T.shape
    keep = [0, 1, 3]
    np.testing.assert_allclose(Z[:, keep].mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(Z[:, keep].std(0), 1, atol=1e-5)          # ddof 0
    np.testing.assert_array_equal(Z[:, 2], 0)                            # centred only
    mu, sd = T.astype(np.float64).mean(0), T.astype(np.float64).std(0)
    np.testing.assert_allclose(Z[:, 0], (T[:, 0] - mu[0]) / sd[0], rtol=1e-5)
    assert rec["kind"] == "lookup" and rec["n_rows"] == 50 and rec["dim"] == 4
    assert rec["n_constant_dims"] == 1 and rec["stats_sha256"] == FC.stats_sha256(mu, sd)
    # deterministic (bit-exact statistics -> same hash); another table -> another hash
    assert FC.standardize_lookup_table(T.copy())[1]["stats_sha256"] == rec["stats_sha256"]
    assert FC.standardize_lookup_table(T * 2)[1]["stats_sha256"] != rec["stats_sha256"]


# ---------------------------------------------------------------- 2. training residues
def _bucket(rng, D=3):
    mk = lambda L, loc: {"pid": f"P{L}{loc}", "x": rng.normal(loc, 2.0, (L, D)).astype(np.float32),
                         "y": np.zeros(2, np.float32)}
    return {"train": [mk(5, 1.0), mk(7, 1.0), mk(3, 1.0)],
            "val": [mk(4, 50.0)], "test": [mk(6, -50.0)]}


def test_continuous_stats_from_training_residues_only():
    rng = np.random.default_rng(1)
    b = _bucket(rng)
    b["train"][0]["x"][:, 1] = 0.0
    for it in b["train"]:
        it["x"][:, 1] = 4.0                                   # constant over training residues
    orig = {s: [it["x"].copy() for it in v] for s, v in b.items()}
    std = FC.InputStandardisation.continuous("protprofile_K8", 3)
    out, rec = std.apply(b)
    Xtr = np.concatenate([it["x"] for it in b["train"]]).astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd_safe = np.where(sd < FC.STD_EPS, 1.0, sd)
    for s in ("train", "val", "test"):                        # train stats applied everywhere
        for it, o in zip(b[s], out[s]):
            np.testing.assert_allclose(o["x"], (it["x"] - mu) / sd_safe, rtol=1e-5, atol=1e-5)
            assert o["x"].dtype == np.float32 and o["pid"] == it["pid"]
    Ztr = np.concatenate([it["x"] for it in out["train"]])
    np.testing.assert_allclose(Ztr[:, [0, 2]].mean(0), 0, atol=1e-5)
    np.testing.assert_array_equal(Ztr[:, 1], 0)               # constant: centred only
    for s, v in b.items():                                    # inputs untouched
        for it, x0 in zip(v, orig[s]):
            np.testing.assert_array_equal(it["x"], x0)
    blk = rec["blocks"][0]
    assert rec["enabled"] and rec["version"] == FC.INPUT_STD_VERSION and rec["rule"] == FC.TRAIN_RULE
    assert blk["n_train_residues"] == 15 and blk["n_train_items"] == 3
    mu2, sd2, n2 = FC.train_residue_stats(b["train"])           # two-pass float64
    np.testing.assert_allclose(mu2, mu, rtol=1e-12); np.testing.assert_allclose(sd2, sd, rtol=1e-12)
    assert blk["stats_sha256"] == FC.stats_sha256(mu2, sd2)
    # val / test values never enter the statistics; training values do
    b2 = {**b, "val": [{**b["val"][0], "x": b["val"][0]["x"] * 100}], "test": []}
    assert std.apply(b2)[1]["stats_sha256"] == rec["stats_sha256"]
    b3 = {**b, "train": b["train"][:2]}
    assert std.apply(b3)[1]["stats_sha256"] != rec["stats_sha256"]
    assert FC.train_residue_stats(b["train"])[2] == 15


def test_run_ec_depth_standardises_per_depth_training_items():
    rng = np.random.default_rng(2)
    split = {"P1": "train", "P2": "train", "P3": "train", "P4": "val", "P5": "test"}
    labels = {"P1": {"1"}, "P2": {"2"}, "P3": {"1"}, "P4": {"1"}, "P5": {"2"}}
    feats = {p: rng.normal(5, 3, (4 + i, 2)).astype(np.float32) for i, p in enumerate(split)}
    seen = {}

    def fn(tr, va, te, n):
        seen["tr"] = np.concatenate([it["x"] for it in tr])
        seen["te"] = te[0]["x"]
        return {"n_test": len(te), "epochs_run": 1, "val": {}}
    r = FC.run_ec_depth(1, labels, split, feats, "random", fn,
                        input_std=FC.InputStandardisation.continuous("random", 2))
    np.testing.assert_allclose(seen["tr"].mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(seen["tr"].std(0), 1, atol=1e-4)
    Xtr = np.concatenate([feats[p] for p in ("P1", "P2", "P3")]).astype(np.float64)
    np.testing.assert_allclose(seen["te"], (feats["P5"] - Xtr.mean(0)) / Xtr.std(0), rtol=1e-5)
    assert r["input_standardisation"]["blocks"][0]["n_train_residues"] == len(Xtr)
    # a lookup rule leaves the items alone (token ids of the Ensembits probes)
    tok = {p: np.arange(4) for p in split}
    T = FC.standardize_lookup_table(np.eye(4))[1]
    FC.run_ec_depth(1, labels, split, tok, "ens", fn,
                    input_std=FC.InputStandardisation.lookup("ens", 4, [T]))
    assert seen["te"] is tok["P5"]


def test_concat_blocks_keep_their_rule():
    rng = np.random.default_rng(3)
    b = _bucket(rng, D=5)
    T = FC.standardize_lookup_table(np.eye(2))[1]
    std = FC.InputStandardisation.continuous("random", 3).concat(
        FC.InputStandardisation.lookup("aa", 2, [T]))
    out, rec = std.apply(b)
    for s in b:                                               # lookup columns 3:5 untouched
        for it, o in zip(b[s], out[s]):
            np.testing.assert_array_equal(o["x"][:, 3:], it["x"][:, 3:])
    Ztr = np.concatenate([it["x"] for it in out["train"]])
    np.testing.assert_allclose(Ztr[:, :3].mean(0), 0, atol=1e-5)
    assert [blk["kind"] for blk in rec["blocks"]] == ["train_residues", "lookup"]
    assert rec["blocks"][1]["dims"] == [3, 5]
    with pytest.raises(FC.FunctionInputError):                # width mismatch
        FC.InputStandardisation.continuous("random", 4).apply(b)


# ---------------------------------------------------------------- 3. refusals
def _good():
    T = FC.standardize_lookup_table(np.eye(3))[1]
    return FC.protocol_record([{"file": "f", "verified": True}],
                              input_std=FC.InputStandardisation.lookup("m", 3, [T]).run_record())


def test_protocol_and_input_std_refusals(tmp_path):
    js = _good()
    assert FC.FUNCTION_PROTOCOL != "misato-function-2026-10-19"        # bumped
    assert FC.function_result_problems(js) == []
    assert FC.function_result_problems({**js, "function_protocol": "misato-function-2026-10-19"})
    no_rec = {k: v for k, v in js.items() if k != "input_standardisation"}
    assert "no input_standardisation" in FC.function_result_problems(no_rec)[0]
    old_v = {**js, "input_standardisation": {**js["input_standardisation"], "version": "v0"}}
    assert FC.function_result_problems(old_v)
    raw = {**js, "input_standardisation": FC.InputStandardisation.disabled("m").run_record()}
    assert "not standardised" in FC.function_result_problems(raw)[0]
    no_sha = {**js, "input_standardisation": {**js["input_standardisation"], "stats_sha256": None}}
    assert FC.function_result_problems(no_sha)
    lig = {**js, "input_standardisation": FC.InputStandardisation("ligand_only", []).run_record()}
    assert FC.function_result_problems(lig) == [] and \
        lig["input_standardisation"]["rule"] == FC.NO_INPUT_RULE
    # the runners refuse to reuse the old results
    import json
    (tmp_path / "old.json").write_text(json.dumps({**js, "function_protocol": "misato-function-2026-10-19"}))
    (tmp_path / "new.json").write_text(json.dumps(js))
    bad = FC.stale_function_outputs([tmp_path / "old.json", tmp_path / "new.json"])
    assert [p.name for p, _ in bad] == ["old.json"]


def test_continuous_run_record_hashes_blocks():
    rng = np.random.default_rng(4)
    std = FC.InputStandardisation.continuous("random", 3)
    r1 = std.apply(_bucket(rng))[1]; r2 = std.apply(_bucket(rng))[1]
    top = std.run_record([("sequence_binding", r1), ("sequence_affinity", r2)])
    assert top["per_block"] == ["sequence_binding", "sequence_affinity"]
    assert top["stats_sha256"] and top["stats_sha256"] != r1["stats_sha256"]
    assert FC.function_result_problems(
        FC.protocol_record([{"file": "f", "verified": True}], input_std=top)) == []
    assert std.run_record([])["stats_sha256"] is None         # no block -> refused


# ---------------------------------------------------------------- 4. same rule for every method
def _caches(tmp_path, cb, cent):
    rng = np.random.default_rng(5)
    toks = {"AAAA": rng.integers(0, cb.shape[0], 9), "BBBB": rng.integers(0, cb.shape[0], 4)}
    for name in ("esm3struct", "aminoaseed", "protoken"):
        np.savez(tmp_path / f"{name}_tokens_misato.npz", **toks)
        np.save(tmp_path / f"{name}_codebook.npy", cb)
    t3 = {k: v % cent.shape[0] for k, v in toks.items()}
    np.savez(tmp_path / "mini3di_tokens_misato.npz", **t3)
    np.savez(tmp_path / "mini3di_tokens_K8_misato.npz",
             **{k: rng.integers(0, cent.shape[0], (len(v), 8)) for k, v in toks.items()})
    np.save(tmp_path / "mini3di_centroids.npy", cent)
    return toks, t3


def test_same_rule_ensembits_and_baselines(tmp_path, monkeypatch):
    import ec_classify_conv1d_misato as E
    import tokenize_misato_and_eval as T
    from ensembits.baselines.mini3di_baselines import vote_from_frame_tokens
    rng = np.random.default_rng(6)
    cb = (rng.normal(0, 0.8, (16, 6)) + 2.0).astype(np.float32)
    cent = rng.normal(0, 1, (20, 2)).astype(np.float32)
    toks, t3 = _caches(tmp_path, cb, cent)
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    keep = set(toks)
    ens_cb, ens_std = T.codebook_input_std(torch.from_numpy(cb), "runX")
    ens_rec = ens_std.base_record()
    for spec in ("esm3struct", "aminoaseed", "protoken"):
        ts = E._TableStd()
        feats, D = E._load_features_all(spec, keep, ts)
        assert D == 6 and len(ts.records) == 1
        for p in keep:                       # baseline feature = row of the Ensembits-rule table
            np.testing.assert_allclose(feats[p], ens_cb.numpy()[toks[p]], rtol=1e-6)
        rec = E.feature_input_std(spec, D, ts.records).base_record()
        assert rec["rule"] == ens_rec["rule"] == FC.LOOKUP_RULE
        assert rec["stats_sha256"] == ens_rec["stats_sha256"]
        raw, _ = E._load_features_all(spec, keep, None)       # --no-input-standardisation
        np.testing.assert_array_equal(raw["AAAA"], cb[toks["AAAA"]])
    # mini3di centroid tables (3Di: token -> row; Vote_3Di: vote on raw centroids -> row)
    cstd, crec = FC.standardize_lookup_table(cent, "mini3di centroids")
    ts = E._TableStd(); f3, _ = E._load_features_all("3di_tokens", keep, ts)
    np.testing.assert_allclose(f3["AAAA"], cstd[t3["AAAA"]], rtol=1e-6)
    assert ts.records[0]["stats_sha256"] == crec["stats_sha256"]
    ts = E._TableStd(); fv, _ = E._load_features_all("vote_3di", keep, ts)
    kt = np.load(tmp_path / "mini3di_tokens_K8_misato.npz")["AAAA"]
    np.testing.assert_allclose(fv["AAAA"], cstd[vote_from_frame_tokens(kt, cent)], rtol=1e-6)
    # aa: one-hot rows of the 21 x 21 identity, standardised over its rows
    oh = {"AAAA": np.eye(21, dtype=np.float32)[[0, 3, 20]]}
    ts = E._TableStd(); fa = E._aa_lookup(oh, ts)
    istd = FC.standardize_lookup_table(np.eye(21))[0]
    np.testing.assert_allclose(fa["AAAA"], istd[[0, 3, 20]], rtol=1e-6)
    assert ts.records[0]["n_rows"] == 21
    # same head output: Ensembits token head on the standardised codebook == baseline head on
    # the standardised lookup features (zero padding in both)
    items = [{"x": toks[p], "y": np.zeros(2, np.float32)} for p in sorted(keep)]
    feat_items = [{"x": ens_cb.numpy()[toks[p]], "y": np.zeros(2, np.float32)} for p in sorted(keep)]
    X, M, _ = T.collate(items); Xf, Mf, _ = E.collate(feat_items)
    torch.manual_seed(0)
    tm = T.TokenConv1D(ens_cb, 2).eval(); bm = E.Conv1DClassifier(6, 2).eval()
    bm.load_state_dict({k: v for k, v in tm.state_dict().items()
                        if k.split(".")[0] in ("conv", "head")})
    with torch.no_grad():
        assert torch.allclose(tm(X, M), bm(Xf, Mf), atol=1e-5)


def test_pca128_lookup_is_standardised(tmp_path, monkeypatch):
    import ec_classify_conv1d_misato as E
    rng = np.random.default_rng(7)
    cb = rng.normal(0, 5, (200, 160)).astype(np.float32)
    toks, _ = _caches(tmp_path, cb, rng.normal(0, 1, (20, 2)).astype(np.float32))
    monkeypatch.setattr(E, "CACHE_DIR", tmp_path)
    ts = E._TableStd()
    feats, D = E._load_features_all("aminoaseed_pca128", set(toks), ts)
    proj, _ = E.codebook_pca(cb, 128)
    assert D == 128 and ts.records[0]["n_rows"] == 200
    np.testing.assert_allclose(feats["AAAA"], FC.standardize_lookup_table(proj)[0][toks["AAAA"]],
                               rtol=1e-5, atol=1e-5)


def test_every_runner_feature_has_a_rule_and_std_features_removed():
    import ec_classify_conv1d_misato as E
    import run_misato_baselines as R
    import conv1d_binding_baselines as CB  # noqa: F401
    assert not (E.LOOKUP_SPECS & E.CONTINUOUS_SPECS)
    for f in R.BASELINES:
        assert f in E.LOOKUP_SPECS | E.CONTINUOUS_SPECS, f
    assert {"aa", "esm3struct", "aminoaseed", "protoken", "3di_tokens", "vote_3di"} <= E.LOOKUP_SPECS
    assert {"protprofile_K8", "random", "pca_eig"} <= E.CONTINUOUS_SPECS
    src = (DOWNSTREAM / "baselines" / "conv1d_binding_baselines.py").read_text()
    choices = src[src.index("choices=['aa'"):src.index("ap.add_argument('--device'")]
    import re
    for f in re.findall(r"'([A-Za-z0-9_]+)'", choices):
        if f != "ligand_only":
            assert f in E.LOOKUP_SPECS | E.CONTINUOUS_SPECS, f
    for f in ("aminoaseed_std", "esm3struct_std"):
        assert f not in E.FEATURE_CACHES
        with pytest.raises(ValueError):
            E.load_features_checked(f)
    with pytest.raises(FC.FunctionInputError):
        E.feature_input_std("unknown_feature", 3)
    with pytest.raises(FC.FunctionInputError):            # a lookup feature without its table
        E.feature_input_std("aminoaseed", 3, [])
    assert E.feature_input_std("aa", 21, [], enabled=False).base_record()["enabled"] is False


def test_probes_default_and_flag_and_rmsf_untouched():
    for rel in PROBES:
        s = (DOWNSTREAM / rel).read_text()
        assert "add_input_std_arg(ap)" in s and "input_std_tag(args)" in s, rel
        assert "standardize_codebook_rows" not in s and "--standardize-codebook'" not in s, rel
        assert "input_std=" in s or "input_standardisation" in s, rel
    import argparse
    ap = argparse.ArgumentParser(); FC.add_input_std_arg(ap)
    a = ap.parse_args([]); a.tag = "stab"; FC.input_std_tag(a)
    assert a.tag == "stab" and a.no_input_standardisation is False
    a = ap.parse_args(["--no-input-standardisation"]); a.tag = "stab"; FC.input_std_tag(a)
    assert a.tag == "stab_rawin"
    for rel in ("probe_rmsf_eval.py", "probe_rmsf_misato.py", "probe_rmsf_baselines.py",
                "rmsf_common.py"):
        s = (DOWNSTREAM / rel).read_text()
        assert "InputStandardisation" not in s and "standardize_lookup_table" not in s, rel
