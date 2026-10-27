"""Tests of the camera-ready CryptoBench port (step 5 of camera_ready/cryptobench_audit/TRACKER.md).

What it does
------------
Synthetic data only (no CryptoBench run, no metric of any real model):
  * cryptobench_common: gap-aware residue index from label_seq_id, placed carbonyl O, entry
    layout (apo id, chains in sorted chain-id order), per-entry / pooled metrics against
    sklearn, per-seed result + sidecar round trip, token-file provenance gate, non-cryptic
    mask, move_aside (never overwrites), train-only truncation, no silent chain drop;
  * cryptobench_build_apo: selection parsing and the (number, insertion code) label key;
  * conv probe: a residue's output does not depend on the padding of its batch (pads zeroed
    before every convolution); MLP shape and the 20-dim amino-acid one-hot;
  * protein_level_stats: task `cryptobench` with spec-given clusters (sha256 checked),
    subsets (>= k positives, residue mask; a mask that drops a positive is refused), bundles
    Holm-adjusted per family, refusal without clusters;
  * the runner: the pre-specified family file equals `make_family_template()` and the cluster
    file it names has the recorded sha256; the plan has every row, probe and dependency.

Paper
-----
not described in the paper (infrastructure for the CryptoBench experiment)

How to run
----------
    python -m pytest tests/test_cryptobench_port.py
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
CB = CODE_ROOT / "downstream" / "cryptobench"
for _p in (CODE_ROOT, CODE_ROOT / "downstream", CB):
    sys.path.insert(0, str(_p))

import cryptobench_common as CC  # noqa: E402
import protein_preds as PP  # noqa: E402
import protein_level_stats as S  # noqa: E402


# ---------------------------------------------------------------- structures
def test_chain_residue_index_gap_aware():
    ri = CC.chain_residue_index([5, 6, 7, 10, 11, 20])
    assert ri.tolist() == [0, 1, 2, 5, 6, 15]
    with pytest.raises(ValueError):
        CC.chain_residue_index([1, 2, 0, 3])          # missing label_seq_id
    with pytest.raises(ValueError):
        CC.chain_residue_index([1, 3, 3, 4])          # not strictly increasing


def test_virtual_o_geometry():
    rng = np.random.default_rng(0)
    L = 12
    ca = np.cumsum(rng.normal(size=(L, 3)) + [3.8, 0, 0], axis=0)
    n = ca + rng.normal(scale=0.3, size=(L, 3)) - [1.2, 0, 0]
    c = ca + rng.normal(scale=0.3, size=(L, 3)) + [1.2, 0, 0]
    o = CC.virtual_o(n, ca, c)
    assert np.allclose(np.linalg.norm(o - c, axis=1), 1.231, atol=1e-3)


def test_backbone4_places_only_missing_o():
    L = 20
    bb = np.random.default_rng(1).normal(size=(1, L, 3, 3)).astype(np.float32)
    o = np.random.default_rng(2).normal(size=(L, 3)).astype(np.float32)
    o[3] = np.nan
    class Z(dict):                                    # mimics np.load's NpzFile
        files = ["bb", "o"]
    bb4, n = CC.backbone4(Z({"bb": bb, "o": o}))
    assert n == 1 and np.isfinite(bb4).all()
    assert np.array_equal(bb4[np.arange(L) != 3, 3], o[np.arange(L) != 3])


def test_build_apo_label_keys():
    sys.path.insert(0, str(CB))
    import cryptobench_build_apo as BA
    sel = BA.parse_selection(["B_12", "B_60A", "A_7", "x"])
    assert sel == {"B": {"12", "60A"}, "A": {"7"}}
    assert BA.res_key("60", "A") == "60A" and BA.res_key("60", "") == "60"
    assert BA.res_key("60", " ") == "60"
    assert BA.BUILD_VERSION == CC.STRUCT_BUILD_VERSION


# ---------------------------------------------------------------- entries and metrics
def test_entry_layout_and_concat():
    keys = ["1abc_B", "1abc_A", "2xyz_A", "1abc_AA"]
    lay = CC.entry_layout(keys)
    assert list(lay) == ["1abc", "2xyz"]
    assert lay["1abc"] == ["1abc_A", "1abc_AA", "1abc_B"]
    per = {k: np.full(i + 1, i) for i, k in enumerate(keys)}
    cat = CC.entry_concat(per, lay)
    assert cat["1abc"].tolist() == [1, 1] + [3, 3, 3, 3] + [0]


def test_entry_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(3)
    y = {f"e{i}": (rng.random(40) < 0.2).astype(int) for i in range(5)}
    for k in y:
        y[k][0] = 1; y[k][1] = 0
    s = {k: rng.random(40) + 0.5 * v for k, v in y.items()}
    m = CC.entry_metrics(s, y)
    assert m["n_entries"] == 5 and m["n_entries_scored"] == 5
    assert np.isclose(m["auroc_per_entry"], np.mean([roc_auc_score(y[k], s[k]) for k in y]))
    assert np.isclose(m["ap_per_entry"], np.mean([average_precision_score(y[k], s[k]) for k in y]))
    Y = np.concatenate([y[k] for k in sorted(y)]); P = np.concatenate([s[k] for k in sorted(y)])
    assert np.isclose(m["auroc_pooled"], roc_auc_score(Y, P))
    assert np.isclose(m["ap_pooled"], average_precision_score(Y, P))


def _toy_split(rng, entries, chains_per=(1, 2)):
    preds, labels = {}, {}
    for e in entries:
        for c in "AB"[:int(rng.integers(*chains_per)) + 1]:
            k = f"{e}_{c}"
            y = (rng.random(30) < 0.25).astype(np.int64); y[0] = 1; y[1] = 0
            labels[k] = y; preds[k] = (rng.random(30) + 0.3 * y).astype(np.float32)
    return preds, labels


def _write_run(tmp, probe, row, seed, preds, labels, shift=0.0):
    pr = {s: {k: v + shift * labels[s][k] for k, v in preds[s].items()} for s in preds}
    lay = {s: CC.entry_layout(list(labels[s])) for s in labels}
    f = tmp / probe / f"{row}__token__seed{seed}.json"
    CC.write_seed_result(f, preds=pr, labels=labels, layout=lay,
                         meta={"row": row, "probe": probe, "seed": seed, "set": "apo",
                               "readout": "token"},
                         result={"name": row, "metrics": {}})
    return f


def test_seed_result_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    pv, yv = _toy_split(rng, ["v1", "v2"])
    pt, yt = _toy_split(rng, ["t1", "t2", "t3"])
    f = _write_run(tmp_path, "mlp", "rowA", 0, {"val": pv, "test": pt}, {"val": yv, "test": yt})
    assert CC.seed_result_ok(f)
    run = S.load_run(f)
    assert run["meta"]["task"] == "cryptobench" and run["meta"]["probe"] == "mlp"
    assert run["test"]["ids"].tolist() == ["t1", "t2", "t3"]
    lay = CC.entry_layout(list(yt))
    cat = CC.entry_concat(yt, lay)
    for i, e in enumerate(run["test"]["ids"]):
        sl = slice(run["test"]["offsets"][i], run["test"]["offsets"][i + 1])
        assert np.array_equal(run["test"]["label"][sl], cat[e])
    f.with_name(f.stem + ".preds.npz").write_bytes(b"changed")
    assert not CC.seed_result_ok(f)


# ---------------------------------------------------------------- token provenance gate
def test_check_token_file(tmp_path):
    p = tmp_path / "cryptobench_tokens_x.npz"
    np.savez(p, a=np.zeros(3))
    man = tmp_path / "apo_manifest.json"; man.write_text("{}")
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man)                       # no sidecar (rebuttal-era cache)
    side = p.with_name(p.name + ".provenance.json")
    side.write_text(json.dumps({"tokenizer": "aminoaseed", "chain_rule": "split_5A"}))
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man)                       # wrong chain rule
    side.write_text(json.dumps({"tokenizer": "aminoaseed", "chain_rule": CC.BASELINE_CHAIN_RULE,
                                "manifest_sha256": "0" * 64, "n_failed": 0}))
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man)                       # other structure manifest
    side.write_text(json.dumps({"tokenizer": "aminoaseed", "chain_rule": CC.BASELINE_CHAIN_RULE,
                                "manifest_sha256": CC.file_sha256(man), "n_failed": 0}))
    assert CC.check_token_file(p, man)["tokenizer"] == "aminoaseed"
    ms = CC.file_sha256(man)
    side.write_text(json.dumps({"checkpoint": {"variant": "ensembits_bm_lam1_rv",
                                               "best_pt_sha256": "ab"}, "n_failed": 2,
                                "manifest_sha256": ms}))
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man)                       # failed chains
    side.write_text(json.dumps({"checkpoint": {"variant": "v", "best_pt_sha256": "ab"},
                                "n_failed": 0}))
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man)                       # Ensembits file without manifest
    side.write_text(json.dumps({"checkpoint": {"variant": "v", "best_pt_sha256": "ab"},
                                "n_failed": 0, "manifest_sha256": "0" * 64}))
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man)                       # Ensembits file, other manifest
    side.write_text(json.dumps({"checkpoint": {"variant": "v", "best_pt_sha256": "ab"},
                                "n_failed": 0, "manifest_sha256": ms}))
    assert CC.check_token_file(p, man)["checkpoint"]["variant"] == "v"


def test_load_chains_truncates_train_only_and_never_drops(tmp_path):
    sd = tmp_path / "cryptobench_apo"; sd.mkdir()
    ents = []
    for k, fold, L in (("1aaa_A", "train-0", 30), ("2bbb_A", "test", 30)):
        np.savez(sd / f"{k}.npz", y=np.zeros(L, np.int8), seq=np.array("A" * L))
        ents.append({"apo": k[:4], "chain": "A", "fold": fold})
    tok = tmp_path / "t.npz"
    np.savez(tok, **{"1aaa_A": np.arange(30), "2bbb_A": np.arange(30)})
    ch = CC.load_chains({"entries": ents}, sd, "tokens", tok, max_len=10)
    assert [len(x[2]) for x in ch] == [10, 30]
    np.savez(tok, **{"1aaa_A": np.arange(30)})
    with pytest.raises(CC.TokenFileError):
        CC.load_chains({"entries": ents}, sd, "tokens", tok)


def test_noncryptic_mask(tmp_path):
    sd = tmp_path / "cryptobench_apo"; sd.mkdir()
    y = np.array([0, 1, 0, 0, 1], np.int8); nc = np.array([1, 1, 0, 1, 0], np.int8)
    np.savez(sd / "1aaa_A.npz", y=y, nc=nc)
    np.savez(sd / "1aaa_B.npz", y=y, nc=nc)
    man = {"entries": [{"apo": "1aaa", "chain": c, "fold": "test"} for c in "BA"]}
    m = CC.noncryptic_mask(man, sd)
    assert m == {"1aaa": {"n_res": 10, "drop": [0, 3, 5, 8]}}
    np.savez(sd / "1aaa_A.npz", y=y)
    with pytest.raises(ValueError):
        CC.noncryptic_mask(man, sd)


def test_move_aside_never_overwrites(tmp_path):
    f = tmp_path / "cryptobench_tokens_a.npz"
    for i in range(2):
        f.write_text(str(i)); f.with_name(f.name + ".provenance.json").write_text("{}")
        CC.move_aside(f, date="20990101")
    d = tmp_path / "_pre_20990101"
    names = sorted(p.name for p in d.iterdir())
    assert names == ["cryptobench_tokens_a.npz", "cryptobench_tokens_a.npz.1",
                     "cryptobench_tokens_a.npz.provenance.json",
                     "cryptobench_tokens_a.npz.provenance.json.1"]
    assert not f.exists()


# ---------------------------------------------------------------- probes
def test_conv_output_independent_of_batch_padding():
    import cryptobench_probe as CP
    torch.manual_seed(0)
    model = CP.ResidueConv1D(torch.randn(50, 8)).eval()
    a = np.random.default_rng(0).integers(0, 50, 12)
    b = np.random.default_rng(1).integers(0, 50, 40)
    alone = CP.predict(model, [("a", a, np.zeros(12, np.float32))], "cpu", 16)["a"]
    both = CP.predict(model, [("a", a, np.zeros(12, np.float32)),
                              ("b", b, np.zeros(40, np.float32))], "cpu", 16)["a"]
    assert np.allclose(alone, both, atol=1e-6)
    dense = CP.ResidueConv1D(None, in_dim=6).eval()
    x = np.random.default_rng(2).normal(size=(9, 6)).astype(np.float32)
    x2 = np.random.default_rng(3).normal(size=(30, 6)).astype(np.float32)
    p1 = CP.predict(dense, [("x", x, np.zeros(9, np.float32))], "cpu", 16)["x"]
    p2 = CP.predict(dense, [("x", x, np.zeros(9, np.float32)),
                            ("y", x2, np.zeros(30, np.float32))], "cpu", 16)["x"]
    assert np.allclose(p1, p2, atol=1e-6)


def test_mlp_shape_and_aa_onehot():
    import cryptobench_mlp_probe as MP
    m = MP.build_mlp(7)
    assert m(torch.zeros(3, 7)).shape == (3, 2)
    ch = MP.featurize([("k", "test", np.array([0, 19, 20]), np.zeros(3))], "aa")
    F = ch[0][2]
    assert F.shape == (3, 20) and F[0, 0] == 1 and F[1, 19] == 1 and F[2].sum() == 0


# ---------------------------------------------------------------- statistics
def _cluster_file(tmp, entries):
    p = tmp / "clusters.json"
    p.write_text(json.dumps({e: ("c0" if i < 2 else f"c{i}") for i, e in enumerate(entries)}))
    return {"path": str(p), "sha256": PP.file_sha256(p)}


def _runs(tmp, probe, row, preds, labels, shift, seeds=(0, 1)):
    return [str(_write_run(tmp, probe, row, s, preds, labels, shift=shift + 0.001 * s))
            for s in seeds]


def test_stats_cryptobench_family_bundle_and_subsets(tmp_path):
    rng = np.random.default_rng(5)
    pv, yv = _toy_split(rng, ["v1", "v2"])
    tents = [f"t{i}" for i in range(8)]
    pt, yt = _toy_split(rng, tents)
    P, Y = {"val": pv, "test": pt}, {"val": yv, "test": yt}
    a = _runs(tmp_path, "mlp", "ensA", P, Y, 0.5)
    b = _runs(tmp_path, "mlp", "base", P, Y, 0.0)
    c = _runs(tmp_path, "conv", "base", P, Y, 0.0)
    clus = _cluster_file(tmp_path, tents)
    lay = CC.entry_layout(list(yt))
    ycat = CC.entry_concat(yt, lay)
    mask = {"schema": CC.MASK_SCHEMA, "entries": {
        e: {"n_res": len(v), "drop": np.flatnonzero(v == 0)[:3].tolist()} for e, v in ycat.items()}}
    mp = tmp_path / "mask.json"; mp.write_text(json.dumps(mask))
    comp = {"task": "cryptobench", "metric": "auroc_per_protein", "a": a, "b": b}
    bundle = {"bundle": "t", "B": 200, "seed": 1, "clusters": clus, "families": [
        {"family": "f1", "comparisons": [{**comp, "id": "x1"},
                                         {**comp, "id": "x2", "metric": "auroc_pooled"}]},
        {"family": "f2", "comparisons": [
            {**comp, "id": "y1", "subset": {"min_positives": 5}},
            {**comp, "id": "y2", "subset": {"mask": {"path": str(mp)}}}]}]}
    assert S.run_bundle(bundle, check_only=True)["checked"] == {"f1": ["x1", "x2"],
                                                               "f2": ["y1", "y2"]}
    res = S.run_bundle(bundle)
    assert [len(f["comparisons"]) for f in res["families"]] == [2, 2]
    for f in res["families"]:
        for r in f["comparisons"]:
            assert np.isfinite(r["delta"]) and np.isfinite(r["p_holm"]) and r["n_clusters"] >= 1
            assert r["cluster_source"]["sha256"] == clus["sha256"]
    y1 = res["families"][1]["comparisons"][0]
    assert y1["n_items"] == sum(int(v.sum() >= 5) for v in ycat.values())
    # a mask that drops a positive is refused
    e0 = sorted(ycat)[0]
    mask["entries"][e0]["drop"] = [int(np.flatnonzero(ycat[e0] == 1)[0])]
    mp.write_text(json.dumps(mask))
    with pytest.raises(S.RefusedError):
        S.run_bundle(bundle)
    # pairing two probes is refused; no clusters refused; changed cluster file refused
    with pytest.raises(S.RefusedError):
        S.run_family({"family": "z", "B": 50, "clusters": clus,
                      "comparisons": [{**comp, "id": "z1", "b": c}]})
    with pytest.raises(S.RefusedError):
        S.run_family({"family": "z", "B": 50, "comparisons": [{**comp, "id": "z1"}]})
    with pytest.raises(S.RefusedError):
        S.run_family({"family": "z", "B": 50, "clusters": {**clus, "sha256": "0" * 64},
                      "comparisons": [{**comp, "id": "z1"}]})


# ---------------------------------------------------------------- runner and family
def test_family_file_is_the_prespecified_template():
    import run_cryptobench_rv as R
    tpl = json.loads(R.FAMILY_TEMPLATE.read_text())
    assert tpl == R.make_family_template()
    prim = [f for f in tpl["families"] if f["role"] == "primary"]
    assert len(prim) == 1 and len(prim[0]["comparisons"]) == 10
    assert {c["metric"] for c in prim[0]["comparisons"]} == {"auroc_per_protein"}
    assert all("/mlp/" in c["a"][0] and "/mlp/" in c["b"][0] for c in prim[0]["comparisons"])
    assert len({f["family"] for f in tpl["families"]}) == len(tpl["families"])
    assert S.check_family({**tpl["families"][0], "clusters": tpl["clusters"]}) == []
    from ensembits.paths import REPO_ROOT
    cf = REPO_ROOT / tpl["clusters"]["path"]
    if cf.exists():
        assert PP.file_sha256(cf) == tpl["clusters"]["sha256"]
        assert len(json.loads(cf.read_text())) == 222
    inst = R.instantiate_family(tpl, REPO_ROOT / "output" / "x", REPO_ROOT / "data" / "y", B=7)
    s = json.dumps(inst)
    assert "{RUN}" not in s and "{STRUCT}" not in s and inst["B"] == 7


def by_name(jobs):
    return {j.name: j for j in jobs}


def test_runner_plan(tmp_path):
    import _cb_paths as CBP
    import run_cryptobench_rv as R
    args = Namespace(run_tag="t", legacy_layout=False, cryptobench_dir=None,
                     structs_root=str(tmp_path / "s"), manifest_dir=str(tmp_path / "s"),
                     tokens_dir=str(tmp_path / "k"), out_dir=str(tmp_path / "o"),
                     gemmi_python="py", workers=2, seeds=5, smoke_epochs=None, stats_B=None,
                     modified_residues="parent", esm2_row="on", conv_standardize="train",
                     selection="val_pooled_auroc", aa_order="collaborator", esm2_layer=33)
    jobs = R.plan(args, CBP.resolve_paths(args))
    names = [j.name for j in jobs]
    assert len(names) == len(set(names))
    probes = [j for j in jobs if j.name.startswith(("mlp_", "conv_"))]
    assert len(probes) == 2 * (5 + 8 + 6) + 1                 # + ESM-2 (MLP only)
    assert "conv_esm2_3b__esm2" not in names and "mlp_esm2_3b__esm2" in names
    assert "mlp_esm2_3b__esm2" not in by_name(jobs)["family"].deps
    assert all(j.gpu for j in probes)
    for r in R.ENS_RUNS:
        assert f"mlp_ens_{r}_rv__token" in names and f"conv_ens_{r}_rv__token" in names
    for r in R.LAMBDA_CANDIDATES:
        for ro in ("z", "q", "q0"):
            assert f"mlp_ens_{r}_rv__{ro}" in names
    by = {j.name: j for j in jobs}
    assert "dense_ens_lam1_rv" in by["mlp_ens_lam1_rv__z"].deps
    assert "tok_protoken_wc" in by["conv_protoken_wc__token"].deps
    assert set(by["family"].deps) >= {j.name for j in probes} - {"mlp_esm2_3b__esm2"}
    assert by["build"].python == "py" and not by["tok_mini3di_wc"].gpu
    assert all("--seeds" in j.argv and "5" in j.argv for j in probes)
