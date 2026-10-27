"""Tests of the per-protein prediction sidecars and the protein-level significance tests.

What it does
------------
Synthetic data only (no probe output, no test metric of any real run):
  * downstream/protein_preds.py: regrouping of residues by protein, deterministic sidecar
    bytes, verification (missing / changed sidecar refused), item-hash check of the function
    writers, `needs_rerun`;
  * the probes' train functions with return_preds=True: the dumped predictions reproduce the
    metrics the probe stores (RMSF per-protein Spearman, binding per-protein AUROC, EC / GO
    Fmax, affinity Spearman) and keep the item order;
  * downstream/protein_level_stats.py: per-protein metrics equal the probes' rules, seed
    averaging and its refusals, paired cluster bootstrap (no difference -> p = 1, a clear
    shift -> small p and a CI around it), fold-paired mean, Holm, family checks and the
    refusal of a result without a sidecar.

Paper
-----
not described in the paper (infrastructure for A4 (a))

How to run
----------
    python -m pytest tests/test_protein_level_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = CODE_ROOT / "downstream"
for _p in (CODE_ROOT, DOWNSTREAM, DOWNSTREAM / "baselines", CODE_ROOT / "data_prep"):
    sys.path.insert(0, str(_p))

import protein_preds as PP  # noqa: E402
import protein_level_stats as S  # noqa: E402
import function_common as FC  # noqa: E402


# ---------------------------------------------------------------- helpers
def _residue_block(rng, n_prot=6, L=(12, 20), names=None):
    ids, P, Y = [], [], []
    for i in range(n_prot):
        n = int(rng.integers(*L))
        y = rng.standard_normal(n).astype(np.float32)
        ids.append(names[i] if names else f"p{i:02d}")
        P.append((y + rng.standard_normal(n)).astype(np.float32)); Y.append(y)
    return ids, P, Y


def _write_rmsf_like(tmp, name, ids, P, Y, seed=0, extra_meta=None):
    split = PP.residue_split_from_items(ids, P, Y)
    js = tmp / f"{name}.json"
    rec = PP.write_preds(PP.sidecar_path(js), kind="residue_regression",
                         splits={"val": split, "test": split},
                         meta={"task": "rmsf", "dataset": "mdcath_div", "seed": seed,
                               "residue_set_sha256": "rs", **(extra_meta or {})},
                         json_dir=tmp)
    js.write_text(json.dumps({"seed": seed, "residue_set_sha256": "rs", PP.PREDS_KEY: rec}))
    return js


# ---------------------------------------------------------------- protein_preds
def test_residue_regrouping_and_roundtrip(tmp_path):
    names = np.array(["b", "a", "b", "a", "a"])
    pred = np.array([0.2, 1.0, 0.1, 3.0, 2.0], np.float32)
    lab = np.array([2, 10, 1, 30, 20], np.float32)
    ri = np.array([1, 0, 0, 2, 1])
    blk = PP.residue_split_from_residues(names, pred, lab, ri, row=np.array([11, 0, 10, 2, 1]))
    assert blk["ids"].tolist() == ["a", "b"] and blk["offsets"].tolist() == [0, 3, 5]
    assert blk["label"].tolist() == [10, 20, 30, 1, 2]           # residue order inside items
    assert blk["row"].tolist() == [0, 1, 2, 10, 11]
    with pytest.raises(PP.PredsError):
        PP.residue_split_from_residues(names, pred, lab, np.zeros(5))   # duplicate index
    js = tmp_path / "r.json"
    rec = PP.write_preds(PP.sidecar_path(js), kind="residue_regression",
                         splits={"val": blk, "test": blk}, meta={"task": "rmsf"}, json_dir=tmp_path)
    rec2 = PP.write_preds(PP.sidecar_path(js), kind="residue_regression",
                          splits={"val": blk, "test": blk}, meta={"task": "rmsf"}, json_dir=tmp_path)
    assert rec["sha256"] == rec2["sha256"] == PP.file_sha256(tmp_path / "r.preds.npz")
    got = PP.load_preds(rec, js)
    np.testing.assert_array_equal(got["test"]["pred"], blk["pred"])
    assert got["test"]["pred"].dtype == np.float32 and got["meta"]["n_items"]["val"] == 2


def test_sidecar_refusals(tmp_path):
    js = tmp_path / "r.json"
    assert "no per-protein" in PP.preds_problem(None, js)
    assert PP.needs_rerun(None, js)
    rec = PP.not_selected_record("constant")
    assert PP.needs_rerun(rec, js) is None and "no selected model" in PP.preds_problem(rec, js)
    blk = PP.protein_split(["x", "y"], [1.0, 2.0], [1.0, 3.0])
    rec = PP.write_preds(PP.sidecar_path(js), kind="protein_regression",
                         splits={"val": blk, "test": blk}, meta={}, json_dir=tmp_path)
    assert PP.preds_problem(rec, js) is None and PP.needs_rerun(rec, js) is None
    p = tmp_path / rec["file"]
    p.write_bytes(p.read_bytes() + b"x")
    assert "changed" in PP.preds_problem(rec, js) and PP.needs_rerun(rec, js)
    with pytest.raises(PP.PredsError):
        PP.load_preds(rec, js)
    p.unlink()
    assert "missing" in PP.preds_problem(rec, js)


def test_function_writer_checks_items(tmp_path):
    out = tmp_path / "go.json"
    items = {"val": [{"pid": "B"}, {"pid": "A"}], "test": [{"pid": "C"}]}
    r = {**FC.items_record({**items, "train": []}), "seed": 3}
    S_ = np.array([[0.1, 0.9], [0.8, 0.2]]); Y = np.array([[0, 1], [1, 0]])
    preds = {"val": PP.items_preds(items["val"], S_, Y),
             "test": PP.items_preds(items["test"], S_[:1], Y[:1])}
    rec = PP.write_function_preds(out, preds, r, task="GO", classes=["c1", "c2"])
    got = PP.load_preds(rec, out)
    assert got["val"]["ids"].tolist() == ["A", "B"]                 # sorted, rows follow
    np.testing.assert_allclose(got["val"]["pred"][0], [0.8, 0.2])
    assert got["classes"] == ["c1", "c2"] and got["meta"]["seed"] == 3
    bad = {"val": PP.items_preds([{"pid": "Z"}, {"pid": "A"}], S_, Y), "test": preds["test"]}
    with pytest.raises(PP.PredsError):
        PP.write_function_preds(out, bad, r, task="GO", classes=["c1", "c2"])


def test_contiguous_residue_index():
    np.testing.assert_array_equal(PP.contiguous_residue_index([3, 3, 1, 1, 1, 2]),
                                  [0, 1, 0, 1, 2, 0])
    with pytest.raises(PP.PredsError):
        PP.contiguous_residue_index([1, 2, 1])


# ---------------------------------------------------------------- probes' return_preds
def test_rmsf_fit_returns_selected_predictions():
    import probe_rmsf_eval as pe
    import probe_rmsf_misato as pm
    from rmsf_common import per_protein_spearman
    g = np.random.default_rng(0)
    M, d = 16, 8
    cb = torch.tensor(g.standard_normal((M, d)), dtype=torch.float32)
    w = g.standard_normal(d).astype(np.float32)

    def make(n):
        t = g.integers(0, M, n)
        return torch.tensor(t), torch.tensor(cb.numpy()[t] @ w + 0.3 * g.standard_normal(n),
                                             dtype=torch.float32)

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding.from_pretrained(cb, freeze=True)
            self.mlp = torch.nn.Linear(d, 1)
        def forward(self, t): return self.mlp(self.emb(t)).squeeze(-1)

    (ttr, ytr), (tva, yva), (tte, yte) = make(200), make(120), make(120)
    gva, gte = np.repeat(np.arange(8), 15), np.repeat(np.arange(8), 15)
    torch.manual_seed(0)
    res = pe.fit_probe_early_stopping(Probe(), ttr, ytr, tva, yva, tte, yte, device="cpu", seed=0,
                                      batch_size=32, max_epochs=5, patience=2,
                                      groups_va=gva, groups_te=gte, return_preds=True)
    pr = res.pop("_preds")
    assert res["val_spearman_per_protein_mean"] == pytest.approx(
        per_protein_spearman(pr["val"], yva.numpy(), gva)["mean"])
    json.dumps(res)                                              # no arrays left
    torch.manual_seed(0)
    res2 = pm.train_and_eval(Probe(), ttr.numpy(), ytr.numpy(), tva.numpy(), yva.numpy(),
                             tte.numpy(), yte.numpy(), "cpu", max_epochs=5, patience=2,
                             dtype_x=torch.long, groups_va=gva, groups_te=gte, return_preds=True)
    pr2 = res2.pop("_preds")
    assert res2["test_spearman_per_protein_mean"] == pytest.approx(
        per_protein_spearman(pr2["test"], yte.numpy(), gte)["mean"])
    assert "_preds" not in pe.fit_probe_early_stopping(
        Probe(), ttr, ytr, tva, yva, tte, yte, device="cpu", seed=0, max_epochs=1)


def _items(n, L, D, n_cls, seed):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = np.zeros(n_cls, np.float32); y[i % n_cls] = 1
        out.append({"pid": f"P{seed}_{i}", "x": g.standard_normal((L + i % 3, D)).astype(np.float32),
                    "y": y})
    return out


def test_function_heads_return_predictions_in_item_order():
    import ec_classify_conv1d_misato as E
    import tokenize_misato_and_eval as T
    import conv1d_binding_baselines as CB
    import tokenize_misato_binding_eval as BE
    from multilabel_metrics import fmax
    tr, va, te = _items(8, 6, 3, 2, 0), _items(4, 6, 3, 2, 1), _items(5, 6, 3, 2, 2)
    r = E.train_classifier(tr, va, te, 2, 3, "cpu", epochs=2, batch_size=4, return_preds=True)
    pr = r.pop("_preds")
    assert pr["test"]["ids"] == [it["pid"] for it in te]
    assert r["fmax"] == pytest.approx(fmax(pr["test"]["pred"], pr["test"]["label"])[0])
    cb = torch.randn(5, 3)
    tok = lambda items: [{**it, "x": np.arange(len(it["x"])) % 5} for it in items]
    r = T.train_one(tok(tr), tok(va), tok(te), cb, 2, "cpu", epochs=2, batch_size=4,
                    return_preds=True)
    assert r["val"]["fmax"] == pytest.approx(fmax(r["_preds"]["val"]["pred"],
                                                  r["_preds"]["val"]["label"])[0])
    bs = lambda items: [{**it, "y": (np.arange(len(it["x"])) % 2).astype(np.float32)}
                        for it in items]
    for rb in (CB.train_binding_cont(bs(tr), bs(va), bs(te), 3, "cpu", epochs=2, batch_size=4,
                                     return_preds=True),
               BE.train_binding(tok(bs(tr)), tok(bs(va)), tok(bs(te)), cb, "cpu", epochs=2,
                                batch_size=4, return_preds=True)):
        p = rb.pop("_preds")
        blk = PP.residue_split_from_items(p["test"]["ids"], p["test"]["pred"], p["test"]["label"])
        vals = S.binding_per_protein_values(blk)
        assert rb["auroc_per_protein_mean"] == pytest.approx(np.nanmean(vals))
        assert [len(x) for x in p["test"]["pred"]] == [len(it["x"]) for it in te]
    af = lambda items: [{**it, "y": float(i),
                         "lig": (np.arange(167) % (i + 2) == 0).astype(np.float32)}
                        for i, it in enumerate(items)]
    from scipy.stats import spearmanr
    for ra in (CB.train_affinity_cont(af(tr), af(va), af(te), 3, "cpu", epochs=2,
                                      ligand_dim=167, return_preds=True),
               CB.train_ligand_only_affinity(af(tr), af(va), af(te), "cpu", epochs=2,
                                             return_preds=True),
               BE.train_affinity(tok(af(tr)), tok(af(va)), tok(af(te)), cb, "cpu", epochs=2,
                                 ligand_dim=167, return_preds=True)):
        p = ra.pop("_preds")
        assert ra["val"]["spearman"] == pytest.approx(
            spearmanr(p["val"]["pred"], p["val"]["label"]).correlation)
        json.dumps(ra)


# ---------------------------------------------------------------- statistics
def test_per_protein_rules_match_the_probes():
    from rmsf_common import per_protein_spearman
    g = np.random.default_rng(1)
    ids, P, Y = _residue_block(g, 7, (5, 25))
    P[2] = np.ones_like(P[2])                                   # constant prediction -> 0
    Y[3] = np.ones_like(Y[3])                                   # constant label -> skipped
    blk = PP.residue_split_from_items(ids, P, Y)
    v = S.rmsf_per_protein_values(blk)
    ref = per_protein_spearman(np.concatenate(P), np.concatenate(Y),
                               np.repeat(np.arange(7), [len(y) for y in Y]))
    assert np.nanmean(v) == pytest.approx(ref["mean"]) and np.sum(~np.isnan(v)) == ref["n_scored"]
    Yb = [(g.random(len(y)) < 0.3).astype(np.float32) for y in Y]
    Pb = [g.random(len(y)) for y in Y]
    m = FC.binding_metrics(Pb, Yb)
    vb = S.binding_per_protein_values(PP.residue_split_from_items(ids, Pb, Yb))
    assert np.nanmean(vb) == pytest.approx(m["auroc_per_protein_mean"])


def test_bootstrap_null_and_shift():
    g = np.random.default_rng(2)
    n = 60
    clusters = np.repeat(np.arange(20), 3)
    base = g.standard_normal(n)
    same = S.paired_cluster_bootstrap(lambda idx: float(np.mean(base[idx] - base[idx])),
                                      clusters, B=200, seed=0)
    assert same["delta"] == 0 and same["p"] == 1.0
    d = 0.5 + 0.1 * g.standard_normal(n)
    r = S.paired_cluster_bootstrap(lambda idx: float(np.mean(d[idx])), clusters, B=500, seed=0)
    assert r["ci_low"] < r["delta"] < r["ci_high"] and r["ci_low"] > 0.3
    assert r["p"] < 0.01 and r["n_clusters"] == 20 and r["n_items"] == n
    r2 = S.paired_cluster_bootstrap(lambda idx: float(np.mean(d[idx])), clusters, B=500, seed=0)
    assert r2 == r                                              # fixed seed -> reproducible
    members = S.cluster_members(np.array(["b", "a", "b"]))
    assert [m.tolist() for m in members] == [[1], [0, 2]]


def test_fold_paired_mean():
    g = np.random.default_rng(3)
    ds = [0.1 + 0.05 * g.standard_normal(30), 0.3 + 0.05 * g.standard_normal(40)]
    stats = [lambda idx, d=d: float(np.mean(d[idx])) for d in ds]
    cls = [np.repeat(np.arange(10), 3), np.repeat(np.arange(8), 5)]
    r = S.fold_paired_bootstrap(stats, cls, B=300, seed=1)
    assert r["fold_mean"]["delta"] == pytest.approx(np.mean([d.mean() for d in ds]))
    assert [f["n_clusters"] for f in r["per_fold"]] == [10, 8]
    assert r["fold_mean"]["ci_low"] < r["fold_mean"]["delta"] < r["fold_mean"]["ci_high"]


def test_holm():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    # A.9 fix: NaN p now raises RefusedError instead of being silently dropped
    with pytest.raises(S.RefusedError, match='NaN p-value'):
        S.holm([0.02, float("nan"), 0.5])


def test_seed_average_and_comparison(tmp_path):
    g = np.random.default_rng(4)
    ids, P, Y = _residue_block(g, 12, (12, 20))
    better = [y + 0.2 * g.standard_normal(len(y)) for y in Y]
    worse = [y + 2.0 * g.standard_normal(len(y)) for y in Y]
    a = [S.load_run(_write_rmsf_like(tmp_path, f"a{s}", ids,
                                     [p + 0.01 * s for p in better], Y, seed=s)) for s in (0, 1)]
    b = [S.load_run(_write_rmsf_like(tmp_path, f"b{s}", ids, worse, Y, seed=s)) for s in (0, 1)]
    avg = S.seed_average(a, "val")
    np.testing.assert_allclose(avg["pred"], np.concatenate(better) + 0.005, rtol=1e-5, atol=1e-5)
    with pytest.raises(S.RefusedError):
        S.seed_average([a[0], a[0]], "val")                    # repeated seed
    clusters = {i: f"c{k // 3}" for k, i in enumerate(ids)}
    r = S.compare("rmsf", "spearman_per_protein", a, b, split="val", B=300, seed=0,
                  clusters=clusters)
    assert r["delta"] > 0 and r["n_clusters"] == 4 and r["seeds_a"] == [0, 1]
    r2 = S.compare("rmsf", "spearman_pooled", a, b, split="val", B=50, seed=0, clusters=clusters)
    assert r2["delta"] > 0
    # other labels for method B -> refused
    Y2 = [y + 1 for y in Y]
    c = [S.load_run(_write_rmsf_like(tmp_path, "c0", ids, worse, Y2, seed=0))]
    with pytest.raises(S.RefusedError):
        S.compare("rmsf", "spearman_per_protein", a, c, split="val", B=10, clusters=clusters)


def test_family_refuses_results_without_sidecar(tmp_path):
    g = np.random.default_rng(5)
    ids, P, Y = _residue_block(g, 4)
    ok = _write_rmsf_like(tmp_path, "ok", ids, P, Y)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"seed": 0, "test_spearman": 0.5}))   # a table-only result
    spec = {"family": "t", "comparisons": [{"id": "x", "task": "rmsf",
                                            "metric": "spearman_per_protein",
                                            "a": [str(ok)], "b": [str(old)]}]}
    with pytest.raises(S.RefusedError, match="no per-protein"):
        S.run_family(spec, split="val")
    assert S.check_family({"comparisons": [{"id": "y", "task": "GO", "metric": "fmax",
                                            "a": ["f"], "b": ["g"]}]})
    assert S.check_family({"comparisons": [{"id": "z", "task": "rmsf", "metric": "fmax",
                                            "a": ["f"], "b": ["g"]}]})
    assert S.check_family(S.example_family()) == []


def test_multilabel_metrics_and_fmax_stat():
    from multilabel_metrics import fmax
    g = np.random.default_rng(6)
    Y = (g.random((30, 5)) < 0.3).astype(np.uint8); Y[:, 0] = 1
    a = {"ids": np.array([f"p{i}" for i in range(30)]), "pred": g.random((30, 5)), "label": Y}
    b = {**a, "pred": g.random((30, 5))}
    stat, point, n, _ = S.make_delta_stat("GO", "fmax", a, b)
    assert point == pytest.approx(fmax(a["pred"], Y)[0] - fmax(b["pred"], Y)[0])
    stat, point, n, vals = S.make_delta_stat("GO", "protein_ap", a, b)
    from sklearn.metrics import average_precision_score
    assert vals["a"][3] == pytest.approx(average_precision_score(Y[3], a["pred"][3]))
    assert n == 30


# ---------------------------------------------------------------- parallel == serial
def test_parallel_identical_to_serial_synthetic(tmp_path):
    """FIXED(numbers2, 2026-10-26): serial == parallel proof on a synthetic family.

    Two comparisons (pooled and per-protein Spearman) with 3 seeds, B = 200;
    clusters given in the spec (bypasses the real manifest).  run_family with
    workers=1 and workers=2 must produce bit-identical JSON."""
    g = np.random.default_rng(7)
    ids = [f"p{i:02d}" for i in range(8)]
    Y = [g.standard_normal(int(g.integers(12, 25))).astype(np.float32) for _ in ids]
    clusters = {i: f"c{k // 2}" for k, i in enumerate(ids)}
    cluster_file = tmp_path / "clusters.json"
    cluster_file.write_text(json.dumps(clusters))
    for s in range(3):
        better = [y + 0.2 * g.standard_normal(len(y)).astype(np.float32) for y in Y]
        worse = [y + 1.0 * g.standard_normal(len(y)).astype(np.float32) for y in Y]
        _write_rmsf_like(tmp_path, f"a_seed{s}", ids, better, Y, seed=s)
        _write_rmsf_like(tmp_path, f"b_seed{s}", ids, worse, Y, seed=s)
    spec = {
        "family": "test_parallel",
        "B": 200, "seed": 42,
        "clusters": {"path": str(cluster_file)},
        "comparisons": [
            {"id": "pooled", "task": "rmsf", "metric": "spearman_pooled",
             "a": [str(tmp_path / "a_seed*.json")],
             "b": [str(tmp_path / "b_seed*.json")]},
            {"id": "protein", "task": "rmsf", "metric": "spearman_per_protein",
             "a": [str(tmp_path / "a_seed*.json")],
             "b": [str(tmp_path / "b_seed*.json")]},
        ],
    }
    serial = S.run_family(spec, split="val", workers=1)
    parallel = S.run_family(spec, split="val", workers=2)
    assert json.dumps(S._jsonable(serial), sort_keys=True) == \
           json.dumps(S._jsonable(parallel), sort_keys=True)


def test_parallel_identical_to_serial_real_family():
    """FIXED(numbers2, 2026-10-26): serial == parallel proof on a real family at small B.

    Runs the rv_test_rmsf_lam10 family at B = 50 with workers=1 and workers=2.
    Skipped if the data files do not exist (CI / test environments)."""
    spec_path = S.ROOT / "camera_ready" / "rv_test_rmsf_lam10_family.json"
    if not spec_path.exists():
        pytest.skip("real family spec not found")
    spec = json.loads(spec_path.read_text())
    spec["B"] = 50
    try:
        serial = S.run_family(spec, split="test", workers=1)
    except S.RefusedError:
        pytest.skip("real family data files not available")
    parallel = S.run_family(spec, split="test", workers=4)
    assert json.dumps(S._jsonable(serial), sort_keys=True) == \
           json.dumps(S._jsonable(parallel), sort_keys=True)
