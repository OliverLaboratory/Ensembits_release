"""Tests of the MISATO function-task pre-run fixes (camera_ready/function_reaudit.md, 2026-10-19).

What it does
------------
  1. aa baseline from misato_real_bb (`function_common.misato_aa_onehot`): one-hot, refuses a
     missing file or an old parser;
  2. same items and classes for every method: EC classes from all labelled training
     complexes, a missing feature is an error, sorted item order, item / class hashes
     (`run_ec_depth`, `go_items`); the compiler refuses a cell whose methods disagree;
  3. binding / affinity items: a length mismatch is an error, drops counted, every affinity
     mode (protein + ligand, protein only, ligand only) scores the same complexes; affinity
     label / MACCS records; the ported builders' rules;
  4. D5 controls in the runners (dry run) and the ligand-only batch size;
  5. validation metrics in every result block; `--split-metric val` compiles from them and
     never sees a test field; MSE ranks ascending; floor columns are not ranked;
     EC columns of the main tables = mAP (2026-10-19), ranked on mAP in both modes; the
     "Metrics: EC" tables start with mAP, Fmax and end with the unranked majority top-1;
  6. Ensembits padding: pad positions are zero vectors, so the token heads equal the
     baselines' continuous heads on codebook-lookup features;
  7. Vote_3Di of the RMSF probes = mini3di_baselines.vote_from_frame_tokens.

Paper
-----
not described in the paper (infrastructure only)

How to run
----------
  python -m pytest tests/test_function_prerun_fixes.py -q      # from the codebase root (CPU)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = CODE_ROOT / "downstream"
for _p in (CODE_ROOT, DOWNSTREAM, DOWNSTREAM / "baselines", CODE_ROOT / "data_prep"):
    sys.path.insert(0, str(_p))

import function_common as FC  # noqa: E402


# ---------------------------------------------------------------- 1. aa from misato_real_bb
def _real_bb(d: Path, pid: str, seq: str, parser=None):
    from ensembits.data_versions import MISATO_PARSER_VERSION
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{pid}.npz", seq=np.array(seq), parser_version=np.array(parser or MISATO_PARSER_VERSION))


def test_aa_onehot_from_real_bb(tmp_path):
    d = tmp_path / "misato_real_bb"
    _real_bb(d, "AAAA", "ACH")
    _real_bb(d, "BBBB", "WZ")                        # Z is not a standard letter -> X channel
    feats, rec = FC.misato_aa_onehot(["aaaa", "BBBB"], real_bb_dir=d)
    assert sorted(feats) == ["AAAA", "BBBB"]
    x = feats["AAAA"]
    assert x.shape == (3, 21) and x.dtype == np.float32 and (x.sum(1) == 1).all()
    assert [FC.AA_VOCAB[i] for i in x.argmax(1)] == ["A", "C", "H"]
    assert FC.AA_VOCAB[feats["BBBB"].argmax(1)[1]] == "X"
    assert rec["kind"] == "misato_real_bb_seq" and rec["verified"] and rec["n_pids"] == 2
    with pytest.raises(FC.FunctionInputError):
        FC.misato_aa_onehot(["AAAA", "CCCC"], real_bb_dir=d)      # no file
    _real_bb(d, "OLD1", "AC", parser="old")
    with pytest.raises(FC.FunctionInputError):
        FC.misato_aa_onehot(["OLD1"], real_bb_dir=d)              # old parser


def test_function_probes_no_longer_read_pairs_pickle_for_aa():
    src = (DOWNSTREAM / "baselines" / "ec_classify_conv1d_misato.py").read_text()
    body = src[src.index("def load_features(spec: str, pids=None"):src.index("def _aa_lookup")]
    assert "misato_aa_onehot" in body and "PAIRS" not in body
    for rel in ("baselines/conv1d_GO_misato.py", "baselines/conv1d_binding_baselines.py",
                "baselines/ec_classify_conv1d_misato.py"):
        assert "load_features_checked(" in (DOWNSTREAM / rel).read_text(), rel


# ---------------------------------------------------------------- 2. items and classes
def _ec_setup():
    split = {"P1": "train", "P2": "train", "P3": "train", "P4": "val", "P5": "test", "P6": "test"}
    labels = {"P1": {"1.1"}, "P2": {"2.1"}, "P3": {"3.4"}, "P4": {"1.1"}, "P5": {"2.1", "9.9"},
              "P6": {"9.9"}}                        # P6: no training class -> dropped
    feats = {p: np.zeros((5, 3), np.float32) for p in split}
    return split, labels, feats


def _fake_train(log):
    def fn(tr, va, te, n_cls):
        log.append(([it["pid"] for it in tr], [it["pid"] for it in va], [it["pid"] for it in te], n_cls))
        return {"top1_hit": 0.0, "n_test": len(te), "n_classes": n_cls, "epochs_run": 1,
                "val": {"top1_hit": 0.0}}
    return fn


def test_ec_items_classes_and_order():
    split, labels, feats = _ec_setup()
    log = []
    r = FC.run_ec_depth(2, labels, split, feats, "m", _fake_train(log))
    assert r["ec_classes"] == ["1.1", "2.1", "3.4"] and r["n_items"] == {"train": 3, "val": 1, "test": 1}
    assert r["dropped_no_train_class"] == {"val": 0, "test": 1}
    assert log[0][:3] == (["P1", "P2", "P3"], ["P4"], ["P5"])
    # classes come from every labelled training complex: a method without P3 is an error,
    # not a method with fewer classes
    with pytest.raises(FC.FunctionInputError):
        FC.run_ec_depth(2, labels, split, {k: v for k, v in feats.items() if k != "P3"}, "m2",
                        _fake_train([]))
    # same result whatever the dict order (old Ensembits code iterated a set)
    rev = dict(reversed(list(feats.items())))
    log2 = []
    r2 = FC.run_ec_depth(2, dict(reversed(list(labels.items()))), split, rev, "m", _fake_train(log2))
    assert r2["items_sha256"] == r["items_sha256"] and r2["classes_sha256"] == r["classes_sha256"]
    assert log2[0][:3] == log[0][:3]
    assert "val" in r and "majority_top1_hit" in r["val"]


def test_go_items_require_features_and_sort():
    split = {"A": "train", "B": "val", "C": "test", "D": "train"}
    labels = {"D": {"GO:2"}, "A": {"GO:1"}, "B": {"GO:1"}, "C": {"GO:2"}}
    feats = {p: np.zeros((4, 2), np.float32) for p in "ABCD"}
    b = FC.go_items(labels, ["GO:1", "GO:2"], split, feats, "m")
    assert [it["pid"] for it in b["train"]] == ["A", "D"]
    with pytest.raises(FC.FunctionInputError):
        FC.go_items(labels, ["GO:1", "GO:2"], split, {k: feats[k] for k in "ABC"}, "m")


def test_items_record_and_signature():
    b = {"train": [{"pid": "B"}, {"pid": "A"}], "val": [{"pid": "C"}], "test": []}
    r = FC.items_record(b)
    assert r["n_items"] == {"train": 2, "val": 1, "test": 0}
    assert r["items_pids_sha256"]["train"] == FC.pid_list_sha256(["A", "B"])
    r2 = FC.items_record({"train": [{"pid": "A"}, {"pid": "B"}], "val": [{"pid": "C"}], "test": []})
    assert r2["items_sha256"] == r["items_sha256"]
    r3 = FC.items_record({"train": [{"pid": "A"}], "val": [{"pid": "B"}, {"pid": "C"}], "test": []})
    assert r3["items_sha256"] != r["items_sha256"]            # same pids, other split
    js = {"by_depth": {"1": {"items_sha256": "x", "classes_sha256": "y"}}}
    assert FC.cell_signature(js, "EC", depth=1) == ("y", "x")
    assert FC.cell_signature({"results": {"structure_affinity": {"items_sha256": "z"}}},
                             "affinity", "structure") == (None, "z")
    assert FC.cell_signature({}, "GO") is None


# ---------------------------------------------------------------- 3. binding / affinity items
def test_binding_items_length_error_and_drop_counts():
    split = {"train": {"A", "S"}, "val": {"B"}, "test": {"C", "N"}}
    lab = {"A": np.zeros(12), "B": np.zeros(10), "C": np.zeros(11), "S": np.zeros(9)}
    feats = {"A": np.zeros((12, 2)), "B": np.zeros((10, 2)), "C": np.zeros((11, 2)), "S": np.zeros((9, 2))}
    b, dropped = FC.binding_site_items(split, lab, feats, "m")
    assert [len(b[s]) for s in ("train", "val", "test")] == [1, 1, 1]
    assert dropped["fewer_than_10_residues"] == {"train": 1, "val": 0, "test": 0}
    assert dropped["no_binding_label"] == {"train": 0, "val": 0, "test": 1}
    bad = dict(feats, C=np.zeros((10, 2)))                    # old code cut to min(L)
    with pytest.raises(FC.FunctionInputError, match="length"):
        FC.binding_site_items(split, lab, bad, "m")
    with pytest.raises(FC.FunctionInputError):
        FC.binding_site_items(split, lab, {k: v for k, v in feats.items() if k != "B"}, "m")


def test_affinity_items_same_for_every_mode():
    split = {"train": {"A", "B", "M"}, "val": {"C", "N"}, "test": {"D", "S"}}
    aff = {"A": 5.0, "B": 6.0, "C": 7.0, "D": 4.0, "M": 5.5, "N": float("nan"), "S": 3.0}
    maccs = {p: np.ones(167, np.float32) for p in "ABCDNS"}         # M has no MACCS
    n_res = {"A": 20, "B": 20, "C": 20, "D": 20, "M": 20, "N": 20, "S": 5}
    feats = {p: np.zeros((n_res[p], 3)) for p in n_res}
    lig, d1 = FC.affinity_items(split, aff, maccs, n_res, feats, "m", with_ligand=True)
    prot, d2 = FC.affinity_items(split, aff, maccs, n_res, feats, "m", with_ligand=False)
    only, d3 = FC.affinity_items(split, aff, maccs, n_res, None, "ligand_only", with_ligand=True)
    assert FC.items_record(lig) == FC.items_record(prot) == FC.items_record(only)
    assert [it["pid"] for it in lig["train"]] == ["A", "B"]
    assert "lig" in lig["train"][0] and "lig" not in prot["train"][0] and "x" not in only["train"][0]
    assert d1 == d2 == d3
    assert d1["no_maccs"]["train"] == 1 and d1["non_finite_label"]["val"] == 1
    assert d1["fewer_than_10_residues"]["test"] == 1


def _sidecar(path, schema, **extra):
    from ensembits.data_versions import write_provenance
    from ensembits.featurize import file_sha256
    write_provenance(path, {"schema": schema, "output_sha256": file_sha256(path), **extra})


def test_affinity_label_and_maccs_records(tmp_path, monkeypatch):
    a = tmp_path / "misato_affinity.json"; a.write_text(json.dumps({"abcd": 6.5}))
    m = tmp_path / "maccs.npz"; np.savez(m, ABCD=np.ones(167, np.uint8))
    with pytest.raises(SystemExit):
        FC.load_affinity_labels(a)                                 # no sidecar
    _sidecar(a, FC.AFFINITY_SCHEMA); _sidecar(m, FC.MACCS_SCHEMA)
    lab, rec = FC.load_affinity_labels(a)
    assert lab == {"ABCD": 6.5} and rec["schema"] == FC.AFFINITY_SCHEMA and len(rec["sha256"]) == 64
    mac, rec2 = FC.load_ligand_maccs(m)
    assert mac["ABCD"].dtype == np.float32
    cur = FC.affinity_inputs_record(a, m)
    monkeypatch.setattr(FC, "affinity_inputs_record", lambda *x, **k: cur)
    js = {**FC.protocol_record([{"file": "f", "verified": True}], input_std=FC.InputStandardisation.lookup("t", 2, [FC.standardize_lookup_table(np.eye(2))[1]]).run_record()),
          "affinity_labels": rec, "ligand_maccs": rec2}
    assert FC.function_result_problems(js, "affinity") == []
    assert FC.function_result_problems({**js, "ligand_maccs": {"sha256": "0" * 64}}, "affinity")
    assert FC.function_result_problems({**js, "function_protocol": None})
    assert FC.function_result_problems({**js, "cache_verified": False})
    a.write_text(json.dumps({"abcd": 7.0}))                       # changed after its sidecar
    with pytest.raises(SystemExit):
        FC.load_affinity_labels(a)


def test_ported_affinity_builder_rules():
    import build_misato_affinity_labels as BA
    recs = [{"protein": {"ID": "A", "neglog_aff": 6.0}, "residue": {"residue_number": list(range(12)), "binding_site": [0] * 12}},
            {"protein": {"ID": "S", "neglog_aff": 5.0}, "residue": {"residue_number": list(range(9)), "binding_site": [0] * 9}},
            {"protein": {"ID": "M", "neglog_aff": 5.0}, "residue": {"residue_number": list(range(12)), "binding_site": [0] * 11}},
            {"protein": {"ID": "N", "neglog_aff": float("nan")}, "residue": {"residue_number": list(range(12)), "binding_site": [0] * 12}},
            {"protein": {"ID": "B", "neglog_aff": 7}, "residue": {"residue_number": list(range(12)), "binding_site": [1] * 12}}]
    aff, n = BA.affinity_from_records(recs)
    assert list(aff) == ["A", "B"] and aff["B"] == 7.0 and isinstance(aff["B"], float)
    assert n == {"records": 5, "short": 1, "binding_site_length_mismatch": 1, "no_affinity": 1, "written": 2}
    assert BA.encode(aff) == b'{"A": 6.0, "B": 7.0}'


def test_ported_maccs_builder_rules():
    import build_misato_ligand_maccs as BM
    recs = [{"protein": {"ID": "A", "ligand_id": " atp "}}, {"protein": {"ID": "B", "ligand_id": None}},
            {"protein": {"ID": "C", "ligand_id": "SO4"}}, {"protein": {"ID": "D", "ligand_id": "ABCD"}},
            {"protein": {"ID": "E", "ligand_id": "ATP"}}]
    codes, n = BM.ligand_codes(recs)
    assert codes == {"A": "ATP", "E": "ATP"}
    assert n["no_ligand"] == 1 and n["skipped_artifact"] == 2 and n["unique_ccds"] == 1
    arrays = {"A": np.zeros(167, np.uint8), "B": np.ones(167, np.uint8)}
    assert BM.encode_npz(arrays) == BM.encode_npz(dict(arrays))   # reproducible bytes
    pytest.importorskip("rdkit")
    fp, n2, no_smi, bad = BM.fingerprints({"A": "ETH", "B": "XXX", "C": "BAD"},
                                          {"ETH": "CCO", "XXX": None, "BAD": "C1CC"})
    assert list(fp) == ["A"] and fp["A"].shape == (167,) and no_smi == ["XXX"] and bad == ["BAD"]


# ---------------------------------------------------------------- 4. D5 controls
def test_ligand_only_batch_size_and_control_jobs():
    import inspect
    import conv1d_binding_baselines as CB
    assert inspect.signature(CB.train_ligand_only_affinity).parameters["batch_size"].default == 32
    assert inspect.signature(CB.train_affinity_cont).parameters["batch_size"].default == 32
    import run_misato_baselines as RB
    J = RB.function_jobs(["esm3struct"], ["structure"], [0, 1], RB.RECIPE, controls=True)
    ctrl = [j for j in J if j[0] == "AFFCTRL"]
    assert {j[1] for j in ctrl} == {"ligand_only", "esm3struct", "aa"} and len(ctrl) == 6
    for t, F, p, sp, s, o, c in ctrl:
        assert "--use-ligand-maccs" not in c and c[c.index("--task") + 1] == "affinity"
        assert all(x in c for x in RB.RECIPE) and "_lig_" not in o.name
    sys.path.insert(0, str(DOWNSTREAM))
    import _run_model_battery as MB
    JM = MB.function_jobs("runX", MB.RECIPE)
    cm = [j for j in JM if j[0] == "AFFCTRL"]
    assert len(cm) == 2 * len(MB.SEEDS) and all("--use-ligand-maccs" not in j[6] for j in cm)
    assert all("_lig_" not in j[5].name for j in cm)


def test_seed_rescue_reads_control_cells(tmp_path):
    import seed_rescue as SR
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"results": {"structure_affinity": {"spearman": 0.3, "epochs_run": 12}}}))
    assert SR.read_cells("AFFCTRL", f) == [("structure_affinity", 0.3, 12)]


# ---------------------------------------------------------------- 5. validation metrics
def _items(n, L, D, n_cls, seed):
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = np.zeros(n_cls, np.float32); y[i % n_cls] = 1
        out.append({"pid": f"P{i}", "x": g.standard_normal((L + i % 3, D)).astype(np.float32), "y": y})
    return out


def test_val_block_in_every_head():
    import ec_classify_conv1d_misato as E
    tr, va, te = _items(8, 6, 3, 2, 0), _items(4, 6, 3, 2, 1), _items(4, 6, 3, 2, 2)
    r = E.train_classifier(tr, va, te, 2, 3, "cpu", epochs=2, batch_size=4)
    for k in FC.EC_METRICS:
        assert k in r and k in r["val"]
    import tokenize_misato_and_eval as T
    cb = torch.randn(5, 3)
    tok = lambda items: [{**it, "x": np.arange(len(it["x"])) % 5} for it in items]
    r = T.train_one(tok(tr), tok(va), tok(te), cb, 2, "cpu", epochs=2, batch_size=4)
    for k in FC.EC_METRICS:
        assert k in r["val"]
    import conv1d_binding_baselines as CB
    import tokenize_misato_binding_eval as BE
    bs = lambda items: [{**it, "y": (np.arange(len(it["x"])) % 2).astype(np.float32)} for it in items]
    rb = CB.train_binding_cont(bs(tr), bs(va), bs(te), 3, "cpu", epochs=2, batch_size=4)
    rt = BE.train_binding(tok(bs(tr)), tok(bs(va)), tok(bs(te)), cb, "cpu", epochs=2, batch_size=4)
    for r in (rb, rt):
        for k in FC.BINDING_METRICS:
            assert k in r and k in r["val"]
    af = lambda items, lig: [{**it, "y": float(i), **({"lig": np.ones(167, np.float32)} if lig else {})}
                             for i, it in enumerate(items)]
    for lig in (True, False):
        ra = CB.train_affinity_cont(af(tr, lig), af(va, lig), af(te, lig), 3, "cpu", epochs=2,
                                    ligand_dim=167 if lig else 0)
        rt = BE.train_affinity(tok(af(tr, lig)), tok(af(va, lig)), tok(af(te, lig)), cb, "cpu",
                               epochs=2, ligand_dim=167 if lig else 0)
        for r in (ra, rt):
            for k in FC.AFFINITY_METRICS:
                assert k in r and k in r["val"]
    rl = CB.train_ligand_only_affinity(af(tr, True), af(va, True), af(te, True), "cpu", epochs=2)
    assert set(FC.AFFINITY_METRICS) <= set(rl["val"])


def test_binding_metrics_per_protein():
    m = FC.binding_metrics([np.array([0.1, 0.9, 0.2]), np.array([0.5, 0.5])],
                           [np.array([0, 1, 0]), np.array([1, 1])])
    assert m["auroc_per_protein_mean"] == 1.0 and m["n_proteins_auroc"] == 1 and m["n_residues"] == 5


def _fn_result(test_v, val_v, items="i", classes="c", **kw):
    blk = {k: test_v for k in FC.EC_METRICS}
    return {**blk, "val": {k: val_v for k in FC.EC_METRICS}, "items_sha256": items * 64,
            "classes_sha256": classes * 64, **kw}


def test_split_metric_val_compile_and_refusals(tmp_path, monkeypatch):
    import ensembits.paths as P
    import _compile_rebuttal_results as CR
    from ensembits.data_versions import misato_splits_record
    import ec_labels as EL
    sp = tmp_path / "misato_splits_v2.json"
    sp.write_text(json.dumps({"stats": {"schema": "misato-splits-v2"}}))
    monkeypatch.setattr(P, "MISATO_SPLITS", sp)
    ecl = tmp_path / "misato_ec_v2.json"; ecl.write_text(json.dumps({"A": ["1.1.1.1"]}))
    monkeypatch.setattr(EL, "EC_LABELS", ecl)
    ec = tmp_path / "ec"; ec.mkdir(); monkeypatch.setattr(CR, "EC", ec)
    monkeypatch.setattr(CR, "FN_CV", False)   # the fixed-split path (function CV: test_function_cv.py)
    common = {**misato_splits_record(sp), "ec_labels": EL.ec_labels_record(ecl),
              **FC.protocol_record([{"file": "f", "verified": True}], input_std=FC.InputStandardisation.lookup("t", 2, [FC.standardize_lookup_table(np.eye(2))[1]]).run_record())}
    def write(name, d1, **top):
        for s in range(10):
            (ec / name(s)).write_text(json.dumps({"by_depth": {"1": d1}, **common, **top}))
    write(lambda s: f"tok_runX_structure_{CR.TAG}{CR._sd_ec(s)}__summary.json", _fn_result(0.999, 0.5))
    write(lambda s: f"aa__structure__{CR.TAG}{CR._sd_bl(s)}__summary.json", _fn_result(0.999, 0.25))
    write(lambda s: f"esm3struct__structure__{CR.TAG}{CR._sd_bl(s)}__summary.json",
          _fn_result(0.999, 0.75, items="j"))                  # other items
    write(lambda s: f"protoken__structure__{CR.TAG}{CR._sd_bl(s)}__summary.json",
          _fn_result(0.999, 0.6, classes="d"))                 # other classes
    write(lambda s: f"random__structure__{CR.TAG}{CR._sd_bl(s)}__summary.json",
          _fn_result(0.999, 0.1), function_protocol="old")    # old protocol
    CR.set_rmsf_protocol("test"); CR.set_split_metric("val")
    try:
        for x in (CR._SIG, CR._REF_SIG): x.clear()
        CR._ITEM_REFUSED.clear(); CR._FN_REFUSED.clear()
        md = CR.split_table("structure", [("Ens", "runX")], {"Ens"}, None,
                            specs=[("EC d1", "EC", 1, "top1_hit"), ("mAP", "EC", 1, "macro_ap")])
        assert "0.999" not in md
        row = lambda lab: next(l for l in md.splitlines() if l.startswith(f"| {lab} |"))
        assert "0.500" in row("Ens · P=full") and "0.250" in row("aa")
        assert "✗" in row("esm3struct") and "✗" in row("protoken") and "✗" in row("random")
        reasons = {(r[0], r[3]) for r in CR._ITEM_REFUSED}
        assert ("baseline_esm3struct", "items / classes differ from the other methods") in reasons
        assert ("baseline_protoken", "items / classes differ from the other methods") in reasons
        assert any(r[0] == "baseline_random" for r in CR._FN_REFUSED)
        # the loaded JSON has no test field in val mode
        js = CR._load_result(ec / f"aa__structure__{CR.TAG}__summary.json")
        assert "top1_hit" not in js["by_depth"]["1"] and js["by_depth"]["1"]["val"]["top1_hit"] == 0.25
        CR.set_split_metric("test")
        for x in (CR._SIG, CR._REF_SIG): x.clear()
        md2 = CR.split_table("structure", [("Ens", "runX")], {"Ens"}, None,
                             specs=[("EC d1", "EC", 1, "top1_hit")])
        assert "0.999" in md2
    finally:
        CR.set_split_metric("test")


def test_ec_columns_show_map_and_majority_not_ranked(tmp_path, monkeypatch):
    """EC metric decision (2026-10-19): the main tables' EC d1-d3 cells are mAP (ranked on
    mAP, not top-1), in the test and the val compile; the "Metrics: EC" tables list mAP, Fmax
    first and the majority-class top-1 last, unranked (both compilers)."""
    import ensembits.paths as P
    import _compile_rebuttal_results as CR
    import compile_camera_ready_tables as T
    from ensembits.data_versions import misato_splits_record
    import ec_labels as EL
    assert [s[3] for s in CR.MAIN_SPECS if s[1] == "EC"] == ["macro_ap"] * 3
    assert all("mAP" in s[0] for s in CR.MAIN_SPECS if s[1] == "EC")
    assert not hasattr(CR, "ABL_FN_SPECS")   # A9: no function columns in the ablation tables
    for name, specs in CR.DETAIL_TABLES:
        if name.startswith("EC"):
            assert [m for _, _, _, m in specs][:2] == ["macro_ap", "fmax"]
            assert specs[-1][3] == "majority_top1_hit"
            assert {m for _, _, _, m in specs} >= set(FC.EC_METRICS)
    assert "majority" in CR.NOT_RANKED_COL and "majority" in T.NOT_RANKED_COL
    sp = tmp_path / "misato_splits_v2.json"
    sp.write_text(json.dumps({"stats": {"schema": "misato-splits-v2"}}))
    monkeypatch.setattr(P, "MISATO_SPLITS", sp)
    ecl = tmp_path / "misato_ec_v2.json"; ecl.write_text(json.dumps({"A": ["1.1.1.1"]}))
    monkeypatch.setattr(EL, "EC_LABELS", ecl)
    ec = tmp_path / "ec"; ec.mkdir(); monkeypatch.setattr(CR, "EC", ec)
    monkeypatch.setattr(CR, "FN_CV", False)   # the fixed-split path (function CV: test_function_cv.py)
    common = {**misato_splits_record(sp), "ec_labels": EL.ec_labels_record(ecl),
              **FC.protocol_record([{"file": "f", "verified": True}], input_std=FC.InputStandardisation.lookup("t", 2, [FC.standardize_lookup_table(np.eye(2))[1]]).run_record())}
    def res(top1, mapv, maj):                    # test block; val block = test values / 2
        blk = {"top1_hit": top1, "macro_ap": mapv, "micro_ap": 0.5, "macro_f1@0.5": 0.5,
               "micro_f1@0.5": 0.5, "fmax": 0.5, "majority_top1_hit": maj}
        d = {**blk, "val": {k: v / 2 for k, v in blk.items()}, "items_sha256": "i" * 64,
             "classes_sha256": "c" * 64}
        return {"by_depth": {str(k): d for k in (1, 2, 3)}, **common}
    for s in range(10):
        for pih in ("", "_P1inf"):                             # lower top-1, higher mAP
            (ec / f"tok_runX{pih}_structure_{CR.TAG}{CR._sd_ec(s)}__summary.json").write_text(
                json.dumps(res(0.6, 0.42, 0.7)))
        (ec / f"aa__structure__{CR.TAG}{CR._sd_bl(s)}__summary.json").write_text(
            json.dumps(res(0.9, 0.22, 0.7)))
    ec_main = [s for s in CR.MAIN_SPECS if s[1] == "EC"]
    ec_detail = dict(CR.DETAIL_TABLES)["EC d1"]
    row = lambda md, lab: next(l for l in md.splitlines() if l.startswith(f"| {lab} |"))
    try:
        for mode, (m_hi, m_lo, maj) in (("test", ("0.420", "0.220", "0.700")),
                                        ("val", ("0.210", "0.110", "0.350"))):
            CR.set_rmsf_protocol("test"); CR.set_split_metric(mode)
            for x in (CR._SIG, CR._REF_SIG): x.clear()
            CR._ITEM_REFUSED.clear(); CR._FN_REFUSED.clear()
            md = CR.split_table("structure", [("Ens", "runX")], {"Ens"}, None, specs=ec_main)
            assert "| model | EC d1 mAP | EC d2 mAP | EC d3 mAP |" in md
            ens, aa = row(md, "Ens · P=1"), row(md, "aa")         # single-frame block
            assert f"**{m_hi}" in ens and m_lo in aa and f"**{m_lo}" not in aa
            assert "0.600" not in ens and "0.900" not in aa         # no top-1 in the main cells
            md = CR.split_table("structure", [("Ens", "runX")], {"Ens"}, None, specs=ec_detail,
                                check=False)
            hdr = next(l for l in md.splitlines() if l.startswith("| model |"))
            assert hdr.startswith("| model | mAP | Fmax |") and hdr.endswith(f"{CR.EC_MAJORITY_COL} |")
            cells = row(md, "aa").strip("|").split("|")
            assert cells[-1].strip().startswith(maj) and "**" not in cells[-1] and "<u>" not in cells[-1]
        ranked = T.rank([["a", "0.42", "0.70"], ["b", "0.22", "0.70"]],
                        header=["model", "mAP", CR.EC_MAJORITY_COL])
        assert ranked[0][1] == "**0.42**" and ranked[0][2] == "0.70" and ranked[1][2] == "0.70"
    finally:
        CR.set_split_metric("test")


def test_strip_test_keeps_val_and_provenance():
    import _compile_rebuttal_results as CR
    js = {"results": {"s_affinity": {"spearman": 0.9, "mse": 1.0, "items_sha256": "x",
                                     "val": {"spearman": 0.4, "mse": 2.0}}},
          "test_spearman": 0.8, "best_val_spearman": 0.3, "splits_sha256": "y",
          "frequency_floor": {"fmax": 0.1}}
    out = CR._strip_test(js)
    assert out["results"]["s_affinity"] == {"items_sha256": "x", "val": {"spearman": 0.4, "mse": 2.0}}
    assert "test_spearman" not in out and out["best_val_spearman"] == 0.3 and "frequency_floor" not in out


def test_mse_ranks_ascending_and_floor_not_ranked():
    import _compile_rebuttal_results as CR
    rows = [[CR.agg([1.0]), CR.agg([0.2]), CR.agg([0.5])], [CR.agg([2.0]), CR.agg([0.4]), CR.agg([0.1])]]
    marks = CR._rank_cols(rows, ["Aff MSE", "Aff Spr", "floor top-1"])
    assert marks[0][0] == 0 and marks[1][0] == 1 and marks[2] == (None, None)
    import compile_camera_ready_tables as T
    ranked = T.rank([["a", "1.000", "0.2", "0.5"], ["b", "2.000", "0.4", "0.1"]],
                    header=["model", "Aff MSE", "Aff Spr", "floor top-1"])
    assert ranked[0][1].startswith("**") and ranked[1][2].startswith("**")
    assert not ranked[0][3].startswith("**") and not ranked[1][3].startswith("**")


# ---------------------------------------------------------------- 6. Ensembits padding
def _copy(dst, src, names):
    sd = src.state_dict()
    dst.load_state_dict({k: v for k, v in sd.items() if k.split(".")[0] in names}, strict=False)


def test_ensembits_padding_equals_zero_padded_baseline():
    torch.manual_seed(0)
    import tokenize_misato_and_eval as T
    import ec_classify_conv1d_misato as E
    import tokenize_misato_binding_eval as BE
    import conv1d_binding_baselines as CB
    cb = torch.randn(7, 4)
    cb[0] = 5.0                                   # a loud codebook[0]: must not leak into pads
    toks = [np.array([1, 2, 3, 4, 5, 6]), np.array([3, 1])]
    items = [{"x": t, "y": np.zeros(2, np.float32)} for t in toks]
    X, M, _ = T.collate(items)
    feat_items = [{"x": cb[torch.as_tensor(t)].numpy(), "y": np.zeros(2, np.float32)} for t in toks]
    Xf, Mf, _ = E.collate(feat_items)
    tm = T.TokenConv1D(cb, 2).eval(); bm = E.Conv1DClassifier(4, 2).eval(); _copy(bm, tm, {"conv", "head"})
    with torch.no_grad():
        assert torch.allclose(tm(X, M), bm(Xf, Mf), atol=1e-5)
        old = T.TokenConv1D(cb, 2, zero_pad=False).eval(); old.load_state_dict(tm.state_dict())
        assert not torch.allclose(old(X, M), bm(Xf, Mf), atol=1e-5)   # the Q23 bug
    bt = BE.BindingConv1D(cb).eval(); bc = CB.BindingConv1DCont(4).eval(); _copy(bc, bt, {"conv", "head"})
    with torch.no_grad():
        assert torch.allclose(bt(X, M), bc(Xf, Mf), atol=1e-5)
    at = BE.AffinityConv1D(cb, ligand_dim=167).eval()
    ac = CB.AffinityConv1DCont(4, ligand_dim=167).eval(); _copy(ac, at, {"conv", "head", "lig_proj"})
    lig = torch.ones(2, 167)
    with torch.no_grad():
        assert torch.allclose(at(X, M, ligand=lig), ac(Xf, Mf, ligand=lig), atol=1e-5)


# ---------------------------------------------------------------- 7. Vote_3Di
def test_rmsf_vote_is_the_anova_rule():
    from ensembits.baselines.mini3di_baselines import vote_from_frame_tokens
    import probe_rmsf_baselines as PB
    import rmsf_common as RC
    cent = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, -1.0]], np.float32)
    tk = np.array([[0, 1, 2, 3], [1, 1, 0, 2], [0, 0, 1, 1]])     # tie-only rows included
    assert PB._vote(tk, cent).tolist() == vote_from_frame_tokens(tk, cent).tolist()
    assert RC.vote_rule_problem({"feature": "vote_3di"})
    assert RC.vote_rule_problem({"feature": "vote_3di", "vote_rule": RC.VOTE_RULE}) is None
    assert RC.vote_rule_problem({"feature": "aa"}) is None
    src = (DOWNSTREAM / "probe_rmsf_misato.py").read_text()
    body = src[src.index("def build_vote_3di"):src.index("def build_pca_eig")]
    assert "vote_from_frame_tokens(" in body and "np.argmin(" not in body
    src = (DOWNSTREAM / "baselines" / "ec_classify_conv1d_misato.py").read_text()
    assert "vote_from_frame_tokens(tk, cent)" in src
