"""Tests of the MISATO EC / GO cross-validation (authors' decision option A, 2026-10-19).

What it does
------------
  1. builder rules (data_prep/build_misato_function_cv.py): protein keys (short chains left
     out, polyproteins by mature product, no accession = own key), representatives (most
     residues, ties by PDB id), label disagreement, group-level stratified folds (groups
     never split, deterministic, size cap, label balance), rotations;
  2. an end-to-end build on a synthetic chain table (GO and EC, both modes);
  3. the loader / checks of downstream/function_cv.py (split maps, class check, result
     block, refusal of another CV file, sidecar sha256), the fold aggregation;
  4. the compiler's CV cells (mean over folds per seed, then over seeds; refusal of results
     from another CV file; per-fold values) and the runners' CV jobs (dry run);
  5. the built file data/misato_function_cv_v1.json, if present: every representative in one
     fold, groups whole, no CATH H across folds in the structure mode, class lists equal to
     a recomputation, leakage summary of the sidecar.

Paper
-----
not described in the paper (infrastructure only)

How to run
----------
  python -m pytest tests/test_function_cv.py -q      # from the codebase root (CPU)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = CODE_ROOT / "downstream"
for _p in (CODE_ROOT, DOWNSTREAM, DOWNSTREAM / "baselines", CODE_ROOT / "data_prep"):
    sys.path.insert(0, str(_p))

import build_misato_function_cv as B  # noqa: E402
import function_cv as CV  # noqa: E402


def _chain(length, accs, keys=None, cluster="s1", cath=()):
    return {"length": length, "uniprot": list(accs), "uniprot_keys": keys or list(accs),
            "cluster": cluster, "cath_h": list(cath)}


# ---------------------------------------------------------------- 1. builder rules
def test_protein_key_rules():
    poly = {"P04585"}
    ch = [_chain(99, ["P04585"], [["P04585", "Protease"]]), _chain(12, ["P99999"])]
    assert B.protein_key(ch, poly, "1A30") == ("P04585|Protease",)      # peptide left out
    ch2 = [_chain(150, ["P04585"], [["P04585", "Integrase"]])]
    assert B.protein_key(ch2, poly, "3AO2") != B.protein_key(ch, poly, "1A30")
    # a multi-chain zymogen (not component-resolved) keeps the plain accession
    ch3 = [_chain(250, ["P00734"], [["P00734", "Thrombin heavy chain"]]),
           _chain(36, ["P00734"], [["P00734", "Thrombin light chain"]])]
    assert B.protein_key(ch3, poly, "1XXX") == ("P00734",)
    assert B.protein_key([_chain(80, [])], poly, "9ABC") == ("pid:9ABC",)
    assert B.protein_key([_chain(80, ["B"]), _chain(90, ["A"])], poly, "X") == ("A", "B")


def test_representatives_and_disagreement():
    keys = {"1AAA": ("P1",), "2BBB": ("P1",), "3CCC": ("P1",), "4DDD": ("P2",)}
    n_res = {"1AAA": 100, "2BBB": 120, "3CCC": 120, "4DDD": 50}
    reps = B.choose_representatives(list(keys), keys, n_res)
    assert reps == {"2BBB": ["1AAA", "2BBB", "3CCC"], "4DDD": ["4DDD"]}  # tie -> smaller id
    lab = {"1AAA": {"a"}, "2BBB": {"a", "b"}, "3CCC": {"a", "b"}, "4DDD": {"c"}}
    d = B.label_disagreement(reps, lab)
    assert d["n_proteins"] == 2 and d["n_proteins_with_differing_labels"] == 1
    assert d["n_nonrep_complexes"] == 2 and d["n_nonrep_complexes_differing_from_rep"] == 1
    assert d["n_proteins_rep_labels_not_union"] == 0


def _synthetic_groups(n_groups=60, seed=0):
    rng = np.random.RandomState(seed)
    groups, labels, k = [], {}, 0
    for g in range(n_groups):
        size = int(rng.choice([1, 1, 1, 2, 3, 8]))
        members = []
        lab = {f"L{rng.randint(4)}"} | ({"R"} if g % 10 == 0 else set())
        for _ in range(size):
            p = f"P{k:04d}"; k += 1
            members.append(p); labels[p] = set(lab)
        groups.append(members)
    return groups, labels


def test_stratified_group_folds_whole_groups_deterministic_balanced():
    groups, labels = _synthetic_groups()
    strat = sorted({l for v in labels.values() for l in v})
    f1, info = B.stratified_group_folds(groups, labels, strat, 5, 42)
    f2, _ = B.stratified_group_folds(list(reversed(groups)), labels, strat, 5, 42)
    assert f1 == f2                                              # input order irrelevant
    B.check_folds(f1, groups, list(labels))
    n = len(labels)
    assert all(len(f) <= 1.1 * n / 5 + max(len(g) for g in groups) for f in f1)
    # the rare label "R" (6 groups) is spread over the folds
    r_per_fold = [sum("R" in labels[p] for p in f) > 0 for f in f1]
    assert sum(r_per_fold) == 5
    # a group larger than a fold gets a fold of its own size
    big = [[f"G{i:03d}" for i in range(50)]] + [[f"S{i:03d}"] for i in range(40)]
    lab = {p: {"x"} for g in big for p in g}
    fb, _ = B.stratified_group_folds(big, lab, ["x"], 5, 42)
    B.check_folds(fb, big, list(lab))
    assert max(len(f) for f in fb) == 50


def test_size_cap_and_uncapped():
    groups, labels = _synthetic_groups(80, seed=3)
    strat = sorted({l for v in labels.values() for l in v})
    capped, _ = B.stratified_group_folds(groups, labels, strat, 5, 42, size_tol=0.0)
    free, _ = B.stratified_group_folds(groups, labels, strat, 5, 42, size_tol=float("inf"))
    sp = lambda fs: max(len(f) for f in fs) - min(len(f) for f in fs)
    assert sp(capped) <= sp(free)


def test_always_train_groups():
    groups = [[f"G{i:03d}" for i in range(30)], [f"H{i:03d}" for i in range(10)]] + \
             [[f"S{i:03d}"] for i in range(60)]
    big, rest = B.split_always_train(groups, 100, 5)            # cap 20
    assert big == [sorted(groups[0])] and len(rest) == 61
    folds = [[r[0] for r in rest[i::5]] for i in range(5)]
    for rot in B.rotations_of(folds, big[0]):
        assert set(big[0]) <= set(rot["train"])
        assert not set(big[0]) & (set(rot["val"]) | set(rot["test"]))


def test_rotations():
    folds = [[f"F{f}_{i}" for i in range(3)] for f in range(5)]
    rots = B.rotations_of(folds)
    for r, rot in enumerate(rots):
        assert rot["test"] == folds[r] and rot["val"] == folds[(r + 1) % 5]
        assert set(rot["train"]) == {p for f in range(5) if f not in (r, (r + 1) % 5)
                                     for p in folds[f]}
    # every fold is the test fold exactly once
    assert sorted(r["test_fold"] for r in rots) == list(range(5))


# ---------------------------------------------------------------- 2. end-to-end build
def _table(n=120, seed=1):
    table, n_res, go, ec = {}, {}, {}, {}
    for i in range(n):
        prot = i // 3                          # three complexes per protein
        fam = prot // 4                        # four proteins per family
        pid = f"{i:04d}"
        table[pid] = {"chains": [_chain(200, [f"Q{prot:05d}"], cluster=f"s{fam}",
                                        cath=[f"1.10.{fam % 7}.1"])],
                      "clusters": [f"s{fam}"], "cath_h": [f"1.10.{fam % 7}.1"]}
        n_res[pid] = 200 + (i % 3)
        terms = {f"GO:{fam % 5:07d}", "GO:0009999"} | ({f"GO:{100 + prot % 3:07d}"})
        go[pid] = sorted(terms)
        ec[pid] = [f"{1 + fam % 6}.{1 + prot % 2}.1.{1 + i % 2}"]
    return table, n_res, go, ec


def test_build_end_to_end_synthetic():
    table, n_res, go, ec = _table()
    body, rep = B.build(table, n_res, go, ec, set(), 42)
    for task in ("go", "ec"):
        tb = body["tasks"][task]
        assert tb["n_proteins"] == 40 and len(tb["representatives"]) == 40
        for r, rec in tb["representatives"].items():
            assert r == sorted(rec["members"], key=lambda p: (-n_res[p], p))[0]
        for mode in ("sequence", "structure"):
            m = tb["modes"][mode]
            B.check_folds(m["folds"], m["groups"], list(tb["representatives"]))
            for r, rot in enumerate(m["rotations"]):
                sm = {p: s for s in ("train", "val", "test") for p in rot[s]}
                assert len(sm) == 40 and rot["test_fold"] == r
                if task == "ec":
                    want = sorted({e.split(".")[0] for p in rot["train"] for e in ec[p]})
                    assert rot["classes"]["1"] == want
                else:
                    from go_labels import build_go_labelset
                    _, cls, _ = build_go_labelset({p: go[p] for p in tb["representatives"]},
                                                  rot["train"], "top50")
                    assert rot["classes"] == cls
        assert rep["tasks"][task]["label_disagreement"]["n_proteins"] == 40
    # the structure mode never puts one CATH H code in two folds
    for task in ("go", "ec"):
        folds = body["tasks"][task]["modes"]["structure"]["folds"]
        where = {}
        for f, ps in enumerate(folds):
            for p in ps:
                for h in table[p]["cath_h"]:
                    where.setdefault(h, set()).add(f)
        assert all(len(v) == 1 for v in where.values())


# ---------------------------------------------------------------- 3. loader and checks
def _write_cv(tmp_path, body, inputs=None):
    from ensembits.featurize import file_sha256
    p = tmp_path / "cv.json"
    p.write_text(json.dumps({"schema": CV.CV_SCHEMA, "inputs": inputs or {}, **body}))
    Path(str(p) + ".provenance.json").write_text(json.dumps(
        {"schema": CV.CV_SCHEMA, "output_sha256": file_sha256(p)}))
    return p


def test_loader_split_map_classes_and_refusals(tmp_path):
    table, n_res, go, ec = _table()
    body, _ = B.build(table, n_res, go, ec, set(), 42)
    p = _write_cv(tmp_path, body)
    cv, rec = CV.load_function_cv(p, check_inputs=False)
    assert rec["schema"] == CV.CV_SCHEMA and len(rec["sha256"]) == 64
    sm = CV.cv_split_map(cv, "ec", "structure", 2)
    rot = cv["tasks"]["ec"]["modes"]["structure"]["rotations"][2]
    assert {p for p, s in sm.items() if s == "test"} == set(rot["test"])
    assert rot["val_fold"] == 3
    CV.check_classes(cv, "ec", "structure", 2, rot["classes"]["1"], depth=1)
    with pytest.raises(ValueError):
        CV.check_classes(cv, "ec", "structure", 2, rot["classes"]["1"][:-1], depth=1)
    blk = CV.cv_result_block(rec, cv, "go", "sequence", 4)["function_cv"]
    assert blk["fold"] == 4 and blk["val_fold"] == 0 and blk["sha256"] == rec["sha256"]
    assert CV.cv_result_problem({"function_cv": blk}, "go", p) is None
    assert "task" in CV.cv_result_problem({"function_cv": blk}, "ec", p)
    assert "not a cross-validation run" in CV.cv_result_problem({}, None, p)
    other = {"function_cv": {**blk, "sha256": "0" * 64}}
    assert "is not the current" in CV.cv_result_problem(other, None, p)
    p.write_text(p.read_text() + " ")                       # edited after the build
    with pytest.raises(SystemExit):
        CV.load_function_cv(p, check_inputs=False)
    with pytest.raises(ValueError):
        CV.rotation(cv, "ec", "structure", 5)


def test_loader_refuses_stale_inputs(tmp_path, monkeypatch):
    table, n_res, go, ec = _table(30)
    body, _ = B.build(table, n_res, go, ec, set(), 42)
    p = _write_cv(tmp_path, body, inputs={"go_labels": {"sha256": "a" * 64}})
    monkeypatch.setattr(CV, "current_inputs", lambda: {"go_labels": "b" * 64})
    with pytest.raises(SystemExit):
        CV.load_function_cv(p)
    monkeypatch.setattr(CV, "current_inputs", lambda: {"go_labels": "a" * 64})
    CV.load_function_cv(p)


def test_aggregate_cv():
    vals = {0: {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5},          # seed mean 0.3
            1: {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},          # seed mean 0.5
            2: {0: 0.9, 1: 0.9}}                                   # incomplete: left out
    (m, s, n), per_fold = CV.aggregate_cv(vals)
    assert n == 2 and abs(m - 0.4) < 1e-12 and abs(s - np.std([0.3, 0.5], ddof=1)) < 1e-12
    assert per_fold[0] == [0.1, 0.5, 0.9] and per_fold[4] == [0.5, 0.5]
    assert CV.aggregate_cv({0: {0: 0.1}})[0] is None
    assert CV.aggregate_cv({0: {0: 0.1}}, require_all_folds=False)[0][0] == 0.1


def test_probe_cli_pairs_cv_args():
    import argparse
    ap = argparse.ArgumentParser(); CV.add_cv_args(ap)
    assert CV.check_cv_args(ap, ap.parse_args(["--cv-mode", "sequence", "--fold", "0"]))
    assert not CV.check_cv_args(ap, ap.parse_args([]))
    with pytest.raises(SystemExit):
        CV.check_cv_args(ap, ap.parse_args(["--cv-mode", "sequence"]))
    with pytest.raises(SystemExit):
        ap.parse_args(["--cv-mode", "sequence", "--fold", "5"])
    for rel in ("tokenize_misato_and_eval.py", "tokenize_misato_GO_eval.py",
                "baselines/ec_classify_conv1d_misato.py", "baselines/conv1d_GO_misato.py"):
        src = (DOWNSTREAM / rel).read_text()
        assert "add_cv_args(ap)" in src and "cv_result_block(" in src and "check_classes(" in src


# ---------------------------------------------------------------- 4. compiler and runners
def test_compiler_cv_cells(tmp_path, monkeypatch):
    import ensembits.paths as P
    import _compile_rebuttal_results as CR
    import function_common as FC
    import ec_labels as EL
    from ensembits.data_versions import misato_splits_record
    sp = tmp_path / "misato_splits_v2.json"
    sp.write_text(json.dumps({"stats": {"schema": "misato-splits-v2"}}))
    monkeypatch.setattr(P, "MISATO_SPLITS", sp)
    ecl = tmp_path / "misato_ec_v2.json"; ecl.write_text(json.dumps({"A": ["1.1.1.1"]}))
    monkeypatch.setattr(EL, "EC_LABELS", ecl)
    table, n_res, go, ec = _table(30)
    body, _ = B.build(table, n_res, go, ec, set(), 42)
    cvp = _write_cv(tmp_path, body)
    monkeypatch.setattr(P, "MISATO_FUNCTION_CV", cvp)
    cv, rec = CV.load_function_cv(cvp, check_inputs=False)
    ecd = tmp_path / "ec"; ecd.mkdir(); monkeypatch.setattr(CR, "EC", ecd)
    monkeypatch.setattr(CR, "FN_CV", True)
    common = {**misato_splits_record(sp), "ec_labels": EL.ec_labels_record(ecl),
              **FC.protocol_record([{"file": "f", "verified": True}], input_std=FC.InputStandardisation.lookup("t", 2, [FC.standardize_lookup_table(np.eye(2))[1]]).run_record())}

    def res(v, fold, sha=None):
        blk = {k: v for k in FC.EC_METRICS}
        d = {**blk, "val": {k: v / 2 for k in FC.EC_METRICS}, "items_sha256": f"{fold}" * 64,
             "classes_sha256": "c" * 64}
        cvb = CV.cv_result_block(rec, cv, "ec", "structure", fold)
        if sha:
            cvb["function_cv"]["sha256"] = sha
        return {"by_depth": {"1": d}, **common, **cvb}
    for s in range(10):
        for r in range(5):
            v = 0.1 * (r + 1) + 0.01 * s                     # fold mean 0.3 + 0.01 s
            (ecd / f"aa__cv_structure_f{r}__{CR.TAG}{CR._sd_bl(s)}__summary.json").write_text(
                json.dumps(res(v, r)))
            (ecd / f"random__cv_structure_f{r}__{CR.TAG}{CR._sd_bl(s)}__summary.json").write_text(
                json.dumps(res(v, r, sha="0" * 64)))         # another CV file
    CR.set_rmsf_protocol("test")
    try:
        for mode, scale in (("test", 1.0), ("val", 0.5)):
            CR.set_split_metric(mode)
            for x in (CR._SIG, CR._REF_SIG, CR._CV_FOLDS): x.clear()
            CR._ITEM_REFUSED.clear(); CR._FN_REFUSED.clear(); CR._CV_REFUSED.clear()
            got = CR.b_ec("aa", "structure", 1, "macro_ap")
            seeds = [(0.3 + 0.01 * s) * scale for s in range(10)]
            assert got[2] == 10 and abs(got[0] - np.mean(seeds)) < 1e-9
            assert abs(got[1] - np.std(seeds, ddof=1)) < 1e-9
            key = "baseline_aa|structure|EC d1|macro_ap"
            assert len(CR._CV_FOLDS[key]["per_fold"]["0"]) == 10
            assert getattr(CR.b_ec("random", "structure", 1, "macro_ap"), "refused", False)
            assert any(r[0] == "baseline_random" for r in CR._CV_REFUSED)
            # one signature per seed over the five folds
            assert len(CR._SIG[("baseline_aa", "structure", "EC d1")]) == 1
        # the random split has no EC / GO columns under CV
        md = CR.split_table("random", [], (), None, specs=CR.MAIN_SPECS)
        assert "EC d1" not in md.splitlines()[0] and "GO" not in md.splitlines()[0]
    finally:
        CR.set_split_metric("test")


def _dry(cmd, env_root):
    import os
    env = {**os.environ, "ENSEMBITS_REPO_ROOT": str(env_root)}
    return subprocess.run([sys.executable] + cmd, cwd=str(CODE_ROOT), env=env,
                          capture_output=True, text=True, timeout=600)


def test_runners_cv_jobs_dry_run(tmp_path):
    out = _dry(["downstream/baselines/run_misato_baselines.py", "--dry-run", "--baselines", "aa",
                "--seeds", "0", "--tasks", "ec,go", "--folds", "0,1"], tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    lines = [l for l in out.stdout.splitlines() if "ec_classify" in l or "conv1d_GO" in l]
    assert len(lines) == 2 * 2 * 2                 # 2 modes x 2 folds x (EC, GO)
    assert all("--cv-mode" in l and "--split" not in l for l in lines)
    old = _dry(["downstream/baselines/run_misato_baselines.py", "--dry-run", "--baselines", "aa",
                "--seeds", "0", "--tasks", "ec,go", "--function-protocol", "split"], tmp_path)
    lines = [l for l in old.stdout.splitlines() if "ec_classify" in l or "conv1d_GO" in l]
    assert len(lines) == 3 * 2 and all("--split" in l for l in lines)
    import _run_model_battery as RB
    RB.FN_PROTOCOL, RB.FOLDS = "cv", [0, 3]
    try:
        J = [j for j in RB.function_jobs("runX", RB.RECIPE) if j[0] in ("EC", "GO")]
        assert len(J) == 2 * 2 * 2 * len(RB.SEEDS) * 2            # P x modes x folds x seeds x task
        assert {j[3] for j in J} == {"cv_sequence_f0", "cv_sequence_f3",
                                     "cv_structure_f0", "cv_structure_f3"}
        assert all("_cv_" in Path(j[5]).name for j in J)
    finally:
        RB.FN_PROTOCOL, RB.FOLDS = "cv", list(range(5))


# ---------------------------------------------------------------- 5. the built file
def _built():
    p = CV.cv_path()
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return CV.load_function_cv(p)


def test_built_file_folds_groups_and_classes():
    cv, rec = _built()
    from ensembits.paths import MISATO_SPLITS
    chains = B.read_chain_table(Path(MISATO_SPLITS).with_name(
        Path(MISATO_SPLITS).stem + ".chains.json"))
    from go_labels import load_go_labels, build_go_labelset
    from ec_labels import load_ec_labels
    from tokenize_misato_and_eval import build_label_sets
    go, _ = load_go_labels()
    ec = {k.upper(): v for k, v in load_ec_labels()[0].items()}
    for task in CV.CV_TASKS:
        tb = cv["tasks"][task]
        R = sorted(tb["representatives"])
        members = [p for r in R for p in tb["representatives"][r]["members"]]
        assert len(members) == len(set(members)) == tb["n_complexes_labelled"]
        for mode in CV.CV_MODES:
            m = tb["modes"][mode]
            always = set(m["always_train"])
            n = len(R)
            assert all((len(g) > n / 5) == bool(set(g) & always) for g in m["groups"])
            B.check_folds(m["folds"], [g for g in m["groups"] if not set(g) & always],
                          sorted(set(R) - always))
            for r in range(5):
                rot = CV.rotation(cv, task, mode, r)
                assert always <= set(rot["train"])
                if task == "ec":
                    assert len(rot["classes"]["1"]) == 7      # every d1 class in train
            if mode == "structure":
                where = {}
                for f, ps in enumerate(m["folds"]):
                    for p in ps:
                        for h in chains[p]["cath_h"]:
                            where.setdefault(h, set()).add(f)
                assert all(len(v) == 1 for v in where.values())
            r = 3
            rot = CV.rotation(cv, task, mode, r)
            if task == "go":
                _, cls, _ = build_go_labelset({p: go[p] for p in R}, rot["train"], "top50")
                assert rot["classes"] == cls and len(cls) == 50
            else:
                for d in (1, 2, 3):
                    lab = build_label_sets(ec, d)
                    assert rot["classes"][str(d)] == sorted(
                        {c for p in rot["train"] for c in lab.get(p, ())})


def test_built_file_leakage_report():
    cv, rec = _built()
    from ensembits.data_versions import read_provenance
    prov = read_provenance(CV.cv_path())
    for task in CV.CV_TASKS:
        seq = prov["report"]["tasks"][task]["modes"]["sequence"]["leakage"]["max_over_folds"]
        st = prov["report"]["tasks"][task]["modes"]["structure"]["leakage"]["max_over_folds"]
        assert seq["cov05_short_ge30"] == 0 and seq["nocov_aln50_ge50"] == 0
        assert seq["identical_chain"] == 0 and st["identical_chain"] == 0
        assert st["shares_cath_h"] == 0 and st["nocov_aln50_ge50"] == 0


def test_go_main_column_is_map(tmp_path, monkeypatch):
    """GO metric (coordinator, 2026-10-19): main / ablation GO column = mAP; the Metrics: GO
    table starts with mAP, Fmax, top-1; floor columns are not ranked (both compilers)."""
    import ensembits.paths as P
    import _compile_rebuttal_results as CR
    import compile_camera_ready_tables as T
    import function_common as FC
    import go_labels as GL
    from ensembits.data_versions import misato_splits_record
    assert [s for s in CR.MAIN_SPECS if s[1] == "GO"] == [("GO mAP", "GO", None, "macro_ap")]
    assert not hasattr(CR, "ABL_FN_SPECS")   # A9: no function columns in the ablation tables
    go_tab = dict(CR.DETAIL_TABLES)["GO"]
    assert [m for _, _, _, m in go_tab][:3] == ["macro_ap", "fmax", "top1_hit"]
    assert all("floor" in h for h, _, _, m in go_tab if m.startswith("floor:"))
    assert "floor" in CR.NOT_RANKED_COL and "floor" in T.NOT_RANKED_COL
    sp = tmp_path / "misato_splits_v2.json"
    sp.write_text(json.dumps({"stats": {"schema": "misato-splits-v2"}}))
    monkeypatch.setattr(P, "MISATO_SPLITS", sp)
    monkeypatch.setattr(GL, "current_label_sha", lambda path=None: "g" * 64)
    table, n_res, go, ec = _table(30)
    body, _ = B.build(table, n_res, go, ec, set(), 42)
    cvp = _write_cv(tmp_path, body)
    monkeypatch.setattr(P, "MISATO_FUNCTION_CV", cvp)
    cv, rec = CV.load_function_cv(cvp, check_inputs=False)
    god = tmp_path / "go"; god.mkdir(); monkeypatch.setattr(CR, "GO", god)
    monkeypatch.setattr(CR, "FN_CV", True)
    common = {**misato_splits_record(sp), "go_labels": {"sha256": "g" * 64},
              **FC.protocol_record([{"file": "f", "verified": True}], input_std=FC.InputStandardisation.lookup("t", 2, [FC.standardize_lookup_table(np.eye(2))[1]]).run_record())}
    for s in range(10):
        for r in range(5):
            blk = {"macro_ap": 0.4, "top1_hit": 0.9, "fmax": 0.5, "micro_ap": 0.5,
                   "macro_f1@0.5": 0.5, "micro_f1@0.5": 0.5}
            js = {**blk, "val": blk, "items_sha256": f"{r}" * 64, "go_classes_sha256": "c" * 64,
                  **common, **CV.cv_result_block(rec, cv, "go", "sequence", r)}
            (god / f"go_aa_top50_cv_sequence_f{r}_{CR.TAG}{CR._sd_ec(s)}.json").write_text(
                json.dumps(js))
    for x in (CR._SIG, CR._REF_SIG, CR._CV_FOLDS): x.clear()
    CR._ITEM_REFUSED.clear(); CR._FN_REFUSED.clear(); CR._CV_REFUSED.clear(); CR._GO_REFUSED.clear()
    md = CR.split_table("sequence", [], (), None, specs=[s for s in CR.MAIN_SPECS if s[1] == "GO"])
    row = next(l for l in md.splitlines() if l.startswith("| aa |"))
    assert "0.400" in row and "0.900" not in row
