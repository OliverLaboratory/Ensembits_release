"""Tests of the MISATO probe splits v2 (benchmark_audit D3) and of the split-file checks.

What it does
------------
Synthetic, CPU-only tests of `data_prep/build_misato_splits_v2.py`:
  * union-find (path compression, deterministic roots) and the grouping of complexes by
    shared sequence clusters (a complex with chains in clusters A and B joins both);
  * the structure grouping: shared CATH H codes over all chains, complexes without CATH
    attached through their sequence clusters (labelled complexes are not joined through
    sequence clusters), singletons when nothing else shares a cluster;
  * connected components of an all-vs-all hit list;
  * the group assignment: deterministic (input order, seed), large groups forced to train,
    no group straddles train / val / test (`check_no_straddle` also catches a straddle);
  * the leakage report on a hand-made case;
  * the split-file helpers of `ensembits.data_versions` (v2 schema required, result
    records, refusal of results without / with another split record) and the compiler's
    refusal of EC cells built on the old split file.
When `data/misato_splits_v2.json` exists in the research repo, it is checked too
(schema, universe and the copied random / ligand_disjoint splits, no straddling group).
No probe is run and no test metric is read.

Paper
-----
`appendix.tex` \\label{app:experiments} (MISATO splits).

How to run
----------
    python -m pytest tests/test_misato_splits_v2.py
"""
import json
import random
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "data_prep"))
sys.path.insert(0, str(CODE_ROOT / "downstream"))

import build_misato_splits_v2 as B  # noqa: E402


def test_union_find():
    uf = B.UnionFind("abcdef")
    uf.union("a", "b"); uf.union("c", "d"); uf.union("b", "d")
    assert uf.find("a") == uf.find("c") == uf.find("d")
    assert uf.find("e") != uf.find("a")
    assert uf.groups() == [["a", "b", "c", "d"], ["e"], ["f"]]
    uf2 = B.UnionFind("abcdef")                      # other union order, same root
    uf2.union("d", "b"); uf2.union("d", "c"); uf2.union("a", "b")
    assert uf2.find("a") == uf.find("a")


def test_groups_by_shared_keys_joins_both_clusters():
    cc = {"p1": ["A"], "p2": ["A", "B"], "p3": ["B"], "p4": ["C"], "p5": []}
    assert B.groups_by_shared_keys(cc) == [["p1", "p2", "p3"], ["p4"], ["p5"]]


def test_structure_groups_no_cath_attached_via_sequence():
    h = {"a": ["1.10.10.10"], "b": ["1.10.10.10", "2.40.10.10"], "c": ["2.40.10.10"],
         "d": ["3.40.50.300"], "e": [], "f": [], "g": [], "h": ["4.10.1.1"]}
    cc = {"a": ["s1"], "b": ["s2"], "c": ["s3"], "d": ["s4"], "e": ["s4"],   # e -> d via s4
          "f": ["s9"], "g": ["s9"],                                          # f, g no CATH
          "h": ["s1"]}               # h shares a sequence cluster with a but has its own H
    groups, info = B.structure_groups(h, cc)
    assert ["a", "b", "c"] in groups                  # all H codes of all chains
    assert ["d", "e"] in groups                       # no-CATH e attached to d
    assert ["f", "g"] in groups                       # no-CATH group without CATH
    assert ["h"] in groups                            # labelled: not joined via sequence
    assert info["n_complexes_without_cath"] == 3
    assert info["n_without_cath_in_group_with_cath"] == 1
    assert info["n_without_cath_in_group_without_cath_size_gt1"] == 2
    assert info["n_without_cath_singleton"] == 0
    groups, info = B.structure_groups({"x": [], "y": ["1.1.1.1"]}, {"x": ["s7"], "y": ["s8"]})
    assert info["n_without_cath_singleton"] == 1 and ["x"] in groups


def test_connected_clusters():
    ids = {"AAA": "s0", "CCC": "s1", "DDD": "s2", "EEE": "s3"}
    rep = B.connected_clusters(ids, [("s0", "s1"), ("s1", "s2")])
    assert rep["s0"] == rep["s1"] == rep["s2"] == "s0" and rep["s3"] == "s3"


def test_link_rules():
    # hit = (q, t, fident, alnlen, qlen, tlen, qcov, tcov)
    dom_vs_full = ("q", "t", 0.9, 100, 110, 400, 0.91, 0.25)   # domain vs full-length protein
    assert B.link_graph(dom_vs_full) and not B.link_cov08(dom_vs_full)
    near = ("q", "t", 0.97, 60, 300, 300, 0.2, 0.2)             # same protein, short overlap
    assert B.link_graph(near) and not B.link_cov08(near)
    weak = ("q", "t", 0.35, 80, 300, 300, 0.27, 0.27)           # local 35 % over 27 %
    assert not B.link_graph(weak)
    short_aln = ("q", "t", 0.5, 25, 40, 300, 0.62, 0.08)        # < 30 aligned residues
    assert not B.link_graph(short_aln)
    assert B.link_cov08(("q", "t", 0.31, 200, 220, 230, 0.9, 0.87))
    assert B.cov_short(dom_vs_full) == 0.91


def _groups(n=400, seed=0):
    rng = random.Random(seed)
    pids = [f"p{i:04d}" for i in range(n)]
    groups, i = [], 0
    while i < n:
        k = rng.choice([1, 1, 1, 2, 3, 4])               # <= cap (0.1 x val target)
        groups.append(pids[i:i + k]); i += k
    groups.append([f"big{j:03d}" for j in range(30)])   # > 10 % of a 43-complex val target
    return groups


def test_assign_groups_deterministic_and_no_straddle():
    g = _groups()
    sp1, info1 = B.assign_groups(g, seed=42)
    shuffled = [list(reversed(x)) for x in g][::-1]
    sp2, _ = B.assign_groups(shuffled, seed=42)
    assert sp1 == sp2                                  # input order does not matter
    sp3, _ = B.assign_groups(g, seed=7)
    assert sp3 != sp1                                  # the seed does
    B.check_no_straddle(sp1, g)
    n = sum(len(x) for x in g)
    assert sum(len(v) for v in sp1.values()) == n
    assert all(p in sp1["train"] for p in g[-1])       # large group forced to train
    assert info1["n_groups_forced_train"] == 1 and info1["giant_component"]["size"] == 30
    for k, f in zip(B.SPLITS, (0.8, 0.1, 0.1)):
        assert abs(len(sp1[k]) / n - f) < 0.03


def test_check_no_straddle_catches():
    g = [["a", "b"], ["c"]]
    with pytest.raises(ValueError, match="straddles"):
        B.check_no_straddle({"train": ["a"], "val": ["b"], "test": ["c"]}, g)
    with pytest.raises(ValueError):
        B.check_no_straddle({"train": ["a", "b"], "val": ["c"], "test": ["c"]}, g)


def test_leakage_report():
    chains = {"t1": [{"seq": "AAAA"}], "t2": [{"seq": "CCCC"}],
              "v1": [{"seq": "AAAA"}],                       # identical chain in train
              "v2": [{"seq": "DDDD"}, {"seq": "GGGG"}],       # 0.6 hit to train via DDDD
              "x1": [{"seq": "EEEE"}]}                        # nothing
    ids = {"AAAA": "s0", "CCCC": "s1", "DDDD": "s2", "EEEE": "s3", "GGGG": "s4"}
    hits = [("s2", "s1", 0.6, 100, 120, 130, 0.9, 0.9), ("s1", "s2", 0.6, 100, 130, 120, 0.9, 0.9),
            ("s3", "s1", 0.4, 60, 300, 130, 0.2, 0.46)]          # x1: local hit only
    h = {"t1": ["1.1.1.1"], "v1": ["1.1.1.1"], "v2": ["2.2.2.2"], "x1": []}
    acc = {"t2": ["P1"], "x1": ["P1"]}
    sp = {"train": ["t1", "t2"], "val": ["v1", "v2"], "test": ["x1"]}
    r = B.leakage(sp, chains, ids, hits, h, acc)
    a = r["val"]["any_chain"]
    for f in B.LEAK_FILTERS:
        assert a[f]["ge30"] == 2 and a[f]["ge50"] == 2 and a[f]["ge95"] == 1
    assert r["val"]["identical_chain_in_train"] == 1
    assert r["val"]["shares_cath_h_with_train"] == 1
    t = r["test"]["any_chain"]
    assert t["nocov_aln50"]["ge30"] == 1 and t["cov05_short"]["ge30"] == 0
    assert t["cov08_both"]["ge30"] == 0 and r["test"]["shares_uniprot_with_train"] == 1


def _v2_file(path, sp=None):
    sp = sp or {"train": ["a"], "val": ["b"], "test": ["c"]}
    path.write_text(json.dumps({"sequence": sp, "structure": sp, "random": sp,
                                "stats": {"schema": "misato-splits-v2"}}))
    return path


def test_split_helpers(tmp_path):
    from ensembits import data_versions as DV
    p = _v2_file(tmp_path / "misato_splits_v2.json")
    rec = DV.misato_splits_record(p)
    assert rec["splits_schema"] == "misato-splits-v2" and len(rec["splits_sha256"]) == 64
    assert DV.misato_split_problem(rec, p) is None
    assert "old misato_splits.json" in DV.misato_split_problem({}, p)
    assert "is not the current" in DV.misato_split_problem({"splits_sha256": "0" * 64}, p)
    old = tmp_path / "misato_splits.json"
    old.write_text(json.dumps({"sequence": {}, "stats": {"n_pdbs": 1}}))
    with pytest.raises(SystemExit, match="schema"):
        DV.load_misato_splits(old)
    good = tmp_path / "good.json"; good.write_text(json.dumps(rec))
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"top1_hit": 0.5}))
    stale = DV.stale_split_outputs([good, bad, tmp_path / "missing.json"], p)
    assert [o.name for o, _ in stale] == ["bad.json"]


def test_compiler_refuses_old_split_results(tmp_path, monkeypatch):
    import ensembits.paths as P
    import _compile_rebuttal_results as CR
    p = _v2_file(tmp_path / "misato_splits_v2.json")
    monkeypatch.setattr(P, "MISATO_SPLITS", p)
    from ensembits.data_versions import misato_splits_record
    ec = tmp_path / "ec"; ec.mkdir()
    monkeypatch.setattr(CR, "EC", ec)
    monkeypatch.setattr(CR, "FN_CV", False)   # fixed-split files (function CV: test_function_cv.py)
    # benchmark_audit D8 (2026-10-18): EC results must also record the current EC labels
    import ec_labels as EL
    ecl = tmp_path / "misato_ec_v2.json"; ecl.write_text(json.dumps({"A": ["3.4.23.16"]}))
    monkeypatch.setattr(EL, "EC_LABELS", ecl)
    ec_rec = {"ec_labels": EL.ec_labels_record(ecl)}
    V = "runX"
    for s in range(10):
        f = ec / f"tok_{V}_{CR.TAG}{CR._sd_ec(s)}__summary.json"
        f.write_text(json.dumps({"by_depth": {"1": {"top1_hit": 0.5}}, **misato_splits_record(p),
                                 **ec_rec}))
    CR._SPLIT_REFUSED.clear()
    CR.m_ec(V, False, "sequence", 1)          # (other checks, e.g. EC labels D8, may refuse)
    assert not CR._SPLIT_REFUSED                  # ... but not the split-file check
    f = ec / f"tok_{V}_{CR.TAG}_seed3__summary.json"
    f.write_text(json.dumps({"by_depth": {"1": {"top1_hit": 0.5}}}))   # old split: no record
    v = CR.m_ec(V, False, "sequence", 1)
    assert getattr(v, "refused", False)
    assert CR._SPLIT_REFUSED and CR._SPLIT_REFUSED[0][2] == f.name
    CR._SPLIT_REFUSED.clear()


REAL = B.MISATO_SPLITS


@pytest.mark.skipif(not REAL.exists(), reason="data/misato_splits_v2.json not built")
def test_real_split_file():
    from ensembits.data_versions import load_misato_splits, read_provenance
    new = load_misato_splits(REAL)
    old = json.loads(B.MISATO_SPLITS_LEGACY.read_text())
    uni = B.universe(old)
    for name in B.COPIED_SPLITS:
        assert new[name] == old[name]
    for name in B.NEW_SPLITS:
        assert sorted(p for k in B.SPLITS for p in new[name][k]) == uni
    side = read_provenance(REAL)
    ch = json.loads(REAL.with_name(REAL.stem + B.CHAINS_OUT_SUFFIX).read_text())["complexes"]
    cc = {p: v["clusters"] for p, v in ch.items()}
    hh = {p: v["cath_h"] for p, v in ch.items()}
    B.check_no_straddle(new["sequence"], B.groups_by_shared_keys(cc))
    B.check_no_straddle(new["structure"], B.structure_groups(hh, cc)[0])
    for name in B.NEW_SPLITS:                          # the leakage the builder measured
        for k in ("val", "test"):
            lk = side["report_new"][name]["leakage"][k]
            assert lk["identical_chain_in_train"] == 0
            assert lk["any_chain"]["cov08_both"]["ge30"] == 0
            assert lk["any_chain"]["nocov_aln50"]["ge50"] == 0
    for k in ("val", "test"):
        assert side["report_new"]["structure"]["leakage"][k]["shares_cath_h_with_train"] == 0


def test_uniprot_keys_polyprotein_products(tmp_path):
    import gzip
    sifts = tmp_path / "pdb_chain_uniprot.tsv.gz"
    rows = ["# SIFTS", "PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\tPDB_BEG\tPDB_END\tSP_BEG\tSP_END",
            "1aaa\tA\tPOLY1\t1\t99\t1\t99\t1\t99",           # protease product
            "2bbb\tA\tPOLY1\t1\t99\t1\t99\t1\t99",           # protease again -> same key
            "3ccc\tA\tPOLY1\t1\t500\t1\t500\t120\t620",      # reverse transcriptase
            "4ddd\tB\tP12345\t1\t50\t1\t50\t10\t59"]         # ordinary accession
    with gzip.open(sifts, "wt") as f:
        f.write("\n".join(rows) + "\n")
    feat = lambda d, s, e: {"type": "Chain", "description": d,
                            "location": {"start": {"value": s}, "end": {"value": e}}}
    entries = {"entries": {
        "POLY1": {"proteinDescription": {"recommendedName": {"fullName": {"value": "Gag-Pol"}}},
                  "features": [feat("Protease", 1, 99), feat("Reverse transcriptase", 100, 660)]},
        "P12345": {"proteinDescription": {"recommendedName": {"fullName": {"value": "X"}}},
                   "features": [feat("X", 1, 300)]}}}
    ep = tmp_path / "entries.json.gz"
    with gzip.open(ep, "wt") as f:
        json.dump(entries, f)
    keys, info = B.chain_uniprot_keys({"1aaa", "2bbb", "3ccc", "4ddd"}, sifts, ep)
    assert keys[("1aaa", "A")] == keys[("2bbb", "A")] == {("POLY1", "Protease")}
    assert keys[("3ccc", "A")] == {("POLY1", "Reverse transcriptase")}
    assert keys[("4ddd", "B")] == {"P12345"}
    assert info["n_polyprotein_accessions"] == 1 and info["n_polyprotein_chains_by_product"] == 3
