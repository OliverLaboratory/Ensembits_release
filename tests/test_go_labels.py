"""Tests of the MISATO GO labels v2 and the shared multi-label metrics (benchmark_audit D7).

What it does
------------
  * builder (`data_prep/build_misato_go_labels.py`) on a toy ontology: propagation over
    is_a + part_of within MF (cross-namespace edges ignored), alt-id mapping, obsolete
    terms (single MF replaced_by mapped, others removed), exclusion of BP / CC, the MF
    root, GO:0005515 and do-not-annotate terms before and after propagation, proteins
    left without labels omitted;
  * class selection (`downstream/go_labels.py`): frequency band (>= 10, < 50 % of the
    annotated training proteins), top-50 with ties broken by GO id, independent of the
    order of the training pids;
  * label loader sha check and the compilers' result check (`go_result_problems`);
  * `multilabel_metrics`: CAFA Fmax on a hand-computed case, the frequency floor
    (including its tie rule), one `metrics` shared by the EC / GO scripts.

Paper
-----
not described in the paper (tests)

How to run
----------
  python -m pytest tests/test_go_labels.py -q      # from the codebase root (CPU, seconds)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream"))
sys.path.insert(0, str(CODE_ROOT / "downstream" / "baselines"))
sys.path.insert(0, str(CODE_ROOT / "data_prep"))

import build_misato_go_labels as B   # noqa: E402
import go_labels as GL               # noqa: E402
import multilabel_metrics as MM      # noqa: E402

TOY_OBO = """format-version: 1.2
data-version: releases/toy

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function

[Term]
id: GO:0000001
name: a
namespace: molecular_function
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0000002
name: b
namespace: molecular_function
is_a: GO:0000001 ! a

[Term]
id: GO:0000003
name: c
namespace: molecular_function
is_a: GO:0000002 ! b
relationship: part_of GO:0000004 ! d

[Term]
id: GO:0000004
name: d
namespace: molecular_function
alt_id: GO:0000012
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0005515
name: protein binding
namespace: molecular_function
is_a: GO:0000001 ! a

[Term]
id: GO:0000005
name: grouping term
namespace: molecular_function
subset: gocheck_do_not_annotate
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0000006
name: e
namespace: molecular_function
is_a: GO:0000005 ! grouping term

[Term]
id: GO:0000007
name: obsolete x
namespace: molecular_function
is_obsolete: true
replaced_by: GO:0000002

[Term]
id: GO:0000008
name: obsolete y
namespace: molecular_function
is_obsolete: true

[Term]
id: GO:0000009
name: obsolete z
namespace: molecular_function
is_obsolete: true
replaced_by: GO:0000010

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000010
name: some process
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0000011
name: f
namespace: molecular_function
is_a: GO:0003674 ! molecular_function
relationship: part_of GO:0000010 ! some process

[Typedef]
id: part_of
name: part of
"""


@pytest.fixture()
def toy(tmp_path):
    p = tmp_path / "toy.obo"
    p.write_text(TOY_OBO)
    return B.parse_obo(p)


def test_parse_obo(toy):
    terms, alt, version = toy
    assert version == "releases/toy"
    assert alt == {"GO:0000012": "GO:0000004"}
    assert terms["GO:0000003"]["part_of"] == {"GO:0000004"}
    assert terms["GO:0000007"]["obsolete"] and terms["GO:0000007"]["replaced_by"] == ["GO:0000002"]
    assert "gocheck_do_not_annotate" in terms["GO:0000005"]["subsets"]
    assert "part_of" not in terms                      # [Typedef] stanzas are skipped


def test_propagation_and_exclusions(toy):
    terms, alt, _ = toy
    source = {"p1": ["GO:0000003"],                    # c -> b -> a, and part_of d
              "P2": ["GO:0005515", "GO:0000010"],      # protein binding + BP only -> dropped
              "P3": ["GO:0000006"],                    # ancestor is do-not-annotate
              "P4": ["GO:0000007", "GO:0000008"],      # obsolete: mapped / removed
              "P5": ["GO:0000009", "GO:0000012"],      # obsolete -> BP (removed); alt id
              "P6": ["GO:0000011"],                    # part_of to BP ignored
              "P7": ["GO:0003674", "GO:0000005"]}      # root + do-not-annotate only
    out, rep = B.clean_and_propagate(source, terms, alt)
    assert out == {"P1": ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"],
                   "P3": ["GO:0000006"],
                   "P4": ["GO:0000001", "GO:0000002"],
                   "P5": ["GO:0000004"],
                   "P6": ["GO:0000011"]}
    assert rep["obsolete_mapped"] == {"GO:0000007": {"to": "GO:0000002", "n": 1}}
    assert dict(rep["obsolete_removed"]) == {"GO:0000008": 1, "GO:0000009": 1}
    assert dict(rep["alt_mapped"]) == {"GO:0000012->GO:0000004": 1}
    assert dict(rep["excluded_direct"]) == {"GO:0005515": 1, "GO:0003674": 1, "GO:0000005": 1}
    assert rep["excluded_after_propagation"]["GO:0000005"] == 1      # from P3's ancestor
    assert rep["excluded_after_propagation"]["GO:0003674"] == 5
    assert dict(rep["non_mf"]) == {"biological_process": 1}


def _labels_for_selection():
    """400 annotated proteins: a filler term on all (100 %), a term on 250 (>= 50 %), a
    rare term on 9, and 53 terms T00..T52 with counts 100 - i, except T49..T52 tied."""
    n = 400
    labels = {f"P{k:03d}": ["GO:9999999"] for k in range(n)}
    for k in range(250):
        labels[f"P{k:03d}"].append("GO:8888888")
    for k in range(9):
        labels[f"P{k:03d}"].append("GO:7777777")
    for i in range(53):
        c = 100 - i if i < 49 else 51
        tid = f"GO:00{i:05d}"
        for k in range(c):
            labels[f"P{(k + 7 * i) % n:03d}"].append(tid)
    return labels


def test_class_selection_band_and_ties():
    labels = _labels_for_selection()
    train = list(labels)
    ranked, counts, info = GL.select_go_classes(labels, train, "top50")
    assert len(ranked) == 50
    assert "GO:9999999" not in ranked and "GO:8888888" not in ranked   # >= 50 %
    assert "GO:7777777" not in ranked                                  # < 10 proteins
    assert ranked[:3] == ["GO:0000000", "GO:0000001", "GO:0000002"]
    # four terms tied at 51 for the last slot: the smallest id wins
    assert ranked[-1] == "GO:0000049" and counts["GO:0000049"] == 51
    assert info["n_tied_beyond_last_rank"] == 3 and info["count_at_last_rank"] == 51
    assert info["n_train_annotated"] == 400 and info["n_too_frequent"] == 2
    # independent of the order of the training pids (old code: read order decided ties)
    for seed in range(3):
        sh = train[:]; random.Random(seed).shuffle(sh)
        assert GL.select_go_classes(labels, sh, "top50")[0] == ranked
    lab, classes, sel = GL.build_go_labelset(labels, train, "top50")
    assert classes == sorted(ranked)
    assert all(set(v) <= set(classes) for v in lab.values())
    assert sel["train_freq"]["GO:0000000"] == pytest.approx(100 / 400)
    with pytest.raises(ValueError):
        GL.select_go_classes(labels, train, "nope")


def test_fmax_hand_computed():
    y = np.array([[1, 0, 0], [0, 1, 1]])
    p = np.array([[0.9, 0.2, 0.1], [0.6, 0.7, 0.3]])
    f, t = MM.fmax(p, y)
    # t in (0.20, 0.30]: P1 predicts {0} (pr 1, rc 1), P2 predicts {0,1,2} (pr 2/3, rc 1)
    assert f == pytest.approx(2 * (5 / 6) / (5 / 6 + 1))
    assert t == pytest.approx(0.21)
    # a protein without true labels is not scored
    f2, _ = MM.fmax(np.vstack([p, [[0.9, 0.9, 0.9]]]), np.vstack([y, [[0, 0, 0]]]))
    assert f2 == pytest.approx(f)
    # perfect scores
    assert MM.fmax(y.astype(float), y)[0] == pytest.approx(1.0)


def test_frequency_floor_and_shared_metrics():
    y = np.array([[1, 0, 0], [0, 1, 1]])
    fl = MM.frequency_floor([0.3, 0.5, 0.1], y)
    assert fl["top1_hit"] == pytest.approx(0.5) and fl["n_test"] == 2
    assert fl["fmax"] == pytest.approx(MM.fmax(np.tile([0.3, 0.5, 0.1], (2, 1)), y)[0])
    # tie in the training frequency: the first column (smallest GO id) is the top-1 pick
    assert MM.frequency_floor([0.4, 0.4, 0.1], np.array([[1, 0, 0]] * 2))["top1_hit"] == 1.0
    assert MM.frequency_floor([0.4, 0.4, 0.1], np.array([[0, 1, 0]] * 2))["top1_hit"] == 0.0
    m = MM.metrics(np.array([[0.9, 0.2, 0.1], [0.6, 0.7, 0.3]]), y)
    assert set(m) == {"macro_ap", "micro_ap", "top1_hit", "macro_f1@0.5", "micro_f1@0.5",
                      "fmax", "fmax_threshold", "n_classes_in_test"}
    assert m["top1_hit"] == pytest.approx(1.0) and m["fmax"] == pytest.approx(10 / 11)
    nan = MM.metrics(np.zeros((2, 3)), np.zeros((2, 3)))
    assert set(nan) == set(m) and nan["n_classes_in_test"] == 0


def test_one_metrics_function_for_ec_and_go():
    pytest.importorskip("torch")
    import tokenize_misato_and_eval as T
    import ec_classify_conv1d_misato as E
    assert T.metrics is MM.metrics and E.metrics is MM.metrics


def _write_labels(tmp_path, labels):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels, sort_keys=True))
    Path(str(path) + ".provenance.json").write_text(json.dumps(
        {"schema": GL.GO_SCHEMA, "sha256": GL.file_sha256(path),
         "ontology": {"data_version": "releases/toy"}, "source": {"sha256": "s"}}))
    return path


def test_loader_and_result_checks(tmp_path):
    path = _write_labels(tmp_path, {"1abc": ["GO:0000001"]})
    labels, info = GL.load_go_labels(path)
    assert labels == {"1ABC": ["GO:0000001"]} and info["sha256"] == GL.file_sha256(path)
    assert GL.go_result_problems({"go_labels": info}, path) == []
    assert GL.go_result_problems({"top1_hit": 0.5}, path)               # old result
    assert GL.go_result_problems({"go_labels": {"sha256": "x" * 64}}, path)
    out = tmp_path / "go_x_top50.json"
    out.write_text(json.dumps({"go_labels": info}))
    assert GL.go_result_is_current(out, path)
    path.write_text(json.dumps({"1ABC": ["GO:0000002"]}))                # edited file
    with pytest.raises(SystemExit):
        GL.load_go_labels(path)


def test_go_result_fields():
    labels = _labels_for_selection()
    train = list(labels)[:300]
    lab, classes, sel = GL.build_go_labelset(labels, train, "top50")
    split = {p: ("train" if i < 300 else "test") for i, p in enumerate(labels)}
    idx = {c: i for i, c in enumerate(classes)}
    bucket = {"train": [], "val": [], "test": []}
    for p, s in split.items():
        if p in lab:
            y = np.zeros(len(classes), dtype=np.float32)
            for c in lab[p]:
                y[idx[c]] = 1
            bucket[s].append({"pid": p, "y": y})
    info = {"file": "x", "sha256": "h", "schema": GL.GO_SCHEMA}
    r = GL.go_result_fields(info, sel, classes, bucket, split, labels, labels)
    assert r["go_classes"] == classes and r["go_labels"] == info
    assert r["test_pids_sha256"] == GL.pid_list_sha256([it["pid"] for it in bucket["test"]])
    assert r["go_split_counts"]["test"]["kept"] == len(bucket["test"])
    assert r["go_split_counts"]["train"]["with_features"] == 300
    assert 0 <= r["frequency_floor"]["top1_hit"] <= 1
