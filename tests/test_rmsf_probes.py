"""RMSF probes: common residue set, per-protein Spearman, sequences, labels, compilers.

What it does
------------
Synthetic, CPU-only tests (pytest tmp dirs; ENSEMBITS_REPO_ROOT points at a fake repo):
  * rmsf_common: per-protein Spearman, residue-set hash / tamper check / row mask,
    strict feature placement, His normalisation (preretrain_audit_r2 Q2, Q3, Q29);
  * data_prep/build_rmsf_residue_sets.py on fake mdCATH-div and MISATO data;
  * end to end: probe_rmsf_baselines.py (mdCATH-div), probe_rmsf_eval.py (with the
    checkpoint in ckpt/), probe_rmsf_misato.py (aa mode): 100% coverage of the set or
    an error, residue-set record and per-protein fields in the JSON;
  * the MISATO coords hash is recomputed from the file (R2-C10);
  * label builders: all frames of K identical copies = the label of the frames,
    pose independence, raw-frame check of the stored ca_10 frames, MISATO all frames;
  * compilers refuse mixed RMSF rows; the selection table shows validation numbers only;
  * ProToken token-count mismatch tolerance.

Paper
-----
`2_results.tex` \\label{result:rmsf}; `appendix.tex` \\label{app:baselines}.

How to run
----------
  python -m pytest tests/test_rmsf_probes.py
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = CODE_ROOT / "downstream"
DATA_PREP = CODE_ROOT / "data_prep"
for _p in (CODE_ROOT, DOWNSTREAM, DATA_PREP, DATA_PREP / "baselines"):
    sys.path.insert(0, str(_p))

import rmsf_common as RC  # noqa: E402


def _env(root, **extra):
    e = dict(os.environ)
    e["ENSEMBITS_REPO_ROOT"] = str(root)
    e["ENSEMBITS_MDCATH_DIV_SRC"] = str(Path(root) / "data" / "external" / "mdcath_div")
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.update(extra)
    return e


def _run(root, module, argv, folder=DOWNSTREAM):
    code = (f"import sys; sys.path.insert(0, {str(folder)!r}); import {module} as M; "
            f"sys.exit(M.main({argv!r}))")
    return subprocess.run([sys.executable, "-c", code], env=_env(root), capture_output=True,
                          text=True, timeout=600)


# ---------------------------------------------------------------- rmsf_common units
def test_per_protein_spearman_rules():
    g = np.array([0] * 12 + [1] * 12 + [2] * 5 + [3] * 12)
    y = np.concatenate([np.arange(12.), np.arange(12.), np.arange(5.), np.ones(12)])
    p = np.concatenate([np.arange(12.), -np.arange(12.), np.arange(5.), np.arange(12.)])
    r = RC.per_protein_spearman(p, y, g)
    # protein 2 too short, protein 3 constant label -> 2 proteins: +1 and -1
    assert r["n_scored"] == 2 and r["n_too_short"] == 1 and r["n_constant_label"] == 1
    assert r["mean"] == pytest.approx(0.0) and r["median"] == pytest.approx(0.0)
    # constant prediction inside an eligible protein scores 0 (same proteins for every method)
    p2 = np.concatenate([np.full(12, 3.), np.arange(12.), np.arange(5.), np.arange(12.)])
    r2 = RC.per_protein_spearman(p2, y, g)
    assert r2["n_scored"] == 2 and r2["n_constant_pred"] == 1 and r2["mean"] == pytest.approx(0.5)
    f = RC.per_protein_fields("val", r)
    assert set(f) == {"val_spearman_per_protein_mean", "val_spearman_per_protein_median",
                      "n_val_proteins_scored", "n_val_proteins", "n_val_proteins_constant_pred"}


def test_residue_set_hash_mask_and_tamper(tmp_path):
    prot = {"a": {"row_start": 0, "length": 3}, "c": {"row_start": 5, "length": 2}}
    rs = RC.make_residue_set("mdcath_div", prot, "rule", n_rows_total=8, excluded={"b": "x"})
    # hash independent of dict order and of the rule text
    rs2 = RC.make_residue_set("mdcath_div", dict(reversed(list(prot.items()))), "other",
                              n_rows_total=8)
    assert rs["sha256"] == rs2["sha256"]
    p = tmp_path / "set.json"; p.write_text(json.dumps(rs))
    got = RC.load_residue_set(p)
    m = RC.residue_set_row_mask(got, 8)
    assert m.tolist() == [True, True, True, False, False, True, True, False]
    with pytest.raises(RC.ResidueCoverageError):
        RC.residue_set_row_mask(got, 9)
    sp = RC.restrict_splits_to_rows({"train": [0, 1, 3], "val": [2, 4], "test": [5, 6, 7]}, m)
    assert sp["train"].tolist() == [0, 1] and sp["test"].tolist() == [5, 6]
    with pytest.raises(RC.ResidueCoverageError):      # row 6 of the set in no split
        RC.restrict_splits_to_rows({"train": [0, 1, 2], "val": [5], "test": []}, m)
    bad = dict(rs); bad["proteins"] = {"a": {"row_start": 0, "length": 4}}
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        RC.load_residue_set(p)
    rec = RC.resolve_residue_set("all", "mdcath_div")[1]
    assert rec["residue_set"] == "all" and rec["residue_set_sha256"] is None
    with pytest.raises(FileNotFoundError):
        RC.resolve_residue_set(str(tmp_path / "none.json"), "mdcath_div")


def test_place_features_strict():
    offsets = {"a": (0, 3), "b": (3, 5)}
    f, m = RC.place_per_pid_features({"a": np.ones((3, 2))}, offsets, 5, 2, required={"a"})
    assert m.tolist() == [True] * 3 + [False] * 2
    with pytest.raises(RC.ResidueCoverageError):        # missing protein inside the set
        RC.place_per_pid_features({"a": np.ones((3, 2))}, offsets, 5, 2, required={"a", "b"})
    with pytest.raises(RC.ResidueCoverageError):        # length mismatch (was min(...) cut)
        RC.place_per_pid_features({"a": np.ones((4, 2))}, offsets, 5, 2)


def test_residue_names():
    assert RC.normalize_mdcath_real_bb_seq("AXGX") == "AHGH"
    with pytest.raises(ValueError):
        RC.normalize_mdcath_real_bb_seq("AHX")
    assert [RC.three_to_one(n) for n in ("HSD", "HSE", "HSP", "HID", "HIE", "HIP", "CYX",
                                         "CYM", "ASH", "GLH", "LYN", "ALA", "UNK")] == \
        list("HHHHHHCCDEKAX")


# ---------------------------------------------------------------- fake mdCATH-div repo
N_DOM, L_DOM = 10, 14
EXCL = ("d8",)                          # no real-backbone file (C-alpha only)


def _fake_mdcath(root, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    src = root / "data" / "external" / "mdcath_div"; src.mkdir(parents=True)
    rb = root / "data" / "cached_descriptors" / "mdcath_real_bb"; rb.mkdir(parents=True)
    pids = [f"d{i}" for i in range(N_DOM)]
    (src / "_manifest.json").write_text(json.dumps({"representatives": {p: {} for p in pids}}))
    for p in pids:
        ca = rng.normal(size=(10, L_DOM, 3)).astype(np.float32)
        np.savez(src / f"{p}.npz", ca_5=ca[:5], ca_10=ca)
        if p not in EXCL:
            np.savez(rb / f"{p}.npz", seq=np.array("ACXDEFGXIKLMNP"))
    total = N_DOM * L_DOM
    cd = root / "output" / "phase_b" / "cached_desc"; cd.mkdir(parents=True)
    np.save(cd / "rmsf_div_allframes.npy", rng.gamma(2.0, 1.0, total).astype(np.float32))
    (cd / "rmsf_div_allframes.npy.provenance.json").write_text(json.dumps(
        {"label_name": "allframes", "n_rows": total}))
    rows = lambda ds: [r for d in ds for r in range(d * L_DOM, (d + 1) * L_DOM)]
    ft = root / "output" / "full_train" / "cached_desc"; ft.mkdir(parents=True)
    ft.joinpath("splits.json").write_text(json.dumps(
        {"train": rows(range(0, 6)), "val": rows([6, 7]), "test": rows([8, 9])}))
    return pids, total


def _build_set(root, dataset, extra=()):
    return _run(root, "build_rmsf_residue_sets", ["--dataset", dataset, *extra], DATA_PREP)


def test_build_mdcath_residue_set(tmp_path):
    _fake_mdcath(tmp_path)
    out = tmp_path / "data" / "cached_descriptors" / "rmsf_residue_set_mdcath_div.json"
    # N1 (2026-10-17): a representative without a real-backbone file is an error ...
    r = _build_set(tmp_path, "mdcath_div")
    assert r.returncode == 1 and "d8" in r.stdout and "--allow-missing" in r.stdout
    assert not out.exists()
    # ... unless --allow-missing, which excludes it and records the choice
    r = _build_set(tmp_path, "mdcath_div", ["--allow-missing"])
    assert r.returncode == 0, r.stdout + r.stderr
    rs = RC.load_residue_set(out)
    assert set(rs["excluded"]) == set(EXCL) and rs["n_proteins"] == N_DOM - len(EXCL)
    assert rs["proteins"]["d9"] == {"row_start": 9 * L_DOM, "length": L_DOM}
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["sha256"] == rs["sha256"] and prov["n_rows_total"] == N_DOM * L_DOM
    assert prov["allow_missing"] is True and prov["missing_real_bb"] == list(EXCL)
    assert _build_set(tmp_path, "mdcath_div", ["--allow-missing"]).returncode == 1  # no overwrite
    # with every file present the flag is not needed and nothing is excluded
    rb = tmp_path / "data" / "cached_descriptors" / "mdcath_real_bb"
    np.savez(rb / "d8.npz", seq=np.array("A" * L_DOM))
    r = _build_set(tmp_path, "mdcath_div", ["--overwrite"])
    assert r.returncode == 0, r.stdout + r.stderr
    rs = RC.load_residue_set(out)
    assert rs["excluded"] == {} and rs["n_proteins"] == N_DOM
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["allow_missing"] is False and prov["missing_real_bb"] == []
    # a real-backbone sequence of the wrong length is an error
    np.savez(tmp_path / "data" / "cached_descriptors" / "mdcath_real_bb" / "d1.npz",
             seq=np.array("ACD"))
    assert _build_set(tmp_path, "mdcath_div", ["--overwrite"]).returncode == 1


def test_probe_rmsf_baselines_common_set(tmp_path):
    _fake_mdcath(tmp_path)
    assert _build_set(tmp_path, "mdcath_div", ["--allow-missing"]).returncode == 0
    args = ["--device", "cpu", "--max-epochs", "3", "--patience", "2"]
    r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", "aa", *args])
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "output" / "probe_rmsf_div" / "baseline_aa__test.json").read_text())
    n_set = (N_DOM - len(EXCL)) * L_DOM
    assert js["n_train_residues"] + js["n_val_residues"] + js["n_test_residues"] == n_set
    assert js["n_test_residues"] == L_DOM            # d8 (excluded) is a test domain
    assert js["residue_set"] == "common" and len(js["residue_set_sha256"]) == 64
    assert js["rmsf_label"] == "allframes" and js["n_test_proteins"] == 1
    for k in ("val_spearman_per_protein_mean", "test_spearman_per_protein_median",
              "n_val_proteins_scored"):
        assert k in js
    # ProtProfile without the excluded domain is fine; without a set domain it is an error
    cache = tmp_path / "data" / "cached_descriptors"
    np.save(cache / "mini3di_centroids.npy", np.zeros((20, 2), np.float32))
    hist = {p: np.full((L_DOM, 20), 0.05, np.float32) for p in [f"d{i}" for i in range(N_DOM)]
            if p not in EXCL}
    np.savez(cache / "protprofile_K10_mdcath.npz", **hist)
    # F1: without a provenance sidecar the cache is refused
    r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", "protprofile_K10", *args])
    assert r.returncode == 1 and "provenance" in r.stdout
    RC.write_cache_provenance(cache / "protprofile_K10_mdcath.npz", dataset="mdcath", builder="test",
                              inputs={}, n_proteins=len(hist), chain_split=RC.chain_split_for("mdcath"),
                              frames="first 10 of bb_10",
                              sequence_source=RC.SEQUENCE_SOURCES["mdcath_div"])
    r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", "protprofile_K10", *args])
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "output" / "probe_rmsf_div" / "baseline_protprofile_K10__test.json").read_text())
    assert js["feature_caches"][0]["file"] == "protprofile_K10_mdcath.npz" and js["rmsf_label_sha256"]
    assert js["feature_caches"][0]["chain_split"] == "whole protein (single chain)"
    del hist["d3"]; np.savez(cache / "protprofile_K10_mdcath.npz", **hist)
    RC.write_cache_provenance(cache / "protprofile_K10_mdcath.npz", dataset="mdcath", builder="test",
                              inputs={}, n_proteins=len(hist), chain_break_A=2.5,
                              frames="first 10 of bb_10",
                              sequence_source=RC.SEQUENCE_SOURCES["mdcath_div"])
    r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", "protprofile_K10", "--seed", "1", *args])
    assert r.returncode == 1 and "d3" in r.stdout
    # without the residue-set file the probe refuses unless --residue-set all
    (cache / "rmsf_residue_set_mdcath_div.json").unlink()
    r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", "random", *args])
    assert r.returncode == 1 and "residue set" in r.stdout
    r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", "random", "--residue-set", "all", *args])
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "output" / "probe_rmsf_div" / "baseline_random__test.json").read_text())
    assert js["residue_set"] == "all" and js["n_test_residues"] == 2 * L_DOM


def _link_ckpt(root, name="ckptrun"):
    d = root / "output" / name; d.mkdir(parents=True)
    for f in ("best.pt", "config.json", "stats.npz", "history.json"):
        os.symlink(CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10" / f, d / f)
    return name


def test_probe_rmsf_eval_end_to_end(tmp_path):
    pids, total = _fake_mdcath(tmp_path)
    assert _build_set(tmp_path, "mdcath_div", ["--allow-missing"]).returncode == 0
    v = _link_ckpt(tmp_path)
    rng = np.random.default_rng(1)
    (tmp_path / "output" / "phase_b" / "cached_desc" / "esm3desc_dihedral_k16_P10.npy.provenance.json"
     ).write_text(json.dumps({"dataset": "mdcath", "n_rows": total, "IDEAL_PIDS": ["d8"]}))
    np.save(tmp_path / "output" / "phase_b" / "cached_desc" / "esm3desc_dihedral_k16_P10.npy",
            rng.normal(size=(total, 10, 256)).astype(np.float32))
    r = _run(tmp_path, "probe_rmsf_eval", ["--variant", v, "--device", "cpu", "--max-epochs", "3",
                                           "--patience", "2", "--p-inference", "1"])
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "output" / "probe_rmsf_div" / f"{v}_P1inf__test.json").read_text())
    assert js["n_test_residues"] == L_DOM and js["n_val_proteins"] == 2
    assert js["residue_set_n_proteins"] == N_DOM - len(EXCL)
    assert js["checkpoint"]["best_pt_sha256"] and js["dataset"] == "mdcath_div"
    assert np.isfinite(js["val_spearman_per_protein_mean"])
    # an existing result made on another residue set is not reused
    js["residue_set_sha256"] = "0" * 64
    (tmp_path / "output" / "probe_rmsf_div" / f"{v}_P1inf__test.json").write_text(json.dumps(js))
    r = _run(tmp_path, "probe_rmsf_eval", ["--variant", v, "--device", "cpu", "--p-inference", "1"])
    assert r.returncode == 1 and "residue set" in r.stdout
    # F9: a frozen snapshot is accepted as the variant and recorded as such
    r = _run(tmp_path, "snapshot_checkpoint", ["--run", v, "--name", "snapA"])
    assert r.returncode == 0, r.stdout + r.stderr
    r = _run(tmp_path, "probe_rmsf_eval", ["--variant", "snapA", "--device", "cpu", "--max-epochs", "2",
                                           "--patience", "1", "--p-inference", "1"])
    assert r.returncode == 0, r.stdout + r.stderr
    js2 = json.loads((tmp_path / "output" / "probe_rmsf_div" / "snapA_P1inf__test.json").read_text())
    assert js2["snapshot_of"] == v and js2["checkpoint"]["best_pt_sha256"] == js["checkpoint"]["best_pt_sha256"]


# ---------------------------------------------------------------- fake MISATO repo
MIS = [f"P{i:03d}" for i in range(12)]


def _fake_misato(root, L=11):
    from ensembits.data_versions import MISATO_PARSER_VERSION
    rng = np.random.default_rng(2)
    cache = root / "data" / "cached_descriptors"; rb = cache / "misato_real_bb"
    rb.mkdir(parents=True, exist_ok=True)
    for p in MIS:
        np.savez(rb / f"{p}.npz", bb_8=rng.normal(size=(8, L, 4, 3)).astype(np.float32),
                 seq=np.array("ACDEFGHIKLM"), parser_version=np.array(MISATO_PARSER_VERSION))
    labels = {p: rng.gamma(2.0, 1.0, L).astype(np.float32) for p in MIS[:-1]}  # last: no label
    lab = cache / "misato_rmsf_allframes.npz"; np.savez(lab, **labels)
    Path(str(lab) + ".provenance.json").write_text(json.dumps(
        {"label_name": "allframes", "parser_version": MISATO_PARSER_VERSION}))
    lo = [p.lower() for p in MIS]
    sp = {"train": lo[:6], "val": lo[6:9], "test": lo[9:]}
    # benchmark_audit D3 (2026-10-18): the probes read misato_splits_v2.json (v2 schema)
    (root / "data" / "misato_splits_v2.json").write_text(json.dumps(
        {"sequence": sp, "structure": sp, "random": sp, "stats": {"schema": "misato-splits-v2"}}))
    return labels


def test_misato_set_and_probe(tmp_path):
    _fake_misato(tmp_path)
    r = _build_set(tmp_path, "misato")
    assert r.returncode == 0, r.stdout + r.stderr
    rs = RC.load_residue_set(tmp_path / "data" / "cached_descriptors" / "rmsf_residue_set_misato.json")
    assert rs["n_proteins"] == len(MIS) - 1 and rs["excluded"] == {MIS[-1]: "no RMSF label"}
    r = _run(tmp_path, "probe_rmsf_misato", ["--mode", "aa", "--device", "cpu", "--max-epochs",
                                             "3", "--patience", "2"])
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "output" / "probe_rmsf_misato" /
                     "baseline_aa_structure__test.json").read_text())
    assert js["rmsf_label"] == "allframes" and js["residue_set_sha256"] == rs["sha256"]
    assert js["n_test_residues"] == 2 * 11 and js["n_test_proteins"] == 2
    # a feature of the wrong length inside the set is an error (was: cut to the shorter)
    # F6: an existing, matching result is kept (skip), not overwritten
    r = _run(tmp_path, "probe_rmsf_misato", ["--mode", "aa", "--device", "cpu", "--max-epochs", "3",
                                             "--patience", "2"])
    assert r.returncode == 0 and "[skip]" in r.stdout, r.stdout + r.stderr
    from ensembits.data_versions import MISATO_PARSER_VERSION
    np.savez(tmp_path / "data" / "cached_descriptors" / "misato_real_bb" / "P003.npz",
             bb_8=np.zeros((8, 11, 4, 3), np.float32), seq=np.array("ACD"),
             parser_version=np.array(MISATO_PARSER_VERSION))
    r = _run(tmp_path, "probe_rmsf_misato", ["--mode", "aa", "--device", "cpu", "--max-epochs", "2",
                                             "--seed", "1"])
    assert r.returncode == 1 and "P003" in r.stderr
    # a sequence from an old parser is refused (F1)
    np.savez(tmp_path / "data" / "cached_descriptors" / "misato_real_bb" / "P003.npz",
             bb_8=np.zeros((8, 11, 4, 3), np.float32), seq=np.array("ACDEFGHIKLM"),
             parser_version=np.array("old"))
    r = _run(tmp_path, "probe_rmsf_misato", ["--mode", "aa", "--device", "cpu", "--max-epochs", "2",
                                             "--seed", "2"])
    assert r.returncode == 1 and "parser" in r.stderr


def test_misato_gather_legacy_and_strict():
    import probe_rmsf_misato as PM
    feats = {"A": np.ones((6, 2)), "B": np.ones((3, 2)), "C": np.ones((7, 2))}
    labels = {"A": np.arange(6.), "B": np.arange(3.), "C": np.arange(6.)}
    split = {"A": "train", "B": "val", "C": "test"}
    with pytest.raises(PM.ResidueCoverageError):       # C: 7 vs 6 (was cut to 6)
        PM.gather_residues(feats, labels, split)
    feats["C"] = np.ones((6, 2))
    b = PM.gather_residues(feats, labels, split)        # legacy: B (< 5 residues) skipped
    assert b["val"][0] is None and len(b["test"][2]) == 6
    rs = {"proteins": {"A": {"length": 6}, "D": {"length": 4}}}
    with pytest.raises(PM.ResidueCoverageError):
        PM.gather_residues(feats, labels, split, residue_set=rs)


def test_misato_coords_hash_recomputed(tmp_path, capsys):
    import probe_rmsf_misato as PM
    from ensembits.data_versions import hash_arrays_by_pid
    coords = {"A": np.ones((8, 4, 3), np.float32)}
    cp = tmp_path / "coords_ca8_misato.pkl"; cp.write_bytes(pickle.dumps(coords))
    Path(str(cp) + ".provenance.json").write_text(json.dumps({"coords_sha256": "f" * 64}))
    lab = tmp_path / "misato_rmsf.npz"; np.savez(lab, A=np.ones(4))
    side = Path(str(lab) + ".provenance.json")
    side.write_text(json.dumps({"coords_sha256": "f" * 64}))   # matches only the stale sidecar
    with pytest.raises(ValueError):
        PM.check_label_provenance(lab, cp)
    assert "stale sidecar" in capsys.readouterr().out
    side.write_text(json.dumps({"coords_sha256": hash_arrays_by_pid(coords)}))
    assert PM.check_label_provenance(lab, cp) == "match"


# ---------------------------------------------------------------- label builders
def test_allframes_label_of_repeated_frames_equals_label_of_frames():
    from build_mdcath_div_rmsf_labels import converged_rmsf, kabsch_onto
    rng = np.random.default_rng(5)
    base = np.cumsum(rng.normal(size=(30, 3)) * 2, 0)
    X = np.stack([base + rng.normal(size=(30, 3)) * 0.4 for _ in range(7)])
    r1, _, ok = converged_rmsf(X)
    rK, _, okK = converged_rmsf(np.concatenate([X] * 4))
    assert ok and okK
    np.testing.assert_allclose(r1, rK, atol=1e-6)
    # batched Kabsch == the per-frame definition
    P = X[1] @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T + 3.0
    Pc = P - P.mean(0); Qc = X[0] - X[0].mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc); d = np.sign(np.linalg.det(Vt.T @ U.T))
    ref = Pc @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T + X[0].mean(0)
    np.testing.assert_allclose(kabsch_onto(P, X[0]), ref, atol=1e-9)


def _fake_mdcath_h5(path, dom, frames_by_rep, n_extra_atoms=2):
    import h5py
    L = frames_by_rep[0].shape[1]
    lines = []
    atom = 0
    for i in range(L):
        for name in ("N", "CA", "C", "O"):
            lines.append(f"ATOM  {atom + 1:5d}  {name:<3s} ALA A{i + 1:4d}    "); atom += 1
    with h5py.File(path, "w") as f:
        g = f.create_group(dom)
        g["pdbProteinAtoms"] = "\n".join(lines)
        for rep, fr in enumerate(frames_by_rep):
            full = np.zeros((fr.shape[0], 4 * L, 3), np.float32)
            full[:, 1::4] = fr                       # CA is the 2nd atom of each residue
            g.create_dataset(f"320/{rep}/coords", data=full)


def test_mdcath_allframes_builder(tmp_path):
    import build_mdcath_div_rmsf_labels as B
    rng = np.random.default_rng(7)
    src = tmp_path / "src"; src.mkdir(); h5d = tmp_path / "h5"; h5d.mkdir()
    pids, alls = ["x1", "x2"], {}
    for p in pids:
        reps = [rng.normal(size=(23, 9, 3)).astype(np.float32) for _ in range(3)]
        _fake_mdcath_h5(h5d / f"mdcath_dataset_{p}.h5", p, reps)
        a = np.concatenate(reps); alls[p] = a
        pick = a[::7][:10]
        np.savez(src / f"{p}.npz", ca_5=pick[:5], ca_10=pick)
    (src / "_manifest.json").write_text(json.dumps({"representatives": {p: {} for p in pids}}))
    out = tmp_path / "lab.npy"
    assert B.main(["--src", str(src), "--h5-dir", str(h5d), "--out", str(out), "--n-rows", "18"]) == 0
    lab = np.load(out)
    np.testing.assert_allclose(lab[:9], B.converged_rmsf(alls["x1"])[0], atol=1e-5)
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["label_name"] == "allframes" and prov["n_frames_per_pid"]["x2"] == 69
    assert B.main(["--src", str(src), "--h5-dir", str(h5d), "--out", str(out), "--n-rows", "18"]) == 1
    # a stored ca_10 frame that is not among the raw frames fails the build
    np.savez(src / "x2.npz", ca_5=alls["x2"][:5], ca_10=alls["x2"][:10] + 0.5)
    assert B.main(["--src", str(src), "--h5-dir", str(h5d), "--out", str(tmp_path / "l2.npy"),
                   "--n-rows", "18"]) == 1


def test_misato_allframes_one(tmp_path, monkeypatch):
    import h5py
    import build_misato_real_bb as BB
    import build_misato_rmsf_labels as ML
    from build_mdcath_div_rmsf_labels import converged_rmsf
    rng = np.random.default_rng(3)
    traj = rng.normal(size=(100, 20, 3))
    with h5py.File(tmp_path / "MD.hdf5", "w") as f:
        f.create_dataset("PX/trajectory_coordinates", data=traj)
    ca_idx = np.arange(1, 20, 2)
    monkeypatch.setattr(BB, "parse_complex", lambda g: {"ca": ca_idx})
    rb = tmp_path / "rb"; rb.mkdir()
    ch = np.array([0, 5, 9, 20, 33, 50, 70, 99])
    np.savez(rb / "PX.npz", ca_8=traj[ch][:, ca_idx].astype(np.float32), chosen_indices=ch)
    pid, r, info = ML.allframes_one("PX", str(tmp_path / "MD.hdf5"), str(rb))
    assert info["checked_real_bb"] and info["n_frames"] == 100
    np.testing.assert_allclose(r, converged_rmsf(traj[:, ca_idx])[0], atol=1e-6)
    np.savez(rb / "PX.npz", ca_8=traj[ch][:, ca_idx].astype(np.float32) + 1, chosen_indices=ch)
    assert ML.allframes_one("PX", str(tmp_path / "MD.hdf5"), str(rb))[1] is None


# ---------------------------------------------------------------- compilers / selection
def _rm_json(v=0.6, label="allframes", rs="a" * 64, ck="b" * 64, variant="newrun"):
    return {"test_spearman": v, "test_spearman_per_protein_mean": v - 0.2,
            "best_val_spearman": v + 0.01, "val_spearman_per_protein_mean": v - 0.19,
            "rmsf_label": label, "residue_set_sha256": rs, "rmsf_label_sha256": "e" * 64,
            "feature_caches": [], "cache_verified": True, "splits_sha256": "f" * 64,
            "max_epochs": 500, "patience": 20, "min_delta": 1e-4, "batch_size": 2048, "selection": "val_early_stopping",   # K2: the probes' defaults
            "checkpoint": {"variant": variant, "best_pt_sha256": ck}}


def _pg_fixture(root):
    pg = root / "ensembits-repro" / "runs" / "proteingym"; pg.mkdir(parents=True)
    for n in ("esm3struct", "aminoaseed", "protoken", "3di_tokens", "combined_esm3"):
        (pg / f"{n}.json").write_text(json.dumps({"n_assays": 96, "mean_spearman_ours": 0.3,
                                                  "mean_zmix_a03": 0.5, "mean_spearman_esm2": 0.48}))


def test_compiler_refuses_mixed_rmsf_rows(tmp_path):
    _pg_fixture(tmp_path)
    rm = tmp_path / "output" / "probe_rmsf_div"; rm.mkdir(parents=True)
    for s in range(3):
        suf = "" if s == 0 else f"_seed{s}"
        (rm / f"newrun{suf}__test.json").write_text(json.dumps(_rm_json(0.6 + 0.01 * s)))
        (rm / f"baseline_aa{suf}__test.json").write_text(json.dumps(
            {k: v for k, v in _rm_json(0.3).items() if k != "checkpoint"}))
    out = tmp_path / "res.md"; cam = tmp_path / "cam.md"
    # H6: 3 of 10 seeds -> refused unless --allow-incomplete (preview)
    r = _run(tmp_path, "_compile_rebuttal_results", ["--variant", "newrun", "--out", str(out)])
    assert r.returncode == 1 and "lack seeds" in r.stderr and "✗ (3)" in out.read_text()
    r = _run(tmp_path, "_compile_rebuttal_results", ["--variant", "newrun", "--out", str(out),
                                                     "--allow-incomplete"])
    assert r.returncode == 0, r.stderr
    md = out.read_text()
    assert "## Per-protein RMSF" in md and "0.410±0.010" in md        # per-protein mean column
    assert "✗" not in md
    r = _run(tmp_path, "compile_camera_ready_tables", ["--variant", "newrun", "--src", str(out),
                                                       "--out", str(cam)])
    assert r.returncode == 1 and "Not written" in r.stderr         # H6: (k) cells, missing rows
    r = _run(tmp_path, "compile_camera_ready_tables", ["--variant", "newrun", "--src", str(out),
                                                       "--out", str(cam), "--allow-incomplete"])
    assert r.returncode == 0, r.stderr
    assert "Per-protein RMSF" in cam.read_text()
    # one seed of the baseline on another residue set -> the whole cell is refused
    (rm / "baseline_aa_seed2__test.json").write_text(json.dumps(
        {k: v for k, v in _rm_json(0.3, rs="c" * 64).items() if k != "checkpoint"}))
    # one seed of the model with another checkpoint
    (rm / "newrun_seed1__test.json").write_text(json.dumps(_rm_json(0.61, ck="d" * 64)))
    # H2: a recorded input cache that changed on disk refuses the cell too
    tracked = tmp_path / "tracked.npz"; np.savez(tracked, a=np.arange(3))
    js = _rm_json(0.62); js["feature_caches"] = [RC.file_record(tracked)]
    js["feature_caches"][0]["path"] = str(tracked)
    for s_ in range(3):
        j2 = dict(js, test_spearman=0.3 + 0.01 * s_); j2.pop("checkpoint")
        (rm / f"baseline_random{'' if s_ == 0 else f'_seed{s_}'}__test.json").write_text(json.dumps(j2))
    np.savez(tracked, a=np.arange(4))
    r = _run(tmp_path, "_compile_rebuttal_results", ["--variant", "newrun", "--out", str(out),
                                                     "--allow-incomplete"])
    assert r.returncode == 1 and "refused" in r.stderr
    assert "tracked.npz changed" in out.read_text()
    md = out.read_text()
    assert "## Refused RMSF cells" in md and "| aa | ✗" in md.replace("**", "")
    r = _run(tmp_path, "compile_camera_ready_tables", ["--variant", "newrun", "--src", str(out),
                                                       "--out", str(cam)])
    assert r.returncode == 1


def test_selection_table_validation_only(tmp_path):
    v = _link_ckpt(tmp_path, "selrun")
    code = ("import sys; sys.path.insert(0, %r); from ensembits.featurize import checkpoint_provenance; "
            "import json; print(json.dumps(checkpoint_provenance(%r)))"
            % (str(CODE_ROOT), str(tmp_path / "output" / v)))
    ck = json.loads(subprocess.run([sys.executable, "-c", code], env=_env(tmp_path),
                                   capture_output=True, text=True).stdout)
    rm = tmp_path / "output" / "probe_rmsf_div"; rm.mkdir(parents=True)
    for s in range(3):
        for suf in ("", "_P1inf"):
            js = _rm_json(0.6 + 0.01 * s); js["checkpoint"] = ck; js["test_spearman"] = 0.987
            (rm / f"{v}{suf}{'' if s == 0 else f'_seed{s}'}__test.json").write_text(json.dumps(js))
    out = tmp_path / "sel.md"
    r = _run(tmp_path, "rmsf_selection_table", ["--variants", v, "--out", str(out)])
    assert r.returncode == 1 and "missing seeds [3, 4, 5, 6, 7, 8, 9]" in r.stderr   # F10
    r = _run(tmp_path, "rmsf_selection_table", ["--variants", v, "--seeds", "0-2", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    md = out.read_text()
    assert "0.620±0.010 (n=3)" in md and "0.987" not in md and "best val recon" in md
    # G9: a current residue-set file with another hash refuses the table
    rsf = tmp_path / "data" / "cached_descriptors" / "rmsf_residue_set_mdcath_div.json"
    rsf.parent.mkdir(parents=True, exist_ok=True)
    rsf.write_text(json.dumps(RC.make_residue_set("mdcath_div", {"z": {"row_start": 0, "length": 1}}, "r")))
    r = _run(tmp_path, "rmsf_selection_table", ["--variants", v, "--seeds", "0-2"])
    assert r.returncode == 1 and "residue set" in r.stderr
    rsf.unlink()
    js["rmsf_label"] = "converged5"
    (rm / f"{v}_seed2__test.json").write_text(json.dumps(js))
    r = _run(tmp_path, "rmsf_selection_table", ["--variants", v, "--seeds", "0-2"])
    assert r.returncode == 1 and "rmsf_label" in r.stderr


def test_builder_failure_tolerance(tmp_path, capsys):
    # F7: failures of proteins in the RMSF set fail the build (default 0); outside: reported
    prot = {"a": {"row_start": None, "length": 3}, "b": {"row_start": None, "length": 3}}
    rs = RC.make_residue_set("misato", prot, "rule")
    code = ("import sys, json; sys.path.insert(0, %r); import rmsf_common as RC; "
            "print(json.dumps([RC.check_builder_failures({'c': 'x'}, ['a', 'b', 'c'], 'misato'), "
            "RC.check_builder_failures({'a': 'x'}, ['a', 'b'], 'misato'), "
            "RC.check_builder_failures({'a': 'x'}, ['a', 'b'], 'misato', 0.5), "
            "RC.check_builder_failures({}, ['a'], 'misato'), "
            "RC.check_builder_failures({}, ['a'], 'misato', only=['a'])]))") % str(DOWNSTREAM)
    (tmp_path / "data" / "cached_descriptors").mkdir(parents=True)
    (tmp_path / "data" / "cached_descriptors" / "rmsf_residue_set_misato.json").write_text(json.dumps(rs))
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [True, False, True, False, True]
    assert "FAILED b: no input" in r.stdout


def test_cache_provenance_and_token_ids(tmp_path):
    # F1 / H4: a baseline cache is read only with a current, complete sidecar; F4: no clipping
    code = ("import sys, numpy as np; sys.path.insert(0, %r); import rmsf_common as RC\n"
            "from ensembits.data_versions import MISATO_PARSER_VERSION as V\n"
            "c = RC.CACHE / 'mini3di_tokens_misato.npz'; c.parent.mkdir(parents=True, exist_ok=True)\n"
            "np.savez(c, A=np.arange(5))\n"
            "def bad(f):\n"
            "    try: f()\n"
            "    except RC.CacheError: return True\n"
            "    return False\n"
            "chk = lambda ds='misato', fr='frame 0', seq=False: RC.check_cache_provenance(c, ds, frames=fr, uses_sequence=seq)\n"
            "good = {'parser_version': [V], 'K': [8], 'fps': ['kabsch'], 'scan': 'all files'}\n"
            "import json; RC.MISATO_CHAIN_BREAKS.write_text(json.dumps({'schema': 'misato-chain-breaks-v2', 'complexes': {}}))\n"
            "w = lambda **k: RC.write_cache_provenance(c, **{**dict(dataset='misato', builder='t', inputs=good, n_proteins=1, frames='frame 0', **RC.split_record('misato'), sequence_source=RC.SEQ_NOT_USED), **k})\n"
            "assert bad(chk)                                                # no sidecar\n"
            "w(inputs={**good, 'parser_version': ['old']}); assert bad(chk)  # old parser\n"
            "w(inputs={**good, 'K': [10]}); assert bad(chk)                  # K10 backbones\n"
            "w(inputs={**good, 'fps': ['frobenius']}); assert bad(chk)       # other FPS rule\n"
            "w(chain_split=None); assert bad(chk)                            # chain split not recorded\n"
            "w(chain_split=None, chain_break_A=2.5); assert bad(chk)         # 2026-10-18: old 2.5 A split refused\n"
            "w(chain_split='whole protein (single chain)'); assert bad(chk)  # not the MISATO split\n"
            "w(chain_split='chain_index'); assert bad(chk)                   # chain_index runs (2026-10-18 morning)\n"
            "w(chain_split='true_chain_v1'); assert bad(chk)                 # v1 table (2026-10-18)\n"
            "w(chain_breaks={'sha256': 'x'}); assert bad(chk)                # another chain-break table\n"
            "w(sequence_source=None); assert bad(chk)                        # sequence source not recorded\n"
            "w(); assert bad(lambda: chk(fr='all 8'))                        # frames of another feature\n"
            "assert bad(lambda: chk(seq=True))                                # feature that needs a sequence\n"
            "rec = chk(); assert rec['sha256'] and rec['path'] and rec['chain_split'] == 'true_chain_v2'\n"
            "assert bad(lambda: chk(ds='bioemu'))                            # other dataset\n"
            "np.savez(c, A=np.arange(6))\n"
            "assert bad(chk)                                                 # changed after sidecar\n"
            "assert RC.records_changed([rec]) == ['mini3di_tokens_misato.npz changed (sha256)']\n"
            "assert bad(lambda: RC.check_token_ids(np.array([0, 5]), 5))\n"
            "assert RC.check_token_ids(np.array([0, 4]), 5).tolist() == [0, 4]\n"
            "print('OK')") % str(DOWNSTREAM)
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


def test_per_protein_refuses_nan():
    with pytest.raises(ValueError):
        RC.per_protein_spearman(np.array([np.nan] + [1.0] * 11), np.arange(12.), np.zeros(12))


# ---------------------------------------------------------------- runners / snapshots (F8, F9)
def test_battery_tasks_dry_run(tmp_path):
    (tmp_path / "output").mkdir()
    code = ("import sys; sys.argv = ['b', '--variant', 'xrun', '--tasks', 'RMDIV,RMBIO', '--dry-run', "
            "'--gpus', '0']; sys.path.insert(0, %r); import _run_model_battery as B; B.main()") % str(DOWNSTREAM)
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "[downstream] total=40" in r.stdout and "--dataset bioemu" in r.stdout
    assert "tokenize_misato_and_eval" not in r.stdout and "[pg] skipped" in r.stdout
    assert "[compile] skipped" in r.stdout
    assert sorted(p.name for p in (tmp_path / "output").iterdir()) == []      # nothing written


def test_snapshot_checkpoint(tmp_path):
    import snapshot_checkpoint as SC
    run = tmp_path / "output" / "runA"; run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"weights")
    (run / "config.json").write_text(json.dumps({"status": "running", "code_sha256": "c" * 64}))
    np.savez(run / "stats.npz", mean=np.zeros(1))
    # epoch 3 improves by < 1e-5 over epoch 2, so best.pt is from epoch 2 (trainer's rule)
    (run / "history.json").write_text(json.dumps({"eval_epochs": [1, 2, 3],
                                                  "val_recon": [1.0, 0.5, 0.499995]}))
    dest = SC.snapshot(run, tmp_path / "output")
    assert dest.name == "snap_runA_ep2"
    info = RC.snapshot_info(dest)
    assert info["snapshot_of"] == "runA" and info["best_epoch"] == 2
    assert (dest / "best.pt").read_bytes() == b"weights" and (dest / "stats.npz").exists()
    with pytest.raises(FileExistsError):
        SC.snapshot(run, tmp_path / "output")


# ---------------------------------------------------------------- hardening (H1, H3, H5, H7)
def test_misato_variant_token_cache_inputs(tmp_path):
    # H1: the Ensembits MISATO token cache needs a sidecar with current-parser K8 Kabsch inputs
    _fake_misato(tmp_path)
    assert _build_set(tmp_path, "misato").returncode == 0
    v = _link_ckpt(tmp_path, "mrun")
    cache = tmp_path / "data" / "cached_descriptors" / "ensembits_mrun_P1inf_tokens_misato.npz"
    rng = np.random.default_rng(0)
    np.savez(cache, **{p: rng.integers(0, 2048, 11) for p in MIS})
    code = ("import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "from ensembits.featurize import checkpoint_provenance, write_provenance_sidecar\n"
            "from ensembits.data_versions import MISATO_PARSER_VERSION as V\n"
            "from rmsf_common import ROOT\n"
            "from ensembits.featurize import file_sha256\n"
            "inp = {'parser_version': [V], 'K': [int(sys.argv[1])], 'fps': ['kabsch'], 'scan': 'all files'}\n"
            "write_provenance_sidecar(%r, checkpoint_provenance(ROOT / 'output' / 'mrun'), inputs=inp,"
            " frames='first 1 of bb_8', cache_sha256=file_sha256(%r))\n") % (
                str(CODE_ROOT), str(DOWNSTREAM), str(cache), str(cache))
    args = ["--mode", "variant", "--variant", v, "--p-inference", "1", "--device", "cpu",
            "--max-epochs", "2", "--patience", "1"]
    r = _run(tmp_path, "probe_rmsf_misato", args)                       # no sidecar
    assert r.returncode == 1 and "no provenance sidecar" in r.stderr
    subprocess.run([sys.executable, "-c", code, "10"], env=_env(tmp_path), check=True)
    r = _run(tmp_path, "probe_rmsf_misato", args)                       # K10 inputs
    assert r.returncode == 1 and "K=[10]" in r.stderr
    subprocess.run([sys.executable, "-c", code, "8"], env=_env(tmp_path), check=True)
    r = _run(tmp_path, "probe_rmsf_misato", args)
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "output" / "probe_rmsf_misato" /
                     "mrun_P1inf_structure__test.json").read_text())
    assert js["feature_caches"][0]["inputs"]["K"] == [8] and js["feature_caches"][0]["sha256"]
    # G8: a cache changed after its sidecar, and a cache of another checkpoint, are clean errors
    np.savez(cache, **{p: rng.integers(0, 2048, 11) for p in MIS})
    r = _run(tmp_path, "probe_rmsf_misato", args + ["--seed", "1"])
    assert r.returncode == 1 and "changed after its sidecar" in r.stderr
    subprocess.run([sys.executable, "-c", code, "8"], env=_env(tmp_path), check=True)
    sc = Path(str(cache) + ".provenance.json"); d = json.loads(sc.read_text())
    d["checkpoint"]["best_pt_sha256"] = "0" * 64; sc.write_text(json.dumps(d))
    r = _run(tmp_path, "probe_rmsf_misato", args + ["--seed", "2"])
    assert r.returncode == 1 and "another checkpoint" in r.stderr and "Traceback" not in r.stderr


def test_merge_protoken_shards_sidecars(tmp_path):
    # H3: shards must agree, match their sidecars and not share pids
    sd = tmp_path / "data" / "cached_descriptors" / "protoken_shards"; sd.mkdir(parents=True)
    cb = tmp_path / "data" / "cached_descriptors" / "protoken_codebook.npy"; np.save(cb, np.zeros((4, 2)))
    base = dict(dataset="mdcath", builder="t", inputs={}, frames="frame 0",
                chain_split=RC.chain_split_for("mdcath"),
                sequence_source=RC.SEQUENCE_SOURCES["mdcath_div"], codebook=cb)

    def shard(k, pids, **over):
        f = sd / f"mdcath_shard{k}of2.npz"; np.savez(f, **{p: np.arange(3) for p in pids})
        RC.write_cache_provenance(f, **{**base, **over}, n_proteins=len(pids), sharded=True,
                                  attempted=pids)

    def merge():
        out = tmp_path / "data" / "cached_descriptors" / "protoken_tokens_mdcath.npz"
        if out.exists(): out.unlink()
        return _run(tmp_path, "merge_protoken_shards", ["--datasets", "mdcath", "--num-shards", "2"],
                    DATA_PREP / "baselines")
    shard(0, ["a", "b"]); shard(1, ["c"])
    r = merge(); assert r.returncode == 0, r.stdout + r.stderr
    side = json.loads((tmp_path / "data" / "cached_descriptors" /
                       "protoken_tokens_mdcath.npz.provenance.json").read_text())
    assert side["n_proteins"] == 3 and side["frames"] == "frame 0"
    assert side["chain_split"] == "whole protein (single chain)"
    shard(1, ["c"], chain_split="chain_index")
    r = merge(); assert r.returncode == 1 and "disagree on chain_split" in r.stdout
    shard(0, ["a", "b"], chain_split=None, chain_break_A=2.5)        # 2026-10-18: old shards
    shard(1, ["c"], chain_split=None, chain_break_A=2.5)
    r = merge(); assert r.returncode == 1 and "expected" in r.stdout and "chain_split" in r.stdout
    shard(0, ["a", "b"])
    shard(1, ["b"])
    r = merge(); assert r.returncode == 1 and "pid b in both" in r.stdout
    shard(1, ["c"]); np.savez(sd / "mdcath_shard1of2.npz", c=np.arange(4))
    r = merge(); assert r.returncode == 1 and "changed after its sidecar" in r.stdout


def test_snapshot_retries_when_history_moves(tmp_path, monkeypatch):
    # H5: train.py writes best.pt before history.json; a copy in between is retried
    import snapshot_checkpoint as SC
    run = tmp_path / "output" / "runB"; run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"w1")
    (run / "config.json").write_text(json.dumps({"status": "running"}))
    hist = run / "history.json"
    hist.write_text(json.dumps({"eval_epochs": [1, 2], "val_recon": [1.0, 0.5]}))
    real_copy, calls = SC.shutil.copy2, {"n": 0}

    def copy2(src, dst):
        real_copy(src, dst)
        if Path(src).name == "best.pt" and calls["n"] == 0:       # a new best epoch lands
            calls["n"] += 1
            hist.write_text(json.dumps({"eval_epochs": [1, 2, 3], "val_recon": [1.0, 0.5, 0.4]}))
    monkeypatch.setattr(SC.shutil, "copy2", copy2)
    dest = SC.snapshot(run, tmp_path / "output", wait_s=0)
    info = RC.snapshot_info(dest)
    assert dest.name == "snap_runB_ep3" and info["best_epoch"] == 3 and info["attempts"] == 2


def test_unfinished_checkpoint_clean_error(tmp_path):
    # H7: an unfinished checkpoint gives one [error] line pointing at the two ways out
    _fake_mdcath(tmp_path)
    assert _build_set(tmp_path, "mdcath_div", ["--allow-missing"]).returncode == 0
    d = tmp_path / "output" / "unfin"; d.mkdir(parents=True)
    for f in ("best.pt", "stats.npz", "history.json"):
        os.symlink(CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10" / f, d / f)
    cfg = json.loads((CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10" / "config.json").read_text())
    (d / "config.json").write_text(json.dumps({**cfg, "status": "running", "finished_at": None}))
    r = _run(tmp_path, "probe_rmsf_eval", ["--variant", "unfin", "--device", "cpu"])
    assert r.returncode == 1 and "[error]" in r.stdout and "snapshot_checkpoint.py" in r.stdout
    assert "Traceback" not in r.stderr


# ---------------------------------------------------------------- audit G items (2026-10-17)
def test_eval_descriptor_and_label_provenance(tmp_path):
    # G1: the descriptor-cache sidecar is required and must match; no silent P fallback
    pids, total = _fake_mdcath(tmp_path)
    assert _build_set(tmp_path, "mdcath_div", ["--allow-missing"]).returncode == 0
    v = _link_ckpt(tmp_path)
    cd = tmp_path / "output" / "phase_b" / "cached_desc"
    np.save(cd / "esm3desc_dihedral_k16_P10.npy",
            np.random.default_rng(1).normal(size=(total, 10, 256)).astype(np.float32))
    args = ["--variant", v, "--device", "cpu", "--max-epochs", "2", "--patience", "1"]
    r = _run(tmp_path, "probe_rmsf_eval", args)
    assert r.returncode == 1 and "no provenance sidecar" in r.stdout
    side = cd / "esm3desc_dihedral_k16_P10.npy.provenance.json"
    side.write_text(json.dumps({"dataset": "mdcath", "n_rows": total, "IDEAL_PIDS": ["d1"]}))
    r = _run(tmp_path, "probe_rmsf_eval", args)
    assert r.returncode == 1 and "ideal geometry" in r.stdout
    side.write_text(json.dumps({"dataset": "bioemu", "n_rows": total}))
    r = _run(tmp_path, "probe_rmsf_eval", args)
    assert r.returncode == 1 and "built for 'bioemu'" in r.stdout
    side.write_text(json.dumps({"dataset": "mdcath", "n_rows": total}))
    (cd / "rmsf_div_allframes.npy.provenance.json").unlink()
    r = _run(tmp_path, "probe_rmsf_eval", args)
    assert r.returncode == 1 and "rmsf_div_allframes.npy has no provenance sidecar" in r.stdout
    (cd / "esm3desc_dihedral_k16_P10.npy").rename(cd / "esm3desc_dihedral_k16_P5.npy")
    r = _run(tmp_path, "probe_rmsf_eval", args)
    assert r.returncode == 1 and "--desc-p" in r.stdout       # no silent fallback to P5


def test_split_and_settings_checks(tmp_path):
    # G4: one protein in one split; BioEmu split built from the current layout
    with pytest.raises(RC.ResidueCoverageError):
        RC.check_one_split(np.array([0, 0, 1, 1]), {"train": [0], "val": [1, 2], "test": [3]})
    RC.check_one_split(np.array([0, 0, 1, 1]), {"train": [0, 1], "val": [2, 3], "test": []})
    rows, total, cath = __import__("test_rmsf_bioemu")._fake_bioemu_repo(tmp_path)
    assert _run(tmp_path, "build_bioemu_rmsf_splits", ["--cath-list", str(cath)], DATA_PREP).returncode == 0
    code = ("import sys; sys.path.insert(0, %r); import rmsf_common as RC\n"
            "RC.check_bioemu_split_layout()\n"
            "p = RC.BIOEMU_LAYOUT; p.write_text(p.read_text().replace('\"L\": 13', '\"L\": 13 '))\n"
            "try:\n    RC.check_bioemu_split_layout(); print('NOT REFUSED')\n"
            "except RC.CacheError: print('REFUSED')\n") % str(DOWNSTREAM)
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert "REFUSED" in r.stdout and "NOT" not in r.stdout, r.stderr
    # G4 MISATO: a pid in two parts of a split
    _fake_misato(tmp_path)
    sp = json.loads((tmp_path / "data" / "misato_splits_v2.json").read_text())
    sp["structure"]["val"].append(sp["structure"]["train"][0])
    (tmp_path / "data" / "misato_splits_v2.json").write_text(json.dumps(sp))
    assert _build_set(tmp_path, "misato").returncode == 0
    r = _run(tmp_path, "probe_rmsf_misato", ["--mode", "aa", "--device", "cpu", "--max-epochs", "2"])
    assert r.returncode == 1 and "is in train and val" in r.stderr


def test_cache_records_ideal_and_codebook(tmp_path):
    # G7 ideal frames inside the set; G11 codebook re-check, path-less records, unverified;
    # G12 identical codebook kept
    code = ("import sys, json, numpy as np; sys.path.insert(0, %r); import rmsf_common as RC\n"
            "from ensembits.data_versions import MISATO_PARSER_VERSION as V\n"
            "c = RC.CACHE / 'protoken_tokens_mdcath.npz'; c.parent.mkdir(parents=True, exist_ok=True)\n"
            "cb = RC.CACHE / 'protoken_codebook.npy'\n"
            "assert RC.write_codebook(cb, np.zeros((4, 2), np.float32)) is True\n"
            "m = cb.stat().st_mtime_ns\n"
            "assert RC.write_codebook(cb, np.zeros((4, 2), np.float32)) is False and cb.stat().st_mtime_ns == m\n"
            "np.savez(c, d1=np.arange(3))\n"
            "kw = dict(dataset='mdcath', builder='t', inputs={}, n_proteins=1, frames='frame 0',"
            " chain_break_A=2.5, sequence_source=RC.SEQ_NOT_USED, codebook=cb)\n"
            "RC.write_cache_provenance(c, **kw, n_ideal_frames=1, ideal_pids=['d8'])\n"
            "rec = RC.check_cache_provenance(c, 'mdcath', frames='frame 0', codebook=cb, required={'d1'})\n"
            "try:\n    RC.check_cache_provenance(c, 'mdcath', frames='frame 0', codebook=cb, required={'d8'}); print('BAD1')\n"
            "except RC.CacheError: pass\n"
            "assert RC.records_changed([rec]) == []\n"
            "RC.write_codebook(cb, np.ones((4, 2), np.float32))\n"
            "assert 'changed' in RC.records_changed([rec])[0]\n"
            "assert RC.records_changed([{'file': 'x'}]) == ['x: record cannot be re-checked (no path)']\n"
            "o = RC.ROOT / 'output' / 'probe_rmsf_div'; o.mkdir(parents=True)\n"
            "(o / 'r.json').write_text(json.dumps({'residue_set_sha256': 'a', 'rmsf_label': 'allframes', 'cache_verified': False}))\n"
            "print(RC.stale_rmsf_outputs([o / 'r.json'], 'mdcath_div'))\n") % str(DOWNSTREAM)
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0 and "BAD" not in r.stdout, r.stderr
    assert "not verified" in r.stdout


def test_misato_label_builder_lists_failures(tmp_path, monkeypatch, capsys):
    # G3: default tolerance 0, failures listed
    import build_misato_rmsf_labels as ML
    monkeypatch.setattr(ML, "allframes_one", lambda pid, h5, rb, *a: (
        (pid, None, "parse error") if pid == "P1" else (pid, np.ones(3, np.float32),
                                                         {"n_frames": 100, "iter": 1,
                                                          "checked_real_bb": False})))
    pids = tmp_path / "pids.txt"; pids.write_text("P0\nP1\nP2\n")
    out = tmp_path / "lab.npz"
    assert ML.main(["--pids", str(pids), "--out", str(out), "--h5", str(tmp_path / "x.h5")]) == 1
    assert "FAILED P1: parse error" in capsys.readouterr().out and not out.exists()
    assert ML.main(["--pids", str(pids), "--out", str(out), "--h5", str(tmp_path / "x.h5"),
                    "--max-fail-frac", "0.5"]) == 0 and out.exists()


def test_compiler_settings_proteingym_and_battery_names(tmp_path):
    # G5 settings identical across rows; G10 no ProteinGym inputs; G13 tag_of names
    rm = tmp_path / "output" / "probe_rmsf_div"; rm.mkdir(parents=True)
    for s_ in range(10):
        suf = "" if s_ == 0 else f"_seed{s_}"
        (rm / f"newrun{suf}__test.json").write_text(json.dumps(_rm_json(0.6)))
        j = {k: v for k, v in _rm_json(0.3).items() if k != "checkpoint"}
        j["max_epochs"] = 7                              # another setting than the model row
        (rm / f"baseline_aa{suf}__test.json").write_text(json.dumps(j))
    out = tmp_path / "res.md"
    r = _run(tmp_path, "_compile_rebuttal_results", ["--variant", "newrun", "--out", str(out),
                                                     "--allow-incomplete"])
    assert r.returncode == 1 and "differ from the defaults" in out.read_text()   # G5/K2 (aa refused)
    # A.5 fix: PG table removed in favour of pg_aggregate.py; check redirect text
    assert "pg_aggregate.py" in out.read_text()                             # redirect pointer
    import _run_model_battery as B
    names = [str(o) for (*_, o, c) in B.function_jobs("final_model_abc", B.RECIPE)]
    assert any("tok_abc" in n for n in names) and not any("final_model" in n for n in names)


def test_probes_reject_removed_options(tmp_path):
    # G14: side-experiment options without a builder are gone
    r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", "esm2"])
    assert r.returncode == 2 and "invalid choice" in r.stderr
    r = _run(tmp_path, "probe_rmsf_misato", ["--mode", "aminoaseed_ourdata"])
    assert r.returncode == 2 and "invalid choice" in r.stderr


# ---------------------------------------------------------------- K items (2026-10-17)
def test_current_split_file_and_default_settings(tmp_path):
    # K2: the defaults equal the probes' constants
    import probe_rmsf_eval as PE
    import probe_rmsf_misato as PM
    d = RC.DEFAULT_PROBE_SETTINGS
    assert (PE.MAX_EPOCHS, PE.PATIENCE, PE.MIN_DELTA) == (d["max_epochs"], d["patience"], d["min_delta"])
    assert (PM.MAX_EPOCHS, PM.PATIENCE, PM.MIN_DELTA) == (d["max_epochs"], d["patience"], d["min_delta"])
    # K1: a current split file exists -> results must carry its hash (compiler + stale check)
    sp = tmp_path / "output" / "full_train" / "cached_desc" / "splits.json"
    sp.parent.mkdir(parents=True); sp.write_text(json.dumps({"train": [0], "val": [1], "test": [2]}))
    code = ("import sys, json; sys.path.insert(0, %r); import rmsf_common as RC\n"
            "print(json.dumps(RC.current_splits_sha256('mdcath_div')))") % str(DOWNSTREAM)
    cur = json.loads(subprocess.run([sys.executable, "-c", code], env=_env(tmp_path),
                                    capture_output=True, text=True).stdout)
    rm = tmp_path / "output" / "probe_rmsf_div"; rm.mkdir(parents=True)
    for s_ in range(10):
        suf = "" if s_ == 0 else f"_seed{s_}"
        j = _rm_json(0.6); j["splits_sha256"] = cur
        j["max_epochs"] = 7                                  # K2: the model row deviates ...
        (rm / f"newrun{suf}__test.json").write_text(json.dumps(j))
        jb = {k: v for k, v in _rm_json(0.3).items() if k != "checkpoint"}
        jb["splits_sha256"] = cur                            # ... the baseline row does not
        (rm / f"baseline_aa{suf}__test.json").write_text(json.dumps(jb))
        jr = dict(jb); jr["splits_sha256"] = "0" * 64        # K1: another split file
        (rm / f"baseline_random{suf}__test.json").write_text(json.dumps(jr))
    out = tmp_path / "res.md"
    r = _run(tmp_path, "_compile_rebuttal_results", ["--variant", "newrun", "--out", str(out),
                                                     "--allow-incomplete"])
    md = out.read_text()
    refused = [l for l in md.splitlines() if l.startswith("| mdcath_div |")]
    assert any("| newrun |" in l and "differ from the defaults" in l for l in refused)
    assert any("| baseline_random |" in l and "current split file" in l for l in refused)
    assert not any("| baseline_aa |" in l for l in refused)            # the deviant row is refused
    code2 = ("import sys; sys.path.insert(0, %r); import rmsf_common as RC; from pathlib import Path\n"
             "print(RC.stale_rmsf_outputs([Path(%r)], 'mdcath_div'))") % (
                str(DOWNSTREAM), str(rm / "baseline_random__test.json"))
    r = subprocess.run([sys.executable, "-c", code2], env=_env(tmp_path), capture_output=True, text=True)
    assert "another split file" in r.stdout, r.stderr


def test_residue_set_refuses_label_with_failures(tmp_path):
    # K4: a MISATO label built with --max-fail-frac > 0 needs --allow-label-failures
    _fake_misato(tmp_path)
    side = tmp_path / "data" / "cached_descriptors" / "misato_rmsf_allframes.npz.provenance.json"
    d = json.loads(side.read_text()); d.update(max_fail_frac=0.01, n_failed=1); side.write_text(json.dumps(d))
    r = _build_set(tmp_path, "misato")
    assert r.returncode == 1 and "--allow-label-failures" in r.stdout
    r = _build_set(tmp_path, "misato", ["--allow-label-failures"])
    assert r.returncode == 0, r.stdout + r.stderr
