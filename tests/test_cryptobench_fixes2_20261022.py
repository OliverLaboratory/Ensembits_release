"""Tests of the CryptoBench fix round 2 (FIXLIST2_20261022.md items 1-9) and of the fixes that
FIXAUDIT FX1 F5 / FX2 F11 found untested.

What it does
------------
Synthetic structures only (random backbones); no CryptoBench run, no metric of any real model is
computed or read. Uses the retrained checkpoint folder output/ensembits_bm_lam0p25_rv read only
on CPU where an Ensembits job is exercised (skipped if absent). Covers:
  1  baseline fingerprint with builder code + weights (mini3di end to end, is_done);
  2  gate: --expect-checkpoint refuses baseline / pLM files; pLM model / layer / failure checks;
     relative --expect-checkpoint resolved against REPO_ROOT;
  3  conv standardisation: sd = 1 for zero-variance columns;
  4  runner build / mask done-checks with build code, release and mask-code fingerprints;
  5  cryptobench_tokenize.py: a due rerun moves the old file aside (also for another
     checkpoint, with --rerun-on-change), a manual run still refuses another checkpoint, exit 1
     on a failed chain, is_done;
  6  --gpus "" refused at once;
  7  ESM-2 layer option in the fingerprint, role "descriptive" in the results, completeness
     restricted to the plan, option routing per job;
  8  CCD parent lookup (4CF -> F, AME -> M, PIA falls back), gz reader; the gemmi end-to-end
     build test (move-aside, microheterogeneity) runs when CB_GEMMI_PYTHON names a python with
     gemmi;
  9  stats results moved aside before a rerun; stats_done freshness rule;
  featurize exit 1 on a level-0 mismatch.

Paper
-----
not described in the paper (infrastructure for the CryptoBench experiment)

How to run
----------
    python -m pytest tests/test_cryptobench_fixes2_20261022.py
    CB_GEMMI_PYTHON=<python with gemmi> python -m pytest tests/test_cryptobench_fixes2_20261022.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
CB = CODE_ROOT / "downstream" / "cryptobench"
for _p in (CODE_ROOT, CODE_ROOT / "downstream", CB):
    sys.path.insert(0, str(_p))

import cryptobench_common as CC  # noqa: E402
from ensembits.paths import REPO_ROOT  # noqa: E402

CKPT = REPO_ROOT / "output" / "ensembits_bm_lam0p25_rv"
needs_ckpt = pytest.mark.skipif(not (CKPT / "best.pt").exists(), reason="retrained checkpoint absent")


def helix(L, seed=0):
    """A rough backbone (N, CA, C) along a helix-like path, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(L) * 1.7
    ca = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(L)], 1)
    n = ca + np.array([-1.0, 0.4, -0.5]) + rng.normal(0, 0.05, (L, 3))
    c = ca + np.array([1.0, 0.3, 0.6]) + rng.normal(0, 0.05, (L, 3))
    return np.stack([n, ca, c], 1).astype(np.float32)


def make_structs(tmp, lengths=None, folds=None):
    """v4-style structure caches + manifest (meta build_version current)."""
    lengths = lengths or [24, 30, 26]
    folds = folds or ["train-0", "train-3", "test"][:len(lengths)]
    sd = tmp / "cryptobench_apo"; sd.mkdir(parents=True, exist_ok=True)
    ents = []
    for i, (L, f) in enumerate(zip(lengths, folds)):
        apo = f"9a{i:02d}"
        bb = helix(L, i)
        y = np.zeros(L, np.int8); y[:3] = 1
        np.savez(sd / f"{apo}_A.npz", bb=bb[None], y=y, seq=np.array("A" * L),
                 cb=np.full((L, 3), np.nan, np.float32), o=np.full((L, 3), np.nan, np.float32),
                 label_seq=np.arange(1, L + 1, dtype=np.int32), nc=np.zeros(L, np.int8),
                 auth=np.array([str(k) for k in range(L)]), icode=np.array([""] * L))
        ents.append({"apo": apo, "chain": "A", "fold": f, "status": "ok", "L": L})
    man = tmp / "apo_manifest.json"
    man.write_text(json.dumps({"entries": ents, "meta": {"build_version": CC.STRUCT_BUILD_VERSION}}))
    return man


def loc(tmp):
    return ["--run-tag", "t", "--structs-root", str(tmp), "--manifest-dir", str(tmp),
            "--tokens-dir", str(tmp), "--out-dir", str(tmp / "out")]


# ---------------------------------------------------------------- 1 baseline fingerprint
def test_baseline_fingerprint_has_builder_code_and_weights(tmp_path, monkeypatch):
    import cryptobench_tokenize_baselines as TB
    make_structs(tmp_path)
    argv = ["--tokenizer", "mini3di", *loc(tmp_path)]
    assert TB.main(argv) == 0 and TB.is_done(argv)
    side = json.loads((tmp_path / "cryptobench_tokens_mini3di_wc_apo.npz.provenance.json").read_text())
    assert "_real_or_ideal_bb.py" in side["builder_code_sha256"]
    assert "encoder.py" in side["weights_sha256"] and "encoder_weights_3di.kerasify" in side["weights_sha256"]
    # a changed builder file makes the output stale
    fake = tmp_path / "builder_copy.py"; fake.write_text("x = 1\n")
    monkeypatch.setitem(TB.BUILDER_FILES, "mini3di", [fake])
    assert not TB.is_done(argv)
    assert TB.main(argv) == 0 and TB.is_done(argv)
    assert list(tmp_path.glob("_pre_*/cryptobench_tokens_mini3di_wc_apo.npz"))
    for t in TB.TOKENIZERS:
        assert TB.BUILDER_FILES[t] and all(Path(f).exists() for f in TB.BUILDER_FILES[t])


def test_aminoaseed_weight_path_equals_builder():
    import cryptobench_tokenize_baselines as TB
    sys.path.insert(0, str(CODE_ROOT / "data_prep" / "baselines"))
    import build_aminoaseed_tokens as BA
    assert Path(BA.CKPT) == TB.AMINOASEED_CKPT


# ---------------------------------------------------------------- 2 gate
def test_gate_expect_checkpoint_refuses_baseline_and_plm(tmp_path):
    man = tmp_path / "m.json"; man.write_text("{}")
    p = tmp_path / "cryptobench_tokens_x.npz"; np.savez(p, a=np.zeros(1))
    side = p.with_name(p.name + ".provenance.json")
    ms = CC.file_sha256(man)
    side.write_text(json.dumps({"tokenizer": "mini3di", "chain_rule": CC.BASELINE_CHAIN_RULE,
                                "manifest_sha256": ms, "n_failed": 0}))
    assert CC.check_token_file(p, man)
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man, expect_checkpoint=CKPT)
    plm = {"tokenizer": "esm2_3b", "model": "facebook/esm2_t36_3B_UR50D", "layer": 33,
           "manifest_sha256": ms, "n_failed": 0}
    side.write_text(json.dumps(plm))
    assert CC.check_token_file(p, man, expect_layer=33)
    for bad in ({"model": "facebook/esm2_t33_650M_UR50D"}, {"layer": None}, {"n_failed": None}):
        side.write_text(json.dumps({**plm, **bad}))
        with pytest.raises(CC.TokenFileError):
            CC.check_token_file(p, man)
    side.write_text(json.dumps(plm))
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man, expect_layer=36)
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man, expect_checkpoint=CKPT)


@needs_ckpt
def test_gate_relative_expect_checkpoint(tmp_path):
    from ensembits.featurize import checkpoint_provenance
    man = tmp_path / "m.json"; man.write_text("{}")
    p = tmp_path / "cryptobench_tokens_e.npz"; np.savez(p, a=np.zeros(1))
    cfg = json.loads((CKPT / "config.json").read_text())
    p.with_name(p.name + ".provenance.json").write_text(json.dumps(
        {"checkpoint": checkpoint_provenance(CKPT), "manifest_sha256": CC.file_sha256(man),
         "n_failed": 0, "dihedral_convention": cfg.get("dihedral_convention"),
         "chain_break_cutoff": cfg.get("chain_break_cutoff")}))
    cwd = os.getcwd()
    try:
        os.chdir(CODE_ROOT)                    # not REPO_ROOT: the relative path must still resolve
        assert CC.check_token_file(p, man, expect_checkpoint="output/ensembits_bm_lam0p25_rv")
        with pytest.raises(CC.TokenFileError):
            CC.check_token_file(p, man, expect_checkpoint="output/ensembits_bm_lam1_rv")
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------- 3 conv zero variance
def test_conv_zero_variance_column_keeps_unit_sd():
    import cryptobench_probe as CP
    T = np.eye(21, dtype=np.float32)
    rows = [("a", np.arange(20), None)]                       # id 20 never in training
    mu, sd = CP.train_stats(rows, T)
    assert sd[20] == 1.0 and mu[20] == 0.0
    z = (T[20] - mu) / sd
    assert np.abs(z).max() < 10


# ---------------------------------------------------------------- 4 build / mask done-checks
def _runner_args(tmp, **kw):
    a = dict(run_tag="t", legacy_layout=False, cryptobench_dir=str(tmp / "rel"),
             structs_root=str(tmp), manifest_dir=str(tmp), tokens_dir=str(tmp),
             out_dir=str(tmp / "out"), gemmi_python="py", workers=1, seeds=5, smoke_epochs=None,
             stats_B=None, modified_residues="drop", esm2_row="on", esm2_layer=33,
             conv_standardize="train", selection="val_pooled_auroc", aa_order="collaborator")
    a.update(kw)
    return Namespace(**a)


def test_runner_build_and_mask_done_checks(tmp_path):
    import _cb_paths as CBP
    import run_cryptobench_rv as R
    rel = tmp_path / "rel"; rel.mkdir()
    for f in ("dataset.json", "folds.json", "noncryptic-pockets.json"):
        (rel / f).write_text("{}")
    args = _runner_args(tmp_path)
    paths = CBP.resolve_paths(args)
    meta = {"build_version": CC.STRUCT_BUILD_VERSION, "modified_residues": "drop",
            "build_code_sha256": CC.file_sha256(CB / "cryptobench_build_apo.py"),
            "release_sha256": R.release_sha256(paths), "ccd_sha256": None}
    man = tmp_path / "apo_manifest.json"
    man.write_text(json.dumps({"entries": [], "meta": meta}))
    jobs = {j.name: j for j in R.plan(args, paths)}
    assert jobs["build"].done()
    (rel / "folds.json").write_text('{"test": []}')              # release changed
    assert not jobs["build"].done()
    (rel / "folds.json").write_text("{}")
    man.write_text(json.dumps({"entries": [], "meta": {**meta, "build_code_sha256": "0" * 64}}))
    assert not jobs["build"].done()                              # build code changed
    man.write_text(json.dumps({"entries": [], "meta": meta}))
    mask = tmp_path / "apo_noncryptic_mask.json"
    mask.write_text(json.dumps({"manifest_sha256": CC.file_sha256(man),
                                "mask_code_sha256": CC.mask_code_sha256()}))
    assert jobs["mask"].done()
    mask.write_text(json.dumps({"manifest_sha256": CC.file_sha256(man), "mask_code_sha256": "x"}))
    assert not jobs["mask"].done()


# ---------------------------------------------------------------- 5 cryptobench_tokenize.py
@needs_ckpt
def test_tokenize_rerun_on_change_and_exit_codes(tmp_path):
    import cryptobench_tokenize as T
    from ensembits.featurize import ProvenanceMismatchError
    make_structs(tmp_path)
    base = ["--model-dir", str(CKPT), "--tag", "e", "--device", "cpu", *loc(tmp_path)]
    assert T.main(base) == 0 and T.is_done(base)
    side = tmp_path / "cryptobench_tokens_e.npz.provenance.json"
    d = json.loads(side.read_text())
    d["checkpoint"] = {**d["checkpoint"], "best_pt_sha256": "0" * 64}   # "another checkpoint"
    side.write_text(json.dumps(d))
    assert not T.is_done(base)
    with pytest.raises(ProvenanceMismatchError):                       # manual run: refused
        T.main(base)
    assert T.main(base + ["--rerun-on-change"]) == 0 and T.is_done(base)
    assert list(tmp_path.glob("_pre_*/cryptobench_tokens_e.npz"))
    # a failed chain gives exit code 1
    t2 = tmp_path / "b"; t2.mkdir()
    make_structs(t2, lengths=[24, 10, 26])                             # L = 10 < K = 16
    assert T.main(["--model-dir", str(CKPT), "--tag", "e", "--device", "cpu", *loc(t2)]) == 1


@needs_ckpt
def test_featurize_exit_1_on_level0_mismatch(tmp_path):
    import cryptobench_featurize as F
    import cryptobench_tokenize as T
    make_structs(tmp_path)
    assert T.main(["--model-dir", str(CKPT), "--tag", "e", "--device", "cpu", *loc(tmp_path)]) == 0
    argv = ["--model-dir", str(CKPT), "--tag", "e", "--tokens-tag", "e", "--device", "cpu",
            *loc(tmp_path)]
    assert F.main(argv) == 0 and F.is_done(argv)
    tok = tmp_path / "cryptobench_tokens_e.npz"
    t = dict(np.load(tok))
    k = sorted(t)[0]; t[k] = (t[k].astype(np.int64) + 1).astype(np.uint16)
    np.savez_compressed(tok, **t)                                      # tokens no longer match
    assert not F.is_done(argv)
    assert F.main(argv) == 1
    side = json.loads((tmp_path / "cryptobench_dense_e.npz.provenance.json").read_text())
    assert side["n_tokenfile_mismatch"] > 0
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(tmp_path / "cryptobench_dense_e.npz", tmp_path / "apo_manifest.json")


# ---------------------------------------------------------------- 6 empty --gpus
def test_empty_gpus_refused_at_once(tmp_path):
    import run_cryptobench_rv as R
    rc = R.main(["--gpus", "", "--gemmi-python", sys.executable, "--structs-root", str(tmp_path),
                 "--manifest-dir", str(tmp_path), "--tokens-dir", str(tmp_path),
                 "--out-dir", str(tmp_path / "o"), "--run-tag", "t"])
    assert rc == 2
    assert not (tmp_path / "o" / "logs").exists()


# ---------------------------------------------------------------- 7 ESM-2, completeness, routing
def test_esm2_layer_in_fingerprint(tmp_path):
    import cryptobench_esm2 as E
    man = make_structs(tmp_path)
    args = E.parser().parse_args(loc(tmp_path))
    assert args.layer == 33
    out, _, want = E.expected(args)
    assert want["layer"] == 33 and want["model"] == "facebook/esm2_t36_3B_UR50D"
    np.savez(out, a=np.zeros(1))
    out.with_name(out.name + ".provenance.json").write_text(json.dumps({**want, "n_entries": 3}))
    assert E.is_done(loc(tmp_path))
    assert not E.is_done(loc(tmp_path) + ["--layer", "36"])
    with pytest.raises(SystemExit):
        E.main(loc(tmp_path) + ["--layer", "40"])
    assert man.exists()


def test_mlp_esm2_row_is_marked_descriptive(tmp_path):
    import cryptobench_mlp_probe as MP
    man = make_structs(tmp_path, lengths=[24, 22, 26, 25, 23],
                       folds=["train-0", "train-1", "train-3", "test", "test"])
    rng = np.random.default_rng(0)
    m = json.loads(man.read_text())
    dense = {f"{e['apo']}_A|esm2": rng.normal(size=(e["L"], 6)).astype(np.float16)
             for e in m["entries"]}
    f = tmp_path / "cryptobench_dense_esm2_3b.npz"; np.savez(f, **dense)
    f.with_name(f.name + ".provenance.json").write_text(json.dumps(
        {"tokenizer": "esm2_3b", "model": "facebook/esm2_t36_3B_UR50D", "layer": 33,
         "manifest_sha256": CC.file_sha256(man), "n_failed": 0}))
    argv = ["--row", "esm2_3b", "--features", "dense", "--dense", "esm2_3b", "--readout", "esm2",
            "--expect-layer", "33", "--seeds", "1", "--epochs", "1", "--patience", "1",
            "--device", "cpu", *loc(tmp_path)]
    assert MP.main(argv) == 0
    out = tmp_path / "out" / "mlp"
    for p in (out / "esm2_3b__esm2.json", out / "esm2_3b__esm2__seed0.json"):
        assert json.loads(p.read_text())["role"].startswith("descriptive")
    bad = list(argv); bad[bad.index("--expect-layer") + 1] = "36"
    with pytest.raises(CC.TokenFileError):
        MP.main(bad + ["--redo"])


def test_option_routing_and_planned_completeness(tmp_path):
    import _cb_paths as CBP
    import run_cryptobench_rv as R
    args = _runner_args(tmp_path, esm2_layer=36, aa_order="alphabetical")
    jobs = R.plan(args, CBP.resolve_paths(args))
    for j in jobs:
        a = " ".join(j.argv)
        assert ("--standardize" in a) == j.name.startswith("conv_")
        assert ("--aa-order" in a) == j.name.endswith("aa__onehot")
        assert ("--layer 36" in a) == (j.name == "tok_esm2_3b")
        assert ("--expect-layer 36" in a) == (j.name == "mlp_esm2_3b__esm2")
        assert ("--rerun-on-change" in a) == (j.name.startswith("tok_ens_"))
        assert ("--expect-checkpoint" in a) == ("_ens_" in j.name and j.name.split("_")[0]
                                                in ("mlp", "conv"))
    planned = R.planned_outputs(jobs)
    assert "esm2_3b__esm2" in planned["mlp"] and "esm2_3b__esm2" not in planned["conv"]
    off = R.planned_outputs(R.plan(_runner_args(tmp_path, esm2_row="off"),
                                   CBP.resolve_paths(_runner_args(tmp_path))))
    assert "esm2_3b__esm2" not in off["mlp"]
    pdir = tmp_path / "out" / "mlp"; pdir.mkdir(parents=True)
    (pdir / "esm2_3b__esm2.json").write_text("{}")
    rep = R.completeness(args, CBP.resolve_paths(args), {}, off)
    assert rep["probes"]["mlp"]["aggregates"] == 0
    assert rep["probes"]["mlp"]["outputs_not_in_plan"] == ["esm2_3b__esm2"]


# ---------------------------------------------------------------- 8 modified residues
def test_ccd_parent_priority_and_reader(tmp_path):
    import cryptobench_build_apo as BA
    ccd = {"4CF": "PHE", "AME": "MET", "PIA": "ALA,TYR,GLY", "MLY": "LYS", "ACE": "?"}
    assert BA.residue_letter("4CF", {"4CF": "ILE"}, "parent", ccd) == ("F", "ccd")
    assert BA.residue_letter("AME", {}, "parent", ccd) == ("M", "ccd")
    assert BA.residue_letter("PIA", {"PIA": "SER"}, "parent", ccd) == ("S", "pdbx_struct_mod_residue")
    assert BA.residue_letter("ACE", {}, "parent", ccd)[0] is None
    assert BA.residue_letter("4CF", {}, "drop", ccd) == (None, "dropped_by_mode")
    gz = tmp_path / "c.cif.gz"
    with gzip.open(gz, "wt") as f:
        f.write("data_4CF\n_chem_comp.id 4CF\n_chem_comp.mon_nstd_parent_comp_id PHE\n"
                "data_PIA\n_chem_comp.id PIA\n_chem_comp.mon_nstd_parent_comp_id\n\"ALA,TYR,GLY\"\n"
                "data_ACE\n_chem_comp.id ACE\n_chem_comp.mon_nstd_parent_comp_id ?\n")
    assert BA.load_ccd_parents(gz) == {"4CF": "PHE", "PIA": "ALA,TYR,GLY", "ACE": "?"}


@pytest.mark.skipif(not os.environ.get("CB_GEMMI_PYTHON"), reason="set CB_GEMMI_PYTHON to a python with gemmi")
def test_build_with_gemmi_end_to_end(tmp_path):
    """Two tiny release entries built twice: parent mapping (CCD), backbone-drop counting,
    microheterogeneity retry, and the move-aside of the previous build."""
    rel = tmp_path / "rel"; (rel / "cif-files").mkdir(parents=True)
    atoms = []
    def res(seq_id, comp, label_seq, names, alt="."):
        for k, nm in enumerate(names):
            atoms.append(f"ATOM {len(atoms) + 1} {nm[0]} {nm} {alt} {comp} A 1 {label_seq} ? "
                         f"{1.0 * label_seq:.3f} {k * 1.1:.3f} 0.000 1.00 0.0 {seq_id} {comp} A 1")
    for i in range(1, 21):
        res(i, "MLY" if i == 5 else "ALA", i, ["N", "CA", "C", "O", "CB"])
    res(21, "GLY", 21, ["N", "C", "O"])                       # no CA: dropped at the backbone check
    head = ("data_1abc\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n_atom_site.label_entity_id\n_atom_site.label_seq_id\n"
            "_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n"
            "_atom_site.auth_seq_id\n_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n"
            "_atom_site.pdbx_PDB_model_num\n")
    (rel / "cif-files" / "1abc.cif").write_text(head + "\n".join(a + "" for a in atoms) + "\n")
    sel = [f"A_{i}" for i in (3, 4, 5)]
    (rel / "dataset.json").write_text(json.dumps({"1abc": [{"holo_pdb_id": "2xyz", "apo_chain": "A",
                                                            "apo_pocket_selection": sel}]}))
    (rel / "folds.json").write_text(json.dumps({"test": ["1abc"]}))
    (rel / "noncryptic-pockets.json").write_text("{}")
    ccd = tmp_path / "ccd.cif.gz"
    with gzip.open(ccd, "wt") as f:
        f.write("data_MLY\n_chem_comp.id MLY\n_chem_comp.mon_nstd_parent_comp_id LYS\n")
    cmd = [os.environ["CB_GEMMI_PYTHON"], str(CB / "cryptobench_build_apo.py"), "--workers", "1",
           "--cryptobench-dir", str(rel), "--ccd", str(ccd), "--structs-root", str(tmp_path),
           "--manifest-dir", str(tmp_path), "--run-tag", "t"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for _ in range(2):
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
    m = json.loads((tmp_path / "apo_manifest.json").read_text())
    e = m["entries"][0]
    assert e["L"] == 20 and e["n_pos"] == 3                    # MLY kept (5 is a positive)
    assert m["meta"]["modified_mapping"]["MLY"]["parent"] == "K"
    assert m["meta"]["dropped_incomplete_backbone"] == {"GLY": 1}
    assert m["meta"]["build_code_sha256"] and m["meta"]["release_sha256"]["folds.json"]
    d = np.load(tmp_path / "cryptobench_apo" / "1abc_A.npz", allow_pickle=True)
    assert str(d["seq"])[4] == "K" and d["comp"][4] == "MLY"
    assert list(tmp_path.glob("_pre_*/cryptobench_apo")) and list(tmp_path.glob("_pre_*/apo_manifest.json"))


# ---------------------------------------------------------------- 9 stats results / freshness
def test_stats_results_moved_aside_and_freshness(tmp_path):
    import _cb_paths as CBP
    import run_cryptobench_rv as R
    import protein_preds as PP
    args = _runner_args(tmp_path)
    paths = CBP.resolve_paths(args)
    jobs = {j.name: j for j in R.plan(args, paths)}
    res = paths.out_dir / "stats" / "cryptobench_families_v1_results.json"
    res.parent.mkdir(parents=True)
    # a seed result with a sidecar, recorded by the "results"
    f = tmp_path / "out" / "mlp" / "x__token__seed0.json"
    lab = {"t1_A": np.array([1, 0, 0, 1])}
    CC.write_seed_result(f, preds={"val": {"v_A": np.array([.1, .2])}, "test": {"t1_A": np.array([.3, .1, .2, .9])}},
                         labels={"val": {"v_A": np.array([1, 0])}, "test": lab},
                         layout={"val": {"v": ["v_A"]}, "test": {"t1": ["t1_A"]}},
                         meta={"row": "x", "probe": "mlp", "seed": 0}, result={"name": "x"})
    rec = json.loads(f.read_text())[PP.PREDS_KEY]
    fam_sha = hashlib.sha256(R.family_bytes(args, paths)).hexdigest()
    body = {"family_spec": {"sha256": fam_sha}, "families": [{"comparisons": [
        {"sidecars_a": [{"result": str(f), "sha256": rec["sha256"]}], "sidecars_b": []}]}]}
    res.write_text(json.dumps(body))
    assert R.stats_done(res, args, paths)
    res.write_text(json.dumps({**body, "family_spec": {"sha256": "0" * 64}}))
    assert not R.stats_done(res, args, paths)                 # other family instance
    res.write_text(json.dumps(body))
    f.with_name(f.stem + ".preds.npz").write_bytes(b"changed")
    assert not R.stats_done(res, args, paths)                 # sidecar changed
    jobs["stats"].pre()
    assert not res.exists() and list(res.parent.glob("_pre_*/cryptobench_families_v1_results.json"))
