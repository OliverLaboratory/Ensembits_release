"""Tests of the CryptoBench fixes after the port re-audit (FIXLIST_20261022.md items 1-8).

What it does
------------
Synthetic data only (no CryptoBench run, no metric of any real model is computed or read):
  * item 1: probe input fingerprints (a changed input file, codebook, option or extra seed
    file makes a finished run "not done"; a rerun moves the old outputs aside), the runner's
    dependency propagation and stats freshness rule;
  * item 2: dense files with a level-0 mismatch, another manifest or another row checkpoint
    are refused by the gate;
  * item 3: the statistics refuse a method whose seed set differs from the declared one, a
    split mismatch and a non-finite p in a cryptobench family; sidecar hashes are recorded;
  * item 4: the runner's hard-coded bundle sha256 equals the file and the generator;
  * item 5: modified-residue mapping (parent / drop), the collaborator's aa order, conv
    training-fold standardisation (table == per-residue inputs; q0 == token stays), the
    selection option, ESM-2 chunking;
  * item 6: `cryptobench_tokenize.py` needs --model-dir;
  * item 7: the GPU guard fails closed;
  * item 8: the cluster generator reproduces the pre-registered partition.

Paper
-----
not described in the paper (infrastructure for the CryptoBench experiment)

How to run
----------
    python -m pytest tests/test_cryptobench_fixes_20261022.py
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
CB = CODE_ROOT / "downstream" / "cryptobench"
for _p in (CODE_ROOT, CODE_ROOT / "downstream", CB):
    sys.path.insert(0, str(_p))

import cryptobench_common as CC  # noqa: E402
import protein_level_stats as S  # noqa: E402
import protein_preds as PP  # noqa: E402


# ---------------------------------------------------------------- a tiny synthetic set
def make_set(tmp, n_per_fold=None, L=24, seed=0):
    """Structure caches + manifest + an Ensembits-like token file with a valid sidecar."""
    rng = np.random.default_rng(seed)
    n_per_fold = n_per_fold or {"train-0": 3, "train-1": 2, "train-2": 2, "train-3": 3, "test": 3}
    sd = tmp / "cryptobench_apo"; sd.mkdir(parents=True)
    ents, toks = [], {}
    i = 0
    for fold, n in n_per_fold.items():
        for _ in range(n):
            apo = f"{i:01d}x{i:02d}"[:4]; i += 1
            y = (rng.random(L) < 0.25).astype(np.int8); y[0] = 1; y[1] = 0
            seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), L))
            np.savez(sd / f"{apo}_A.npz", y=y, seq=np.array(seq), nc=np.zeros(L, np.int8))
            ents.append({"apo": apo, "chain": "A", "fold": fold, "status": "ok", "L": L})
            toks[f"{apo}_A"] = rng.integers(0, 16, L).astype(np.uint16)
    man = tmp / "apo_manifest.json"
    man.write_text(json.dumps({"entries": ents, "meta": {"build_version": CC.STRUCT_BUILD_VERSION}}))
    tok = tmp / "cryptobench_tokens_t.npz"
    np.savez(tok, **toks)
    np.save(tmp / "cryptobench_tokens_t_codebook.npy", rng.normal(size=(16, 8)).astype(np.float32))
    (tmp / "cryptobench_tokens_t.npz.provenance.json").write_text(json.dumps(
        {"checkpoint": {"variant": "v", "best_pt_sha256": "ab"}, "n_failed": 0,
         "manifest_sha256": CC.file_sha256(man)}))
    return man


def loc(tmp):
    return ["--run-tag", "t", "--structs-root", str(tmp), "--manifest-dir", str(tmp),
            "--tokens-dir", str(tmp), "--out-dir", str(tmp / "out")]


# ---------------------------------------------------------------- item 1
def test_mlp_fingerprint_skip_and_rerun(tmp_path):
    import cryptobench_mlp_probe as MP
    make_set(tmp_path)
    argv = ["--row", "r", "--features", "tokens", "--tokens", "t", "--seeds", "2",
            "--epochs", "2", "--patience", "2", "--device", "cpu", *loc(tmp_path)]
    assert not MP.is_done(argv)
    assert MP.main(argv) == 0
    assert MP.is_done(argv)
    out = tmp_path / "out" / "mlp"
    js = json.loads((out / "r__token__seed0.json").read_text())
    assert js["options"]["selection"] == "val_pooled_auroc" and js["input_fingerprint"]["code"]
    # an option enters the fingerprint
    assert not MP.is_done(argv + ["--selection", "val_per_entry_auroc"])
    # aa_order does not enter the fingerprint of a token row
    assert MP.is_done(argv + ["--aa-order", "alphabetical"])
    # a changed codebook makes the run stale; a rerun moves the old outputs aside
    cb = tmp_path / "cryptobench_tokens_t_codebook.npy"
    np.save(cb, np.load(cb) * 2)
    assert not MP.is_done(argv)
    assert MP.main(argv) == 0 and MP.is_done(argv)
    assert list(out.glob("_pre_*"))                              # old outputs moved, not deleted
    # an extra seed file makes the method not done
    (out / "r__token__seed7.json").write_text("{}")
    assert not MP.is_done(argv)


def test_conv_runs_with_standardisation_and_records_options(tmp_path):
    import cryptobench_probe as CP
    make_set(tmp_path)
    argv = ["--row", "aa", "--features", "aa", "--seeds", "2", "--max-epochs", "2",
            "--min-epochs", "1", "--patience", "1", "--warmup", "1", "--device", "cpu",
            *loc(tmp_path)]
    assert CP.main(argv) == 0 and CP.is_done(argv)
    js = json.loads((tmp_path / "out" / "conv" / "aa__onehot__seed1.json").read_text())
    assert js["options"] == {"standardize": "train", "selection": "val_pooled_auroc",
                             "aa_order": "collaborator"}
    assert not CP.is_done(argv + ["--aa-order", "alphabetical"])      # aa row: affects
    assert not CP.is_done(argv + ["--standardize", "none"])


def test_conv_train_stats_table_equals_inputs():
    import cryptobench_probe as CP
    rng = np.random.default_rng(1)
    T = rng.normal(size=(12, 5)).astype(np.float32) * 7
    rows = [("a", rng.integers(0, 12, 30), None), ("b", rng.integers(0, 12, 17), None)]
    mu_t, sd_t = CP.train_stats(rows, T)
    dense = [(k, T[X], None) for k, X, _ in rows]
    mu_d, sd_d = CP.train_stats(dense)
    assert np.allclose(mu_t, mu_d, atol=1e-5) and np.allclose(sd_t, sd_d, atol=1e-4)
    Z = np.concatenate([(T[X] - mu_t) / sd_t for _, X, _ in rows])
    assert np.allclose(Z.mean(0), 0, atol=1e-4) and np.allclose(Z.std(0), 1, atol=1e-3)


def test_runner_dependency_propagation_and_gpu_guard():
    import run_cryptobench_rv as R
    jobs = [R.Job("a", [], done=lambda: False), R.Job("b", [], ["a"], done=lambda: True),
            R.Job("c", [], done=lambda: True), R.Job("d", [], ["c"], done=lambda: True)]
    st = R.initial_states(jobs, redo_all=False)
    assert st == {"a": "todo", "b": "todo", "c": "skip", "d": "skip"}
    assert set(R.initial_states(jobs, redo_all=True).values()) == {"todo"}
    assert R.gpu_refusals([0, 3], None, 2048) == [0, 3]               # query failed
    assert R.gpu_refusals([0, 3], {0: 10, 3: 130000}, 2048) == [3]
    assert R.gpu_refusals([0, 5], {0: 10}, 2048) == [5]               # unknown GPU
    j = R.Job("x", ["s.py"], redo_flag=True)
    assert j.cmd(redo=True)[-1] == "--redo" and "--redo" not in j.cmd()


# ---------------------------------------------------------------- item 2
def test_gate_dense_mismatch_and_row_checkpoint(tmp_path):
    from ensembits.featurize import checkpoint_provenance
    man = tmp_path / "apo_manifest.json"; man.write_text("{}")
    ck = tmp_path / "ensembits_bm_lam1_rv"; ck.mkdir()
    (ck / "best.pt").write_bytes(b"w")
    (ck / "config.json").write_text(json.dumps({"dihedral_convention": "iupac",
                                                "chain_break_cutoff": 2.5}))
    prov = checkpoint_provenance(ck)
    p = tmp_path / "cryptobench_dense_x.npz"; np.savez(p, a=np.zeros(2))
    side = p.with_name(p.name + ".provenance.json")
    base = {"checkpoint": prov, "manifest_sha256": CC.file_sha256(man), "n_failed": 0,
            "readouts": ["z", "q", "q0"], "dihedral_convention": "iupac",
            "chain_break_cutoff": 2.5, "n_token_mismatch": 0, "n_tokenfile_mismatch": 0}
    side.write_text(json.dumps(base))
    assert CC.check_token_file(p, man, expect_checkpoint=ck)
    for bad in ({"n_token_mismatch": 3}, {"n_tokenfile_mismatch": 1},
                {"n_tokenfile_mismatch": None}, {"dihedral_convention": "legacy_plus180"}):
        side.write_text(json.dumps({**base, **bad}))
        with pytest.raises(CC.TokenFileError):
            CC.check_token_file(p, man, expect_checkpoint=ck)
    side.write_text(json.dumps({**base, "checkpoint": {**prov, "variant": "other"}}))
    with pytest.raises(CC.TokenFileError):
        CC.check_token_file(p, man, expect_checkpoint=ck)
    # the ESM-2 file is accepted with its manifest
    side.write_text(json.dumps({"tokenizer": "esm2_3b", "manifest_sha256": CC.file_sha256(man),
                                "n_failed": 0, "readouts": ["esm2"], "layer": 33,
                                "model": "facebook/esm2_t36_3B_UR50D"}))
    assert CC.check_token_file(p, man)["tokenizer"] == "esm2_3b"


# ---------------------------------------------------------------- item 3
def _seed_runs(tmp, probe, row, seeds, shift):
    rng = np.random.default_rng(7)
    labels, preds = {}, {}
    for split, ents in (("val", ["v1", "v2"]), ("test", [f"t{i}" for i in range(6)])):
        labels[split], preds[split] = {}, {}
        for e in ents:
            y = (rng.random(25) < 0.3).astype(np.int64); y[0] = 1; y[1] = 0
            labels[split][f"{e}_A"] = y
            preds[split][f"{e}_A"] = (rng.random(25) + shift * y).astype(np.float32)
    lay = {s: CC.entry_layout(list(labels[s])) for s in labels}
    out = []
    for s in seeds:
        f = tmp / probe / f"{row}__token__seed{s}.json"
        CC.write_seed_result(f, preds=preds, labels=labels, layout=lay,
                             meta={"row": row, "probe": probe, "seed": s, "set": "apo"},
                             result={"name": row})
        out.append(str(f))
    return out


def test_stats_seed_set_split_nan_and_hashes(tmp_path, monkeypatch):
    a = _seed_runs(tmp_path, "mlp", "A", [0, 1, 2], 0.5)
    b = _seed_runs(tmp_path, "mlp", "B", [0, 1, 2, 7], 0.0)
    cl = tmp_path / "cl.json"
    cl.write_text(json.dumps({f"t{i}": f"c{i // 2}" for i in range(6)}))
    comp = {"id": "x", "task": "cryptobench", "metric": "auroc_per_protein", "a": a, "b": b[:3]}
    spec = {"family": "f", "B": 50, "clusters": {"path": str(cl)}, "comparisons": [comp]}
    r = S.run_family({**spec, "seeds": [0, 1, 2], "split": "test"})
    rec = r["comparisons"][0]["sidecars_a"]
    assert len(rec) == 3 and all(len(x["sha256"]) == 64 for x in rec)
    with pytest.raises(S.RefusedError):                          # extra seed 7 in B
        S.run_family({**spec, "seeds": [0, 1, 2], "comparisons": [{**comp, "b": b}]})
    with pytest.raises(S.RefusedError):
        S.run_family({**spec, "seeds": [0, 1, 2, 3, 4]})
    with pytest.raises(S.RefusedError):
        S.run_family({**spec, "split": "val"})
    real = S.compare
    monkeypatch.setattr(S, "compare", lambda *x, **k: {**real(*x, **k), "p": float("nan")})
    with pytest.raises(S.RefusedError):
        S.run_family(spec)


# ---------------------------------------------------------------- item 4
def test_bundle_sha_is_hard_coded():
    import run_cryptobench_rv as R
    assert CC.file_sha256(R.FAMILY_TEMPLATE) == R.FAMILY_SHA256
    assert R.template_problem() is None
    tpl = json.loads(R.FAMILY_TEMPLATE.read_text())
    for f in tpl["families"]:                                    # ESM-2 in no family
        for c in f["comparisons"]:
            assert "esm2" not in json.dumps(c)


def test_family_instance_bytes_declare_seeds(tmp_path):
    import _cb_paths as CBP
    import run_cryptobench_rv as R
    args = Namespace(run_tag="t", legacy_layout=False, cryptobench_dir=None,
                     structs_root=str(tmp_path), manifest_dir=str(tmp_path),
                     tokens_dir=str(tmp_path), out_dir=str(tmp_path / "o"), stats_B=None, seeds=5)
    paths = CBP.resolve_paths(args)
    b1, b2 = R.family_bytes(args, paths), R.family_bytes(args, paths)
    assert b1 == b2
    inst = json.loads(b1)
    assert inst["seeds"] == [0, 1, 2, 3, 4] and inst["template"]["sha256"] == R.FAMILY_SHA256
    assert R.seed_set_problems(inst)                             # no seed files yet


# ---------------------------------------------------------------- item 5
def test_modified_residue_mapping_rule():
    import cryptobench_build_apo as BA
    parents = {"MLY": "LYS", "SEP": "SER", "XYZ": "HOH"}
    assert BA.residue_letter("ALA", parents, "parent") == ("A", "standard")
    assert BA.residue_letter("MSE", parents, "drop") == ("M", "standard")
    assert BA.residue_letter("MLY", parents, "parent") == ("K", "pdbx_struct_mod_residue")
    assert BA.residue_letter("SEP", parents, "parent")[0] == "S"
    assert BA.residue_letter("MLY", parents, "drop") == (None, "dropped_by_mode")
    assert BA.residue_letter("XYZ", parents, "parent")[0] is None   # parent not an amino acid
    assert BA.BUILD_VERSION == CC.STRUCT_BUILD_VERSION


def test_collaborator_aa_order():
    assert CC.AA_ORDERS["collaborator"] == "ARNDCQEGHILKMFPSTWYV"
    ids = CC.aa_ids("ARVXU", "collaborator")
    assert ids.tolist() == [0, 1, 19, 20, 20]
    assert CC.aa_ids("C", "alphabetical").tolist() == [1]


def test_mlp_per_entry_selection_option():
    import cryptobench_mlp_probe as MP
    torch.manual_seed(0)
    m = MP.build_mlp(4)
    X = np.random.default_rng(0).normal(size=(40, 4)).astype(np.float32)
    Y = np.r_[np.ones(10), np.zeros(10), np.ones(5), np.zeros(15)].astype(np.int64)
    B = [("e1_A", 0, 20), ("e2_A", 20, 40)]
    v1 = MP.val_selection_score(m, X, Y, B, "cpu", "val_pooled_auroc")
    v2 = MP.val_selection_score(m, X, Y, B, "cpu", "val_per_entry_auroc")
    assert np.isfinite(v1) and np.isfinite(v2)


def test_esm2_chunks():
    import cryptobench_esm2 as E
    assert E.chunks(10) == [(0, 10)]
    assert E.chunks(2500) == [(0, 1022), (1022, 2044), (2044, 2500)]


# ---------------------------------------------------------------- item 6
def test_tokenize_requires_model_dir(capsys):
    import cryptobench_tokenize as T
    with pytest.raises(SystemExit):
        T.parser().parse_args(["--tag", "x"])


# ---------------------------------------------------------------- item 8
def test_cluster_generator_reproduces_partition():
    import cryptobench_test_clusters as TC
    from ensembits.paths import REPO_ROOT
    d = REPO_ROOT / "camera_ready" / "cryptobench_audit" / "leak_D"
    if not (d / "apo_vs_apo.m8").exists():
        pytest.skip("audit hit file not present")
    meta = json.loads((d / "cb_apo_meta.json").read_text())
    ref = json.loads((d / "test_clusters_30.json").read_text())
    cl = TC.clusters_from_m8(open(d / "apo_vs_apo.m8"), {k: v["fold"] for k, v in meta.items()},
                             list(ref))

    def part(m):
        g = defaultdict(set)
        for k, v in m.items():
            g[v].add(k)
        return sorted(sorted(x) for x in g.values())
    assert part(cl) == part(ref) and len(set(cl.values())) == 201
