"""Tests of the 2026-10-16 downstream fixes (WP3: code_fix P2, P9, P12, P13, P16, P17, P19).

What it does
------------
  * seed rescue (P2): the Ensembits battery and the baseline runner flag the same seeds
    on a synthetic table of result JSONs (shared rule in downstream/seed_rescue.py), and
    `mark_rescued` annotates a re-run JSON;
  * compiler (P19 / P16 / P9): importing `_compile_rebuttal_results` (and
    `significance_tests`, which imports it) writes nothing; `main()` on a synthetic
    REPO_ROOT writes the reported row, the pre-fix labels, the ablation section and the
    seed-completeness list, with sample std; `compile_camera_ready_tables` selects the
    reported rows by label;
  * checkpoint guard (P12): `featurize.load_model` / `check_checkpoint_finished` refuse a
    folder without config.json or with an unfinished run, unless allow_unfinished;
  * provenance (P17): matching / mismatching / missing checkpoint blocks;
  * mdCATH-div probe (P13): the row-count and sidecar checks of probe_rmsf_eval.
  * pair table (P21): build_pair_table on a synthetic token comparison.

Paper
-----
not described in the paper (infrastructure only)

How to run
----------
  python -m pytest tests/test_downstream_wp3.py -q      # from the codebase root (CPU, < 1 min)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = CODE_ROOT / "downstream"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(DOWNSTREAM))
sys.path.insert(0, str(DOWNSTREAM / "baselines"))


# ---------------------------------------------------------------- seed rescue (P2)
def _write_results(EC, GO, BD, ec_name, go_name, bd_name, values, epochs):
    """Synthetic EC / GO / BIND JSONs for seeds 0-9 (same value in every metric)."""
    for s, (v, ep) in enumerate(zip(values, epochs)):
        cell = {"top1_hit": v, "epochs_run": ep}
        (EC / ec_name(s)).write_text(json.dumps({"by_depth": {"1": cell, "2": cell, "3": cell}}))
        # GO results carry the label hash (benchmark_audit D7; patched as current below).
        (GO / go_name(s)).write_text(json.dumps({**cell, "go_labels": {"sha256": "synthetic"}}))
        res = {f"{sp}_{t}": ({"auroc": v, "epochs_run": ep} if t == "binding"
                             else {"spearman": v, "epochs_run": ep})
               for sp in ("sequence", "structure", "random") for t in ("binding", "affinity")}
        (BD / bd_name(s)).write_text(json.dumps({"results": res}))


VALUES = [0.80, 0.81, 0.79, 0.20, 0.82, 0.80, 0.78, 0.25, 0.81, 0.80]
EPOCHS = [40, 40, 40, 12, 40, 40, 40, 40, 40, 40]   # seed 3 premature, seed 7 genuine


def test_seed_rescue_same_decisions_for_both_callers(tmp_path, monkeypatch):
    import seed_rescue as SR
    import _run_model_battery as BAT
    import run_misato_baselines as RB
    EC, GO, BD = (tmp_path / "ec", tmp_path / "go", tmp_path / "bd")
    for d in (EC, GO, BD):
        d.mkdir()
    for mod in (BAT, RB):
        monkeypatch.setattr(mod, "EC", EC); monkeypatch.setattr(mod, "GO", GO)
        monkeypatch.setattr(mod, "BD", BD)
    import go_labels
    monkeypatch.setattr(go_labels, "current_label_sha", lambda path=None: "synthetic")
    # the fixed-split job names (function CV jobs, 2026-10-19: tests/test_function_cv.py)
    for mod in (BAT, RB):
        monkeypatch.setattr(mod, "FN_PROTOCOL", "split")
    V, F = "newrun", "esm3struct"
    sd = lambda s: "" if s == 0 else f"_seed{s}"
    sdb = lambda s: "" if s == 0 else f"__seed{s}"
    # Ensembits files (battery names) for P=full, sequence split; baseline files likewise.
    _write_results(EC, GO, BD, lambda s: f"tok_{V}_stab{sd(s)}__summary.json",
                   lambda s: f"go_{V}_stab{sd(s)}_top50.json",
                   lambda s: f"{V}_lig_stab{sd(s)}.json", VALUES, EPOCHS)
    _write_results(EC, GO, BD, lambda s: f"{F}__stab{sdb(s)}__summary.json",
                   lambda s: f"go_{F}_top50_stab{sd(s)}.json",
                   lambda s: f"baseline_{F}_lig_stab{sd(s)}.json", VALUES, EPOCHS)

    ens = SR.find_rescues(BAT.function_jobs(V, SR.RESCUE_RECIPE), verbose=False)
    base = SR.find_rescues(RB.function_jobs([F], RB.SPLITS, RB.SEEDS, SR.RESCUE_RECIPE), verbose=False)
    # Only seed 3 (premature low outlier) is re-run, for EC, GO and BIND, in both callers.
    assert sorted(Path(p).name for p in ens) == sorted([f"go_{V}_stab_seed3_top50.json",
                                                        f"tok_{V}_stab_seed3__summary.json",
                                                        f"{V}_lig_stab_seed3.json"])
    assert sorted(Path(p).name for p in base) == sorted([f"baseline_{F}_lig_stab_seed3.json",
                                                         f"go_{F}_top50_stab_seed3.json",
                                                         f"{F}__stab__seed3__summary.json"])
    # Same metric cells and original epochs in both.
    norm = lambda d: sorted((Path(p).suffix, tuple(sorted(e["original"])),
                             tuple(x["epochs_run"] for x in e["original"].values())) for p, e in d.items())
    assert norm(ens) == norm(base)
    # The rescue commands use the rescue recipe; the Ensembits BIND re-run recomputes.
    for p, e in ens.items():
        assert e["cmd"][-len(SR.RESCUE_RECIPE):] == SR.RESCUE_RECIPE or "--overwrite" in e["cmd"]
    assert "--overwrite" in ens[str(BD / f"{V}_lig_stab_seed3.json")]["cmd"]
    # Baseline wrapper keeps its old return type.
    assert set(RB.rescue_jobs(RB.function_jobs([F], RB.SPLITS, RB.SEEDS, SR.RESCUE_RECIPE))) == set(base)


def test_outlier_rule_and_mark_rescued(tmp_path):
    import seed_rescue as SR
    fix, gen, info = SR.outlier_decision({s: (v, e) for s, (v, e) in enumerate(zip(VALUES, EPOCHS))})
    assert fix == [3] and gen == [7] and info is not None
    # Low spread: not checked.
    assert SR.outlier_decision({s: (0.5 + 0.001 * s, 10) for s in range(10)}) == ([], [], None)
    f = tmp_path / "r.json"; f.write_text(json.dumps({"top1_hit": 0.8}))
    assert SR.mark_rescued(f, {"go": {"epochs_run": 12, "value": 0.2}})
    d = json.loads(f.read_text())
    assert d["rescued"] is True and d["rescue_original"]["go"]["epochs_run"] == 12
    assert not SR.mark_rescued(tmp_path / "missing.json", {})


# ---------------------------------------------------------------- compiler (P19, P16, P9)
def _env(root):
    e = dict(os.environ); e["ENSEMBITS_REPO_ROOT"] = str(root)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return e


def _all_files(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file())


def test_compile_import_has_no_side_effects(tmp_path):
    before = _all_files(tmp_path)
    code = ("import sys; sys.path.insert(0, %r); import _compile_rebuttal_results as C; "
            "import significance_tests; print('ROOT', C.ROOT)") % str(DOWNSTREAM)
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "wrote" not in r.stdout
    assert str(tmp_path.resolve()) in r.stdout          # the import used the tmp REPO_ROOT
    assert _all_files(tmp_path) == before                # nothing written


def _synthetic_repo(root):
    pg = root / "ensembits-repro" / "runs" / "proteingym"; pg.mkdir(parents=True)
    for n in ("esm3struct", "aminoaseed", "protoken", "3di_tokens", "combined_esm3"):
        (pg / f"{n}.json").write_text(json.dumps({"n_assays": 96, "mean_spearman_ours": 0.3,
                                                  "mean_zmix_a03": 0.5, "mean_spearman_esm2": 0.48}))
    ec = root / "output" / "ec_conv1d_misato"; ec.mkdir(parents=True)
    rm = root / "output" / "probe_rmsf_div"; rm.mkdir(parents=True)
    # benchmark_audit D3 (2026-10-18): EC / GO / binding results must record the current
    #   MISATO split file (data/misato_splits_v2.json of the fake repo)
    from ensembits.data_versions import misato_splits_record
    spf = root / "data" / "misato_splits_v2.json"; spf.parent.mkdir(parents=True, exist_ok=True)
    sp = {"train": ["a"], "val": ["b"], "test": ["c"]}
    spf.write_text(json.dumps({"sequence": sp, "structure": sp, "random": sp,
                               "stats": {"schema": "misato-splits-v2"}}))
    rec = misato_splits_record(spf)
    # benchmark_audit D8 (2026-10-18): EC results must record the current EC label file
    #   (data/annotations/misato_ec_v2.json of the fake repo)
    from ensembits.featurize import file_sha256
    ecf = root / "data" / "annotations" / "misato_ec_v3.json"; ecf.parent.mkdir(parents=True)
    ecf.write_text(json.dumps({"A": ["3.4.23.16"]}))
    rec = {**rec, "ec_labels": {"path": "data/annotations/misato_ec_v3.json",
                                "sha256": file_sha256(ecf), "schema": "misato-ec-v3"}}
    # function_reaudit (2026-10-19): function results carry the protocol tag, verified inputs
    #   and the item / class hashes
    sys.path.insert(0, str(DOWNSTREAM))
    import function_common as FC   # + input standardisation record (2026-10-19)
    rec = {**rec, **FC.protocol_record([{"file": "tokens.npz", "verified": True}],
                                       input_std=FC.InputStandardisation.lookup("t", 2, [FC.standardize_lookup_table(np.eye(2))[1]]).run_record())}
    # reported model: 10 seeds of EC (structure split, P=full), RMSF-div with 3 seeds only
    for s in range(10):
        suf = "" if s == 0 else f"_seed{s}"
        (ec / f"tok_newrun_structure_stab{suf}__summary.json").write_text(
            # EC metric decision (2026-10-19): the main tables' EC cells are mAP (macro_ap);
            #   the top-1 hit (0.3) must not appear there
            json.dumps({"by_depth": {"1": {"macro_ap": 0.5 + 0.01 * s, "top1_hit": 0.3,
                                           "items_sha256": "i" * 64,
                                           "classes_sha256": "c" * 64}}, **rec}))
    for s in range(3):
        suf = "" if s == 0 else f"_seed{s}"
        # (2026-10-16, Q17: an RMSF result needs the label, the common residue set and the
        #  checkpoint block to enter a table)
        (rm / f"newrun{suf}__test.json").write_text(json.dumps(
            {"test_spearman": 0.6 + 0.01 * s, "test_spearman_per_protein_mean": 0.4,
             "rmsf_label": "allframes", "residue_set_sha256": "a" * 64,
             "rmsf_label_sha256": "c" * 64, "feature_caches": [], "cache_verified": True,
             "splits_sha256": "d" * 64, "max_epochs": 500, "patience": 20, "min_delta": 1e-4, "batch_size": 2048, "selection": "val_early_stopping",
             "checkpoint": {"variant": "newrun", "best_pt_sha256": "b" * 64}}))


def test_compile_main_reported_rows_and_seed_check(tmp_path):
    _synthetic_repo(tmp_path)
    out = tmp_path / "res.md"; cam = tmp_path / "cam.md"
    code = ("import sys; sys.path.insert(0, %r); import _compile_rebuttal_results as C; "
            "sys.exit(C.main(['--variant', 'newrun', '--out', %r, '--allow-incomplete', "
            "'--function-protocol', 'split', "
            "'--ablation-variants', 'vqrun=VQ-2048,lamrun=λ = 0.5', "
            "'--abl-vq', 'newrun=e0,vqrun=VQ-2048', '--abl-sftd', 'lamrun=λ = 0.5']))") % (str(DOWNSTREAM), str(out))
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    md = out.read_text()
    # reported row: mean ± SAMPLE std of 0.50 .. 0.59
    sd = float(np.std([0.5 + 0.01 * s for s in range(10)], ddof=1))
    assert f"| Ensembits (reported) · P=full | {0.545:.3f}±{sd:.3f}" in md.replace("**", "")
    assert "| model | EC d1 mAP | EC d2 mAP | EC d3 mAP |" in md
    # final_fix_list A9: VQ / SFTD ablation tables without function columns
    assert "## Ablations: VQ (E1)" in md and "VQ-2048 · P = full" in md and "## Seed completeness" in md
    assert "## Ablations: SFTD (E2)" in md and "λ = 0.5 · P = 1" in md
    assert "| RMSF | Ensembits (reported) · P=full | mdCATH-div | 3 |" in md
    assert "fewer than 10 seeds" in r.stderr
    assert "rebuttal_analyses" not in md
    # camera-ready selector picks the reported rows by label
    code2 = ("import sys; sys.path.insert(0, %r); import compile_camera_ready_tables as T; "
             "sys.exit(T.main(['--variant', 'newrun', '--src', %r, '--out', %r, '--allow-incomplete']))"
             % (str(DOWNSTREAM), str(out), str(cam)))
    r2 = subprocess.run([sys.executable, "-c", code2], env=_env(tmp_path), capture_output=True,
                        text=True, timeout=300)
    assert r2.returncode == 0, r2.stderr
    cm = cam.read_text()
    assert "Ensembits (BioEmu-CATH+MISATO) · P=full" in cm and "output/newrun" in cm
    assert "sample std" in cm and "## Ablations" in cm


def test_compile_labels_prefix():
    import _compile_rebuttal_results as C
    assert all(d.endswith(C.PREFIX) for d, _ in C.MODELS + C.RM_MODELS)
    assert all(d.endswith(C.PREFIX) for d, *_ in C.PG_ROWS)
    assert "protoken_pooled" in C.BASELINES
    import run_misato_baselines as RB
    assert "protoken_pooled" in RB.BASELINES
    assert C.agg([0.1, 0.3])[1] == pytest.approx(np.std([0.1, 0.3], ddof=1))


# ---------------------------------------------------------------- checkpoint guard (P12)
def test_featurize_refuses_unfinished_or_missing_config(tmp_path):
    from ensembits.featurize import (check_checkpoint_finished, load_model,
                                     UnfinishedCheckpointError)
    with pytest.raises(UnfinishedCheckpointError):
        check_checkpoint_finished(tmp_path)                       # no config.json
    with pytest.raises(UnfinishedCheckpointError):
        load_model(tmp_path, device="cpu")
    (tmp_path / "config.json").write_text(json.dumps({"status": "running", "finished_at": None}))
    with pytest.raises(UnfinishedCheckpointError):
        check_checkpoint_finished(tmp_path)
    with pytest.raises(UnfinishedCheckpointError):
        load_model(tmp_path, device="cpu")                        # refused before stats.npz
    assert check_checkpoint_finished(tmp_path, allow_unfinished=True)["status"] == "running"
    (tmp_path / "config.json").write_text(json.dumps({"finished_at": None}))
    with pytest.raises(UnfinishedCheckpointError):
        check_checkpoint_finished(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"status": "finished",
                                                      "finished_at": "2026-10-16T00:00:00"}))
    assert check_checkpoint_finished(tmp_path)["status"] == "finished"
    (tmp_path / "config.json").write_text(json.dumps({"input_dim": 256}))   # legacy config
    assert check_checkpoint_finished(tmp_path)["input_dim"] == 256


def test_load_model_robust_refuses_unfinished(tmp_path):
    from tokenize_misato_and_eval import load_model_robust
    from ensembits.featurize import UnfinishedCheckpointError
    (tmp_path / "config.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(UnfinishedCheckpointError):
        load_model_robust(tmp_path, device="cpu")


def test_pretokenize_exits_1_without_config(tmp_path):
    (tmp_path / "output" / "norun").mkdir(parents=True)
    r = subprocess.run([sys.executable, str(DOWNSTREAM / "_pretokenize_misato_dihedral.py"),
                        "--variant", "norun", "--device", "cpu"], env=_env(tmp_path),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert not list((tmp_path / "output" / "norun").iterdir())


# ---------------------------------------------------------------- provenance (P17)
def test_provenance_checks(tmp_path):
    from ensembits.featurize import (checkpoint_provenance, write_provenance_sidecar,
                                     check_provenance_sidecar, ProvenanceMismatchError)
    ck = tmp_path / "run"; ck.mkdir()
    (ck / "best.pt").write_bytes(b"weights-1")
    (ck / "config.json").write_text(json.dumps({"code_sha256": "abc"}))
    prov = checkpoint_provenance(ck)
    assert prov["variant"] == "run" and prov["code_sha256"] == "abc" and len(prov["best_pt_sha256"]) == 64
    cache = tmp_path / "tokens.npz"; np.savez(cache, a=np.arange(3))
    with pytest.raises(ProvenanceMismatchError):                  # missing sidecar, new trainer
        check_provenance_sidecar(cache, prov)
    assert check_provenance_sidecar(cache, prov, force=True) == "forced"
    write_provenance_sidecar(cache, prov)
    assert check_provenance_sidecar(cache, prov) == "match"
    other = dict(prov, best_pt_sha256="0" * 64)
    with pytest.raises(ProvenanceMismatchError):
        check_provenance_sidecar(cache, other)
    legacy = dict(prov, code_sha256=None)
    cache2 = tmp_path / "old.npz"; np.savez(cache2, a=np.arange(3))
    assert check_provenance_sidecar(cache2, legacy) == "legacy"   # pre-fix checkpoint, no sidecar
    assert "a" in np.load(cache).files and len(np.load(cache).files) == 1   # nothing added to the npz


# ---------------------------------------------------------------- mdCATH-div probe (P13)
def test_probe_rmsf_eval_row_count_and_sidecar(tmp_path):
    import probe_rmsf_eval as PE
    desc = np.zeros((12, 10, 4), np.float32); rmsf = np.zeros(12, np.float32)
    PE.check_cache_rows(desc.shape[0], len(rmsf))
    with pytest.raises(ValueError):
        PE.check_cache_rows(desc[:-1].shape[0], len(rmsf))
    PE.check_desc_sidecar({"dihedral_convention": "iupac", "chain_break_cutoff": "auto"}, "iupac", 2.5)
    PE.check_desc_sidecar({"dihedral_convention": "iupac", "chain_break_cutoff": 2.5}, "iupac", "auto")
    with pytest.raises(ValueError):
        PE.check_desc_sidecar({"dihedral_convention": "legacy_plus180"}, "iupac", "auto")
    with pytest.raises(ValueError):
        PE.check_desc_sidecar({"dihedral_convention": "iupac", "chain_break_cutoff": None},
                              "iupac", "auto")
    p = tmp_path / "esm3desc_dihedral_iupac_k16_P10.npy"; np.save(p, desc)
    assert PE.find_desc_sidecar(p) is None
    (tmp_path / (p.name + ".provenance.json")).write_text("{}")
    assert PE.find_desc_sidecar(p).name == p.name + ".provenance.json"


# ---------------------------------------------------------------- pair table (P21)
def test_build_pair_table_from_synthetic_comparison(tmp_path):
    sys.path.insert(0, str(DOWNSTREAM / "fold_pairs"))
    import bioemu_token_compare as TC
    import build_pair_table as BT
    rng = np.random.default_rng(0)
    names = ['3Di', 'Ensembits P=1', 'Ensembits P=full']
    sim = {n: rng.uniform(0.2, 0.6, 12) for n in names}
    div = {n: rng.uniform(0.0, 0.4, 9) for n in names}
    con = TC.bootstrap_contrast(sim, div, names, n_boot=200, seed=0)
    comp = {'floor': {'n_systems': 50, 'per_tokenizer': {n: {'n_residues': 5000} for n in names}},
            'sets': {s: {'n_pairs': k, 'dropped': [], 'rows': [{'tokenizer': n, 'codebook': c}
                                                              for n, c in zip(names, (20, 2048, 2048))]}
                     for s, k in (('divergent', 9), ('similar', 12))},
            'contrast': {'ALL': con, 'DYNAMIC': con},
            'checkpoint': {'variant': 'newrun', 'best_pt_sha256': 'f' * 64}}
    md, tex = BT.build_tables(comp, {'divergent': (9, 25), 'similar': (12, 50)})
    text = '\n'.join(md)
    assert 'divergent: 9 valid of 25 verified, 9 used' in text and 'floor population: 50 systems' in text
    d = con['per_tokenizer']['3Di']['delta']
    assert f'{d:+.3f}' in text
    e = con['ensembits_minus_baseline']['Ensembits P=full - 3Di']['diff']
    assert f'{e:+.3f}' in text
    assert len([l for l in tex if not l.startswith('%')]) == 2 * len(names)
