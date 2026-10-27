"""BioEmu-CATH RMSF probe (phase 2): extras, labels, splits, probes end to end.

What it does
------------
Synthetic, CPU-only tests in pytest tmp dirs:
  * A: `build_bioemu_extras` re-derives the FPS picks of `build_bioemu_real_bb._process`
    on a tiny fake system (topology + two xtc runs) exactly (stored bb_10 == re-derived),
    the pick indices point at the right raw frames, sequence names are normalised, Gly
    has a NaN Cβ; a changed stored file is reported as a mismatch;
  * B: the all-frames label of `build_bioemu_rmsf_labels` equals the converged RMSF of
    every frame and does not depend on per-frame rigid poses;
  * C: the CATH-H split keeps H groups and the two systems of a domain together and
    reports the overlap with a tokenizer corpus split;
  * F: probe_rmsf_eval.py / probe_rmsf_baselines.py --dataset bioemu on a fake repo
    (residue set, per-protein fields, output/probe_rmsf_bioemu), and the compiler column.

Paper
-----
`2_results.tex` \\label{result:rmsf} (BioEmu-CATH column).

How to run
----------
  python -m pytest tests/test_rmsf_bioemu.py
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
DATA_PREP = CODE_ROOT / "data_prep"
for _p in (CODE_ROOT, DOWNSTREAM, DATA_PREP):
    sys.path.insert(0, str(_p))

md = pytest.importorskip("mdtraj")


def _env(root):
    e = dict(os.environ)
    e["ENSEMBITS_REPO_ROOT"] = str(root)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return e


def _run(root, module, argv, folder=DOWNSTREAM):
    code = (f"import sys; sys.path.insert(0, {str(folder)!r}); import {module} as M; "
            f"sys.exit(M.main({argv!r}))")
    return subprocess.run([sys.executable, "-c", code], env=_env(root), capture_output=True,
                          text=True, timeout=600)


def _rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


RES = ["ALA", "GLY", "HSD", "CYX", "LYS", "SER", "GLY", "VAL", "HIE", "LEU", "THR", "ASP"]


def _fake_system(sysdir, rng, n_runs=(14, 17), rigid=False):
    """topology.pdb + trajs/run00k_protein.cmprsd.xtc (+ one extra short xtc that must be
    ignored). Returns the xyz (Å) of all frames, concatenated."""
    top = md.Topology(); ch = top.add_chain()
    for name in RES:
        r = top.add_residue(name, ch)
        for a, el in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")) + \
                     ((("CB", "C"),) if name != "GLY" else ()):
            top.add_atom(a, md.element.get_by_symbol(el), r)
    n = top.n_atoms
    base = np.cumsum(rng.normal(size=(n, 3)) * 1.5, 0)
    (sysdir / "trajs").mkdir(parents=True)
    allx, pose_rng = [], np.random.default_rng(99)
    xs = [np.stack([base + rng.normal(size=(n, 3)) * 0.6 for _ in range(T)]) for T in n_runs]
    for k, x in enumerate(xs):
        if rigid:   # a random rigid pose per frame (separate generator: same frames otherwise)
            x = np.stack([f @ _rot(pose_rng).T + pose_rng.normal(size=3) * 20 for f in x])
        t = md.Trajectory(x / 10.0, top)
        t.save_xtc(str(sysdir / "trajs" / f"run{k:03d}_protein.cmprsd.xtc"))
        allx.append(md.load(str(sysdir / "trajs" / f"run{k:03d}_protein.cmprsd.xtc"), top=top).xyz * 10)
    md.Trajectory(base[None] / 10.0, top).save_pdb(str(sysdir / "topology.pdb"))
    md.Trajectory(base[None] / 10.0, top).save_xtc(str(sysdir / "trajs" / "x_0.cmprsd.xtc"))
    return np.concatenate(allx)


def test_extras_rederive_picks(tmp_path):
    import build_bioemu_real_bb as BB
    import build_bioemu_extras as EX
    rng = np.random.default_rng(0)
    root = tmp_path / "raw"; sysdir = root / "MSR_cath2" / "cath2_1abcA00"
    allx = _fake_system(sysdir, rng)
    real_bb = tmp_path / "rb"
    system, pid, err, L = BB._process(str(sysdir), str(real_bb), 10, 20)   # subsample 31 -> 20
    assert err is None and L == len(RES)
    row = {"system": system, "pid": pid, "deposit": "MSR_cath2", "L": L}
    out = tmp_path / "extras"
    assert EX.process(row, str(root), str(real_bb), str(out), 10, 20, False) == (system, "ok")
    z = np.load(out / f"{system}.npz")
    assert str(z["seq"]) == "AGHCKSGVHLTD"
    assert np.isnan(z["cb_10"][:, [1, 6]]).all() and not np.isnan(z["cb_10"][:, 0]).any()
    bb = np.load(real_bb / f"{system}.npz")["bb_10"]
    ca_idx = [i for i, a in enumerate(md.load_topology(str(sysdir / "topology.pdb")).atoms)
              if a.name == "CA"]
    np.testing.assert_array_equal(allx[z["pick_concat"]][:, ca_idx], bb[:, :, 1])
    assert z["run_n_frames"].tolist() == [14, 17] and len(z["run_files"]) == 2
    first = z["pick_concat"] - np.where(z["pick_run"] == 1, 14, 0)
    assert (first == z["pick_frame"]).all()
    # a stored backbone that differs from the re-derived one is a mismatch
    np.savez_compressed(real_bb / f"{system}.npz", bb_10=bb + 1e-3)
    st = EX.process(row, str(root), str(real_bb), str(out), 10, 20, True)[1]
    assert st.startswith("mismatch")


def test_bioemu_label_all_frames_pose_independent(tmp_path):
    import build_bioemu_real_bb as BB
    import build_bioemu_rmsf_labels as BL
    from build_mdcath_div_rmsf_labels import converged_rmsf
    rows = []
    for name, rigid in (("cath2_1aaaA00", False), ("cath2_1bbbA00", True)):
        rng = np.random.default_rng(4)          # same frames; the second with rigid poses
        sysdir = tmp_path / "raw" / "MSR_cath2" / name
        _fake_system(sysdir, rng, rigid=rigid)
        assert BB._process(str(sysdir), str(tmp_path / "rb"), 10, 2000)[2] is None
        rows.append({"system": name, "pid": name.split("_")[1], "deposit": "MSR_cath2",
                     "L": len(RES)})
    (tmp_path / "rb" / "layout_rows.json").write_text(json.dumps(rows))
    out = tmp_path / "lab.npy"; rows_out = tmp_path / "rows.json"
    assert BL.main(["--root", str(tmp_path / "raw"), "--real-bb", str(tmp_path / "rb"),
                    "--layout", str(tmp_path / "rb" / "layout_rows.json"), "--out", str(out),
                    "--rows-out", str(rows_out), "--workers", "1"]) == 0
    lab = np.load(out)
    L = len(RES)
    # pose independence (xtc precision 0.01 Å)
    np.testing.assert_allclose(lab[:L], lab[L:], atol=0.03)
    rm = json.loads(rows_out.read_text())
    assert rm[1]["row_start"] == L and rm[0]["length"] == L
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["n_frames_per_system"]["cath2_1aaaA00"] == 31 and prov["label_name"] == "allframes"
    assert BL.main(["--root", str(tmp_path / "raw"), "--real-bb", str(tmp_path / "rb"),
                    "--layout", str(tmp_path / "rb" / "layout_rows.json"), "--out", str(out),
                    "--rows-out", str(rows_out), "--workers", "1"]) == 1    # refuses overwrite
    # the label equals the converged RMSF over every frame of every run
    sysdir = tmp_path / "raw" / "MSR_cath2" / "cath2_1aaaA00"
    ca, run_n = BL.all_frames_ca(sysdir)
    assert run_n == [14, 17] and ca.shape[0] == 31
    np.testing.assert_allclose(lab[:L], converged_rmsf(ca)[0], atol=1e-5)


def test_bioemu_split_groups(tmp_path):
    import build_bioemu_rmsf_splits as BS
    doms = [f"{i}xyzA00" for i in range(20)]
    rows = [{"system": f"cath2_{d}", "pid": d, "deposit": "MSR_cath2", "L": 3} for d in doms]
    rows.insert(5, {"system": f"cath1_{doms[4]}", "pid": doms[4], "deposit": "ONE_cath1", "L": 3})
    h = {d: f"1.10.{i % 12}.1" for i, d in enumerate(doms[:-1])}     # last domain: no CATH
    sp = BS.build(rows, h)
    which = {d: k for k, ds in sp["domains"].items() for d in ds}
    for a in doms[:-1]:
        for b in doms[:-1]:
            if h[a] == h[b]:
                assert which[a] == which[b]
    s4 = [k for k, ss in sp["systems"].items() for s in ss if s.split("_", 1)[1] == doms[4]]
    assert len(s4) == 2 and len(set(s4)) == 1          # both systems of the domain together
    assert sp["h_superfamily"][doms[-1]] == f"no_cath_{doms[-1]}"
    assert sorted(sp["train"] + sp["val"] + sp["test"]) == list(range(3 * len(rows)))
    # same procedure as scripts/baselines/build_mdcath_splits.py cluster_split
    cl = sorted(set(sp["h_superfamily"].values())); rng = np.random.RandomState(42); rng.shuffle(cl)
    assert set(sp["domains"]["train"]) == {d for d, c in sp["h_superfamily"].items()
                                           if c in set(cl[:int(0.8 * len(cl))])}
    # overlap with a tokenizer corpus split (rows of the corpus layout)
    cdir = tmp_path / "corpus"; cdir.mkdir()
    lay = [{"system": r["system"], "pid": r["pid"], "corpus": "bioemu_cath", "L": 3, "start": 3 * i}
           for i, r in enumerate(rows)]
    (cdir / "layout.json").write_text(json.dumps({"rows": lay}))
    (cdir / "splits.json").write_text(json.dumps({"train": list(range(3 * len(rows))), "val": [],
                                                  "test": []}))
    ov = BS.corpus_overlap(sp["domains"], cdir)
    assert ov["test"]["tokenizer_train"] == len(sp["domains"]["test"])


# ---------------------------------------------------------------- end to end (F)
N_SYS, L_SYS = 12, 13


def _fake_bioemu_repo(root):
    rng = np.random.default_rng(9)
    cache = root / "data" / "cached_descriptors"
    rb = cache / "bioemu_cath_real_bb"; rb.mkdir(parents=True)
    ex = cache / "bioemu_cath_extras"; ex.mkdir(parents=True)
    rows = []
    for i in range(N_SYS):
        d = f"{i}abcA00"
        for dep, pre in (("MSR_cath2", "cath2"),) + ((("ONE_cath1", "cath1"),) if i == 0 else ()):
            s = f"{pre}_{d}"
            rows.append({"system": s, "pid": d, "deposit": dep, "corpus": "bioemu_cath", "L": L_SYS})
            np.savez(rb / f"{s}.npz", bb_10=rng.normal(size=(10, L_SYS, 4, 3)).astype(np.float32))
            np.savez(ex / f"{s}.npz", seq=np.array("ACDEFGHIKLMNP"),
                     cb_10=rng.normal(size=(10, L_SYS, 3)).astype(np.float32))
    (rb / "layout_rows.json").write_text(json.dumps(rows))
    (cache / "bioemu_cath_extras.provenance.json").write_text(json.dumps({"mismatch": {}, "n_ok": len(rows)}))
    total = len(rows) * L_SYS
    bdir = root / "output" / "phase_b" / "cached_desc" / "bioemu"; bdir.mkdir(parents=True)
    np.save(bdir / "rmsf_bioemu_allframes.npy", rng.gamma(2.0, 1.0, total).astype(np.float32))
    (bdir / "rmsf_bioemu_allframes.npy.provenance.json").write_text(json.dumps(
        {"label_name": "allframes", "n_rows": total}))
    (bdir / "esm3desc_dihedral_k16_P10.npy.provenance.json").write_text(json.dumps(
        {"dataset": "bioemu", "n_rows": total, "IDEAL_PIDS": []}))
    np.save(bdir / "esm3desc_dihedral_k16_P10.npy",
            rng.normal(size=(total, 10, 256)).astype(np.float32))
    cath = root / "cath-domain-list.txt"
    cath.write_text("".join(f"{i}abcA00 1 10 {i} {i}\n" for i in range(N_SYS)))
    return rows, total, cath


def test_bioemu_probes_end_to_end(tmp_path):
    rows, total, cath = _fake_bioemu_repo(tmp_path)
    r = _run(tmp_path, "build_bioemu_rmsf_splits", ["--cath-list", str(cath)], DATA_PREP)
    assert r.returncode == 0, r.stdout + r.stderr
    r = _run(tmp_path, "build_rmsf_residue_sets", ["--dataset", "bioemu"], DATA_PREP)
    assert r.returncode == 0, r.stdout + r.stderr
    d = tmp_path / "output" / "v1"; d.mkdir(parents=True)
    for f in ("best.pt", "config.json", "stats.npz", "history.json"):
        os.symlink(CODE_ROOT / "ckpt" / "bioemu_cath_misato_lam10" / f, d / f)  # reported λ = 10 checkpoint
    args = ["--dataset", "bioemu", "--device", "cpu", "--max-epochs", "3", "--patience", "2"]
    r = _run(tmp_path, "probe_rmsf_eval", ["--variant", "v1", *args])
    assert r.returncode == 0, r.stdout + r.stderr
    js = json.loads((tmp_path / "output" / "probe_rmsf_bioemu" / "v1__test.json").read_text())
    assert js["dataset"] == "bioemu" and js["rmsf_label"] == "allframes"
    assert js["n_train_residues"] + js["n_val_residues"] + js["n_test_residues"] == total
    assert js["residue_set_n_proteins"] == len(rows) and "test_spearman_per_protein_mean" in js
    for feat in ("aa", "random"):
        r = _run(tmp_path, "probe_rmsf_baselines", ["--feature", feat, *args])
        assert r.returncode == 0, r.stdout + r.stderr
    jb = json.loads((tmp_path / "output" / "probe_rmsf_bioemu" / "baseline_aa__test.json").read_text())
    assert jb["residue_set_sha256"] == js["residue_set_sha256"] and jb["coverage"] == 1.0
    # the compiler shows the BioEmu column for both rows
    pg = tmp_path / "ensembits-repro" / "runs" / "proteingym"; pg.mkdir(parents=True)
    for n in ("esm3struct", "aminoaseed", "protoken", "3di_tokens", "combined_esm3"):
        (pg / f"{n}.json").write_text(json.dumps({"n_assays": 96, "mean_spearman_ours": 0.3,
                                                  "mean_zmix_a03": 0.5, "mean_spearman_esm2": 0.48}))
    out = tmp_path / "res.md"
    r = _run(tmp_path, "_compile_rebuttal_results", ["--variant", "v1", "--out", str(out),
                                                     "--allow-incomplete"])   # 1 seed: preview
    assert r.returncode == 1 and "refused" in r.stderr   # K2: non-default probe settings
    md_text = out.read_text()
    assert "| model | mdCATH-div | MISATO-seq | MISATO-struct | BioEmu-CATH |" in md_text
    # K2: these probes ran with --max-epochs 3 --patience 2, not the defaults -> refused
    assert "differ from the defaults" in md_text
    row = next(l for l in md_text.splitlines() if l.startswith("| Ensembits (reported) · P=full |"))
    assert row.split("|")[5].strip().startswith("✗")   # BioEmu P=full cell refused (K2)
    # the validation-only selection table reads the BioEmu results too
    # BioEmu only with --include-bioemu (warning); mdCATH-div is always required (F10)
    r = _run(tmp_path, "rmsf_selection_table", ["--variants", "v1", "--include-bioemu", "--seeds", "0"])
    assert r.returncode == 1 and "WARNING" in r.stderr and "mdcath_div P=full: missing seeds" in r.stderr


def test_bioemu_baseline_inputs(tmp_path):
    rows, total, _ = _fake_bioemu_repo(tmp_path)
    code = ("import sys, numpy as np; sys.path.insert(0, %r); import _bioemu_inputs as BI; "
            "f = BI.CaFrames(); s = next(iter(f)); ca = f[s]; "
            "bb, cb, src = BI.load_bb_for_frame('bioemu', s, 3, ca[3]); "
            "assert len(f) == %d and src == 'real' and bb.shape == (%d, 4, 3) and cb.shape == (%d, 3); "
            "assert BI.sequence(s) == 'ACDEFGHIKLMNP'; "
            "ok = False\n"
            "try: BI.load_bb_for_frame('bioemu', s, 3, ca[2])\n"
            "except ValueError: ok = True\n"
            "assert ok; print('OK')") % (str(DATA_PREP / "baselines"), len(rows), L_SYS, L_SYS)
    r = subprocess.run([sys.executable, "-c", code], env=_env(tmp_path), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


def test_descriptor_builder_overwrite_and_rows(tmp_path, monkeypatch):
    # G2: refuse an existing cache unless --overwrite; atomic write; sidecar with rows
    monkeypatch.setattr(sys.modules[__name__], "L_SYS", 20)   # the descriptor needs >= 16 residues
    rows, total, _ = _fake_bioemu_repo(tmp_path)
    args = ["--dataset", "bioemu", "--device", "cpu"]
    code = ("import sys; sys.argv = ['b'] + %r; sys.path.insert(0, %r); "
            "import build_esm3desc_dihedral_perdataset as D; sys.exit(D.main())") % (args, str(DATA_PREP))
    run = lambda extra=(): subprocess.run([sys.executable, "-c", code.replace("%r" % (args,), "%r" % (args + list(extra),))],
                                          env=_env(tmp_path), capture_output=True, text=True, timeout=600)
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    bdir = tmp_path / "output" / "phase_b" / "cached_desc" / "bioemu"
    out = bdir / "esm3desc_dihedral_iupac_k16_P10.npy"
    assert np.load(out, mmap_mode="r").shape == (total, 10, 256)
    side = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert side["n_rows"] == total and side["dataset"] == "bioemu"
    assert not list(bdir.glob(".*tmp*"))                       # atomic: no temporary left
    r = run(); assert r.returncode == 1 and "exists" in r.stdout
    r = run(["--overwrite"]); assert r.returncode == 0, r.stdout + r.stderr
