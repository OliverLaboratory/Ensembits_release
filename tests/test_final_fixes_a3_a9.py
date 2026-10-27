"""Tests of the camera-ready fixes A3 (function dot plot) and A9 (ablation tables).

What it does
------------
Synthetic checks (CPU, no real checkpoint, no test metric):
  * plot_function_dotplot: --split-metric test is refused without --unseal (exit 2, nothing
    read or written); every value comes through the compiler's loaders (brow / mrow,
    _check_items, enforce_seeds) in val mode; the random row is the reference line; a
    refused cell is not drawn and makes the script exit 1 (0 with --allow-incomplete);
    the PDF / PNG / JSON are written;
  * recon_by_p: on a tiny synthetic run (real config.json layout, random weights) the loss
    equals the recon path of RVQVAETokenizer.forward with the trainer's loss; --verify-run
    copies the other runs only when the verified run is identical;
  * token_usage_by_p: usage() counts and perplexity, the P = 10 self-check against
    config.json per_level;
  * _compile_rebuttal_results.ablation_tables: no function columns, recon / usage / ANOVA
    cells checked against best.pt, "—†" for a one-token row, refused / missing cells
    reported, in the test and the val mode.

Paper
-----
`2_results.tex` fig:downstream_radar (replaced), `appendix.tex` app:ablation-vq /
app:ablation-sftd (final_fix_list A3, A9).

How to run
----------
  python -m pytest tests/test_final_fixes_a3_a9.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream"))
sys.path.insert(0, str(CODE_ROOT / "downstream" / "analyses"))

import _compile_rebuttal_results as CR      # noqa: E402
import plot_function_dotplot as DP          # noqa: E402
import recon_by_p as RB                     # noqa: E402
import token_usage_by_p as TU               # noqa: E402


# ---------------------------------------------------------------- A3 dot plot
def _fake_loaders(monkeypatch, refused=None):
    """brow / mrow return synthetic cells; records the calls and the split metric."""
    calls = []
    specs_seen = []
    def cellv(name, sp, k):
        if refused and (name, sp, k) in refused:
            return CR.Refused(0)
        base = 0.1 + 0.05 * (hash(name) % 7) + (0.0 if sp == "sequence" else -0.02)
        return (base + 0.01 * k, 0.005, 10)
    def brow(F, sp, specs):
        calls.append(("b", F, sp, CR.SPLIT_METRIC)); specs_seen.append(specs)
        return [cellv(F, sp, k) for k in range(len(specs))]
    def mrow(V, p, sp, specs):
        calls.append(("m", V, p, sp, CR.SPLIT_METRIC))
        return [cellv(f"{V}{p}", sp, k) for k in range(len(specs))]
    monkeypatch.setattr(CR, "brow", brow)
    monkeypatch.setattr(CR, "mrow", mrow)
    monkeypatch.setattr(CR, "_check_items", lambda rowid, sp, specs, vals: vals)
    return calls, specs_seen


def test_dotplot_sealed_without_unseal(tmp_path, monkeypatch):
    calls, _ = _fake_loaders(monkeypatch)
    with pytest.raises(DP.SealedError):
        DP.check_unseal("test", False)
    DP.check_unseal("test", True)
    DP.check_unseal("val", False)
    assert DP.main(["--split-metric", "test", "--out-dir", str(tmp_path)]) == 2
    assert not calls and not list(tmp_path.iterdir())


def test_dotplot_reads_through_compiler_and_draws(tmp_path, monkeypatch):
    calls, specs_seen = _fake_loaders(monkeypatch)
    try:
        rc = DP.main(["--out-dir", str(tmp_path), "--variants", "runA=λ = 1", "runB=λ = 0.25"])
        assert rc == 0
        assert {c[-1] for c in calls} == {"val"}                 # validation fields only
        assert CR.FN_CV is True
        assert [s[3] for s in specs_seen[0]] == ["macro_ap"] * 4 + ["auroc"]
        for ext in ("pdf", "png", "json"):
            assert (tmp_path / f"function_dotplot_val.{ext}").stat().st_size > 0
        rec = json.loads((tmp_path / "function_dotplot_val.json").read_text())
        labels = [r["label"] for r in rec["rows"]]
        assert "random features" not in labels                    # reference line only
        assert set(rec["random_floor"]) == {f"{sp}|{t[0]}" for sp, _ in DP.SPLITS for t in DP.TASKS}
        assert [r["group"] for r in rec["rows"]] == ["single-frame"] * 7 + ["multi-frame"] * 4
        ens = [r for r in rec["rows"] if r["variant"]]
        assert {(r["variant"], r["P"]) for r in ens} == {("runA", "1"), ("runA", "full"),
                                                         ("runB", "1"), ("runB", "full")}
        assert ens[0]["color_idx"] != ens[1]["color_idx"]
        assert not rec["refused"]
    finally:
        CR.set_split_metric("test")


def test_dotplot_refused_cell_not_drawn(tmp_path, monkeypatch):
    calls, _ = _fake_loaders(monkeypatch, refused={("aminoaseed", "structure", 3)})
    try:
        assert DP.main(["--out-dir", str(tmp_path)]) == 1
        rec = json.loads((tmp_path / "function_dotplot_val.json").read_text())
        assert rec["refused"] == [{"row": "AminoAseed", "split": "structure",
                                   "task": "GO molecular function (mAP)", "status": "refused",
                                   "n_seeds": 0}]
        row = next(r for r in rec["rows"] if r["label"] == "AminoAseed")
        assert row["cells"]["structure"]["GO molecular function (mAP)"]["mean"] is None
        assert DP.main(["--out-dir", str(tmp_path), "--allow-incomplete"]) == 0
    finally:
        CR.set_split_metric("test")
        CR.ALLOW_INCOMPLETE = False


def test_dotplot_enforces_ten_seeds(tmp_path, monkeypatch):
    _fake_loaders(monkeypatch)
    orig = CR.brow
    monkeypatch.setattr(CR, "brow", lambda F, sp, specs: [(0.3, 0.01, 7)] * len(specs)
                        if F == "3di_tokens" else orig(F, sp, specs))
    try:
        data = DP.collect(DP.parse_variants(["runA=a"]), "val", allow_incomplete=False)
        bad = {(r["row"], r["status"], r["n_seeds"]) for r in data["refused"]}
        assert bad == {("3Di", "refused", 7)}
    finally:
        CR.set_split_metric("test")


# ---------------------------------------------------------------- A9 recon_by_p / token usage
def _fake_run(root: Path, name: str, seed: int, n_val=40, n_train=30):
    """A tiny run dir with the real config.json layout (random weights), its cache and splits."""
    a = {
        "quantizer": "rvq", "geom_attn_encoder": False, "target_desc": None,
        "codebook_sizes": "16,4,4", "aminoaseed_projection": "0", "hidden_dim": 32,
        "latent_dim": 8, "commitment_cost": 0.5, "encoder_type": "set_transformer",
        "loss_type": "hungarian", "vq_update": "ema", "ema_decay": 0.99, "rotation_trick": False,
        "nsvq_alpha": 0.0, "nsvq_tau": 1.0, "n_queries": 2, "n_heads": 2,
        "n_encoder_layers": 1, "n_decoder_layers": 1, "rvq_residual": "standard",
        "recon_loss_type": "fape", "rect_hungarian": True, "hungarian_cost": "loss",
        "dihedral_weight": 3.0, "fape_clamp": 10.0, "batch_size": 16, "seed": 42,
        "desc_dtype": "float16", "consistency_mse_weight": 0.5, "random_p_range": "1,10",
        "no_standardize": True}
    a = dict(a)
    P, D = 10, 256
    desc = root / "desc.npy"
    if not desc.exists():
        rng = np.random.default_rng(0)
        np.save(desc, rng.normal(size=(n_val + n_train, P, D)).astype(np.float16))
        (root / "splits.json").write_text(json.dumps({"val": list(range(n_val)),
                                                      "train": list(range(n_val, n_val + n_train))}))
    a.update(desc=str(desc), splits=str(root / "splits.json"))
    rd = root / name
    rd.mkdir()
    torch.manual_seed(seed)
    m = RB.build_model(a, P, D)
    torch.save(m.state_dict(), rd / "best.pt")
    np.savez(rd / "stats.npz", mean=np.zeros(D, np.float32), std=np.ones(D, np.float32))
    # Compute actual P=10 loss so best_val_recon matches (needed for A.2 repro_diff check)
    actual_loss = RB.recon_loss(m.eval(), a,
                                np.load(desc, mmap_mode="r")[:n_val], P, "cpu")
    # A.1 fix: best_val_recon must equal history val_recon at codebook_health.epoch
    (rd / "history.json").write_text(json.dumps({"val_recon": [9.0, actual_loss, 8.5],
                                                  "eval_epochs": [5, 10, 15]}))
    cfg = {"args": a, "best_val_recon": actual_loss, "total_epochs": 15, "n_val_residues": n_val,
           "per_level": [{"level": 0, "M": int(str(a["codebook_sizes"]).split(",")[0]),
                          "unique": 0, "ppl": 0.0}],
           "codebook_health": {"epoch": 10}}
    (rd / "config.json").write_text(json.dumps(cfg))
    return rd, a


def test_recon_by_p_matches_forward_and_verify(tmp_path, monkeypatch):
    from ensembits.losses import fape_hungarian_loss
    rd1, a = _fake_run(tmp_path, "run1", 1)
    rd2, _ = _fake_run(tmp_path, "run2", 2)
    val, tr = RB.load_split_data({"args": a}, 10)
    assert val.shape[0] == 40 and tr.shape == (10, 10, 256)
    row = RB.evaluate_run(str(rd1), val, tr, [10, 1], "cpu", a)
    assert row["best_epoch"] == 10 and row["total_epochs"] == 15
    # independent recomputation with the model's forward() recon and the trainer's loss
    # A.2 fix: test uses row-weighted mean (matching the trainer's per-residue mean)
    model, _ = RB.load_run(rd1, 10, 256, "cpu")
    with torch.no_grad():
        losses, weights = [], []
        for s in range(0, 40, a["batch_size"]):
            vb = torch.from_numpy(np.array(val[s:s + a["batch_size"]])).float()[:, :1]
            out = model(vb)
            losses.append(fape_hungarian_loss(out["recon"], vb, a["dihedral_weight"], a["fape_clamp"],
                                              match_cost=a["hungarian_cost"]).item())
            weights.append(vb.shape[0])
    assert row["val_P1"] == pytest.approx(np.average(losses, weights=weights), rel=1e-6)
    # verify / reuse
    out = tmp_path / "out.json"
    full = tmp_path / "full.json"
    assert RB.main(["--runs", f"{rd1},{rd2}", "--ps", "10,1", "--train-sub", "0",
                    "--device", "cpu", "--out", str(full)]) == 0
    res = json.loads(full.read_text())["results"]
    assert RB.main(["--runs", f"{rd1},{rd2}", "--ps", "10,1", "--train-sub", "0", "--device",
                    "cpu", "--out", str(out), "--reuse", str(full), "--verify-run", str(rd1)]) == 0
    got = json.loads(out.read_text())
    assert got["results"] == res and got["provenance"]["verified_run"] == str(rd1)
    bad = json.loads(full.read_text()); bad["results"][str(rd1)]["val_P1"] += 1e-9
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    out.unlink()
    assert RB.main(["--runs", f"{rd1},{rd2}", "--ps", "10,1", "--train-sub", "0", "--device",
                    "cpu", "--out", str(out), "--reuse", str(tmp_path / "bad.json"),
                    "--verify-run", str(rd1)]) == 1
    assert not out.exists()


def test_token_usage(tmp_path):
    u = TU.usage([0, 0, 1, 3], 8)
    assert u["distinct"] == 3 and u["util"] == 3 / 8 and u["n"] == 4
    p = np.array([0.5, 0.25, 0.25])
    assert u["ppl"] == pytest.approx(np.exp(-(p * np.log(p)).sum()))
    assert TU.usage([5] * 10, 8)["ppl"] == pytest.approx(1.0)
    rd, a = _fake_run(tmp_path, "run1", 1)
    res = TU.from_validation([str(rd)], [1, 10], "cpu", 2)[str(rd)]
    assert res["P1"]["n"] == 40 and 1 <= res["P1"]["distinct"] <= 16
    assert res["P10_check"]["distinct_diff"] == res["P10"]["distinct"]   # config says 0
    model, _ = RB.load_run(rd, 10, 256, "cpu")
    val, _ = RB.load_split_data({"args": a}, 0)
    ids = TU.level1_ids(model, val, 1, 7, "cpu")
    assert TU.usage(ids, 16) == res["P1"]


# ---------------------------------------------------------------- A9 compiler tables
def _abl_fixture(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    runs = ["ensembits_bm_e0", "ensembits_bm_vq2048", "ensembits_bm_nosftd", "ensembits_bm_lam1"]
    shas = {}
    for r in runs:
        d = root / "output" / r; d.mkdir(parents=True)
        (d / "best.pt").write_bytes(r.encode())
        from ensembits.featurize import file_sha256
        shas[r] = file_sha256(d / "best.pt")
    key = CR._run_key
    recon = {"results": {key(r): {"best_epoch": 10, "total_epochs": 50, "val_P1": 6.5 + i,
                                  "val_P2": 6.9, "val_P5": 7.0, "val_P10": 7.2}
                         for i, r in enumerate(runs)},
             "provenance": {"runs": {key(r): {"best_pt_sha256": shas[r]} for r in runs}}}
    recon["provenance"]["runs"]["lam1"]["best_pt_sha256"] = "0" * 64        # stale -> refused
    usage = {"results": {key(r): {"P1": {"distinct": 1 if "nosftd" in r else 1800, "util": 0.9,
                                         "ppl": 1.0 if "nosftd" in r else 1600.0},
                                  "P10": {"distinct": 1900, "util": 0.93, "ppl": 1650.0}}
                         for r in runs},
             "provenance": {"runs": {key(r): {"best_pt_sha256": shas[r]} for r in runs}}}
    (root / "output" / "ablations").mkdir(parents=True)
    (root / "output" / "ablations" / "recon_by_p.json").write_text(json.dumps(recon))
    (root / "output" / "ablations" / "token_usage_by_p.json").write_text(json.dumps(usage))
    meth, ho_s1, ho_rm, an = [], {}, {}, {}
    for r in runs[:3]:                                      # lam1 has no ANOVA -> missing
        for p in (10, 1):
            k = f"ens:{r}:P{p}"
            meth.append({"key": k, "prov": {"ckpt_sha256_12": shas[r][:12]}})
            ho_s1[k] = {"r2": 0.2, "ci95": [0.19, 0.21]}
            ho_rm[k] = {"r2": 0.4, "ci95": [0.38, 0.42]}
            an[k] = {"M": 1 if (r.endswith("nosftd") and p == 1) else 1800}
    anova = {"methods": meth, "analysis": {"targets": {"s1": {"heldout_r2": ho_s1, "anova": an},
                                                       "rmsf_allframes": {"heldout_r2": ho_rm}}}}
    af = root / "anova.json"; af.write_text(json.dumps(anova))
    monkeypatch.setattr(CR, "ROOT", root)
    monkeypatch.setattr(CR, "ABL_RECON", root / "output" / "ablations" / "recon_by_p.json")
    monkeypatch.setattr(CR, "ABL_USAGE", root / "output" / "ablations" / "token_usage_by_p.json")
    monkeypatch.setattr(CR, "ANOVA_FILES", [af])
    seen = []
    def rm(stem, p1, *rest, key=None):
        seen.append((stem, p1))
        if stem.endswith("nosftd") and p1:
            return CR.Constant(10)
        return (0.6, 0.01, 10)
    monkeypatch.setattr(CR, "rmsf_div", lambda stem, p1, key=None: rm(stem, p1))
    monkeypatch.setattr(CR, "rmsf_bio", lambda stem, p1, key=None: rm(stem, p1))
    monkeypatch.setattr(CR, "rmsf_mis", lambda stem, split, p1, key=None: rm(stem, p1))
    return runs, seen


@pytest.mark.parametrize("mode", ["test", "val"])
def test_ablation_tables(tmp_path, monkeypatch, mode):
    runs, seen = _abl_fixture(tmp_path, monkeypatch)
    CR.set_rmsf_protocol("test"); CR.set_split_metric(mode)
    try:
        inc = []
        md = "\n".join(CR.ablation_tables(
            [("ensembits_bm_e0", "e0"), ("ensembits_bm_vq2048", "VQ")],
            [("ensembits_bm_nosftd", "SFTD off"), ("ensembits_bm_e0", "λ = 0.1 (e0)"),
             ("ensembits_bm_lam1", "λ = 1")], inc))
    finally:
        CR.set_split_metric("test")
    assert "## Ablations: VQ (E1)" in md and "## Ablations: SFTD (E2)" in md
    for bad in ("EC d1", "GO mAP", "Bind AUROC", "Aff "):
        assert bad not in md                                   # no function columns
    assert ("validation Spearman" in md) == (mode == "val")
    assert "RVQ decodes from 3 levels" in md
    row = lambda lab: next(l for l in md.splitlines() if l.startswith(f"| {lab} |"))
    # VQ table: P = full / P = 1 rows with usage, recon, RMSF, ANOVA
    assert "1,900 (93%)" in row("e0 · P = full") and "7.200" in row("e0 · P = full")
    assert "0.200 [0.190, 0.210]" in row("VQ · P = 1") and "7.500" in row("VQ · P = 1")
    # SFTD off at P = 1: one token -> —† in RMSF and ANOVA
    r = row("SFTD off · P = 1")
    assert r.count(CR.RC.CONSTANT_MARK) == 7 and "0.600" not in r
    assert "1 (90%)" in row("SFTD off") and "10 / 50" in row("SFTD off")
    # λ = 1: stale recon checkpoint -> refused; no ANOVA row -> missing (incomplete)
    assert CR.REFUSED_MARK in row("λ = 1") and "## Refused ablation cells" in md
    assert any(r[0] == "ensembits_bm_lam1" and r[1] == "recon" for r in CR._ABL_REFUSED)
    assert ("ablations", "ensembits_bm_lam1", "ANOVA P=10", 0) in inc
    assert ("ensembits_bm_nosftd", True) in seen and ("ensembits_bm_vq2048", False) in seen


def test_compiler_main_has_ablation_args():
    import inspect
    src = inspect.getsource(CR.main)
    assert "--abl-vq" in src and "--abl-sftd" in src and "--no-ablation-tables" in src
    # FIXED(recon2 A.7, 2026-10-25): defaults switched to _rv runs.
    assert [v for v, _ in CR.DEFAULT_ABL_SFTD] == [
        "ensembits_bm_nosftd_rv", "ensembits_bm_lam0_rv", "ensembits_bm_lam0p05_rv", "ensembits_bm_e0_rv",
        "ensembits_bm_lam0p25_rv", "ensembits_bm_lam0p5_rv", "ensembits_bm_lam1_rv"]
    assert [v for v, _ in CR.DEFAULT_ABL_VQ] == ["ensembits_bm_e0_rv", "ensembits_bm_vq2048_rv"]


# B.10: tests for _resolve_result_key
def test_resolve_result_key():
    # short key
    d = {"e0": 1, "lam1": 2}
    assert CR._resolve_result_key(d, "ensembits_bm_e0") == "e0"
    assert CR._resolve_result_key(d, "ensembits_bm_lam1") == "lam1"
    # absolute path key (from _rv files)
    d2 = {"/long/path/ensembits_bm_e0_rv": 1}
    assert CR._resolve_result_key(d2, "ensembits_bm_e0_rv") == "/long/path/ensembits_bm_e0_rv"
    # missing
    assert CR._resolve_result_key(d, "ensembits_bm_missing") is None


def test_abl_entry_refuses_bad_repro_diff(tmp_path, monkeypatch):
    """A.2 fix: _abl_entry refuses rows with |repro_diff| > 1e-6."""
    root = tmp_path / "repo"
    rd = root / "output" / "ensembits_bm_e0"; rd.mkdir(parents=True)
    (rd / "best.pt").write_bytes(b"test")
    from ensembits.featurize import file_sha256
    sha = file_sha256(rd / "best.pt")
    recon = {"results": {"e0": {"val_P10": 7.2, "repro_diff": 1e-3}},
             "provenance": {"runs": {"e0": {"best_pt_sha256": sha}}}}
    rp = root / "output" / "ablations" / "recon_by_p.json"
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(recon))
    monkeypatch.setattr(CR, "ROOT", root)
    CR._ABL_CACHE.clear(); CR._ABL_REFUSED.clear()
    result = CR._abl_entry(str(rp), "ensembits_bm_e0", "recon")
    assert result == "refused"
    assert any("repro_diff" in r[2] for r in CR._ABL_REFUSED)
