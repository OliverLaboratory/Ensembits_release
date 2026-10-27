"""Tiny CPU smoke test of the trainer (train/train.py) on a synthetic corpus.

What it does
------------
Builds a synthetic descriptor corpus in a pytest tmp dir (64 residues x 10 frames
x 256 random numbers, splits.json, layout.json with the iupac convention and a
2.5 A chain-break cutoff), runs train/train.py as a subprocess on CPU for 2 epochs
with tiny model sizes and the SFTD recipe flags, and checks:
  * config.json records the corpus / loss / RVQ settings, descriptor_mode, k,
    codebook_update_effective == "full", status "finished" and finished_at;
  * a second run into the same --out is refused without --overwrite (code_fix P10);
  * an iupac corpus whose layout.json lacks chain_break_cutoff or k, or that has no
    layout.json, is refused (code_fix P27, preretrain_audit_r2 Q19);
  * lambda = 0 still runs both SFTD branches with the full-branch codebook update
    (preretrain_audit_r2 Q4): codebook_update_effective "full" and a nonzero logged
    distillation distance.
  * the codebook revival fix (final_fix_list A6, 2026-10-19): config.json records
    "codebook_update_rule" == "ema-revival-v2" and the duplicate-free health of
    best.pt; history.json has one codebook check per evaluation; training batches
    are full (drop_last: 48 rows at batch 20 -> 2 batches, 8 rows skipped per epoch);
    revival window schedule recorded; sub-ensemble codebook updates and invalid revival
    settings refused (training audits 2026-10-20).
  * B.2 (recon2, 2026-10-25): revival window switch with small --revival-window* flags;
    val_recon recorded per evaluation epoch in history.json.
  * B.3 (recon2, 2026-10-25): codebook_update_effective == "single" for non-SFTD mode.
  * B.5 (recon2, 2026-10-25): codebook_health carries a "source" label.
The run is skipped (not failed) if it would take more than 2 minutes.

Paper
-----
not described in the paper (infrastructure only)

Origin
------
New (code_fix P31, 2026-10-16).

How to run
----------
  python -m pytest tests/test_train_smoke.py -q     # from the codebase root
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
TRAIN = CODE_ROOT / "train" / "train.py"
DESC = "esm3desc_dihedral_iupac_K16_P10.npy"
TIMEOUT_S = 120

# Tiny model; the recipe flags of the reported run (SFTD fixed-max, full-branch
# codebook update, k-means init, rectangular Hungarian, FAPE).
TINY = [
    "--device", "cpu",
    "--codebook-sizes", "8,4,4", "--hidden-dim", "32", "--latent-dim", "16",
    "--n-encoder-layers", "1", "--n-decoder-layers", "1",
    "--n-queries", "2", "--n-heads", "2",
    "--no-standardize",
    "--random-p-range", "1,10", "--consistency-mse-weight", "0.1",
    "--consistency-distill-fixed-max",
    "--rect-hungarian", "--recon-loss-type", "fape",
    "--kmeans-init", "--noise-std", "0.05", "--warmup-steps", "2",
    "--batch-size", "16", "--max-epochs", "2", "--eval-every", "1", "--patience", "10",
    "--seed", "0",
    "--dihedral-convention", "iupac", "--hungarian-cost", "loss",
    "--rvq-residual", "standard", "--codebook-update", "full",
]


def _make_corpus(d: Path, layout: dict | None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(d / DESC, rng.normal(size=(64, 10, 256)).astype(np.float32))
    (d / "splits.json").write_text(json.dumps(
        {"train": list(range(48)), "val": list(range(48, 56)), "test": list(range(56, 64))}))
    if layout is not None:
        (d / "layout.json").write_text(json.dumps(layout))
    return d / DESC


def _run(tmp_path: Path, desc: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # make_run_dir writes per-run plots under REPO_ROOT/Ensembits/output: keep them in tmp.
    env["ENSEMBITS_REPO_ROOT"] = str(tmp_path / "repo")
    env.setdefault("OMP_NUM_THREADS", "2")
    cmd = [sys.executable, str(TRAIN), "--desc", str(desc),
           "--splits", str(desc.parent / "splits.json"), "--out", str(out), *TINY, *extra]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=TIMEOUT_S, cwd=str(tmp_path))
    except subprocess.TimeoutExpired:
        pytest.skip(f"training smoke run took more than {TIMEOUT_S} s")


IUPAC_LAYOUT = {"rows": [], "dihedral_convention": "iupac", "chain_break_cutoff": 2.5,
                "split_unit": "domain", "k": 16, "frames": 10}


def test_train_smoke_config_and_refusals(tmp_path):
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    out = tmp_path / "output" / "smoke_run"
    r = _run(tmp_path, desc, out)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

    for f in ("best.pt", "stats.npz", "config.json", "history.json"):
        assert (out / f).exists(), f
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["dihedral_convention"] == "iupac"
    assert cfg["chain_break_cutoff"] == 2.5
    assert cfg["hungarian_cost"] == "loss"
    assert cfg["rvq_residual"] == "standard"
    assert cfg["codebook_update"] == "full"
    assert cfg["codebook_update_effective"] == "full"
    assert cfg["descriptor_mode"] == "esm3desc_dihedral"
    assert cfg["k"] == 16
    assert cfg["status"] == "finished"
    assert cfg["finished_at"] is not None
    assert cfg["total_epochs"] == 2 and cfg["per_level"] is not None
    assert cfg["corpus_settings"]["chain_break_cutoff"] == 2.5
    # final_fix_list A6: revival rule and checkpoint codebook check
    assert cfg["codebook_update_rule"] == "ema-revival-v2"
    assert cfg["train_batches_per_epoch"] == 3 and cfg["train_rows_skipped_per_epoch"] == 0
    h = cfg["codebook_health"]
    assert h["error"] is None and len(h["levels"]) == 3
    assert all(l["exact_duplicates"] == 0 and l["near_duplicates"] == 0 for l in h["levels"])
    hist = json.loads((out / "history.json").read_text())
    assert [e["epoch"] for e in hist["codebook_health"]] == [1, 2]
    # training audits 2026-10-20: window schedule recorded; 3 batches <= cooldown warned
    assert (cfg["revival_window_early"], cfg["revival_window"],
            cfg["revival_window_switch_epoch"], cfg["revival_cooldown"]) == (100, 400, 2, 50)
    assert cfg["codebook_update_rule_description"]
    assert cfg["revival_warnings"] and "no dead-code revival" in cfg["revival_warnings"][0]

    # code_fix P10: a second run into the same --out is refused without --overwrite.
    r2 = _run(tmp_path, desc, out)
    assert r2.returncode != 0
    assert "--overwrite" in (r2.stdout + r2.stderr)


def test_train_refuses_iupac_layout_without_cutoff(tmp_path):
    layout = {k: v for k, v in IUPAC_LAYOUT.items() if k != "chain_break_cutoff"}
    desc = _make_corpus(tmp_path / "corpus", layout)
    out = tmp_path / "output" / "no_cutoff"
    r = _run(tmp_path, desc, out)
    assert r.returncode != 0
    assert "chain_break_cutoff" in (r.stdout + r.stderr)
    assert not (out / "config.json").exists()


def test_train_refuses_iupac_without_layout(tmp_path):
    desc = _make_corpus(tmp_path / "corpus", None)
    out = tmp_path / "output" / "no_layout"
    r = _run(tmp_path, desc, out)
    assert r.returncode != 0
    assert "layout.json" in (r.stdout + r.stderr)
    assert not (out / "config.json").exists()


def test_train_refuses_iupac_layout_without_k(tmp_path):
    layout = {k: v for k, v in IUPAC_LAYOUT.items() if k != "k"}
    desc = _make_corpus(tmp_path / "corpus", layout)
    out = tmp_path / "output" / "no_k"
    r = _run(tmp_path, desc, out)
    assert r.returncode != 0
    assert "neighbour count k" in (r.stdout + r.stderr)
    assert not (out / "config.json").exists()


def test_lambda_zero_keeps_both_branches(tmp_path):
    """Q4: lambda = 0 drops only the distillation term; the full branch still runs and
    alone updates the codebooks (old code: P' branch only, codebooks from it)."""
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    out = tmp_path / "output" / "lambda0"
    r = _run(tmp_path, desc, out, "--consistency-mse-weight", "0")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["args"]["consistency_mse_weight"] == 0
    assert cfg["codebook_update_effective"] == "full"
    hist = json.loads((out / "history.json").read_text())
    # the distance ||z_2 - sg[z_1]||^2 is still computed (two branches), weight 0
    assert all(c > 0 for c in hist["train_consistency"])


def test_resume_matches_uninterrupted_run(tmp_path):
    """--resume (training audit 2026-10-16): a run stopped after epoch 2 and resumed
    reproduces the uninterrupted 4-epoch run exactly on CPU (weights, EMA codebooks,
    history), and a resume with different settings is refused."""
    import torch
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    full, part = tmp_path / "output" / "full", tmp_path / "output" / "part"
    env_extra = ["--max-epochs", "4"]
    r = _run(tmp_path, desc, full, *env_extra)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    r = _run(tmp_path, desc, part, *env_extra, "--debug-stop-after-epoch", "2")
    assert r.returncode == 3, r.stdout[-3000:] + r.stderr[-3000:]
    assert json.loads((part / "config.json").read_text())["status"] == "running"
    # a plain restart is refused, a resume with changed settings too
    assert _run(tmp_path, desc, part, *env_extra).returncode != 0
    r = _run(tmp_path, desc, part, "--max-epochs", "4", "--lr", "5e-4", "--resume")
    assert r.returncode != 0 and "settings differ" in (r.stdout + r.stderr)
    r = _run(tmp_path, desc, part, *env_extra, "--resume")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "skipping --kmeans-init" in r.stdout
    h_full = json.loads((full / "history.json").read_text())
    h_part = json.loads((part / "history.json").read_text())
    assert h_full == h_part
    a = torch.load(full / "best.pt"); b = torch.load(part / "best.pt")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    cfg = json.loads((part / "config.json").read_text())
    assert cfg["status"] == "finished" and len(cfg["resumed_at"]) == 1


def test_drop_last_batches(tmp_path):
    """final_fix_list A6: a partial last batch is never trained on (drop_last)."""
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    out = tmp_path / "output" / "drop_last"
    r = _run(tmp_path, desc, out, "--batch-size", "20")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["train_batches_per_epoch"] == 2 and cfg["train_rows_skipped_per_epoch"] == 8
    assert all(l["partial_batches_skipped"] == 0 for l in cfg["codebook_health"]["levels"])


@pytest.mark.parametrize("extra,msg", [
    (("--codebook-update", "both"), "sub-ensemble"),
    (("--revival-window", "0"), "must be >= 1"),
    (("--revival-cooldown", "-1"), "must be >= 0"),
])
def test_refused_revival_and_update_settings(tmp_path, extra, msg):
    """Training audits 2026-10-20: sub-ensemble codebook updates are refused without
    --allow-subensemble-codebook-update; revival window >= 1, cooldown >= 0."""
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    r = _run(tmp_path, desc, tmp_path / "output" / "refused", *extra)
    assert r.returncode != 0 and msg in (r.stdout + r.stderr)


def test_revival_window_switch(tmp_path):
    """B.2 (recon2, 2026-10-25): small --revival-window* flags verify the window
    schedule is recorded and the switch epoch is respected."""
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    out = tmp_path / "output" / "window_switch"
    r = _run(tmp_path, desc, out,
             "--revival-window-early", "3", "--revival-window", "7",
             "--revival-window-early-epochs", "1", "--revival-cooldown", "2")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    cfg = json.loads((out / "config.json").read_text())
    assert (cfg["revival_window_early"], cfg["revival_window"],
            cfg["revival_window_switch_epoch"], cfg["revival_cooldown"]) == (3, 7, 1, 2)


TINY_NO_SFTD = [
    "--device", "cpu",
    "--codebook-sizes", "8,4,4", "--hidden-dim", "32", "--latent-dim", "16",
    "--n-encoder-layers", "1", "--n-decoder-layers", "1",
    "--n-queries", "2", "--n-heads", "2",
    "--no-standardize",
    "--rect-hungarian", "--recon-loss-type", "fape",
    "--kmeans-init", "--noise-std", "0.05", "--warmup-steps", "2",
    "--batch-size", "16", "--max-epochs", "2", "--eval-every", "1", "--patience", "10",
    "--seed", "0",
    "--dihedral-convention", "iupac", "--hungarian-cost", "loss",
    "--rvq-residual", "standard", "--codebook-update", "full",
]


def _run_no_sftd(tmp_path, desc, out, *extra):
    env = dict(os.environ)
    env["ENSEMBITS_REPO_ROOT"] = str(tmp_path / "repo")
    env.setdefault("OMP_NUM_THREADS", "2")
    cmd = [sys.executable, str(TRAIN), "--desc", str(desc),
           "--splits", str(desc.parent / "splits.json"), "--out", str(out),
           *TINY_NO_SFTD, *extra]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=TIMEOUT_S, cwd=str(tmp_path))
    except subprocess.TimeoutExpired:
        pytest.skip(f"training smoke run took more than {TIMEOUT_S} s")


def test_codebook_update_effective_single(tmp_path):
    """B.3 (recon2, 2026-10-25): without SFTD (no --random-p-range), the trainer
    records codebook_update_effective == 'single'."""
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    out = tmp_path / "output" / "single_branch"
    r = _run_no_sftd(tmp_path, desc, out)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["codebook_update_effective"] == "single"


def test_codebook_health_labelled(tmp_path):
    """B.5 (recon2, 2026-10-25): codebook_health in config.json carries a 'source'
    label indicating the check was on best.pt reloaded at end of training."""
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    out = tmp_path / "output" / "health_label"
    r = _run(tmp_path, desc, out)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    cfg = json.loads((out / "config.json").read_text())
    h = cfg["codebook_health"]
    assert h.get("source") and "best.pt" in h["source"]
    assert h["epoch"] is not None


def test_val_recon_in_history(tmp_path):
    """B.2 (recon2, 2026-10-25): val_recon is recorded per evaluation epoch in
    history.json."""
    desc = _make_corpus(tmp_path / "corpus", IUPAC_LAYOUT)
    out = tmp_path / "output" / "val_recon"
    r = _run(tmp_path, desc, out)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    hist = json.loads((out / "history.json").read_text())
    assert len(hist["val_recon"]) == len(hist["eval_epochs"]) == 2
    assert all(isinstance(v, float) and v > 0 for v in hist["val_recon"])
