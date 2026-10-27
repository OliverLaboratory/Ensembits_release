"""Smoke test for plot_training_curve.py: synthetic history, check outputs."""
import json
import tempfile
from pathlib import Path

import pytest


def _make_history(d: Path, n_train=20, n_eval=4):
    import numpy as np
    rng = np.random.RandomState(42)
    tr = 8.0 - np.log1p(np.arange(n_train)).astype(float) * 0.3 + rng.randn(n_train) * 0.01
    vr = tr[::n_train // n_eval][:n_eval] + 0.05
    history = {
        "train_recon": tr.tolist(),
        "train_vq": (rng.rand(n_train) * 0.2).tolist(),
        "val_recon": vr.tolist(),
        "eval_epochs": list(range(n_train // n_eval, n_train + 1, n_train // n_eval)),
        "util_L0": [0.95 + i * 0.01 for i in range(n_eval)],
        "util_L1": [1.0] * n_eval,
        "util_L2": [1.0] * n_eval,
        "ppl_L0": [1500 + i * 30 for i in range(n_eval)],
        "ppl_L1": [100.0] * n_eval,
        "ppl_L2": [100.0] * n_eval,
        "train_consistency": (rng.rand(n_train) * 0.05).tolist(),
        "codebook_health": [{}] * n_eval,
    }
    (d / "history.json").write_text(json.dumps(history))
    config = {"codebook_sizes": [2048, 128, 128], "num_prototypes": 10,
              "args": {"consistency_mse_weight": 10.0}}
    (d / "config.json").write_text(json.dumps(config))
    return history


def test_training_curve_smoke():
    import sys, importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream" / "analyses"))
    with tempfile.TemporaryDirectory() as td:
        md = Path(td) / "model"
        md.mkdir()
        _make_history(md)
        out = Path(td) / "out"
        # patch best.pt so ckpt_hash works
        (md / "best.pt").write_bytes(b"fake")
        import plot_training_curve as ptc
        rc = ptc.main(["--model-dir", str(md), "--out-dir", str(out)])
        assert rc == 0
        assert (out / "training_curve.pdf").exists()
        assert (out / "training_curve.png").exists()
        prov = json.loads((out / "training_curve.provenance.json").read_text())
        assert "best_epoch" in prov
        assert prov["n_train_epochs"] == 20
