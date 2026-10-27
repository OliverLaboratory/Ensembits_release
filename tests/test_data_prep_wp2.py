"""Tests of the pre-retrain data-preparation fixes of 2026-10-16 (code_fix P5, P14, P28, P29).

What it does
------------
  * `build_misato_real_bb` (P14): the resume check refuses an existing file with a
    different K (or fps, or no K / fps), both in `check_out_dir` and in the
    per-pid skip path; the default output directory follows from K.
  * Baseline chain split (P5): `split_chains_by_peptide_bond` and ProtProfileMD's
    `_split_chains` break at C–N > 2.5 Å.
  * Descriptor (P29): fewer than K = 16 residues raises a clear ValueError.
  * `build_misato_rmsf_labels` (P28): writes `<out>.provenance.json` with the hash
    of the coordinates it used, and refuses coordinates that do not match their
    own sidecar. Runs on a tiny synthetic pickle in a temporary folder.
All files are written under pytest's tmp_path; nothing in REPO_ROOT is touched.

Paper
-----
not described in the paper (infrastructure only)

How to run
----------
  python -m pytest tests/test_data_prep_wp2.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "data_prep"))


def _write_misato_npz(path, K=8, fps="kabsch", parser=None, with_k_fps=True):
    from ensembits.data_versions import MISATO_PARSER_VERSION
    fields = {f"bb_{K}": np.zeros((K, 3, 4, 3), np.float32),
              "parser_version": np.array(parser or MISATO_PARSER_VERSION)}
    if with_k_fps:
        fields["K"] = np.array(K, dtype=np.int64)
        fields["fps"] = np.array(fps)
    np.savez_compressed(path, **fields)


# ---- P14: build_misato_real_bb resume / out-dir ---------------------------------------
def test_misato_real_bb_resume_refuses_other_K(tmp_path):
    pytest.importorskip("h5py")
    import build_misato_real_bb as B
    _write_misato_npz(tmp_path / "1ABC.npz", K=10)
    # The whole-directory check refuses it ...
    with pytest.raises(SystemExit):
        B.check_out_dir(tmp_path, K=8, fps="kabsch")
    # ... and so does the per-pid skip path (no HDF5 access: the file exists).
    pid, status = B._process_pid("1ABC", K=8, fps="kabsch", out_dir=tmp_path)
    assert status.startswith("mismatch") and "K" in status
    # A matching file is kept.
    B.check_out_dir(tmp_path, K=10, fps="kabsch")
    assert B._process_pid("1ABC", K=10, fps="kabsch", out_dir=tmp_path)[1] == "skip"
    # Other fps, and files without K / fps (written before 2026-10-16), are refused.
    assert B.existing_file_mismatch(tmp_path / "1ABC.npz", 10, "frobenius") is not None
    _write_misato_npz(tmp_path / "2XYZ.npz", K=10, with_k_fps=False)
    with pytest.raises(SystemExit):
        B.check_out_dir(tmp_path, K=10, fps="kabsch")


def test_misato_real_bb_default_out_dir():
    pytest.importorskip("h5py")
    import build_misato_real_bb as B
    assert B.default_out_dir(8).name == "misato_real_bb"
    assert B.default_out_dir(10).name == "misato_real_bb_kabsch"
    assert B.default_out_dir(10, "frobenius").name not in ("misato_real_bb", "misato_real_bb_kabsch")


# ---- P5: chain-break threshold of the baselines ---------------------------------------
def _two_chain_bb(gap):
    """(6, 4, 3) backbone: residues 0-2 and 3-5 with C(2)–N(3) = `gap` Å, bonded C–N 1.33 Å."""
    bb = np.zeros((6, 4, 3), np.float32)
    x = 0.0
    for i in range(6):
        if i:
            x += gap if i == 3 else 1.33
        bb[i, 0] = [x, 0, 0]
        bb[i, 1] = [x + 1.46, 0.5, 0]
        bb[i, 2] = [x + 2.5, 0, 0]
        bb[i, 3] = [x + 2.5, 1.2, 0]
        x += 2.5
    return bb


def test_baseline_split_threshold_2p5():
    from ensembits.baselines._real_or_ideal_bb import (CHAIN_BREAK_THRESHOLD,
                                                       split_chains_by_peptide_bond)
    assert CHAIN_BREAK_THRESHOLD == 2.5
    assert split_chains_by_peptide_bond(_two_chain_bb(3.0)) == [(0, 3), (3, 6)]
    assert split_chains_by_peptide_bond(_two_chain_bb(2.0)) == [(0, 6)]
    sys.path.insert(0, str(CODE_ROOT / "data_prep" / "baselines"))
    import build_protprofile_K_misato_foldseek as PP
    assert PP._split_chains(_two_chain_bb(3.0)) == [(0, 3), (3, 6)]


# ---- P29: descriptor needs L >= K -----------------------------------------------------
def test_descriptor_short_protein_raises():
    from ensembits.descriptor import K, _descriptor
    with pytest.raises(ValueError, match="at least K"):
        _descriptor(torch.randn(2, K - 1, 3, 3), "cpu")


# ---- P28: RMSF label provenance -------------------------------------------------------
def test_rmsf_labels_write_provenance(tmp_path):
    # (2026-10-16: the default label is now "allframes" from MD.hdf5; this tests the old
    #  8-frame label, selected with --label frames8; tests/test_rmsf_probes.py tests allframes)
    import build_misato_rmsf_labels as R
    from ensembits.data_versions import hash_arrays_by_pid, read_provenance, write_provenance
    rng = np.random.default_rng(0)
    coords = {f"P{i}": rng.normal(size=(8, 20, 3)).astype(np.float32) for i in range(3)}
    cpath, out = tmp_path / "coords_ca8.pkl", tmp_path / "rmsf.npz"
    pickle.dump(coords, open(cpath, "wb"))

    # No coords sidecar: refused unless --allow-no-provenance.
    assert R.main(["--label", "frames8", "--coords", str(cpath), "--out", str(out)]) == 1 and not out.exists()

    write_provenance(cpath, {"coords_sha256": hash_arrays_by_pid(coords),
                             "parser_version": ["cx_test"], "source_dir": "tmp"})
    assert R.main(["--label", "frames8", "--coords", str(cpath), "--out", str(out)]) == 0
    prov = read_provenance(out)
    assert prov["coords_sha256"] == hash_arrays_by_pid(coords)
    assert prov["parser_version"] == ["cx_test"] and prov["n_pids"] == 3
    labels = np.load(out)
    assert sorted(labels.files) == sorted(coords)            # no extra keys in the npz
    assert prov["labels_sha256"] == hash_arrays_by_pid({k: labels[k] for k in labels.files})
    assert json.loads((tmp_path / "rmsf.npz.provenance.json").read_text()) == prov

    # Coordinates changed after their sidecar was written: refused.
    coords["P0"] = coords["P0"] + 1.0
    pickle.dump(coords, open(cpath, "wb"))
    out2 = tmp_path / "rmsf2.npz"
    assert R.main(["--label", "frames8", "--coords", str(cpath), "--out", str(out2)]) == 1 and not out2.exists()
