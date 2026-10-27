"""Tests of the corpus builders (`data_prep/`).

What it does
------------
  * MISATO parser (code_fix §4 / T1.1): on a few complexes of MD.hdf5, every CA
    is bonded to its C, every CB to its CA, and prolines have a real CA.
    Skipped when `REPO_ROOT/data/misato/MD.hdf5` is not present.
  * Corpus split (code_fix §5 / T1.5): the two simulations of a domain always
    land in the same split.

Paper
-----
`appendix.tex` \\subsection{Dataset} (\\label{app:dataset}).

How to run
----------
  python -m pytest tests/test_data_prep.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))


def test_misato_parser_backbone_geometry():
    h5py = pytest.importorskip("h5py")
    import build_misato_real_bb as B
    if not B.H5_PATH.exists():
        pytest.skip("MISATO MD.hdf5 not available")
    with h5py.File(B.H5_PATH, "r") as f:
        pids = sorted(f.keys())[:: max(1, len(f) // 5)][:5]
        n_pro = 0
        for pid in pids:
            g = f[pid]
            P = B.parse_complex(g)
            xyz = g["trajectory_coordinates"][0]
            ca_c = np.linalg.norm(xyz[P["ca"]] - xyz[P["c"]], axis=-1)
            n_ca = np.linalg.norm(xyz[P["ca"]] - xyz[P["n"]], axis=-1)
            assert ca_c.max() < 1.7 and n_ca.max() < 1.7, pid
            cb = P["cb_present"]
            ca_cb = np.linalg.norm(xyz[P["cb"][cb]] - xyz[P["ca"][cb]], axis=-1)
            assert ca_cb.max() < 1.8, pid
            seq = np.array(list(P["seq"]))
            assert np.all(cb == (seq != "G")), pid      # CB exactly for non-glycine
            n_pro += int((seq == "P").sum())
        assert n_pro > 0


def test_domain_split_keeps_duplicates_together():
    from build_bioemu_misato_descriptor import assign_splits
    layout = [{"corpus": "bioemu_cath", "pid": f"d{i // 2}", "system": f"s{i}"} for i in range(200)]
    layout += [{"corpus": "misato", "pid": f"m{i}", "system": f"m{i}"} for i in range(800)]
    grp, n = assign_splits(layout, "domain", 42)
    assert n == 900
    for i in range(0, 200, 2):
        assert grp[i] == grp[i + 1]
    counts = {s: sum(v == s for v in grp.values()) for s in ("train", "val", "test")}
    assert counts["train"] > counts["val"] > 0 and counts["test"] > 0
