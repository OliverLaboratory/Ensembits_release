"""Tests of the token exemplar figure helpers (downstream/analyses/token_exemplars_lib.py).

What it does
------------
  * Canonical kNN: the anchor is first, K = 16 slots, no sequence-gap filter (the chain
    neighbours r - 1 and r + 1 are in), counts <= P, and with ten identical frames the set
    equals the descriptor's kNN list of that frame (`ensembits.geometry.knn_edges`).
  * 3-mers stay inside their chain piece; chain pieces split at chain changes and at
    C-N > 2.5 A; distance to the piece end.
  * Alignment: frames moved rigidly plus a local motion of the 3-mers are superposed back
    exactly on the non-3-mer C-alpha, and the 3-mer motion is kept.
  * The anchor frame puts the anchor C-alpha at the origin, C on the -x axis and N in the
    xy plane.
  * Multi-state PDB: PDB columns, a chain id per piece, TER records, one MODEL per frame.
  * Sequence identity: 1 for identical chains, low for unrelated ones.
  * All-atom loaders (skipped when the data are absent): the MISATO (MD.hdf5) and BioEmu
    (xtc + topology) loaders reproduce the descriptor's backbone bb_10 (N, CA, C, O) exactly
    at the descriptor frames.

Paper
-----
`appendix.tex` \\section{Token visualizations} (\\label{app:viz}): sixteen canonical kNN,
no sequence-gap filter, 3-mers, Kabsch on all C-alpha minus the 3-mers, all-atom frames.

How to run
----------
  python -m pytest tests/test_token_exemplars.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "downstream" / "analyses"))

import token_exemplars_lib as TL  # noqa: E402
from ensembits.paths import REPO_ROOT  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CORPUS_DIR = REPO_ROOT / "output" / "bioemu_cath_misato_e0" / "cached_desc"


@pytest.fixture(scope="module")
def ncac():
    return np.load(FIXTURES / "1ctfA00_P10_ncac.npz")["ncac"].astype(np.float32)   # (10, 68, 3, 3)


def _bb4(ncac):
    """(P, L, 4, 3) with a dummy O (only N/CA/C are used by the kNN)."""
    return np.concatenate([ncac, ncac[:, :, 2:3]], axis=2)


def _rot(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_canonical_knn_k16_no_gap(ncac):
    bb = _bb4(ncac)
    edges = TL.knn_edges_all_frames(bb)
    assert edges.shape == (10, 68, 16)
    r = 30
    nbrs, cnt, md = TL.canonical_knn(edges, bb[:, :, 1], r)
    assert len(nbrs) == 16 and nbrs[0] == r and cnt[0] == 10 and md[0] == 0
    assert {r - 1, r + 1} <= set(nbrs.tolist())          # no sequence-gap filter
    assert (cnt <= 10).all() and (np.diff(cnt) <= 0).all()
    assert len(set(nbrs.tolist())) == 16


def test_canonical_knn_identical_frames_equals_descriptor_list(ncac):
    bb = np.repeat(_bb4(ncac)[:1], 10, axis=0)
    edges = TL.knn_edges_all_frames(bb)
    for r in (0, 20, 67):
        nbrs, cnt, _ = TL.canonical_knn(edges, bb[:, :, 1], r)
        assert (cnt == 10).all()
        assert set(nbrs.tolist()) == set(edges[0, r].tolist())
        # ordered by distance, as the descriptor list
        d = np.linalg.norm(bb[0, nbrs, 1] - bb[0, r, 1], axis=-1)
        assert (np.diff(d) >= -1e-6).all()


def test_segments_threemers_and_ends():
    L = 12
    bb0 = np.zeros((L, 4, 3), np.float32)
    for i in range(L):                                       # a straight chain, C-N bonded
        bb0[i, 0] = [3.8 * i, 0, 0]; bb0[i, 1] = [3.8 * i + 1.2, 0, 0]
        bb0[i, 2] = [3.8 * i + 2.5, 0, 0]
    bb0[7:] += [50.0, 0, 0]                                  # chain break before residue 7
    chain = np.array([0] * 10 + [1] * 2)                     # chain change before residue 10
    seg = TL.segments(bb0, chain)
    assert seg.tolist() == [0] * 7 + [1] * 3 + [2] * 2
    assert TL.dist_to_segment_end(seg).tolist() == [0, 1, 2, 3, 2, 1, 0, 0, 1, 0, 0, 0]
    assert TL.threemer_residues([6, 7, 3], seg) == [2, 3, 4, 5, 6, 7, 8]


def test_alignment_recovers_rigid_motion_and_keeps_local_motion(ncac):
    ca = ncac[:3, :, 1].astype(np.float64).copy()
    ca[1:] = ca[0]
    tm = [10, 11, 12]
    ca[1, tm] += [2.0, 0.0, 0.0]                              # local 3-mer motion in frame 2
    ca[2, tm] += [0.0, -1.5, 0.5]
    moved = ca.copy()
    for p, s in ((1, 1), (2, 2)):
        moved[p] = ca[p] @ _rot(s).T + np.array([5.0, -3.0, 7.0]) * p
    align = [i for i in range(ca.shape[1]) if i not in tm]
    trs = TL.align_to_first(moved, align)
    for p in range(3):
        back = TL.apply_transform(moved[p], trs[p])
        assert np.allclose(back[align], ca[0, align], atol=1e-4)
        assert np.allclose(back[tm], ca[p, tm], atol=1e-4)    # the 3-mer motion is kept
    with pytest.raises(ValueError):
        TL.align_to_first(moved, [1, 2])


def test_anchor_frame_local_coordinates(ncac):
    n, ca, c = (ncac[0, 20, k].astype(np.float64) for k in range(3))
    rot, org = TL.anchor_frame(n, ca, c)
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) and np.linalg.det(rot) > 0
    loc = lambda x: (x - org) @ rot  # noqa: E731
    assert np.allclose(loc(ca), 0, atol=1e-5)
    lc = loc(c); ln = loc(n)
    assert lc[0] < 0 and abs(lc[1]) < 1e-4 and abs(lc[2]) < 1e-4    # x axis along CA - C
    assert abs(ln[2]) < 1e-4                                          # N in the xy plane


def test_multistate_pdb_columns_chains_and_models(tmp_path):
    F, L = 2, 4
    names = ["N", "CA", "C", "O", ""] * L
    atom_res = np.repeat(np.arange(L), 5)
    coords = np.random.default_rng(0).normal(size=(F, 5 * L, 3)) * 10
    seg = np.array([0, 0, 1, 1])
    ids = TL.write_multistate_pdb(tmp_path / "x.pdb", coords, atom_res, names,
                                  ["ALA", "GLY", "SER", "LYS"], seg, keep_res=[0, 1, 3])
    lines = (tmp_path / "x.pdb").read_text().splitlines()
    atoms = [l for l in lines if l.startswith("ATOM")]
    assert len(atoms) == F * 3 * 4                      # residues 0, 1, 3; named atoms only
    assert sum(l.startswith("MODEL") for l in lines) == F
    assert sum(l == "TER" for l in lines) == F * 2      # one between pieces + one at the end
    ca = [l for l in atoms if l[12:16] == " CA "]
    assert len(ca) == F * 3
    assert {l[21] for l in atoms} == {"A", "B"} and ids[3] == ("B", 4) and ids[0] == ("A", 1)
    x = float(atoms[0][30:38]); assert x == pytest.approx(coords[0, 0, 0], abs=1e-3)
    assert atoms[0][17:20] == "ALA" and atoms[0][76:78].strip() == "N"


def test_seq_identity():
    a = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQ"
    assert TL.seq_identity(a, a) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    b = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=len(a)))
    assert TL.seq_identity(a, b) < 0.3
    assert TL.seq_identity(a, a[10:60]) == pytest.approx(1.0)      # a domain of the chain


def _layout():
    if not (CORPUS_DIR / "layout.json").exists():
        pytest.skip("training corpus layout not available")
    lj = json.loads((CORPUS_DIR / "layout.json").read_text())
    return lj["rows"], {k: v for k, v in lj.items() if k != "rows"}


@pytest.mark.parametrize("corpus", ["misato", "bioemu_cath"])
def test_allatom_loader_reproduces_descriptor_backbone(corpus):
    rows, settings = _layout()
    row = next(r for r in rows if r["corpus"] == corpus and 40 <= int(r["L"]) <= 400)
    if corpus == "misato" and not TL.MISATO_H5.exists():
        pytest.skip("MD.hdf5 not available")
    if corpus == "bioemu_cath" and not TL.allatom_available(corpus, row["system"], row["pid"], None):
        pytest.skip("BioEmu trajectories not available")
    s = TL.load_system(row, settings)
    frames, pos = TL.trajectory_frames(s, "descriptor", 10)
    assert pos == list(range(10))
    aa = TL.load_allatom(s, frames)
    assert aa.coords.shape[0] == 10 and len(aa.ca_atom) == s.L and len(aa.resname) == s.L
    for k, nm in enumerate(("N", "CA", "C", "O")):
        idx = [next(a for a in np.where(aa.atom_res == i)[0] if aa.atom_name[a] == nm)
               for i in range(s.L)]
        assert np.abs(aa.coords[:, idx] - s.bb[:, :, k]).max() < 1e-4, nm
    # five frames: positions 0, 2, 4, 7, 9 of the descriptor frames
    f5, p5 = TL.trajectory_frames(s, "descriptor", 5)
    assert p5 == [0, 2, 4, 7, 9] and len(f5) == 5
    fe, pe = TL.trajectory_frames(s, "even", 5)
    assert pe is None and len(fe) == 5
