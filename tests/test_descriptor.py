"""Tests of the 256-D descriptor (`ensembits/descriptor.py`).

What it does
------------
Checks the properties the paper states for the descriptor:
  * phi/psi agree with an independent implementation (mdtraj) under the IUPAC
    convention, and the legacy convention is exactly +180 deg (code_fix §1);
  * the descriptor is invariant to rigid motions of a frame (SE(3));
  * the first neighbour slot is the residue itself (the "self slot");
  * the legacy and IUPAC descriptors differ only by the sign of the 64 dihedral
    columns.

Paper
-----
`appendix.tex` \\label{app:descriptor-esm3}; `1_method.tex` descriptor paragraph.

How to run
----------
  python -m pytest tests/test_descriptor.py
"""
import numpy as np
import pytest
import torch

from conftest import random_rotation
from ensembits.descriptor import _descriptor, _phi_psi_sincos, with_convention
from ensembits import geometry as G

K = 16
PER_NB = 16
DIH_COLS = np.concatenate([np.arange(k * PER_NB + 12, k * PER_NB + 16) for k in range(K)])


def _write_backbone_pdb(path, ncac):
    """One-model PDB of N/CA/C (glycine names; only the backbone is needed)."""
    lines, serial = [], 1
    for i, res in enumerate(ncac):
        for name, xyz in zip(("N", "CA", "C"), res):
            lines.append("ATOM  %5d  %-3s GLY A%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s"
                         % (serial, name, i + 1, *xyz, name[0]))
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_phi_psi_match_mdtraj(ensemble, tmp_path):
    md = pytest.importorskip("mdtraj")
    frame = ensemble[0].numpy()                                     # (L, 3, 3)
    pdb = tmp_path / "bb.pdb"
    _write_backbone_pdb(pdb, frame)
    traj = md.load(str(pdb))
    _, phi = md.compute_phi(traj)                                   # (1, L-1), residues 1..L-1
    _, psi = md.compute_psi(traj)                                   # (1, L-1), residues 0..L-2
    feats = _phi_psi_sincos(ensemble[:1], convention="iupac")[0].numpy()   # (L, 4)
    np.testing.assert_allclose(feats[1:, 0], np.sin(phi[0]), atol=2e-3)
    np.testing.assert_allclose(feats[1:, 1], np.cos(phi[0]), atol=2e-3)
    np.testing.assert_allclose(feats[:-1, 2], np.sin(psi[0]), atol=2e-3)
    np.testing.assert_allclose(feats[:-1, 3], np.cos(psi[0]), atol=2e-3)


def test_legacy_is_plus_180(ensemble):
    """legacy_plus180 = iupac angle + 180 deg, i.e. (sin, cos) negated."""
    iu = _phi_psi_sincos(ensemble, convention="iupac")
    le = _phi_psi_sincos(ensemble, convention="legacy_plus180")
    torch.testing.assert_close(le, -iu, atol=1e-5, rtol=0)


def test_undefined_endpoints_are_zero(ensemble):
    feats = _phi_psi_sincos(ensemble)
    assert torch.all(feats[:, 0, :2] == 0)       # phi of the first residue
    assert torch.all(feats[:, -1, 2:] == 0)      # psi of the last residue


def test_shape(ensemble):
    d = _descriptor(ensemble, "cpu")
    assert d.shape == (10, 68, 256)


def test_se3_invariance(ensemble):
    R = random_rotation(1)
    moved = ensemble @ R.T + torch.tensor([12.0, -7.0, 3.5])
    d0 = _descriptor(ensemble, "cpu")
    d1 = _descriptor(moved, "cpu")
    torch.testing.assert_close(d1, d0, atol=1e-3, rtol=0)


def test_self_slot_first(ensemble):
    """n_{r,1} = r: the nearest 'neighbour' of every residue is itself."""
    rot, trans, mask = G.backbone_frames(ensemble)
    edges = G.knn_edges(ensemble, mask, K)
    L = ensemble.shape[1]
    assert torch.equal(edges[..., 0], torch.arange(L).expand(ensemble.shape[0], L))
    # and its relative frame is the identity: R = I, t = 0
    d = _descriptor(ensemble, "cpu")
    eye = torch.eye(3).reshape(-1)
    torch.testing.assert_close(d[..., :9], eye.expand_as(d[..., :9]), atol=1e-4, rtol=0)
    torch.testing.assert_close(d[..., 9:12], torch.zeros_like(d[..., 9:12]), atol=1e-4, rtol=0)


def test_conventions_differ_only_in_dihedral_sign(ensemble):
    iu = _descriptor(ensemble, "cpu", convention="iupac").numpy()
    le = _descriptor(ensemble, "cpu", convention="legacy_plus180").numpy()
    frame_cols = np.setdiff1d(np.arange(256), DIH_COLS)
    np.testing.assert_array_equal(iu[..., frame_cols], le[..., frame_cols])
    np.testing.assert_allclose(le[..., DIH_COLS], -iu[..., DIH_COLS], atol=1e-5)


def test_unknown_convention_rejected(ensemble):
    with pytest.raises(ValueError):
        _descriptor(ensemble, "cpu", convention="degrees")


def test_cache_names():
    assert with_convention("esm3desc_dihedral_K16_P10.npy", "legacy_plus180") == "esm3desc_dihedral_K16_P10.npy"
    assert with_convention("esm3desc_dihedral_K16_P10.npy", "iupac") == "esm3desc_dihedral_iupac_K16_P10.npy"
    assert with_convention("x.npz", "iupac") == "x_iupac.npz"



# ---- chain breaks (code_fix §6) -----------------------------------------------

def _with_gap(ensemble, at=30, shift=10.0):
    """Move residues at..L-1 away by `shift` Å: no bond between at-1 and at."""
    gapped = ensemble.clone()
    gapped[:, at:] += torch.tensor([shift, 0.0, 0.0])
    return gapped


def test_no_effect_without_breaks(ensemble):
    a = _phi_psi_sincos(ensemble, "iupac", chain_break_cutoff=2.5)
    b = _phi_psi_sincos(ensemble, "iupac", chain_break_cutoff=None)
    assert torch.equal(a, b)


def test_break_zeroes_phi_psi_at_gap(ensemble):
    at = 30
    f = _phi_psi_sincos(_with_gap(ensemble, at), "iupac")          # auto -> 2.5 Å
    assert torch.all(f[:, at, :2] == 0)                            # phi after the gap
    assert torch.all(f[:, at - 1, 2:] == 0)                        # psi before the gap
    ref = _phi_psi_sincos(ensemble, "iupac")
    keep = [i for i in range(ensemble.shape[1]) if i not in (at - 1, at)]
    torch.testing.assert_close(f[:, keep], ref[:, keep], atol=1e-4, rtol=0)  # rest unchanged


def test_legacy_ignores_breaks(ensemble):
    """legacy_plus180 (auto) has no detection: reproduces old checkpoints' features."""
    f = _phi_psi_sincos(_with_gap(ensemble), "legacy_plus180")
    assert torch.any(f[:, 30, :2] != 0)


def test_ideal_geometry_must_disable_detection(ensemble):
    """Ideal N/C rebuilt from Cα have no real peptide bond: detection would zero
    almost every phi/psi, so callers pass chain_break_cutoff=None (all connected)."""
    from ensembits.ideal_backbone import _reconstruct_n_c_ideal
    ca = ensemble[:, :, 1].numpy()
    nc = [_reconstruct_n_c_ideal(ca[p]) for p in range(ca.shape[0])]
    ideal = torch.from_numpy(np.stack([np.stack([n, ca[p], c], 1) for p, (n, c) in enumerate(nc)])).float()
    on = _phi_psi_sincos(ideal, "iupac")                            # auto: 2.5 Å
    off = _phi_psi_sincos(ideal, "iupac", chain_break_cutoff=None)
    assert (on[:, 1:, :2] == 0).all(-1).float().mean() > 0.9        # most phi zeroed
    assert (off[:, 1:, :2] == 0).all(-1).float().mean() == 0        # none zeroed
