"""mdCATH-div real-backbone extractor (data_prep/build_mdcath_real_bb.py, 2026-10-17).

What it does
------------
Synthetic h5 files in a pytest tmp dir, with the two features that broke the research
builder for 8 domains: insertion codes (residue numbers that repeat with iCode A / B)
and residue numbers that are not monotonic in the file. Checks: residues in file order by
(chain, resSeq, iCode), matching ca_10; stored frames found among the raw frames (stride
and non-stride); Cβ NaN for Gly; the sequence keeps the X convention for CHARMM His names
(and rmsf_common's X -> H normalisation gives H); refusal to overwrite; --verify-existing
identical / mismatch; a frame that is not in the trajectory fails.

Paper
-----
`appendix.tex` \\label{app:dataset} (mdCATH-div real backbones).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "data_prep"))
sys.path.insert(0, str(CODE_ROOT / "downstream"))
h5py = pytest.importorskip("h5py")

import build_mdcath_real_bb as MB  # noqa: E402
from rmsf_common import normalize_mdcath_real_bb_seq  # noqa: E402

# (resname, resSeq, iCode): insertion codes 2A / 2B, numbering 10 then 5 (non-monotonic)
RESIDUES = [("ALA", "1", ""), ("GLY", "2", ""), ("HSD", "2", "A"), ("SER", "2", "B"),
            ("HSE", "10", ""), ("LYS", "5", "")]


def _line(serial, name, resn, chain, resseq, icode):
    return (f"ATOM  {serial:5d} {name:<4s} {resn:<4s}{chain}{resseq:>4s}{icode:1s}   "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00")


def _fake_domain(root, dom="9xyzA00", picks=((0, 0), (0, 10), (1, 20), (1, 7)) + ((0, 3),) * 6):
    """h5 + mdCATH-div source npz; picks = (replica, frame) of the 10 stored frames."""
    lines, serial, atoms = ["REMARK cap", _line(1, "CAY", "ACE", "A", "0", "")], 2, []
    for resn, num, ic in RESIDUES:
        for name in ("N", "CA", "C", "O") + (() if resn == "GLY" else ("CB",)):
            lines.append(_line(serial, name, resn, "A", num, ic)); atoms.append((resn, name)); serial += 1
    lines.append("HETATM    99  O   HOH W   1       0.000   0.000   0.000")
    n_atoms = len(atoms) + 1                                   # + the cap atom
    rng = np.random.default_rng(0)
    h5d = root / "h5"; h5d.mkdir(parents=True, exist_ok=True)
    trajs = [rng.normal(size=(25, n_atoms, 3)).astype(np.float32) * 5 for _ in range(2)]
    with h5py.File(h5d / f"mdcath_dataset_{dom}.h5", "w") as f:
        g = f.create_group(dom)
        g["pdbProteinAtoms"] = "\n".join(lines)
        for r, t in enumerate(trajs):
            g.create_dataset(f"320/{r}/coords", data=t)
    ca_atoms = [1 + i for i, (_, n) in enumerate(atoms) if n == "CA"]      # ATOM index (cap = 0)
    ca10 = np.stack([trajs[r][t][ca_atoms] for r, t in picks])
    src = root / "src"; src.mkdir(exist_ok=True)
    np.savez(src / f"{dom}.npz", ca_5=ca10[:5], ca_10=ca10)
    (src / "_manifest.json").write_text(json.dumps({"representatives": {dom: {}}}))
    return dom, src, h5d, trajs, ca_atoms, atoms


def test_insertion_codes_and_file_order(tmp_path):
    dom, src, h5d, trajs, ca_atoms, atoms = _fake_domain(tmp_path)
    with h5py.File(h5d / f"mdcath_dataset_{dom}.h5", "r") as f:
        res = MB.parse_residues(f[dom]["pdbProteinAtoms"][()].decode())
    assert [k for k, _, _ in res] == [("A", n, ic) for _, n, ic in RESIDUES]   # file order kept
    out = tmp_path / "out"
    assert MB.main(["--pids", dom, "--src", str(src), "--h5-dir", str(h5d), "--out", str(out),
                    "--existing", str(tmp_path / "none"), "--workers", "1"]) == 0
    z = np.load(out / f"{dom}.npz")
    assert z["bb_10"].shape == (10, 6, 4, 3) and z["ca_idx"].tolist() == ca_atoms
    np.testing.assert_array_equal(z["bb_10"][:, :, 1], np.load(src / f"{dom}.npz")["ca_10"])
    assert np.isnan(z["cb_10"][:, 1]).all() and not np.isnan(z["cb_10"][:, 0]).any()   # Gly
    # sequence: CHARMM His names stay X (existing convention); normalisation gives H
    assert str(z["seq"]) == "AGXSXK" and "H" not in str(z["seq"])
    assert normalize_mdcath_real_bb_seq(str(z["seq"])) == "AGHSHK"
    assert z["match_dist"].max() == 0.0
    prov = json.loads(next(out.glob("_extract_*.provenance.json")).read_text())
    assert prov["info"][dom]["frames"][3] == ["1", 7]          # a non-stride frame found too
    # an existing output is never overwritten
    assert MB.main(["--pids", dom, "--src", str(src), "--h5-dir", str(h5d), "--out", str(out),
                    "--existing", str(tmp_path / "none"), "--workers", "1"]) == 1


def test_verify_existing_and_missing_frame(tmp_path):
    dom, src, h5d, *_ = _fake_domain(tmp_path)
    ex = tmp_path / "existing"
    assert MB.main(["--pids", dom, "--src", str(src), "--h5-dir", str(h5d), "--out", str(ex),
                    "--existing", str(ex), "--workers", "1"]) == 0
    assert MB.main(["--verify-existing", "--src", str(src), "--h5-dir", str(h5d), "--out",
                    str(tmp_path / "v1"), "--existing", str(ex), "--workers", "1"]) == 0
    z = dict(np.load(ex / f"{dom}.npz")); z["bb_10"] = z["bb_10"] + 1e-3
    np.savez_compressed(ex / f"{dom}.npz", **z)
    assert MB.main(["--verify-existing", "--src", str(src), "--h5-dir", str(h5d), "--out",
                    str(tmp_path / "v2"), "--existing", str(ex), "--workers", "1"]) == 1
    # --only-missing finds nothing to do when the file exists
    assert MB.main(["--only-missing", "--src", str(src), "--h5-dir", str(h5d),
                    "--existing", str(ex), "--workers", "1"]) == 0
    # a stored frame that is not in the trajectory fails the domain
    d = dict(np.load(src / f"{dom}.npz")); d["ca_10"] = d["ca_10"] + 0.5
    np.savez(src / f"{dom}.npz", **d)
    assert MB.main(["--pids", dom, "--src", str(src), "--h5-dir", str(h5d), "--out",
                    str(tmp_path / "v3"), "--existing", str(ex), "--workers", "1"]) == 1


def test_research_numbering_would_merge(tmp_path):
    # the old key (residue number only) merges 2 / 2A / 2B: 4 residues instead of 6
    dom, src, h5d, *_ = _fake_domain(tmp_path)
    with h5py.File(h5d / f"mdcath_dataset_{dom}.h5", "r") as f:
        pa = f[dom]["pdbProteinAtoms"][()].decode()
    nums = {int(l[22:26]) for l in pa.split("\n") if l.startswith("ATOM") and l[12:16].strip() == "CA"}
    assert len(nums) == 4 and len(MB.parse_residues(pa)) == 6


def test_no_ideal_fallback_in_consumers(tmp_path):
    # descriptor cache builder and ANOVA: no ideal geometry for a domain with a real file,
    # and for a domain without one only with --allow-ideal (2026-10-17)
    import os, subprocess
    rb = tmp_path / "data" / "cached_descriptors" / "mdcath_real_bb"; rb.mkdir(parents=True)
    src = tmp_path / "data" / "external" / "mdcath_div"; src.mkdir(parents=True)
    ca = np.random.default_rng(1).normal(size=(10, 20, 3)).astype(np.float32) * 3
    np.savez(src / "d0.npz", ca_10=ca, ca_5=ca[:5]); np.savez(src / "d1.npz", ca_10=ca, ca_5=ca[:5])
    np.savez(rb / "d0.npz", seq=np.array("A" * 20))            # real file without bb_10
    code = ("import sys; sys.path.insert(0, %r); import build_esm3desc_dihedral_perdataset as D\n"
            "assert D._coords_mdcath('d0') is None            # real file exists: no fallback\n"
            "assert D._coords_mdcath('d1') is None            # no file: refused by default\n"
            "D.ALLOW_IDEAL = True\n"
            "assert D._coords_mdcath('d1') is not None and 'd1' in D.IDEAL_PIDS\n"
            "assert D._coords_mdcath('d0') is None\n"
            "sys.path.insert(0, %r); import token_anova as A\n"
            "assert A._coords_mdcath('d0', 10) == (None, False) and A._coords_mdcath('d1', 10) == (None, False)\n"
            "assert A._coords_mdcath('d1', 10, True)[1] is True\n"
            "print('OK')") % (str(CODE_ROOT / "data_prep"), str(CODE_ROOT / "downstream" / "analyses"))
    env = dict(os.environ, ENSEMBITS_REPO_ROOT=str(tmp_path), ENSEMBITS_MDCATH_DIV_SRC=str(src),
               PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
