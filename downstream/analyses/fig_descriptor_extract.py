"""Extract the geometry, descriptor and tokens of one mdCATH domain for the descriptor figure.

What it does
------------
Takes one mdCATH domain, samples P = 10 MD frames, and writes everything the
PyMOL renderer and the matplotlib composite need:

  frames_P10.pdb   heavy-atom multi-MODEL PDB (one MODEL per conformation)
  geom.json        per-frame query/neighbour frames, kNN, phi/psi, tokens
  desc.npy         the (P, L, 256) descriptor the checkpoint sees

The descriptor, kNN and frames are computed with the descriptor code of this
codebase (`ensembits.descriptor._descriptor` with the checkpoint's dihedral
convention and chain-break cutoff; frames / kNN from `ensembits.geometry`), so the
figure shows what the tokenizer actually sees. Tokens: all three RVQ levels at
P = 10 and at P = 1 (first frame).

Paper
-----
  Descriptor figure (`camera_ready/fig_descriptor/`, rendered by the research-repo
  scripts `render_pymol.py`, `render_p3_confs.py`, `sheet.py`, which only read
  the files written here). It illustrates `appendix.tex` \\subsection{ESM3-style
  relative-frame descriptor} (\\label{app:descriptor-esm3}): per-residue frame,
  K = 16 nearest residues incl. the residue itself, relative frames, φ/ψ.

Origin
------
Copied from `camera_ready/fig_descriptor/extract.py`. Changes, each marked
FIXED(code_fix P20, 2026-10-16):
  * model directory from `--model-dir` (default `ensembits.paths.CKPT_DIR`),
    output directory from `--out-dir`, HDF5 folder from `--h5-dir`;
  * the descriptor uses the checkpoint's convention and cutoff (was the legacy
    +180° code path for any checkpoint);
  * the φ/ψ written to geom.json are computed with the IUPAC formula directly, so
    the "+180° undo" of the original (correct only for the legacy formula) is
    removed (code_fix.md §1 "Also update after the fix");
  * frames / kNN from `ensembits.geometry` instead of the ESM3 structure encoder
    (bit-identical, tests/test_geometry.py; the original sanity check against the
    descriptor slice is kept).
Logic otherwise unchanged.

Used by the reported model: yes (re-render with the retrained checkpoint).

Inputs / outputs
----------------
  reads   REPO_ROOT/data/mdcath/data/mdcath_dataset_<domain>.h5 (--h5-dir)
          <model-dir>/config.json, best.pt, stats.npz
  writes  <out-dir>/frames_P10.pdb, geom.json, desc.npy
          (default out-dir REPO_ROOT/camera_ready/fig_descriptor/build)

How to run
----------
    python downstream/analyses/fig_descriptor_extract.py --domain 1ctfA00 --query 0 \\
        --model-dir ckpt/<retrained>      # --query 0 = auto-pick
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _common.py
from ensembits.paths import REPO_ROOT as ROOT, CKPT_DIR  # noqa: E402
from ensembits.descriptor import _descriptor, _phi_psi_sincos, K as KNN_K  # noqa: E402
from _common import resolve_model_dir, ckpt_hash, checkpoint_settings  # noqa: E402

CAPS = {"CAY", "HY1", "HY2", "HY3", "CY", "OY", "NT", "HNT", "CAT", "HT1", "HT2", "HT3"}
OUT = ROOT / "camera_ready" / "fig_descriptor" / "build"
H5_DIR = ROOT / "data" / "mdcath" / "data"


def read_domain(h5_dir, domain, temp, rep, P):
    import h5py
    fn = Path(h5_dir) / f"mdcath_dataset_{domain}.h5"
    with h5py.File(fn) as f:
        d = f[domain]
        pdb = d["pdbProteinAtoms"][()].decode().splitlines()
        g = d[temp][rep]
        xyz = g["coords"][:]                                   # (F, A, 3) Angstrom
        dssp = g["dssp"][:]
    atoms = [l for l in pdb if l.startswith("ATOM")]
    assert len(atoms) == xyz.shape[1]
    idx = np.linspace(0, xyz.shape[0] - 1, P).round().astype(int)
    return atoms, xyz[idx], dssp[idx], idx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=str(CKPT_DIR))
    ap.add_argument("--h5-dir", default=str(H5_DIR))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--domain", default="1ctfA00")
    ap.add_argument("--temp", default="379")
    ap.add_argument("--rep", default="0")
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--query", type=int, default=0, help="1-based residue index; 0=auto")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms, xyz, dssp, fidx = read_domain(args.h5_dir, args.domain, args.temp, args.rep, args.P)

    # residue bookkeeping (sequential 1..L, renumbered)
    res_keys, res_of_atom, names, resn = [], [], [], []
    for l in atoms:
        key = (l[21], l[22:27])
        if not res_keys or res_keys[-1] != key:
            res_keys.append(key); resn.append(l[17:20].strip())
        res_of_atom.append(len(res_keys))
        names.append(l[12:16].strip())
    L = len(res_keys)

    def atom_index(r, name):
        for a, (rr, nn) in enumerate(zip(res_of_atom, names)):
            if rr == r and nn == name:
                return a
        raise KeyError((r, name))

    bb = np.array([[atom_index(r, n) for n in ("N", "CA", "C")] for r in range(1, L + 1)])
    coords = xyz[:, bb, :].astype(np.float32)                   # (P, L, 3, 3)

    # heavy-atom multi-model PDB, caps stripped, chain A, resi 1..L
    lines = []
    for p in range(args.P):
        lines.append(f"MODEL     {p + 1:4d}")
        k = 0
        for a, l in enumerate(atoms):
            nm = names[a]
            if nm in CAPS or nm.startswith("H") or (len(nm) > 1 and nm[0].isdigit() and nm[1] == "H"):
                continue
            k += 1
            x, y, z = xyz[p, a]
            el = nm[0]
            lines.append(f"ATOM  {k:5d} {nm:<4s} {resn[res_of_atom[a]-1]:>3s} A{res_of_atom[a]:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}")
        lines.append("ENDMDL")
    (out_dir / "frames_P10.pdb").write_text("\n".join(lines) + "\nEND\n")

    # ---- model, descriptor + kNN + frames -----------------------------------
    from ensembits.featurize import load_model
    from ensembits import geometry as G
    dev = "cpu"
    # FIXED(code_fix P20, 2026-10-16): model from --model-dir; descriptor in its
    #   φ/ψ convention with its chain-break cutoff (was the legacy code path).
    model_dir = resolve_model_dir(args.model_dir)
    ens = load_model(model_dir, device=dev)
    conv, cutoff = checkpoint_settings(ens)
    print(f"model {model_dir.name} sha={ckpt_hash(model_dir)} convention={conv} cutoff={cutoff}")
    ct = torch.from_numpy(coords)
    desc = _descriptor(ct, dev, convention=conv, chain_break_cutoff=cutoff).numpy()  # (P, L, 256)
    np.save(out_dir / "desc.npy", desc)
    # FIXED(code_fix P20, 2026-10-16): frames / kNN from ensembits.geometry (same
    #   values as ESM3's build_affine3d_from_coordinates / find_knn_edges).
    rot, trans, amask = G.backbone_frames(ct)
    knn = G.knn_edges(ct, amask, KNN_K).numpy()                 # (P, L, K) 0-based
    R = rot.numpy()                                             # (P, L, 3, 3) columns = x,y,z axes
    T = trans.numpy()                                           # (P, L, 3) = CA

    # FIXED(code_fix P20 / §1, 2026-10-16): φ/ψ for the figure from the IUPAC formula
    #   (checked against mdtraj in tests/test_descriptor.py). The original undid the
    #   +180° of the legacy formula with atan2(-sin, -cos); that is wrong for an
    #   iupac checkpoint, so the figure angles no longer depend on the checkpoint.
    sc = _phi_psi_sincos(ct, convention="iupac", chain_break_cutoff=cutoff).numpy()
    phi = np.degrees(np.arctan2(sc[..., 0], sc[..., 1]))
    psi = np.degrees(np.arctan2(sc[..., 2], sc[..., 3]))

    # ---- tokens from the checkpoint ------------------------------------------
    with torch.no_grad():
        d = torch.from_numpy(desc)
        d = ((d - torch.as_tensor(ens.mean)) / torch.as_tensor(ens.std)).float()
        d = d.transpose(0, 1).contiguous()                      # (L, P, 256)
        tok = np.stack([t.numpy() for t in ens.model.encode_all(d)], -1)   # (L, 3)
        tok1 = np.stack([t.numpy() for t in ens.model.encode_all(d[:, :1])], -1)  # single frame

    # ---- query pick: interior residue whose 15 neighbours span >=3 SSEs, far in sequence
    ss = np.array([[c.decode() if isinstance(c, bytes) else c for c in row] for row in dssp])
    def score(r):
        nb = knn[0, r, 1:]
        far = np.abs(nb - r) > 3
        return far.sum() + 0.5 * len(set(ss[0, nb]))
    q = args.query - 1 if args.query > 0 else int(max(range(2, L - 2), key=score))
    print(f"query residue {q + 1} {resn[q]} ss={ss[0, q]}  neighbours={sorted((knn[0, q] + 1).tolist())}")
    print(f"token (P=10) L0/L1/L2 = {tok[q].tolist()}   single-frame = {tok1[q].tolist()}")

    # per-frame kNN overlap with frame 0
    base = set(knn[0, q].tolist())
    print("kNN set overlap vs frame0:", [len(base & set(knn[p, q].tolist())) for p in range(args.P)])

    # sanity: our reconstructed rel frame == descriptor slice
    Rq, tq = R[0, q], T[0, q]
    for jj, j in enumerate(knn[0, q][:3]):
        Rrel = Rq.T @ R[0, j]; trel = Rq.T @ (T[0, j] - tq)
        blk = desc[0, q, jj * 16: jj * 16 + 12]
        assert np.allclose(np.concatenate([Rrel.reshape(-1), trel]), blk, atol=1e-4) or \
               np.allclose(np.concatenate([Rrel.T.reshape(-1), trel]), blk, atol=1e-4), (blk, Rrel, trel)

    # FIXED(code_fix P20, 2026-10-16): geom.json also records the checkpoint (name,
    #   hash, convention, cutoff) so a render can be traced to the model it shows.
    geom = dict(domain=args.domain, temp=args.temp, rep=args.rep, frame_idx=fidx.tolist(),
                L=L, resn=resn, ss=ss.tolist(), query=q + 1,
                knn=(knn + 1).tolist(), R=R.tolist(), T=T.tolist(),
                phi=phi.tolist(), psi=psi.tolist(),
                tokens=tok.tolist(), tokens_single=tok1.tolist(),
                model=model_dir.name, ckpt_sha256_12=ckpt_hash(model_dir),
                dihedral_convention=conv, chain_break_cutoff=cutoff)
    (out_dir / "geom.json").write_text(json.dumps(geom))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
