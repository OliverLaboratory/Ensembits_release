"""Pick diverse level-1 tokens and their most representative residue (codebook row figure).

What it does
------------
  1. Tokenizes N mdCATH domains (P = 10 frames of one temperature / replica, read
     from the mdCATH HDF5 files) with an Ensembits checkpoint and keeps the
     continuous encoder output z_e of every residue.
  2. Among well-used tokens (>= --min-count residues), picks k far apart in
     codebook space (farthest-point sampling on the level-1 code vectors, seeded
     by the most frequent token).
  3. For each picked token, the representative is the residue r minimising
     ||z_e(r) - e_k|| (closest embedding to the code vector), restricted to an
     interior residue (3 < r < L - 3) with that token.
  4. Writes conformation 0 of each representative (heavy-atom PDB in the
     residue's own backbone frame) and its neighbour frames to
     `<out-dir>/tok<n>.pdb` and `<out-dir>/tokens.json`, the inputs of the
     research-repo renderer `camera_ready/fig_codebook/render_tokens.py` (PyMOL).

Paper
-----
  Codebook token figure prepared for the camera-ready (`camera_ready/fig_codebook/`,
  "Codebook row figure", worklog 2026-09-29); it is not yet included in the tex.
  It illustrates `appendix.tex` \\label{app:viz} (token visualisation, the role of
  fig:5x5tokens) with closest-to-codeword exemplars. The descriptor and codebook:
  `appendix.tex` \\label{app:descriptor-esm3}, `1_method.tex` \\paragraph{Residual
  quantizer.} (level-1 codebook, M_1 = 2048).

Origin
------
Copied from `camera_ready/fig_codebook/mine_tokens.py`. Changes, each marked
FIXED(code_fix P20, 2026-10-16):
  * model directory from `--model-dir` (default `ensembits.paths.CKPT_DIR`) instead
    of the hard-coded flagship directory; output directory from `--out-dir`;
    mdCATH HDF5 folder from `--h5-dir` (default REPO_ROOT/data/mdcath/data);
  * the descriptor is computed by `ensembits.descriptor._descriptor` with the
    checkpoint's dihedral convention and chain-break cutoff (the original used
    the research-repo `_descriptor` with the legacy +180° φ/ψ for any checkpoint);
  * backbone frames and kNN for the geometry dump come from `ensembits.geometry`
    (the same routines the descriptor uses, bit-identical to ESM3's, see
    tests/test_geometry.py) instead of ESM3's structure encoder, so ESM3 weights
    are not downloaded.
Logic otherwise unchanged.

Used by the reported model: yes (renders the reported checkpoint's tokens once
it is retrained).

Inputs / outputs
----------------
  reads   REPO_ROOT/data/mdcath/data/mdcath_dataset_<domain>.h5 (--h5-dir)
          <model-dir>/config.json, best.pt, stats.npz
  writes  <out-dir>/tok<n>.pdb, <out-dir>/tokens.json
          (default out-dir REPO_ROOT/camera_ready/fig_codebook/build)

How to run
----------
    python downstream/analyses/fig_codebook_mine_tokens.py --model-dir ckpt/<retrained>
    # smoke
    python downstream/analyses/fig_codebook_mine_tokens.py --n-domains 2 --k 2 --min-count 1 \\
        --out-dir <scratch>/fig_codebook --device cpu
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _common.py
from ensembits.paths import REPO_ROOT as ROOT, CKPT_DIR  # noqa: E402
from ensembits.descriptor import _descriptor, K as KNN_K  # noqa: E402
from _common import resolve_model_dir, ckpt_hash, checkpoint_settings  # noqa: E402

CAPS = {"CAY", "HY1", "HY2", "HY3", "CY", "OY", "NT", "HNT", "CAT", "HT1", "HT2", "HT3"}
OUT = ROOT / "camera_ready" / "fig_codebook" / "build"
H5_DIR = ROOT / "data" / "mdcath" / "data"


def load_domain(fn, temp, rep, P):
    import h5py
    dom = os.path.basename(fn)[len("mdcath_dataset_"):-3]
    with h5py.File(fn) as f:
        d = f[dom]
        atoms = [l for l in d["pdbProteinAtoms"][()].decode().splitlines() if l.startswith("ATOM")]
        g = d[temp][rep]
        n = g["coords"].shape[0]
        idx = np.linspace(0, n - 1, P).round().astype(int)
        xyz = g["coords"][sorted(set(idx.tolist()))]
        ss = g["dssp"][idx[0]]
    res_of, names, resn, keys = [], [], [], []
    for l in atoms:
        k = (l[21], l[22:27])
        if not keys or keys[-1] != k:
            keys.append(k); resn.append(l[17:20].strip())
        res_of.append(len(keys)); names.append(l[12:16].strip())
    L = len(keys)
    lookup = {(r, n_): a for a, (r, n_) in enumerate(zip(res_of, names))}
    bb = np.array([[lookup[(r, n_)] for n_ in ("N", "CA", "C")] for r in range(1, L + 1)])
    ss = [c.decode() if isinstance(c, bytes) else str(c) for c in ss]
    return dict(dom=dom, atoms=atoms, res_of=res_of, names=names, resn=resn, L=L,
                xyz=xyz, coords=xyz[:, bb, :].astype(np.float32), ss=ss)


def fps(vecs, seed, k):
    chosen = [seed]
    d = np.linalg.norm(vecs - vecs[seed], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d)); chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(vecs - vecs[nxt], axis=1))
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=str(CKPT_DIR))
    ap.add_argument("--h5-dir", default=str(H5_DIR))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--n-domains", type=int, default=200)
    ap.add_argument("--temp", default="379")
    ap.add_argument("--rep", default="0")
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ensembits.featurize import load_model
    # FIXED(code_fix P20, 2026-10-16): model dir from the CLI (was a hard-coded
    #   flagship directory); its config fixes the dihedral convention and cutoff.
    model_dir = resolve_model_dir(args.model_dir)
    ens = load_model(model_dir, device=args.device)
    conv, cutoff = checkpoint_settings(ens)
    print(f"model {model_dir.name} sha={ckpt_hash(model_dir)} convention={conv} cutoff={cutoff}")
    model = ens.model.eval()
    mu = torch.as_tensor(ens.mean, device=args.device).float()
    sd = torch.as_tensor(ens.std, device=args.device).float()
    E = model.vq_levels[0].codebook.weight.detach().cpu().numpy()          # (2048, 128)

    files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))[: args.n_domains]
    Z, TOK, WHO = [], [], []
    for fi, fn in enumerate(files):
        try:
            D = load_domain(fn, args.temp, args.rep, args.P)
        except Exception as e:                                             # noqa: BLE001
            print("skip", fn, e); continue
        with torch.no_grad():
            # FIXED(code_fix P20, 2026-10-16): descriptor in the checkpoint's φ/ψ
            #   convention and chain-break cutoff (mdCATH all-atom frames are real
            #   backbones, so the cutoff applies); was the legacy +180° code path.
            desc = _descriptor(torch.from_numpy(D["coords"]), args.device, convention=conv,
                               chain_break_cutoff=cutoff).to(args.device)
            x = ((desc - mu) / sd).transpose(0, 1).contiguous().float()     # (L, P, 256)
            z = model.encoder(x)
            tok = model._quantize(z)[1][0]
        Z.append(z.cpu().numpy()); TOK.append(tok.cpu().numpy())
        WHO += [(fn, r + 1) for r in range(D["L"])]
        if fi % 25 == 0:
            print(f"{fi}/{len(files)} {D['dom']} L={D['L']}", flush=True)
    Z = np.concatenate(Z); TOK = np.concatenate(TOK)
    print(f"residues={len(TOK)}  tokens used={len(np.unique(TOK))}")

    counts = np.bincount(TOK, minlength=len(E))
    cand = np.where(counts >= args.min_count)[0]
    seed = int(np.argmax(counts[cand]))
    picks = [int(cand[i]) for i in fps(E[cand], seed, args.k)]

    reps = []
    for k in picks:
        d = np.linalg.norm(Z - E[k], axis=1)
        order = np.argsort(d)
        for i in order:                               # need interior residue for phi/psi + frames
            fn, r = WHO[i]
            if TOK[i] == k:
                D = load_domain(fn, args.temp, args.rep, args.P)
                if 3 < r < D["L"] - 3:
                    break
        reps.append((k, i, float(d[i]), D, r))

    # FIXED(code_fix P20, 2026-10-16): frames and kNN from ensembits.geometry (the
    #   descriptor's own routines, bit-identical to ESM3's build_affine3d_from_coordinates
    #   and find_knn_edges) instead of loading the ESM3 structure encoder.
    from ensembits import geometry as G
    summary = []
    for n, (k, i, dist, D, r) in enumerate(reps):
        ct = torch.from_numpy(D["coords"][:1])                             # conformation 0
        rot, trans, amask = G.backbone_frames(ct)
        knn = G.knn_edges(ct, amask, KNN_K)
        Rm = rot[0].numpy(); T = trans[0].numpy()
        Rq, tq = Rm[r - 1], T[r - 1]
        # heavy-atom PDB of conformation 0 in the residue's own frame
        lines = []
        for a, l in enumerate(D["atoms"]):
            nm = D["names"][a]
            if nm in CAPS or nm.startswith("H") or (len(nm) > 1 and nm[0].isdigit() and nm[1] == "H"):
                continue
            x, y, zc = (D["xyz"][0, a] - tq) @ Rq
            lines.append(f"ATOM  {len(lines) + 1:5d} {nm:<4s} {D['resn'][D['res_of'][a] - 1]:>3s} A"
                         f"{D['res_of'][a]:4d}    {x:8.3f}{y:8.3f}{zc:8.3f}  1.00  0.00          {nm[0]:>2s}")
        (out_dir / f"tok{n}.pdb").write_text("\n".join(lines) + "\nEND\n")
        nb = [int(j) + 1 for j in knn[0, r - 1].numpy() if int(j) + 1 != r]
        frames = {j: dict(R=(Rq.T @ Rm[j - 1]).tolist(), t=(Rq.T @ (T[j - 1] - tq)).tolist()) for j in nb}
        n_same = int((TOK == k).sum())
        summary.append(dict(slot=n, token=k, domain=D["dom"], resi=r, resn=D["resn"][r - 1],
                            ss=D["ss"][r - 1], dist=dist, count=n_same, knn=nb, frames=frames))
        print(f"tok{n}: token {k:4d} (n={n_same:4d})  {D['dom']} {D['resn'][r - 1]}{r} "
              f"ss={D['ss'][r - 1]!r}  ||z-e_k||={dist:.3f}")
    (out_dir / "tokens.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
