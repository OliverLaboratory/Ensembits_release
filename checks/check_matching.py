"""Does Hungarian matching on squared L2 pick the same pairs as matching on the loss?

What it checks
--------------
The trainer pairs decoder outputs with target frames by squared L2 between raw
256-D descriptors (`ensembits/losses.py`, `matched_pairs`) and only then applies
the FAPE + dihedral loss. The paper (Eq. eq:hungarian) matches on the per-pair
loss itself. For 4,096 random training residues of the BioEmu-CATH + MISATO
corpus and P' = 1, 2, 5, 10 target frames, this script computes both
assignments with the reported checkpoint and reports how often they agree and
how much loss the L2 matching gives up.

Result on the reported checkpoint (2026-10-15): agreement 34% / 14% / 1.9% /
1.9%, loss gap 3.3% / 2.8% / 2.2% / 0.9% (P' = 1 / 2 / 5 / 10); the P' = 10 loss
under L2 matching (7.443) reproduces the run's best validation loss (7.437).
The dihedral term is ~9% of the loss; changing its weight (0 .. 3) barely
changes the gap. The 10 decoder outputs are similar to each other (FAPE spread
~0.9 A vs ~7 A mean), so any pairing scores about the same.

`pair_loss` is a tested, vectorised per-pair loss (FAPE + w_dih x dihedral MSE)
that can be reused to implement the fix (camera_ready/code_fix.md §2).

Paper: `1_method.tex`, \paragraph{Decoder and reconstruction loss.},
Eq. (eq:hungarian), (eq:fape).

Origin: `camera_ready/match_check.py` (written for code_fix.md §2); paths now
come from `ensembits.paths`, the model from `ensembits.featurize`. Logic unchanged.

Run:  python checks/check_matching.py cuda:0      (needs a GPU; reads the 73 GB cache lazily)
"""
import sys, numpy as np, torch
from pathlib import Path
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensembits.paths import REPO_ROOT, LEGACY_CKPT_DIR   # noqa: E402
from ensembits.featurize import load_model                # noqa: E402

# FIXED(code_fix P18, 2026-10-16): old: CKPT_DIR (the reported checkpoint, which becomes the
#   retrained one after E0). New: LEGACY_CKPT_DIR, the pre-fix checkpoint this check is about.
M = LEGACY_CKPT_DIR                                       # legacy (pre-fix) checkpoint
# Training corpus of the reported model (built by data_prep/build_bioemu_misato_descriptor.py).
DESC = REPO_ROOT / "output/reviewer_3_special_corpus_MISATO/cached_desc/esm3desc_dihedral_K16_P10.npy"
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
N_RES, CHUNK, NB, W_DIH, CLAMP = 4096, 256, 16, 3.0, 10.0
N_OFF = torch.tensor([1.458, 0.0, 0.0], device=dev)
C_OFF = torch.tensor([-0.545, 1.421, 0.0], device=dev)


def gram_schmidt(R):                          # same as train.py::_gram_schmidt
    v0, v1 = R[..., :, 0], R[..., :, 1]
    e0 = v0 / v0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v1 = v1 - (e0 * v1).sum(-1, keepdim=True) * e0
    e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.stack([e0, e1, torch.cross(e0, e1, dim=-1)], dim=-1)


def local_points(d):                          # (..., 256) -> (..., 16 ref, 48 pts, 3), dihedrals (..., 64)
    x = d.reshape(*d.shape[:-1], NB, 16)
    R = gram_schmidt(x[..., :9].reshape(*x.shape[:-1], 3, 3))
    t = x[..., 9:12]
    n = t + torch.einsum('...ij,j->...i', R, N_OFF)
    c = t + torch.einsum('...ij,j->...i', R, C_OFF)
    pts = torch.stack([n, t, c], dim=-2).reshape(*x.shape[:-2], NB * 3, 3)      # (...,48,3)
    diff = pts.unsqueeze(-3) - t.unsqueeze(-2)                                  # (...,16,48,3)
    loc = torch.einsum('...kba,...kpb->...kpa', R, diff)                        # R_k^T (x - t_k)
    return loc, x[..., 12:].reshape(*x.shape[:-2], NB * 4)


def pair_loss(pred, tgt):                     # pred (B,Pm,256), tgt (B,Pt,256) -> (B,Pm,Pt)
    lp, dp = local_points(pred); lt, dt = local_points(tgt)
    fape = (lp.unsqueeze(2) - lt.unsqueeze(1)).norm(dim=-1).clamp(max=CLAMP).mean(dim=(-1, -2))
    dih = ((dp.unsqueeze(2) - dt.unsqueeze(1)) ** 2).mean(-1)
    return fape + W_DIH * dih


def assign(cost):
    out = []
    for c in cost:
        r, k = linear_sum_assignment(c)
        out.append(dict(zip(k.tolist(), r.tolist())))          # target -> decoder output
    return out


ens = load_model(M, device=dev); m = ens.model.eval()
arr = np.load(DESC, mmap_mode="r")
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(arr.shape[0], N_RES, replace=False))
data = torch.from_numpy(np.ascontiguousarray(arr[idx])).float().to(dev)           # (N,10,256)

print(f"{N_RES} random training residues, checkpoint = BioEmu-CATH+MISATO")
print(f"{'P_sub':>5} {'same pairing':>13} {'loss (L2 match)':>16} {'loss (loss match)':>18} {'gap':>8} {'gap>5%':>7}")
for p_sub in (1, 2, 5, 10):
    same, l_l2, l_opt = [], [], []
    g = torch.Generator(device="cpu").manual_seed(p_sub)
    for s in range(0, N_RES, CHUNK):
        x = data[s:s + CHUNK]
        fidx = torch.randperm(10, generator=g)[:p_sub].to(dev)
        tgt = x[:, fidx]
        with torch.no_grad():
            z = m.encoder(tgt)
            q = m._quantize(z)[0]
            rec = m.decoder(q)                                       # (B,10,256)
            c_l2 = torch.cdist(rec, tgt).pow(2).cpu().numpy()       # trainer's matching cost
            c_ls = pair_loss(rec, tgt).cpu().numpy()                # actual loss per pair
        a_l2, a_ls = assign(c_l2), assign(c_ls)
        for b in range(len(x)):
            same.append(a_l2[b] == a_ls[b])
            l_l2.append(np.mean([c_ls[b, a_l2[b][j], j] for j in range(p_sub)]))
            l_opt.append(np.mean([c_ls[b, a_ls[b][j], j] for j in range(p_sub)]))
    l_l2, l_opt = np.array(l_l2), np.array(l_opt)
    gap = (l_l2 - l_opt) / np.maximum(l_opt, 1e-9)
    print(f"{p_sub:>5} {np.mean(same):>12.1%} {l_l2.mean():>16.4f} {l_opt.mean():>18.4f} "
          f"{(l_l2.mean() - l_opt.mean()) / l_opt.mean():>7.2%} {np.mean(gap > 0.05):>7.1%}")
