"""Reconstruction loss of Ensembits: Hungarian matching + FAPE + dihedral term.

What it does
------------
The decoder always outputs P_max = 10 descriptors for a residue, while the
target is the set of P' <= 10 descriptors that were encoded (P' is random in
the sub-ensemble branch of SFTD). Frames have no order, so:

  1. `matched_pairs` pairs each target descriptor with one decoder output by a
     rectangular Hungarian assignment (scipy `linear_sum_assignment`); surplus
     decoder outputs stay unmatched and get no loss.
  2. `fape_dihedral_loss` scores the matched pairs: FAPE over pseudo atoms
     attached to the 16 relative frames of each descriptor (clamped at 10 A),
     plus `dihedral_weight` (= 3) x mean squared error of the 64 sin/cos
     dihedral values.
  3. `fape_hungarian_loss` = 1 followed by 2. This is the reconstruction loss
     the reported model was trained with (`--recon-loss-type fape --rect-hungarian`).

Paper
-----
  * `1_method.tex`, \\paragraph{Decoder and reconstruction loss.}:
      per-pair loss  l(f_hat, f) = FAPE(f_hat, f) + w_dih/(4K) ||d_hat - d||^2,  w_dih = 3
      Eq. (eq:hungarian): sigma* = argmin over injective assignments of sum l(...)
      Eq. (eq:fape):     L_recon = (1/P') sum_p l(f_hat^{sigma*(p)}, f^p)
  * `appendix.tex`, \\label{app:model-impl} \\paragraph{Loss.}: "FAPE clamp 10 A",
    "averaged over the two SFTD branches".

Origin
------
  * `fape_dihedral_loss`, `fape_hungarian_loss`, `_gram_schmidt`, `_N_OFFSET`,
    `_C_OFFSET`: moved out of `Ensembits/scripts/train.py` (logic unchanged).
  * `matched_pairs`, `rectangular_hungarian_loss`: copied from
    `src/tokenizer/rectangular_hungarian.py` (logic unchanged).

Used by the reported model: yes (`fape_hungarian_loss` in both SFTD branches).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Pseudo atoms attached to every frame (FAPE needs points, not just frames)
# ---------------------------------------------------------------------------
# Ideal local backbone offsets (Å) in a residue's own Affine3D frame — same
# convention as build_coords_K16_target.py (CA at origin).
#
# NOTE: these are *pseudo atoms*. In ESM3's frame (x axis along CA - C) the real
# atoms sit at C = (-1.51, 0, 0) and N = (0.56, 1.34, 0) (measured on 1ctfA00),
# so these offsets do not land on the real N and C. FAPE only needs a fixed,
# non-collinear point set per frame, so the loss is still a valid frame-aligned
# point error; the paper therefore says "three pseudo atoms per frame"
# (`1_method.tex`, Decoder and reconstruction loss). See require_fix.md #2.
_N_OFFSET = (1.458, 0.000, 0.000)
_C_OFFSET = (-0.545, 1.421, 0.000)


def _gram_schmidt(R: "torch.Tensor") -> "torch.Tensor":
    """Differentiable 6D→SO(3): orthonormalize the first two columns of R
    (..., 3, 3) row-major and complete with the cross product. Returns a proper
    rotation (columns e0,e1,e2).

    Used because the decoder's predicted 3x3 blocks are not exactly rotations;
    after this, R^{-1} = R^T as FAPE assumes. (Paper: "predicted rotations are
    orthonormalised by Gram--Schmidt" is implied by FAPE on relative frames.)
    """
    v0 = R[..., :, 0]; v1 = R[..., :, 1]
    e0 = v0 / v0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v1 = v1 - (e0 * v1).sum(-1, keepdim=True) * e0
    e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-1)


def fape_dihedral_loss(pred: "torch.Tensor", target: "torch.Tensor",
                        dihedral_weight: float = 3.0, clamp: float = 10.0,
                        n_nb: int = 16) -> "torch.Tensor":
    """FAPE on the reconstructed 16 neighbor frames + a scaled dihedral term.

    This is the per-pair loss l(f_hat, f) of the paper, averaged over all pairs
    passed in (`1_method.tex`, Decoder and reconstruction loss).

    `pred`, `target` shape (..., n_nb*16) where each neighbor block is
    [R(9, row-major) | t(3) | phi/psi sincos(4)] — i.e. the 256-D
    dihedral-augmented ESM3 descriptor. FAPE: derive N/CA/C atoms per neighbor
    (CA=t, N/C via R·ideal-offset), then for every neighbor reference frame
    express all atoms in that frame's local coords and measure clamped L2
    between pred and true; mean over (ref, atom). Predicted rotations are
    Gram-Schmidt orthonormalized so R⁻¹=Rᵀ. Dihedral term is sin/cos MSE,
    weighted by `dihedral_weight` to be comparable to the FAPE (Å) magnitude.

    Normalisation (matches the paper): FAPE is the mean over K reference frames
    x 3K points = 1/(3K^2) sum; the dihedral term is the mean over 4K values =
    1/(4K) ||d_hat - d||^2.
    """
    *lead, D = pred.shape
    per_nb = D // n_nb                       # 16 (frame+dihedral) or 12 (frame-only)
    has_dih = per_nb >= 16
    Pm = pred.reshape(-1, n_nb, per_nb)
    Tm = target.reshape(-1, n_nb, per_nb)
    M = Pm.shape[0]
    dev = pred.device
    n_off = torch.tensor(_N_OFFSET, device=dev, dtype=pred.dtype)
    c_off = torch.tensor(_C_OFFSET, device=dev, dtype=pred.dtype)

    def parse(x):
        # Split a descriptor into its 16 frames (R, t) and dihedral values d.
        R = _gram_schmidt(x[..., :9].reshape(M, n_nb, 3, 3))   # (M,nb,3,3)
        t = x[..., 9:12]                                       # (M,nb,3)
        d = x[..., 12:] if has_dih else None                  # (M,nb,4) or None
        return R, t, d

    Rp, tp, dp = parse(Pm)
    Rt, tt, dt = parse(Tm)

    def atoms(R, t):                                           # -> (M, nb*3, 3)
        # Three pseudo atoms per frame: t + R·N_offset, t, t + R·C_offset.
        n = t + torch.einsum('mnij,j->mni', R, n_off)
        c = t + torch.einsum('mnij,j->mni', R, c_off)
        return torch.stack([n, t, c], dim=2).reshape(M, n_nb * 3, 3)

    xp = atoms(Rp, tp); xt = atoms(Rt, tt)                     # (M, 48, 3)

    def local(R, t, x):                                        # refs (R,t), points x -> (M,nb,48,3)
        diff = x.unsqueeze(1) - t.unsqueeze(2)                 # (M,nb_ref,48,3)
        # local[i,p,a] = sum_b R_i[b,a] diff[i,p,b]  (= R_iᵀ · diff)
        return torch.einsum('miba,mipb->mipa', R, diff)

    # FAPE: every point expressed in every frame, predicted vs target, clamped.
    lp = local(Rp, tp, xp); lt = local(Rt, tt, xt)            # (M,nb,48,3)
    fape = (lp - lt).norm(dim=-1).clamp(max=clamp).mean()
    if has_dih:
        # + w_dih/(4K) ||d_hat - d||^2  (mean over the 4K dihedral values)
        return fape + dihedral_weight * ((dp - dt) ** 2).mean()
    return fape                                               # frame-only (192-D) baseline


def _local_points(x: "torch.Tensor", n_nb: int = 16):
    """Descriptors (..., n_nb*per_nb) -> (points in every frame, dihedrals).

    Returns
      loc: (..., n_nb, 3*n_nb, 3)  every pseudo atom expressed in every frame
      dih: (..., n_nb*4) or None   the sin/cos dihedral values
    Same construction as inside `fape_dihedral_loss` (Gram-Schmidt rotations,
    three pseudo atoms per frame), so `pairwise_recon_cost` equals that loss.
    """
    *lead, D = x.shape
    per_nb = D // n_nb
    xm = x.reshape(*lead, n_nb, per_nb)
    R = _gram_schmidt(xm[..., :9].reshape(*lead, n_nb, 3, 3))
    t = xm[..., 9:12]
    n_off = torch.tensor(_N_OFFSET, device=x.device, dtype=x.dtype)
    c_off = torch.tensor(_C_OFFSET, device=x.device, dtype=x.dtype)
    n = t + torch.einsum('...ij,j->...i', R, n_off)
    c = t + torch.einsum('...ij,j->...i', R, c_off)
    pts = torch.stack([n, t, c], dim=-2).reshape(*lead, n_nb * 3, 3)       # (..., 48, 3)
    diff = pts.unsqueeze(-3) - t.unsqueeze(-2)                              # (..., nb, 48, 3)
    loc = torch.einsum('...kba,...kpb->...kpa', R, diff)                    # R_k^T (x - t_k)
    dih = xm[..., 12:].reshape(*lead, n_nb * (per_nb - 12)) if per_nb >= 16 else None
    return loc, dih


@torch.no_grad()
def pairwise_recon_cost(pred: "torch.Tensor", target: "torch.Tensor",
                        dihedral_weight: float = 3.0, clamp: float = 10.0,
                        n_nb: int = 16, chunk: int = 512) -> "torch.Tensor":
    """Per-pair reconstruction loss for every (decoder output, target) pair.

    pred (B, N_pred, D), target (B, N_tgt, D) -> cost (B, N_pred, N_tgt), where
    cost[b, i, j] = FAPE(pred_i, target_j) + dihedral_weight * mean((d_i - d_j)^2),
    i.e. exactly the per-pair loss l of the paper (`1_method.tex`, Decoder and
    reconstruction loss). Computed without gradients, in chunks over B to bound
    memory (chunk 512 x 10 x 10 pairs x 16 frames x 48 points x 3 ~ 0.5 GB fp32).
    Used as the matching cost of Eq. (eq:hungarian) (code_fix §2).
    """
    B, Np, D = pred.shape
    Nt = target.shape[1]
    out = torch.empty(B, Np, Nt, device=pred.device, dtype=pred.dtype)
    for s in range(0, B, chunk):
        lp, dp = _local_points(pred[s:s + chunk], n_nb)                    # (b,Np,nb,48,3)
        lt, dt = _local_points(target[s:s + chunk], n_nb)                  # (b,Nt,nb,48,3)
        c = (lp.unsqueeze(2) - lt.unsqueeze(1)).norm(dim=-1).clamp(max=clamp).mean(dim=(-1, -2))
        if dp is not None:
            c = c + dihedral_weight * ((dp.unsqueeze(2) - dt.unsqueeze(1)) ** 2).mean(-1)
        out[s:s + chunk] = c
    return out


def fape_hungarian_loss(pred: "torch.Tensor", target: "torch.Tensor",
                         dihedral_weight: float = 3.0, clamp: float = 10.0,
                         n_nb: int = 16, match_cost: str = "loss") -> "torch.Tensor":
    """Permutation-invariant FAPE over the decoder's frame set.

    This is L_recon of the paper (Eq. eq:fape) with the matching of
    Eq. eq:hungarian.

    MD frames are an unordered set, so the positional `fape_dihedral_loss`
    (frame i of the decoder vs. frame i of a randomly-permuted target subset)
    imposes an arbitrary ordering penalty. Here we first solve a rectangular
    Hungarian assignment (`matched_pairs`; cost = the per-pair loss itself by
    default, match_cost="loss"; "l2" = squared L2 between descriptors, the cost of
    checkpoints trained before 2026-10-15) to pair each target
    frame with its best decoder frame — the decoder may emit more frames
    (P_max) than targets (p_eff); the surplus are left unmatched and incur no
    loss — then apply FAPE to the matched (B, n_match, D) pairs. Identical
    pred/target → 0; invariant to any permutation of either frame set.
    """
    # FIXED(code_fix §2, 2026-10-15): match_cost="loss" (default) matches on the
    #   per-pair loss itself, as the paper states (Eq. eq:hungarian): cost =
    #   FAPE + w_dih x dihedral MSE (`pairwise_recon_cost`). match_cost="l2"
    #   reproduces the matching of every checkpoint trained before the fix
    #   (squared L2 between raw descriptors). Measured on the old checkpoint
    #   (checks/check_matching.py): the two matchings agree on 2-34% of residues;
    #   the loss under L2 matching is 0.9-3.3% higher.
    mp, mt = matched_pairs(pred, target, cost=match_cost,
                           dihedral_weight=dihedral_weight, clamp=clamp, n_nb=n_nb)
    return fape_dihedral_loss(mp, mt, dihedral_weight, clamp, n_nb)


# ---------------------------------------------------------------------------
# Rectangular Hungarian matching (from src/tokenizer/rectangular_hungarian.py)
# ---------------------------------------------------------------------------
# The decoder outputs N_pred = P_max descriptors; the target has N_target = P'
# descriptors. scipy's linear_sum_assignment accepts rectangular cost matrices
# and returns a matching of size min(N_pred, N_target); the longer side's extra
# elements are left unmatched. Matching is computed on the CPU per residue; the
# cost matrices are built on the GPU and transferred once.

def rectangular_hungarian_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    unmatched_penalty: float = 0.0,
    return_permutation: bool = False,
):
    """Mean squared error over the optimal min-cost matching between
    ``pred`` and ``target``.

    Not used by the reported model (it uses FAPE, `fape_hungarian_loss`); this
    is the `--recon-loss-type default` option of the trainer, kept because the
    trainer still offers it.

    Args:
        pred:   ``(B, N_pred, D)``  — predicted set elements.
        target: ``(B, N_target, D)`` — target set elements.
                ``N_pred`` and ``N_target`` may differ.
        unmatched_penalty: extra MSE-weight applied per *unmatched*
                element on the longer side. ``0.0`` (default) means
                unmatched elements are free.
        return_permutation: if True, also returns the per-sample
                ``(row_idx, col_idx)`` matching arrays.

    Returns:
        loss: scalar 0-D tensor, mean over batch.
        (optional) perms: list of (row_idx, col_idx) numpy arrays.
    """
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError(
            f"pred and target must be (B, N, D); got pred {tuple(pred.shape)}, "
            f"target {tuple(target.shape)}")
    B, N_pred, D = pred.shape
    B_t, N_tgt, D_t = target.shape
    if B != B_t or D != D_t:
        raise ValueError(
            f"Batch / feature mismatch: pred {tuple(pred.shape)} vs "
            f"target {tuple(target.shape)}")

    # Cost matrices on GPU: (B, N_pred, N_target) squared L2 distances.
    costs = torch.cdist(pred, target).pow(2)
    costs_np = costs.detach().cpu().numpy()

    n_match = min(N_pred, N_tgt)
    matched_pred_idx = np.empty((B, n_match), dtype=np.int64)
    matched_tgt_idx = np.empty((B, n_match), dtype=np.int64)
    perms = []
    for b in range(B):
        row_idx, col_idx = linear_sum_assignment(costs_np[b])
        # scipy returns row, col arrays of length min(N_pred, N_target)
        matched_pred_idx[b] = row_idx
        matched_tgt_idx[b] = col_idx
        if return_permutation:
            perms.append((row_idx, col_idx))

    pred_dev = pred.device
    mpi = torch.from_numpy(matched_pred_idx).to(pred_dev)   # (B, n_match)
    mti = torch.from_numpy(matched_tgt_idx).to(pred_dev)    # (B, n_match)
    batch_idx = torch.arange(B, device=pred_dev).unsqueeze(1).expand_as(mpi)
    matched_pred = pred[batch_idx, mpi]      # (B, n_match, D)
    matched_tgt = target[batch_idx, mti]     # (B, n_match, D)
    loss = F.mse_loss(matched_pred, matched_tgt)

    if unmatched_penalty > 0.0:
        # Penalize the longer side's unmatched elements toward zero.
        if N_pred > n_match:
            full_pred_idx = torch.arange(N_pred, device=pred_dev) \
                .unsqueeze(0).expand(B, -1)                              # (B, N_pred)
            matched_mask = torch.zeros(B, N_pred, dtype=torch.bool,
                                         device=pred_dev)
            matched_mask.scatter_(1, mpi, True)
            unmatched_pred = pred[~matched_mask].view(-1, D)              # (B*(N_pred-n_match), D)
            loss = loss + unmatched_penalty * unmatched_pred.pow(2).mean()
        if N_tgt > n_match:
            matched_mask = torch.zeros(B, N_tgt, dtype=torch.bool,
                                         device=pred_dev)
            matched_mask.scatter_(1, mti, True)
            unmatched_tgt = target[~matched_mask].view(-1, D)
            loss = loss + unmatched_penalty * unmatched_tgt.pow(2).mean()

    if return_permutation:
        return loss, perms
    return loss


def matched_pairs(
    pred: torch.Tensor,
    target: torch.Tensor,
    cost: str = "l2",
    **cost_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the matched ``(matched_pred, matched_target)`` tensors of
    shape ``(B, min(N_pred, N_target), D)`` after rectangular Hungarian
    matching. No loss is computed.

    Paper: the assignment sigma* of Eq. eq:hungarian (`1_method.tex`).

    cost="loss": per-pair reconstruction loss (`pairwise_recon_cost`; kwargs
                 dihedral_weight, clamp, n_nb) — the paper's matching.
    cost="l2"  : squared L2 between raw descriptors — the matching of checkpoints
                 trained before code_fix §2 (default here so the function keeps its
                 original behaviour for other callers; `fape_hungarian_loss`
                 defaults to "loss").
    """
    B, N_pred, D = pred.shape
    _, N_tgt, _ = target.shape
    if cost == "loss":
        costs = pairwise_recon_cost(pred, target, **cost_kwargs)
    elif cost == "l2":
        costs = torch.cdist(pred, target).pow(2)
    else:
        raise ValueError(f"cost must be 'loss' or 'l2', got {cost!r}")
    costs_np = costs.detach().cpu().numpy()
    n_match = min(N_pred, N_tgt)
    mpi = np.empty((B, n_match), dtype=np.int64)
    mti = np.empty((B, n_match), dtype=np.int64)
    for b in range(B):
        r, c = linear_sum_assignment(costs_np[b])
        mpi[b] = r; mti[b] = c
    pred_dev = pred.device
    mpi_t = torch.from_numpy(mpi).to(pred_dev)
    mti_t = torch.from_numpy(mti).to(pred_dev)
    batch_idx = torch.arange(B, device=pred_dev).unsqueeze(1).expand_as(mpi_t)
    return pred[batch_idx, mpi_t], target[batch_idx, mti_t]
