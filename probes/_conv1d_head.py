"""Shared conv1d trunk + heads + I/O helpers used by every supervised
downstream task in this package (binding-site, affinity, EC, GO).

Trunk: 3 × Conv1d(hidden=128, kernel=5) with GELU + Dropout(0.1).
Pool : masked mean + masked max over the residue axis (per-protein
       tasks); skipped (kept residue-major) for per-residue tasks.

`load_features_per_pid` reads either a `(pid -> (L,) int)` token cache
or a `(pid -> (L, D) float)` continuous-feature cache and produces the
embedded tensor that the trunk expects, plus the per-pid lengths used
to build padding masks during collation.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Conv1d trunks


class Conv1DTrunk(nn.Module):
    """3-layer Conv1d trunk (in_dim → hidden) used by every downstream head."""

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3,
                  kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [
                nn.Conv1d(prev, hidden, kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, L, in_dim)` → `(B, L, hidden)`."""
        x = x.transpose(1, 2)        # (B, in_dim, L)
        x = self.conv(x)              # (B, hidden, L)
        return x.transpose(1, 2)      # (B, L, hidden)


class PerResidueHead(nn.Module):
    """Trunk + per-residue logit (binding-site)."""

    def __init__(self, in_dim: int, n_classes: int = 1,
                  hidden: int = 128, **kw):
        super().__init__()
        self.trunk = Conv1DTrunk(in_dim, hidden=hidden, **kw)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)                 # (B, L, hidden)
        return self.head(h).squeeze(-1)   # (B, L)  for n_classes=1


def _masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean + masked max along the residue axis."""
    m = mask.unsqueeze(-1).float()
    avg = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
    neg = torch.full_like(x, float("-inf"))
    mx = torch.where(mask.unsqueeze(-1), x, neg).max(1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat([avg, mx], dim=-1)


class PerProteinHead(nn.Module):
    """Trunk + masked pool + small MLP head producing per-protein logits.

    Optional `ligand_dim`: when > 0, a learned projection of the
    per-protein ligand fingerprint is concatenated to the pooled
    representation before the head MLP. Used by the affinity task with
    MACCS-167 ligand fingerprints.
    """

    def __init__(self, in_dim: int, n_classes: int,
                  hidden: int = 128, depth: int = 3, kernel: int = 5,
                  dropout: float = 0.1,
                  ligand_dim: int = 0, lig_proj: int = 64):
        super().__init__()
        self.trunk = Conv1DTrunk(in_dim, hidden=hidden, depth=depth,
                                  kernel=kernel, dropout=dropout)
        head_in = 2 * hidden
        self.ligand_dim = ligand_dim
        if ligand_dim > 0:
            self.lig_proj = nn.Sequential(
                nn.Linear(ligand_dim, lig_proj),
                nn.GELU(),
                nn.Dropout(dropout))
            head_in += lig_proj
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                ligand: torch.Tensor | None = None) -> torch.Tensor:
        h = self.trunk(x)
        pooled = _masked_pool(h, mask)
        if self.ligand_dim > 0:
            if ligand is None:
                raise ValueError("ligand tensor required when ligand_dim > 0")
            pooled = torch.cat([pooled, self.lig_proj(ligand.float())], dim=-1)
        return self.head(pooled).squeeze(-1)


# ----------------------------------------------------------------------
# I/O helpers


def _load_npz_or_pickle(path: Path) -> dict:
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    if path.suffix in (".pkl", ".pickle"):
        return pickle.load(open(path, "rb"))
    raise SystemExit(f"unsupported feature file: {path}")


def load_features_per_pid(features_path: Path,
                            codebook_path: Path | None = None,
                            ) -> tuple[Mapping[str, np.ndarray], int]:
    """Read a per-pid feature cache. Returns `(feat_by_pid, in_dim)`.

    Three modes, dispatched by feature ndim and the presence of `codebook_path`:

    1. `(L,)` int tokens + `(M, D)` codebook → embedding lookup `(L, D)`.
       Used by single-token baselines (3di_tokens, aminoaseed, protoken,
       esm3struct).
    2. `(L, M)` probability histogram + `(M, D)` codebook → canonical
       protprofile interleave: each bin's probability concatenated with
       its 2-D centroid → `(L, M*(D+1))`. Mirrors
       `submission_exp/src/ec_classify_conv1d_misato.py:116-131` (the
       protprofile_K* branch).
    3. `(L, D)` continuous features, no codebook → returned as-is.
    """
    raw = _load_npz_or_pickle(features_path)
    sample = np.asarray(next(iter(raw.values())))

    if codebook_path is not None:
        cb = np.load(codebook_path).astype(np.float32)
        if sample.ndim == 1:
            # mode 1: token-id stream → centroid lookup
            feats = {pid: cb[np.clip(np.asarray(t).astype(np.int64),
                                       0, cb.shape[0] - 1)]
                     for pid, t in raw.items()}
            return feats, int(cb.shape[1])
        if sample.ndim == 2 and sample.shape[1] == cb.shape[0]:
            # mode 2: probability histogram → canonical interleave
            # Each residue (M,) becomes (M*(D+1),) with each bin's prob
            # immediately followed by that bin's centroid coordinates.
            D = int(cb.shape[1])
            M = int(cb.shape[0])
            feats: dict[str, np.ndarray] = {}
            for pid, hist in raw.items():
                h = np.asarray(hist, dtype=np.float32)
                L = h.shape[0]
                cb_bc = np.broadcast_to(cb, (L, M, D))                  # (L, M, D)
                comb = np.concatenate([h[..., None], cb_bc], axis=-1)   # (L, M, D+1)
                feats[pid] = comb.reshape(L, M * (D + 1))
            return feats, int(M * (D + 1))
        raise SystemExit(
            f"--codebook shape {cb.shape} incompatible with feature shape "
            f"{sample.shape}: expected (L,) ints for embedding lookup, or "
            f"(L, M) histogram with codebook of shape (M, D) for "
            f"prob-centroid interleave.")

    if sample.ndim != 2:
        raise SystemExit(
            f"--features without --codebook expects (L, D) per pid; "
            f"got {sample.shape}")
    return ({pid: np.asarray(v, dtype=np.float32) for pid, v in raw.items()},
            int(sample.shape[1]))


_LABEL_KINDS = ("scalar", "per_residue", "per_protein_vector")


def collate_residue_features(batch_pids: list[str],
                               feats: Mapping[str, np.ndarray],
                               labels: Mapping[str, np.ndarray] | None = None,
                               ligand: Mapping[str, np.ndarray] | None = None,
                               *, dtype_x: torch.dtype = torch.float32,
                               label_kind: str | None = None,
                               ) -> dict:
    """Pad a batch of variable-length proteins along the residue axis.

    `label_kind` selects how the per-pid label arrays are stacked:

      - "scalar"             : one float per pid → returns y of shape (B,)
      - "per_residue"        : (L,) per pid; padded to Lmax → returns (B, Lmax)
      - "per_protein_vector" : (n_classes,) constant per pid → returns
                                 (B, n_classes) by stacking
      - None (default)       : inferred — `scalar` when ndim==0, otherwise
                                `per_residue` if every label's length equals
                                its protein's residue count, else
                                `per_protein_vector`. Pass an explicit kind
                                for self-documenting call sites; the inferred
                                path is kept only for backwards compatibility.
    """
    if label_kind is not None and label_kind not in _LABEL_KINDS:
        raise ValueError(f"label_kind must be one of {_LABEL_KINDS}, "
                         f"got {label_kind!r}")
    fs = [feats[p] for p in batch_pids]
    Lmax = max(f.shape[0] for f in fs)
    D = fs[0].shape[1]
    X = torch.zeros((len(fs), Lmax, D), dtype=dtype_x)
    M = torch.zeros((len(fs), Lmax), dtype=torch.bool)
    for i, f in enumerate(fs):
        L = f.shape[0]
        X[i, :L] = torch.as_tensor(f, dtype=dtype_x)
        M[i, :L] = True

    out = {"x": X, "mask": M, "pids": batch_pids}
    if labels is not None:
        ys = [np.asarray(labels[p]) for p in batch_pids]
        kind = label_kind
        if kind is None:
            # Robust inference: compare label lengths to per-pid feature lengths
            # rather than to Lmax. n_classes ≠ Lmax used to misroute multi-hot
            # per-protein vectors as per-residue labels.
            if ys[0].ndim == 0:
                kind = "scalar"
            elif ys[0].ndim == 1 and all(y.shape[0] == fs[i].shape[0]
                                            for i, y in enumerate(ys)):
                kind = "per_residue"
            elif ys[0].ndim == 1:
                kind = "per_protein_vector"
            else:
                raise ValueError(
                    f"could not infer label_kind from first label "
                    f"shape={ys[0].shape}; pass label_kind= explicitly.")

        if kind == "scalar":
            out["y"] = torch.tensor([float(y) for y in ys],
                                     dtype=torch.float32)
        elif kind == "per_residue":
            Y = torch.zeros((len(ys), Lmax), dtype=torch.float32)
            for i, y in enumerate(ys):
                Y[i, :y.shape[0]] = torch.as_tensor(y, dtype=torch.float32)
            out["y"] = Y
        else:  # per_protein_vector
            out["y"] = torch.stack([torch.as_tensor(y, dtype=torch.float32)
                                      for y in ys])

    if ligand is not None:
        out["lig"] = torch.stack([torch.as_tensor(np.asarray(ligand[p]),
                                                    dtype=torch.float32)
                                    for p in batch_pids])
    return out
