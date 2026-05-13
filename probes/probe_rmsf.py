"""Per-residue RMSF probe.

Trains a small MLP to predict per-residue Cα RMSF from a frozen
per-residue representation and reports best validation Spearman across
`--epochs` epochs. The intended inputs are discrete tokens (Ensembits
or any other discrete tokenizer baseline — `3di_tokens`, `aminoaseed`,
`protoken`, `esm3struct`) plus their frozen codebook lookup.

Two input shapes are supported:

  Flat (mdCATH-style):
      --features  (N,) int token ids (an `(N, D)` per-residue feature
                    is also accepted for completeness — used by the
                    aggregate-of-tokens baselines `vote_3di` and
                    `protprofile_K` after their own pre-processing)
      --labels    (N,)   float ndarray of RMSF
      --splits    json with 'train'/'val' keys mapping to int-index lists

  Per-pid (misato-style):
      --features  pickle dict[pid -> (L,) int token ids] or
                    dict[pid -> (L, D)] for the aggregate baselines
      --labels    npz   dict[pid -> (L,) float ndarray]
      --splits    json  {'train': [pid, ...], 'val': [pid, ...], ...}
      --split-name optional sub-key (e.g. 'sequence' / 'structure' /
                    'random') if the splits json is nested by split flavour

For a token feature you must also pass `--codebook PATH.npy`
(a (M, d) array — typically the L_1 codebook of the trained model)
so the probe can build the frozen embedding lookup.

Outputs a small JSON with `best_val_spearman`, the inputs that produced
it, and a couple of sanity counters.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class TokenMLP(nn.Module):
    """Frozen-codebook embedding lookup → MLPProbe."""
    def __init__(self, codebook: np.ndarray, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        cb = torch.as_tensor(codebook, dtype=torch.float32)
        self.emb = nn.Embedding.from_pretrained(cb, freeze=True)
        self.mlp = MLPProbe(in_dim=cb.shape[1], hidden=hidden, dropout=dropout)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.emb(t))


# ----------------------------------------------------------------------
# I/O


def _load_features(path: Path) -> tuple[object, str]:
    """Returns (obj, kind) where kind is 'flat-cont' | 'flat-tok' |
    'per-pid-cont' | 'per-pid-tok'."""
    if path.suffix == ".npy":
        a = np.load(path)
        if a.ndim == 1:
            return a, "flat-tok"
        return a.astype(np.float32), "flat-cont"
    if path.suffix in (".npz",):
        z = np.load(path, allow_pickle=True)
        sample = z[z.files[0]]
        kind = "per-pid-tok" if sample.ndim == 1 else "per-pid-cont"
        return {k: z[k] for k in z.files}, kind
    if path.suffix in (".pkl", ".pickle"):
        d = pickle.load(open(path, "rb"))
        sample = next(iter(d.values()))
        sample = np.asarray(sample)
        kind = "per-pid-tok" if sample.ndim == 1 else "per-pid-cont"
        d = {k: np.asarray(v) for k, v in d.items()}
        return d, kind
    raise SystemExit(f"unsupported feature file extension: {path}")


def _load_labels_flat(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _load_labels_per_pid(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        return {k: z[k].astype(np.float32) for k in z.files}
    if path.suffix in (".pkl", ".pickle"):
        d = pickle.load(open(path, "rb"))
        return {k: np.asarray(v).astype(np.float32) for k, v in d.items()}
    raise SystemExit(f"unsupported label file extension: {path}")


def _stack_per_pid(feats: dict, labels: dict, pid_lists: dict) -> dict:
    """Concat (feature, label) pairs per split bucket."""
    out: dict = {}
    for split_name, pids in pid_lists.items():
        xs, ys = [], []
        for pid in pids:
            pid_u = pid.upper()
            if pid_u not in feats or pid_u not in labels:
                continue
            x = np.asarray(feats[pid_u])
            y = np.asarray(labels[pid_u], dtype=np.float32)
            n = min(len(x), len(y))
            if n < 5:
                continue
            xs.append(x[:n])
            ys.append(y[:n])
        if not xs:
            out[split_name] = (None, None)
        else:
            x_cat = np.concatenate(xs, axis=0)
            y_cat = np.concatenate(ys, axis=0)
            out[split_name] = (x_cat, y_cat)
    return out


# ----------------------------------------------------------------------
# Train


def train_and_eval(model: nn.Module, X_tr, y_tr, X_va, y_va, *,
                    device: str, epochs: int, batch: int,
                    lr: float, wd: float, dtype_x, seed: int) -> float:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x_tr = torch.tensor(X_tr, device=device, dtype=dtype_x)
    y_tr_t = torch.tensor(y_tr, device=device, dtype=torch.float32)
    x_va = torch.tensor(X_va, device=device, dtype=dtype_x)
    y_va_t = torch.tensor(y_va, device=device, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best = -1.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g)
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch].to(device)
            loss = F.mse_loss(model(x_tr[idx]), y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(x_va).cpu().numpy()
        sp = spearmanr(pv, y_va_t.cpu().numpy()).correlation
        if sp is not None and sp > best:
            best = sp
    return float(best)


# ----------------------------------------------------------------------
# Driver


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path,
                     help="(N, D) .npy / (N,) int .npy / .npz / .pkl per-pid")
    ap.add_argument("--labels", required=True, type=Path,
                     help=".npy (flat) or .npz / .pkl (per-pid) of RMSF labels")
    ap.add_argument("--splits", required=True, type=Path,
                     help="json with 'train' / 'val' (and optionally 'test')")
    ap.add_argument("--split-name", default=None,
                     help="sub-key of --splits for nested split files "
                          "(e.g. 'sequence' / 'structure' / 'random')")
    ap.add_argument("--codebook", default=None, type=Path,
                     help="(M, d) .npy frozen codebook (required for token features)")
    ap.add_argument("--out", required=True, type=Path,
                     help="output json path")
    ap.add_argument("--device", default=None,
                     help="auto-detect cuda/cpu by default")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats, kind = _load_features(args.features)

    if kind in ("flat-cont", "flat-tok"):
        labels = _load_labels_flat(args.labels)
        splits = json.loads(args.splits.read_text())
        if args.split_name:
            splits = splits[args.split_name]
        if "train" not in splits or "val" not in splits:
            raise SystemExit(f"--splits needs 'train' / 'val'; got {list(splits)}")
        train_idx = np.asarray(splits["train"], dtype=np.int64)
        val_idx = np.asarray(splits["val"], dtype=np.int64)
        # Drop rows with non-finite labels
        good = np.isfinite(labels)
        train_idx = train_idx[good[train_idx]]
        val_idx = val_idx[good[val_idx]]
        X_tr, X_va = feats[train_idx], feats[val_idx]
        y_tr, y_va = labels[train_idx], labels[val_idx]
        n_train, n_val = len(train_idx), len(val_idx)
    else:
        labels = _load_labels_per_pid(args.labels)
        splits = json.loads(args.splits.read_text())
        if args.split_name:
            splits = splits[args.split_name]
        # Upper-case pids consistently
        pid_lists = {sp: [p.upper() for p in splits.get(sp, [])]
                     for sp in ("train", "val")}
        bucket = _stack_per_pid(feats, labels, pid_lists)
        if bucket["train"][0] is None or bucket["val"][0] is None:
            raise SystemExit("empty split")
        X_tr, y_tr = bucket["train"]
        X_va, y_va = bucket["val"]
        n_train, n_val = len(X_tr), len(X_va)

    print(f"[setup] kind={kind}  train_residues={n_train}  val_residues={n_val}",
          flush=True)

    if kind in ("flat-tok", "per-pid-tok"):
        if args.codebook is None:
            raise SystemExit("--codebook is required for token features")
        cb = np.load(args.codebook).astype(np.float32)
        model = TokenMLP(cb)
        dtype_x = torch.long
        in_dim = int(cb.shape[1])
        M = int(cb.shape[0])
    else:
        in_dim = int(np.asarray(X_tr).shape[1])
        M = None
        model = MLPProbe(in_dim=in_dim)
        dtype_x = torch.float32

    sp = train_and_eval(model,
                         X_tr.astype(np.int64) if dtype_x == torch.long else X_tr.astype(np.float32),
                         y_tr.astype(np.float32),
                         X_va.astype(np.int64) if dtype_x == torch.long else X_va.astype(np.float32),
                         y_va.astype(np.float32),
                         device=device, epochs=args.epochs, batch=args.batch_size,
                         lr=args.lr, wd=args.weight_decay, dtype_x=dtype_x,
                         seed=args.seed)

    out = {
        "best_val_spearman": sp,
        "kind": kind,
        "in_dim": in_dim, "M": M,
        "n_train": int(n_train), "n_val": int(n_val),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}  best_val_spearman={sp:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
