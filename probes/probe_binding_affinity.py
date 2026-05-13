"""Per-protein binding-affinity regression (-log K_d/K_i).

Conv1d trunk (3 × Conv1d(hidden=128, kernel=5) + GELU + Dropout) →
masked mean+max pool → small MLP head producing a single scalar per
protein. Trained with MSE. Reports R², Spearman, MSE on the test split.

**Ligand conditioning** is mandatory (``--ligand-maccs``): a 167-bit
MACCS fingerprint of the bound ligand is fed alongside the pooled
protein representation. Predicting affinity from a protein alone is not
a well-defined task — a single protein has many binding partners with
different affinities, so the label is one-to-many.

**AA concatenation** (``--concat-aa <aa_features.npz>``): concatenate
per-residue AA one-hot (21-D) to the protein feature before the
conv1d. Useful for evaluating whether a structural tokenizer adds
signal *on top of* sequence.

Usage:

    python -m probes.probe_binding_affinity \\
        --features data/cached_descriptors/<feature>.npz \\
        [--codebook data/cached_descriptors/<feature>.codebook.npy] \\
        --labels   data/labels/misato_affinity.json \\
        --splits   data/splits/misato_splits.json \\
        --split-name structure \\
        --ligand-maccs data/cached_descriptors/misato_ligand_maccs.npz \\
        [--concat-aa data/cached_descriptors/aa_tokens_misato.npz] \\
        --out runs/aff__<feature>__structure__seed0.json
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "8")


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from probes._conv1d_head import (  # noqa: E402
    PerProteinHead, collate_residue_features,
    load_features_per_pid,
)


def _load_split(splits_path: Path, split_name: str) -> dict:
    s = json.loads(splits_path.read_text())
    if split_name in s:
        s = s[split_name]
    return {k: [p.upper() for p in s.get(k, [])] for k in ("train", "val", "test")}


def _build_buckets(feats: dict, labels: dict, ligand: dict, split: dict,
                    min_len: int = 10) -> dict[str, list[str]]:
    """Match canonical iteration order: walk `feats` in NPZ-key order and
    bucket each pid into its split. Mirrors
    submission_exp/src/conv1d_binding_baselines.py:396-413 so that, for a
    given seed, the SGD trajectory is identical to the paper's run."""
    split_sets = {s: set(split[s]) for s in ("train", "val", "test")}
    bucket: dict[str, list[str]] = {s: [] for s in ("train", "val", "test")}
    for pid in feats:
        if pid not in labels:
            continue
        if not np.isfinite(float(labels[pid])):
            continue
        if pid not in ligand:
            continue
        if feats[pid].shape[0] < min_len:
            continue
        for s in ("train", "val", "test"):
            if pid in split_sets[s]:
                bucket[s].append(pid)
                break
    return bucket


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path,
                    help="Per-pid protein features (.npz).")
    ap.add_argument("--codebook", default=None, type=Path,
                    help="(L,) token ids in --features get codebook-looked-up by this (M, D) array.")
    ap.add_argument("--concat-aa", default=None, type=Path,
                    help="If set, .npz of per-pid AA one-hot (21-D); concatenated "
                         "to the protein feature before the conv1d.")
    ap.add_argument("--labels", required=True, type=Path,
                    help=".json {pid: float} of -log K_d/K_i targets.")
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--split-name", default="structure",
                    choices=("sequence", "structure", "random"))
    ap.add_argument("--ligand-maccs", required=True, type=Path,
                    help=".npz {pid: (167,) uint8} ligand MACCS fingerprints. "
                         "Required: predicting affinity from protein alone "
                         "is not a well-defined task (one protein has many "
                         "binding partners with different affinities).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ── Labels ──────────────────────────────────────────────────────
    raw_labels = json.loads(args.labels.read_text())
    labels = {p.upper(): float(v) for p, v in raw_labels.items()
              if v is not None and np.isfinite(float(v))}

    # ── Ligand MACCS (mandatory) ─────────────────────────────────────
    z = np.load(args.ligand_maccs, allow_pickle=False)
    ligand = {p.upper(): np.asarray(z[p], dtype=np.float32) for p in z.files}
    ligand_dim = int(next(iter(ligand.values())).shape[0])

    # ── Features ────────────────────────────────────────────────────
    feats, in_dim = load_features_per_pid(args.features, args.codebook)
    feats = {p.upper(): np.asarray(v, dtype=np.float32) for p, v in feats.items()}

    if args.concat_aa is not None:
        aa_feats, aa_dim = load_features_per_pid(args.concat_aa, None)
        aa_feats = {p.upper(): np.asarray(v, dtype=np.float32) for p, v in aa_feats.items()}
        new_feats: dict[str, np.ndarray] = {}
        for pid, f in feats.items():
            if pid not in aa_feats:
                continue
            a = aa_feats[pid]
            L = min(len(f), len(a))
            new_feats[pid] = np.concatenate([f[:L], a[:L]], axis=-1).astype(np.float32)
        feats = new_feats
        in_dim = in_dim + aa_dim
        print(f"[setup] +concat-aa: in_dim={in_dim} (+{aa_dim}), "
              f"n_proteins={len(feats)}", flush=True)
    else:
        print(f"[setup] in_dim={in_dim} ligand_dim={ligand_dim} "
              f"n_proteins={len(feats)}", flush=True)

    # ── Splits ──────────────────────────────────────────────────────
    split = _load_split(args.splits, args.split_name)
    bucket = _build_buckets(feats, labels, ligand, split)
    n = {s: len(bucket[s]) for s in bucket}
    if not bucket["train"] or not bucket["test"]:
        raise SystemExit(f"empty split: {n}")
    print(f"[setup] split={args.split_name}  train={n['train']} val={n['val']} test={n['test']}",
          flush=True)

    # ── Model ───────────────────────────────────────────────────────
    model = PerProteinHead(in_dim=in_dim, n_classes=1,
                            ligand_dim=ligand_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    best_va = float("inf"); best_state = None; no_imp = 0

    def _epoch(items: list[str], train: bool) -> float:
        model.train(train)
        # Mirror canonical's train_affinity_cont (conv1d_binding_baselines.py:175-206)
        # exactly: train pass does NOT read loss (no per-step host sync),
        # val pass uses sample-weighted mean `sum(per-batch-mean * B) / total`.
        if train:
            perm = torch.randperm(len(items), generator=g).tolist()
            items = [items[i] for i in perm]
            for s in range(0, len(items), args.batch_size):
                batch = items[s:s + args.batch_size]
                b = collate_residue_features(
                    batch, feats,
                    labels={p: labels[p] for p in batch},
                    ligand=ligand, label_kind="scalar")
                X = b["x"].to(device); M = b["mask"].to(device)
                y = b["y"].to(device); lig = b["lig"].to(device)
                yhat = model(X, M, ligand=lig)
                loss = F.mse_loss(yhat, y)
                opt.zero_grad(); loss.backward(); opt.step()
            return 0.0
        total = 0.0; n_samples = 0
        with torch.no_grad():
            for s in range(0, len(items), args.batch_size):
                batch = items[s:s + args.batch_size]
                b = collate_residue_features(
                    batch, feats,
                    labels={p: labels[p] for p in batch},
                    ligand=ligand, label_kind="scalar")
                X = b["x"].to(device); M = b["mask"].to(device)
                y = b["y"].to(device); lig = b["lig"].to(device)
                yhat = model(X, M, ligand=lig)
                total += float(F.mse_loss(yhat, y).item()) * X.shape[0]
                n_samples += X.shape[0]
        return total / max(n_samples, 1)

    for ep in range(args.epochs):
        _ = _epoch(bucket["train"], train=True)
        va_loss = _epoch(bucket["val"], train=False)
        if va_loss < best_va - 1e-4:
            best_va = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_y, all_p = [], []
    for s in range(0, len(bucket["test"]), args.batch_size):
        batch = bucket["test"][s:s + args.batch_size]
        b = collate_residue_features(
            batch, feats,
            labels={p: labels[p] for p in batch},
            ligand=ligand, label_kind="scalar")
        X = b["x"].to(device); M = b["mask"].to(device)
        lig = b["lig"].to(device)
        with torch.no_grad():
            pred = model(X, M, ligand=lig).cpu().numpy()
        ytrue = b["y"].numpy()
        all_p.extend(pred.tolist()); all_y.extend(ytrue.tolist())

    p_arr = np.asarray(all_p, dtype=np.float64)
    y_arr = np.asarray(all_y, dtype=np.float64)
    out = {
        "r2":         float(r2_score(y_arr, p_arr)),
        "spearman":   float(spearmanr(p_arr, y_arr).correlation),
        "mse":        float(np.mean((p_arr - y_arr) ** 2)),
        "in_dim":     int(in_dim),
        "ligand_dim": int(ligand_dim),
        "concat_aa":  args.concat_aa is not None,
        "n_train":    n["train"], "n_val": n["val"], "n_test": n["test"],
        "split":      args.split_name, "epochs": ep + 1,
        "best_val_loss": best_va, "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}  R²={out['r2']:.4f}  Sp={out['spearman']:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
