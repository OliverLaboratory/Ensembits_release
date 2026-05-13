"""Per-residue binding-site classification.

Trains a 3-layer Conv1d head (shared with EC / GO / affinity) on
per-residue features and reports AUROC + AP on the test split.
Labels are binary per residue; loss is masked BCE-with-logits.

Usage:

    python downstreams/binding_site.py \\
        --features data/<feature>.npz \\
        [--codebook data/<feature>.codebook.npy] \\
        --labels   data/binding_site_labels.npz \\
        --splits   data/splits.json \\
        --split-name structure \\
        --out      runs/binding_site__<feature>.json

Inputs:
    --features  pickle / npz   dict[pid -> (L,) int64] (token IDs) or
                                 dict[pid -> (L, D) float32] (continuous)
    --codebook  .npy           required iff --features is token IDs
    --labels    .npz           dict[pid -> (L,) int8 in {0, 1}]
    --splits    .json          {<split>: {train: [pid,...], val:[...], test:[...]}}
    --split-name str           sub-key of --splits to use
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
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from probes._conv1d_head import (  # noqa: E402
    PerResidueHead, load_features_per_pid, collate_residue_features,
)


def _load_split(splits_path: Path, split_name: str) -> dict:
    s = json.loads(splits_path.read_text())
    if split_name in s:
        s = s[split_name]
    return {k: [p.upper() for p in s.get(k, [])] for k in ("train", "val", "test")}


def _build_buckets(feats: dict, labels: dict, split: dict,
                    min_len: int = 10) -> dict[str, list[str]]:
    """Match canonical iteration order: walk `feats` in NPZ-key order and
    bucket each pid into its split. Mirrors
    submission_exp/src/conv1d_binding_baselines.py:377-385 so that, for a
    given seed, the SGD trajectory is identical to the paper's run."""
    split_sets = {s: set(split[s]) for s in ("train", "val", "test")}
    bucket: dict[str, list[str]] = {s: [] for s in ("train", "val", "test")}
    for pid in feats:
        if pid not in labels or feats[pid].shape[0] < min_len:
            continue
        for s in ("train", "val", "test"):
            if pid in split_sets[s]:
                bucket[s].append(pid)
                break
    return bucket


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--codebook", default=None, type=Path)
    ap.add_argument("--labels",   required=True, type=Path)
    ap.add_argument("--splits",   required=True, type=Path)
    ap.add_argument("--split-name", default="structure")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    feats, in_dim = load_features_per_pid(args.features, args.codebook)
    feats = {p.upper(): v for p, v in feats.items()}
    labels = {p.upper(): np.asarray(v).astype(np.float32)
              for p, v in np.load(args.labels, allow_pickle=False).items()}
    split = _load_split(args.splits, args.split_name)
    bucket = _build_buckets(feats, labels, split)
    n = {s: len(bucket[s]) for s in bucket}
    if not bucket["train"] or not bucket["test"]:
        raise SystemExit(f"empty split: {n}")
    print(f"[setup] in_dim={in_dim} train={n['train']} val={n['val']} test={n['test']}",
          flush=True)

    model = PerResidueHead(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    best_va = float("inf")
    best_state = None
    no_imp = 0

    def _epoch(items: list[str], train: bool) -> float:
        model.train(train)
        total = 0.0; n_batch = 0
        if train:
            perm = torch.randperm(len(items), generator=g).tolist()
            items = [items[i] for i in perm]
        for s in range(0, len(items), args.batch_size):
            batch = items[s:s + args.batch_size]
            b = collate_residue_features(batch, feats, labels,
                                          label_kind="per_residue")
            X = b["x"].to(device); M = b["mask"].to(device); y = b["y"].to(device)
            with torch.set_grad_enabled(train):
                z = model(X, M)
                loss = F.binary_cross_entropy_with_logits(
                    z, y, weight=M.float(), reduction="sum") / M.float().sum().clamp(1)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()); n_batch += 1
        return total / max(n_batch, 1)

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
    all_p, all_y = [], []
    for s in range(0, len(bucket["test"]), args.batch_size):
        batch = bucket["test"][s:s + args.batch_size]
        b = collate_residue_features(batch, feats, labels)
        X = b["x"].to(device); M = b["mask"].to(device); y = b["y"]
        with torch.no_grad():
            p = torch.sigmoid(model(X, M)).cpu().numpy()
        for i, pid in enumerate(batch):
            L = b["mask"][i].sum().item()
            all_p.append(p[i, :L])
            all_y.append(y[i, :L].numpy())
    p_flat = np.concatenate(all_p)
    y_flat = np.concatenate(all_y)
    auroc = float(roc_auc_score(y_flat, p_flat))
    ap = float(average_precision_score(y_flat, p_flat))

    out = {
        "auroc": auroc, "ap": ap,
        "in_dim": int(in_dim), "n_train": n["train"], "n_val": n["val"],
        "n_test": n["test"], "n_test_residues": int(len(p_flat)),
        "split": args.split_name, "epochs": ep + 1, "best_val_loss": best_va,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}  AUROC={auroc:.4f}  AP={ap:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
