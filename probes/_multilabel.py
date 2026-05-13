"""Shared multi-label train/eval loop for EC and GO.

Both tasks are per-protein multi-label classification on the same
conv1d trunk + masked mean+max pool + MLP head. The scoring metrics
are identical (top-1 hit, macro/micro AP, macro/micro F1@0.5).
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "8")


import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

from ._conv1d_head import (
    PerProteinHead, load_features_per_pid, collate_residue_features,
)


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """Multi-label metric bundle. Macro statistics are averaged only
    over classes with ≥1 positive in test (sklearn convention)."""
    nz = (y.sum(0) > 0)
    if int(nz.sum()) == 0:
        return {"top1_hit": float("nan"), "macro_ap": float("nan"),
                "micro_ap": float("nan"), "macro_f1@0.5": float("nan"),
                "micro_f1@0.5": float("nan"), "n_classes_in_test": 0}
    macro_ap = float(average_precision_score(y[:, nz], p[:, nz], average="macro"))
    micro_ap = float(average_precision_score(y[:, nz], p[:, nz], average="micro"))
    pred = (p > 0.5).astype(np.int8)
    macro_f1 = float(f1_score(y[:, nz], pred[:, nz], average="macro",
                                zero_division=0))
    micro_f1 = float(f1_score(y[:, nz], pred[:, nz], average="micro",
                                zero_division=0))
    pred_top = p.argmax(-1)
    top1 = float(np.mean([y[i, pred_top[i]] > 0 for i in range(y.shape[0])]))
    return {"top1_hit": top1, "macro_ap": macro_ap, "micro_ap": micro_ap,
            "macro_f1@0.5": macro_f1, "micro_f1@0.5": micro_f1,
            "n_classes_in_test": int(nz.sum())}


def train_eval(
    *,
    feats: dict[str, np.ndarray],
    labels_by_pid: dict[str, np.ndarray],
    splits: dict[str, list[str]],
    in_dim: int,
    n_classes: int,
    device: str,
    epochs: int = 40,
    patience: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Train/eval the shared multi-label classifier. Returns the
    metric dict + bookkeeping fields."""
    torch.manual_seed(seed)
    # Match the canonical sweep's bucket-build order so that for a given
    # seed, the SGD trajectory is identical to the paper's run. Canonical
    # iterates `for pid in feats:` (NPZ-key dict order) and drops val/test
    # items whose label vector is all-zero (no overlap with train classes)
    # but KEEPS train items with all-zero labels. See
    # submission_exp/src/conv1d_GO_misato.py:67-78 and
    # submission_exp/src/ec_classify_conv1d_misato.py:533-540 (the latter
    # iterates a set, which we replace with `feats` iteration in
    # ec_classify_conv1d_misato.py at submission time for the same reason).
    split_sets = {s: set(splits[s]) for s in ("train", "val", "test")}
    bucket = {s: [] for s in ("train", "val", "test")}
    for pid in feats:
        # Canonical drops empty-label pids from every split — for EC this
        # is implicit (train_classes derives from train labels, so every
        # train pid has sum>0 automatically), but for GO it is the
        # `if s: out[pid] = s` filter inside build_go_labelset() that
        # excludes pids whose GO terms don't intersect the top-K set.
        # We mirror that filter here for both tasks.
        if pid not in labels_by_pid or labels_by_pid[pid].sum() == 0:
            continue
        for s in ("train", "val", "test"):
            if pid in split_sets[s]:
                bucket[s].append(pid)
                break
    n = {s: len(bucket[s]) for s in bucket}
    if not bucket["train"] or not bucket["test"]:
        return {"error": f"empty split: {n}"}

    model = PerProteinHead(in_dim=in_dim, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best_va = float("inf"); best_state = None; no_imp = 0

    def _epoch(items: list[str], train: bool) -> float:
        model.train(train)
        # Canonical never reads the train loss, so its CUDA stream runs
        # without inter-step host syncs (see
        # ec_classify_conv1d_misato.py:455-461). Reading `loss.detach()`
        # after every train step forces a GPU→CPU transfer that perturbs
        # cuDNN's kernel scheduling and changes the non-determinism
        # pattern. We mirror canonical by tracking the loss only on the
        # val pass (no_grad scope), and discarding train losses.
        if train:
            perm = torch.randperm(len(items), generator=g).tolist()
            items = [items[i] for i in perm]
            for s in range(0, len(items), batch_size):
                batch = items[s:s + batch_size]
                b = collate_residue_features(
                    batch, feats,
                    labels={p: labels_by_pid[p] for p in batch},
                    label_kind="per_protein_vector")
                X = b["x"].to(device); M = b["mask"].to(device); y = b["y"].to(device)
                z = model(X, M)
                loss = F.binary_cross_entropy_with_logits(z, y)
                opt.zero_grad(); loss.backward(); opt.step()
            return 0.0
        # val: sample-weighted mean matching canonical (l.item() * batch_size)
        total = 0.0; n_samples = 0
        with torch.no_grad():
            for s in range(0, len(items), batch_size):
                batch = items[s:s + batch_size]
                b = collate_residue_features(
                    batch, feats,
                    labels={p: labels_by_pid[p] for p in batch},
                    label_kind="per_protein_vector")
                X = b["x"].to(device); M = b["mask"].to(device); y = b["y"].to(device)
                z = model(X, M)
                loss = F.binary_cross_entropy_with_logits(z, y)
                total += float(loss.item()) * X.shape[0]; n_samples += X.shape[0]
        return total / max(n_samples, 1)

    last_ep = 0
    for ep in range(epochs):
        last_ep = ep + 1
        _ = _epoch(bucket["train"], train=True)
        va = _epoch(bucket["val"], train=False)
        if va < best_va - 1e-4:
            best_va = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_p, all_y = [], []
    for s in range(0, len(bucket["test"]), batch_size):
        batch = bucket["test"][s:s + batch_size]
        b = collate_residue_features(
            batch, feats,
            labels={p: labels_by_pid[p] for p in batch})
        X = b["x"].to(device); M = b["mask"].to(device)
        with torch.no_grad():
            all_p.append(torch.sigmoid(model(X, M)).cpu().numpy())
        all_y.append(b["y"].numpy())
    p = np.concatenate(all_p, 0); y = np.concatenate(all_y, 0)
    out = metrics(p, y)
    out.update({"in_dim": int(in_dim), "n_classes": int(n_classes),
                "n_train": n["train"], "n_val": n["val"], "n_test": n["test"],
                "epochs_run": int(last_ep), "best_val_loss": float(best_va),
                "seed": int(seed)})
    return out


# ----------------------------------------------------------------------
# Shared CLI builders


def common_arg_parser(description: str):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--codebook", default=None, type=Path)
    ap.add_argument("--labels",   required=True, type=Path,
                     help=".json {pid: list[str]} of canonical class labels")
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
    return ap


def load_split(splits_path: Path, split_name: str) -> dict:
    s = json.loads(splits_path.read_text())
    if split_name in s:
        s = s[split_name]
    return {k: [p.upper() for p in s.get(k, [])] for k in ("train", "val", "test")}


def encode_labels(labels_raw: dict, train_pids: list[str]) -> tuple[dict, list[str]]:
    """Multi-hot encode (pid -> set of class strings) using the class
    space induced by the train split. Returns
    (pid -> (n_classes,) float32, list[class_str])."""
    train_classes = sorted({c for p in train_pids
                             for c in labels_raw.get(p, [])})
    cls_to_idx = {c: i for i, c in enumerate(train_classes)}
    out: dict[str, np.ndarray] = {}
    n_classes = len(train_classes)
    for pid, classes in labels_raw.items():
        v = np.zeros(n_classes, dtype=np.float32)
        for c in classes:
            if c in cls_to_idx:
                v[cls_to_idx[c]] = 1.0
        out[pid] = v
    return out, train_classes
