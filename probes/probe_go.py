"""GO term prediction (top-K most-frequent terms multi-label).

Same conv1d trunk + per-protein MLP head as binding-site / EC. The
class space is the K most-frequent GO terms in the train split
(default `K=50`). All ontology branches (BP / MF / CC) are pooled
together; the per-protein label is the union of its GO annotations
intersected with the chosen K.

Usage:

    python downstreams/go.py \\
        --features data/<feature>.npz \\
        [--codebook data/<feature>.codebook.npy] \\
        --labels   data/misato_go.json   # {pid: [go_term, ...]}
        --splits   data/splits.json \\
        --split-name structure \\
        --top-k 50 \\
        --out      runs/go_top50__<feature>.json
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "8")


import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from probes._conv1d_head import load_features_per_pid  # noqa: E402
from probes._multilabel import (  # noqa: E402
    common_arg_parser, load_split, train_eval,
)


def main() -> int:
    ap = common_arg_parser(__doc__)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats, in_dim = load_features_per_pid(args.features, args.codebook)
    feats = {p.upper(): v for p, v in feats.items()}

    raw_lower = {p.upper(): list(terms) for p, terms in
                 json.loads(args.labels.read_text()).items() if terms}
    split = load_split(args.splits, args.split_name)
    # Match canonical's class-counter iteration order. Canonical iterates
    # `feats` (NPZ-key order) intersected with the train split. A naive
    # `for p in split['train']` walks JSON-list order instead, and with ties
    # in `Counter.most_common(top_k)` the resulting top-K list is in a
    # different order, which permutes cls_to_idx and the output-layer
    # weight init under the same seed → divergent SGD trajectory.
    train_set = set(split["train"])
    counter: collections.Counter[str] = collections.Counter()
    for p in feats:
        if p in train_set and p in raw_lower:
            counter.update(raw_lower[p])
    # Canonical returns `sorted(classes)` (alphabetical) — see
    # tokenize_misato_GO_eval.py:72. Keeping `most_common()` order here
    # would permute cls_to_idx and the output-layer weight init.
    classes = sorted(c for c, _ in counter.most_common(args.top_k))
    cls_set = set(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    print(f"[setup] top_k={args.top_k} in_dim={in_dim} "
          f"n_classes={n_classes} train={len(split['train'])} "
          f"val={len(split['val'])} test={len(split['test'])}", flush=True)

    labels_by_pid: dict[str, np.ndarray] = {}
    for pid, terms in raw_lower.items():
        v = np.zeros(n_classes, dtype=np.float32)
        for t in terms:
            if t in cls_set:
                v[cls_to_idx[t]] = 1.0
        labels_by_pid[pid] = v

    out = train_eval(
        feats=feats, labels_by_pid=labels_by_pid, splits=split,
        in_dim=in_dim, n_classes=n_classes,
        device=device, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed)
    out["top_k"] = args.top_k
    out["split"] = args.split_name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}  top1={out.get('top1_hit', float('nan')):.4f}  "
          f"microAP={out.get('micro_ap', float('nan')):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
