"""EC classification (depth-1 / 2 / 3 multi-label).

Same conv1d trunk + per-protein MLP head as binding-site / GO; the
only thing that changes is the label set. Each protein gets a
multi-hot label vector over the EC classes that appear in the train
split (truncated to the requested depth — `1.4.3.21` → `1`, `1.4`,
`1.4.3`).

Usage:

    python downstreams/ec.py \\
        --features data/<feature>.npz \\
        [--codebook data/<feature>.codebook.npy] \\
        --labels   data/misato_ec.json   # {pid: [ec_str, ...]}
        --splits   data/splits.json \\
        --split-name structure \\
        --depth 1 \\
        --out      runs/ec_d1__<feature>.json
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "8")


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
    common_arg_parser, load_split, encode_labels, train_eval,
)


def truncate_ec(ec: str, depth: int) -> str | None:
    parts = str(ec).split(".")
    if len(parts) < depth:
        return None
    return ".".join(parts[:depth])


def main() -> int:
    ap = common_arg_parser(__doc__)
    ap.add_argument("--depth", type=int, choices=(1, 2, 3), default=1)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats, in_dim = load_features_per_pid(args.features, args.codebook)
    feats = {p.upper(): v for p, v in feats.items()}

    raw = json.loads(args.labels.read_text())
    truncated: dict[str, list[str]] = {}
    for pid, ecs in raw.items():
        if not ecs:
            continue
        s = {t for ec in ecs if (t := truncate_ec(ec, args.depth)) is not None}
        if s:
            truncated[pid.upper()] = sorted(s)

    split = load_split(args.splits, args.split_name)
    labels_by_pid, classes = encode_labels(truncated, split["train"])
    print(f"[setup] depth={args.depth} in_dim={in_dim} "
          f"n_classes={len(classes)} train={len(split['train'])} "
          f"val={len(split['val'])} test={len(split['test'])}", flush=True)

    out = train_eval(
        feats=feats, labels_by_pid=labels_by_pid, splits=split,
        in_dim=in_dim, n_classes=len(classes),
        device=device, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed)
    out["depth"] = args.depth
    out["split"] = args.split_name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] {args.out}  top1={out.get('top1_hit', float('nan')):.4f}  "
          f"macroAP={out.get('macro_ap', float('nan')):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
