"""CryptoBench: per-residue context conv1d probe on frozen representations (SECONDARY probe).

Role (DECISIONS.md CB1, authors' answer 2026-10-20): the primary CryptoBench probe is the
pLM-NN-shaped MLP of `cryptobench_mlp_probe.py` (the probe behind the rebuttal numbers);
this context conv1d head is the secondary probe, reported in the appendix with the
alphabet-size bound of the context-free MLP. It produced `rebuttal/cryptobench.md` (old
flagship), a different and less favourable ranking than the MLP; that is why both are run
and the choice was fixed before any `_rv` number exists.

What it does
------------
Step 3b of the CryptoBench pipeline (see `README.md` in this folder).

  1. Inputs per chain: token ids of `cryptobench_tokens_<tag>.npz` embedded with the
     tokenizer's FROZEN codebook (`--features tokens`), amino-acid ids embedded with an
     identity table (`aa`, 21 incl. unknown), uniform random ids (`random`, chance control),
     or Ensembits z / q / q0 (`dense`, `cryptobench_dense_<tag>.npz`, `--readout`).
     FIXED(REAUDIT CB2 F3; authors' TENTATIVE answer 2026-10-22): every row's inputs are
     standardised with the training-fold residue mean / sd (`--standardize train`, default;
     the frozen table rows are transformed, which equals standardising the per-residue
     inputs; codebook spread across tokenizers was 30x in RMS); `--standardize none` is the
     2026-10-21 behaviour. The table / features keep q0 identical to the token readout. Token / dense files must carry the
     port's provenance sidecar (`cryptobench_common.check_token_file`). TRAIN chains are cut
     at `--max-len` 1024 (3 train chains: 6j35_B, 6kqf_C, 6kqf_D); validation / test chains
     stay whole (2 validation chains, 2po4_A and 7oum_C, are longer than 1,024 and are kept
     whole, while the source cut every split).
  2. Folds: train = train-0, train-1, train-2; val = train-3; test = test.
  3. Head (`ResidueConv1D`, same shape as the paper's `BindingConv1D`): input (L, D) ->
     3 x [Conv1d(k=5, 128) + GELU + Dropout 0.1] -> Linear -> one logit per residue; padded
     positions are zeroed before EVERY convolution.
  4. Training: masked BCE with pos_weight = #neg / #pos over train residues, AdamW lr 1e-3,
     weight decay 1e-4, batch 16 chains (length-bucketed), grad clip 5, LR warmup 5 epochs ->
     cosine, max 60 epochs, early stopping on the VALIDATION pooled AUROC (min 20 epochs,
     patience 12; `--selection`), best val state restored.
  5. Metrics (written, never printed; sealed until the lambda choice, DECISIONS.md CB8):
     per-entry AUROC (entry = apo id, chains concatenated; primary), per-entry AP, pooled
     AUROC, pooled AP on test and validation. 5 seeds (0-4).
  6. Outputs per seed: `<name>__seed<s>.json` + `.preds.npz` (sigmoid probabilities of the
     selected epoch, val and test, items = entries; read by protein_level_stats.py);
     aggregate `<name>.json`. Outputs that exist, verify and carry the current input
     fingerprint are skipped; otherwise they are moved to `_pre_<date>/` and redone.

Options for the authors' TENTATIVE answers of 2026-10-22 (defaults = the tentative choice,
recorded in every result, part of the input fingerprint of the rows they affect):
  --standardize {train, none}   conv input scale (all conv rows)
  --selection {val_pooled_auroc, val_per_entry_auroc}   epoch selection (all rows)
  --aa-order {collaborator, alphabetical}   aa row only
Input fingerprint: as cryptobench_mlp_probe.py (manifest, input file + sidecar + codebook,
code sha256, result-affecting arguments).

Paper
-----
Not in the paper yet: CryptoBench goes into the main text
(`camera_ready/require_fix.md` #24, `camera_ready/code_fix.md` E3).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Head shape: `appendix.tex` \\label{app:downstreams} \\paragraph{Binding Site}
(per-residue 1D-CNN head; `downstream/tokenize_misato_binding_eval.py` BindingConv1D).
Benchmark: Škrhák et al., Bioinformatics 41(1):btae745, 2025. Published references
(CryptoBench README): pLM-NN (ESM2-3B + NN) 0.86 AUC / 0.36 AUPRC on the full test set;
PocketMiner and P2Rank were evaluated on SUBSETS (CB-PM, CB-P2RANK-apo), not on this test set
(audit A finding C5).

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/cryptobench_probe.py`.
Changes: comments; ROOT from `ensembits.paths.REPO_ROOT`; locations from `_cb_paths.py`;
the output folder is created before the first write.
Fixes 2026-10-16 (code_fix P22 / P9 / P17): --tokens has no default; sample std over seeds;
the token file's provenance copied into the JSON.
Fixes 2026-10-21 (port step 5, camera_ready/cryptobench_audit/PORT.md; FIXED markers):
  * padded positions zeroed before every convolution (audit B finding 10; the first fix of
    the morning masked the input embedding only, so layer-2 / 3 inputs at pad positions
    were GELU(bias) and the last residues still depended on the batch partners);
  * per-ENTRY metrics (DECISIONS.md CB6; was per chain) and per-entry AP (was missing);
  * per-seed result JSON + prediction sidecar (DECISIONS.md CB7; replaces the seed-mean
    preds_*.npz, which had no sidecar record);
  * dense inputs (Ensembits z / q / q0, DECISIONS.md CB4);
  * provenance gate on token files; no metric printed (CB8);
  * `--max-len` cuts train chains only. Correction 2026-10-22 (REAUDIT CB1 F1 / CB2 F8): 3
    TRAIN chains are longer than 1,024 (6j35_B, 6kqf_C, 6kqf_D) and are cut; 2 VALIDATION
    chains (2po4_A 1,094, 7oum_C 1,033), which the source cut at 1,024, are now kept whole
    (deliberate deviation: every evaluated residue is scored); no test chain exceeds 901.
Fixes 2026-10-22: tentative options above; aa ids in the collaborator's order by default;
input fingerprint and stale-output handling; row checkpoint check (--expect-checkpoint).
Model, optimiser, schedule, early stopping and loss are unchanged.

Used by the reported model: yes (secondary CryptoBench probe).

Inputs / outputs
----------------
Read  : `<manifest-dir>/<set>_manifest.json`, `<structs-root>/cryptobench_<set>/*.npz` (y, seq),
        `<tokens-dir>/cryptobench_tokens_<tag>.npz` (+ codebook, provenance) or
        `<tokens-dir>/cryptobench_dense_<tag>.npz` (+ provenance).
Write : `<out-dir>/conv/<name>__seed<s>.json`, `.preds.npz`; `<out-dir>/conv/<name>.json`.

Dependencies left in the research repo: none (torch, scikit-learn).

How to run
----------
  $PY downstream/cryptobench/cryptobench_probe.py --run-tag <tag> --row ens_lam0p25_rv \\
      --features tokens --tokens ens_lam0p25_rv
  $PY downstream/cryptobench/cryptobench_probe.py --run-tag <tag> --row aa --features aa
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402,F401  data/ and output/ stay in the research repo
import _cb_paths as CBP  # noqa: E402
import cryptobench_common as CC  # noqa: E402

PROBE = "conv1d_context"
AAS = CC.AAS   # length 20; aa ids come from CC.aa_ids in --aa-order


class ResidueConv1D(nn.Module):
    """Per-residue binary classifier (same shape as the paper's BindingConv1D).

    emb_table given: input = token ids embedded with the frozen table; else input = (B, L, D)
    features (dense readouts) with D = in_dim."""

    def __init__(self, emb_table=None, hidden=128, depth=3, kernel=5, dropout=0.1, in_dim=None):
        super().__init__()
        # frozen lookup: the tokenizer's codebook is the residue feature
        self.emb = (nn.Embedding.from_pretrained(emb_table.float(), freeze=True)
                    if emb_table is not None else None)
        layers, prev = [], (emb_table.shape[1] if emb_table is not None else int(in_dim))
        for _ in range(depth):
            layers += [nn.Conv1d(prev, hidden, kernel, padding=kernel // 2),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, inputs, mask=None):
        x = self.emb(inputs) if self.emb is not None else inputs.float()
        x = x.transpose(1, 2)                                          # (B, D, L)
        m = None if mask is None else mask.unsqueeze(1).to(x.dtype)    # (B, 1, L)
        for layer in self.conv:
            # FIXED(CB audit B finding 10, 2026-10-21): zero the padded positions before
            #   EVERY convolution (was: before the first only), so a residue's output does
            #   not depend on the padding added for its batch partners.
            if m is not None and isinstance(layer, nn.Conv1d):
                x = x * m
            x = layer(x)
        return self.head(x.transpose(1, 2)).squeeze(-1)


def batches(items, bs, device, shuffle, rng):
    """Length-bucketed padded batches of (key, X, y); yields (inputs, y, mask, keys).
    X is int64 ids (L,) or float features (L, D)."""
    order = np.argsort([len(x[2]) for x in items], kind="stable")   # length-bucketed padding
    blocks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        rng.shuffle(blocks)
    for b in blocks:
        chunk = [items[i] for i in b]
        L = max(len(c[2]) for c in chunk)
        X0 = chunk[0][1]
        if X0.ndim == 1:
            X = np.zeros((len(chunk), L), dtype=np.int64)
        else:
            X = np.zeros((len(chunk), L, X0.shape[1]), dtype=np.float32)
        y = np.zeros((len(chunk), L), dtype=np.float32)
        m = np.zeros((len(chunk), L), dtype=np.float32)
        for j, (_, a, t) in enumerate(chunk):
            X[j, :len(a)] = a
            y[j, :len(t)] = t
            m[j, :len(a)] = 1.0
        yield (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device),
               torch.from_numpy(m).to(device), [c[0] for c in chunk])


@torch.no_grad()
def predict(model, items, device, bs):
    """{key: sigmoid probabilities (L,)} of every chain (eval mode, fixed order)."""
    model.eval()
    out = {}
    for X, y, m, keys in batches(items, bs, device, False, None):
        p = torch.sigmoid(model(X, m).float())
        for j, k in enumerate(keys):
            n = int(m[j].sum().item())
            out[k] = p[j, :n].cpu().numpy()
    return out


def pooled_auc(preds, items):
    from sklearn.metrics import roc_auc_score
    Y = np.concatenate([t for k, _, t in items])
    P = np.concatenate([preds[k] for k, _, _ in items])
    return float(roc_auc_score(Y, P))


def run_seed(tr, va, te, emb, seed, args, device, in_dim=None):
    """Train one probe (one seed) with early stopping on val AUROC; predict val and test."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ResidueConv1D(emb, hidden=args.hidden, depth=args.depth, in_dim=in_dim).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    # class weighting: pos_weight = negatives / positives over all train residues
    npos = sum(float(t.sum()) for _, _, t in tr)
    ntot = sum(len(t) for _, _, t in tr)
    pos_weight = torch.tensor([(ntot - npos) / max(npos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def lr_at(ep):
        # linear warmup, then cosine decay to 0 at max_epochs
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        p = (ep - args.warmup) / max(1, args.max_epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    best, best_state, best_ep, bad = -1.0, None, -1, 0
    ep = -1
    for ep in range(args.max_epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(ep)
        model.train()
        for X, y, m, _ in batches(tr, args.batch, device, True, rng):
            opt.zero_grad()
            l = (lossf(model(X, m), y) * m).sum() / m.sum()  # masked mean over real residues
            l.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
        # epoch selection on the VALIDATION fold (train-3) only
        pv = predict(model, va, device, args.batch)
        v = (pooled_auc(pv, va) if args.selection == "val_pooled_auroc"
             else split_metrics(pv, va)[0]["auroc_per_entry"])
        if v > best + 1e-5:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if ep + 1 >= args.min_epochs and bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return {"val": predict(model, va, device, args.batch),
            "test": predict(model, te, device, args.batch),
            "best_epoch": best_ep, "epochs_run": ep + 1,
            "pos_weight": float(pos_weight.item())}


def split_metrics(preds, items):
    keys = [k for k, _, _ in items]
    lay = CC.entry_layout(keys)
    lab = {k: t.astype(np.int64) for k, _, t in items}
    return CC.entry_metrics(CC.entry_concat(preds, lay), CC.entry_concat(lab, lay)), lay, lab


def out_name(args) -> str:
    readout = {"tokens": "token", "aa": "onehot", "random": "random",
               "dense": args.readout}[args.features]
    return f"{args.row}__{readout}"


def train_stats(rows, table=None):
    """Training-fold residue mean / sd (population sd + 1e-6, the MLP's rule; sd = 1 for a
    column with zero training variance, FIXLIST2 item 3 / FIXAUDIT FX2 F2) of the probe
    inputs: rows = [(key, X, y)] with X ids (table given) or (L, D) features."""
    if table is not None:
        T = np.asarray(table, dtype=np.float64)
        counts = np.bincount(np.concatenate([X for _, X, _ in rows]), minlength=len(T))
        n = counts.sum()
        mu = counts @ T / n
        var = np.maximum(counts @ (T ** 2) / n - mu ** 2, 0.0)
    else:
        F = np.concatenate([X for _, X, _ in rows]).astype(np.float64)
        mu, var = F.mean(0), F.var(0)
    # FIXED(FIXAUDIT FX2 F2 / FX1 F3, 2026-10-22): a column constant over the training residues
    #   (e.g. the aa "unknown" column when no training residue is unknown) got sd = 1e-6, which
    #   turns a val / test residue there into ~1e6; such columns now keep sd = 1.
    sd = np.where(var <= 1e-12, 1.0, np.sqrt(var) + 1e-6)
    return mu.astype(np.float32), sd.astype(np.float32)


def input_paths(args, paths):
    if args.features in ("tokens", "random"):
        return (paths.tokens_dir / f"cryptobench_tokens_{args.tokens}.npz",
                paths.tokens_dir / f"cryptobench_tokens_{args.tokens}_codebook.npy")
    if args.features == "dense":
        return paths.tokens_dir / f"cryptobench_dense_{args.dense}.npz", None
    return None, None


def fingerprint(args, paths) -> dict:
    man = CBP.manifest_path(paths, args.set)
    src, cb = input_paths(args, paths)
    here = Path(__file__)
    drop = () if args.features == "aa" else ("aa_order",)
    return {"manifest_sha256": CC.file_sha256(man) if man.exists() else None,
            "input": CC.input_record(src) if src is not None and src.exists() else None,
            "codebook_sha256": CC.file_sha256(cb) if cb is not None and cb.exists() else None,
            "code": CC.code_sha256(here, CC.__file__, here.parents[1] / "protein_preds.py"),
            "args": CC.args_record(args, drop)}


def is_done(argv) -> bool:
    args = parser().parse_args(argv)
    paths = CBP.resolve_paths(args)
    return CC.probe_outputs_done(paths.out_dir / "conv", out_name(args), args.seeds,
                                 fingerprint(args, paths))


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--row", required=True, help="row name (e.g. ens_lam0p25_rv, aminoaseed_wc, aa)")
    # FIXED(code_fix P22, 2026-10-16): old: default "dih256_fapeH" (the old mdCATH+MISATO
    #   flagship's tokens). New: no default; required with --features tokens / random.
    ap.add_argument("--tokens", default=None,
                    help="cryptobench_tokens_<tag>.npz (required with --features tokens / random)")
    ap.add_argument("--dense", default=None, help="cryptobench_dense_<tag>.npz (--features dense)")
    ap.add_argument("--readout", choices=["z", "q", "q0"], default=None)
    ap.add_argument("--standardize", choices=["train", "none"], default="train",
                    help="input scale (tentative default: training-fold mean / sd for every row)")
    ap.add_argument("--selection", choices=["val_pooled_auroc", "val_per_entry_auroc"],
                    default="val_pooled_auroc", help="epoch selection metric (tentative default)")
    ap.add_argument("--aa-order", choices=sorted(CC.AA_ORDERS), default="collaborator",
                    help="aa row only: id order (tentative default: collaborator)")
    ap.add_argument("--expect-checkpoint", default=None,
                    help="Ensembits rows: the row's checkpoint folder (gate)")
    ap.add_argument("--features", choices=["tokens", "aa", "random", "dense"], default="tokens")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--emb-dim", type=int, default=128, help="random table dim for --features random")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--set", default="apo",
                    help="which built set the labels/structures come from: "
                         "apo | holo | apo_main | holo_main")
    ap.add_argument("--redo", action="store_true", help="recompute even if the outputs verify")
    ap.add_argument("--allow-legacy-tokens", action="store_true",
                    help="skip the provenance gate (comparisons with old caches only)")
    CBP.add_path_args(ap)
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = parser()
    args = ap.parse_args(argv)
    if args.features in ("tokens", "random") and not args.tokens:
        ap.error("--tokens <tag> is required with --features tokens / random (code_fix P22)")
    if args.features == "dense" and not (args.dense and args.readout):
        ap.error("--dense and --readout required with --features dense")
    paths = CBP.resolve_paths(args, create=("out_dir",))
    out_dir = paths.out_dir / "conv"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_name(args)
    fp = fingerprint(args, paths)
    if CC.probe_outputs_done(out_dir, name, args.seeds, fp) and not args.redo:
        print(f"[conv] {name}: outputs exist, verify and match the inputs; skipped")
        return 0
    moved = CC.move_aside_probe_outputs(out_dir, name)
    if moved:
        print(f"[conv] {name}: {len(moved)} old output files moved aside (stale or --redo)")
    device = args.device if torch.cuda.is_available() else "cpu"
    man_path = CBP.manifest_path(paths, args.set)
    man = json.loads(man_path.read_text())
    set_dir = CBP.structs_dir(paths, args.set)

    prov, src, emb, in_dim = None, None, None, None
    if args.features in ("tokens", "random"):
        src = paths.tokens_dir / f"cryptobench_tokens_{args.tokens}.npz"
        prov = CC.check_token_file(src, man_path, args.allow_legacy_tokens,
                                   expect_checkpoint=args.expect_checkpoint)
    elif args.features == "dense":
        src = paths.tokens_dir / f"cryptobench_dense_{args.dense}.npz"
        prov = CC.check_token_file(src, man_path, args.allow_legacy_tokens,
                                   expect_checkpoint=args.expect_checkpoint)
    mode = "tokens" if args.features in ("tokens", "random") else args.features
    chains = CC.load_chains(man, set_dir, mode, src, args.readout, max_len=args.max_len,
                            aa_order=args.aa_order)
    if args.features == "random":
        rng0 = np.random.default_rng(0)
        chains = [(k, f, rng0.integers(0, int(X.max()) + 1, size=X.shape), y)
                  for k, f, X, y in chains]
    by = {s: [(k, X, y.astype(np.float32)) for k, f, X, y in chains if CC.split_of(f) == s]
          for s in ("train", "val", "test")}
    tr, va, te = by["train"], by["val"], by["test"]

    # Frozen embedding table per feature mode (dense: none, features fed directly).
    if args.features == "tokens":
        emb = torch.from_numpy(np.load(paths.tokens_dir /
                                       f"cryptobench_tokens_{args.tokens}_codebook.npy"))
    elif args.features == "aa":
        emb = torch.eye(len(AAS) + 1)                       # one-hot AA (+unknown)
    elif args.features == "random":
        g = torch.Generator().manual_seed(0)
        M = int(max(int(a.max()) for _, a, _ in tr + va + te)) + 1
        emb = torch.randn(M, args.emb_dim, generator=g) * 0.02
    else:
        in_dim = tr[0][1].shape[1]
    # FIXED(REAUDIT CB2 F3; tentative answer 2026-10-22): training-fold standardisation.
    if args.standardize == "train":
        if emb is not None:
            mu, sd = train_stats(tr, emb.numpy())
            emb = (emb - torch.from_numpy(mu)) / torch.from_numpy(sd)
        else:
            mu, sd = train_stats(tr)
            tr, va, te = ([(k, ((X - mu) / sd).astype(np.float32), y) for k, X, y in part]
                          for part in (tr, va, te))
    print(f"[conv] {name}: chains train/val/test={len(tr)}/{len(va)}/{len(te)} "
          f"input={'table ' + str(tuple(emb.shape)) if emb is not None else 'dense ' + str(in_dim)} "
          f"seeds={args.seeds} (metrics not printed)", flush=True)

    rows, seed_files = [], []
    for s in range(args.seeds):
        r = run_seed(tr, va, te, emb, s, args, device, in_dim)
        mv, lay_v, yv = split_metrics(r["val"], va)
        mt, lay_t, yt = split_metrics(r["test"], te)
        f = out_dir / f"{name}__seed{s}.json"
        res = {"name": name, "row": args.row, "probe": PROBE, "set": args.set, "seed": s,
               "features": args.features, "readout": args.readout or args.features,
               "args": vars(args), "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
               "pos_weight": r["pos_weight"], "selection": f"validation (train-3) {args.selection}",
               "options": {"standardize": args.standardize, "selection": args.selection,
                           "aa_order": args.aa_order if args.features == "aa" else None},
               "metrics": {"val": mv, "test": mt}, "input_provenance": prov,
               "checkpoint": (prov or {}).get("checkpoint"), "input_fingerprint": fp}
        CC.write_seed_result(f, preds={"val": r["val"], "test": r["test"]},
                             labels={"val": yv, "test": yt}, layout={"val": lay_v, "test": lay_t},
                             meta={"row": args.row, "probe": PROBE, "seed": s, "set": args.set,
                                   "readout": res["readout"]}, result=res)
        rows.append(mt); seed_files.append(f.name)
        print(f"  seed {s}: done (best epoch {r['best_epoch']}, {r['epochs_run']} epochs)", flush=True)
    agg = {"name": name, "row": args.row, "probe": PROBE, "set": args.set,
           "features": args.features, "readout": args.readout or args.features,
           "test_agg": CC.agg_over_seeds(rows), "std": "sample (ddof=1) over probe seeds",
           "seed_files": seed_files, "input_provenance": prov,
           "options": {"standardize": args.standardize, "selection": args.selection,
                       "aa_order": args.aa_order if args.features == "aa" else None},
           "input_fingerprint": fp, "args": vars(args)}
    (out_dir / f"{name}.json").write_text(json.dumps(CC._jsonable(agg), indent=1))
    print(f"[conv] {name}: wrote {len(seed_files)} seed results + aggregate in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
