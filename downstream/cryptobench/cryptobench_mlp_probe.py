"""CryptoBench: per-residue MLP probe on frozen representations (pLM-NN shape; PRIMARY probe).

What it does
------------
Step 3 of the CryptoBench pipeline (see README.md in this folder). The primary probe of
the camera-ready CryptoBench experiment (DECISIONS.md CB1, authors' answer: "use what we
used", i.e. the collaborator's pLM-NN-shaped MLP behind the rebuttal numbers); the context
conv1d head (`cryptobench_probe.py`) is the secondary probe.

  1. Inputs per residue (one feature set per run):
       --features tokens  : the codebook row of each token (`cryptobench_tokens_<tag>.npz` +
                            `_codebook.npy`): Ensembits level-0 code vector (the token
                            readout, primary), AminoAseed projected codebook, ESM3 raw
                            codebook, ProToken codebook, 3Di one-hot(20);
       --features aa      : amino-acid one-hot (20; any other residue -> all zeros), as the
                            collaborator's `aa_onehot`, columns in `--aa-order`;
       --features dense   : Ensembits z / q / q0 (`cryptobench_dense_<tag>.npz`,
                            `cryptobench_featurize.py`; secondary readouts, --readout), or
                            ESM-2 3B (`--dense esm2_3b --readout esm2`, descriptive row,
                            `cryptobench_esm2.py`; its results carry "role": "descriptive";
                            `--expect-layer` makes the gate check the recorded ESM-2 layer).
     Token / dense files must carry the port's provenance sidecar
     (`cryptobench_common.check_token_file`: whole-chain rule for baselines, checkpoint block
     for Ensembits).
  2. Folds: train = train-0..2, validation = train-3, test = test.
  3. Inputs standardised with the train mean / sd.
  4. Probe: Linear(D -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 256) -> ReLU ->
     Dropout(0.3) -> Linear(256 -> 2); cross-entropy with class weights [1, n_neg / n_pos];
     Adam lr 1e-4, weight decay 1e-5, batch 4,096 residues, <= 80 epochs, early stopping on
     the VALIDATION pooled AUROC (patience 15; `--selection`), best state restored; seeds 0-4.
  5. Metrics (written, never printed; CryptoBench numbers are sealed, DECISIONS.md CB8):
     per-entry AUROC (entry = apo id, chains concatenated; primary), per-entry AP, pooled
     AUROC, pooled AP (average precision) on test, and the same on validation.
  6. Outputs per seed: `<name>__seed<s>.json` + `<name>__seed<s>.preds.npz` (prediction
     sidecar of the selected epoch, val and test, items = entries; read by
     downstream/protein_level_stats.py); aggregate `<name>.json` (mean, sample sd over seeds,
     effective bits of token features on test). A run whose outputs all exist, verify and
     carry the current input fingerprint is skipped; otherwise the old outputs are moved to
     `_pre_<date>/` and the run is redone.

Options for the authors' TENTATIVE answers of 2026-10-22 (defaults = the tentative choice;
each is recorded in every result and enters the input fingerprint of the rows it affects, so a
change reruns only those rows):
  --aa-order {collaborator, alphabetical}   one-hot column order of the aa row (aa row only)
  --selection {val_pooled_auroc, val_per_entry_auroc}   epoch selection metric (all rows)
Input fingerprint (FIXED REAUDIT CB5 F1 / CB1 F3 / CB3 F2 / CB2 F10, 2026-10-22): structure
manifest sha256, sha256 of the input file, its provenance sidecar and codebook, sha256 of this
script, cryptobench_common.py and protein_preds.py, and every result-affecting CLI argument.

Paper
-----
Not in the paper yet (CryptoBench text after the lambda choice, DECISIONS.md CB9).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Benchmark: Škrhák et al., Bioinformatics 41(1):btae745, 2025 (pLM-NN head shape).

Origin
------
Ported from the collaborator's `ensembits-work/submission_exp/src/cryptobench_repr.py`
(`_build`, `_evaluate`, `run_probe`; sha256
3208d2bbe1383e10e52f4238a3269d8e8432ab720f1112c21077d99660d58d54; full path in
camera_ready/cryptobench_plan.md, read only). Probe architecture, optimiser, batch, class
weights, standardisation, epochs, patience, the strict ">" improvement rule, seeds and the
20-dim amino-acid one-hot are the source's.
Changes from the source (FIXED markers below; camera_ready/cryptobench_audit/PORT.md):
  * FIXED(CB audit B finding 8): AP (average_precision_score) instead of the trapezoid
    auc(recall, precision).
  * FIXED(DECISIONS.md CB6 / audit D Q3): per-ENTRY metrics (apo id, chains concatenated;
    was per chain), primary = per-entry AUROC.
  * FIXED(DECISIONS.md CB7 / audit D Q4): prediction sidecars per seed for the protein-level
    cluster bootstrap; sample sd (ddof 1; was ddof 0); no Welch test over seeds.
  * FIXED(DECISIONS.md CB8): no metric printed (sealed).
  * inputs from the port's caches (our structure stage keeps MSE / SEC / PYL; Ensembits via
    the camera-ready loader; baselines whole chain, label_seq index), not the source's own
    feature stage; provenance gate on every token file.
  * 2026-10-22: ESM-2 3B row (`plm_full`, recomputed embeddings) added as a descriptive dense
    row (authors' tentative answer); `plm_pca128` and the one-hot variants of AminoAseed /
    ESM3 (`aminoaseed_onehot`, `esm3struct_onehot`) and the readouts of the collaborator's
    ablation checkpoints are not ported (not camera-ready rows).
  * 2026-10-22: aa one-hot columns in the collaborator's order by default (REAUDIT CB6 F1;
    the 2026-10-21 port used the alphabetical one-letter order).
  A partial edit of this file by a killed session (2026-10-21, z / q / q0 modes) was
  superseded by this version (PORT.md).

Used by the reported model: yes (primary CryptoBench probe, DECISIONS.md CB1).

Inputs / outputs
----------------
Read  : `<manifest-dir>/<set>_manifest.json`, `<structs-root>/cryptobench_<set>/*.npz` (y, seq),
        `<tokens-dir>/cryptobench_tokens_<tag>.npz` (+ codebook, provenance) or
        `<tokens-dir>/cryptobench_dense_<tag>.npz` (+ provenance).
Write : `<out-dir>/mlp/<name>__seed<s>.json`, `.preds.npz`; `<out-dir>/mlp/<name>.json`.

How to run
----------
  $PY downstream/cryptobench/cryptobench_mlp_probe.py --run-tag <tag> --row ens_lam0p25_rv \\
      --features tokens --tokens ens_lam0p25_rv
  $PY downstream/cryptobench/cryptobench_mlp_probe.py --run-tag <tag> --row ens_lam0p25_rv \\
      --features dense --dense ens_lam0p25_rv --readout z
  $PY downstream/cryptobench/cryptobench_mlp_probe.py --run-tag <tag> --row aa --features aa
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402,F401
import _cb_paths as CBP  # noqa: E402
import cryptobench_common as CC  # noqa: E402

PROBE = "mlp_plmnn"


def build_mlp(in_dim, units=256, n_layers=2, dropout=0.3):
    """Source `_build`: [Linear -> ReLU -> Dropout] x n_layers -> Linear(., 2)."""
    layers, d = [], in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, units), nn.ReLU(), nn.Dropout(dropout)]
        d = units
    layers.append(nn.Linear(d, 2))
    return nn.Sequential(*layers)


def featurize(chains, features, codebook=None):
    """[(key, fold, X, y)] -> same with X (L, D) float32 (tokens -> codebook rows, aa ->
    one-hot 20 as the source's `_onehot(tokens, 20)`: index 20 (other) -> all zeros)."""
    out = []
    for key, fold, X, y in chains:
        if features == "tokens":
            F = codebook[X].astype(np.float32)
        elif features == "aa":
            F = np.zeros((len(X), 20), dtype=np.float32)
            ok = X < 20
            F[np.flatnonzero(ok), X[ok]] = 1.0
        else:
            F = X.astype(np.float32)
        out.append((key, fold, F, y))
    return out


def stack(chains, split):
    items = [(k, F, y) for k, f, F, y in chains if CC.split_of(f) == split]
    if not items:
        raise SystemExit(f"no chains in split {split}")
    X = np.concatenate([F for _, F, _ in items]).astype(np.float32)
    Y = np.concatenate([y for _, _, y in items]).astype(np.int64)
    bounds, pos = [], 0
    for k, F, _ in items:
        bounds.append((k, pos, pos + len(F))); pos += len(F)
    return X, Y, bounds


@torch.no_grad()
def scores_of(model, X, device, batch=65536):
    model.eval()
    s = np.zeros(len(X), dtype=np.float32)
    for a in range(0, len(X), batch):
        xb = torch.from_numpy(X[a:a + batch]).to(device)
        s[a:a + batch] = torch.softmax(model(xb).float(), -1)[:, 1].cpu().numpy()
    return s


def pooled_auroc(Y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(Y, s))


def per_chain(bounds, arr):
    return {k: arr[a:b] for k, a, b in bounds}


def split_metrics(bounds, Y, s):
    keys = [k for k, _, _ in bounds]
    lay = CC.entry_layout(keys)
    pc, yc = per_chain(bounds, s), per_chain(bounds, Y)
    return CC.entry_metrics(CC.entry_concat(pc, lay), CC.entry_concat(yc, lay)), lay, pc, yc


def val_selection_score(model, Xva, Yva, Bva, device, rule):
    """Selection metric on the validation fold: the source's pooled AUROC (default) or the
    per-entry AUROC (option for the authors' tentative answer)."""
    s = scores_of(model, Xva, device)
    if rule == "val_pooled_auroc":
        return pooled_auroc(Yva, s)
    m, *_ = split_metrics(Bva, Yva, s)
    return m["auroc_per_entry"]


def train_seed(data, seed, args, device):
    """One seed (source `run_probe` loop body): returns the selected model's val / test
    scores and the selection record."""
    (Xtr, Ytr, _), (Xva, Yva, Bva), (Xte, Yte, Bte) = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_mlp(Xtr.shape[1], args.units, args.n_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pw = float((len(Ytr) - Ytr.sum()) / max(Ytr.sum(), 1))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], dtype=torch.float32,
                                                    device=device))
    best_val, best_state, best_ep, bad = -1.0, None, -1, 0
    order = np.arange(len(Ytr))
    ep = -1
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        for a in range(0, len(order), args.batch_size):
            idx = order[a:a + args.batch_size]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(Ytr[idx]).to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # epoch selection on the VALIDATION fold (train-3) only, pooled AUROC (source rule)
        v = val_selection_score(model, Xva, Yva, Bva, device, args.selection)
        if v > best_val:
            best_val, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return {"val_scores": scores_of(model, Xva, device), "test_scores": scores_of(model, Xte, device),
            "best_epoch": best_ep, "epochs_run": ep + 1, "pos_weight": pw}


def out_name(args) -> str:
    readout = {"tokens": "token", "aa": "onehot", "dense": args.readout}[args.features]
    return f"{args.row}__{readout}"


def input_paths(args, paths):
    """(input file, codebook file or None) of a run."""
    if args.features == "tokens":
        return (paths.tokens_dir / f"cryptobench_tokens_{args.tokens}.npz",
                paths.tokens_dir / f"cryptobench_tokens_{args.tokens}_codebook.npy")
    if args.features == "dense":
        return paths.tokens_dir / f"cryptobench_dense_{args.dense}.npz", None
    return None, None


def fingerprint(args, paths) -> dict:
    """Input fingerprint of a run (see the module docstring)."""
    man = CBP.manifest_path(paths, args.set)
    src, cb = input_paths(args, paths)
    here = Path(__file__)
    drop = () if args.features == "aa" else ("aa_order",)
    return {"manifest_sha256": CC.file_sha256(man) if man.exists() else None,
            "input": CC.input_record(src) if src is not None and src.exists() else None,
            "codebook_sha256": CC.file_sha256(cb) if cb is not None and cb.exists() else None,
            "code": CC.code_sha256(here, CC.__file__,
                                   here.parents[1] / "protein_preds.py"),
            "args": CC.args_record(args, drop)}


def is_done(argv) -> bool:
    args = parser().parse_args(argv)
    paths = CBP.resolve_paths(args)
    return CC.probe_outputs_done(paths.out_dir / "mlp", out_name(args), args.seeds,
                                 fingerprint(args, paths))


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--row", required=True, help="row name (e.g. ens_lam0p25_rv, aminoaseed_wc, aa)")
    ap.add_argument("--features", choices=["tokens", "aa", "dense"], default="tokens")
    ap.add_argument("--tokens", default=None, help="cryptobench_tokens_<tag>.npz (--features tokens)")
    ap.add_argument("--dense", default=None, help="cryptobench_dense_<tag>.npz (--features dense)")
    ap.add_argument("--readout", choices=["z", "q", "q0", "esm2"], default=None)
    ap.add_argument("--aa-order", choices=sorted(CC.AA_ORDERS), default="collaborator",
                    help="aa row only: one-hot column order (tentative default: collaborator)")
    ap.add_argument("--selection", choices=["val_pooled_auroc", "val_per_entry_auroc"],
                    default="val_pooled_auroc",
                    help="epoch selection metric on the validation fold (tentative default: "
                         "pooled AUROC, the source rule)")
    ap.add_argument("--expect-layer", type=int, default=None,
                    help="pLM rows: the layer the dense file must record (gate)")
    ap.add_argument("--expect-checkpoint", default=None,
                    help="Ensembits rows: the row's checkpoint folder; the input file must be "
                         "written by it (gate)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--set", default="apo")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--units", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--redo", action="store_true", help="recompute even if the outputs verify")
    ap.add_argument("--allow-legacy-tokens", action="store_true",
                    help="skip the provenance gate (comparisons with old caches only)")
    CBP.add_path_args(ap)
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = parser()
    args = ap.parse_args(argv)
    if args.features == "tokens" and not args.tokens:
        ap.error("--tokens required with --features tokens")
    if args.features == "dense" and not (args.dense and args.readout):
        ap.error("--dense and --readout required with --features dense")
    paths = CBP.resolve_paths(args, create=("out_dir",))
    out_dir = paths.out_dir / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_name(args)
    fp = fingerprint(args, paths)
    if CC.probe_outputs_done(out_dir, name, args.seeds, fp) and not args.redo:
        print(f"[mlp] {name}: outputs exist, verify and match the inputs; skipped")
        return 0
    moved = CC.move_aside_probe_outputs(out_dir, name)
    if moved:
        print(f"[mlp] {name}: {len(moved)} old output files moved aside (stale or --redo)")
    set_dir = CBP.structs_dir(paths, args.set)
    man_path = CBP.manifest_path(paths, args.set)
    man = json.loads(man_path.read_text())
    device = args.device if torch.cuda.is_available() else "cpu"

    prov, codebook, src = None, None, None
    if args.features == "tokens":
        src = paths.tokens_dir / f"cryptobench_tokens_{args.tokens}.npz"
        prov = CC.check_token_file(src, man_path, args.allow_legacy_tokens,
                                   expect_checkpoint=args.expect_checkpoint)
        codebook = np.load(paths.tokens_dir / f"cryptobench_tokens_{args.tokens}_codebook.npy")
    elif args.features == "dense":
        src = paths.tokens_dir / f"cryptobench_dense_{args.dense}.npz"
        prov = CC.check_token_file(src, man_path, args.allow_legacy_tokens,
                                   expect_checkpoint=args.expect_checkpoint,
                                   expect_layer=args.expect_layer)
    # FIXED(FIXAUDIT FX4 F7, 2026-10-22): the ESM-2 row says in its results that it is
    #   descriptive (in no significance family).
    role = ("descriptive (no test; ESM-2 3B calibration row, in no significance family)"
            if args.readout == "esm2" else None)
    chains = CC.load_chains(man, set_dir, "tokens" if args.features == "tokens" else
                            args.features, src, args.readout, aa_order=args.aa_order)
    test_ids = [X for k, f, X, y in chains if f == CC.TEST_FOLD] if args.features == "tokens" else []
    chains = featurize(chains, args.features, codebook)
    tr, va, te = stack(chains, "train"), stack(chains, "val"), stack(chains, "test")
    # standardise on train statistics (source: "no-op in spirit for one-hots, harmless")
    mu = tr[0].mean(0, keepdims=True)
    sd = tr[0].std(0, keepdims=True) + 1e-6
    for A in (tr[0], va[0], te[0]):
        A -= mu
        A /= sd
    print(f"[mlp] {name}: D={tr[0].shape[1]} chains train/val/test="
          f"{len(tr[2])}/{len(va[2])}/{len(te[2])} seeds={args.seeds} (metrics not printed)",
          flush=True)

    rows, seed_files = [], []
    for s in range(args.seeds):
        r = train_seed((tr, va, te), s, args, device)
        mv, lay_v, pv, yv = split_metrics(va[2], va[1], r["val_scores"])
        mt, lay_t, pt, yt = split_metrics(te[2], te[1], r["test_scores"])
        f = out_dir / f"{name}__seed{s}.json"
        res = {"name": name, "row": args.row, "probe": PROBE, "set": args.set, "seed": s,
               "features": args.features, "readout": args.readout or args.features,
               "args": vars(args), "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
               "pos_weight": r["pos_weight"], "selection": f"validation (train-3) {args.selection}",
               "options": {"selection": args.selection,
                           "aa_order": args.aa_order if args.features == "aa" else None},
               "metrics": {"val": mv, "test": mt}, "input_provenance": prov,
               "input_fingerprint": fp, **({"role": role} if role else {})}
        CC.write_seed_result(f, preds={"val": pv, "test": pt}, labels={"val": yv, "test": yt},
                             layout={"val": lay_v, "test": lay_t},
                             meta={"row": args.row, "probe": PROBE, "seed": s, "set": args.set,
                                   "readout": res["readout"]}, result=res)
        rows.append(mt); seed_files.append(f.name)
        print(f"  seed {s}: done (best epoch {r['best_epoch']}, {r['epochs_run']} epochs)", flush=True)

    agg = {"name": name, "row": args.row, "probe": PROBE, "set": args.set,
           "features": args.features, "readout": args.readout or args.features,
           "test_agg": CC.agg_over_seeds(rows), "std": "sample (ddof=1) over probe seeds",
           "seed_files": seed_files, "input_provenance": prov,
           "effective_bits_test": CC.effective_bits(np.concatenate(test_ids)) if test_ids else None,
           "effective_bits_rule": "entropy of the token ids over the test residues (equals the "
                                  "collaborator's entropy of distinct q0 rows when the codebook "
                                  "rows are distinct)",
           "options": {"selection": args.selection,
                       "aa_order": args.aa_order if args.features == "aa" else None},
           "input_fingerprint": fp, **({"role": role} if role else {}), "args": vars(args)}
    (out_dir / f"{name}.json").write_text(json.dumps(CC._jsonable(agg), indent=1))
    print(f"[mlp] {name}: wrote {len(seed_files)} seed results + aggregate in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
