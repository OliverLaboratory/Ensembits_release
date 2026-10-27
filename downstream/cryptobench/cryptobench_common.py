"""CryptoBench: shared helpers of the camera-ready port (entries, metrics, sidecars, masks).

What it does
------------
Library used by the probes (`cryptobench_mlp_probe.py`, `cryptobench_probe.py`), the
tokenizers and the runner (`run_cryptobench_rv.py`); also a small CLI that writes the
non-cryptic-pocket residue mask of a built set.

  * Folds (CryptoBench release, identical in both audited implementations): train =
    train-0, train-1, train-2; validation = train-3 (selects the epoch); test = test.
  * Entry = apo PDB id (DECISIONS.md CB6). One CryptoBench entry can hold 2-3 chains
    (homomers); its residues are the chains of the entry concatenated in sorted chain-id
    order (as CryptoBench's own per-structure evaluation, which concatenates the chains of
    one apo structure). `entry_layout` / `entry_concat` implement that order; the residue
    index of a residue inside its entry (`residue_index` of the prediction sidecar) is its
    position in that concatenation.
  * Metrics (`entry_metrics`): per-entry AUROC (mean over entries with both classes;
    primary), per-entry AP, pooled AUROC and pooled AP (average precision, the step
    estimator, not the trapezoid). The probes write them to their result JSONs only; no
    script of this folder prints a metric value (CryptoBench numbers are sealed until the
    lambda choice, DECISIONS.md CB8).
  * Prediction sidecars (`write_seed_result`): one result JSON + one
    `downstream/protein_preds.py` sidecar (kind residue_binary, items = entries, val and
    test) per probe seed, read by `downstream/protein_level_stats.py`.
  * Gap-aware residue index of the baselines (`chain_residue_index`): label_seq_id (SEQRES
    position) minus its first value, so a chain with missing residues keeps its gaps
    (audit C findings 2 and 5).
  * `virtual_o`: carbonyl O placed from the peptide plane for residues whose O is not
    resolved (ProToken input; ProToken's encoder does not read O, audit C section 5).
  * `noncryptic_mask`: residues that line a NON-cryptic pocket of the entry
    (`noncryptic-pockets.json` of the release) and are labelled 0; dropped from the
    negatives in the pre-declared sensitivity analysis (DECISIONS.md CB6, audit D Q3).
  * `move_aside`: moves an existing output (and its sidecars) into a dated `_pre_<date>/`
    folder next to it instead of overwriting or deleting it.

Paper
-----
Not in the paper yet (CryptoBench text after the lambda choice, DECISIONS.md CB9).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.

Origin
------
New file written for this codebase (2026-10-21, step 5 of
camera_ready/cryptobench_audit/TRACKER.md). The per-structure evaluation follows the
collaborator's `cryptobench_repr.py` (`_evaluate`, sha256 3208d2bb...d58d54, see
camera_ready/cryptobench_plan.md) with the fixes decided in DECISIONS.md: unit = apo entry
(not chain), AP instead of the trapezoid AUPRC.

Used by the reported model: evaluation only.

Inputs / outputs
----------------
CLI `noncryptic-mask` reads `<structs-root>/cryptobench_<set>/*.npz` (y, nc) and
`<manifest-dir>/<set>_manifest.json`; writes `<manifest-dir>/<set>_noncryptic_mask.json`.

How to run
----------
  $PY downstream/cryptobench/cryptobench_common.py noncryptic-mask --set apo --run-tag <tag>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
for _p in (CODE_ROOT, CODE_ROOT / "downstream", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TRAIN_FOLDS = ("train-0", "train-1", "train-2")
VAL_FOLD = "train-3"
TEST_FOLD = "test"
SPLIT_NAME = "cryptobench_folds_train012_val3_test"   # recorded in every sidecar
TASK = "cryptobench"
MASK_SCHEMA = "cryptobench-residue-mask-v1"
# Rule of the baseline token files of the port (whole deposited chain, gap-aware residue
# index from label_seq_id; audit C findings 2 / 5). Written into their provenance sidecars;
# the probes refuse a baseline token file without it (the rebuttal files were cut at
# C-N > 5 A with an arange index).
BASELINE_CHAIN_RULE = "whole_chain_label_seq_v1"
STRUCT_BUILD_VERSION = "cb_build_v4_2026-10-22"


# ---------------------------------------------------------------- entries
def entry_of(key: str) -> str:
    """'<apo>_<chain>' -> '<apo>' (apo ids carry no underscore; chain ids may)."""
    return key.split("_", 1)[0]


def entry_layout(keys) -> dict:
    """{entry: [chain keys in sorted chain-id order]} (entries sorted)."""
    out: dict = {}
    for k in keys:
        out.setdefault(entry_of(k), []).append(k)
    return {e: sorted(v, key=lambda k: k.split("_", 1)[1]) for e, v in sorted(out.items())}


def entry_concat(per_chain: dict, layout: dict) -> dict:
    """{entry: concatenation of per_chain[key] over the entry's chains (layout order)}."""
    return {e: np.concatenate([np.asarray(per_chain[k]) for k in ks]) for e, ks in layout.items()}


# ---------------------------------------------------------------- metrics
def entry_metrics(scores: dict, labels: dict) -> dict:
    """Metrics of one split from per-ENTRY score / label arrays (see module docstring).

    Returns auroc_per_entry (mean over entries with both classes), ap_per_entry, auroc_pooled,
    ap_pooled, n_entries, n_entries_scored, n_res, n_pos. Never printed by the scripts."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    ids = sorted(scores)
    pe_roc, pe_ap = [], []
    for e in ids:
        y = np.asarray(labels[e]); s = np.asarray(scores[e], dtype=np.float64)
        if 0 < y.sum() < len(y):
            pe_roc.append(roc_auc_score(y, s)); pe_ap.append(average_precision_score(y, s))
    Y = np.concatenate([np.asarray(labels[e]) for e in ids])
    S = np.concatenate([np.asarray(scores[e], dtype=np.float64) for e in ids])
    pooled_ok = 0 < Y.sum() < len(Y)
    return {"auroc_per_entry": float(np.mean(pe_roc)) if pe_roc else float("nan"),
            "ap_per_entry": float(np.mean(pe_ap)) if pe_ap else float("nan"),
            "auroc_pooled": float(roc_auc_score(Y, S)) if pooled_ok else float("nan"),
            "ap_pooled": float(average_precision_score(Y, S)) if pooled_ok else float("nan"),
            "n_entries": len(ids), "n_entries_scored": len(pe_roc),
            "n_res": int(len(Y)), "n_pos": int(Y.sum())}


def residue_set_sha256(labels_by_split: dict) -> str:
    """sha256 over (split, entry, length, labels) of the evaluated splits: two runs are
    comparable only if they score the same residues with the same labels."""
    h = hashlib.sha256()
    for s in sorted(labels_by_split):
        for e in sorted(labels_by_split[s]):
            y = np.asarray(labels_by_split[s][e]).astype(np.uint8)
            h.update(f"{s}|{e}|{len(y)}|".encode()); h.update(y.tobytes())
    return h.hexdigest()


def effective_bits(token_ids) -> dict:
    """Shannon entropy (bits) of a token distribution and its perplexity (collaborator's
    `effective_bits.py` definition: entropy of the level-0 codeword over the test residues)."""
    counts = np.bincount(np.asarray(token_ids, dtype=np.int64))
    counts = counts[counts > 0]
    p = counts.astype(np.float64) / counts.sum()
    H = float(-(p * np.log2(p)).sum())
    return {"H_bits": H, "perplexity": 2.0 ** H, "n_tokens": int(counts.sum()),
            "n_distinct": int(len(counts))}


def agg_over_seeds(rows: list, keys=("auroc_per_entry", "ap_per_entry",
                                     "auroc_pooled", "ap_pooled")) -> dict:
    """{metric: [mean, sample sd (ddof 1; 0 with one seed)]} over seed results."""
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        out[k] = [float(np.mean(v)) if v else float("nan"),
                  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
    return out


# ---------------------------------------------------------------- sidecars
def write_seed_result(out_json, *, preds: dict, labels: dict, layout: dict, meta: dict,
                      result: dict) -> dict:
    """Write the result JSON of one probe seed and its prediction sidecar.

    preds / labels: {'val': {chain key: (L,) array}, 'test': {...}} of the selected epoch;
    layout: {'val': entry_layout(...), 'test': ...}. meta: sidecar facts (row, readout,
    probe, seed, set, ...); task / dataset / split / residue_set_sha256 are added here.
    Returns the result dict written (with 'per_protein_preds')."""
    import protein_preds as PP
    out_json = Path(out_json)
    splits, lab_e = {}, {}
    for s in ("val", "test"):
        pe = entry_concat(preds[s], layout[s])
        ye = entry_concat(labels[s], layout[s])
        lab_e[s] = ye
        ids = sorted(pe)
        splits[s] = PP.residue_split_from_items(
            ids, [np.asarray(pe[e], dtype=np.float32) for e in ids],
            [np.asarray(ye[e]).astype(np.uint8) for e in ids])
    rs = residue_set_sha256(lab_e)
    meta = {**meta, "task": TASK, "dataset": f"cryptobench_{meta.get('set', 'apo')}",
            "split": SPLIT_NAME, "residue_set_sha256": rs,
            "entry_rule": "apo id; chains concatenated in sorted chain-id order"}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rec = PP.write_preds(PP.sidecar_path(out_json), kind="residue_binary", splits=splits,
                         meta=meta, json_dir=out_json.parent)
    res = {**result, "residue_set_sha256": rs, PP.PREDS_KEY: rec}
    tmp = out_json.with_name(f".{out_json.name}.tmp")
    tmp.write_text(json.dumps(_jsonable(res), indent=1))
    tmp.replace(out_json)
    return res


def seed_result_ok(path) -> bool:
    """True if a seed result JSON exists and its sidecar verifies (used to skip reruns)."""
    import protein_preds as PP
    p = Path(path)
    if not p.exists():
        return False
    try:
        js = json.loads(p.read_text())
    except ValueError:
        return False
    return PP.preds_problem(js.get(PP.PREDS_KEY), p) is None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


# ---------------------------------------------------------------- structures
def chain_residue_index(label_seq) -> np.ndarray:
    """Gap-aware residue index of one chain: label_seq_id - label_seq_id[0] (0 at the first
    kept residue; a run of n missing residues advances it by n + 1). Raises ValueError if
    label_seq_id is missing or not strictly increasing (then no gap-aware index exists)."""
    ls = np.asarray(label_seq, dtype=np.int64).ravel()
    if ls.size == 0:
        raise ValueError("empty label_seq_id")
    if (ls < 1).any():
        raise ValueError("label_seq_id missing (stored as 0) for some residues")
    if (np.diff(ls) <= 0).any():
        raise ValueError("label_seq_id not strictly increasing")
    return ls - ls[0]


def virtual_o(n, ca, c) -> np.ndarray:
    """Carbonyl O of every residue from the peptide plane: O = C + 1.231 A along the bisector
    pointing away from CA(i) and N(i+1) (the standard placement); the last residue (no next
    N) uses its own N in place of N(i+1)."""
    n = np.asarray(n, np.float64); ca = np.asarray(ca, np.float64); c = np.asarray(c, np.float64)
    nn = np.vstack([n[1:], n[-1:]])
    u1 = ca - c; u1 /= np.linalg.norm(u1, axis=1, keepdims=True) + 1e-9
    u2 = nn - c; u2 /= np.linalg.norm(u2, axis=1, keepdims=True) + 1e-9
    d = -(u1 + u2); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return (c + 1.231 * d).astype(np.float32)


def backbone4(d) -> tuple[np.ndarray, int]:
    """(L, 4, 3) N/CA/C/O from a structure npz (real O; a missing O placed by `virtual_o`)
    and the number of placed O atoms."""
    bb3 = np.asarray(d["bb"][0], dtype=np.float32)
    L = bb3.shape[0]
    bb4 = np.zeros((L, 4, 3), dtype=np.float32)
    bb4[:, :3] = bb3
    o = np.asarray(d["o"], dtype=np.float32) if "o" in d.files else np.full((L, 3), np.nan,
                                                                            np.float32)
    miss = ~np.isfinite(o).all(axis=1)
    if miss.any():
        o = o.copy()
        o[miss] = virtual_o(bb3[:, 0], bb3[:, 1], bb3[:, 2])[miss]
    bb4[:, 3] = o
    return bb4, int(miss.sum())


def mask_code_sha256() -> str:
    """sha256 of the source of `noncryptic_mask` (the mask's code fingerprint; FIXLIST2 item 4)."""
    import inspect
    return hashlib.sha256(inspect.getsource(noncryptic_mask).encode()).hexdigest()


def noncryptic_mask(manifest: dict, set_dir, folds=(VAL_FOLD, TEST_FOLD)) -> dict:
    """{entry: {'n_res', 'drop'}}: entry residue indices (layout order) that line a
    non-cryptic pocket (`nc` flag of the structure npz) and are labelled 0."""
    set_dir = Path(set_dir)
    keys = [f"{e['apo']}_{e['chain']}" for e in manifest["entries"] if e["fold"] in folds]
    out = {}
    for ent, ks in entry_layout(keys).items():
        drop, pos = [], 0
        for k in ks:
            d = np.load(set_dir / f"{k}.npz", allow_pickle=True)
            if "nc" not in d.files:
                raise ValueError(f"{k}: structure cache without the 'nc' flag (built before "
                                 f"{STRUCT_BUILD_VERSION}); rebuild with cryptobench_build_apo.py")
            y = np.asarray(d["y"]); nc = np.asarray(d["nc"])
            drop.extend((pos + np.flatnonzero((nc > 0) & (y == 0))).tolist())
            pos += len(y)
        out[ent] = {"n_res": int(pos), "drop": [int(i) for i in drop]}
    return out


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- probe inputs
# Amino-acid order of the one-hot rows. FIXED(REAUDIT CB6 F1; authors' TENTATIVE answer
#   2026-10-22): the collaborator's order (AA3 list of his cryptobench_probe.py: ALA, ARG, ASN,
#   ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL) is the
#   default; "alphabetical" (one-letter A, C, D, ...) is the 2026-10-21 port order. The probes
#   take it as --aa-order (part of the aa rows' fingerprint only).
AA_ORDERS = {"collaborator": "ARNDCQEGHILKMFPSTWYV", "alphabetical": "ACDEFGHIKLMNPQRSTVWY"}
AAS = AA_ORDERS["alphabetical"]           # kept for old callers; the probes use AA_ORDERS
BASELINE_TOKENIZERS = ("aminoaseed", "esm3struct", "protoken", "mini3di")
PLM_KINDS = ("esm2_3b",)                  # descriptive pLM row (authors 2026-10-22)
PLM_MODELS = {"esm2_3b": "facebook/esm2_t36_3B_UR50D"}


def aa_ids(seq: str, order: str = "collaborator") -> np.ndarray:
    """Residue letters -> indices in the chosen order; anything else -> 20 ('unknown')."""
    alpha = AA_ORDERS[order]
    return np.fromiter((alpha.find(c) if alpha.find(c) >= 0 else 20 for c in seq),
                       dtype=np.int64, count=len(seq))


class TokenFileError(RuntimeError):
    """A token / feature file cannot be used by the probes of the port."""


def check_token_file(npz_path, manifest_path=None, allow_legacy: bool = False,
                     expect_checkpoint=None, expect_layer=None) -> dict:
    """Provenance of a token or dense-feature file; refuse what the port must not use.

    * no `<file>.provenance.json`: refused (rebuttal-era caches have none);
    * every file must name the structure manifest it was built from (sha256) and, if
      `manifest_path` is given, it must be that one (FIXED REAUDIT CB1 F3 / CB3 F2 / CB5 F2,
      2026-10-22: was baselines only);
    * a baseline file (sidecar 'tokenizer' in BASELINE_TOKENIZERS) must state
      chain_rule == BASELINE_CHAIN_RULE; a pLM file ('tokenizer' in PLM_KINDS) is accepted
      with its manifest;
    * an Ensembits file must name its checkpoint (variant + best.pt sha256); with
      `expect_checkpoint` (the row's checkpoint folder) variant and best.pt sha256 must be
      that checkpoint's, and its dihedral convention / break cutoff its config's;
    * a dense file must have zero level-0 mismatches (`n_token_mismatch`,
      `n_tokenfile_mismatch`; FIXED REAUDIT CB1 F4 / CB2 F1 / CB5 F3);
    * any failed chain refuses the file.
    Fix round 2 (2026-10-22, FIXAUDIT FX2 F1 / FX4 F6): with `expect_checkpoint` a baseline or
    pLM file is refused (an Ensembits row must read an Ensembits file); a pLM file must name
    its model (PLM_MODELS) and an integer `layer` (equal to `expect_layer` when given).
    allow_legacy skips the checks (reading old caches for comparisons only)."""
    p = Path(npz_path)
    side = p.with_name(p.name + ".provenance.json")
    if not p.exists():
        raise TokenFileError(f"{p} missing")
    if allow_legacy:
        return json.loads(side.read_text()) if side.exists() else {}
    if not side.exists():
        raise TokenFileError(f"{p.name}: no provenance sidecar (rebuttal-era cache?); rebuild "
                             f"with the port's tokenizers")
    prov = json.loads(side.read_text())
    if manifest_path is not None:
        if not prov.get("manifest_sha256"):
            raise TokenFileError(f"{p.name}: sidecar names no structure manifest (rebuild)")
        if prov["manifest_sha256"] != file_sha256(manifest_path):
            raise TokenFileError(f"{p.name}: built from another structure manifest")
    tk = prov.get("tokenizer")
    if expect_checkpoint is not None and (tk in BASELINE_TOKENIZERS or tk in PLM_KINDS):
        raise TokenFileError(f"{p.name}: a {tk} file, but the row expects an Ensembits file of "
                             f"{expect_checkpoint}")
    if tk in BASELINE_TOKENIZERS:
        if prov.get("chain_rule") != BASELINE_CHAIN_RULE:
            raise TokenFileError(f"{p.name}: chain rule {prov.get('chain_rule')!r}, the port "
                                 f"needs {BASELINE_CHAIN_RULE!r} (whole chain, label_seq index)")
    elif tk in PLM_KINDS:
        if prov.get("model") != PLM_MODELS[tk]:
            raise TokenFileError(f"{p.name}: model {prov.get('model')!r}, expected {PLM_MODELS[tk]!r}")
        if not isinstance(prov.get("layer"), int):
            raise TokenFileError(f"{p.name}: pLM file without an integer `layer`")
        if expect_layer is not None and prov["layer"] != int(expect_layer):
            raise TokenFileError(f"{p.name}: layer {prov['layer']}, the row expects {expect_layer}")
        if prov.get("n_failed") is None:
            raise TokenFileError(f"{p.name}: pLM file without a failure count")
    else:
        ck = prov.get("checkpoint") or {}
        if not (ck.get("variant") and ck.get("best_pt_sha256")):
            raise TokenFileError(f"{p.name}: neither a whole-chain baseline, a pLM file nor an "
                                 f"Ensembits file with a checkpoint block")
        if expect_checkpoint is not None:
            from ensembits.featurize import checkpoint_provenance
            from ensembits.paths import REPO_ROOT
            ck_dir = Path(expect_checkpoint)
            if not ck_dir.is_absolute() and not ck_dir.exists():
                ck_dir = REPO_ROOT / ck_dir            # same rule as _cb_paths.repo_path
            if not (ck_dir / "best.pt").exists():
                raise TokenFileError(f"expected checkpoint {expect_checkpoint}: no best.pt")
            expect_checkpoint = ck_dir
            want = checkpoint_provenance(ck_dir)
            if (ck.get("variant"), ck.get("best_pt_sha256")) != (want["variant"],
                                                                 want["best_pt_sha256"]):
                raise TokenFileError(f"{p.name}: written by {ck.get('variant')}, the row needs "
                                     f"{want['variant']} (or another best.pt)")
            cfg = json.loads((Path(expect_checkpoint) / "config.json").read_text())
            conv = cfg.get("dihedral_convention", "legacy_plus180")
            cut = cfg.get("chain_break_cutoff", "auto")
            if (prov.get("dihedral_convention"), prov.get("chain_break_cutoff")) != (conv, cut):
                raise TokenFileError(f"{p.name}: descriptor convention / cutoff differ from "
                                     f"the checkpoint's config")
        if "readouts" in prov and prov.get("n_tokenfile_mismatch") is None:
            raise TokenFileError(f"{p.name}: Ensembits dense file without the token-file "
                                 f"cross-check")
    for k in ("n_token_mismatch", "n_tokenfile_mismatch"):
        if prov.get(k):
            raise TokenFileError(f"{p.name}: {k} = {prov[k]} (level-0 tokens disagree)")
    if prov.get("n_failed"):
        raise TokenFileError(f"{p.name}: {prov['n_failed']} chains failed to tokenize")
    return prov


# ---------------------------------------------------------------- fingerprints
def code_sha256(*files) -> dict:
    """{file name: sha256} of code files (default: the caller passes its own path)."""
    return {Path(f).name: file_sha256(f) for f in files}


def input_record(path) -> dict:
    """sha256 of an input file and of its provenance sidecar (if any)."""
    p = Path(path)
    side = p.with_name(p.name + ".provenance.json")
    return {"file": p.name, "sha256": file_sha256(p),
            "provenance_sha256": file_sha256(side) if side.exists() else None}


# CLI arguments that do not change a probe's result (locations, device, control flags)
VOLATILE_ARGS = {"device", "redo", "run_tag", "legacy_layout", "cryptobench_dir", "structs_root",
                 "manifest_dir", "tokens_dir", "out_dir", "expect_checkpoint", "expect_layer"}


def args_record(args, drop=()) -> dict:
    """The result-affecting CLI arguments (VOLATILE_ARGS and `drop` removed)."""
    return {k: v for k, v in sorted(vars(args).items())
            if k not in VOLATILE_ARGS and k not in set(drop)}


def seed_files(out_dir, name) -> dict:
    """{seed: path} of the `<name>__seed<s>.json` files present."""
    out = {}
    for p in Path(out_dir).glob(f"{name}__seed*.json"):
        tail = p.stem.rsplit("__seed", 1)[-1]
        if tail.isdigit():
            out[int(tail)] = p
    return out


def probe_outputs_done(out_dir, name, seeds: int, fingerprint: dict) -> bool:
    """True iff the aggregate and exactly the seeds 0..seeds-1 exist, every seed sidecar
    verifies and every file carries `fingerprint` (FIXED REAUDIT CB5 F1, CB4 F1)."""
    out_dir = Path(out_dir)
    agg = out_dir / f"{name}.json"
    sf = seed_files(out_dir, name)
    if not agg.exists() or sorted(sf) != list(range(seeds)):
        return False
    try:
        if json.loads(agg.read_text()).get("input_fingerprint") != fingerprint:
            return False
        for p in sf.values():
            if not seed_result_ok(p):
                return False
            if json.loads(p.read_text()).get("input_fingerprint") != fingerprint:
                return False
    except ValueError:
        return False
    return True


def move_aside_probe_outputs(out_dir, name) -> list:
    """Move the aggregate and every seed result (+ sidecars) of `name` to _pre_<date>/."""
    out_dir = Path(out_dir)
    moved = []
    for p in [out_dir / f"{name}.json", *seed_files(out_dir, name).values()]:
        if p.exists():
            moved += move_aside(p)
    return moved


# ---------------------------------------------------------------- probe inputs
def load_chains(manifest: dict, set_dir, features: str, src=None, readout: str | None = None,
                max_len: int | None = None, aa_order: str = "collaborator") -> list:
    """Per-chain probe inputs: [(key, fold, X, y)] in manifest order.

    features: 'tokens' (src = (token npz, codebook): X = ids (L,) int64, embedded by the
    probe), 'aa' (X = amino-acid ids in `aa_order` (AA_ORDERS) -> 0..19, anything else -> 20),
    'dense' (src = dense npz of cryptobench_featurize.py, readout z / q / q0: X (L, 128)
    float32). A chain missing from the token / dense file raises (no silent drop).
    max_len truncates TRAIN chains only (val / test must stay whole for the sidecars)."""
    set_dir = Path(set_dir)
    tok = np.load(src) if features in ("tokens", "dense") else None
    out = []
    for e in manifest["entries"]:
        key = f"{e['apo']}_{e['chain']}"
        d = np.load(set_dir / f"{key}.npz", allow_pickle=True)
        y = np.asarray(d["y"]).astype(np.int64)
        if features == "aa":
            X = aa_ids(str(d["seq"]), aa_order)
        elif features == "tokens":
            if key not in tok:
                raise TokenFileError(f"{key} missing from {Path(src).name}")
            X = tok[key].astype(np.int64)
        elif features == "dense":
            k = f"{key}|{readout}"
            if k not in tok:
                raise TokenFileError(f"{k} missing from {Path(src).name}")
            X = tok[k].astype(np.float32)
        else:
            raise ValueError(features)
        if len(X) != len(y):
            raise TokenFileError(f"{key}: {len(X)} inputs for {len(y)} labels")
        if max_len and len(y) > max_len and e["fold"] in TRAIN_FOLDS:
            X, y = X[:max_len], y[:max_len]
        out.append((key, e["fold"], X, y))
    return out


def split_of(fold: str) -> str | None:
    return ("train" if fold in TRAIN_FOLDS else "val" if fold == VAL_FOLD
            else "test" if fold == TEST_FOLD else None)


# ---------------------------------------------------------------- outputs
def sidecars_of(path) -> list:
    """The files that belong to an output: itself, <file>.provenance.json, <stem>_codebook.npy
    (token files) and <stem>.preds.npz (result JSONs)."""
    p = Path(path)
    out = [p, p.with_name(p.name + ".provenance.json")]
    if p.suffix == ".npz":
        out.append(p.with_name(p.stem + "_codebook.npy"))
    if p.suffix == ".json":
        out.append(p.with_name(p.stem + ".preds.npz"))
    return [q for q in out if q.exists()]


def move_aside_tree(folder, date: str | None = None) -> list:
    """Move a whole folder into <parent>/_pre_<YYYYMMDD>/<name>[.k] (never delete)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    date = date or _dt.date.today().strftime("%Y%m%d")
    dest_dir = folder.parent / f"_pre_{date}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest, k = dest_dir / folder.name, 1
    while dest.exists():
        dest = dest_dir / f"{folder.name}.{k}"; k += 1
    shutil.move(str(folder), str(dest))
    return [dest]


def move_aside(path, date: str | None = None) -> list:
    """Move an existing output and its sidecars into <dir>/_pre_<YYYYMMDD>/ (never delete,
    never overwrite there: a numbered suffix is added if the name is taken). Returns the
    new paths."""
    date = date or _dt.date.today().strftime("%Y%m%d")
    moved = []
    for q in sidecars_of(path):
        dest_dir = q.parent / f"_pre_{date}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / q.name
        k = 1
        while dest.exists():
            dest = dest_dir / f"{q.name}.{k}"; k += 1
        shutil.move(str(q), str(dest))
        moved.append(dest)
    return moved


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    import _cb_paths as CBP
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("noncryptic-mask", help="write <set>_noncryptic_mask.json")
    m.add_argument("--set", default="apo")
    CBP.add_path_args(m)
    args = ap.parse_args(argv)
    paths = CBP.resolve_paths(args)
    man = json.loads(CBP.manifest_path(paths, args.set).read_text())
    mask = noncryptic_mask(man, CBP.structs_dir(paths, args.set))
    out = paths.manifest_dir / f"{args.set}_noncryptic_mask.json"
    if out.exists():
        move_aside(out)
    out.write_text(json.dumps({"schema": MASK_SCHEMA, "set": args.set,
                               "manifest_sha256": file_sha256(CBP.manifest_path(paths, args.set)),
                               "mask_code_sha256": mask_code_sha256(),
                               "rule": "residues lining a non-cryptic pocket of the entry "
                                       "(noncryptic-pockets.json, apo side) and labelled 0",
                               "entries": mask}, indent=0))
    print(f"[mask] {out.name}: {len(mask)} entries, "
          f"{sum(len(v['drop']) for v in mask.values())} residues to drop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
