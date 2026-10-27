"""CryptoBench: ESM-2 3B per-residue embeddings of every chain (descriptive pLM row).

What it does
------------
Step 2d of the CryptoBench pipeline. Authors' TENTATIVE answer of 2026-10-22 (DECISIONS.md,
"after the port re-audit"; REAUDIT CB1 F2 / CB2 F2): an ESM-2 3B row, MLP probe only, in no
significance family, as the calibration against CryptoBench's own pLM-NN (ESM-2 3B + MLP).
The embeddings are RECOMPUTED on the port's sequences (structure stage of
`cryptobench_build_apo.py`: MSE -> M, modified residues mapped to their parent, polymer
residues only), never copied from the collaborator's cache (his structure stage drops MSE).

  sequence of the chain (npz `seq`) -> HF `facebook/esm2_t36_3B_UR50D` (`EsmModel`, fp16, eval)
  -> hidden state of layer `--layer` (default 33) without <cls> / <eos> -> (L, 2560); chains
  longer than 1,022 residues
  are embedded in non-overlapping chunks of 1,022 and concatenated (the source's rule) ->
  stored as float16 (as the source) in `cryptobench_dense_esm2_3b.npz`, key `<key>|esm2`.

The runner runs this row only with `--esm2-row on` (default, tentative); the MLP probe reads it
with `--features dense --dense esm2_3b --readout esm2`.

Layer (fix round 2, 2026-10-22; FIXAUDIT FX4 F1 major; coordinator's TENTATIVE choice, runner
option `--esm2-layer`, here `--layer`): upstream pLM-NN was trained on ESM-2 3B layer 33 (its
shipped example embedding 7w19A.npy matches our layer-33 hidden states, mean cosine 0.9987, vs
0.897 for the final layer 36), so the default is 33 = HF `hidden_states[33]` = fair-esm
`repr_layers=[33]` (output of block 33, before the final LayerNorm; values up to ~1,300, within
float16). `--layer 36` = HF `hidden_states[36]` = `last_hidden_state` (after the final
LayerNorm) = what the collaborator's script embedded. The layer is recorded in the sidecar and is
part of the fingerprint (`is_done`), so a change reruns this row and its probe only.

Paper
-----
Not in the paper yet (descriptive calibration row, appendix; DECISIONS.md CB9).
Benchmark: Škrhák et al., Bioinformatics 41(1):btae745, 2025 (pLM-NN = ESM-2 3B + MLP).

Origin
------
Ported from the collaborator's `ensembits-work/submission_exp/src/cryptobench_plmnn.py`
(`_load_esm2`, `_embed_one`, `run_embed`; sha256
12a5b51a399f847bdd0add450593655811ac38821680920151fbf80b683a0d62; full path in
camera_ready/cryptobench_plan.md, read only). Model, dtype, layer, chunking and storage dtype
are the source's EXCEPT the layer (the source took the final layer 36; default here 33, see
above). Changes: sequences from the port's structure stage (see above); one npz
with a provenance sidecar (manifest sha256, model id and snapshot, code sha256) instead of
one .npy per chain; the source cleared HF_HUB_OFFLINE to download, here the weights must be
in the local cache (HF_HUB_OFFLINE stays as set); failures recorded (exit 1).

Used by the reported model: no (reference row).

Inputs / outputs
----------------
Read  : `<structs-root>/cryptobench_<set>/*.npz` (seq), `<manifest-dir>/<set>_manifest.json`,
        the HF cache of facebook/esm2_t36_3B_UR50D.
Write : `<tokens-dir>/cryptobench_dense_esm2_3b.npz` (float16) + `.npz.provenance.json`.

How to run
----------
  CUDA_VISIBLE_DEVICES=0 $PY downstream/cryptobench/cryptobench_esm2.py --set apo --run-tag <tag>
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
import _cb_paths as CBP  # noqa: E402
import cryptobench_common as CC  # noqa: E402

ESM2_HF = "facebook/esm2_t36_3B_UR50D"
ESM2_DIM = 2560
ESM2_MAXLEN = 1022          # ESM-2 positional limit; longer chains are chunked (source rule)
TAG = "esm2_3b"
DEFAULT_LAYER = 33          # TENTATIVE (coordinator, 2026-10-22): upstream pLM-NN's layer
N_LAYERS = 36
SOURCE_SHA256 = "12a5b51a399f847bdd0add450593655811ac38821680920151fbf80b683a0d62"


def chunks(L: int, n: int = ESM2_MAXLEN) -> list:
    """Non-overlapping [start, end) windows of at most n residues covering [0, L)."""
    return [(c, min(c + n, L)) for c in range(0, L, n)]


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="apo")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                    help="hidden-state layer 1..36 (tentative default 33 = upstream pLM-NN; 36 = "
                         "the final, layer-normed state the collaborator used)")
    ap.add_argument("--redo", action="store_true")
    CBP.add_path_args(ap)
    return ap


def expected(args):
    paths = CBP.resolve_paths(args)
    man_path = CBP.manifest_path(paths, args.set)
    out = paths.tokens_dir / f"cryptobench_dense_{TAG}.npz"
    want = {"tokenizer": TAG, "model": ESM2_HF, "layer": int(args.layer),
            "manifest_sha256": CC.file_sha256(man_path) if man_path.exists() else None,
            "code_sha256": CC.file_sha256(__file__), "limit": args.limit, "n_failed": 0}
    return out, man_path, want


def is_done(argv) -> bool:
    args = parser().parse_args(argv)
    out, _, want = expected(args)
    side = out.with_name(out.name + ".provenance.json")
    if not (out.exists() and side.exists()):
        return False
    got = json.loads(side.read_text())
    return all(got.get(k) == v for k, v in want.items())


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    paths = CBP.resolve_paths(args, create=("tokens_dir",))
    out, man_path, want = expected(args)
    if not 1 <= args.layer <= N_LAYERS:
        raise SystemExit(f"--layer must be in 1..{N_LAYERS}")
    if out.exists():
        if is_done(argv) and not args.redo:
            print(f"[esm2] {out.name} up to date; skipped")
            return 0
        moved = CC.move_aside(out)
        # FIXED(FIXAUDIT FX3 F8, 2026-10-22): logged like the other writers.
        print(f"[esm2] moved the old {out.name} aside -> {moved[0].parent.name}/")
    import torch
    from transformers import AutoTokenizer, EsmModel
    device = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ESM2_HF)
    model = EsmModel.from_pretrained(ESM2_HF, torch_dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    snap = getattr(model.config, "_commit_hash", None)
    man = json.loads(man_path.read_text())
    entries = man["entries"][:args.limit] if args.limit else man["entries"]
    set_dir = CBP.structs_dir(paths, args.set)

    @torch.no_grad()
    def embed_one(seq: str) -> np.ndarray:
        enc = tok(seq, return_tensors="pt", add_special_tokens=True)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        # FIXED(FIXAUDIT FX4 F1): the chosen layer (HF hidden_states[k] = output of block k;
        #   hidden_states[36] = last_hidden_state, after the final LayerNorm).
        out_ = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        h = out_.hidden_states[args.layer]
        return h[0, 1:len(seq) + 1].float().cpu().numpy()   # drop <cls>, <eos>

    store, fails, n_chunked, t0 = {}, [], 0, time.time()
    for i, e in enumerate(entries, 1):
        key = f"{e['apo']}_{e['chain']}"
        try:
            seq = str(np.load(set_dir / f"{key}.npz", allow_pickle=True)["seq"])
            parts = [embed_one(seq[a:b]) for a, b in chunks(len(seq))]
            n_chunked += len(parts) > 1
            emb = np.concatenate(parts, axis=0)
            if emb.shape != (len(seq), ESM2_DIM) or not np.isfinite(emb).all():
                raise ValueError(f"embedding shape {emb.shape} / non-finite for L={len(seq)}")
            store[f"{key}|esm2"] = emb.astype(np.float16)
        except Exception as ex:                                  # noqa: BLE001
            fails.append((key, f"{type(ex).__name__}: {ex}"))
        if i % 100 == 0:
            print(f"  {i}/{len(entries)} ok={len(store)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(out, **store)
    out.with_name(out.name + ".provenance.json").write_text(json.dumps(
        {**want, "n_failed": len(fails), "failed": [k for k, _ in fails],
         "n_entries": len(store), "n_chunked": n_chunked, "chunk": ESM2_MAXLEN,
         "model_snapshot": snap, "model_dtype": "float16", "stored_dtype": "float16",
         "device": (torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"),
         "layer_rule": "HF hidden_states[layer]; 33 = upstream pLM-NN, 36 = last_hidden_state",
         "readouts": ["esm2"], "set": args.set, "checkpoint": None,
         "build_version": man.get("meta", {}).get("build_version"),
         "source": {"file": "cryptobench_plmnn.py", "sha256": SOURCE_SHA256},
         "written_by": "downstream/cryptobench/cryptobench_esm2.py"}, indent=1))
    print(f"wrote {out.name} ({len(store)} chains, {n_chunked} chunked)")
    if fails:
        print(f"FAILED {len(fails)}:")
        for k, m in fails[:10]:
            print("  ", k, m)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
