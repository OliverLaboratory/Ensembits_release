"""CryptoBench: tokenize every chain with the static baselines (AminoAseed, ESM3-struct, ProToken, 3Di).

What it does
------------
Step 2b of the CryptoBench pipeline (see `README.md` in this folder). Covers the
baseline alphabets of the paper's tables that apply to a single structure
(Vote-3Di and ProtProfileMD pool several frames and have no P = 1 definition).

Every CryptoBench example is ONE deposited chain, so each baseline encodes the chain
WHOLE (no cut at missing-residue gaps) with a gap-aware residue index (the camera-ready
rule of `data_prep/baselines/README.md`, "true chains" option A; StructTokenBench usage).
The residue index is `label_seq_id - label_seq_id[0]` of the chain
(`cryptobench_common.chain_residue_index`; stored by `cryptobench_build_apo.py`).

  aminoaseed  camera-ready `data_prep/baselines/build_aminoaseed_tokens.tokenize_protein`
              (`pieces=[(0, L)]`, `residue_index=`); codebook = PROJECTED
              linear_proj(codebook.weight) (512 x 1024), the space the quantizer assigns in
  esm3struct  camera-ready `data_prep/baselines/build_esm3struct_tokens.tokenize_protein`
              (`pieces=[(0, L)]`, `residue_index=` -> `encode(..., residue_index=)`);
              codebook = the encoder's raw 4096 x 128 table (no projection in ESM3)
  protoken    camera-ready `data_prep/baselines/build_protoken_tokens.tokenize_by_chain`
              (`pieces=[(0, L)]`, PDB residue numbers = residue_index + 1) with the
              camera-ready wrapper `ensembits.baselines.protoken.ProToken`; real N / CA / C / O
              (an unresolved O placed from the peptide plane, `cryptobench_common.virtual_o`);
              codebook 512 x 32
  mini3di     analytic encoder on the whole chain; real C-beta, NaN (Gly / missing) ->
              mini3di's own approximation (`ensembits.baselines._real_or_ideal_bb.
              cb_from_bb_or_real`); masked terminal residues -> the invalid state 2;
              embedded as a full-rank one-hot(20) table instead of the (20, 2) centroids
Outputs one token npz + codebook + `<npz>.provenance.json` (chain rule
`whole_chain_label_seq_v1`, builder, structure manifest sha256) per tokenizer. The probes
refuse a baseline token file without that sidecar (the rebuttal caches were cut at
C-N > 5 A with an arange index).

GPU: the job's GPU is chosen with CUDA_VISIBLE_DEVICES (the runner sets it); `--device`
is only for a manual run (it then sets CUDA_VISIBLE_DEVICES, as before). Run all rows on
the same device type (ProToken GPU vs CPU tokens differ on ~2 %, audit C finding 9).

Paper
-----
Not in the paper yet: CryptoBench goes into the main text
(`camera_ready/require_fix.md` #24, `camera_ready/code_fix.md` E3).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Baselines as in `appendix.tex` \\label{app:baselines}.
Benchmark: Škrhák et al., Bioinformatics 41(1):btae745, 2025.

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/cryptobench_tokenize_baselines.py`.
Changes before 2026-10-21: ROOT from `ensembits.paths.REPO_ROOT`; locations from
`_cb_paths.py`; `--protoken-dir`; the literal Hugging Face token of the original removed
(export HF_TOKEN yourself if the hub is needed; HF_HUB_OFFLINE=1 is kept).
FIXED(CB audit C findings 2 / 5 / 6 / 9, 2026-10-21; port step 5, see
  camera_ready/cryptobench_audit/PORT.md):
  * old: the research-repo builders (`submission_exp/src/build_*_tokens.py`,
    `_real_or_ideal_bb`, `Ensembits/utils/baselines`), which cut every chain at
    C-N > 5.0 A and gave AminoAseed / ESM3 an arange index per piece and ProToken numbers
    1..L with O at the origin. New: the camera-ready builders, whole chain, gap-aware
    residue index from label_seq_id, real O (audit C findings 2, 5, 9). An index from
    auth_seq_id (tried in a killed session) was rejected: auth numbers have bonded jumps
    and non-monotone steps (audit C section 2).
  * ESM3-struct folded in here (`--tokenizer esm3struct`; `cryptobench_tokenize_esm3struct.py`
    is a thin wrapper); its codebook is read from the loaded encoder instead of the MISATO
    cache.
  * mini3di: masked terminal residues -> state 2 (audit C finding 6, earlier today).
  * default tags carry `_wc` (whole chain) so no rebuttal file name is reused; an existing
    output of the same tag is skipped if its sidecar matches the current rule and
    structure manifest, else moved to `_pre_<date>/` (never deleted, never overwritten).

Fixes 2026-10-22 (port re-audit, FIXLIST_20261022.md items 1 / 7):
  * the sidecar records the sha256 of this script and of cryptobench_common.py; `is_done`
    (used by the runner) compares tokenizer, chain rule, structure manifest, code and limit;
  * ProToken refuses to run without a visible GPU (all ProToken rows on one device type,
    GPU vs CPU tokens differ by ~2 %, REAUDIT CB3 F4) unless --allow-cpu; the GPU name is
    recorded; XLA_PYTHON_CLIENT_PREALLOCATE=false (REAUDIT CB5 F6).

Fix round 2 (2026-10-22, FIXLIST2 item 1, FIXAUDIT FX1 F2 / FX1 F4):
  * the fingerprint also holds the sha256 of the builder code each tokenizer calls
    (`BUILDER_FILES`: data_prep/baselines/build_*_tokens.py, _chain_pieces.py,
    ensembits/baselines/{protoken,_real_or_ideal_bb}.py) and of the weights
    (`weight_files`: AminoAseed checkpoint, ESM3 structure-encoder weights of the snapshot esm
    loads, ProToken models/ + codebook, the mini3di package files);
  * ProToken also records JAX's device platform and refuses a non-GPU JAX unless --allow-cpu.

Used by the reported model: no (baseline rows of the CryptoBench table).
# Not added (code_fix C13, authors' decision 2026-10-15: released checkpoints only): the
#   same-data baselines `aminoaseed_ourdata`, `esm3style_ourdata`.

Inputs / outputs
----------------
Read  : `<structs-root>/cryptobench_<set>/<apo>_<chain>.npz` (bb, seq, cb, o, label_seq),
        `<manifest-dir>/<set>_manifest.json`; tokenizer weights (AminoAseed checkpoint via
        the builder, ESM3 open weights from the HF cache, ProToken in `--protoken-dir`).
Write : `<tokens-dir>/cryptobench_tokens_<tag>.npz`, `_codebook.npy`, `.npz.provenance.json`;
        <tag> = `--tag` or "<tokenizer>_wc_<set>".

Dependencies left in the research repo: none (data_prep/baselines/ builders; StructTokenBench
and ProToken sources through `ensembits.paths`; packages `mini3di`, `omegaconf`, `esm`,
tensorflow + jax for ProToken).

How to run
----------
  $PY downstream/cryptobench/cryptobench_tokenize_baselines.py --tokenizer aminoaseed --set apo
  $PY downstream/cryptobench/cryptobench_tokenize_baselines.py --tokenizer esm3struct --set apo
  $PY downstream/cryptobench/cryptobench_tokenize_baselines.py --tokenizer protoken   --set apo
  $PY downstream/cryptobench/cryptobench_tokenize_baselines.py --tokenizer mini3di    --set apo
(with CUDA_VISIBLE_DEVICES=<gpu> for the first three)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Original: os.environ.setdefault("HF_TOKEN", "<literal token>") -- removed (secret).
if "HF_TOKEN" in os.environ:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # ProToken (JAX) on shared GPUs

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
from ensembits.paths import REPO_ROOT as ROOT, PROTOKEN_SRC  # noqa: E402,F401
import _cb_paths as CBP  # noqa: E402
import cryptobench_common as CC  # noqa: E402

# Camera-ready builders (2026-10-21; were the research-repo copies).
BASELINES_DIR = CODE_ROOT / "data_prep" / "baselines"
sys.path.insert(0, str(BASELINES_DIR))

TOKENIZERS = ("aminoaseed", "esm3struct", "protoken", "mini3di")
_BL = CODE_ROOT / "data_prep" / "baselines"
_EB = CODE_ROOT / "ensembits" / "baselines"
BUILDER_FILES = {"aminoaseed": [_BL / "build_aminoaseed_tokens.py", _BL / "_chain_pieces.py"],
                 "esm3struct": [_BL / "build_esm3struct_tokens.py", _BL / "_chain_pieces.py"],
                 "protoken": [_BL / "build_protoken_tokens.py", _BL / "_chain_pieces.py",
                              _EB / "protoken.py"],
                 "mini3di": [_EB / "_real_or_ideal_bb.py"]}
AMINOASEED_CKPT = ROOT / ("data/struct_tokenizer_ckpts/codebook_512x1024-1e+19-linear-fixed-last.ckpt/"
                          "checkpoint/mp_rank_00_model_states.pt")   # = build_aminoaseed_tokens.CKPT


BUILDERS = {"aminoaseed": "data_prep/baselines/build_aminoaseed_tokens.tokenize_protein",
            "esm3struct": "data_prep/baselines/build_esm3struct_tokens.tokenize_protein",
            "protoken": "data_prep/baselines/build_protoken_tokens.tokenize_by_chain "
                        "+ ensembits.baselines.protoken.ProToken",
            "mini3di": "mini3di.Encoder.encode_atoms (whole chain) + "
                       "ensembits.baselines._real_or_ideal_bb.cb_from_bb_or_real"}


def weight_files(name: str, protoken_dir=None) -> list:
    """[(key, path)] of the weight / parameter files a tokenizer loads (hashed into its
    fingerprint); key = path relative to the tokenizer's weight folder."""
    if name == "aminoaseed":
        return [(AMINOASEED_CKPT.name, AMINOASEED_CKPT)]
    if name == "esm3struct":
        from esm.utils.constants.esm3 import data_root
        f = Path(data_root("esm3")) / "data/weights/esm3_structure_encoder_v0.pth"
        return [(f.name, f)]
    if name == "protoken":
        d = Path(CBP.repo_path(protoken_dir or PROTOKEN_SRC))
        files = [d / "ProToken_Code_Book.pkl",
                 *sorted(p for p in (d / "models").rglob("*") if p.is_file())]
        return [(str(f.relative_to(d)), f) for f in files]
    if name == "mini3di":
        import mini3di
        d = Path(mini3di.__file__).parent
        return [(f.name, f) for f in sorted(d.iterdir())
                if f.is_file() and f.suffix in (".py", ".kerasify")]
    raise ValueError(name)


def weights_sha256(name: str, protoken_dir=None) -> dict:
    """{key: sha256} of `weight_files` (sha cached per path / size / mtime in the process)."""
    from ensembits.featurize import file_sha256 as cached_sha
    out = {}
    for key, f in weight_files(name, protoken_dir):
        if not Path(f).exists():
            raise SystemExit(f"{name}: weight file {f} missing")
        out[key] = cached_sha(f)
    return out


def make_encoder(name: str, protoken_dir: str | None = None, allow_cpu: bool = False):
    """(encode(bb4, seq, cbeta, residue_index) -> (L,) int64, codebook (M, D) float32, info)."""
    if name == "aminoaseed":
        import torch
        from build_aminoaseed_tokens import build_aminoaseed_model, tokenize_protein, CKPT
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = build_aminoaseed_model(dev)
        with torch.no_grad():
            cb = model.quantizer.linear_proj(model.quantizer.codebook.weight)
        codebook = cb.detach().float().cpu().numpy()

        def encode(bb4, seq, cbeta, ri):
            return tokenize_protein(model, bb4[:, 1, :], dev, bb=bb4,
                                    pieces=[(0, bb4.shape[0])], residue_index=ri)
        return encode, codebook, {"device": dev, "checkpoint": str(Path(CKPT).relative_to(ROOT))
                                  if str(CKPT).startswith(str(ROOT)) else Path(CKPT).name,
                                  "codebook": "projected linear_proj(codebook.weight)"}
    if name == "esm3struct":
        import torch
        from build_esm3struct_tokens import tokenize_protein
        from esm.pretrained import ESM3_structure_encoder_v0
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        enc = ESM3_structure_encoder_v0(dev)
        enc.eval()
        codebook = enc.codebook.embeddings.detach().float().cpu().numpy()   # raw 4096 x 128

        def encode(bb4, seq, cbeta, ri):
            return tokenize_protein(enc, bb4[:, 1, :], dev, bb=bb4,
                                    pieces=[(0, bb4.shape[0])], residue_index=ri)
        return encode, codebook, {"device": dev, "weights": "esm3-sm-open-v1 structure encoder",
                                  "codebook": "raw encoder codebook"}
    if name == "protoken":
        from build_protoken_tokens import tokenize_by_chain
        from ensembits.baselines.protoken import ProToken
        tok = ProToken(repo_path=str(CBP.repo_path(protoken_dir or PROTOKEN_SRC)), device="")
        codebook = tok.codebook.astype(np.float32)
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        # FIXED(REAUDIT CB3 F4 / FIXLIST item 7, 2026-10-22): one device type for every
        #   ProToken row (GPU); a CPU run is refused unless --allow-cpu.
        if not gpus and not allow_cpu:
            raise SystemExit("ProToken: no GPU visible to TensorFlow; refusing a CPU run "
                             "(tokens differ ~2 % from GPU); pass --allow-cpu to override")
        gpu_name = gpus[0].name if gpus else "cpu"
        import jax
        jax_platform = jax.devices()[0].platform
        # FIXED(FIXAUDIT FX1 F4, 2026-10-22): ProToken's model runs in JAX; require a JAX GPU too.
        if jax_platform != "gpu" and not allow_cpu:
            raise SystemExit(f"ProToken: JAX platform is {jax_platform!r}, not gpu; refusing "
                             f"(pass --allow-cpu to override)")

        def encode(bb4, seq, cbeta, ri):
            return tokenize_by_chain(tok, bb4[:, 1, :], seq, bb4, "real",
                                     pieces=[(0, bb4.shape[0])], residue_index=ri)
        return encode, codebook, {"device": gpu_name, "jax_platform": jax_platform,
                                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                  "residue_numbers": "residue_index + 1"}
    if name == "mini3di":
        import mini3di
        from ensembits.baselines._real_or_ideal_bb import cb_from_bb_or_real
        enc = mini3di.Encoder()
        codebook = np.eye(20, dtype=np.float32)   # full-rank one-hot(20) table

        def encode(bb4, seq, cbeta, ri):
            cbf = cb_from_bb_or_real(bb4[:, :3, :], cbeta)
            states = enc.encode_atoms(ca=bb4[:, 1, :], n=bb4[:, 0, :], c=bb4[:, 2, :], cb=cbf)
            # FIXED(CB audit C finding 6, 2026-10-21): masked terminal residues -> the
            #   invalid state 2 ("D"), as mini3di's `.filled()`, not 0 ("A").
            return np.asarray(states.filled(2)).astype(np.int64)
        return encode, codebook, {"device": "cpu", "masked_state": 2}
    raise ValueError(name)


def provenance_matches(out: Path, want: dict) -> bool:
    side = out.with_name(out.name + ".provenance.json")
    if not (out.exists() and side.exists()):
        return False
    got = json.loads(side.read_text())
    return all(got.get(k) == v for k, v in want.items())


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", required=True, choices=TOKENIZERS)
    ap.add_argument("--set", default="apo")
    ap.add_argument("--device", default="",
                    help="CUDA_VISIBLE_DEVICES value for a manual run (default: leave the "
                         "environment as is; the runner sets it per job)")
    ap.add_argument("--tag", default=None, help="default <tokenizer>_wc_<set>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--protoken-dir", default=None,
                    help="ProToken source tree with weights (default ensembits.paths.PROTOKEN_SRC)")
    ap.add_argument("--allow-cpu", action="store_true", help="ProToken only: allow a CPU run")
    ap.add_argument("--redo", action="store_true",
                    help="rebuild even if the output matches; the old file goes to _pre_<date>/")
    CBP.add_path_args(ap)
    return ap


def expected(args):
    """(output path, manifest path, the sidecar fields a valid output must carry)."""
    paths = CBP.resolve_paths(args)
    tag = args.tag or f"{args.tokenizer}_wc_{args.set}"
    man_path = CBP.manifest_path(paths, args.set)
    out = paths.tokens_dir / f"cryptobench_tokens_{tag}.npz"
    want = {"chain_rule": CC.BASELINE_CHAIN_RULE, "tokenizer": args.tokenizer,
            "manifest_sha256": CC.file_sha256(man_path) if man_path.exists() else None,
            "limit": args.limit, "n_failed": 0,
            "code_sha256": CC.code_sha256(__file__, CC.__file__),
            "builder_code_sha256": CC.code_sha256(*BUILDER_FILES[args.tokenizer]),
            "weights_sha256": weights_sha256(args.tokenizer, args.protoken_dir)}
    return out, man_path, want


def is_done(argv) -> bool:
    args = parser().parse_args(argv)
    out, _, want = expected(args)
    return provenance_matches(out, want)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    paths = CBP.resolve_paths(args, create=("tokens_dir",))
    CD = paths.tokens_dir
    tag = args.tag or f"{args.tokenizer}_wc_{args.set}"
    data_dir = CBP.structs_dir(paths, args.set)
    out, man_path, want = expected(args)
    man = json.loads(man_path.read_text())
    if man.get("meta", {}).get("build_version") != CC.STRUCT_BUILD_VERSION:
        raise SystemExit(f"{man_path}: structures not built by {CC.STRUCT_BUILD_VERSION}; "
                         f"rebuild with cryptobench_build_apo.py")
    entries = man["entries"][:args.limit] if args.limit else man["entries"]
    if out.exists():
        if provenance_matches(out, want) and not args.redo:
            print(f"[{args.tokenizer}] {out.name} up to date; skipped", flush=True)
            return 0
        moved = CC.move_aside(out)
        print(f"[{args.tokenizer}] moved the old {out.name} aside -> {moved[0].parent.name}/")
    print(f"[{args.tokenizer}] set={args.set} entries={len(entries)} tag={tag}", flush=True)

    encode, codebook, info = make_encoder(args.tokenizer, args.protoken_dir, args.allow_cpu)
    toks, fails, n_o_placed, t0 = {}, [], 0, time.time()
    for i, e in enumerate(entries, 1):
        key = f"{e['apo']}_{e['chain']}"
        try:
            d = np.load(data_dir / f"{key}.npz", allow_pickle=True)
            bb4, n_o = CC.backbone4(d)                       # real N/CA/C/O (O placed if absent)
            n_o_placed += n_o
            L = bb4.shape[0]
            ri = CC.chain_residue_index(d["label_seq"])      # gap-aware, 0 at the first residue
            t = np.asarray(encode(bb4, str(d["seq"]), d["cb"], ri))
            if len(t) != L:
                fails.append((key, f"length {len(t)} != {L}"))
                continue
            toks[key] = t.astype(np.uint16)
        except Exception as ex:                               # noqa: BLE001
            fails.append((key, f"{type(ex).__name__}: {ex}"))
        if i % 200 == 0:
            print(f"  {i}/{len(entries)} ok={len(toks)} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(out, **toks)
    np.save(CD / f"cryptobench_tokens_{tag}_codebook.npy", codebook.astype(np.float32))
    side = out.with_name(out.name + ".provenance.json")
    side.write_text(json.dumps({**want, "n_failed": len(fails), "checkpoint": None,
                                "builder": BUILDERS[args.tokenizer],
                                "build_version": CC.STRUCT_BUILD_VERSION, "set": args.set,
                                "n_entries": len(toks), "failed": [k for k, _ in fails],
                                "n_o_placed": n_o_placed, "codebook_shape": list(codebook.shape),
                                "residue_index": "label_seq_id - label_seq_id[0], whole chain",
                                **info,
                                "written_by": "downstream/cryptobench/cryptobench_tokenize_baselines.py"},
                               indent=1))
    used = np.unique(np.concatenate(list(toks.values()))) if toks else np.array([])
    print(f"\nwrote {out} ({len(toks)} entries, {sum(v.size for v in toks.values())} residues)")
    print(f"codebook {codebook.shape}; distinct tokens used: {used.size}/{codebook.shape[0]}; "
          f"O placed: {n_o_placed}")
    if fails:
        print(f"FAILED {len(fails)}:")
        for k, m in fails[:10]:
            print("  ", k, m)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
