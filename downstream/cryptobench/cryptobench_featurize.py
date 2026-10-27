"""CryptoBench: continuous Ensembits readouts z / q / q0 at P = 1 (secondary readouts).

What it does
------------
Step 2a' of the CryptoBench pipeline (next to `cryptobench_tokenize.py`, which writes the
level-0 tokens, the PRIMARY readout; DECISIONS.md CB4). For every chain of a built set:

  N/CA/C (1, L, 3, 3) -> 256-D descriptor (`ensembits.descriptor._descriptor` with the
  checkpoint's dihedral convention and chain-break cutoff) -> standardise with the
  checkpoint's stats -> (L, P = 1, 256) -> encoder -> z (L, 128);
  z -> residual quantizer `_quantize` -> q = sum of the codewords of all levels (L, 128),
  all_tokens; q0 = level-0 codebook vector of all_tokens[0] (L, 128).

  z  = pre-quantization latent (not a token; upper bound of what the encoder knows)
  q  = full code vector (all RVQ levels; 3 tokens per residue for [2048, 128, 128])
  q0 = level-0 code vector = the codebook row of the paper's token (identical to the
       token readout of the probes, which embed the token with the same codebook)

Checks written to the provenance sidecar: `n_token_mismatch` = residues whose level-0 token
differs from `model.encode` on the same descriptor and `n_tokenfile_mismatch` = residues whose
level-0 token differs from the token file of `cryptobench_tokenize.py` (`--tokens-tag`,
required). FIXED(REAUDIT CB1 F4 / CB2 F1 / CB3 F2 / CB5 F3, 2026-10-22): both must be 0: this
script exits 1 otherwise, the probes' gate (`cryptobench_common.check_token_file`) refuses a
file with a non-zero count, and the runner's completeness report counts them.

Paper
-----
Not in the paper yet (secondary CryptoBench readouts, appendix; DECISIONS.md CB4).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Encoder / RVQ: `1_method.tex` \\paragraph{Residual quantizer.}; descriptor `appendix.tex`
\\label{app:descriptor-esm3}.

Origin
------
Ported from the collaborator's `ensembits-work/submission_exp/src/cryptobench_repr.py`
(class `EnsembitsFeaturizer.__call__`; sha256
3208d2bbe1383e10e52f4238a3269d8e8432ab720f1112c21077d99660d58d54; full path and the
collaborator's tree in camera_ready/cryptobench_plan.md, read only).
Changes from the source:
  * FIXED(CB audit C finding 1 / audit B finding 4, 2026-10-21): the source used its own
    `ensembits_featurize.load_model` and `build_esm3desc_dihedral._descriptor`, hard-wired to
    the legacy +180 deg phi/psi with no chain-break handling (right for the old flagship,
    wrong for the `_rv` checkpoints: 99.8 % of tokens change). Here: the camera-ready
    `ensembits.featurize.load_model` and `ensembits.descriptor._descriptor(convention=
    ens.dihedral_convention, chain_break_cutoff=ens.chain_break_cutoff)`, as
    `cryptobench_tokenize.py` does; unfinished checkpoints refused (code_fix P12).
  * structures from our structure stage (`cryptobench_build_apo.py`: MSE / SEC / PYL kept),
    not the source's Biopython stage (which drops MSE, audit B finding 6).
  * one npz per checkpoint (`<key>|z`, `<key>|q`, `<key>|q0`) with a provenance sidecar,
    level-0 token cross-checks; existing output skipped if its sidecar names the same
    checkpoint and structure manifest, else moved to `_pre_<date>/`.
  * `--model-dir` has no default (the default CKPT_DIR is the pre-retrain checkpoint).
  * 2026-10-22: `--tokens-tag` required; the sidecar records the structure manifest, the
    token file's sha256 and this script's sha256 (input fingerprint, `is_done`).
  The extraction itself (encoder -> _quantize -> q, codebook(all_tokens[0]) -> q0) is the
  source's. A first version of this file written by a killed session on 2026-10-21
  (same extraction, three files, no checks) was replaced by this one (PORT.md).

Used by the reported model: yes (secondary readouts of the two lambda candidates).

Inputs / outputs
----------------
Read  : `<model-dir>/config.json, best.pt, stats.npz`; `<structs-root>/cryptobench_<set>/*.npz`;
        `<manifest-dir>/<set>_manifest.json`; optional `cryptobench_tokens_<tokens-tag>.npz`.
Write : `<tokens-dir>/cryptobench_dense_<tag>.npz` (float32) + `.npz.provenance.json`.

How to run
----------
  CUDA_VISIBLE_DEVICES=0 $PY downstream/cryptobench/cryptobench_featurize.py --set apo \\
      --model-dir output/ensembits_bm_lam0p25_rv --tag ens_lam0p25_rv --tokens-tag ens_lam0p25_rv
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
import torch

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(HERE))
from ensembits.paths import REPO_ROOT as ROOT  # noqa: E402,F401
import _cb_paths as CBP  # noqa: E402
import cryptobench_common as CC  # noqa: E402

READOUTS = ("z", "q", "q0")


class EnsembitsFeaturizer:
    """z / q / q0 of one chain at P = 1 (collaborator's EnsembitsFeaturizer on our loader)."""

    def __init__(self, model_dir, device: str, allow_unfinished: bool = False):
        from ensembits.featurize import load_model
        self.device = device
        self.ens = load_model(model_dir, device=device, allow_unfinished=allow_unfinished)
        if self.ens.config.get("input_dim") != 256:
            raise SystemExit(f"256-D dihedral descriptor expected, model wants "
                             f"{self.ens.config.get('input_dim')}")
        if not hasattr(self.ens.model, "_quantize"):
            raise SystemExit("z / q / q0 need a residual-quantizer model (RVQVAETokenizer)")
        self.mu = torch.as_tensor(self.ens.mean, dtype=torch.float32, device=device)
        self.sd = torch.as_tensor(self.ens.std, dtype=torch.float32, device=device)

    @torch.no_grad()
    def __call__(self, bb: np.ndarray):
        """bb (1, L, 3, 3) N/CA/C -> dict z, q, q0 (L, 128) float32, tok0, tok0_encode (L,)."""
        from ensembits.descriptor import _descriptor
        m = self.ens.model
        d = _descriptor(torch.from_numpy(np.asarray(bb, np.float32)), self.device,
                        convention=self.ens.dihedral_convention,
                        chain_break_cutoff=self.ens.chain_break_cutoff).to(self.device)
        d = ((d - self.mu) / self.sd).transpose(0, 1).contiguous()     # (L, P=1, 256)
        z = m.encoder(d)                                                # (L, 128)
        q, all_tokens, _ = m._quantize(z)                               # (L, 128)
        q0 = m.vq_levels[0].codebook(all_tokens[0])                     # (L, 128)
        tok_enc = m.encode(d)                                           # the token path
        return {"z": z.float().cpu().numpy(), "q": q.float().cpu().numpy(),
                "q0": q0.float().cpu().numpy(),
                "tok0": all_tokens[0].cpu().numpy().astype(np.int64),
                "tok0_encode": tok_enc.cpu().numpy().astype(np.int64)}


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True,
                    help="checkpoint folder (no default: the default CKPT_DIR is pre-retrain)")
    ap.add_argument("--tag", required=True, help="output cryptobench_dense_<tag>.npz")
    ap.add_argument("--tokens-tag", required=True,
                    help="token file of the same checkpoint to cross-check level 0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--set", default="apo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-unfinished", action="store_true")
    ap.add_argument("--redo", action="store_true")
    CBP.add_path_args(ap)
    return ap


def expected(args):
    """(output path, sidecar fields a valid output must carry)."""
    from ensembits.featurize import checkpoint_provenance
    paths = CBP.resolve_paths(args)
    man_path = CBP.manifest_path(paths, args.set)
    tok = paths.tokens_dir / f"cryptobench_tokens_{args.tokens_tag}.npz"
    out = paths.tokens_dir / f"cryptobench_dense_{args.tag}.npz"
    want = {"checkpoint": checkpoint_provenance(CBP.repo_path(args.model_dir)),
            "manifest_sha256": CC.file_sha256(man_path) if man_path.exists() else None,
            "tokens_file_sha256": CC.file_sha256(tok) if tok.exists() else None,
            "code_sha256": CC.file_sha256(__file__), "limit": args.limit,
            "n_failed": 0, "n_token_mismatch": 0, "n_tokenfile_mismatch": 0}
    return out, want


def is_done(argv) -> bool:
    args = parser().parse_args(argv)
    out, want = expected(args)
    side = out.with_name(out.name + ".provenance.json")
    if not (out.exists() and side.exists()):
        return False
    got = json.loads(side.read_text())
    return all(got.get(k) == v for k, v in want.items())


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    paths = CBP.resolve_paths(args, create=("tokens_dir",))
    set_dir = CBP.structs_dir(paths, args.set)
    man_path = CBP.manifest_path(paths, args.set)
    from ensembits.featurize import check_checkpoint_finished
    model_dir = CBP.repo_path(args.model_dir)
    check_checkpoint_finished(model_dir, allow_unfinished=args.allow_unfinished)
    out, want = expected(args)
    side = out.with_name(out.name + ".provenance.json")
    if want["tokens_file_sha256"] is None:
        raise SystemExit(f"token file cryptobench_tokens_{args.tokens_tag}.npz missing")
    if out.exists():
        if is_done(argv) and not args.redo:
            print(f"[featurize] {out.name} up to date; skipped")
            return 0
        CC.move_aside(out)
        print(f"[featurize] moved the old {out.name} aside")
    man = json.loads(man_path.read_text())
    entries = man["entries"][:args.limit] if args.limit else man["entries"]
    device = args.device if torch.cuda.is_available() else "cpu"
    fz = EnsembitsFeaturizer(model_dir, device, args.allow_unfinished)
    tokfile = np.load(paths.tokens_dir / f"cryptobench_tokens_{args.tokens_tag}.npz")
    store, fails, n_mis, n_mis_file, n_res, t0 = {}, [], 0, 0, 0, time.time()
    for i, e in enumerate(entries, 1):
        key = f"{e['apo']}_{e['chain']}"
        try:
            bb = np.load(set_dir / f"{key}.npz", allow_pickle=True)["bb"]
            r = fz(bb)
            for k in READOUTS:
                store[f"{key}|{k}"] = r[k].astype(np.float32)
            n_res += len(r["tok0"])
            n_mis += int((r["tok0"] != r["tok0_encode"]).sum())
            if key not in tokfile:
                raise KeyError(f"{key} missing from the token file")
            n_mis_file += int((tokfile[key].astype(np.int64) != r["tok0"]).sum())
        except Exception as ex:                                  # noqa: BLE001
            fails.append((key, f"{type(ex).__name__}: {ex}"))
        if i % 200 == 0:
            print(f"  {i}/{len(entries)} ok={len(store)//3} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(out, **store)
    side.write_text(json.dumps({**want, "set": args.set, "readouts": list(READOUTS),
                                "n_entries": len(store) // 3, "n_failed": len(fails),
                                "failed": [k for k, _ in fails], "n_residues": n_res,
                                "n_token_mismatch": n_mis, "tokens_tag": args.tokens_tag,
                                "n_tokenfile_mismatch": n_mis_file,
                                "build_version": man.get("meta", {}).get("build_version"),
                                "dihedral_convention": fz.ens.dihedral_convention,
                                "chain_break_cutoff": fz.ens.chain_break_cutoff,
                                "written_by": "downstream/cryptobench/cryptobench_featurize.py"},
                               indent=1))
    print(f"wrote {out.name} ({len(store)//3} chains, {n_res} residues); level-0 mismatches: "
          f"encode {n_mis}, token file {n_mis_file}")
    rc = 0
    if fails:
        print(f"FAILED {len(fails)}:")
        for k, m in fails[:10]:
            print("  ", k, m)
        rc = 1
    if n_mis or n_mis_file:
        # FIXED(REAUDIT CB1 F4 / CB2 F1, 2026-10-22): a level-0 mismatch is an error.
        print("ERROR: level-0 tokens of the readouts disagree with the token path")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
