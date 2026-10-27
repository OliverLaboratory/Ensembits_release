"""CryptoBench: tokenize every chain with an Ensembits checkpoint at P = 1 (level-0 tokens).

What it does
------------
Step 2a of the CryptoBench pipeline (see `README.md` in this folder).

Each CryptoBench entry is a SINGLE crystal structure, so this is the P = 1
regime: the encoder sees a set of one descriptor per residue and must infer
the residue's accessible geometries from one frame. The labels come from the
holo partner, so nothing about the pocket is visible in the input.

  1. Loads the checkpoint given by `--model-dir` (required since 2026-10-22;
     `ensembits.featurize.load_model`) and refuses any model whose
     input is not the 256-D dihedral descriptor.
  2. For every manifest entry: N/CA/C (1, L, 3, 3) -> 256-D descriptor
     (`ensembits.descriptor._descriptor`, dynamical kNN, K = 16) -> standardise
     with the checkpoint's stats -> (L, P = 1, 256) -> `model.encode` -> one
     level-0 token per residue (uint16).
  3. Saves all tokens in one npz keyed "<apo>_<chain>" and the level-0 codebook
     (2048 x 128 for the reported model); the probe embeds tokens with this
     frozen codebook ("level-0 code vector").

Paper
-----
Not in the paper yet: CryptoBench goes into the main text
(`camera_ready/require_fix.md` #24, `camera_ready/code_fix.md` E3).
# FIXED(require_fix #24, tex pass 2026-10-26): CryptoBench section exists in appendix.
Descriptor: `appendix.tex` \\label{app:descriptor-esm3}; tokenizer:
`1_method.tex` \\paragraph{Residual quantizer.} (level-0 token, M_1 = 2048).
Benchmark: Škrhák et al., Bioinformatics 41(1):btae745, 2025.

Origin
------
Copied from `submission_exp/src/rebuttal_analyses/cryptobench_tokenize.py`.
Changes:
  * imports: `build_esm3desc_dihedral._descriptor` -> `ensembits.descriptor._descriptor`,
    `ensembits_featurize.load_model` -> `ensembits.featurize.load_model`
    (CONVENTIONS §4); the sys.path inserts of the research repo were dropped.
  * the descriptor call passes `convention=ens.dihedral_convention`: the codebase's
    `_descriptor` defaults to the fixed ("iupac") convention, the original call
    used the +180 deg one; with this argument a checkpoint trained before the fix
    gets exactly the features of the original script, a retrained one the fixed
    features (same pattern as `downstream/_pretokenize_misato_dihedral.py`).
  * `--model-dir` default: originally output/dihedral256_fapeH_rvq_2048_128_128_PRODSPLIT
    (the old mdCATH+MISATO flagship, which produced every rebuttal number), then
    `ensembits.paths.CKPT_DIR`; since 2026-10-22 no default (required).
  * locations from `_cb_paths.py` (originally ROOT = the research repo root;
    inputs data/cached_descriptors/cryptobench_<set>/, manifest
    data/cryptobench/<set>_manifest.json; output data/cached_descriptors/).
Tokenization logic is unchanged.
Fixes 2026-10-16 (FIXED markers): provenance sidecar written, an existing token
file of another checkpoint is refused (--force) (code_fix P17); --allow-unfinished (P12).
Fixes 2026-10-22 (CryptoBench port re-audit, FIXLIST_20261022.md items 1 / 6):
  * `--model-dir` is REQUIRED (the old default CKPT_DIR is the pre-retrain checkpoint;
    REAUDIT CB1 F5 / CB3 F3); exit code 1 when any chain fails;
  * the sidecar also records the structure manifest sha256, its build version and the
    sha256 of this script (input fingerprint; the probes' gate checks the manifest);
  * an existing output is skipped when its sidecar matches (`is_done`), otherwise moved to
    `_pre_<date>/` before the rebuild (was overwritten in place, REAUDIT CB5 F4);
  * the TODOs about z / q and effective bits are done (`cryptobench_featurize.py`,
    `cryptobench_common.effective_bits`).
Fix round 2 (2026-10-22, FIXLIST2 item 5, FIXAUDIT FX3 F1): when a rerun is due (the existing
file's sidecar does not match the current inputs, e.g. a retrained checkpoint), the old file is
moved to `_pre_<date>/` BEFORE the checkpoint check, so the job reruns instead of failing; the
runner passes `--rerun-on-change`. Without it (a manual run), a file written by ANOTHER
checkpoint is still refused unless --force / --redo (code_fix P17); `--redo` lifting that
refusal is intended (the old file is moved aside, never overwritten; FX2 F9).

Used by the reported model: yes, the "Ensembits level-0 token" row.
# (2026-10-22) z / q / q0 readouts: cryptobench_featurize.py. The 1-frame ablation rows of
#   the rebuttal have no BioEmu-CATH + MISATO checkpoint and are not part of the camera-ready run.

Inputs / outputs
----------------
Read  : `<model-dir>/config.json, best.pt, stats.npz`;
        `<structs-root>/cryptobench_<set>/<apo>_<chain>.npz` (bb (1, L, 3, 3));
        `<manifest-dir>/<set>_manifest.json` (both from `cryptobench_build_apo.py`).
Write : `<tokens-dir>/cryptobench_tokens_<tag>.npz` {"<apo>_<chain>": (L,) uint16},
        `<tokens-dir>/cryptobench_tokens_<tag>.npz.provenance.json` (checkpoint block, P17),
        `<tokens-dir>/cryptobench_tokens_<tag>_codebook.npy` (M_1, d).
        Default <tokens-dir> = data/cached_descriptors/cryptobench_runs/<run_tag>/.
The rebuttal tokens are data/cached_descriptors/cryptobench_tokens_dih256_fapeH*.npz
(old flagship, legacy layout).

Dependencies left in the research repo: none (core library only; `esm` for the
ESM3 frame / kNN utilities).

How to run
----------
  $PY downstream/cryptobench/cryptobench_tokenize.py --set apo --tag ens_lam1_rv \\
      --model-dir output/ensembits_bm_lam1_rv --device cuda:0
(The original docstring also exported the CUDA libraries of its environment on
LD_LIBRARY_PATH before running.)
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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
from ensembits.paths import CKPT_DIR, REPO_ROOT as ROOT  # noqa: E402,F401
import _cb_paths as CBP  # noqa: E402

APO = None     # structure cache of the chosen set, set from --set
OUTDIR = None  # token output folder, set from --tokens-dir / --run-tag


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # FIXED(REAUDIT CB1 F5 / CB3 F3, 2026-10-22): old: default str(CKPT_DIR) (pre-retrain
    #   checkpoint). New: required.
    ap.add_argument("--model-dir", required=True,
                    help="checkpoint folder (config.json, best.pt, stats.npz)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", default="apo",
                    help="which built set: apo | holo | apo_main | holo_main")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing token file written by another checkpoint "
                         "(code_fix P17)")
    ap.add_argument("--redo", action="store_true",
                    help="rebuild even if the output matches (old file moved to _pre_<date>/)")
    ap.add_argument("--rerun-on-change", action="store_true",
                    help="move an outdated file aside and rerun even if it was written by another "
                         "checkpoint (the runner's mode)")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="accept a checkpoint whose training run has not finished (code_fix P12)")
    CBP.add_path_args(ap)
    return ap


def expected(args):
    """(output path, the sidecar fields a valid output must carry)."""
    import cryptobench_common as CC
    from ensembits.featurize import checkpoint_provenance
    paths = CBP.resolve_paths(args)
    man = Path(args.manifest) if args.manifest else CBP.manifest_path(paths, args.set)
    out = paths.tokens_dir / f"cryptobench_tokens_{args.tag}.npz"
    want = {"checkpoint": checkpoint_provenance(CBP.repo_path(args.model_dir)),
            "manifest_sha256": CC.file_sha256(man) if man.exists() else None,
            "code_sha256": CC.code_sha256(__file__, Path(CC.__file__))["cryptobench_tokenize.py"],
            "n_failed": 0, "limit": args.limit, "set": args.set}
    return out, want


def is_done(argv) -> bool:
    """True iff the output exists and its sidecar matches the current inputs (runner)."""
    args = parser().parse_args(argv)
    out, want = expected(args)
    side = out.with_name(out.name + ".provenance.json")
    if not (out.exists() and side.exists()):
        return False
    got = json.loads(side.read_text())
    return all(got.get(k) == v for k, v in want.items())


def main(argv=None):
    args = parser().parse_args(argv)
    paths = CBP.resolve_paths(args, create=("tokens_dir",))
    global APO, OUTDIR
    APO = CBP.structs_dir(paths, args.set)
    OUTDIR = paths.tokens_dir
    if args.manifest is None:
        args.manifest = str(CBP.manifest_path(paths, args.set))

    import cryptobench_common as CC
    from ensembits.descriptor import _descriptor
    from ensembits.featurize import (load_model, checkpoint_provenance, check_provenance_sidecar,
                                     write_provenance_sidecar, check_checkpoint_finished)
    # FIXED(code_fix P12 / P17, 2026-10-16): refuse an unfinished checkpoint; an existing
    #   token file of this --tag is overwritten only if its provenance sidecar names the same
    #   checkpoint (or with --force); the new file gets <file>.provenance.json.
    model_dir = CBP.repo_path(args.model_dir)
    check_checkpoint_finished(model_dir, allow_unfinished=args.allow_unfinished)
    prov = checkpoint_provenance(model_dir)
    out, want = expected(args)
    if out.exists():
        if is_done(argv if argv is not None else sys.argv[1:]) and not args.redo:
            print(f"[tokenize] {out.name} up to date; skipped")
            return 0
        # FIXED(FIXAUDIT FX3 F1, 2026-10-22): a due rerun moves the old file aside first; the
        #   other-checkpoint refusal (P17) remains for manual runs only.
        if not (args.force or args.redo or args.rerun_on_change):
            check_provenance_sidecar(out, prov, force=False)
        # FIXED(REAUDIT CB5 F4, 2026-10-22): move the old file aside instead of overwriting.
        CC.move_aside(out)
        print(f"[tokenize] moved the old {out.name} aside")

    man = json.load(open(args.manifest))
    entries = man["entries"][:args.limit] if args.limit else man["entries"]
    ens = load_model(model_dir, device=args.device, allow_unfinished=args.allow_unfinished)
    dim = ens.config.get("input_dim")
    print(f"[tokenize] model={Path(args.model_dir).name} input_dim={dim} "
          f"dihedral_convention={ens.dihedral_convention} entries={len(entries)}", flush=True)
    if dim != 256:
        raise SystemExit(f"this script builds the 256-D dihedral descriptor; model wants {dim}")

    # Standardisation stats saved with the checkpoint (identity for the reported model).
    mu = torch.as_tensor(ens.mean, dtype=torch.float32, device=args.device)
    sd = torch.as_tensor(ens.std, dtype=torch.float32, device=args.device)

    toks, t0, fails = {}, time.time(), []
    for i, e in enumerate(entries, 1):
        key = f"{e['apo']}_{e['chain']}"
        try:
            bb = np.load(APO / f"{key}.npz", allow_pickle=True)["bb"].astype(np.float32)
            with torch.no_grad():
                # Paper: appendix.tex app:descriptor-esm3 — f_r in R^256 for the single frame.
                # code_fix §1: the φ/ψ convention and the chain-break cutoff follow the
                #   checkpoint (iupac + 2.5 Å for the retrained `_rv` runs).
                desc = _descriptor(torch.from_numpy(bb), args.device,
                                   convention=ens.dihedral_convention,
                                   chain_break_cutoff=ens.chain_break_cutoff).to(args.device)
                desc = ((desc - mu) / sd).transpose(0, 1).contiguous()   # (L, P, 256)
                # Set encoder + RVQ; `encode` returns the level-0 token only.
                t = ens.model.encode(desc).cpu().numpy().astype(np.uint16)
            toks[key] = t
        except Exception as ex:                                  # noqa: BLE001
            fails.append((key, f"{type(ex).__name__}: {ex}"))
        if i % 200 == 0:
            print(f"  {i}/{len(entries)}  ok={len(toks)} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(out, **toks)
    write_provenance_sidecar(out, prov, set=args.set, n_entries=len(toks), n_failed=len(fails),
                             failed=[k for k, _ in fails],
                             dihedral_convention=ens.dihedral_convention,
                             chain_break_cutoff=ens.chain_break_cutoff,
                             manifest_sha256=want["manifest_sha256"],
                             build_version=man.get("meta", {}).get("build_version"),
                             code_sha256=want["code_sha256"], limit=args.limit,
                             written_by="downstream/cryptobench/cryptobench_tokenize.py")

    # the codebook the probe will embed with: RVQ level 0 (the L_0 alphabet every
    # downstream task uses). For an aminoaseed-projection level the stored
    # weight is already basis @ projection, i.e. the space codes are assigned in.
    vq0 = ens.model.vq_levels[0] if hasattr(ens.model, "vq_levels") else ens.model.vq
    w = vq0.codebook.weight.detach().float().cpu().numpy()
    np.save(OUTDIR / f"cryptobench_tokens_{args.tag}_codebook.npy", w)

    used = np.unique(np.concatenate(list(toks.values()))) if toks else np.array([])
    print(f"\nwrote {out}  ({len(toks)} entries, {sum(v.size for v in toks.values())} residues)")
    print(f"codebook {w.shape} -> cryptobench_tokens_{args.tag}_codebook.npy")
    print(f"distinct tokens used: {used.size} / {w.shape[0]}")
    if fails:
        print(f"FAILED {len(fails)}:")
        for k, m in fails[:10]:
            print("  ", k, m)
        # FIXED(REAUDIT CB1 F5, 2026-10-22): non-zero exit (was 0).
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
