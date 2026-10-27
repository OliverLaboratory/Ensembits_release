"""Frame-pooled static baseline features for MISATO (mean over the 8 frames).

What it does
------------
Rebuttal experiment (reviewer szcw #4: give the static tokenizers the whole
ensemble). For each protein every one of the 8 MISATO frames
(`coords_ca8_misato.pkl`) is tokenized by the baseline, each frame's tokens are
looked up in the frozen codebook, and the per-residue embeddings are averaged
over frames -> (L, D):
  * `esm3struct`: ESM3 structure encoder (`build_esm3struct_tokens.tokenize_protein`);
  * `aminoaseed`: AminoAseed (`build_aminoaseed_tokens.build_aminoaseed_model`
    + `tokenize_protein`, projected codebook from `aminoaseed_codebook.npy`);
  * `3di`: CPU only, mean of the mini3di centroids of the 8 per-frame tokens in
    `mini3di_tokens_K8_misato.npz`.
Proteins that fail are dropped. The probes read the result with feature spec
`<baseline>_pooled`.

Paper
-----
not described in the paper (rebuttal only; the pooled rows are in the rebuttal
tables compiled by `downstream/_compile_rebuttal_results.py`). Baselines in
`appendix.tex` \\label{app:baselines}.

Origin
------
Copied from `submission_exp/src/build_pooled_baseline_features.py`. Changed:
`ROOT` from `ensembits.paths.REPO_ROOT`; the sibling token builders are
imported from this folder (was `submission_exp/src` on sys.path); new flag
`--allow-old-parser` and a MISATO parser check (code_fix C5); comments and this
docstring. Logic unchanged.
2026-10-16: exit 1 on an existing output unless `--overwrite` and on an
ideal-geometry MISATO frame unless `--allow-ideal` (code_fix P15); chains split
at 2.5 Å, real backbones only (code_fix P5).
2026-10-18 (authors' decisions, baseline fidelity audit, option A): every frame is
tokenized per true chain (`misato_chain_breaks.json`: cut at boundary / unresolved breaks,
whole across missing-residue gaps, gap-aware residue index passed to AminoAseed / ESM3;
a MISATO `chain_index` is a molecule that can hold several chains; `_chain_pieces.py`),
never at the 2.5 Å C–N rule; `3di` refuses a `mini3di_tokens_K8_misato.npz` whose
sidecar does not record the current chain split ("true_chain_v2" + the chain-break table
sha256); each output gets `<output>.provenance.json` with `chain_split` / `chain_breaks`.

Used by the reported model: no (rebuttal baseline rows `*_pooled`).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/coords_ca8_misato.pkl
          data/cached_descriptors/mini3di_tokens_K8_misato.npz, mini3di_centroids.npy   (3di)
          data/cached_descriptors/aminoaseed_codebook.npy                               (aminoaseed)
          the checkpoints read by build_esm3struct_tokens.py / build_aminoaseed_tokens.py
  writes  data/cached_descriptors/<baseline>_pooled_misato.npz   pid -> (L, D) float32
          (+ .provenance.json: builder, chain_split, n_proteins; 2026-10-18)

How to run
----------
    python data_prep/baselines/build_pooled_baseline_features.py --baseline esm3struct --device cuda:0
    python data_prep/baselines/build_pooled_baseline_features.py --baseline aminoaseed --device cuda:0
    python data_prep/baselines/build_pooled_baseline_features.py --baseline 3di      # CPU (uses the K8 cache)
    [--allow-old-parser] [--overwrite] [--allow-ideal]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.data_versions import require_misato_parser   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling builders in data_prep/baselines/
sys.path.insert(0, str(CODE_ROOT / "downstream"))          # rmsf_common (split_record, 2026-10-18)
CACHE = ROOT / "data" / "cached_descriptors"
COORDS = CACHE / "coords_ca8_misato.pkl"
BUILDER_TAG = "v2026-10-18"   # recorded in the output sidecars


def _write_sidecar(out, baseline, n, **extra):
    """<out>.provenance.json with the chain split (2026-10-18)."""
    from ensembits.data_versions import write_provenance
    from rmsf_common import split_record
    write_provenance(out, {"builder": f"{Path(__file__).name} {BUILDER_TAG}", "baseline": baseline,
                           "dataset": "misato", **split_record("misato"), "n_proteins": int(n),
                           "created": time.strftime("%Y-%m-%d %H:%M:%S"), **extra})


def pool_3di(out):
    # FIXED(baseline fidelity audit, 2026-10-18): the K8 tokens must be cut at true chains
    #   only (their sidecar records the current split_record("misato"): "true_chain_v2" and
    #   the sha256 of the current chain-break table); old K8 caches cut at 2.5 Å are refused.
    from ensembits.data_versions import read_provenance
    from rmsf_common import split_record
    k8 = CACHE / "mini3di_tokens_K8_misato.npz"
    side = read_provenance(k8) or {}
    want = split_record("misato")
    if side.get("chain_split") != want["chain_split"] or \
            (side.get("chain_breaks") or {}).get("sha256") != want["chain_breaks"]["sha256"]:
        print(f"[error] {k8.name}: chain_split {side.get('chain_split')!r} (or another chain-break "
              f"table), expected {want['chain_split']!r} with the current misato_chain_breaks.json; "
              f"rebuild it with build_mini3di_tokens_K_misato.py", flush=True)
        raise SystemExit(1)
    toks = np.load(k8)                                          # pid -> (L, 8) uint8
    cent = np.load(CACHE / "mini3di_centroids.npy").astype(np.float32)  # (20, 2)
    feats = {}
    for pid in toks.files:
        t = toks[pid].astype(np.int64)                          # (L, 8)
        feats[pid] = cent[t].mean(axis=1).astype(np.float32)    # (L, 2)
    np.savez_compressed(out, **feats)
    _write_sidecar(out, "3di", len(feats), source=k8.name, source_cache_sha256=side.get("cache_sha256"))
    print(f"[3di_pooled] {len(feats)} proteins -> {out.name}", flush=True)


def pool_model(baseline, device, out, allow_ideal=False):
    coords = pickle.load(open(COORDS, "rb"))                    # pid -> (8, L, 3)
    if baseline == "esm3struct":
        import build_esm3struct_tokens as B
        from esm.pretrained import ESM3_structure_encoder_v0
        enc = ESM3_structure_encoder_v0(device=device)
        cb = enc.codebook.embeddings.detach().cpu().numpy().astype(np.float32)
        # FIXED(code_fix C12, 2026-10-15): every frame now uses its real backbone
        #   from misato_real_bb when it exists (ideal geometry only as fallback),
        #   like the single-frame tokens. Before, bb=None -> ideal geometry for all.
        # FIXED(code_fix P5, 2026-10-16): `ideal` passed on (old: not), so an
        #   ideal-geometry frame (--allow-ideal only) stays one chain at the 2.5 Å split.
        tok = lambda ca, bb, ideal, pk: B.tokenize_protein(enc, ca, device, bb=bb, ideal=ideal,
                                                           **pk)
    elif baseline == "aminoaseed":
        import build_aminoaseed_tokens as B
        model = B.build_aminoaseed_model(device)
        cb = np.load(CACHE / "aminoaseed_codebook.npy").astype(np.float32)
        # FIXED(code_fix C12, 2026-10-15): real backbone per frame (see above).
        tok = lambda ca, bb, ideal, pk: B.tokenize_protein(model, ca, device, bb=bb, ideal=ideal,
                                                           **pk)
    else:
        raise ValueError(baseline)
    from ensembits.baselines._real_or_ideal_bb import load_bb_for_frame, refuse_ideal_misato_frame
    from _chain_pieces import chain_pieces   # 2026-10-18: true chains only
    feats, dropped, t0 = {}, 0, time.time()
    n_real = n_ideal = 0
    pids = sorted(coords.keys())
    for i, pid in enumerate(pids):
        ca8 = np.asarray(coords[pid], dtype=np.float32)         # (8, L, 3)
        try:
            embs = []
            # FIXED(baseline fidelity audit, 2026-10-18, option A): pieces = true chains of
            #   misato_chain_breaks.json with the gap-aware residue index (old: the 2.5 Å split
            #   inside tokenize_protein).
            pieces, ri = chain_pieces('misato', pid, ca8.shape[1])
            pk = {'pieces': pieces, 'residue_index': ri}
            for f in range(ca8.shape[0]):
                bb, _, src = load_bb_for_frame('misato', pid, f, ca8[f])   # real if it exists
                # FIXED(code_fix P15, 2026-10-16): exit 1 on an ideal MISATO frame (old: counted).
                refuse_ideal_misato_frame('misato', pid, f, src, allow_ideal)
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                embs.append(cb[tok(ca8[f], bb, src == 'ideal', pk)])      # (L, D)
            feats[pid] = np.mean(embs, axis=0).astype(np.float32)  # (L, D)
        except Exception as e:
            dropped += 1
            if dropped < 15:
                print(f"  DROP {pid}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(pids)}] {time.time()-t0:.0f}s dropped={dropped}", flush=True)
    np.savez_compressed(out, **feats)
    _write_sidecar(out, baseline, len(feats), n_dropped=dropped, n_ideal_frames=n_ideal)
    print(f"[{baseline}_pooled] frames with real backbone: {n_real}, ideal fallback: {n_ideal}", flush=True)
    print(f"[{baseline}_pooled] {len(feats)} proteins (D={cb.shape[1]}, {dropped} dropped) "
          f"-> {out.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["esm3struct", "aminoaseed", "3di"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-old-parser", action="store_true",
                    help="accept MISATO input written before the code_fix §4 parser "
                         "fix (only to reproduce old results)")
    # FIXED(code_fix P15, 2026-10-16): old: an existing output was overwritten silently
    #   and ideal-geometry frames were only counted. New: exit 1 on an existing output
    #   unless --overwrite, and on an ideal-geometry MISATO frame unless --allow-ideal.
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output (default: exit 1 if it exists)")
    ap.add_argument("--allow-ideal", action="store_true",
                    help="accept MISATO frames that fall back to ideal geometry "
                         "(default: exit 1; after the rebuild every frame has a real backbone)")
    args = ap.parse_args()
    # code_fix C5: the inputs (coords_ca8_misato.pkl, mini3di_tokens_K8_misato.npz)
    # are built from misato_real_bb; refuse it if it comes from the old parser.
    require_misato_parser(CACHE / "misato_real_bb", allow_old=args.allow_old_parser)
    out = CACHE / f"{args.baseline}_pooled_misato.npz"
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs
    refuse_existing_outputs([out], args.overwrite)   # code_fix P15
    if args.baseline == "3di":
        pool_3di(out)
    else:
        pool_model(args.baseline, args.device, out, allow_ideal=args.allow_ideal)


if __name__ == "__main__":
    sys.exit(main())
