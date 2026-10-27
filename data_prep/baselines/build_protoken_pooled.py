"""Frame-pooled ProToken feature for MISATO (mean 32-D codebook embedding over the 8 frames).

What it does
------------
Rebuttal experiment (reviewer szcw #4). Every one of the 8 MISATO frames
(`coords_ca8_misato.pkl`) is tokenized by ProToken with the real backbone of
that frame (`load_bb_for_frame`, chain by chain), each frame's tokens are
looked up in the (512, 32) codebook and averaged over the frames that
tokenized cleanly -> (L, 32). A frame whose token count does not match the
residue count is skipped; a protein with no usable frame or no sequence is
dropped. `--shard k --num-shards n` writes
`protoken_pooled_misato_shard<k>of<n>.npz`; merge them with
`merge_protoken_shards.py --pooled --num-shards n`. The probes read the result
with feature spec `protoken_pooled`. The sequence is `seq` of
`misato_real_bb/<pid>.npz`, the pairs pickle only as fallback (code_fix P30;
the reverse before 2026-10-16).

Paper
-----
not described in the paper (rebuttal only). ProToken baseline:
`appendix.tex` \\label{app:baselines}.

Origin
------
Copied from `submission_exp/src/build_protoken_pooled.py`. Changed:
`PROTOKEN_DIR` from `ensembits.paths.PROTOKEN_SRC` (was an absolute path);
`ROOT` from `ensembits.paths.REPO_ROOT`; `ProToken` and the backbone helpers
from `ensembits.baselines` (were `utils.baselines` /
`submission_exp/src/_real_or_ideal_bb.py`, same code), so the sys.path entries
`REPO_ROOT`, `REPO_ROOT/Ensembits`, `submission_exp/src` were dropped; new flag
`--allow-old-parser` and a MISATO parser check (code_fix C5); comments and this
docstring. Logic unchanged.
2026-10-16: exit 1 on an existing output unless `--overwrite` and on an
ideal-geometry MISATO frame unless `--allow-ideal` (code_fix P15); chains split
at 2.5 Å, real backbones only (code_fix P5); sequence from `misato_real_bb`
first (code_fix P30).
2026-10-18 (authors' decisions, baseline fidelity audit, option A): every frame is
tokenized per true chain (`misato_chain_breaks.json`: cut at boundary / unresolved breaks,
whole across missing-residue gaps, gap-aware residue numbers in ProToken's PDB; a MISATO
`chain_index` is a molecule that can hold several chains; `_chain_pieces.py`), never at
the 2.5 Å C–N rule; the output (or shard) gets `<output>.provenance.json` with
`chain_split` "true_chain_v2" and the chain-break table sha256, which
`merge_protoken_shards.py --pooled` requires of every shard.

Used by the reported model: no (rebuttal baseline row `protoken_pooled`).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/protoken/full/                                   (PROTOKEN_SRC)
          data/cached_descriptors/coords_ca8_misato.pkl
          data/cached_descriptors/aa2ensembits_pairs_misato_v2.pkl
          data/cached_descriptors/misato_real_bb/<pid>.npz
  writes  data/cached_descriptors/protoken_pooled_misato.npz     pid -> (L, 32) float32
          data/cached_descriptors/protoken_pooled_misato_shard<k>of<n>.npz   (with --num-shards > 1;
          merged by merge_protoken_shards.py --pooled, added 2026-10-16)

How to run
----------
    python data_prep/baselines/build_protoken_pooled.py [--limit N] \\
        [--shard k --num-shards n] [--allow-old-parser] [--overwrite] [--allow-ideal]
    (ProToken is loaded with device "" and uses the GPU of CUDA_VISIBLE_DEVICES; --device is
    kept for compatibility only.) Sharded: ONE GPU PER SHARD,
    for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k python data_prep/baselines/build_protoken_pooled.py \\
        --shard $k --num-shards 4 & done; wait
    python data_prep/baselines/merge_protoken_shards.py --pooled --num-shards 4
"""
from __future__ import annotations
import os, sys, pickle, time, argparse
from pathlib import Path
import numpy as np
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _chain_pieces (2026-10-18)
sys.path.insert(0, str(CODE_ROOT / "downstream"))          # rmsf_common (split_record)
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.paths import PROTOKEN_SRC   # noqa: E402  ProToken-1.0 source tree
from ensembits.data_versions import require_misato_parser   # noqa: E402
CACHE = ROOT/"data"/"cached_descriptors"; COORDS = CACHE/"coords_ca8_misato.pkl"
PAIRS = CACHE/"aa2ensembits_pairs_misato_v2.pkl"; REAL_BB = CACHE/"misato_real_bb"
PROTOKEN_DIR = str(PROTOKEN_SRC)   # was an absolute path
OUT = CACHE/"protoken_pooled_misato.npz"
BUILDER_TAG = "v2026-10-18"   # recorded in the output sidecar

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--allow-old-parser", action="store_true",
                    help="accept MISATO input written before the code_fix §4 parser "
                         "fix (only to reproduce old results)")
    # FIXED(code_fix P15, 2026-10-16): old: an existing output was overwritten silently and
    #   ideal-geometry frames were not even counted. New: exit 1 on an existing output
    #   unless --overwrite, and on an ideal-geometry MISATO frame unless --allow-ideal.
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output (default: exit 1 if it exists)")
    ap.add_argument("--allow-ideal", action="store_true",
                    help="accept MISATO frames that fall back to ideal geometry "
                         "(default: exit 1; after the rebuild every frame has a real backbone)")
    args = ap.parse_args()
    # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
    require_misato_parser(REAL_BB, allow_old=args.allow_old_parser)
    global OUT
    if args.num_shards > 1:
        OUT = CACHE/f"protoken_pooled_misato_shard{args.shard}of{args.num_shards}.npz"
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs, refuse_ideal_misato_frame
    refuse_existing_outputs([OUT], args.overwrite)   # code_fix P15, before ProToken loads
    from ensembits.baselines import ProToken
    sys.path.insert(0, PROTOKEN_DIR); sys.path.insert(0, PROTOKEN_DIR+"/data_process")
    from ensembits.baselines._real_or_ideal_bb import load_bb_for_frame
    from _chain_pieces import chain_pieces   # 2026-10-18: true chains only
    # device="" => ProToken does NOT touch CUDA_VISIBLE_DEVICES, so it respects
    # the per-shard CUDA_VISIBLE_DEVICES set by the launcher. Passing "cuda:0"
    # would set CUDA_VISIBLE_DEVICES="cuda:0" (invalid) -> TF silently runs on
    # CPU (GPU 0%, ~100x slower). The --device arg is kept only for API compat.
    tok = ProToken(repo_path=PROTOKEN_DIR, device="")
    cb = np.asarray(tok.codebook, dtype=np.float32)   # (M, 32)
    coords = pickle.load(open(COORDS, "rb"))
    pairs = pickle.load(open(PAIRS, "rb")) if PAIRS.exists() else {}
    seq_by_pid = {(k.split(':')[-1] if ':' in k else k): v.get('seq','') for k,v in pairs.items()}

    def toks_frame(pid, f, ca, seq):
        bb,_,src = load_bb_for_frame("misato", pid, f, ca)
        refuse_ideal_misato_frame("misato", pid, f, src, args.allow_ideal)   # code_fix P15
        # FIXED(code_fix P5, 2026-10-16): split at C–N > 2.5 Å (was 5.0 Å), real backbones
        #   only; an ideal frame (--allow-ideal) stays one chain.
        # FIXED(baseline fidelity audit, 2026-10-18, option A): pieces = true chains of
        #   misato_chain_breaks.json, gap-aware residue numbers (ri + 1), never the 2.5 Å rule.
        chunks, ri = chain_pieces("misato", pid, ca.shape[0])
        if len(chunks)==1:
            idx = tok.tokenize(ca, seq, bb=bb, residue_numbers=ri + 1)
            return idx if idx.shape[0]==ca.shape[0] else None
        L=ca.shape[0]; out=np.zeros(L,dtype=np.int64); ok=True
        for s,e in chunks:
            if e-s<1: continue
            ic=tok.tokenize(ca[s:e], seq[s:e], bb=bb[s:e], residue_numbers=ri[s:e] + 1)
            if ic.shape[0]==(e-s): out[s:e]=ic
            else: ok=False
        return out if ok else None

    pids=sorted(coords.keys())
    if args.num_shards>1: pids=[p for j,p in enumerate(pids) if j%args.num_shards==args.shard]
    if args.limit: pids=pids[:args.limit]
    feats={}; dropped=0; t0=time.time()
    for i,pid in enumerate(pids):
        # FIXED(code_fix P30, 2026-10-16): misato_real_bb 'seq' first, the pairs pickle only
        #   as fallback (old: the pairs pickle first).
        bp=REAL_BB/f"{pid}.npz"
        seq_full=str(np.load(bp,allow_pickle=False)["seq"]) if bp.exists() else ""
        if not seq_full:
            seq_full=seq_by_pid.get(pid,"")
            if not seq_full: dropped+=1; continue
        try:
            ca8=np.asarray(coords[pid],dtype=np.float32)   # (8,L,3)
            # Caveat: sequence cut to the structure length without an alignment check.
            L=ca8.shape[1]; seq=seq_full[:L]
            if len(seq)!=L: dropped+=1; continue
            embs=[]
            for f in range(ca8.shape[0]):
                idx=toks_frame(pid,f,ca8[f],seq)
                if idx is not None: embs.append(cb[idx])
            if not embs: dropped+=1; continue
            feats[pid]=np.mean(embs,axis=0).astype(np.float32)  # (L,32)
        except Exception as e:
            dropped+=1
            if dropped<15: print(f"  DROP {pid}: {type(e).__name__}: {e}",flush=True)
        if (i+1)%500==0: print(f"  [{i+1}/{len(pids)}] {time.time()-t0:.0f}s dropped={dropped}",flush=True)
    np.savez_compressed(OUT, **feats)
    from ensembits.data_versions import write_provenance   # 2026-10-18: chain split recorded
    from rmsf_common import split_record
    write_provenance(OUT, {"builder": f"{Path(__file__).name} {BUILDER_TAG}", "baseline": "protoken",
                           "dataset": "misato", **split_record("misato"),
                           "n_proteins": len(feats), "n_dropped": dropped,
                           "shard": [args.shard, args.num_shards],
                           "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"[protoken_pooled] {len(feats)} proteins (D=32, {dropped} dropped) -> {OUT.name}",flush=True)

if __name__=="__main__": sys.exit(main())
