"""Per-frame mini3di (Foldseek 3Di) tokens for the 8 MISATO frames.

What it does
------------
For every (pid, frame) of `coords_ca8_misato.pkl` it loads the real N / C /
C-beta of that frame from `misato_real_bb`
(`ensembits.baselines._real_or_ideal_bb.load_bb_for_frame`; ideal geometry only
if no real frame matches), cuts the protein into its chains (MISATO: runs of
`misato_chain_breaks.json`: true chains, whole across missing-residue gaps, the same
for every frame;
mdCATH / BioEmu: the whole protein; `_chain_pieces.py`, 2026-10-18) and
tokenizes each with `mini3di.Encoder.encode_atoms(...).filled()` (terminal residues
become the invalid state 2). Output per pid: (L, 8) uint8 tokens in [0, 20).
These feed the Vote_3di baseline and the pooled 3Di feature.

Paper
-----
`appendix.tex` \\label{app:baselines}, *Vote_3di* (majority vote over the
frames; the probes break ties by the nearest centroid, require_fix #22).
Results:
  * `2_results.tex` \\label{result:rmsf}, Table \\label{tab:rmsf} (baseline rows,
    mdCATH-div and MISATO columns);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * `appendix.tex` \\label{app:downstreams} (Tables \\label{tab:ec_d1_structure},
    \\label{tab:go_structure}, \\label{tab:binding_structure},
    \\label{tab:affinity_structure}) and \\label{app:experiments} (all splits).

Origin
------
Copied from `submission_exp/src/build_mini3di_tokens_K_misato.py`. Changed:
`ROOT` from `ensembits.paths.REPO_ROOT`; backbone helpers from
`ensembits.baselines` (same code); a CLI with `--allow-old-parser` and a MISATO
parser check (code_fix C5; the original had no flags); comments and this
docstring. Logic unchanged.
2026-10-16: exit 1 on an existing output unless `--overwrite` and on an
ideal-geometry MISATO frame unless `--allow-ideal` (code_fix P15); chains split
at 2.5 Å, real backbones only (code_fix P5; replaced 2026-10-18, see below).
2026-10-18 (authors' decision, baseline fidelity audit): pieces = true chains only,
never the 2.5 Å C–N rule; the sidecar records `chain_split` instead of chain_break_A.

Phase 2 (2026-10-16): `--dataset bioemu` tokenizes all 10 frames of bioemu_cath_real_bb
bb_10 (real Cβ from bioemu_cath_extras) -> `mini3di_tokens_K10_bioemu.npz`.
2026-10-16 (preretrain_audit_r2 F1 / F2 / F7): `--dataset mdcath` (port of the research
`build_mini3di_tokens_K_mdcath.py`: all 10 ca_10 frames, real backbone of mdcath_real_bb,
whole domain) -> `mini3di_tokens_K10_mdcath.npz`; every output gets
`<output>.provenance.json`; a protein of the RMSF residue set that fails makes the
build exit 1 (`--max-fail-frac`); a frame count other than K is an error.

Used by the reported model: no (Vote_3di and 3di_pooled baseline rows).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/coords_ca8_misato.pkl   (data_prep/build_coords_ca8_misato.py)
          data/cached_descriptors/misato_real_bb/<pid>.npz
          (--dataset mdcath: data/external/mdcath_div ca_10 + mdcath_real_bb;
           --dataset bioemu: bioemu_cath_real_bb + bioemu_cath_extras)
  writes  data/cached_descriptors/mini3di_tokens_K8_misato.npz   pid -> (L, 8) uint8
          data/cached_descriptors/mini3di_tokens_K10_{mdcath,bioemu}.npz  (L, 10) uint8
          <each output>.provenance.json

How to run
----------
    python data_prep/baselines/build_mini3di_tokens_K_misato.py [--dataset misato|mdcath|bioemu] \\
        [--allow-old-parser] [--overwrite] [--allow-ideal] [--max-fail-frac 0]
    (needs the `mini3di` package)
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse, pickle, sys, time
from pathlib import Path
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _bioemu_inputs
sys.path.insert(0, str(CODE_ROOT / 'downstream'))          # rmsf_common (sidecars, tolerances)
BUILDER_TAG = 'v2026-10-18'   # recorded in the cache sidecars
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.data_versions import require_misato_parser   # noqa: E402
MISATO_REAL_BB = ROOT / 'data' / 'cached_descriptors' / 'misato_real_bb'   # read through _real_or_ideal_bb
COORDS = ROOT / 'data' / 'cached_descriptors' / 'coords_ca8_misato.pkl'
OUT = ROOT / 'data' / 'cached_descriptors' / 'mini3di_tokens_K8_misato.npz'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-old-parser', action='store_true',
                    help='accept MISATO input written before the code_fix §4 parser '
                         'fix (only to reproduce old results)')
    # FIXED(code_fix P15, 2026-10-16): old: an existing output was skipped silently and
    #   MISATO frames that fell back to ideal geometry were only counted. New: exit 1 on
    #   an existing output unless --overwrite; exit 1 on an ideal-geometry MISATO frame
    #   unless --allow-ideal.
    ap.add_argument('--overwrite', action='store_true',
                    help='replace existing outputs (default: exit 1 if one exists)')
    ap.add_argument('--allow-ideal', action='store_true',
                    help='accept MISATO frames that fall back to ideal geometry '
                         '(default: exit 1; after the rebuild every frame has a real backbone)')
    # Phase 2 (2026-10-16): --dataset bioemu = all 10 frames of bioemu_cath_real_bb bb_10
    #   (real N/CA/C, real Cβ from bioemu_cath_extras) -> mini3di_tokens_K10_bioemu.npz.
    # FIXED(preretrain_audit_r2 F2 / Q29d, 2026-10-16): new --dataset mdcath = the research
    #   builder submission_exp/src/build_mini3di_tokens_K_mdcath.py ported: all 10 frames of
    #   the mdCATH-div ca_10 with the real N / C / Cβ of mdcath_real_bb, the whole domain
    #   (2026-10-18; was a 2.5 Å split, a no-op on these domains) -> mini3di_tokens_K10_mdcath.npz; domains
    #   without a real backbone (none since data_prep/build_mdcath_real_bb.py, 2026-10-17) are not written.
    ap.add_argument('--dataset', default='misato', choices=('misato', 'mdcath', 'bioemu'))
    # FIXED(preretrain_audit_r2 F7, 2026-10-16): new: failures inside the RMSF set fail.
    ap.add_argument('--max-fail-frac', type=float, default=0.0,
                    help='allowed fraction of RMSF-set proteins that fail (default 0)')
    args = ap.parse_args(argv)
    K = 8 if args.dataset == 'misato' else 10
    out_path = OUT.with_name(f'mini3di_tokens_K{K}_{args.dataset}.npz')
    if args.dataset == 'misato':
        # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
        require_misato_parser(MISATO_REAL_BB, allow_old=args.allow_old_parser)
    # FIXED(code_fix E0, 2026-10-20): MISATO caches rebuilt with the fixed parser (CX proline Cα).
    # FIXED(code_fix P15, 2026-10-16): old: `print('cache exists'); return 0`.
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs, refuse_ideal_misato_frame
    refuse_existing_outputs([out_path], args.overwrite)
    import mini3di
    enc = mini3di.Encoder()

    from ensembits.baselines._real_or_ideal_bb import cb_from_bb_or_real
    from _chain_pieces import chain_pieces   # 2026-10-18: true chains only
    import _bioemu_inputs as BI   # load_bb_for_frame: bioemu here, MISATO / mdCATH passed through
    load_bb_for_frame = BI.load_bb_for_frame
    from rmsf_common import (SEQ_NOT_USED, split_record, check_builder_failures,
                             dataset_input_record, write_cache_provenance)

    coords = (BI.CaFrames() if args.dataset == 'bioemu' else
              BI.MdcathCaFrames() if args.dataset == 'mdcath' else pickle.load(open(COORDS, 'rb')))
    pids = sorted(coords.keys())
    print(f'[mini3di-K{K}-{args.dataset}] {len(pids)} proteins, K={K}', flush=True)
    out = {}
    failed = {}; ideal_pids = []
    n_real = n_ideal = 0
    n_chains_total = 0
    t0 = time.time()
    for i, pid in enumerate(pids):
        try:
            caK = np.asarray(coords[pid], dtype=np.float32)  # (K, L, 3)
            P, L, _ = caK.shape
            if P != K:
                raise ValueError(f'{P} frames, expected {K}')
            tokens = np.zeros((L, P), dtype=np.uint8)
            # FIXED(baseline fidelity audit, 2026-10-18): one piece per true chain, the same
            #   for every frame (MISATO: true chains of misato_chain_breaks.json, whole across
            #   internal gaps; mdCATH / BioEmu: the whole protein). Old (code_fix P5): split
            #   at C–N > 2.5 Å per frame on real backbones.
            chunks, _ = chain_pieces(args.dataset, pid, L)   # mini3di: no numbering
            n_chains_total += len(chunks)
            for p in range(P):
                ca = caK[p]
                bb, cb_real, src = load_bb_for_frame(args.dataset, pid, p, ca)
                refuse_ideal_misato_frame(args.dataset, pid, p, src, args.allow_ideal)   # code_fix P15
                if args.dataset == 'mdcath' and src != 'real':
                    raise ValueError(f'frame {p}: no real backbone (C-alpha only)')
                cb = cb_from_bb_or_real(bb, cb_real)
                n_atom = bb[:, 0].astype(np.float32)
                c_atom = bb[:, 2].astype(np.float32)
                # Per-chain tokenisation (mini3di was trained on single
                # chains; multi-chain inputs introduce fictitious peptide
                # bonds at the boundaries).
                for s, e in chunks:
                    states_ma = enc.encode_atoms(ca[s:e].astype(np.float32),
                                                  cb[s:e].astype(np.float32),
                                                  n_atom[s:e], c_atom[s:e])
                    st = np.asarray(states_ma.filled(), dtype=np.uint8)
                    if st.shape[0] != e - s:
                        raise ValueError(f'frame {p}: {st.shape[0]} tokens for {e - s} residues')
                    tokens[s:e, p] = st
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                if src == 'ideal': ideal_pids.append(pid)   # G7 (2026-10-17)
            out[pid] = tokens
        except Exception as e:
            failed[pid] = f'{type(e).__name__}: {e}'
            if len(failed) < 5:
                print(f'  {pid}: FAIL {failed[pid]}', flush=True)
            continue
        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(pids)} ({time.time()-t0:.0f}s, '
                  f'real_frames={n_real} ideal_frames={n_ideal} '
                  f'chains={n_chains_total})', flush=True)
    print(f'{len(out)} ok, {len(failed)} failed, '
          f'real_frames={n_real} ideal_frames={n_ideal}, chains={n_chains_total}')
    if not check_builder_failures(failed, pids, args.dataset, args.max_fail_frac):
        print('[error] failures inside the RMSF residue set; nothing written', flush=True)
        return 1
    np.savez(out_path, **out)
    # FIXED(preretrain_audit_r2 F1, 2026-10-16): provenance sidecar (old: none).
    inputs = dataset_input_record(args.dataset)
    if args.dataset == 'misato':
        inputs['coords'] = str(COORDS)
    write_cache_provenance(out_path, dataset=args.dataset,
                           builder=f'{Path(__file__).name} {BUILDER_TAG}', inputs=inputs,
                           n_proteins=len(out), frames=f'all {K}',
                           **split_record(args.dataset),
                           sequence_source=SEQ_NOT_USED, failed=failed, n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids)))
    print(f'wrote {out_path.name} (+ .provenance.json)')
    return 0


if __name__ == '__main__': sys.exit(main())
