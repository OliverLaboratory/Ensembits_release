"""Single-frame mini3di (Foldseek 3Di) tokens and 10-D features for MISATO.

What it does
------------
For every pid of `coords_ca5_misato.pkl` it takes frame 0, loads the real
N / C / C-beta of that frame from `misato_real_bb`
(`ensembits.baselines._real_or_ideal_bb.load_bb_for_frame`; ideal geometry
only if no real frame matches; glycine C-beta from mini3di's own helper),
cuts the protein into its true chains (MISATO: `misato_chain_breaks.json`, cut at
boundary / unresolved breaks, whole across missing-residue gaps; mdCATH / BioEmu: the
whole protein; `_chain_pieces.py`, 2026-10-18 option A) and runs mini3di per chain:
`feature_encoder.encode_atoms(...).data` gives the 10-D descriptor and
`encode_atoms(...).filled()` the token in [0, 20) (terminal residues, which
mini3di masks, become the invalid state 2).

Paper
-----
`appendix.tex` \\label{app:baselines}, *3Di_tokens* ("applied to the single
representative conformation", frozen mini3di centroids). Results:
  * `2_results.tex` \\label{result:rmsf}, Table \\label{tab:rmsf} (baseline rows,
    mdCATH-div and MISATO columns);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * `appendix.tex` \\label{app:downstreams} (Tables \\label{tab:ec_d1_structure},
    \\label{tab:go_structure}, \\label{tab:binding_structure},
    \\label{tab:affinity_structure}) and \\label{app:experiments} (all splits).

Origin
------
Copied from `submission_exp/src/build_mini3di_misato.py`. Changed: `ROOT` from
`ensembits.paths.REPO_ROOT`; backbone helpers from `ensembits.baselines` (same
code); a CLI with `--allow-old-parser` and a MISATO parser check (code_fix C5;
the original had no flags); comments and this docstring. Logic unchanged.
2026-10-16: exit 1 on an existing output unless `--overwrite` and on an
ideal-geometry MISATO frame unless `--allow-ideal` (code_fix P15); chains split
at 2.5 Å, real backbones only (code_fix P5; replaced 2026-10-18, see below).
2026-10-18 (authors' decision, baseline fidelity audit): pieces = true chains only
(MISATO: true chains of misato_chain_breaks.json; mdCATH / BioEmu whole), never the
2.5 Å C–N rule; the sidecars
record `chain_split` (rmsf_common.chain_split_for) instead of chain_break_A.

Phase 2 (2026-10-16): `--dataset bioemu` uses frame 0 of bioemu_cath_real_bb bb_10 and
the real Cβ of bioemu_cath_extras -> `mini3di_{features,tokens}_bioemu.npz`.
2026-10-16 (preretrain_audit_r2 F1 / F2 / F7): `--dataset mdcath` (port of the research
`build_mini3di_features.py`: frame 0, real backbone of mdcath_real_bb, whole domain)
-> `mini3di_{features,tokens}_mdcath.npz`; every output gets `<output>.provenance.json`
(rmsf_common.write_cache_provenance; the RMSF probes refuse caches without it); a
protein of the RMSF residue set that fails makes the build exit 1 (`--max-fail-frac`).

Used by the reported model: no (3Di_tokens baseline rows).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/coords_ca5_misato.pkl   (data_prep/build_coords_ca5_misato.py)
          data/cached_descriptors/misato_real_bb/<pid>.npz
          (--dataset mdcath: data/external/mdcath_div ca_10 + mdcath_real_bb;
           --dataset bioemu: bioemu_cath_real_bb + bioemu_cath_extras)
  writes  data/cached_descriptors/mini3di_features_<dataset>.npz   pid -> (L, 10) float32
          data/cached_descriptors/mini3di_tokens_<dataset>.npz     pid -> (L,) uint8
          <each output>.provenance.json

How to run
----------
    python data_prep/baselines/build_mini3di_misato.py [--dataset misato|mdcath|bioemu] \\
        [--allow-old-parser] [--overwrite] [--allow-ideal] [--max-fail-frac 0]
    (needs the `mini3di` package)
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
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
COORDS = ROOT / 'data' / 'cached_descriptors' / 'coords_ca5_misato.pkl'
OUT_FEAT = ROOT / 'data' / 'cached_descriptors' / 'mini3di_features_misato.npz'
OUT_TOK = ROOT / 'data' / 'cached_descriptors' / 'mini3di_tokens_misato.npz'


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
    # Phase 2 (2026-10-16): --dataset bioemu = frame 0 of bioemu_cath_real_bb bb_10 (real
    #   N/CA/C), real Cβ from bioemu_cath_extras; outputs mini3di_{features,tokens}_bioemu.npz.
    # FIXED(preretrain_audit_r2 F2 / Q29d, 2026-10-16): new --dataset mdcath = the research
    #   builder submission_exp/src/build_mini3di_features.py ported: frame 0 of the
    #   mdCATH-div ca_10 (= ca_5[0]) with the real N / C / Cβ of mdcath_real_bb, the whole
    #   domain (2026-10-18; was a 2.5 Å split, a no-op on these domains); domains without a real backbone (the 8
    #   Cα-only domains before data_prep/build_mdcath_real_bb.py, 2026-10-17) are not written.
    ap.add_argument('--dataset', default='misato', choices=('misato', 'mdcath', 'bioemu'))
    # FIXED(preretrain_audit_r2 F7, 2026-10-16): new: failures inside the RMSF set fail.
    ap.add_argument('--max-fail-frac', type=float, default=0.0,
                    help='allowed fraction of RMSF-set proteins that fail (default 0)')
    args = ap.parse_args(argv)
    out_feat = OUT_FEAT.with_name(f'mini3di_features_{args.dataset}.npz')
    out_tok = OUT_TOK.with_name(f'mini3di_tokens_{args.dataset}.npz')
    if args.dataset == 'misato':
        # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
        require_misato_parser(MISATO_REAL_BB, allow_old=args.allow_old_parser)
    # FIXED(code_fix E0, 2026-10-20): MISATO caches rebuilt with the fixed parser (CX proline Cα).
    # FIXED(code_fix P15, 2026-10-16): old: returned 0 when both outputs existed.
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs, refuse_ideal_misato_frame
    refuse_existing_outputs([out_feat, out_tok], args.overwrite)
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
    print(f'[mini3di-{args.dataset}] {len(pids)} proteins', flush=True)
    feats: dict = {}
    toks: dict = {}
    failed: dict = {}
    ideal_pids: list = []
    n_real = n_ideal = 0
    n_chains_total = 0
    t0 = time.time()
    for i, pid in enumerate(pids):
        try:
            ca5 = coords[pid]
            ca = np.asarray(ca5[0], dtype=np.float32)
            bb, cb_real, src = load_bb_for_frame(args.dataset, pid, 0, ca)
            refuse_ideal_misato_frame(args.dataset, pid, 0, src, args.allow_ideal)   # code_fix P15
            if args.dataset == 'mdcath' and src != 'real':
                raise ValueError('no real backbone (C-alpha only)')
            cb = cb_from_bb_or_real(bb, cb_real)
            n_atom = bb[:, 0].astype(np.float32)
            c_atom = bb[:, 2].astype(np.float32)
            # Tokenise per-chain — mini3di was trained on single chains
            # and treats input as one continuous chain otherwise; without
            # this, chain-boundary residues see a fictitious 40+ Å
            # peptide bond and emit nonsense descriptors.
            # FIXED(baseline fidelity audit, 2026-10-18): one piece per true chain (MISATO:
            #   true chains of misato_chain_breaks.json (option A: a chain_index is a molecule;
            #   cut at boundary / unresolved breaks, whole across missing gaps); mdCATH /
            #   BioEmu: the whole protein). Old (code_fix P5): split at C–N > 2.5 Å on real
            #   backbones, which also cut chains at internal gaps.
            chunks, _ = chain_pieces(args.dataset, pid, len(bb))   # mini3di: no numbering
            n_chains_total += len(chunks)
            f10_parts = []
            tok_parts = []
            for s, e in chunks:
                ca_c = ca[s:e]
                cb_c = cb[s:e].astype(np.float32)
                n_c = n_atom[s:e]
                c_c = c_atom[s:e]
                desc = enc.feature_encoder.encode_atoms(ca_c, cb_c, n_c, c_c)
                f10_parts.append(np.asarray(desc.data, dtype=np.float32))
                states_ma = enc.encode_atoms(ca_c, cb_c, n_c, c_c)
                tok_parts.append(np.asarray(states_ma.filled(), dtype=np.uint8))
            f10 = np.concatenate(f10_parts, axis=0)
            tok = np.concatenate(tok_parts, axis=0)
            if f10.shape[0] != ca.shape[0] or tok.shape[0] != ca.shape[0]:
                raise ValueError(f'{tok.shape[0]} tokens for {ca.shape[0]} residues')
        except Exception as e:
            failed[pid] = f'{type(e).__name__}: {e}'
            if len(failed) < 5:
                print(f'  {pid}: FAIL {failed[pid]}', flush=True)
            continue
        feats[pid] = f10
        toks[pid] = tok
        n_real += (src == 'real'); n_ideal += (src == 'ideal')
        if src == 'ideal': ideal_pids.append(pid)   # G7 (2026-10-17)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(pids)} ({time.time()-t0:.0f}s, '
                  f'real={n_real} ideal={n_ideal} '
                  f'chains={n_chains_total})', flush=True)
    print(f'[mini3di-{args.dataset}] {len(feats)} ok, {len(failed)} failed, '
          f'real={n_real} ideal={n_ideal}, total chains={n_chains_total}',
          flush=True)
    if not check_builder_failures(failed, pids, args.dataset, args.max_fail_frac):
        print('[error] failures inside the RMSF residue set; nothing written', flush=True)
        return 1
    np.savez(out_feat, **feats)
    np.savez(out_tok, **toks)
    # FIXED(preretrain_audit_r2 F1, 2026-10-16): provenance sidecars (old: none).
    inputs = dataset_input_record(args.dataset)
    if args.dataset == 'misato':
        inputs['coords'] = str(COORDS)
    for out in (out_feat, out_tok):
        write_cache_provenance(out, dataset=args.dataset,
                               builder=f'{Path(__file__).name} {BUILDER_TAG}',
                               inputs=inputs, n_proteins=len(feats), frames='frame 0',
                               **split_record(args.dataset),
                               sequence_source=SEQ_NOT_USED, failed=failed,
                               n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids)))
    print(f'  wrote {out_feat.name}, {out_tok.name} (+ .provenance.json)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
