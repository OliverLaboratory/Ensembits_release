"""Tokenize MISATO and mdCATH-div (frame 0) with ProToken-1.0.

What it does
------------
Loads ProToken through `ensembits.baselines.ProToken` (TF SavedModel + the
ProToken source tree at `ensembits.paths.PROTOKEN_SRC`) and saves its (512, 32)
codebook. Per MISATO protein: frame 0 of `coords_ca8_misato.pkl`, the sequence
`seq` stored in `misato_real_bb/<pid>.npz` (or, if missing, from
`aa2ensembits_pairs_misato_v2.pkl`; the order was the reverse before
2026-10-16, code_fix P30), the real N / CA / C / O of that frame from
`misato_real_bb` (`load_bb_for_frame`; ideal geometry only if no real frame
matches), tokenized chain by chain (ProToken assumes one chain per PDB).
Per mdCATH-div protein: frame 0 of `ca_10` and `mdcath_real_bb`, sequence from
`mdcath_real_bb/<pid>.npz` with 'X' -> 'H' (since 2026-10-16; was the pairs pickle,
with poly-alanine for 645 domains); domains without a backbone file are not written.
Per BioEmu-CATH system (`--datasets bioemu`, phase 2): frame 0 of
`bioemu_cath_real_bb` bb_10, sequence from `bioemu_cath_extras`.
A protein whose token count differs from its residue count is a failure; failures of
proteins in the RMSF residue set make the dataset fail (exit 1, nothing written;
`--max-fail-frac`); the sidecar `<output>.provenance.json` lists every failure. `--shard k --num-shards n` writes shard
files to `protoken_shards/`; merge them with `merge_protoken_shards.py`.

(The original docstring said MISATO only has C-alpha and always uses the ideal
fallback; the code loads the real backbone. The mdCATH h5 helpers
`_parse_backbone_idx` / `_load_mdcath_bb_for_frame` are defined but not called
by `main`; they are kept unchanged.)

Paper
-----
`appendix.tex` \\label{app:baselines}, *ProToken* (minimal N / CA / C / O
backbone, VQ index in [0, 512), frozen 32-D codebook). Results:
  * `2_results.tex` \\label{result:rmsf}, Table \\label{tab:rmsf} (baseline rows,
    mdCATH-div and MISATO columns);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * `appendix.tex` \\label{app:downstreams} (Tables \\label{tab:ec_d1_structure},
    \\label{tab:go_structure}, \\label{tab:binding_structure},
    \\label{tab:affinity_structure}) and \\label{app:experiments} (all splits).

Origin
------
Copied from `submission_exp/src/build_protoken_tokens.py`. Changed:
`PROTOKEN_DIR` from `ensembits.paths.PROTOKEN_SRC` (was an absolute path),
`MDCATH_DIV` from `ensembits.paths.MDCATH_DIV_SRC` (was another user's
absolute path); `ROOT` from `ensembits.paths.REPO_ROOT`; `ProToken` and the
backbone helpers from `ensembits.baselines` (were `utils.baselines` /
`submission_exp/src/_real_or_ideal_bb.py`, same code), so the sys.path entries
`REPO_ROOT` and `REPO_ROOT/Ensembits` were dropped; `REAL_BB_DIR` uses `ROOT`
(was `Path(__file__).parents[2]`, the same folder in the research repo); new
flag `--allow-old-parser` and a MISATO parser check (code_fix C5); comments and
this docstring. Logic unchanged.
2026-10-16: exit 1 on an existing output unless `--overwrite` and on an
ideal-geometry MISATO frame unless `--allow-ideal` (code_fix P15); chains split
at 2.5 Å, real backbones only (code_fix P5).
MISATO sequence from `misato_real_bb` first (code_fix P30).
2026-10-18 (authors' decision, baseline fidelity audit): the builder passes the true
chains as pieces (MISATO: the true chains of `misato_chain_breaks.json`, cut at boundary /
unresolved breaks, whole across missing-residue gaps, with the gap-aware residue
index passed to the encoder (option A, 2026-10-18; a MISATO `chain_index` is a
molecule that can hold several chains); mdCATH / BioEmu: the whole protein;
`_chain_pieces.py`), never the 2.5 Å C–N rule; the sidecars record `chain_split`
("true_chain_v2" + the chain-break table sha256 for MISATO) instead of chain_break_A.
The gap-aware residue index + 1 is written as the residue number of ProToken's PDB.
2026-10-16 (preretrain_audit_r2 Q29 / Q29b): mdCATH sequences from `mdcath_real_bb`
with His normalised (no poly-Ala, no pairs pickle); token-count mismatches are errors
(were token 0 / a wrong-length array); `<output>.provenance.json`; `--datasets bioemu`.
Later the same day (preretrain_audit_r2 F1 / F7): the 0.5% mismatch tolerance became the
RMSF-set rule (default 0, `--max-fail-frac`); the sidecars follow the common baseline-cache
schema (rmsf_common.write_cache_provenance, codebook sha256, sequence source) that the
RMSF probes require; MISATO sequences must have the structure's length (no `seq[:L]`).

Used by the reported model: no (ProToken baseline rows).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/protoken/full/                                    (PROTOKEN_SRC)
          data/cached_descriptors/coords_ca8_misato.pkl
          data/cached_descriptors/aa2ensembits_pairs_misato_v2.pkl
          data/cached_descriptors/misato_real_bb/<pid>.npz
          data/cached_descriptors/mdcath_real_bb/<pid>.npz
          data/external/mdcath_div/{_manifest.json,<pid>.npz}   (MDCATH_DIV_SRC)
  writes  data/cached_descriptors/protoken_codebook.npy            (512, 32) float32
          data/cached_descriptors/protoken_tokens_misato.npz       pid -> (L,) uint16
          data/cached_descriptors/protoken_tokens_mdcath.npz       pid -> (L,) uint16
          data/cached_descriptors/protoken_tokens_bioemu.npz       system -> (L,) uint16
          <each output>.provenance.json   (written, skipped, mismatched pids)
          data/cached_descriptors/protoken_shards/{misato,mdcath}_shard<k>of<n>.npz
                                                                   (with --num-shards > 1)

How to run
----------
    python data_prep/baselines/build_protoken_tokens.py --device 0 \\
        [--datasets misato,mdcath,bioemu] [--shard k --num-shards n] [--allow-old-parser] \
        [--overwrite] [--allow-ideal]
    (every requested dataset's output / shard must not exist: pass --datasets
    misato to rebuild only MISATO, or --overwrite)
    # MISATO with shards: ONE GPU PER SHARD, chosen by CUDA_VISIBLE_DEVICES=k, with
    # --device "" (so the builder does not overwrite CUDA_VISIBLE_DEVICES):
    for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k python data_prep/baselines/build_protoken_tokens.py \\
        --datasets misato --shard $k --num-shards 4 --device "" & done; wait
    python data_prep/baselines/merge_protoken_shards.py --datasets misato --num-shards 4
"""
from __future__ import annotations
import os, sys, json, pickle, time
from pathlib import Path

# Suppress TF spew
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.paths import PROTOKEN_SRC, MDCATH_DIV_SRC   # noqa: E402
from ensembits.data_versions import require_misato_parser   # noqa: E402
sys.path.insert(0, str(CODE_ROOT / 'downstream'))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _chain_pieces (2026-10-18)
BUILDER_TAG = 'v2026-10-18'   # recorded in the cache sidecars
# Force-field residue names and the mdCATH real-backbone sequence (preretrain_audit_r2 Q29).
from rmsf_common import load_mdcath_real_bb_seq, three_to_one   # noqa: E402
# ProToken-1.0 source tree (data_process/, codebook, models); was an absolute path.
PROTOKEN_DIR = str(PROTOKEN_SRC)

CACHE = ROOT / 'data' / 'cached_descriptors'
MISATO_COORDS = CACHE / 'coords_ca8_misato.pkl'
MISATO_PAIRS = CACHE / 'aa2ensembits_pairs_misato_v2.pkl'
MDCATH_PAIRS = CACHE / 'aa2ensembits_pairs_mdcath_v2.pkl'
MISATO_REAL_BB = CACHE / 'misato_real_bb'       # read through _real_or_ideal_bb
MDCATH_DIV = str(MDCATH_DIV_SRC)
MDCATH_H5_DIR = ROOT / 'data' / 'mdcath' / 'data'

OUT_MISATO = CACHE / 'protoken_tokens_misato.npz'
OUT_MDCATH = CACHE / 'protoken_tokens_mdcath.npz'
OUT_BIOEMU = CACHE / 'protoken_tokens_bioemu.npz'   # phase 2 (2026-10-16): --datasets bioemu
OUT_CB = CACHE / 'protoken_codebook.npy'
SHARD_DIR = CACHE / 'protoken_shards'


class _TokenCountMismatch(Exception):
    """ProToken returned a different number of tokens than input residues."""


def tokenize_by_chain(tok, ca, seq, bb, src, pieces=None, residue_index=None):
    """ProToken tokens of one structure, chain by chain (ProToken's preprocess assumes one
    chain per PDB): chains split at C–N > 2.5 Å on real backbones (code_fix P5); an ideal
    frame stays one chain. A chain whose token count differs from its length raises
    _TokenCountMismatch (FIXED preretrain_audit_r2 Q29b: was token 0 for that chain).
    2026-10-16 (H4): factored out of the MISATO loop and used for mdCATH-div and BioEmu too
    (they were tokenized as one chain).
    FIXED(baseline fidelity audit, 2026-10-18): the builders pass `pieces` = the true
    chains (`_chain_pieces.chain_pieces`: MISATO true chains of misato_chain_breaks.json,
    whole across missing gaps; mdCATH / BioEmu whole); the 2.5 Å split remains only for
    callers without. `residue_index` (option A): (L,) residue number within each piece;
    residue_index + 1 is written as the PDB residue number ProToken reads (default 1..n).
    """
    ri = None if residue_index is None else np.asarray(residue_index, dtype=np.int64)
    num = (lambda s, e: None) if ri is None else (lambda s, e: ri[s:e] + 1)
    if pieces is not None:
        from _chain_pieces import check_pieces
        chunks = check_pieces(pieces, ca.shape[0])
    else:
        from ensembits.baselines._real_or_ideal_bb import split_chains_by_peptide_bond
        chunks = split_chains_by_peptide_bond(bb) if src == 'real' else [(0, len(bb))]
    if len(chunks) == 1:
        idx = np.asarray(tok.tokenize(ca, seq, bb=bb, residue_numbers=num(0, ca.shape[0])))
    else:
        idx = np.empty(ca.shape[0], dtype=np.int64)
        for s, e in chunks:
            if e - s < 1: continue
            idx_chunk = np.asarray(tok.tokenize(ca[s:e], seq[s:e], bb=bb[s:e],
                                                residue_numbers=num(s, e)))
            if idx_chunk.shape[0] != (e - s):
                raise _TokenCountMismatch(f'chain {s}-{e}: {idx_chunk.shape[0]} tokens')
            idx[s:e] = idx_chunk
    if len(idx) != ca.shape[0]:
        raise _TokenCountMismatch(f'{len(idx)} tokens for {ca.shape[0]} residues')
    return idx


def _pieces_kw(dataset, pid, L):
    """{'pieces', 'residue_index'} of one protein for the tokenize call (2026-10-18, option A:
    true chains, gap-aware residue index; `_chain_pieces.chain_pieces`)."""
    from _chain_pieces import chain_pieces
    pieces, ri = chain_pieces(dataset, pid, L)
    return {'pieces': pieces, 'residue_index': ri}


# ---- mdCATH all-atom helpers -------------------------------------------

def _parse_backbone_idx(pdb_atoms_str: str) -> dict:
    """Parse mdCATH `pdbProteinAtoms` → per-residue {N,CA,C,O} atom indices.

    The mdCATH PDB string includes CHARMM-style caps (CAY/CY/OY/HY*) and
    explicit hydrogens. We index ATOM records (matching `coords` dim 1)
    and pick the four backbone atoms per residue. Returns:
        {res_id: {"N": int, "CA": int, "C": int, "O": int, "resname": str}}
    Residues missing any of the four are dropped.
    """
    bb = {}  # resid -> dict
    atom_i = 0
    for line in pdb_atoms_str.split("\n"):
        if not line.startswith("ATOM"):
            continue
        aname = line[12:16].strip()
        try:
            resid = int(line[22:26].strip())
        except ValueError:
            atom_i += 1
            continue
        resname = line[17:21].strip()
        if aname in ("N", "CA", "C", "O"):
            entry = bb.setdefault(resid, {"resname": resname})
            entry[aname] = atom_i
        atom_i += 1
    # Keep only complete residues (all four atoms present)
    return {r: v for r, v in bb.items()
            if all(k in v for k in ("N", "CA", "C", "O"))}


def _load_mdcath_bb_for_frame(h5_path, target_ca, stride: int = 10):
    """Open mdCATH h5, find frame whose Cα coords match `target_ca`, and
    return its all-atom backbone `(L, 4, 3)` plus the 1-letter sequence.

    Walks 320K replicas at the same stride used by `build_mdcath_div.py`
    so the candidate set is identical. Picks the closest frame (by
    Frobenius distance on Cα coords) — the FPS-selected frame is in
    that set, so the closest match is the FPS frame itself (distance ≈ 0).
    """
    import h5py

    target_ca = np.asarray(target_ca, dtype=np.float32)
    L = target_ca.shape[0]
    domain = Path(h5_path).stem.replace('mdcath_dataset_', '')

    with h5py.File(h5_path, 'r') as f:
        g = f[domain]
        patoms = g['pdbProteinAtoms'][()]
        if isinstance(patoms, bytes):
            patoms = patoms.decode()
        bb_idx = _parse_backbone_idx(patoms)
        if len(bb_idx) != L:
            raise ValueError(
                f'{domain}: backbone-residue count {len(bb_idx)} != '
                f'target Cα length {L} (cap residues likely)')

        # Sequence: use resname order (sorted by resid)
        # FIXED(preretrain_audit_r2 Q29, 2026-10-16): old = ProToken's restype_3to1, which
        #   maps CHARMM HSD/HSE/HSP to X; new = rmsf_common.three_to_one (force-field names
        #   -> H / C / D / E / K). (This helper is not called by main.)
        ordered = sorted(bb_idx.items())
        seq = ''.join(three_to_one(v['resname']) for _, v in ordered)

        ca_idx = np.array([v['CA'] for _, v in ordered], dtype=np.int64)
        n_idx = np.array([v['N'] for _, v in ordered], dtype=np.int64)
        c_idx = np.array([v['C'] for _, v in ordered], dtype=np.int64)
        o_idx = np.array([v['O'] for _, v in ordered], dtype=np.int64)

        if '320' not in g:
            raise KeyError(f'{domain}: no 320K data')

        best_d = np.inf
        best_bb = None
        for rep in sorted(g['320'].keys()):
            coords = g['320'][rep]['coords']
            T = coords.shape[0]
            for t in range(0, T, stride):
                frame = coords[t]                                  # (N_atoms, 3)
                ca_frame = frame[ca_idx]                           # (L, 3)
                d = float(np.linalg.norm(ca_frame - target_ca))
                if d < best_d:
                    best_d = d
                    best_bb = np.stack([frame[n_idx], ca_frame,
                                        frame[c_idx], frame[o_idx]],
                                       axis=1).astype(np.float32)
                    if d < 1e-3:
                        return best_bb, seq, best_d
        if best_bb is None:
            raise RuntimeError(f'{domain}: no candidate frames found')
        return best_bb, seq, best_d


# ---- main --------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='0',
                    help='CUDA_VISIBLE_DEVICES value (or empty for CPU)')
    ap.add_argument('--datasets', default='misato,mdcath')
    ap.add_argument('--shard', type=int, default=0,
                    help='shard index (0-indexed)')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--mdcath-stride', type=int, default=10,
                    help='Trajectory stride to match build_mdcath_div.py.')
    ap.add_argument('--max-frame-mismatch', type=float, default=2.0,
                    help='Max ‖best_ca − target_ca‖_F to accept (Å) when '
                         'matching the FPS frame in the h5.')
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
    # FIXED(preretrain_audit_r2 F7, 2026-10-16): new: failures inside the RMSF set fail
    #   (replaces the 0.5% token-count-mismatch tolerance of earlier the same day).
    ap.add_argument('--max-fail-frac', type=float, default=0.0,
                    help='allowed fraction of RMSF-set proteins that fail (default 0)')
    args = ap.parse_args()
    # FIXED(preretrain_audit_r2 H8, 2026-10-16): --device takes a CUDA_VISIBLE_DEVICES value
    #   ("0"); the torch form "cuda:0" is accepted too (it was passed through verbatim).
    if args.device and args.device.startswith('cuda:'):
        args.device = args.device.split(':', 1)[1]
    from rmsf_common import finish_cache, SEQUENCE_SOURCES, SEQ_NOT_USED, split_record
    bname = f'{Path(__file__).name} {BUILDER_TAG}'
    if 'misato' in args.datasets.split(','):
        # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
        require_misato_parser(MISATO_REAL_BB, allow_old=args.allow_old_parser)
    # code_fix P15: every requested output (or shard) must be absent, checked before
    # ProToken is loaded. Same names as below.
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs, refuse_ideal_misato_frame
    _sharded = args.num_shards > 1
    _outs = {'misato': (SHARD_DIR / f'misato_shard{args.shard}of{args.num_shards}.npz'
                        if _sharded else OUT_MISATO),
             'mdcath': (SHARD_DIR / f'mdcath_shard{args.shard}of{args.num_shards}.npz'
                        if _sharded else OUT_MDCATH)}
    # Phase 2 (2026-10-16): BioEmu-CATH (unsharded; 1,090 systems).
    if 'bioemu' in args.datasets.split(','):
        if _sharded:
            print('[error] --datasets bioemu is not sharded; run it with --num-shards 1'); return 1
        _outs['bioemu'] = OUT_BIOEMU
    refuse_existing_outputs([_outs[d] for d in args.datasets.split(',') if d in _outs],
                            args.overwrite)

    if args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # TF/JAX setup happens inside the wrapper.

    from ensembits.baselines import ProToken
    sys.path.insert(0, PROTOKEN_DIR)
    sys.path.insert(0, PROTOKEN_DIR + '/data_process')

    tok = ProToken(repo_path=PROTOKEN_DIR, device=args.device or '')
    # FIXED(preretrain_audit_r2 H8, 2026-10-16): only shard 0 (or an unsharded run) writes the
    #   codebook, as in build_esm3struct_tokens.py, so parallel shards never read a file that
    #   another shard is rewriting (their sidecars record its sha256).
    if args.shard == 0:
        # G12 (2026-10-17): atomic write; an identical codebook is kept.
        from rmsf_common import write_codebook
        if write_codebook(OUT_CB, tok.codebook.astype(np.float32)):
            print(f'  wrote codebook {tok.codebook.shape} → {OUT_CB.name}', flush=True)

    datasets = args.datasets.split(',')

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    is_sharded = args.num_shards > 1
    out_misato = (SHARD_DIR / f'misato_shard{args.shard}of{args.num_shards}.npz'
                  if is_sharded else OUT_MISATO)
    out_mdcath = (SHARD_DIR / f'mdcath_shard{args.shard}of{args.num_shards}.npz'
                  if is_sharded else OUT_MDCATH)

    from ensembits.baselines._real_or_ideal_bb import load_bb_for_frame

    # ===== Misato (real bb if available, else ideal-geometry fallback) =====
    # FIXED(code_fix E0, 2026-10-20): MISATO caches rebuilt with the fixed parser (CX proline Cα).
    # FIXED(code_fix P15, 2026-10-16): old: `and not out_misato.exists()` (silent skip).
    if 'misato' in datasets:
        print(f'[misato] tokenizing (shard {args.shard}/{args.num_shards})...',
              flush=True)
        coords = pickle.load(open(MISATO_COORDS, 'rb'))
        # FIXED(code_fix P30, 2026-10-16): the pairs pickle is only the fallback now and
        #   may be absent (old: loaded unconditionally).
        pairs = pickle.load(open(MISATO_PAIRS, 'rb')) if MISATO_PAIRS.exists() else {}
        seq_by_pid = {}
        for k, v in pairs.items():
            pid = k.split(':')[-1] if ':' in k else k
            seq_by_pid[pid] = v.get('seq', '')
        # Primary seq source since 2026-10-16 (code_fix P30): per-pid sequence stored in
        # the misato_real_bb cache (MISATO residue names, fixed parser). Original note
        # (old parser, heavy-atom-derived sequence): "Empirically agrees with
        # `aa2ensembits_pairs_misato_combined.pkl`'s seq at ~98.65% per-residue, with
        # most disagreements being GT's 'X' resolved here via heavy-atom signature."
        REAL_BB_DIR = (ROOT
                       / 'data' / 'cached_descriptors' / 'misato_real_bb')
        all_pids = sorted(coords.keys())
        my_pids = [p for j, p in enumerate(all_pids)
                   if j % args.num_shards == args.shard]
        out = {}; failed = {}; ideal_pids = []; n_real = n_ideal = 0
        n_seq_gt = n_seq_derived = 0
        t0 = time.time()
        for i, pid in enumerate(my_pids):
            # FIXED(code_fix P30, 2026-10-16): the sequence comes from misato_real_bb
            #   'seq' first (fixed parser, MISATO residue names, always length L), the old
            #   pairs pickle only as fallback. Old: the pairs pickle first.
            #   (n_seq_derived now counts real-bb sequences, n_seq_gt pickle fallbacks.)
            seq_full = ''
            bb_path = REAL_BB_DIR / f'{pid}.npz'
            if bb_path.exists():
                seq_full = str(np.load(bb_path, allow_pickle=False)['seq'])
            if seq_full:
                n_seq_derived += 1
            else:
                seq_full = seq_by_pid.get(pid, '')
                if not seq_full:
                    failed[pid] = 'no sequence'; continue
                n_seq_gt += 1
            try:
                ca = np.asarray(coords[pid][0], dtype=np.float32)
                # FIXED(preretrain_audit_r2 F7, 2026-10-16): old = seq_full[:L] (cut without
                #   an alignment check); new = the sequence must have the structure's length.
                seq = seq_full
                if len(seq) != ca.shape[0]:
                    failed[pid] = f'sequence {len(seq)} != structure {ca.shape[0]}'; continue
                bb, _, src = load_bb_for_frame('misato', pid, 0, ca)
                refuse_ideal_misato_frame('misato', pid, 0, src, args.allow_ideal)   # code_fix P15
                # Tokenise chain-by-chain — ProToken's preprocess
                # (`task_mode="single"`) explicitly assumes one chain
                # per PDB. Concatenated multi-chain inputs would emit
                # wrong tokens at chain boundaries via a fictitious
                # peptide bond.
                idx = tokenize_by_chain(tok, ca, seq, bb, src,      # true chains (2026-10-18)
                                        **_pieces_kw('misato', pid, ca.shape[0]))
                if len(idx) != ca.shape[0]:
                    raise _TokenCountMismatch(f'{len(idx)} tokens for {ca.shape[0]} residues')
                out[pid] = idx.astype(np.uint16)
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                if src == 'ideal': ideal_pids.append(pid)   # G7 (2026-10-17)
            except _TokenCountMismatch as e:
                failed[pid] = f'token-count mismatch: {e}'
                print(f'  {pid}: token-count mismatch ({e})', flush=True)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
            if (i+1) % 100 == 0:
                print(f'  misato {i+1}/{len(my_pids)} '
                      f'({time.time()-t0:.0f}s, ok={len(out)} '
                      f'gt_seq={n_seq_gt} derived_seq={n_seq_derived} '
                      f'real={n_real} ideal={n_ideal})', flush=True)
        print(f'misato: {len(out)} ok, {len(failed)} failed, '
              f'real={n_real} ideal={n_ideal}', flush=True)
        # FIXED(preretrain_audit_r2 F1 / F7, 2026-10-16): tolerance check + provenance sidecar
        #   (a pickle-fallback sequence makes the source "mixed", which the probes refuse).
        if not finish_cache(out_misato, out, failed, my_pids, 'misato', builder=bname,
                            max_fail_frac=args.max_fail_frac, sharded=is_sharded,
                            frames='frame 0', codebook=OUT_CB, **split_record('misato'),
                            # G6 (2026-10-17): ProToken's encoder has no amino-acid input, so
                            #   the sequence it is given is recorded for information only.
                            sequence_source=SEQ_NOT_USED,
                            sequence_given=(SEQUENCE_SOURCES['misato'] if not n_seq_gt else
                                            f'mixed ({n_seq_gt} from the pairs pickle)'),
                            n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
            return 1

    # ===== mdCATH (real bb from cache; ideal fallback if cache missing) =====
    # FIXED(code_fix P15, 2026-10-16): old: `and not out_mdcath.exists()` (silent skip).
    if 'mdcath' in datasets:
        print(f'[mdcath] tokenizing (shard {args.shard}/{args.num_shards})...',
              flush=True)
        with open(os.path.join(MDCATH_DIV, '_manifest.json')) as f:
            man = json.load(f)
        all_pids = list(man['representatives'].keys())
        pids = [p for j, p in enumerate(all_pids)
                if j % args.num_shards == args.shard]
        # FIXED(preretrain_audit_r2 Q29b, 2026-10-16): old = sequence from the pairs pickle,
        #   645 of 2,442 domains (no or wrong-length sequence) written as poly-alanine, His as
        #   UNK (the pickle / cache write HSD/HSE/HSP as X). New = the sequence of
        #   mdcath_real_bb/<pid>.npz with X -> H (rmsf_common.normalize_mdcath_real_bb_seq);
        #   domains without a backbone file (none since 2026-10-17; the 8 before were outside the common
        #   RMSF residue set) or with a wrong-length sequence are not written (counted).
        out = {}; failed = {}; ideal_pids = []; n_real = n_ideal = 0
        t0 = time.time()
        for i, pid in enumerate(pids):
            try:
                d = np.load(os.path.join(MDCATH_DIV, f'{pid}.npz'),
                             allow_pickle=False)
                ca = d['ca_10'][0].astype(np.float32)
                seq = load_mdcath_real_bb_seq(pid)
                if seq is None or len(seq) != ca.shape[0]:
                    failed[pid] = 'no real-backbone sequence of the right length'; continue
                bb, _, src = load_bb_for_frame('mdcath', pid, 0, ca)
                # FIXED(preretrain_audit_r2 G7, 2026-10-17): old = an ideal-geometry fallback was
                #   tokenized and written; new = a failure (counted; fails the build inside
                #   the RMSF set), as in the mini3di builders.
                if src != 'real':
                    raise ValueError('no real backbone (ideal-geometry fallback)')
                try:
                    idx = tokenize_by_chain(tok, ca, seq, bb, src,   # whole domain (2026-10-18)
                                            **_pieces_kw('mdcath', pid, ca.shape[0]))
                except _TokenCountMismatch as e:
                    failed[pid] = f'token-count mismatch: {e}'
                    print(f'  {pid}: {failed[pid]}', flush=True)
                    continue
                out[pid] = idx.astype(np.uint16)
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                if src == 'ideal': ideal_pids.append(pid)   # G7 (2026-10-17)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
            if (i+1) % 50 == 0:
                print(f'  mdcath {i+1}/{len(pids)} '
                      f'({time.time()-t0:.0f}s, ok={len(out)} '
                      f'real={n_real} ideal={n_ideal})', flush=True)
        print(f'mdcath: {len(out)} ok, {len(failed)} failed (a domain without a real backbone '
              f'real-backbone sequence; outside the RMSF set), real={n_real} ideal={n_ideal}',
              flush=True)
        if not finish_cache(out_mdcath, out, failed, pids, 'mdcath', builder=bname,
                            max_fail_frac=args.max_fail_frac, sharded=is_sharded,
                            frames='frame 0', codebook=OUT_CB, **split_record('mdcath'),
                            sequence_source=SEQ_NOT_USED,
                            sequence_given=SEQUENCE_SOURCES['mdcath_div'],   # G6: information only
                            n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
            return 1

    # ===== BioEmu-CATH (phase 2, 2026-10-16) =====
    # Frame 0 of bioemu_cath_real_bb bb_10 (real N/CA/C/O), sequence from bioemu_cath_extras
    # (data_prep/baselines/_bioemu_inputs.py); single-frame like the other datasets. A
    # failed system or a token count != residue count fails the dataset (every system is in
    # the common RMSF residue set).
    if 'bioemu' in datasets:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _bioemu_inputs as BI
        print('[bioemu] tokenizing...', flush=True)
        frames = BI.CaFrames()
        out = {}; failed = {}; ideal_pids = []
        for i, pid in enumerate(frames):
            try:
                ca = np.asarray(frames[pid][0], dtype=np.float32)    # frame 0
                bb, _, src = BI.load_bb_for_frame('bioemu', pid, 0, ca)
                idx = tokenize_by_chain(tok, ca, BI.sequence(pid), bb, src,   # whole (2026-10-18)
                                        **_pieces_kw('bioemu', pid, ca.shape[0]))
                out[pid] = np.asarray(idx).astype(np.uint16)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
        print(f'bioemu: {len(out)} ok, {len(failed)} failed', flush=True)
        if not finish_cache(OUT_BIOEMU, out, failed, list(frames), 'bioemu', builder=bname,
                            max_fail_frac=args.max_fail_frac, frames='frame 0', codebook=OUT_CB,
                            **split_record('bioemu'), sequence_source=SEQ_NOT_USED,
                            sequence_given=SEQUENCE_SOURCES['bioemu']):   # G6: information only
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
