"""Tokenize MISATO and mdCATH-div (frame 0) with the ESM3 structure tokenizer.

What it does
------------
Loads `ESM3_structure_encoder_v0` (esm3-sm-open-v1 StructureTokenEncoder) and
saves its frozen (4096, 128) codebook. For every MISATO protein it takes frame 0
of `coords_ca8_misato.pkl`, loads the real N / CA / C of that frame from
`misato_real_bb` (`ensembits.baselines._real_or_ideal_bb.load_bb_for_frame`;
ideal geometry only if no real frame matches) and encodes it chain by chain
(atom37 layout, N / CA / C slots). mdCATH-div proteins use frame 0 of `ca_10`
and `mdcath_real_bb`. Tokens are stored as uint16 per pid. `--shard k
--num-shards n` processes every n-th pid and writes shard files instead.

The Hugging Face token is read from the environment (`HF_TOKEN`, copied to
`HUGGING_FACE_HUB_TOKEN`) or from the cached `huggingface-cli login`. The
original script hard-coded a personal access token here; it was removed.

Paper
-----
`appendix.tex` \\label{app:baselines}, *ESM3struct* (VQ index in [0, 4096),
frozen 128-D EMA codebook). Results:
  * `2_results.tex` \\label{result:rmsf}, Table \\label{tab:rmsf} (baseline rows,
    mdCATH-div and MISATO columns);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * `appendix.tex` \\label{app:downstreams} (Tables \\label{tab:ec_d1_structure},
    \\label{tab:go_structure}, \\label{tab:binding_structure},
    \\label{tab:affinity_structure}) and \\label{app:experiments} (all splits).

Origin
------
Copied from `submission_exp/src/build_esm3struct_tokens.py`. Changed: the
hard-coded HF token was replaced by the environment / cached login (see above);
`ROOT` from `ensembits.paths.REPO_ROOT`; `MDCATH_DIV` from
`ensembits.paths.MDCATH_DIV_SRC` (was another user's absolute path); the
backbone helpers come from `ensembits.baselines` (were `utils.baselines` /
`submission_exp/src/_real_or_ideal_bb.py`, same code); new flag
`--allow-old-parser` and a MISATO parser check (code_fix C5); comments and this
docstring. Logic unchanged.
2026-10-16: exit 1 on an existing output unless `--overwrite` and on an
ideal-geometry MISATO frame unless `--allow-ideal` (code_fix P15); chains split
at 2.5 Å, real backbones only (code_fix P5).
`tokenize_protein` takes `ideal=` (default False) for that.
2026-10-18 (authors' decision, baseline fidelity audit): the builder passes the true
chains as pieces (MISATO: the true chains of `misato_chain_breaks.json`, cut at boundary /
unresolved breaks, whole across missing-residue gaps, with the gap-aware residue
index passed to the encoder (option A, 2026-10-18; a MISATO `chain_index` is a
molecule that can hold several chains); mdCATH / BioEmu: the whole protein;
`_chain_pieces.py`), never the 2.5 Å C–N rule; the sidecars record `chain_split`
("true_chain_v2" + the chain-break table sha256 for MISATO) instead of chain_break_A.
The gap-aware residue index is passed as `encode(coords, residue_index=)`.

Phase 2 (2026-10-16): `--datasets bioemu` tokenizes frame 0 of each BioEmu-CATH
system (bioemu_cath_real_bb bb_10, real backbone; data_prep/baselines/_bioemu_inputs.py)
-> `esm3struct_tokens_bioemu.npz` (unsharded; any failed system fails the dataset).

2026-10-16 (preretrain_audit_r2 F1 / F7): every token cache (and shard) gets
`<output>.provenance.json` (with the codebook's sha256); a token-count mismatch is a
failure, and failures of proteins in the RMSF residue set make the build exit 1
(`--max-fail-frac`).

Used by the reported model: no (ESM3struct baseline rows). `tokenize_protein`
is also imported by `build_pooled_baseline_features.py`.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/cached_descriptors/coords_ca8_misato.pkl
          data/cached_descriptors/misato_real_bb/<pid>.npz
          data/cached_descriptors/mdcath_real_bb/<pid>.npz
          data/external/mdcath_div/{_manifest.json,<pid>.npz}   (MDCATH_DIV_SRC)
  writes  data/cached_descriptors/esm3struct_codebook.npy          (4096, 128) float32
          data/cached_descriptors/esm3struct_tokens_misato.npz     pid -> (L,) uint16
          data/cached_descriptors/esm3struct_tokens_mdcath.npz     pid -> (L,) uint16
          data/cached_descriptors/esm3struct_shards/{misato,mdcath}_shard<k>of<n>.npz
                                                                   (with --num-shards > 1)

How to run
----------
    HF_TOKEN=... python data_prep/baselines/build_esm3struct_tokens.py --device 0 \\
        [--datasets misato,mdcath,bioemu] [--shard k --num-shards n] [--allow-old-parser] \
        [--overwrite] [--allow-ideal]
    (every requested dataset's output must not exist: pass --datasets misato to
    rebuild only MISATO, or --overwrite)
"""
from __future__ import annotations
import os, sys, json, pickle, time
from pathlib import Path

# HF token must be set before any esm import. It is read from the environment
# (HF_TOKEN) or the cached `huggingface-cli login`; the original hard-coded a
# personal token here (removed).
if os.environ.get('HF_TOKEN'):
    os.environ.setdefault('HUGGING_FACE_HUB_TOKEN', os.environ['HF_TOKEN'])

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))   # rmsf_common (sidecars, tolerances)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _chain_pieces (2026-10-18)
BUILDER_TAG = 'v2026-10-18'   # recorded in the cache sidecars
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.paths import MDCATH_DIV_SRC      # noqa: E402  data/external/mdcath_div
from ensembits.data_versions import require_misato_parser   # noqa: E402
CACHE = ROOT / 'data' / 'cached_descriptors'
MISATO_COORDS = CACHE / 'coords_ca8_misato.pkl'
MISATO_REAL_BB = CACHE / 'misato_real_bb'       # read through _real_or_ideal_bb
MDCATH_DIV = str(MDCATH_DIV_SRC)

OUT_MISATO = CACHE / 'esm3struct_tokens_misato.npz'
OUT_MDCATH = CACHE / 'esm3struct_tokens_mdcath.npz'
OUT_BIOEMU = CACHE / 'esm3struct_tokens_bioemu.npz'   # phase 2 (2026-10-16): --datasets bioemu
OUT_CB = CACHE / 'esm3struct_codebook.npy'
SHARD_DIR = CACHE / 'esm3struct_shards'


def reconstruct_backbone_3atoms(ca: np.ndarray):
    """Cα → ideal-geometry (N, CA, C). (L, 3, 3) [N, CA, C].

    Uses the shared `reconstruct_backbone_ideal` helper so the encoder's
    rigid frames aren't degenerate.
    """
    from ensembits.baselines._backbone import reconstruct_backbone_ideal
    n, c, _ = reconstruct_backbone_ideal(ca)
    return np.stack([n, ca, c], axis=1).astype(np.float32)


@torch.no_grad()
def _encode_chain_esm3(enc, ca_c, bb_c, device, residue_index=None):
    # `residue_index` (2026-10-18, option A): (L,) residue numbers of the segment, passed to
    #   enc.encode (None = ESM3's default, the positions 0..L-1).
    L = ca_c.shape[0]
    coords37 = np.zeros((L, 37, 3), dtype=np.float32)
    coords37[:, 0, :] = bb_c[:, 0]
    coords37[:, 1, :] = bb_c[:, 1]
    coords37[:, 2, :] = bb_c[:, 2]
    coords = torch.tensor(coords37, device=device).unsqueeze(0)
    if residue_index is None:
        _z, idx = enc.encode(coords)
    else:
        ri = torch.as_tensor(np.asarray(residue_index), device=device, dtype=torch.long)
        _z, idx = enc.encode(coords, residue_index=ri.unsqueeze(0))
    return idx.squeeze(0).cpu().numpy().astype(np.int64)


@torch.no_grad()
def tokenize_protein(enc, ca: np.ndarray, device: str,
                      bb: np.ndarray | None = None, ideal: bool = False,
                      pieces=None, residue_index=None) -> np.ndarray:
    """Cα (L, 3) → token ids (L,) int64 via ESM3 structure encoder.

    Tokenises chain-by-chain — ESM3's encoder uses a KNN graph from
    coords; concatenated multi-chain inputs put residues from
    different chains in each other's neighborhoods at chain boundaries
    via the fictitious peptide bond.

    `bb`: optional `(L, 4, 3)` real `[N, CA, C, O]`. When given, encoder
    reads real N/CA/C; else falls back to ideal-geometry reconstruction.
    `ideal`: True when `bb` itself was rebuilt with ideal geometry
    (`load_bb_for_frame` source 'ideal'); it is then not split.
    `pieces` (2026-10-18): the chains as [(start, end), ...] covering [0, L)
    (`_chain_pieces.chain_pieces`); the builders of this folder always pass them.
    Without `pieces` the old 2.5 Å split is used (other callers, unchanged).
    `residue_index` (2026-10-18, option A): (L,) residue number within each piece
    (`_chain_pieces.chain_pieces`), given to `enc.encode(..., residue_index=)`.
    """
    L = ca.shape[0]
    bb_real = bb is not None
    ri = None if residue_index is None else np.asarray(residue_index, dtype=np.int64)
    if ri is not None and ri.shape != (L,):
        raise ValueError(f'residue_index has shape {ri.shape}, expected ({L},)')
    if bb is None:
        bb3 = reconstruct_backbone_3atoms(ca)
        bb = np.zeros((L, 4, 3), dtype=np.float32)
        bb[:, :3, :] = bb3
    from ensembits.baselines._real_or_ideal_bb import split_chains_by_peptide_bond
    # FIXED(code_fix P5, 2026-10-16): chains split at C–N > 2.5 Å (was 5.0 Å). Only real
    #   backbones are split; an ideal-geometry backbone (`bb` None, or `ideal=True`) is one
    #   chain, as in the Ensembits descriptor (its C–N reaches 4.42 Å, never split at 5.0 Å).
    # FIXED(baseline fidelity audit, 2026-10-18): the builders pass `pieces` = the true
    #   chains (no cut at internal gaps); the 2.5 Å split remains only for callers without.
    if pieces is not None:
        from _chain_pieces import check_pieces
        chunks = check_pieces(pieces, L)
    else:
        chunks = split_chains_by_peptide_bond(bb) if (bb_real and not ideal) else [(0, L)]
    if len(chunks) == 1:
        return _encode_chain_esm3(enc, ca, bb, device, residue_index=ri)
    out = np.empty(L, dtype=np.int64)
    for s, e in chunks:
        out[s:e] = _encode_chain_esm3(enc, ca[s:e], bb[s:e], device,
                                      residue_index=None if ri is None else ri[s:e])
    return out


def _pieces_kw(dataset, pid, L):
    """{'pieces', 'residue_index'} of one protein for the tokenize call (2026-10-18, option A:
    true chains, gap-aware residue index; `_chain_pieces.chain_pieces`)."""
    from _chain_pieces import chain_pieces
    pieces, ri = chain_pieces(dataset, pid, L)
    return {'pieces': pieces, 'residue_index': ri}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='0')
    ap.add_argument('--datasets', default='misato,mdcath')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
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
    # FIXED(preretrain_audit_r2 F7, 2026-10-16): new: failures inside the RMSF set fail.
    ap.add_argument('--max-fail-frac', type=float, default=0.0,
                    help='allowed fraction of RMSF-set proteins that fail (default 0)')
    args = ap.parse_args()
    # FIXED(preretrain_audit_r2 H8, 2026-10-16): --device takes a CUDA_VISIBLE_DEVICES value
    #   ("0"); the torch form "cuda:0" is accepted too (it was passed through verbatim).
    if args.device and args.device.startswith('cuda:'):
        args.device = args.device.split(':', 1)[1]
    from rmsf_common import finish_cache, split_record, SEQ_NOT_USED   # H4: recorded in every sidecar
    bname = f'{Path(__file__).name} {BUILDER_TAG}'
    if 'misato' in args.datasets.split(','):
        # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
        require_misato_parser(MISATO_REAL_BB, allow_old=args.allow_old_parser)
    # code_fix P15: every requested output (or shard) must be absent, checked before the
    # model is loaded. Same names as below.
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
    device = 'cuda:0'

    print(f'[setup] loading ESM3 structure encoder on {device}...', flush=True)
    t0 = time.time()
    from esm.pretrained import ESM3_structure_encoder_v0
    enc = ESM3_structure_encoder_v0(device=device)
    enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    print(f'  loaded in {time.time()-t0:.0f}s', flush=True)

    cb = enc.codebook.embeddings.detach().cpu().numpy().astype(np.float32)
    if args.shard == 0:
        # G12 (2026-10-17): atomic write; an identical codebook is kept.
        from rmsf_common import write_codebook
        if write_codebook(OUT_CB, cb):
            print(f'  wrote codebook {cb.shape} → {OUT_CB.name}', flush=True)

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    is_sharded = args.num_shards > 1
    out_misato = (SHARD_DIR / f'misato_shard{args.shard}of{args.num_shards}.npz'
                  if is_sharded else OUT_MISATO)
    out_mdcath = (SHARD_DIR / f'mdcath_shard{args.shard}of{args.num_shards}.npz'
                  if is_sharded else OUT_MDCATH)

    datasets = args.datasets.split(',')

    from ensembits.baselines._real_or_ideal_bb import load_bb_for_frame

    # ===== misato =====
    # Paper: appendix.tex app:baselines — single-frame method, frame 0 of 8.
    # FIXED(code_fix E0, 2026-10-20): MISATO caches rebuilt with the fixed parser (CX proline Cα).
    # FIXED(code_fix P15, 2026-10-16): old: `and not out_misato.exists()` (silent skip).
    if 'misato' in datasets:
        print(f'[misato] tokenizing (shard {args.shard}/{args.num_shards})...',
              flush=True)
        coords = pickle.load(open(MISATO_COORDS, 'rb'))
        all_pids = sorted(coords.keys())
        my_pids = [p for j, p in enumerate(all_pids)
                   if j % args.num_shards == args.shard]
        out = {}; failed = {}; ideal_pids = []; n_real = n_ideal = 0
        t0 = time.time()
        for i, pid in enumerate(my_pids):
            try:
                ca = np.asarray(coords[pid][0], dtype=np.float32)   # frame 0
                bb, _, src = load_bb_for_frame('misato', pid, 0, ca)
                refuse_ideal_misato_frame('misato', pid, 0, src, args.allow_ideal)   # code_fix P15
                tok = tokenize_protein(enc, ca, device, bb=bb, ideal=(src == 'ideal'),
                                       **_pieces_kw('misato', pid, ca.shape[0]))
                if len(tok) != ca.shape[0]:   # F7: a token-count mismatch is a failure
                    raise ValueError(f'{len(tok)} tokens for {ca.shape[0]} residues')
                out[pid] = tok.astype(np.uint16)
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                if src == 'ideal': ideal_pids.append(pid)   # G7 (2026-10-17)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
            if (i+1) % 200 == 0:
                print(f'  misato {i+1}/{len(my_pids)} '
                      f'({time.time()-t0:.0f}s, ok={len(out)} '
                      f'real={n_real} ideal={n_ideal})', flush=True)
        print(f'misato: {len(out)} ok, {len(failed)} failed, '
              f'real={n_real} ideal={n_ideal}', flush=True)
        # FIXED(preretrain_audit_r2 F1 / F7, 2026-10-16): tolerance check + provenance sidecar.
        if not finish_cache(out_misato, out, failed, my_pids, 'misato', builder=bname,
                            max_fail_frac=args.max_fail_frac, sharded=is_sharded,
                            frames='frame 0', **split_record('misato'), sequence_source=SEQ_NOT_USED, codebook=OUT_CB, n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
            return 1

    # ===== mdcath =====
    # FIXED(code_fix P15, 2026-10-16): old: `and not out_mdcath.exists()` (silent skip).
    if 'mdcath' in datasets:
        print(f'[mdcath] tokenizing (shard {args.shard}/{args.num_shards})...',
              flush=True)
        with open(os.path.join(MDCATH_DIV, '_manifest.json')) as f:
            man = json.load(f)
        all_pids = list(man['representatives'].keys())
        pids = [p for j, p in enumerate(all_pids)
                if j % args.num_shards == args.shard]
        out = {}; failed = {}; ideal_pids = []; n_real = n_ideal = 0
        t0 = time.time()
        for i, pid in enumerate(pids):
            try:
                d = np.load(os.path.join(MDCATH_DIV, f'{pid}.npz'),
                             allow_pickle=False)
                ca = d['ca_10'][0].astype(np.float32)
                bb, _, src = load_bb_for_frame('mdcath', pid, 0, ca)
                # FIXED(preretrain_audit_r2 G7, 2026-10-17): old = an ideal-geometry fallback was
                #   tokenized and written; new = a failure (counted; fails the build inside
                #   the RMSF set), as in the mini3di builders.
                if src != 'real':
                    raise ValueError('no real backbone (ideal-geometry fallback)')
                tok = tokenize_protein(enc, ca, device, bb=bb, ideal=(src == 'ideal'),
                                       **_pieces_kw('mdcath', pid, ca.shape[0]))
                if len(tok) != ca.shape[0]:   # F7: a token-count mismatch is a failure
                    raise ValueError(f'{len(tok)} tokens for {ca.shape[0]} residues')
                out[pid] = tok.astype(np.uint16)
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                if src == 'ideal': ideal_pids.append(pid)   # G7 (2026-10-17)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
            if (i+1) % 200 == 0:
                print(f'  mdcath {i+1}/{len(pids)} '
                      f'({time.time()-t0:.0f}s, ok={len(out)} '
                      f'real={n_real} ideal={n_ideal})', flush=True)
        print(f'mdcath: {len(out)} ok, {len(failed)} failed, '
              f'real={n_real} ideal={n_ideal}', flush=True)
        if not finish_cache(out_mdcath, out, failed, pids, 'mdcath', builder=bname,
                            max_fail_frac=args.max_fail_frac, sharded=is_sharded,
                            frames='frame 0', **split_record('mdcath'), sequence_source=SEQ_NOT_USED, codebook=OUT_CB, n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
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
                idx = tokenize_protein(enc, ca, device, bb=bb, ideal=False,
                                       **_pieces_kw('bioemu', pid, ca.shape[0]))
                if len(idx) != ca.shape[0]:
                    raise ValueError(f'{len(idx)} tokens for {ca.shape[0]} residues')
                out[pid] = np.asarray(idx).astype(np.uint16)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
        print(f'bioemu: {len(out)} ok, {len(failed)} failed', flush=True)
        if not finish_cache(OUT_BIOEMU, out, failed, list(frames), 'bioemu', builder=bname,
                            max_fail_frac=args.max_fail_frac, frames='frame 0', **split_record('bioemu'), sequence_source=SEQ_NOT_USED, codebook=OUT_CB):
            return 1


if __name__ == '__main__': sys.exit(main())
