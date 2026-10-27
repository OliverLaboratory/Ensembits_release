"""Tokenize MISATO and mdCATH-div (frame 0) with AminoAseed (StructTokenBench).

What it does
------------
Builds StructTokenBench's `VQVAEModel` (codebook 512 x 1024, linear projection)
from `data/struct_tokenizer_ckpts/codebook_512x1024-1e+19-linear-fixed-last.ckpt`
and saves the *projected* codebook `linear_proj(codebook.weight)` (the vectors
the quantizer actually compares against). For every MISATO protein it takes
frame 0 of `coords_ca8_misato.pkl`, loads the real N / CA / C of that frame from
`misato_real_bb` (`ensembits.baselines._real_or_ideal_bb.load_bb_for_frame`;
ideal geometry only if no real frame matches) and runs encoder + quantizer chain
by chain. mdCATH-div proteins use frame 0 of `ca_10` and `mdcath_real_bb`.
`deepspeed` is stubbed so `vqvae_model` imports without it.

Paper
-----
`appendix.tex` \\label{app:baselines}, *Aminoaseed* (VQ index in [0, 512),
frozen 1024-D codebook). Results:
  * `2_results.tex` \\label{result:rmsf}, Table \\label{tab:rmsf} (baseline rows,
    mdCATH-div and MISATO columns);
  * `2_results.tex` \\label{sec:downstreams} (Fig. \\label{fig:downstream_radar});
  * `appendix.tex` \\label{app:downstreams} (Tables \\label{tab:ec_d1_structure},
    \\label{tab:go_structure}, \\label{tab:binding_structure},
    \\label{tab:affinity_structure}) and \\label{app:experiments} (all splits).

Origin
------
Copied from `submission_exp/src/build_aminoaseed_tokens.py`. Changed: the
StructTokenBench source path comes from `ensembits.paths.STRUCTTOKENBENCH_SRC`
and `MDCATH_DIV` from `ensembits.paths.MDCATH_DIV_SRC` (both were another
user's absolute paths); `ROOT` from `ensembits.paths.REPO_ROOT`; the backbone
helpers come from `ensembits.baselines` (same code); new flag
`--allow-old-parser` and a MISATO parser check (code_fix C5); comments and this
docstring. Logic unchanged.
2026-10-16: exit 1 on an existing output unless `--overwrite` and on an
ideal-geometry MISATO frame unless `--allow-ideal` (code_fix P15); chains split
at 2.5 Å, real backbones only (code_fix P5).
`tokenize_protein` takes `ideal=` (default False) for that.
2026-10-18 (authors' decision, baseline fidelity audit): the builder passes
`tokenize_protein(..., pieces=, residue_index=)` = the true chains and their gap-aware
residue index (MISATO: `misato_chain_breaks.json`, option A, a `chain_index` being a
molecule that can hold several chains; mdCATH / BioEmu whole, 0..L-1;
`_chain_pieces.py`), never the 2.5 Å rule; the residue index replaces the encoder's
arange; the sidecars record `chain_split`. The projected codebook is kept (disclosed).

Phase 2 (2026-10-16): `--datasets bioemu` tokenizes frame 0 of each BioEmu-CATH
system (bioemu_cath_real_bb bb_10, real backbone; data_prep/baselines/_bioemu_inputs.py)
-> `aminoaseed_tokens_bioemu.npz` (unsharded; any failed system fails the dataset).

2026-10-16 (preretrain_audit_r2 F1 / F7): every token cache gets `<output>.provenance.json`
(with the codebook's sha256; the RMSF probes refuse caches without it); a token-count
mismatch is a failure, and failures of proteins in the RMSF residue set make the build
exit 1 (`--max-fail-frac`).

Used by the reported model: no (AminoAseed baseline rows).
`build_aminoaseed_model` and `tokenize_protein` are also imported by
`build_pooled_baseline_features.py`.

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   data/struct_tokenizer_ckpts/codebook_512x1024-1e+19-linear-fixed-last.ckpt/
              checkpoint/mp_rank_00_model_states.pt
          data/external/StructTokenBench/src/vqvae_model.py   (STRUCTTOKENBENCH_SRC)
          data/cached_descriptors/coords_ca8_misato.pkl
          data/cached_descriptors/misato_real_bb/<pid>.npz
          data/cached_descriptors/mdcath_real_bb/<pid>.npz
          data/external/mdcath_div/{_manifest.json,<pid>.npz}   (MDCATH_DIV_SRC)
  writes  data/cached_descriptors/aminoaseed_codebook.npy          (512, 1024) float32
          data/cached_descriptors/aminoaseed_tokens_misato.npz     pid -> (L,) uint16
          data/cached_descriptors/aminoaseed_tokens_mdcath.npz     pid -> (L,) uint16

How to run
----------
    python data_prep/baselines/build_aminoaseed_tokens.py --device cuda:0 \\
        [--datasets misato,mdcath,bioemu] [--allow-old-parser] [--overwrite] [--allow-ideal]
    (every requested dataset's output must not exist: pass --datasets misato to
    rebuild only MISATO, or --overwrite)
"""
from __future__ import annotations
import os, sys, json, pickle, time, types
from pathlib import Path

# Stub deepspeed so vqvae_model imports without the training-only ds dep
ds = types.ModuleType('deepspeed'); ds.utils = types.ModuleType('deepspeed.utils')
ds.utils.safe_get_full_grad = lambda p: None
sys.modules['deepspeed'] = ds; sys.modules['deepspeed.utils'] = ds.utils
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))   # rmsf_common (sidecars, tolerances)
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _chain_pieces (2026-10-18)
BUILDER_TAG = 'v2026-10-18'   # recorded in the cache sidecars
from ensembits.paths import REPO_ROOT as ROOT   # noqa: E402  data/ and output/ stay in the research repo
from ensembits.paths import STRUCTTOKENBENCH_SRC, MDCATH_DIV_SRC   # noqa: E402  data/external/...
from ensembits.data_versions import require_misato_parser   # noqa: E402
# StructTokenBench source (vqvae_model.py); was another user's absolute path.
sys.path.insert(0, str(STRUCTTOKENBENCH_SRC))

import numpy as np
import torch
from omegaconf import OmegaConf

CKPT = ROOT / ('data/struct_tokenizer_ckpts/'
                'codebook_512x1024-1e+19-linear-fixed-last.ckpt/checkpoint/'
                'mp_rank_00_model_states.pt')
CACHE = ROOT / 'data' / 'cached_descriptors'

MISATO_COORDS = CACHE / 'coords_ca8_misato.pkl'   # 8 frames; we use frame 0
MISATO_REAL_BB = CACHE / 'misato_real_bb'       # read through _real_or_ideal_bb
MDCATH_DIV = str(MDCATH_DIV_SRC)

OUT_MISATO = CACHE / 'aminoaseed_tokens_misato.npz'
OUT_MDCATH = CACHE / 'aminoaseed_tokens_mdcath.npz'
OUT_BIOEMU = CACHE / 'aminoaseed_tokens_bioemu.npz'   # phase 2 (2026-10-16): --datasets bioemu
OUT_CB = CACHE / 'aminoaseed_codebook.npy'


def reconstruct_backbone_atom_set(ca: np.ndarray):
    """Cα → ideal-geometry (N, CA, C) per residue. (L, 3, 3) layout: [N, CA, C].

    Uses the shared `reconstruct_backbone_ideal` helper so the encoder's
    rigid frames aren't degenerate (the legacy midpoint rule forced
    N[i] = C[i-1], collapsing each residue's N/CA/C onto a line).
    """
    from ensembits.baselines._backbone import reconstruct_backbone_ideal
    n, c, _ = reconstruct_backbone_ideal(ca)
    return np.stack([n, ca, c], axis=1).astype(np.float32)   # (L, 3, 3)


def build_aminoaseed_model(device='cuda:0'):
    cfg = OmegaConf.create({
        'class_name': 'VQVAEModel',
        'pretrained_ckpt_path': '', 'ckpt_path': None,
        'quantizer': {
            'quantizer_type': 'StraightThroughQuantizer',
            'loss_weight': {'commitment_loss_weight': 0.25,
                            'quantization_loss_weight': 1.0,
                            'reconstruction_loss_weight': 1.0},
            'codebook_size': 512, 'codebook_embed_size': 1024,
            '_need_init': False, 'freeze_codebook': False,
            'use_linear_project': True,
        },
        'encoder': {'d_model': 1024, 'n_heads': 1, 'v_heads': 128,
                    'n_layers': 2, 'd_out': 1024},
        'decoder': {'d_model': 1024, 'n_heads': 16, 'n_layers': 8},
    })
    from vqvae_model import VQVAEModel
    m = VQVAEModel(model_cfg=cfg)
    state = torch.load(str(CKPT), map_location='cpu', weights_only=False)['module']
    new_state = {k[6:]: v for k, v in state.items() if k.startswith('model.')}
    m.load_state_dict(new_state, strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    m = m.to(device)
    return m


@torch.no_grad()
def _encode_chain(model, ca_c, bb_c, device, residue_index=None):
    """Tokenise a single chain segment via AminoAseed (`residue_index`: (L,) residue
    numbers of the segment, 2026-10-18; default 0..L-1 as before)."""
    L = ca_c.shape[0]
    coords37 = np.zeros((L, 37, 3), dtype=np.float32)
    coords37[:, 0, :] = bb_c[:, 0]   # N
    coords37[:, 1, :] = bb_c[:, 1]   # CA
    coords37[:, 2, :] = bb_c[:, 2]   # C
    coords = torch.tensor(coords37, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, L), device=device, dtype=torch.bool)
    if residue_index is None:
        residue_index = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)
    else:   # FIXED(baseline fidelity audit, 2026-10-18, option A): gap-aware numbering
        residue_index = torch.as_tensor(np.asarray(residue_index), device=device,
                                        dtype=torch.long).unsqueeze(0)
    z = model.encoder.encode(coords, attention_mask, None, residue_index)
    _, indices, _, _ = model.quantizer(z)
    return indices.squeeze(0).cpu().numpy().astype(np.int64)


def tokenize_protein(model, ca: np.ndarray, device='cuda:0',
                      bb: np.ndarray | None = None, ideal: bool = False,
                      pieces=None, residue_index=None) -> np.ndarray:
    """Cα (L, 3) → token ids (L,) int64 via AminoAseed encoder + quantizer.

    Tokenises chain-by-chain so chain boundaries don't see a fictitious
    40+ Å peptide bond (StructTokenBench / AminoAseed's encoder uses a
    KNN graph + relative positional encoding that assume a single
    contiguous chain).

    `bb`: optional `(L, 4, 3)` real `[N, CA, C, O]` backbone. When given,
    the encoder reads real N/CA/C; otherwise falls back to ideal-geometry
    reconstruction from Cα. `ideal`: True when `bb` itself was rebuilt with
    ideal geometry (`load_bb_for_frame` source 'ideal'); it is then not split.
    `pieces` (2026-10-18): the chains as [(start, end), ...] covering [0, L)
    (`_chain_pieces.chain_pieces`); the builders of this folder always pass them.
    Without `pieces` the old 2.5 Å split is used (other callers, unchanged).
    `residue_index` (2026-10-18, option A): (L,) residue number within each piece
    (`_chain_pieces.chain_pieces`), given to the encoder instead of 0..len-1.
    """
    L = ca.shape[0]
    bb_real = bb is not None
    ri = None if residue_index is None else np.asarray(residue_index, dtype=np.int64)
    if ri is not None and ri.shape != (L,):
        raise ValueError(f'residue_index has shape {ri.shape}, expected ({L},)')
    if bb is None:
        bb3 = reconstruct_backbone_atom_set(ca)   # (L, 3, 3) ideal
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
        return _encode_chain(model, ca, bb, device, residue_index=ri)
    out = np.empty(L, dtype=np.int64)
    for s, e in chunks:
        out[s:e] = _encode_chain(model, ca[s:e], bb[s:e], device,
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
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--datasets', default='misato,mdcath')
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
    # FIXED(preretrain_audit_r2 H8, 2026-10-16): --device is a torch device ("cuda:0"); a bare
    #   index ("0", the form of the ProToken / ESM3struct builders) is accepted too.
    if args.device.isdigit():
        args.device = f'cuda:{args.device}'
    from rmsf_common import finish_cache, split_record, SEQ_NOT_USED   # H4: recorded in every sidecar
    bname = f'{Path(__file__).name} {BUILDER_TAG}'
    if 'misato' in args.datasets.split(','):
        # code_fix C5: refuse MISATO backbones from the old parser (proline CA = Cδ).
        require_misato_parser(MISATO_REAL_BB, allow_old=args.allow_old_parser)
    # code_fix P15: every requested output must be absent (before loading the model).
    from ensembits.baselines._real_or_ideal_bb import refuse_existing_outputs, refuse_ideal_misato_frame
    refuse_existing_outputs([{'misato': OUT_MISATO, 'mdcath': OUT_MDCATH, 'bioemu': OUT_BIOEMU}[d]
                             for d in args.datasets.split(',')
                             if d in ('misato', 'mdcath', 'bioemu')], args.overwrite)

    print(f'[setup] loading AminoAseed on {args.device}...', flush=True)
    t0 = time.time()
    model = build_aminoaseed_model(args.device)
    print(f'  loaded in {time.time()-t0:.0f}s, '
          f'{sum(p.numel() for p in model.parameters())/1e6:.1f}M params',
          flush=True)

    # Save the *projected* codebook. The AminoAseed quantizer is configured
    # with `use_linear_project=True`, which means the model computes
    # nearest-neighbor in projected space:
    #     weight = self.linear_proj(self.codebook.weight)   # (M, d)
    #     dist   = cdist(z, weight); idx = argmin(dist, dim=-1)
    # so the actual quantization-target vector for token i is
    # `linear_proj(codebook.weight[i])`, NOT the raw `codebook.weight[i]`.
    # Downstream lookups (PG distance, conv1d/MLP head input) must use the
    # same vectors the model trained against — saving the raw codebook
    # decouples lookup-space distances from the model's geometry and silently
    # destroys downstream signal (PG was hit hardest because its scoring is
    # a fixed `||cb[wt] - cb[mut]||` distance with no learnable head).
    with torch.no_grad():
        cb_proj = model.quantizer.linear_proj(model.quantizer.codebook.weight)
    cb = cb_proj.detach().cpu().numpy().astype(np.float32)
    # FIXED(preretrain_audit_r2 G12, 2026-10-17): atomic write; an identical codebook is kept,
    #   so a later single-dataset call does not touch the file the other caches recorded.
    from rmsf_common import write_codebook
    if write_codebook(OUT_CB, cb):
        print(f'  wrote codebook (projected) {cb.shape} → {OUT_CB.name}', flush=True)
    else:
        print(f'  codebook unchanged ({OUT_CB.name} kept)', flush=True)

    datasets = args.datasets.split(',')

    from ensembits.baselines._real_or_ideal_bb import load_bb_for_frame

    # Paper: appendix.tex app:baselines — single-frame method, frame 0 of 8.
    # FIXED(code_fix E0, 2026-10-20): MISATO caches rebuilt with the fixed parser (CX proline Cα).
    # FIXED(code_fix P15, 2026-10-16): old: `and not OUT_MISATO.exists()` (silent skip).
    if 'misato' in datasets:
        print(f'[misato] tokenizing...', flush=True)
        coords = pickle.load(open(MISATO_COORDS, 'rb'))
        out = {}
        n_real = n_ideal = 0; failed = {}; ideal_pids = []
        t0 = time.time()
        attempted = sorted(coords.keys())
        for i, pid in enumerate(attempted):
            try:
                ca = np.asarray(coords[pid][0], dtype=np.float32)   # frame 0
                bb, _, src = load_bb_for_frame('misato', pid, 0, ca)
                refuse_ideal_misato_frame('misato', pid, 0, src, args.allow_ideal)   # code_fix P15
                tok = tokenize_protein(model, ca, args.device, bb=bb, ideal=(src == 'ideal'),
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
                print(f'  misato {i+1}/{len(coords)} '
                      f'({time.time()-t0:.0f}s, real={n_real} ideal={n_ideal})',
                      flush=True)
        print(f'misato: {len(out)} ok, {len(failed)} failed, '
              f'real={n_real} ideal={n_ideal}', flush=True)
        # FIXED(preretrain_audit_r2 F1 / F7, 2026-10-16): tolerance check + provenance sidecar.
        if not finish_cache(OUT_MISATO, out, failed, attempted, 'misato', builder=bname,
                            max_fail_frac=args.max_fail_frac, frames='frame 0', **split_record('misato'), sequence_source=SEQ_NOT_USED,
                            codebook=OUT_CB, n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
            return 1

    # FIXED(code_fix P15, 2026-10-16): old: `and not OUT_MDCATH.exists()` (silent skip).
    if 'mdcath' in datasets:
        print(f'[mdcath] tokenizing...', flush=True)
        with open(os.path.join(MDCATH_DIV, '_manifest.json')) as f:
            man = json.load(f)
        pids = list(man['representatives'].keys())
        out = {}
        n_real = n_ideal = 0; failed = {}; ideal_pids = []
        t0 = time.time()
        for i, pid in enumerate(pids):
            try:
                d = np.load(os.path.join(MDCATH_DIV, f'{pid}.npz'),
                             allow_pickle=False)
                ca = d['ca_10'][0].astype(np.float32)   # frame 0
                bb, _, src = load_bb_for_frame('mdcath', pid, 0, ca)
                # FIXED(preretrain_audit_r2 G7, 2026-10-17): old = an ideal-geometry fallback was
                #   tokenized and written; new = a failure (counted; fails the build inside
                #   the RMSF set), as in the mini3di builders.
                if src != 'real':
                    raise ValueError('no real backbone (ideal-geometry fallback)')
                tok = tokenize_protein(model, ca, args.device, bb=bb, ideal=(src == 'ideal'),
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
                      f'({time.time()-t0:.0f}s, real={n_real} ideal={n_ideal})',
                      flush=True)
        print(f'mdcath: {len(out)} ok, {len(failed)} failed, '
              f'real={n_real} ideal={n_ideal}', flush=True)
        if not finish_cache(OUT_MDCATH, out, failed, pids, 'mdcath', builder=bname,
                            max_fail_frac=args.max_fail_frac, frames='frame 0', **split_record('mdcath'), sequence_source=SEQ_NOT_USED,
                            codebook=OUT_CB, n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
            return 1

    # Phase 2 (2026-10-16): BioEmu-CATH systems, frame 0 of bioemu_cath_real_bb bb_10 (real
    # N/CA/C/O; data_prep/baselines/_bioemu_inputs.py), single-frame like the other datasets.
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
                tok = tokenize_protein(model, ca, args.device, bb=bb, ideal=False,
                                       **_pieces_kw('bioemu', pid, ca.shape[0]))
                if len(tok) != ca.shape[0]:
                    raise ValueError(f'{len(tok)} tokens for {ca.shape[0]} residues')
                out[pid] = tok.astype(np.uint16)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
        print(f'bioemu: {len(out)} ok, {len(failed)} failed', flush=True)
        if not finish_cache(OUT_BIOEMU, out, failed, list(frames), 'bioemu', builder=bname,
                            max_fail_frac=args.max_fail_frac, frames='frame 0', **split_record('bioemu'), sequence_source=SEQ_NOT_USED, codebook=OUT_CB):
            return 1


if __name__ == '__main__': sys.exit(main())
