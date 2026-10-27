"""Tokenize MISATO, mdCATH-div and BioEmu-CATH with a retrained static tokenizer checkpoint.

What it does
------------
Loads a retrained VQVAEModel from a PyTorch Lightning checkpoint (produced by
``train/retrain_static/train_static.py``) and tokenizes the same proteins as
``build_aminoaseed_tokens.py`` / ``build_esm3struct_tokens.py``.  Supports both
AminoAseed-style (StraightThroughQuantizer, projected codebook) and ESM3-style
(EMAQuantizer, direct codebook) architectures, determined by the training config.

The tokenization logic (``build_aminoaseed_tokens.tokenize_protein``) is reused
exactly: the same encoder, quantizer, chain splitting and real-backbone loading.
Output: ``<name>_tokens_{misato,mdcath,bioemu}.npz``, ``<name>_codebook.npy``
and provenance sidecars in ``data/cached_descriptors/``.

Paper
-----
``appendix.tex`` static retrain appendix (promises P-17, P-26, P-42):
"every static tokenizer, retrained on exactly our data with exactly our
splits and budget".

Origin
------
New file (2026-10-27, camera-ready retrain task). Delegates tokenization to
``build_aminoaseed_tokens.tokenize_protein`` (unchanged); the model construction
and checkpoint loading are the only new code.

Used by the reported model: no (retrained-baseline rows).

Inputs / outputs
----------------
Relative to REPO_ROOT:
  reads   the checkpoint given by --checkpoint (best val RMSD from convergence run)
          the config YAML given by --config (architecture + quantizer type)
          data/cached_descriptors/coords_ca8_misato.pkl, misato_real_bb/, mdcath_real_bb/
          data/external/mdcath_div/
  writes  data/cached_descriptors/<name>_codebook.npy
          data/cached_descriptors/<name>_tokens_misato.npz
          data/cached_descriptors/<name>_tokens_mdcath.npz
          data/cached_descriptors/<name>_tokens_bioemu.npz
          (each with a .provenance.json sidecar)

How to run
----------
    python data_prep/baselines/build_retrained_tokens.py \\
        --config train/retrain_static/config/aminoaseed_retrained.yaml \\
        --checkpoint output/static_retrain/aminoaseed_converged/checkpoints/<best>.ckpt \\
        --name aminoaseed_retrained --device cuda:0 \\
        [--datasets misato,mdcath,bioemu] [--overwrite]
"""
from __future__ import annotations
import os, sys, json, pickle, time, types, hashlib
from pathlib import Path

# Stub deepspeed so vqvae_model imports without the training-only ds dep
ds = types.ModuleType('deepspeed'); ds.utils = types.ModuleType('deepspeed.utils')
ds.utils.safe_get_full_grad = lambda p: None
ds.modules = types.ModuleType('deepspeed.zero')

class _NoopInit:
    def __init__(self, *a, **k): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False

ds.modules.Init = _NoopInit
sys.modules.setdefault('deepspeed', ds)
sys.modules.setdefault('deepspeed.utils', ds.utils)
sys.modules.setdefault('deepspeed.zero', ds.modules)

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'downstream'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
BUILDER_TAG = 'v2026-10-27'
from ensembits.paths import REPO_ROOT as ROOT
from ensembits.paths import STRUCTTOKENBENCH_SRC, MDCATH_DIV_SRC
from ensembits.data_versions import require_misato_parser
sys.path.insert(0, str(STRUCTTOKENBENCH_SRC))
# Also add the retrain_static dir for ema_quantizer
sys.path.insert(0, str(CODE_ROOT / 'train' / 'retrain_static'))

import numpy as np
import torch
from omegaconf import OmegaConf

CACHE = ROOT / 'data' / 'cached_descriptors'
MISATO_COORDS = CACHE / 'coords_ca8_misato.pkl'
MISATO_REAL_BB = CACHE / 'misato_real_bb'
MDCATH_DIV = str(MDCATH_DIV_SRC)


def build_retrained_model(config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
    """Load a retrained VQVAEModel from a Lightning checkpoint.

    Returns (model, codebook_np, quantizer_type).
    """
    from ema_quantizer import EMAQuantizer
    import vqvae_model
    vqvae_model.EMAQuantizer = EMAQuantizer

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model

    # Build the model
    m = vqvae_model.VQVAEModel(model_cfg=model_cfg)

    # Load from Lightning checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    # Strip 'model.' prefix (Lightning wraps the model)
    new_state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
    m.load_state_dict(new_state, strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m = m.to(device)

    # Extract codebook
    qt = cfg.model.quantizer.quantizer_type
    with torch.no_grad():
        if qt == 'StraightThroughQuantizer' and cfg.model.quantizer.get('use_linear_project', False):
            # AminoAseed-style: projected codebook
            cb = m.quantizer.linear_proj(m.quantizer.codebook.weight)
        else:
            # EMAQuantizer or no projection: direct codebook
            cb = m.quantizer.codebook.weight
    codebook = cb.detach().cpu().numpy().astype(np.float32)

    print(f'  quantizer_type={qt}, codebook shape={codebook.shape}', flush=True)
    print(f'  checkpoint epoch={ckpt.get("epoch", "?")}, step={ckpt.get("global_step", "?")}',
          flush=True)
    return m, codebook, qt


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='training config YAML')
    ap.add_argument('--checkpoint', required=True, help='Lightning checkpoint path')
    ap.add_argument('--name', required=True, help='output name prefix (e.g. aminoaseed_retrained)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--datasets', default='misato,mdcath,bioemu')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--allow-old-parser', action='store_true')
    ap.add_argument('--allow-ideal', action='store_true')
    ap.add_argument('--max-fail-frac', type=float, default=0.0)
    args = ap.parse_args()

    if args.device.isdigit():
        args.device = f'cuda:{args.device}'

    # Import the tokenize_protein function from the aminoaseed builder
    from build_aminoaseed_tokens import tokenize_protein, _pieces_kw

    from rmsf_common import finish_cache, split_record, SEQ_NOT_USED, write_codebook
    from ensembits.baselines._real_or_ideal_bb import (
        load_bb_for_frame, refuse_existing_outputs, refuse_ideal_misato_frame)

    OUT_CB = CACHE / f'{args.name}_codebook.npy'
    OUT_MISATO = CACHE / f'{args.name}_tokens_misato.npz'
    OUT_MDCATH = CACHE / f'{args.name}_tokens_mdcath.npz'
    OUT_BIOEMU = CACHE / f'{args.name}_tokens_bioemu.npz'

    datasets = args.datasets.split(',')
    if 'misato' in datasets:
        require_misato_parser(MISATO_REAL_BB, allow_old=args.allow_old_parser)
    refuse_existing_outputs([{'misato': OUT_MISATO, 'mdcath': OUT_MDCATH, 'bioemu': OUT_BIOEMU}[d]
                             for d in datasets if d in ('misato', 'mdcath', 'bioemu')], args.overwrite)

    bname = f'{Path(__file__).name} {BUILDER_TAG}'

    print(f'[setup] loading retrained model from {args.checkpoint} on {args.device}...', flush=True)
    t0 = time.time()
    model, codebook, qt = build_retrained_model(args.config, args.checkpoint, args.device)
    print(f'  loaded in {time.time()-t0:.0f}s, '
          f'{sum(p.numel() for p in model.parameters())/1e6:.1f}M params', flush=True)

    if write_codebook(OUT_CB, codebook):
        print(f'  wrote codebook {codebook.shape} -> {OUT_CB.name}', flush=True)
    else:
        print(f'  codebook unchanged ({OUT_CB.name} kept)', flush=True)

    if 'misato' in datasets:
        print('[misato] tokenizing...', flush=True)
        coords = pickle.load(open(MISATO_COORDS, 'rb'))
        out = {}; n_real = n_ideal = 0; failed = {}; ideal_pids = []
        t0 = time.time()
        attempted = sorted(coords.keys())
        for i, pid in enumerate(attempted):
            try:
                ca = np.asarray(coords[pid][0], dtype=np.float32)
                bb, _, src = load_bb_for_frame('misato', pid, 0, ca)
                refuse_ideal_misato_frame('misato', pid, 0, src, args.allow_ideal)
                tok = tokenize_protein(model, ca, args.device, bb=bb, ideal=(src == 'ideal'),
                                       **_pieces_kw('misato', pid, ca.shape[0]))
                if len(tok) != ca.shape[0]:
                    raise ValueError(f'{len(tok)} tokens for {ca.shape[0]} residues')
                out[pid] = tok.astype(np.uint16)
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                if src == 'ideal': ideal_pids.append(pid)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
            if (i+1) % 200 == 0:
                print(f'  misato {i+1}/{len(coords)} ({time.time()-t0:.0f}s, real={n_real} ideal={n_ideal})', flush=True)
        print(f'misato: {len(out)} ok, {len(failed)} failed, real={n_real} ideal={n_ideal}', flush=True)
        if not finish_cache(OUT_MISATO, out, failed, attempted, 'misato', builder=bname,
                            max_fail_frac=args.max_fail_frac, frames='frame 0',
                            **split_record('misato'), sequence_source=SEQ_NOT_USED,
                            codebook=OUT_CB, n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
            return 1

    if 'mdcath' in datasets:
        print('[mdcath] tokenizing...', flush=True)
        with open(os.path.join(MDCATH_DIV, '_manifest.json')) as f:
            man = json.load(f)
        pids = list(man['representatives'].keys())
        out = {}; n_real = n_ideal = 0; failed = {}; ideal_pids = []
        t0 = time.time()
        for i, pid in enumerate(pids):
            try:
                d = np.load(os.path.join(MDCATH_DIV, f'{pid}.npz'), allow_pickle=False)
                ca = d['ca_10'][0].astype(np.float32)
                bb, _, src = load_bb_for_frame('mdcath', pid, 0, ca)
                if src != 'real':
                    raise ValueError('no real backbone (ideal-geometry fallback)')
                tok = tokenize_protein(model, ca, args.device, bb=bb, ideal=(src == 'ideal'),
                                       **_pieces_kw('mdcath', pid, ca.shape[0]))
                if len(tok) != ca.shape[0]:
                    raise ValueError(f'{len(tok)} tokens for {ca.shape[0]} residues')
                out[pid] = tok.astype(np.uint16)
                n_real += (src == 'real'); n_ideal += (src == 'ideal')
                if src == 'ideal': ideal_pids.append(pid)
            except Exception as e:
                failed[pid] = f'{type(e).__name__}: {e}'
                if len(failed) < 5:
                    print(f'  {pid}: FAIL {type(e).__name__} {e}', flush=True)
            if (i+1) % 200 == 0:
                print(f'  mdcath {i+1}/{len(pids)} ({time.time()-t0:.0f}s, real={n_real} ideal={n_ideal})', flush=True)
        print(f'mdcath: {len(out)} ok, {len(failed)} failed, real={n_real} ideal={n_ideal}', flush=True)
        if not finish_cache(OUT_MDCATH, out, failed, pids, 'mdcath', builder=bname,
                            max_fail_frac=args.max_fail_frac, frames='frame 0',
                            **split_record('mdcath'), sequence_source=SEQ_NOT_USED,
                            codebook=OUT_CB, n_ideal_frames=n_ideal, ideal_pids=sorted(set(ideal_pids))):
            return 1

    if 'bioemu' in datasets:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _bioemu_inputs as BI
        print('[bioemu] tokenizing...', flush=True)
        frames = BI.CaFrames()
        out = {}; failed = {}
        for i, pid in enumerate(frames):
            try:
                ca = np.asarray(frames[pid][0], dtype=np.float32)
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
                            max_fail_frac=args.max_fail_frac, frames='frame 0',
                            **split_record('bioemu'), sequence_source=SEQ_NOT_USED, codebook=OUT_CB):
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
