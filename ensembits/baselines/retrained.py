"""Retrained static-tokenizer wrapper (AminoAseed / ESM3-style from a Lightning checkpoint).

What it does
------------
Loads a ``VQVAEModel`` from a PyTorch Lightning checkpoint (produced by
``train/retrain_static/train_static.py``) and exposes the same interface as
``AminoAseed`` / ``ESM3Struct``: ``tokenize(ca, n=None, c=None) -> (L,) int64``
and ``.codebook`` (numpy, float32).

Paper
-----
``appendix.tex`` static retrain appendix (P-17, P-26, P-42).

Origin
------
New file (2026-10-27, camera-ready retrain task).

Used by the reported model: no (retrained-baseline rows).

How to run
----------
    from ensembits.baselines.retrained import RetrainedTokenizer
    tok = RetrainedTokenizer(config_path=..., checkpoint_path=..., device="cuda:0")
    tokens = tok.tokenize(ca, n=n, c=c)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from ._backbone import atom37_from_ca


class RetrainedTokenizer:
    """Wrapper around a retrained VQVAEModel from a Lightning checkpoint."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path,
                 device: str = "cuda") -> None:
        import torch
        from omegaconf import OmegaConf

        config_path = Path(config_path)
        checkpoint_path = Path(checkpoint_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Stub deepspeed
        ds = types.ModuleType("deepspeed")
        ds.utils = types.ModuleType("deepspeed.utils")
        ds.utils.safe_get_full_grad = lambda p: None
        ds.zero = types.ModuleType("deepspeed.zero")

        class _NoopInit:
            def __init__(self, *a, **k): pass
            def __enter__(self): return None
            def __exit__(self, *a): return False

        ds.zero.Init = _NoopInit
        sys.modules.setdefault("deepspeed", ds)
        sys.modules.setdefault("deepspeed.utils", ds.utils)
        sys.modules.setdefault("deepspeed.zero", ds.zero)

        # Add StructTokenBench source
        code_root = Path(__file__).resolve().parents[1].parent  # ensembits_code/
        from ensembits.paths import STRUCTTOKENBENCH_SRC
        if str(STRUCTTOKENBENCH_SRC) not in sys.path:
            sys.path.insert(0, str(STRUCTTOKENBENCH_SRC))
        retrain_dir = code_root / "train" / "retrain_static"
        if str(retrain_dir) not in sys.path:
            sys.path.insert(0, str(retrain_dir))

        from ema_quantizer import EMAQuantizer  # type: ignore[import]
        import vqvae_model  # type: ignore[import]
        vqvae_model.EMAQuantizer = EMAQuantizer

        cfg = OmegaConf.load(str(config_path))
        m = vqvae_model.VQVAEModel(model_cfg=cfg.model)

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        new_state = {k[len("model."):]: v for k, v in state.items()
                     if k.startswith("model.")}
        m.load_state_dict(new_state, strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        m = m.to(device)

        qt = cfg.model.quantizer.quantizer_type
        with torch.no_grad():
            if qt == "StraightThroughQuantizer" and cfg.model.quantizer.get("use_linear_project", False):
                cb = m.quantizer.linear_proj(m.quantizer.codebook.weight)
            else:
                cb = m.quantizer.codebook.weight
        self._codebook = cb.detach().cpu().numpy().astype(np.float32)
        self._model = m
        self._torch = torch
        self._device = device

    @property
    def codebook(self) -> np.ndarray:
        return self._codebook

    def tokenize(self, ca: np.ndarray,
                 n: np.ndarray | None = None,
                 c: np.ndarray | None = None) -> np.ndarray:
        if ca.ndim != 2 or ca.shape[-1] != 3:
            raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")
        L = ca.shape[0]
        torch = self._torch
        device = self._device

        if n is not None and c is not None:
            atom37 = np.zeros((L, 37, 3), dtype=np.float32)
            atom37[:, 0, :] = n.astype(np.float32)
            atom37[:, 1, :] = ca.astype(np.float32)
            atom37[:, 2, :] = c.astype(np.float32)
        else:
            atom37 = atom37_from_ca(ca.astype(np.float32))
        coords = torch.tensor(atom37, device=device).unsqueeze(0)
        attention_mask = torch.ones((1, L), device=device, dtype=torch.bool)
        residue_index = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            z = self._model.encoder.encode(coords, attention_mask, None, residue_index)
            _, indices, _, _ = self._model.quantizer(z)
        return indices.squeeze(0).cpu().numpy().astype(np.int64)
