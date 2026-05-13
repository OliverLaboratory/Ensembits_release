"""AminoAseed structural-VQ tokenizer wrapper.

Source: https://github.com/KatarinaYuan/StructTokenBench
Checkpoint: ``codebook_512x1024-1e+19-linear-fixed-last``

Per-residue input: backbone N/Cα/C atoms (atom37 layout, slots 0/1/2).
Per-residue output: VQ index in ``[0, 512)`` → look up in the
``(512, 1024)`` projected codebook for a 1024-D feature.

Real backbone is preferred when available; pass ``bb=(L, 4, 3)``
N/CA/C/O for the canonical post-fix behaviour. Without real bb, N and C
are reconstructed via ideal peptide geometry (the legacy midpoint rule
is no longer used anywhere).

Multi-chain inputs are split on peptide-bond breaks before encoding —
AminoAseed assumes single-chain inputs and produces wrong tokens at
artificial cross-chain "peptide bonds" in concatenated multi-chain
proteins.

Usage:

    tok = AminoAseed(
        weights_path="…/codebook_512x1024-1e+19-linear-fixed-last.ckpt"
                     "/checkpoint/mp_rank_00_model_states.pt",
        repo_path="…/StructTokenBench/src",
        device="cuda",
    )
    ids = tok.tokenize(ca, bb=bb)          # (L,) int64 in [0, 512)
    cb = tok.codebook                       # (512, 1024) float32
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np

from ._backbone import atom37_from_bb, atom37_from_ca, split_chains_by_peptide_bond


class AminoAseed:
    """Thin wrapper around StructTokenBench's ``VQVAEModel`` encoder + quantizer."""

    NUM_CODES: int = 512
    CODE_DIM: int = 1024

    def __init__(self, weights_path: str | Path, repo_path: str | Path,
                  device: str = "cuda") -> None:
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                "AminoAseed requires `torch` and `omegaconf`. "
                "Install the optional extra: `pip install -e .[aminoaseed]`."
            ) from e

        weights_path = Path(weights_path)
        repo_path = Path(repo_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"AminoAseed weights not found: {weights_path}")
        if not repo_path.exists():
            raise FileNotFoundError(
                f"StructTokenBench source not found at {repo_path}. "
                "Clone https://github.com/KatarinaYuan/StructTokenBench and "
                "point `repo_path` at its `src/` directory.")

        # Stub `deepspeed` so vqvae_model imports cleanly.
        ds = types.ModuleType("deepspeed")
        ds.utils = types.ModuleType("deepspeed.utils")
        ds.utils.safe_get_full_grad = lambda p: None
        sys.modules.setdefault("deepspeed", ds)
        sys.modules.setdefault("deepspeed.utils", ds.utils)
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        cfg = OmegaConf.create({
            "class_name": "VQVAEModel",
            "pretrained_ckpt_path": "", "ckpt_path": None,
            "quantizer": {
                "quantizer_type": "StraightThroughQuantizer",
                "loss_weight": {"commitment_loss_weight": 0.25,
                                "quantization_loss_weight": 1.0,
                                "reconstruction_loss_weight": 1.0},
                "codebook_size": self.NUM_CODES,
                "codebook_embed_size": self.CODE_DIM,
                "_need_init": False, "freeze_codebook": False,
                "use_linear_project": True,
            },
            "encoder": {"d_model": 1024, "n_heads": 1, "v_heads": 128,
                        "n_layers": 2, "d_out": 1024},
            "decoder": {"d_model": 1024, "n_heads": 16, "n_layers": 8},
        })
        from vqvae_model import VQVAEModel       # type: ignore[import]
        m = VQVAEModel(model_cfg=cfg)
        state = torch.load(str(weights_path), map_location="cpu",
                            weights_only=False)["module"]
        new_state = {k[len("model."):]: v for k, v in state.items()
                     if k.startswith("model.")}
        m.load_state_dict(new_state, strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        m = m.to(device)

        # Use the *projected* codebook — quantizer compares encoder
        # outputs to linear_proj(codebook.weight), not raw codebook.
        with torch.no_grad():
            cb_proj = m.quantizer.linear_proj(m.quantizer.codebook.weight)
        self._codebook = cb_proj.detach().cpu().numpy().astype(np.float32)
        self._model = m
        self._torch = torch
        self._device = device

    @property
    def codebook(self) -> np.ndarray:
        """``(512, 1024)`` float32 — projected codebook used at quantization."""
        return self._codebook

    def _tokenize_single_chain(self, ca: np.ndarray,
                                bb: Optional[np.ndarray]) -> np.ndarray:
        torch = self._torch
        device = self._device
        if bb is not None:
            atom37 = atom37_from_bb(bb)
        else:
            atom37 = atom37_from_ca(ca.astype(np.float32))
        L = atom37.shape[0]
        coords = torch.tensor(atom37, device=device).unsqueeze(0)
        # encoder.encode() uses True = valid (post-flip convention). The
        # data-loader path flips True = padded inside VQVAEModel.forward,
        # but we call encoder.encode directly so we use the post-flip
        # convention ourselves.
        attention_mask = torch.ones((1, L), device=device, dtype=torch.bool)
        residue_index = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            z = self._model.encoder.encode(coords, attention_mask, None,
                                            residue_index)
            _, indices, _, _ = self._model.quantizer(z)
        return indices.squeeze(0).cpu().numpy().astype(np.int64)

    def tokenize(self, ca: np.ndarray,
                 bb: Optional[np.ndarray] = None,
                 *, chain_split: bool = True) -> np.ndarray:
        """Tokenize one protein (single- or multi-chain).

        Args:
            ca: ``(L, 3)`` Cα coordinates.
            bb: optional ``(L, 4, 3)`` real backbone ``[N, CA, C, O]``.
                When ``None``, N/C are reconstructed via ideal geometry.
            chain_split: if True (default) and ``bb`` is provided, split
                the protein on peptide-bond breaks (>5 Å) and tokenize
                each chain independently, then concatenate.

        Returns:
            ``(L,) int64`` token IDs in ``[0, 512)``.
        """
        if ca.ndim != 2 or ca.shape[-1] != 3:
            raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")

        if chain_split and bb is not None:
            chains = split_chains_by_peptide_bond(np.asarray(bb, dtype=np.float32))
            if len(chains) > 1:
                outs = [self._tokenize_single_chain(ca[s:e], bb[s:e])
                        for s, e in chains]
                return np.concatenate(outs)
        return self._tokenize_single_chain(ca, bb)
