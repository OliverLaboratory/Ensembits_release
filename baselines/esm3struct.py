"""ESM3 structure-track tokenizer wrapper.

Source: ESM3 (``esm3-sm-open-v1`` ``StructureTokenEncoder``)
https://github.com/evolutionaryscale/esm

Per-residue input: backbone N/Cα/C atoms (atom37 layout, slots 0/1/2).
Per-residue output: VQ index in ``[0, 4096)`` → ``(4096, 128)`` frozen
EMA codebook lookup → 128-D feature.

Real backbone is preferred when available; pass ``bb=(L, 4, 3)``
N/CA/C/O. Without real bb, N and C are reconstructed via ideal peptide
geometry.

Multi-chain inputs are split on peptide-bond breaks before encoding —
ESM3's KNN-based partner-index implementation produces wrong tokens at
artificial cross-chain "peptide bonds".

NB: ESM3's structure tokenizer is co-trained with an inverse-folding
cross-entropy loss, so its tokens implicitly carry sequence-derived
information. We report this caveat in the manuscript whenever we
compare against ``esm3struct``.

ESM3 downloads its weights from HuggingFace Hub on first use; set
``HF_TOKEN`` if the model is gated.

Usage::

    tok = ESM3Struct(device="cuda")              # downloads from HF
    ids = tok.tokenize(ca, bb=bb)                # (L,) int64 in [0, 4096)
    cb = tok.codebook                             # (4096, 128) float32
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from ._backbone import atom37_from_bb, atom37_from_ca, split_chains_by_peptide_bond


class ESM3Struct:
    """Thin wrapper around ESM3's ``StructureTokenEncoder``."""

    NUM_CODES: int = 4096
    CODE_DIM: int = 128

    def __init__(self, weights_path: str | Path | None = None,
                  device: str = "cuda") -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("ESM3Struct requires `torch`.") from e
        try:
            from esm.pretrained import ESM3_structure_encoder_v0  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "ESM3Struct requires the `esm` package (`pip install esm`). "
                "Set HF_TOKEN before importing if the model is gated."
            ) from e

        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"weights_path does not exist: {weights_path}")
            os.environ.setdefault("HF_HOME", str(weights_path))

        enc = ESM3_structure_encoder_v0(device=device)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        self._enc = enc
        self._codebook = enc.codebook.embeddings.detach().cpu().numpy().astype(np.float32)
        self._torch = torch
        self._device = device

    @property
    def codebook(self) -> np.ndarray:
        """``(4096, 128)`` float32 — ESM3's frozen EMA codebook."""
        return self._codebook

    def _tokenize_single_chain(self, ca: np.ndarray,
                                bb: Optional[np.ndarray]) -> np.ndarray:
        torch = self._torch
        if bb is not None:
            atom37 = atom37_from_bb(bb)
        else:
            atom37 = atom37_from_ca(ca.astype(np.float32))
        coords = torch.tensor(atom37, device=self._device).unsqueeze(0)
        with torch.no_grad():
            _, idx = self._enc.encode(coords)
        return idx.squeeze(0).cpu().numpy().astype(np.int64)

    def tokenize(self, ca: np.ndarray,
                 bb: Optional[np.ndarray] = None,
                 *, chain_split: bool = True) -> np.ndarray:
        """Tokenize one protein (single- or multi-chain).

        Args:
            ca: ``(L, 3)`` Cα coordinates.
            bb: optional ``(L, 4, 3)`` real backbone ``[N, CA, C, O]``.
            chain_split: if True (default) and ``bb`` is provided, split
                the protein on peptide-bond breaks (>5 Å) and tokenize
                each chain independently, then concatenate.

        Returns:
            ``(L,) int64`` token IDs in ``[0, 4096)``.
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
