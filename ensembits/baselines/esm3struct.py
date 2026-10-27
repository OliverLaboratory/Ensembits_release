"""ESM3 structure-track tokenizer wrapper (esm3-sm-open-v1 StructureTokenEncoder).

What it does
------------
Wraps `esm.pretrained.ESM3_structure_encoder_v0`: `tokenize(ca, n=None,
c=None) -> (L,) int64` in [0, 4096) and the frozen (4096, 128) EMA codebook.
Without real N / C it uses ideal geometry (`_backbone.atom37_from_ca`).
The weights come from the Hugging Face Hub; the token is read from the
`HF_TOKEN` environment variable (or the cached `huggingface-cli login`), never
from the source.

Paper
-----
`appendix.tex` \\label{app:baselines}, *ESM3struct*.

Origin
------
Copied from `Ensembits/utils/baselines/esm3struct.py`. Logic unchanged;
docstring rewritten. Not called by the scripts of this codebase (the token
builder `data_prep/baselines/build_esm3struct_tokens.py` calls the esm package
directly); copied because `ensembits/baselines/__init__.py` imports it.

Used by the reported model: no.

Inputs / outputs
----------------
downloads / reads the ESM3 weights through the Hugging Face cache

How to run
----------
    from ensembits.baselines import ESM3Struct
    tok = ESM3Struct(device="cuda")
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._backbone import atom37_from_ca


class ESM3Struct:
    """Thin wrapper around ESM3's `StructureTokenEncoder`."""

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
                "ESM3Struct requires the `esm` package "
                "(`pip install esm`). Set HF_TOKEN before importing if "
                "you need to download the weights from HuggingFace Hub."
            ) from e

        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"weights_path does not exist: {weights_path}")
            # ESM3's loader pulls from the HF hub by default; if the user
            # wants to point at a local cache they can populate
            # HF_HOME / TRANSFORMERS_CACHE before calling.
            os.environ.setdefault("HF_HOME", str(weights_path))

        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            # The HF gate doesn't always require auth, but the user should
            # be told upfront which env var matters if it does.
            print("[ESM3Struct] HF_TOKEN not set; the public release is "
                  "ungated as of this writing but auth may still be "
                  "required depending on Hub state.")

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
        """`(4096, 128) float32` — ESM3's frozen EMA codebook."""
        return self._codebook

    def tokenize(self, ca: np.ndarray,
                 n: np.ndarray | None = None,
                 c: np.ndarray | None = None) -> np.ndarray:
        """`(L, 3)` Cα → `(L,) int64` token IDs in `[0, 4096)`.

        Args:
            ca: `(L, 3)` Cα coords (required).
            n:  optional `(L, 3)` real N atom positions. If None, N is
                reconstructed via ideal-geometry rule.
            c:  optional `(L, 3)` real C atom positions. If None, C is
                reconstructed via ideal-geometry rule.
        """
        if ca.ndim != 2 or ca.shape[-1] != 3:
            raise ValueError(f"expected ca shape (L, 3), got {ca.shape}")
        torch = self._torch
        if n is not None and c is not None:
            L = ca.shape[0]
            atom37 = np.zeros((L, 37, 3), dtype=np.float32)
            atom37[:, 0, :] = n.astype(np.float32)
            atom37[:, 1, :] = ca.astype(np.float32)
            atom37[:, 2, :] = c.astype(np.float32)
        else:
            atom37 = atom37_from_ca(ca.astype(np.float32))
        coords = torch.tensor(atom37, device=self._device).unsqueeze(0)  # (1, L, 37, 3)
        with torch.no_grad():
            _, idx = self._enc.encode(coords)
        return idx.squeeze(0).cpu().numpy().astype(np.int64)
