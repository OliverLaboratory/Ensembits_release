"""AminoAseed (StructTokenBench) structural-VQ tokenizer wrapper.

What it does
------------
Loads StructTokenBench's `VQVAEModel` (checkpoint
`codebook_512x1024-1e+19-linear-fixed-last`) from its source tree and exposes
`tokenize(ca, n=None, c=None) -> (L,) int64` in [0, 512) and the *projected*
(512, 1024) codebook. Without real N / C it uses ideal geometry
(`_backbone.atom37_from_ca`).

Paper
-----
`appendix.tex` \\label{app:baselines}, *Aminoaseed*.

Origin
------
Copied from `Ensembits/utils/baselines/aminoaseed.py`. Logic unchanged;
docstring rewritten. Not called by the scripts of this codebase (the token
builder `data_prep/baselines/build_aminoaseed_tokens.py` builds the model
itself); copied because `ensembits/baselines/__init__.py` imports it. The
StructTokenBench source path is passed by the caller (`repo_path`); in this
codebase it is `ensembits.paths.STRUCTTOKENBENCH_SRC`.

Used by the reported model: no.

Inputs / outputs
----------------
reads the checkpoint at `weights_path` and the StructTokenBench source at
`repo_path` (both given by the caller)

How to run
----------
    from ensembits.baselines import AminoAseed
    tok = AminoAseed(weights_path=..., repo_path=STRUCTTOKENBENCH_SRC, device="cuda")
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

from ._backbone import atom37_from_ca


class AminoAseed:
    """Thin wrapper around StructTokenBench's `VQVAEModel` encoder + quantizer.

    The constructor pulls in StructTokenBench's source from
    `repo_path` and loads the checkpoint at `weights_path`.
    """

    NUM_CODES: int = 512
    CODE_DIM: int = 1024

    def __init__(self, weights_path: str | Path, repo_path: str | Path,
                  device: str = "cuda") -> None:
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                "AminoAseed requires `torch` and `omegaconf`. Install with "
                "`pip install torch omegaconf`."
            ) from e

        weights_path = Path(weights_path)
        repo_path = Path(repo_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"AminoAseed weights not found: {weights_path}")
        if not repo_path.exists():
            raise FileNotFoundError(
                f"StructTokenBench source not found at {repo_path}. "
                "Clone https://github.com/KatarinaYuan/StructTokenBench and "
                "point `repo_path` at its `src/` directory."
            )

        # Stub `deepspeed` so vqvae_model imports cleanly without the
        # training-only dependency.
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

        # Capture the *projected* codebook — the quantizer compares
        # encoder outputs to `linear_proj(codebook.weight)` (not the raw
        # codebook), so any downstream consumer must use the projected
        # vectors to stay in the same metric space.
        with torch.no_grad():
            cb_proj = m.quantizer.linear_proj(m.quantizer.codebook.weight)
        self._codebook = cb_proj.detach().cpu().numpy().astype(np.float32)
        self._model = m
        self._torch = torch
        self._device = device

    @property
    def codebook(self) -> np.ndarray:
        """`(512, 1024) float32` — projected codebook used at quantization time."""
        return self._codebook

    def tokenize(self, ca: np.ndarray,
                 n: np.ndarray | None = None,
                 c: np.ndarray | None = None) -> np.ndarray:
        """`(L, 3)` Cα → `(L,) int64` token IDs in `[0, 512)`.

        Args:
            ca: `(L, 3)` Cα coords (required).
            n:  optional `(L, 3)` real N atom positions. If None, N is
                reconstructed from Cα via ideal-geometry rule.
            c:  optional `(L, 3)` real C atom positions. If None, C is
                reconstructed from Cα via ideal-geometry rule.
        """
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
            atom37 = atom37_from_ca(ca.astype(np.float32))               # (L, 37, 3)
        coords = torch.tensor(atom37, device=device).unsqueeze(0)        # (1, L, 37, 3)
        # encoder.encode() expects True = valid. The data-loader path uses
        # True = padded and `VQVAEModel.forward()` flips it before reaching
        # the encoder, but we bypass forward() and call encoder.encode()
        # directly, so we must pass the post-flip convention ourselves.
        attention_mask = torch.ones((1, L), device=device, dtype=torch.bool)
        residue_index = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            z = self._model.encoder.encode(coords, attention_mask, None,
                                            residue_index)
            _, indices, _, _ = self._model.quantizer(z)
        return indices.squeeze(0).cpu().numpy().astype(np.int64)
